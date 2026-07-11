import os
import torch
import torch.nn.functional as F
from torch import nn

from util.extract_variants import extract_feat_dino_single, extract_feat_dino_fusion


class SSP_MatchingNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        *,

        # [Checkpoint Configuration]
        # These paths correspond to the default DINOv3 implementation.
        # Replace them when using another local repository or pretrained checkpoint.
        dinov3_repo: str = '/HERA/dinov3',
        dinov3_ckpt_path: str = '/HERA/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_type = backbone

        # [Backbone Initialization]
        # Load the selected backbone and register its feature-extraction functions here.
        # Additional models should provide the same feature interface used below.
        if self.backbone_type == 'DINOv3':
            hubconf_path = os.path.join(dinov3_repo, 'hubconf.py')
            if not os.path.exists(hubconf_path):
                raise FileNotFoundError(
                    f"[DINOv3] hubconf.py not found under repo dir: {dinov3_repo}\n"
                )
            
            self.backbone = torch.hub.load(dinov3_repo, 'dinov3_vitl16', source='local', weights=dinov3_ckpt_path)

            if freeze_backbone:
                for p in self.backbone.parameters(): p.requires_grad_(False)

            # [Layer Configuration]
            # The default 24-block backbone uses candidate layers 12-23.
            # Update this range when the replacement model uses a different depth.
            self.feat_ids = list(range(12, 24))

            self.extract_feat_dino_single = extract_feat_dino_single
            self.extract_feat_dino_fusion = extract_feat_dino_fusion
            self.extract_feats = self.extract_feat_dino_single

        else:
            raise ValueError(f'Unavailable backbone: {backbone}')


    # Forward supports three modes:
    def forward(self, img_s_list, mask_s_list, img_q, mask_q, tta_flag: bool = False, q_flag: bool = False):
        # - q_flag=True  : extract query features only
        if q_flag:
            if self.backbone_type == 'DINOv3':
                feature_q = self.extract_feats(img_q, self.backbone, self.feat_ids)
            else:
                raise ValueError(f"Unknown backbone type {self.backbone_type}")
            return feature_q

        feature_s_list = []

        if self.backbone_type == 'DINOv3':
            for k in range(len(img_s_list)):
                feats = self.extract_feats(img_s_list[k], self.backbone, self.feat_ids)
                feature_s_list.append(feats)
            # - tta_flag=True: extract support features only
            if tta_flag: return feature_s_list

            feature_q = self.extract_feats(img_q, self.backbone, self.feat_ids)

        else:
            raise ValueError(f"Unknown backbone type {self.backbone_type}")

        return self.ssp_func(img_s_list, mask_s_list, img_q, mask_q, feature_s_list, feature_q)


    def get_feat_and_attn(self, img, return_attn: bool = False):
        """
        return_attn=False -> feat
        return_attn=True  -> (feat, A_pp or None)
        """

        last_id = self.feat_ids[-1] if isinstance(self.feat_ids, (list, tuple)) else int(self.feat_ids)

        if return_attn: setattr(self.backbone, "_last_attn_maps", {})

        out = self.extract_feats(img, self.backbone, self.feat_ids, return_attn=return_attn)

        if not return_attn: return out

        if isinstance(out, tuple):
            feat, second = out
            if isinstance(second, dict):
                attn = second.get(last_id, None)
                if attn is None and len(second) > 0: attn = next(iter(second.values()))
            else:
                attn = second
            return feat, attn
        else:
            feat = out
            maps = getattr(self.backbone, "_last_attn_maps", {})
            attn = None
            if isinstance(maps, dict):
                attn = maps.get(last_id, None)
                if attn is None and len(maps) > 0: attn = next(iter(maps.values()))
            return feat, attn


    def ssp_func(self, img_s_list, mask_s_list, img_q, mask_q, feature_s_list, feature_q):
        h, w = img_q.shape[-2:]

        feature_fg_list, feature_bg_list, supp_out_ls = [], [], []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg); feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0
                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        FP = torch.cat(feature_fg_list, dim=1).squeeze(0).mean(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.cat(feature_bg_list, dim=1).squeeze(0).mean(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        out_0 = self.similarity_func(feature_q, FP, BP)

        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)

        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0
            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)

            supp_out = torch.cat(supp_out_ls, 0)
            out_ls.append(self_out); out_ls.append(supp_out)

        return out_ls


    def SSP_func(self, feature_q, out):
        bs, C, f_h, f_w = feature_q.shape
        pred_1 = out.softmax(1).view(bs, 2, -1)
        pred_fg, pred_bg = pred_1[:, 1], pred_1[:, 0]  # [B, H'*W']

        fg_ls, bg_ls, fg_local_ls, bg_local_ls = [], [], [], []
        for epi in range(bs):
            fg_thres, bg_thres = 0.7, 0.6
            cur_feat = feature_q[epi].view(C, -1)

            fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)] if (pred_fg[epi] > fg_thres).sum() > 0 \
                      else cur_feat[:, torch.topk(pred_fg[epi], 12).indices]
            bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)] if (pred_bg[epi] > bg_thres).sum() > 0 \
                      else cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            fg_proto, bg_proto = fg_feat.mean(-1), bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0)); bg_ls.append(bg_proto.unsqueeze(0))

            eps = 1e-6
            fg_feat_norm = fg_feat / (torch.norm(fg_feat, 2, 0, True) + eps)     # [C, N1]
            bg_feat_norm = bg_feat / (torch.norm(bg_feat, 2, 0, True) + eps)     # [C, N2]
            cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True) + eps)  # [C, N3]

            cur_feat_norm_t = cur_feat_norm.t()                         # [N3, C]
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0  # [N3, N1]
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # [N3, N2]
            fg_sim, bg_sim = fg_sim.softmax(-1), bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()).t().view(C, f_h, f_w).unsqueeze(0)
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()).t().view(C, f_h, f_w).unsqueeze(0)
            fg_local_ls.append(fg_proto_local); bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)   # [B,C,f_h,f_w,1,1]
        new_bg_local = torch.cat(bg_local_ls, 0)                               # [B,C,f_h,f_w]

        return new_fg, new_bg, new_fg_local, new_bg_local


    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        return torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0


    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        return torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
