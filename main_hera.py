# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# [Model Configuration]
# The default setup uses a 24-block ViT with candidate layers 12-23,
# layer 23 as the fusion anchor, and 1024-dimensional feature maps.
# Update the marked settings when these assumptions differ.
# -----------------------------------------------------------------------------

import copy
import argparse
from itertools import combinations

import torch
import torch.nn as nn

from common.logger import Logger, AverageMeter
from data_util.datasets import FSSDataset
from common import utils
from common.evaluation import Evaluator

from model.SSP_matching_mlp import SSP_MatchingNet
from util.refine_utils import RefineModule_Simple as RefineModule
from util.refine_voter import should_apply_refine_by_vote


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test with best-layer finetuning + refine voting (minimal-change)'
    )
    parser.add_argument('--dataset', type=str, default='pascal')

    # [Backbone Selection]
    # Select a backbone supported by SSP_MatchingNet. Additional backbones require
    # a corresponding model-loading and feature-extraction implementation.
    parser.add_argument('--backbone', type=str, default='DINOv3', choices=['DINOv2', 'DINOv3', 'CLIP'])
    
    parser.add_argument('--nshot', type=int, default=5)
    parser.add_argument('--benchmark', type=str, default='fss')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test_datapath', type=str, default='./data/fss')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--logfile', type=str, default='test_debug_log.txt')

    # [Layer Configuration]
    # This option specifies the initial feature layer only. The complete HLS
    # candidate range is defined separately in the layer-selection procedure.
    parser.add_argument('--feat_id', type=int, nargs='+', default=[12])
    parser.add_argument('--attn_strategy', type=str, default='dual_attn_gauss',
                        choices=['raw', 'dual_attn_gauss'])
    parser.add_argument('--refine', type=str, default='auto',
                        choices=['off', 'auto', 'always'])
    parser.add_argument('--fusion', type=str, default='on', choices=['on', 'off'])

    # [Checkpoint Configuration]
    # Repository and checkpoint settings for the default DINOv3 implementation.
    # Replace these values when using another model source or pretrained weight.
    parser.add_argument('--dinov3_backend', type=str, default='hub', choices=['hf', 'hub'])
    parser.add_argument('--dinov3_repo', type=str,
                        default='HERA/dinov3')
    parser.add_argument('--dinov3_ckpt', type=str,
                        default='/HERA/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    return parser.parse_args()


def soft_copy_paste(background_img, background_mask, instance_img, instance_mask):
    _, H, W = background_img.shape

    instance_img = instance_img[:, :H, :W]
    instance_mask = instance_mask[:H, :W]

    instance = instance_img * instance_mask.unsqueeze(0).float()

    h_i, w_i = instance.shape[1], instance.shape[2]
    if h_i < H and w_i < W:
        x = torch.randint(0, H - h_i, (1,)).item()
        y = torch.randint(0, W - w_i, (1,)).item()
    else:
        x, y = 0, 0

    background_img[:, x:x + h_i, y:y + w_i] += instance
    background_mask[x:x + h_i, y:y + w_i] += instance_mask

    background_mask = torch.clamp(background_mask, 0, 1)
    return background_img, background_mask


def simulate_multi_shot(img_list, mask_list, num_aug=2):
    new_imgs = [img_list[0].unsqueeze(0)]
    new_masks = [mask_list[0].unsqueeze(0)]
    for _ in range(num_aug):
        aug_img, aug_mask = soft_copy_paste(
            background_img=img_list[0].clone(),
            background_mask=mask_list[0].clone(),
            instance_img=img_list[0],
            instance_mask=mask_list[0]
        )
        new_imgs.append(aug_img.unsqueeze(0))
        new_masks.append(aug_mask.unsqueeze(0))
    return new_imgs, new_masks


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim, device=device) * init_values)

    def forward(self, x):
        return self.gamma * x


class FusionMLP(nn.Module):
    def __init__(self,
                 
                 # [Feature Dimension]
                 # C must match the channel dimension of the backbone feature
                 # maps. The default DINOv3 ViT-L backbone uses C=1024.
                 C: int = 1024,
                 drop: float = 0.0,
                 ls_init: float = 1e-3,
                 residual_scale: float = 0.30,
                 style: str = "v3",
                 pre_norm: bool = True,
                 zero_init: bool = False):
        super().__init__()
        self.style = style
        self.norm = nn.LayerNorm(C) if pre_norm else nn.Identity()
        self.mlp = Mlp(in_features=C, hidden_features=4 * C, out_features=C,
                       drop=drop, bias=True)
        self.ls2 = LayerScale(C, init_values=ls_init) if style == "v3" else nn.Identity()
        self.residual_scale = residual_scale

        if zero_init:
            nn.init.zeros_(self.mlp.fc2.weight)
            nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, x):
        y = self.mlp(self.norm(x))
        if self.style == "v3":
            y = self.ls2(y)
        return x + self.residual_scale * y


# Main pipeline
def test(model, dataloader, nshot, device, fusion_mlp, refine_module, args):
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset, device)
    ori_model = copy.deepcopy(model.state_dict())

    # ========= Fusion Candidate Generation: Dynamically Derived from the Best Single Layer l* (with Pivoting and Deduplication) =========
    def build_fusion_layer_map(l_star: int):

        # [Candidate Layer Range]
        # Layers 12-23 correspond to the latter half of the default 24-block ViT.
        # Update both bounds when using a backbone with a different depth.
        L_MIN, L_MAX = 12, 23
        assert L_MIN <= l_star <= L_MAX  # If L_MAX is changed, the hard-coded upper bound 23 here must also be changed.

        def _centered_window(center: int, W: int):
            r = (W - 1) // 2  # radius
            lo = max(L_MIN, center - r)
            hi = min(L_MAX, center + r)
            while (hi - lo + 1) < W:
                if lo > L_MIN:
                    lo -= 1
                elif hi < L_MAX:
                    hi += 1
                else:
                    break
            return list(range(lo, hi + 1))

        S24 = _centered_window(l_star, 7)                 
        S25 = _centered_window(l_star, 9)                

        S26_all = list(range(L_MIN, min(L_MAX, l_star + 3) + 1))  
        S26 = S26_all[:8] if len(S26_all) >= 8 else S26_all      

        S27_all = list(range(max(L_MIN, l_star - 3), L_MAX + 1))  
        S27 = S27_all[-8:] if len(S27_all) >= 8 else S27_all      

        S28_base = _centered_window(l_star, 9)            
        S28 = S28_base[::2] if len(S28_base) > 0 else []  

        S29_small = _centered_window(l_star, 5)           
        S29_big = _centered_window(l_star, 9)[::2]        
        S29 = sorted(set(S29_small + S29_big))            

        def _dedup(curr, prev_sets, bias: str):
            if any(curr == s for s in prev_sets):
                if bias == 'front':
                    if len(curr) > 0:
                        m = max(curr)
                        curr = [x for x in curr if x != m]
                        if len(curr) > 0 and min(curr) > L_MIN:
                            curr = [min(curr) - 1] + curr
                elif bias == 'tail':
                    if len(curr) > 0:
                        m = min(curr)
                        curr = [x for x in curr if x != m]
                        if len(curr) > 0 and max(curr) < L_MAX:
                            curr = curr + [max(curr) + 1]
                elif bias == 'sparse':
                    alt = S28_base[1::2] if curr == S28_base[::2] else S28_base[::2]
                    curr = alt if len(alt) > 0 else curr
            return sorted(set([x for x in curr if L_MIN <= x <= L_MAX]))

        uniq = []
        S24 = _dedup(S24, uniq, 'center'); uniq.append(S24)
        S25 = _dedup(S25, uniq, 'center'); uniq.append(S25)
        S26 = _dedup(S26, uniq, 'front');  uniq.append(S26)  
        S27 = _dedup(S27, uniq, 'tail');   uniq.append(S27)  
        S28 = _dedup(S28, uniq, 'sparse'); uniq.append(S28)  
        S29 = _dedup(S29, uniq, 'center'); uniq.append(S29)

        def _pack(core):
            body = sorted(set([l for l in core if L_MIN <= l <= L_MAX]))

            # [Fusion Anchor]
            # The final transformer block is retained as the global semantic anchor.
            # Replace 23 with the last valid block index for another model depth.
            if 23 not in body:
                body.append(23)
            body = [l for l in body if l != 23] + [23]

            if len(body) >= 23:
                body_no23 = body[:-1]
                keep = 19 - 1
                mid = len(body_no23) // 2
                half = keep // 2
                lo = max(0, mid - half)
                hi = min(len(body_no23), lo + keep)
                body = sorted(body_no23[lo:hi]) + [23]
            return body

        return {
            24: _pack(S24), 25: _pack(S25), 26: _pack(S26),
            27: _pack(S27), 28: _pack(S28), 29: _pack(S29)
        }

    # =============== Main loop ===============
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch, device=device)
        sup_rgb = batch['support_imgs'][0]
        sup_msk = batch['support_masks'][0]
        qry_rgb = batch['query_img']

        # 1-shot agumentation
        if nshot == 1:
            img_s_list, mask_s_list = simulate_multi_shot(sup_rgb, sup_msk, num_aug=2)
        else:
            img_s_list = [i.unsqueeze(0) for i in sup_rgb]
            mask_s_list = [i.unsqueeze(0) for i in sup_msk]
        real_nshot = len(img_s_list)

        model.eval()
        model.load_state_dict(ori_model)

        Logger.info(
            f"[Episode {idx}] Evaluating layers: single 12-23"
            f"{' + fusion 24-29' if args.fusion == 'on' else ''} ..."
        )
        Logger.info(f"➤ Scoring uses RAW (stable). Finetune/Inference uses: {args.attn_strategy}")

        support_miou_logs = []

        # ========= Phase A: single(12..23) =========
        single_layer_logs = []

        # [Candidate Layer Search]
        # Evaluate physical transformer blocks 12-23. This range must remain
        # consistent with L_MIN and L_MAX in build_fusion_layer_map().
        for lid in range(12, 24):
            model.extract_feats = (
                lambda lid=lid: (
                    lambda img, bb, ids, **kwargs:
                    model.extract_feat_dino_v0(
                        img, model.backbone, [lid], attn_strategy='raw', **kwargs
                    )
                )
            )()
            model.feat_ids = [lid]

            feature_s_list = model(img_s_list, mask_s_list, None, None, tta_flag=True)
            support_miou_list = []

            # leave-one-out
            for i in range(len(img_s_list)):
                f_s_new = feature_s_list[:i] + feature_s_list[i + 1:]
                img_s_new = img_s_list[:i] + img_s_list[i + 1:]
                mask_s_new = mask_s_list[:i] + mask_s_list[i + 1:]

                preds_list = model.ssp_func(
                    img_s_new, mask_s_new,
                    img_s_list[i], mask_s_list[i],
                    f_s_new, feature_s_list[i]
                )
                pred_mask = torch.argmax(preds_list[0], dim=1)  # [1,H,W]
                gt_mask = mask_s_list[i].squeeze(0)             # [H,W]

                pred_binary = (pred_mask == 1).float()
                gt_binary = (gt_mask == 1).float()
                intersection = (pred_binary * gt_binary).sum()
                union = (pred_binary + gt_binary).clamp(max=1).sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                support_miou_list.append(iou.item())

            avg_support_miou = sum(support_miou_list) / max(1, len(support_miou_list))
            single_layer_logs.append({'layer': lid, 'support_miou': avg_support_miou})

        # best single l*
        l_star_entry = max(single_layer_logs, key=lambda x: x['support_miou'])
        l_star = l_star_entry['layer']
        support_miou_logs.extend(single_layer_logs)
        Logger.info(f"[Episode {idx:04d}] Single-layer best l* = {l_star}")

        # ========= Phase B: fusion(24..29) =========
        if args.fusion == 'on':
            fusion_layer_map = build_fusion_layer_map(l_star)

            single_scores = {d['layer']: d['support_miou'] for d in single_layer_logs}

            setattr(model.backbone, "_debug_fusion", False)
            setattr(model.backbone, "_rescale_mode", "norm")  # 'none' / 'count' / 'norm'
            setattr(model.backbone, "_rescale_max", 1.0)
            setattr(model.backbone, "_dist_tau", 2.0)

            def make_extractor(feat_ids_for_mode, fusion_scores_list, beta=26.0):
                def _extract(img, bb, ids, **kwargs):
                    return model.extract_feat_dino_v1style_mlp(
                        img, model.backbone, feat_ids_for_mode,
                        apply_fc=False, fusion_mlp=None,
                        attn_strategy='raw',
                        fusion_scores=fusion_scores_list,
                        beta=beta,
                        use_token_norm=True,
                        **kwargs
                    )
                return _extract

            for lid in sorted(fusion_layer_map.keys()):  # 24..29
                fuse_layers = [l for l in fusion_layer_map[lid]]
                avg_ss = sum(single_scores.values()) / len(single_scores)
                fs = [single_scores.get(l, avg_ss) for l in fuse_layers]

                model.extract_feats = make_extractor(fusion_layer_map[lid], fs, beta=26.0)
                model.feat_ids = fusion_layer_map[lid]

                feature_s_list = model(img_s_list, mask_s_list, None, None, tta_flag=True)
                support_miou_list = []

                for i in range(len(img_s_list)):
                    f_s_new = feature_s_list[:i] + feature_s_list[i + 1:]
                    img_s_new = img_s_list[:i] + img_s_list[i + 1:]
                    mask_s_new = mask_s_list[:i] + mask_s_list[i + 1:]

                    preds_list = model.ssp_func(
                        img_s_new, mask_s_new,
                        img_s_list[i], mask_s_list[i],
                        f_s_new, feature_s_list[i]
                    )
                    pred_mask = torch.argmax(preds_list[0], dim=1)
                    gt_mask = mask_s_list[i].squeeze(0)

                    pred_binary = (pred_mask == 1).float()
                    gt_binary = (gt_mask == 1).float()
                    intersection = (pred_binary * gt_binary).sum()
                    union = (pred_binary + gt_binary).clamp(max=1).sum()
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    support_miou_list.append(iou.item())

                avg_support_miou = sum(support_miou_list) / max(1, len(support_miou_list))
                support_miou_logs.append({'layer': lid, 'support_miou': avg_support_miou})
        else:
            fusion_layer_map = {}
            Logger.info(f"[Episode {idx:04d}] Fusion OFF → 仅在单层(12-23)中选层")

        # ========= select best from "single" + "fusion" =========
        best_entry = max(support_miou_logs, key=lambda x: x['support_miou'])
        best_layer = best_entry['layer']
        Logger.info(f"[Episode {idx:04d}] Selected by support mIoU → best layer: {best_layer}")
        for log in support_miou_logs:
            name = f"L{log['layer']:02d}"
            score = log['support_miou']
            tag = "⭐" if log['layer'] == best_layer else "   "
            Logger.info(f"{tag} {name} | Support mIoU = {score * 100:.4f}")

        # ========= Fine-tuning branch =========
        miou_map = {d['layer']: d['support_miou'] for d in support_miou_logs}

        if best_layer in fusion_layer_map:
            # Fine-tune the fusion layer by updating only the FC and ls2 parameters in fusion_mlp.
            model.feat_ids = fusion_layer_map[best_layer]
            fuse_layers = model.feat_ids[:-1]
            fusion_scores = [miou_map[l] for l in fuse_layers]

            setattr(model.backbone, "_debug_fusion", False)
            setattr(model.backbone, "_rescale_mode", "norm")
            setattr(model.backbone, "_rescale_max", 1.0)
            setattr(model.backbone, "_dist_tau", 2.0)

            model.extract_feats = (
                lambda feat_ids=model.feat_ids, fs=fusion_scores, _attn=args.attn_strategy: (
                    lambda img, bb, ids, **kwargs:
                    model.extract_feat_dino_v1style_mlp(
                        img, model.backbone, feat_ids,
                        apply_fc=True, fusion_mlp=fusion_mlp,
                        attn_strategy=_attn, fusion_scores=fs, beta=26.0,
                        use_token_norm=True, **kwargs
                    )
                )
            )()

            for p in model.parameters():
                p.requires_grad = False
            for name, p in fusion_mlp.named_parameters():
                if name.startswith("mlp.fc") or name.startswith("ls2"):
                    p.requires_grad = True

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, fusion_mlp.parameters()),
                lr=1.3e-3
            )

        else:
            # Fine-tune the selected layer by updating only its FC parameters.
            model.extract_feats = (
                lambda bl=best_layer, _attn=args.attn_strategy: (
                    lambda img, bb, ids, **kwargs:
                    model.extract_feat_dino_v0(
                        img, model.backbone, [bl],
                        attn_strategy=_attn, **kwargs
                    )
                )
            )()
            model.feat_ids = [best_layer]

            for p in model.parameters():
                p.requires_grad = False
            
            # [Parameter Naming]
            # Single-layer TTA updates the selected block's MLP projection layers.
            # Update these patterns only if the backbone uses a different module hierarchy.
            for name, p in model.named_parameters():
                if (f'blocks.{best_layer}.mlp.fc1' in name or
                        f'blocks.{best_layer}.mlp.fc2' in name):
                    p.requires_grad = True

            optimizer = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=1.3e-3
            )
            Logger.info(
                f"[Episode {idx:04d}] Single layer selected: L{best_layer} → "
                f"Fine-tune FC of same layer"
            )

        # ========= Fine-tuning =========
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        model.train()
        for supp_num in range(1, real_nshot):
            optimizer.zero_grad()
            feature_s_list = model(img_s_list, mask_s_list, None, None, tta_flag=True)

            loss_list = []
            for i in range(real_nshot):  # 1shot: 3; 5shot: 5
                f_s_new = feature_s_list[:i] + feature_s_list[i + 1:]
                img_s_new = img_s_list[:i] + img_s_list[i + 1:]
                mask_s_new = mask_s_list[:i] + mask_s_list[i + 1:]

                index_ls = list(range(len(f_s_new)))
                for pidx in combinations(index_ls, supp_num):  # supp_num: 1shot:1,2; 5shot:1..4, Number of combinations：1shot：6+3=9；5shot：75；
                    f_s_list_p = [f_s_new[j] for j in pidx]
                    img_s_list_p = [img_s_new[j] for j in pidx]
                    mask_s_list_p = [mask_s_new[j] for j in pidx]

                    preds_list = model.ssp_func(
                        img_s_list_p, mask_s_list_p,
                        img_s_list[i], mask_s_list[i],
                        f_s_list_p, feature_s_list[i]
                    )
                    loss_i = criterion(preds_list[0], mask_s_list[i].long())
                    loss_list.append(loss_i)

            loss = sum(loss_list) / max(1, len(loss_list))
            Logger.info(
                f"[Finetune] Episode {idx} | Supp Num {supp_num} | "
                f"Loss: {loss.item():.4f}"
            )
            loss.backward()
            optimizer.step()

        # ========= Query inference + refine =========
        model.eval()
        with torch.no_grad():
            feature_s_list = model(img_s_list, mask_s_list, None, None, tta_flag=True)

            apply_refine_vote, stats = should_apply_refine_by_vote(
                model=model,
                img_s_list=img_s_list,
                mask_s_list=mask_s_list,
                feature_s_list=feature_s_list,
                refine_module=refine_module,
                dataset_name=args.benchmark,
                nshot=nshot
            )

            if args.refine == 'off':
                apply_refine = False
                decision_note = "refine=OFF"
            elif args.refine == 'always':
                apply_refine = True
                decision_note = "refine=ALWAYS"
            else:
                apply_refine = apply_refine_vote
                decision_note = f"refine=AUTO(Vote={apply_refine})"

        Logger.info(
            f"[Ep {idx:04d}] vote={stats['soft_votes']:.1f}/{stats['need_votes']:.1f} | "
            f"Δavg={stats['delta_avg']:.4f} -> {decision_note}"
        )

        with torch.no_grad():
            pred = model([x.unsqueeze(0) for x in sup_rgb],
                         [m.unsqueeze(0) for m in sup_msk],
                         qry_rgb, None)[0]
            if apply_refine:
                feat_q, attn_prior = model.get_feat_and_attn(qry_rgb, return_attn=True)
                try:
                    pred = refine_module(feat_q, pred, qry_rgb, attn_prior)
                except TypeError:
                    pred = refine_module(feat_q, pred, qry_rgb)

        pred_mask = torch.argmax(pred, dim=1)
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)

        miou_episode = ((area_inter + 1e-6) / (area_union + 1e-6))[1].mean().item() * 100
        Logger.info(f"[Episode {idx:04d}] Final Query mIoU: {miou_episode:.2f}")
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(args):
    Logger.initialize(logdir=args.logdir, logfile=args.logfile)

    FSSDataset.initialize(img_size=400, datapath=args.test_datapath)
    test_loader = FSSDataset.build_dataloader(
        args.benchmark, 1, 0, 'test', args.nshot, fold=args.fold
    )

    Logger.info(f"Testing with {len(test_loader)} episodes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSP_MatchingNet(args.backbone).to(device)
    fusion_mlp = FusionMLP().to(device)
    refine_module = RefineModule().to(device)  # Introduces no additional hyperparameters; the internal implementation can be freely modified or replaced.

    Evaluator.initialize()
    miou, fb = test(model, test_loader, args.nshot, device, fusion_mlp, refine_module, args)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (miou.item(), fb.item()))
    Logger.info('==================== Finished ====================')


if __name__ == '__main__':
    args = parse_args()
    main(args)