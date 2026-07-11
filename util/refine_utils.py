# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineModule_Simple(nn.Module):
    """
    - l_sim : cosine logit from feature prototypes (same as the original)
    - l_img : cosine logit from image-channel prototypes (L2-normalized per channel; works for pseudo-RGB/multi-channel inputs)
    - l_attn: if attn_prior is provided, apply one-step propagation (row-normalized) and convert to logits
    """
    def __init__(self,
                 w_sim: float = 0.35,
                 w_img: float = 0.3,
                 w_attn: float = 1.1,
                 calibrate_scale: bool = True,
                 calib_clip=(0.5, 2.0),
                 eps: float = 1e-6):
        super().__init__()
        self.w_sim = float(w_sim)
        self.w_img = float(w_img)
        self.w_attn = float(w_attn)
        self.calibrate_scale = bool(calibrate_scale)
        self.calib_clip = (float(calib_clip[0]), float(calib_clip[1]))
        self.eps = float(eps)

    # ---------- utils ----------
    @staticmethod
    def _sigmoid_to_logit(p: torch.Tensor, eps: float):
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    @staticmethod
    def _stdz(x: torch.Tensor):
        m = x.mean(dim=(2, 3), keepdim=True)
        s = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - m) / s

    def _calibrate_like(self, ref: torch.Tensor, inc: torch.Tensor):
        lo, hi = self.calib_clip
        num = ref.abs().mean()
        den = inc.abs().mean() + 1e-6
        scale = (num / den).clamp(min=lo, max=hi)
        return inc * scale

    # ---------- forward ----------
    @torch.no_grad()
    def forward(self,
                feat_q: torch.Tensor,         # [B,C,Hf,Wf]
                mask_logits: torch.Tensor,    # [B,2,Hm,Wm]
                query_img: torch.Tensor,      # [B,Ch,Hm,Wm](Ch≥1)
                attn_prior: torch.Tensor = None  # [B,N,N] or [N,N] or None (N=Hf*Wf)
                ):
        B, C, Hf, Wf = feat_q.shape
        _, _, Hm, Wm = mask_logits.shape
        eps = self.eps

        # 1) base prob & logit
        prob = mask_logits.softmax(dim=1)
        p0 = prob[:, 1:2]
        l0 = self._sigmoid_to_logit(p0, eps)

        # 2) p0 to token scale
        p0_tok = F.adaptive_avg_pool2d(p0, (Hf, Wf))  # [B,1,Hf,Wf]
        w_fg = p0_tok
        w_bg = 1.0 - w_fg
        denom_fg = (w_fg.sum(dim=(2, 3), keepdim=True) + eps)
        denom_bg = (w_bg.sum(dim=(2, 3), keepdim=True) + eps)

        # 3) feature prototypes → l_sim
        Fq = F.normalize(feat_q, dim=1)
        mu_fg = F.normalize((Fq * w_fg).sum(dim=(2, 3), keepdim=True) / denom_fg, dim=1)
        mu_bg = F.normalize((Fq * w_bg).sum(dim=(2, 3), keepdim=True) / denom_bg, dim=1)
        d_sim = (Fq * mu_fg).sum(dim=1, keepdim=True) - (Fq * mu_bg).sum(dim=1, keepdim=True)
        d_sim = self._stdz(d_sim)
        l_sim = F.interpolate(d_sim, size=(Hm, Wm), mode="bilinear", align_corners=False)

        # 4) image-vector prototypes→ l_img
        x_tok = F.adaptive_avg_pool2d(query_img, (Hf, Wf))     # [B,Ch,Hf,Wf]
        Xn = F.normalize(x_tok, dim=1)
        mu_img_fg = F.normalize((Xn * w_fg).sum(dim=(2, 3), keepdim=True) / denom_fg, dim=1)
        mu_img_bg = F.normalize((Xn * w_bg).sum(dim=(2, 3), keepdim=True) / denom_bg, dim=1)
        d_img = (Xn * mu_img_fg).sum(dim=1, keepdim=True) - (Xn * mu_img_bg).sum(dim=1, keepdim=True)
        d_img = self._stdz(d_img)
        l_img = F.interpolate(d_img, size=(Hm, Wm), mode="bilinear", align_corners=False)

        # 5) attention one-hop → l_attn
        l_attn = None
        if attn_prior is not None:
            A = attn_prior
            if A.dim() == 2:
                A = A.unsqueeze(0).expand(B, -1, -1)
            A = A.to(dtype=feat_q.dtype, device=feat_q.device)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
            p0_flat = p0_tok.view(B, -1)                          # [B,N]
            pattn = torch.bmm(A, p0_flat.unsqueeze(-1)).squeeze(-1).view(B, 1, Hf, Wf)
            l_attn = self._sigmoid_to_logit(
                F.interpolate(pattn, size=(Hm, Wm), mode="bilinear", align_corners=False), eps
            )

        # 6) scale calibration
        if self.calibrate_scale:
            l_sim = self._calibrate_like(l0, l_sim)
            l_img = self._calibrate_like(l0, l_img)
            if l_attn is not None:
                l_attn = self._calibrate_like(l0, l_attn)

        # 7) fuse(no gating)
        l_final = l0 + self.w_sim * l_sim + self.w_img * l_img
        if l_attn is not None and self.w_attn != 0.0:
            l_final = l_final + self.w_attn * l_attn

        # 8) return two-channel logits
        return torch.cat([-l_final, l_final], dim=1)
