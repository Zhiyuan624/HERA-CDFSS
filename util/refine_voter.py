import torch
import numpy as np
from typing import List, Tuple, Dict


@torch.no_grad()
def _compute_iou_from_logits(logits: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_mask = torch.argmax(logits, dim=1).float()  # [B, H, W]
    gt = gt_mask.float()                             # [B, H, W]
    inter = (pred_mask * gt).sum()
    union = (pred_mask + gt).clamp(max=1).sum()
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(iou.item())


@torch.no_grad()
def should_apply_refine_by_vote(model, img_s_list: List[torch.Tensor], mask_s_list: List[torch.Tensor], feature_s_list: List[torch.Tensor], 
                                refine_module, dataset_name: str = "fss", nshot: int = 5) -> Tuple[bool, Dict]:
    """
    img_s_list : List[[1,3,H,W]]
    mask_s_list: List[[1,H,W]]
    feature_s_list : List[[1,C,Hf,Wf]]
    """
    real_nshot = len(img_s_list)
    soft_votes = 0.0
    pos = near = neg = 0
    deltas: List[float] = []

    def _refine_with_optional_attn(feat_q, base_logits, img_q):
        try:
            _, attn_prior = model.get_feat_and_attn(img_q, return_attn=True)  # [1,784,784] or None
        except Exception:
            attn_prior = None

        try:
            return refine_module(feat_q, base_logits, img_q, attn_prior)
        except TypeError:
            return refine_module(feat_q, base_logits, img_q)

    # 1-shot
    if nshot == 1:
        f_q  = feature_s_list[0]       # [1,C,Hf,Wf]
        img_q = img_s_list[0]          # [1,3,H,W]
        gt_q  = mask_s_list[0].long()  # [1,H,W]

        f_s_new   = feature_s_list[1:]
        img_s_new = img_s_list[1:]
        msk_s_new = mask_s_list[1:]

        # base
        base_logits_list = model.ssp_func(img_s_new, msk_s_new, img_q, gt_q, f_s_new, f_q)
        base_logits = base_logits_list[0]
        iou_base = _compute_iou_from_logits(base_logits, gt_q)

        # refine
        ref_logits = _refine_with_optional_attn(f_q, base_logits, img_q)
        iou_ref = _compute_iou_from_logits(ref_logits, gt_q)

        delta = float(iou_ref - iou_base)
        deltas.append(delta)
        if delta >= 0.0:
            soft_votes += 1.0
            pos += 1

    # n-shot, n>=2
    else:
        for i in range(real_nshot):
            f_q  = feature_s_list[i]
            img_q = img_s_list[i]
            gt_q  = mask_s_list[i].long()

            f_s_new   = feature_s_list[:i] + feature_s_list[i+1:]
            img_s_new = img_s_list[:i]    + img_s_list[i+1:]
            msk_s_new = mask_s_list[:i]   + mask_s_list[i+1:]

            # base
            base_logits_list = model.ssp_func(img_s_new, msk_s_new, img_q, gt_q, f_s_new, f_q)
            base_logits = base_logits_list[0]
            iou_base = _compute_iou_from_logits(base_logits, gt_q)

            # refine
            ref_logits = _refine_with_optional_attn(f_q, base_logits, img_q)
            iou_ref = _compute_iou_from_logits(ref_logits, gt_q)

            delta = float(iou_ref - iou_base)
            deltas.append(delta)
            if delta >= 0.0:
                soft_votes += 1.0
                pos += 1

    # gate
    need_votes = 2.0 if nshot >= 5 else 1.0
    apply_refine = (soft_votes >= need_votes)

    stats = {
        "soft_votes": soft_votes,
        "need_votes": need_votes,
        "delta_avg": float(np.mean(deltas)) if deltas else 0.0,
        "pos": pos,
        "near": near,
        "neg": neg,
        "deltas": deltas,
    }
    return apply_refine, stats
