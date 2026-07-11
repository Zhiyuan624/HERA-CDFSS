# -*- coding: utf-8 -*-
"""
 - extract_feat_dino_single: Extract single-layer features from the post-norm patch tokens of a specified block.
 - extract_feat_dino_fusion: Fuse features from multiple layers using softmax(beta * fusion_scores) weights, with optional FusionMLP refinement.
"""

from __future__ import annotations
from typing import List, Sequence, Dict, Optional, Tuple, Union

import os
import torch
from torch import Tensor

from util.modified_attention import patch_attn_on_layers


EPS: float = 1e-6
DEBUG_ATTN: bool = bool(int(os.getenv("DINO_DEBUG_ATTN", "0")))

# [Token Layout]
# The default layout assumes one CLS token followed by optional register tokens
# and spatial patch tokens. Update the token offset if another backbone uses a
# different special-token arrangement.
def attn_heads_to_patch_map(
    attn_bhxx: torch.Tensor,
    *,
    n_patch: int,                  
    make_symmetric: bool = True,
    normalize: str = "row",        
    n_register: int | None = None  
) -> torch.Tensor:
    assert attn_bhxx.dim() == 4, f"expect [B,H,N,N], got {tuple(attn_bhxx.shape)}"
    B, H, N, _ = attn_bhxx.shape  # torch.Size([1, 16, 630, 630])

    A = attn_bhxx.mean(dim=1)  # [B, N, N] torch.Size([1, 630, 630])

    if n_register is None:
        r = N - 1 - n_patch
        assert r >= 0, f"invalid tokens: N={N}, n_patch={n_patch} ⇒ r={r}<0"
        n_register = r
    else:
        assert 1 + n_register + n_patch <= N, \
            f"N={N} too small for CLS+{n_register}+{n_patch}"

    start = 1 + n_register
    end   = start + n_patch      
    A = A[:, start:end, start:end]  # [B, n_patch, n_patch] torch.Size([1, 625, 625])

    if make_symmetric:
        A = 0.5 * (A + A.transpose(-1, -2))

    if normalize == "softmax":
        A = torch.softmax(A, dim=-1)
    elif normalize == "row":
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    assert A.shape[-1] == n_patch and A.shape[-2] == n_patch
    return A


def _as_list(x: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, int): return [x]
    raise TypeError(f"feat_ids must be int or Sequence[int], got {type(x)}")


def _l2_norm_tokens(x: Tensor, dim: int = 1, eps: float = EPS) -> Tensor:
    n = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / n

# [Intermediate Feature Interface]
# The backbone must provide get_intermediate_layers() and return feature maps
# compatible with [B, C, H, W] when reshape=True. Adapt this function if the
# replacement model exposes intermediate features through another interface.
def _get_patch_maps_via_api(
    backbone,
    x: Tensor,
    layers: Sequence[int],
    *,
    reshape: bool = True,
    norm: bool = False,
    return_class_token: bool = False,
) -> List[Tensor]:

    if not hasattr(backbone, "get_intermediate_layers"):
        raise NotImplementedError(
            "Backbone does not implement the get_intermediate_layers interface."
        )

    outs = backbone.get_intermediate_layers(
        x, n=list(layers), reshape=reshape, return_class_token=return_class_token, norm=norm
    )

    return [p for (p, *_) in outs] if return_class_token else list(outs)


def extract_feat_dino_single(
    img: Tensor,
    backbone,
    feat_ids: Sequence[int],
    *,
    attn_strategy: str = "raw",
    use_token_norm: bool = True,
    return_attn: bool = False,
    force_fp32_attn: bool = False,
    gamma_temp: float = 1.8,
    sigma_k:   float = 3.0,
    sigma_q:   float = 5.0,
    alpha_k:   float = 0.25,
    alpha_q:   float = 0.45,
    alpha_min: float = 0.25,
    alpha_max: float = 0.45,
) -> Tensor | Tuple[Tensor, Dict[int, Tensor]]:

    layers = _as_list(feat_ids)
    assert len(layers) == 1, f"extract_feat_dino_single expects a single layer ID, but received {layers}"
    lid = layers[0]

    with patch_attn_on_layers(
        backbone, [lid],attn_strategy=attn_strategy,force_fp32_attn=force_fp32_attn, 
        record_even_if_raw=return_attn,gamma_temp=gamma_temp, sigma_k=sigma_k, 
        sigma_q=sigma_q, alpha_k=alpha_k, alpha_q=alpha_q, alpha_min=alpha_min, alpha_max=alpha_max,
    ):
        maps = _get_patch_maps_via_api(
            backbone, img, layers=[lid], reshape=True, norm=False, return_class_token=False
        )

        if return_attn:
            Araw = getattr(backbone.blocks[lid].attn, "last_attn", None)
            if Araw is not None:
                B, C, Hf, Wf = maps[0].shape  # torch.Size([1, 1024, 25, 25])
                A_pp = attn_heads_to_patch_map(Araw, n_patch=Hf * Wf, make_symmetric=True, normalize="row")
                if not hasattr(backbone, "_last_attn_maps"): backbone._last_attn_maps = {}
                backbone._last_attn_maps[lid] = A_pp

    # [Feature Map Format]
    # The extractor is expected to return a spatial feature map in [B, C, H, W].
    # Replacement backbones must preserve this format before entering SSP and PAC.
    feat = maps[0]  # [B, C, Hf, Wf]
    if use_token_norm: feat = _l2_norm_tokens(feat, dim=1)
    return feat


def extract_feat_dino_fusion(
    img: Tensor,
    backbone,
    feat_ids: Sequence[int],
    *,
    apply_fc: bool = False,
    fusion_mlp = None,
    attn_strategy: str = "raw",
    fusion_scores: Optional[List[float]] = None,
    beta: float = 26.0,        # Softmax temperature for feature fusion
    use_token_norm: bool = True,
    return_attn: bool = False,
    force_fp32_attn: bool = False,   
    gamma_temp: float = 1.8,
    sigma_k:   float = 3.0,
    sigma_q:   float = 5.0,
    alpha_k:   float = 0.25,
    alpha_q:   float = 0.45,
    alpha_min: float = 0.25,
    alpha_max: float = 0.45,
) -> Tensor | Tuple[Tensor, Dict[int, Tensor]]:
    layers = _as_list(feat_ids)
    last_id = layers[-1]

    with patch_attn_on_layers(
        backbone, [last_id], attn_strategy=attn_strategy,
        force_fp32_attn=force_fp32_attn, record_even_if_raw=return_attn, 
        gamma_temp=gamma_temp, sigma_k=sigma_k, sigma_q=sigma_q, alpha_k=alpha_k, 
        alpha_q=alpha_q, alpha_min=alpha_min, alpha_max=alpha_max,
    ):
        maps = _get_patch_maps_via_api(
            backbone, img, layers=layers, reshape=True, norm=False, return_class_token=False
        )

        if return_attn:
            Araw = getattr(backbone.blocks[last_id].attn, "last_attn", None)
            if Araw is not None:
                B, C, Hf, Wf = maps[-1].shape
                A_pp = attn_heads_to_patch_map(Araw, n_patch=Hf * Wf, make_symmetric=True, normalize="row")
                if not hasattr(backbone, "_last_attn_maps"): backbone._last_attn_maps = {}
                backbone._last_attn_maps[last_id] = A_pp

    feats = [_l2_norm_tokens(m, dim=1) if use_token_norm else m for m in maps]

    mode = getattr(backbone, "_rescale_mode", "none")  # 'none' | 'count' | 'norm'
    if mode == "norm":
        rmax = float(getattr(backbone, "_rescale_max", 1.0))
        feats = [
            (fi / (torch.sqrt(torch.mean(fi.pow(2), dim=(1, 2, 3), keepdim=True) + 1e-8)
             ).clamp_max(rmax).clamp_min(1e-8))
            for fi in feats
        ]
    elif mode == "count":
        pass

    L = len(layers)
    if fusion_scores is None:
        s = torch.zeros(L, device=feats[0].device, dtype=feats[0].dtype)
    else:
        sc = list(fusion_scores)
        if len(sc) == L - 1: sc = sc + [float(sum(sc)) / max(1, len(sc))]
        s = torch.tensor(sc, device=feats[0].device, dtype=feats[0].dtype)
        if s.numel() != L: s = torch.zeros(L, device=feats[0].device, dtype=feats[0].dtype)

    tau = float(getattr(backbone, "_dist_tau", 0.8))
    pivot = layers[-1]
    z = beta * (s - s.mean())  # [L]
    if tau > 0:
        d = torch.tensor([abs(l - pivot) for l in layers], device=z.device, dtype=z.dtype)
        z = z - d / tau  # log-space multiply by exp(-d/tau)

    w = torch.softmax(z, dim=0)
    w = torch.clamp(w, min=0.02 / len(w))
    w = w / w.sum()

    fused = torch.zeros_like(feats[0])
    for wi, fi in zip(w, feats): fused = fused + wi.view(1, 1, 1, 1) * fi

    if apply_fc and fusion_mlp is not None:
        B, C, H, W = fused.shape
        tokens = fused.permute(0, 2, 3, 1).reshape(B * H * W, C)
        tokens = fusion_mlp(tokens)
        fused = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    return fused
