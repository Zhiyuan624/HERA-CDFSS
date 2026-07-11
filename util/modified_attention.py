from __future__ import annotations
import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Token layout (default): [CLS (optional=1)] + [R register tokens] + [P patch tokens]
# Total tokens: N = (0/1) + R + P
#   P, side        : number of patch tokens and grid size (P = side * side)
#   R              : number of register tokens
#   patch_slice    : slice selecting patch tokens only (rows/cols)
#   reg_slice      : slice selecting register tokens only (rows/cols)
#   rows_slice     : slice selecting register + patch tokens (used for γ statistics)
# ---------------------------------------------------------------------
def _infer_patch_and_reg_from_N(N: int, n_regs_hint: int | None = None):
    if n_regs_hint is None:
        s = int(math.isqrt(max(1, N - 1)))  # 25
        P0 = s * s  # 625
        R0 = N - 1 - P0  # 4
        if R0 < 0:
            R0, P0 = 0, N - 1
        P, R = P0, R0
        s = int(math.isqrt(P))
    else:
        R = max(0, int(n_regs_hint))
        P_raw = max(1, N - 1 - R)
        s = int(math.isqrt(P_raw))
        P = s * s
        if P <= 0:
            s = int(math.isqrt(max(1, N - 1)))
            P = s * s
            R = N - 1 - P

    has_cls = (N - (R + P)) >= 1
    start = (1 if has_cls else 0) + R
    stop = start + P

    patch_slice = slice(start, stop)
    reg_start = (1 if has_cls else 0)
    reg_slice = slice(reg_start, reg_start + R)
    rows_slice = slice(reg_start, reg_start + R + P)  # reg+patch 行

    return P, s, R, patch_slice, reg_slice, rows_slice  # 625, 25, 4, slice(5,630), slice(1,5), slice(1,630)


# ---------------------------------------------------------------------
# ModifiedAttention
# Notes:
#   - attn_strategy='raw'             : standard QK attention
#   - attn_strategy='dual_attn_gauss' : apply a Gaussian prior reweighting within the patch×patch block
#   - mix_qk_kk (default: False)      : if enabled, logits = γ·KK + (1-γ)·QK (only for 'dual_attn_gauss')
#   - γ is computed from the entropy gap of rows (register+patch) to patch columns
#   - reweighting preserves the total mass to patch tokens (row-sum), only redistributing within the patch block
# Parameters and default values are kept unchanged; no logic is modified.
# ---------------------------------------------------------------------
class ModifiedAttention(nn.Module):
    def __init__(self, original_attn: nn.Module, attn_strategy: str = 'raw'):
        super().__init__()

        self._orig = original_attn
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = getattr(original_attn, "proj_drop", nn.Identity())
        self.q_norm = getattr(original_attn, "q_norm", None)
        self.k_norm = getattr(original_attn, "k_norm", None)

        self.num_heads = (
            getattr(original_attn, "num_heads", None)
            or getattr(original_attn, "heads", None)
            or getattr(original_attn, "n_heads", None)
        )
        if self.num_heads is None:
            raise AttributeError("no num_heads")

        embed_from_proj = getattr(self.proj, "in_features", None)
        embed_from_qkv = getattr(self.qkv, "in_features", None)
        self.embed_dim = embed_from_proj if embed_from_proj is not None else embed_from_qkv  # 1024
        self.head_dim = self.embed_dim // int(self.num_heads)  # 64
        self.scale = getattr(original_attn, "scale", (self.head_dim ** -0.5))  # 0.125

        self.attn_strategy = attn_strategy
        self.is_fusion_mode = False
        self.force_fp32_attn = False

        # Stable default for DINOv3 (preserves the original value)
        self.gamma_temp = 1.0
        self.sigma_k = 3.0
        self.sigma_q = 5.0
        self.alpha_k = 0.25  # High γ → lower α_k → reduced interference from the KK path → conservative calibration.
        self.alpha_q = 0.45  # Low γ → higher α_q → stronger reweighting through the QK path → aggressive calibration.        
        self.alpha_min = 0.25
        self.alpha_max = 0.45

        self.lambda_reg_for_gamma = 0.2  # register tokens，primarily used for global semantic routing.

        self.patch_mass_preserve = True

        # Structural hint (DINOv3 typically uses R=4)
        self.n_registers = None

        self.mix_qk_kk = False

        self.addition_cache = {}
        self.last_attn = None

    @staticmethod  # Rotary positional embedding: treat every two dimensions as a pair and rotate them by 90° in the 2D plane.
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    @staticmethod
    def _apply_rope_basic(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
        while sin.ndim < x.ndim:
            sin = sin.unsqueeze(0)
        return (x * cos) + (ModifiedAttention._rotate_half(x) * sin)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope):
        for name in ["apply_rope", "rope_apply", "apply_rotary"]:
            fn = getattr(self._orig, name, None)
            if callable(fn):
                try:
                    return fn(q, k, rope)
                except Exception:
                    pass
        for name in ["apply_rotary", "apply_to_qk", "forward", "__call__"]:
            fn = getattr(rope, name, None)
            if callable(fn):
                out = fn(q, k)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    return out[0], out[1]

        cos = sin = None
        if isinstance(rope, (tuple, list)) and len(rope) >= 2:
            cos, sin = rope[0], rope[1]
        elif isinstance(rope, dict):
            cos, sin = rope.get("cos", None), rope.get("sin", None)

        if cos is not None and sin is not None:
            cos, sin = cos.to(q.dtype).to(q.device), sin.to(q.dtype).to(q.device)
            q = self._apply_rope_basic(q, cos, sin)
            k = self._apply_rope_basic(k, cos, sin)
        return q, k

    # ---- Gaussian Prior Generation ----
    @staticmethod
    def _gaussian_window(h2m1: int, w2m1: int, std: float = 3.5, device=None, dtype=None) -> torch.Tensor:
        c = 1.0 / (std * math.sqrt(2))
        xs = torch.linspace(-(h2m1 - 1) / 2 * c, (h2m1 - 1) / 2 * c, steps=h2m1, device=device, dtype=dtype)
        ys = torch.linspace(-(w2m1 - 1) / 2 * c, (w2m1 - 1) / 2 * c, steps=w2m1, device=device, dtype=dtype)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        return torch.exp(-(X ** 2 + Y ** 2))

    @staticmethod
    def _prior_from_window(h: int, w: int, window: torch.Tensor) -> torch.Tensor:
        device, dtype = window.device, window.dtype
        eye_h = torch.eye(h, device=device, dtype=dtype)
        eye_w = torch.eye(w, device=device, dtype=dtype)
        m = torch.einsum('ij,kl->ijkl', eye_h, eye_w).permute(0, 3, 1, 2).contiguous()
        out = F.conv2d(
            m.view(-1, h, w).unsqueeze(1),
            window.unsqueeze(0).unsqueeze(1),
            padding='same'
        ).squeeze(1)
        return out.view(h * w, h * w)

    def forward(self, x: torch.Tensor, rope=None, attn_bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_fp32_attn:
            x = x.float()

        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B,H,N,D]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if rope is not None:
            try:
                q, k = self._apply_rope(q, k, rope)
            except Exception:
                pass

        P, side, R, patch_slice, reg_slice, rows_slice = _infer_patch_and_reg_from_N(
            N, self.n_registers if self.n_registers is not None else None
        )

        # ---- γ Gating
        k_rows = k[:, :, rows_slice, :]         # [B,H,R+P,D] slice(1, 630)
        q_rows = q[:, :, rows_slice, :]
        k_patch = k[:, :, patch_slice, :]       # [B,H,P,D] slice(5, 630)

        # num_heads=16，each head independently computes an [R+P, P] distribution matrix → inspect the distributions across all 16 heads.
        K_rp = (k_rows @ k_patch.transpose(-2, -1)) * self.scale  # [B,H,R+P,P]
        Q_rp = (q_rows @ k_patch.transpose(-2, -1)) * self.scale

        A_K_rp = F.softmax(K_rp, dim=-1)
        A_Q_rp = F.softmax(Q_rp, dim=-1)

        eps = 1e-8
        Hk_rows = -(A_K_rp.clamp_min(eps) * A_K_rp.clamp_min(eps).log()).sum(-1)  # [B,H,R+P]
        Hq_rows = -(A_Q_rp.clamp_min(eps) * A_Q_rp.clamp_min(eps).log()).sum(-1)

        lam = float(self.lambda_reg_for_gamma)
        lam = max(0.0, min(0.5, lam))

        if R > 0:
            Hk_reg_mean = Hk_rows[:, :, :R].mean(-1, keepdim=True)
            Hq_reg_mean = Hq_rows[:, :, :R].mean(-1, keepdim=True)
        else:
            Hk_reg_mean = 0.0
            Hq_reg_mean = 0.0

        Hk_patch_mean = Hk_rows[:, :, R:].mean(-1, keepdim=True)
        Hq_patch_mean = Hq_rows[:, :, R:].mean(-1, keepdim=True)

        Hk = (1 - lam) * Hk_patch_mean + lam * Hk_reg_mean  # Hq 大、Hk 小 ⇒ QK 更不确定而 KK 更确定
        Hq = (1 - lam) * Hq_patch_mean + lam * Hq_reg_mean

        gamma = torch.sigmoid((Hq - Hk) / self.gamma_temp).to(K_rp.dtype).unsqueeze(-1)  # [B,H,1,1]

        # ---- Base logits: QK by default ----
        logits_q = (q @ k.transpose(-2, -1)) * self.scale
        if (self.attn_strategy == 'dual_attn_gauss') and self.mix_qk_kk:
            logits_k = (k @ k.transpose(-2, -1)) * self.scale
            logits = gamma * logits_k + (1.0 - gamma) * logits_q
        else:
            logits = logits_q

        if attn_bias is not None:
            logits = logits + attn_bias

        A_raw = F.softmax(logits, dim=-1)  # [B,H,N,N]

        # ---- RAW ----
        if self.attn_strategy == 'raw':
            A = self.attn_drop(A_raw)
            out = (A @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            if self.force_fp32_attn:
                out = out.to(in_dtype)
            self.last_attn = A.detach()
            return out

        if self.attn_strategy != 'dual_attn_gauss':
            raise NotImplementedError(self.attn_strategy)

        key_k = (side, round(float(self.sigma_k), 6))  # Narrow → stronger local-neighborhood bias; wide → smoother, more global coverage.
        key_q = (side, round(float(self.sigma_q), 6))

        add_k = self.addition_cache.get(key_k)
        add_q = self.addition_cache.get(key_q)

        if (add_k is None) or (add_q is None):
            win_k = self._gaussian_window(
                2 * side - 1, 2 * side - 1, std=float(self.sigma_k), device=x.device, dtype=x.dtype
            )  # (49,49)
            win_q = self._gaussian_window(
                2 * side - 1, 2 * side - 1, std=float(self.sigma_q), device=x.device, dtype=x.dtype
            )
            add_k = self._prior_from_window(side, side, win_k).to(x.dtype)
            add_q = self._prior_from_window(side, side, win_q).to(x.dtype)
            if not self.training:
                self.addition_cache[key_k] = add_k
                self.addition_cache[key_q] = add_q

        Pk = add_k.clamp_min(1e-8)
        Pk = Pk / (Pk.sum(dim=-1, keepdim=True) + 1e-6)
        Pq = add_q.clamp_min(1e-8)
        Pq = Pq / (Pq.sum(dim=-1, keepdim=True) + 1e-6)

        g = gamma.squeeze(-1).squeeze(-1)               # [B,H]
        g = g.unsqueeze(-1).unsqueeze(-1)               # [B,H,1,1]
        P_prior = g * Pk.unsqueeze(0).unsqueeze(0) + (1-g) * Pq.unsqueeze(0).unsqueeze(0)  # [B,H,P,P]
        logP = torch.log(P_prior.clamp_min(1e-8)).to(A_raw.dtype)                          # [B,H,P,P]

        A = A_raw.clone()
        A_pp = A[:, :, patch_slice, patch_slice].clamp_min(1e-12)       # [B, H, P, P]; slice(5, 630) extracts the pure patch-to-patch submatrix.
        z = A_pp.sum(dim=-1, keepdim=True)                              # [B,H,P,1]

        base_k = self.alpha_k if not self.is_fusion_mode else (self.alpha_k + 0.02)
        base_q = self.alpha_q if not self.is_fusion_mode else (self.alpha_q + 0.02)
        alpha_h = (gamma * base_k + (1.0 - gamma) * base_q).clamp(self.alpha_min, self.alpha_max)  # [B,H,1,1]

        logits_pp = torch.log(A_pp) + alpha_h * logP                    # [B,H,P,P]
        A_pp_tilde = F.softmax(logits_pp, dim=-1)
        A[:, :, patch_slice, patch_slice] = z * A_pp_tilde

        A = self.attn_drop(A)
        self.last_attn = A.detach()

        out = (A @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if self.force_fp32_attn:
            out = out.to(in_dtype)
        return out


@contextlib.contextmanager
def patch_attn_on_layers(
    backbone,
    layer_ids,
    *,
    attn_strategy='raw',
    is_fusion_mode=False,
    force_fp32_attn=False,
    gamma_temp=1.5,
    sigma_k=3.5,
    sigma_q=5.0,
    alpha_k=0.10,
    alpha_q=0.15,
    alpha_min=0.00,
    alpha_max=0.20,
    lambda_reg_for_gamma=0.2,
    record_even_if_raw: bool = True,
    n_registers_hint: int | None = 4,
    mix_qk_kk: bool = False,
):
    if attn_strategy == 'raw' and not record_even_if_raw:
        yield
        return

    blocks = (
        getattr(backbone, "blocks", None)
        or getattr(getattr(backbone, "model", backbone), "blocks", None)
        or getattr(getattr(backbone, "transformer", backbone), "blocks", None)
    )
    if blocks is None:
        raise AttributeError("no backbone.blocks")

    originals = []
    try:
        lids = layer_ids if isinstance(layer_ids, (list, tuple)) else [layer_ids]
        for lid in lids:
            blk = blocks[int(lid)]
            originals.append((blk, blk.attn))

            pat = ModifiedAttention(blk.attn, attn_strategy=attn_strategy)
            pat._orig = blk.attn
            pat.is_fusion_mode = bool(is_fusion_mode)
            pat.force_fp32_attn = bool(force_fp32_attn)

            pat.gamma_temp = float(gamma_temp)
            pat.sigma_k, pat.sigma_q = float(sigma_k), float(sigma_q)
            pat.alpha_k, pat.alpha_q = float(alpha_k), float(alpha_q)
            pat.alpha_min, pat.alpha_max = float(alpha_min), float(alpha_max)
            pat.lambda_reg_for_gamma = float(lambda_reg_for_gamma)
            pat.n_registers = int(n_registers_hint) if n_registers_hint is not None else None
            pat.mix_qk_kk = bool(mix_qk_kk)

            blk.attn = pat
        yield
    finally:
        for blk, attn in originals:
            blk.attn = attn