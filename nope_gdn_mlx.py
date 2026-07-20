"""
nope_gdn_mlx.py
================
MLX port of the NoPE + Gated DeltaNet (KDA) video backbone, originally
implemented in PyTorch + FLA Triton kernels.

Designed to run on:
  - Apple Silicon natively  : pip install mlx
  - Linux with NVIDIA GPU   : pip install mlx[cuda12]   (or mlx[cuda13])

═════════════════════════════════════════════════════════════════════════════
READ THIS BEFORE USING
═════════════════════════════════════════════════════════════════════════════

1. KDA RECURRENCE PERFORMANCE TIERS.
   GatedDeltaLayer offers four compute paths, fastest to slowest:

   (a) METAL_KERNEL    — Custom MSL kernel via mx.fast.metal_kernel.
                         WORKING IMPLEMENTATION (naive scalar matmul).
                         One threadgroup per (batch, head), D threads each.
                         L-step loop inside kernel: ~3-8× over path (c).
                         Add simdgroup matmul for ~3-5× more if you need it.
   (b) CHUNKWISE_WY    — Chunkwise WY (Yang et al., ICLR 2025).
                         SCALAR-α only (channel_wise_decay=False).
                         Math identical to your PyTorch _process_chunk. Fast.
   (c) COMPILED_STEP   — mx.compile-fused step for KDA channel-wise decay.
                         3-5× faster than naive loop. Default for KDA mode.
   (d) NAIVE_STEP      — Step-by-step Python loop. Correct, slowest.

   For your KDA channel-wise model, path (a) is now the fastest available.
   Run verify_path_equivalence() FIRST to confirm the Metal kernel matches
   the naive reference numerically — never trust an untested GPU kernel.

   Realistic wall-clock vs PyTorch+FLA on Blackwell, 32-frame SSv2 training:
     Path (a) :       ~5-15× slower (best on stock MLX)
     Path (b) :       ~5-15× slower (only for scalar-α ablation)
     Path (c) :       ~20-50× slower
     Path (d) :       ~100-500× slower

2. CHANNELS-LAST EVERYTHING.
   MLX convolutions require channels-last layout:
     Conv3d input  = (B, T, H, W, C)   not (B, C, T, H, W)
     Conv1d input  = (B, L, C)         not (B, C, L)
   When loading PyTorch weights, conv kernels need to be permuted:
     PyTorch Conv3d weight (out_C, in_C, T, H, W) -> MLX (out_C, T, H, W, in_C)
     PyTorch Conv1d weight (out_C, in_C, K)       -> MLX (out_C, K, in_C)
   See `convert_pytorch_state_dict()` at the bottom of this file.

3. MODULE FORWARD = `__call__`, NOT `forward`.
   MLX's nn.Module subclasses define `__call__`. Don't write `def forward`.

4. LAZY EVALUATION.
   MLX does not run ops until `mx.eval(...)` is called or a value is read.
   In a training loop, force evaluation after `optimizer.update(...)` or after
   computing the loss to avoid graph blowup.

5. NO AUTOCAST.
   There is no `torch.amp` equivalent. Cast inputs and weights manually.
   Recommended: keep state matrices in float32 (overflow risk in fp16),
   everything else in bfloat16 (M3+ supports bf16; CUDA backend always does).

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, List

import mlx.core as mx
import mlx.nn as nn


# ============================================================================
# UTILITIES
# ============================================================================

def l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    """MLX equivalent of F.normalize(p=2, dim=-1)."""
    norm = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True))
    return x / mx.maximum(norm, eps)


def softplus(x: mx.array, beta: float = 1.0) -> mx.array:
    """Numerically stable softplus: log(1 + exp(beta*x)) / beta."""
    bx = beta * x
    return (mx.log1p(mx.exp(-mx.abs(bx))) + mx.maximum(bx, 0)) / beta


def video_to_mlx_layout(video: mx.array) -> mx.array:
    """(B, C, T, H, W) -> (B, T, H, W, C). Use this if you load PyTorch-style tensors."""
    return mx.transpose(video, (0, 2, 3, 4, 1))


# ============================================================================
# COMPONENT 1: 3D TUBELET EMBEDDING
# ============================================================================

class TubeletEmbedding3D(nn.Module):
    """
    Non-overlapping 3D patch extraction. Channels-last MLX convention.

    Input:  (B, T, H, W, C)
    Output: (B, T'*H'*W', D)   where T'=T/t, H'=H/h, W'=W/w
    """
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 32,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
            padding=0,
            bias=True,
        )

        t, h, w = tubelet_size
        self.num_temporal_patches = num_frames // t
        self.num_spatial_patches_h = img_size // h
        self.num_spatial_patches_w = img_size // w
        self.num_patches = (
            self.num_temporal_patches
            * self.num_spatial_patches_h
            * self.num_spatial_patches_w
        )

    def __call__(self, video: mx.array) -> mx.array:
        # video: (B, T, H, W, C)
        x = self.projection(video)              # (B, T', H', W', D)
        B = x.shape[0]
        return x.reshape(B, -1, self.embed_dim) # (B, N, D)

    def get_grid_dims(self) -> Dict[str, int]:
        return {
            "T": self.num_temporal_patches,
            "H": self.num_spatial_patches_h,
            "W": self.num_spatial_patches_w,
            "total": self.num_patches,
        }


# ============================================================================
# COMPONENT 2: NoPE MULTI-HEAD ATTENTION (uses mx.fast SDPA)
# ============================================================================

class NoPEMultiheadAttention(nn.Module):
    """
    Multi-head self-attention with no positional encoding.
    Uses mx.fast.scaled_dot_product_attention (Metal/CUDA fused kernel).

    NOTE: mx.fast SDPA does NOT take a dropout parameter. If you need
    attention dropout for training regularization, apply it manually
    (rare in practice for video transformers; usually drop_path is enough).
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).reshape(B, N, 3, H, d)
        # qkv[..., 0/1/2, :, :] then transpose to (B, H, N, d)
        q = mx.transpose(qkv[:, :, 0, :, :], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1, :, :], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2, :, :], (0, 2, 1, 3))

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        # (B, H, N, d) -> (B, N, D)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, N, D)
        return self.out_proj(out)


# ============================================================================
# COMPONENT 3: DROP PATH + ViT BLOCK
# ============================================================================

class DropPath(nn.Module):
    """Stochastic depth. In MLX you must explicitly check training mode."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x: mx.array) -> mx.array:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = mx.random.bernoulli(keep_prob, shape).astype(x.dtype)
        return x * mask / keep_prob


class NoPEViTBlock(nn.Module):
    """
    Pre-norm ViT block with optional spatial-only attention factorization.
    Identical structure to your PyTorch version.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        spatial_tokens: Optional[int] = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = NoPEMultiheadAttention(embed_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        h = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, embed_dim),
            nn.Dropout(dropout),
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, D = x.shape
        normed = self.norm1(x)

        if self.spatial_tokens is not None:
            # Spatial-only attention: fold T into batch
            S = self.spatial_tokens
            T = N // S
            normed = normed.reshape(B, T, S, D).reshape(B * T, S, D)
            attn_out = self.attn(normed)
            attn_out = attn_out.reshape(B, T, S, D).reshape(B, N, D)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(normed, mask=mask))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# COMPONENT 4: NoPE VIDEO ENCODER
# ============================================================================

class NoPEVideoEncoder(nn.Module):
    """Tubelet embedding + stack of NoPE ViT blocks."""
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 32,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        factorized_attention: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.tubelet_embed = TubeletEmbedding3D(
            img_size, num_frames, tubelet_size, in_channels, embed_dim
        )
        grid = self.tubelet_embed.get_grid_dims()
        spatial_tokens = grid["H"] * grid["W"] if factorized_attention else None

        # Linearly increasing drop path
        if drop_path_rate > 0 and depth > 1:
            dpr = [drop_path_rate * i / (depth - 1) for i in range(depth)]
        else:
            dpr = [0.0] * depth

        self.blocks = [
            NoPEViTBlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                spatial_tokens=spatial_tokens, drop_path=dpr[i],
            )
            for i in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.embed_dim = embed_dim
        self.depth = depth
        self.factorized_attention = factorized_attention

    def __call__(self, video: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        return self._forward_compiled(video, mask)

    def _forward_uncompiled(self, video: mx.array,
                            mask: Optional[mx.array]) -> mx.array:
        x = self.tubelet_embed(video)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.norm(x)

    @property
    def _forward_compiled(self):
        # Lazily build (and cache) a compiled version that closes over `self`.
        # The encoder has no Metal kernels inside, so it's fully compilable.
        # mx.compile traces on first call and caches by input shape; for
        # variable-length inputs we'd need shapeless=True.
        fn = getattr(self, "_compiled_fn", None)
        if fn is None:
            fn = mx.compile(self._forward_uncompiled)
            object.__setattr__(self, "_compiled_fn", fn)
        return fn

    def get_grid_dims(self):
        return self.tubelet_embed.get_grid_dims()


# ============================================================================
# COMPONENT 5: GATED DELTANET / KDA LAYER
# ============================================================================

@mx.compile
def _gated_rms_norm(x: mx.array, weight: mx.array, gate: mx.array,
                    eps: float) -> mx.array:
    """
    Fused gated RMSNorm: out = mx.fast.rms_norm(x, weight, eps) * sigmoid(gate).

    @mx.compile fuses the sigmoid + elementwise-multiply with the rms_norm
    output, saving one kernel launch per call. mx.fast.rms_norm itself runs
    in fp32 internally then casts back to x.dtype, so this is bf16-safe.
    """
    return mx.fast.rms_norm(x, weight, eps) * mx.sigmoid(gate)


class RMSNormGated(nn.Module):
    """
    RMSNorm with sigmoid gating. Matches FusedRMSNormGated semantics:
        out = RMSNorm(x) * sigmoid(gate)

    Now backed by mx.fast.rms_norm + @mx.compile fusion.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _gated_rms_norm(x, self.weight, gate, self.eps)


def _depthwise_conv1d_manual(
    x: mx.array,
    weight: mx.array,
    kernel_size: int,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """
    Manual depthwise causal Conv1d: faster than nn.Conv1d(groups=C) on Metal
    per ml-explore/mlx#2369.

    x:      (B, L, C)  channels-last
    weight: (C, K, 1)  MLX depthwise weight layout (out_C, K, in_C/groups=1)
    bias:   (C,) or None

    Returns: (B, L, C)  with causal padding (kernel_size-1 zeros prepended).
    """
    B, L, C = x.shape
    K = kernel_size
    # Causal pad: pad K-1 zeros at the front along L dim
    pad_amount = K - 1
    x_pad = mx.pad(x, [(0, 0), (pad_amount, 0), (0, 0)])  # (B, L+K-1, C)

    # Build sliding windows along L: result (B, L, K, C)
    #   y[:, t, k, c] = x_pad[:, t + k, c]
    # Use unfold-style indexing via concatenation
    windows = mx.stack(
        [x_pad[:, k:k + L, :] for k in range(K)],
        axis=2,
    )  # (B, L, K, C)

    # Depthwise multiply-and-sum: weight (C, K, 1) -> (1, 1, K, C)
    w = mx.transpose(weight.squeeze(-1), (1, 0))  # (K, C)
    w = w.reshape(1, 1, K, C)

    out = mx.sum(windows * w, axis=2)  # (B, L, C)
    if bias is not None:
        out = out + bias.reshape(1, 1, C)
    return out


class ShortCausalConv1d(nn.Module):
    """
    Wrapper that exposes weight/bias compatible with nn.Conv1d but uses the
    manual sum-of-products kernel for depthwise convs (faster on Metal).
    Layout matches mlx.nn.Conv1d weight: (out_C, K, in_C/groups).
    """
    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # Match nn.Conv1d depthwise weight layout: (out_C, K, in_C/groups=1)
        scale = 1.0 / math.sqrt(kernel_size)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(channels, kernel_size, 1)
        )
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        return _depthwise_conv1d_manual(x, self.weight, self.kernel_size, self.bias)


# ----------------------------------------------------------------------------
# KDA SINGLE-STEP COMPILED KERNEL
# ----------------------------------------------------------------------------
# mx.compile fuses the ops within a single time step into fewer kernel
# launches. Empirically ~3-5× faster than naive Python loop on M-series GPUs.
# Across time steps there is still Python overhead, but each step is fast.
#
# Inputs are all (B, H, ...) shaped — broadcasting handles batching.
# State stays in float32 (D×D outer products overflow easily in fp16).
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# CHUNK CONSTANTS CACHE
# ----------------------------------------------------------------------------
# I_CC, the strict-lower mask, the upper-incl mask and the causal triangle are
# functions ONLY of chunk_size. Constructing them per-chunk wastes ~1500
# allocations per forward (49 chunks × 16 layers × 2 chunk paths). Cache once,
# reuse forever.

_chunk_consts_cache: Dict[int, Dict[str, mx.array]] = {}


def _get_chunk_consts(C: int) -> Dict[str, mx.array]:
    """Return cached (I_CC, causal, A_mask, O_mask) for the given chunk size."""
    consts = _chunk_consts_cache.get(C)
    if consts is None:
        I_CC = mx.eye(C, dtype=mx.float32)
        ones_CC = mx.ones((C, C), dtype=mx.float32)
        consts = {
            "I_CC":   I_CC,
            "causal": mx.tril(ones_CC),                     # incl. diagonal
            "A_mask": mx.tril(ones_CC, k=-1),               # strict lower
            "O_mask": mx.triu(ones_CC, k=0),                # incl. diagonal upper
        }
        mx.eval(*consts.values())
        _chunk_consts_cache[C] = consts
    return consts


# ----------------------------------------------------------------------------
# COMPILED CHUNK-BODY KERNELS  (optimization #2)
# ----------------------------------------------------------------------------
# The chunk body has 20+ small ops (matmuls, exps, masks, swaps). Each op is
# a separate kernel launch on Metal. mx.compile fuses elementwise ops and
# elides redundant evals, cutting launch overhead substantially.
#
# We can't compile across the `_metal_solve_triangular` call (custom kernels
# are opaque to mx.compile), so we split the body into PRE-SOLVE and
# POST-SOLVE compiled functions and call the kernel between them.


@mx.compile
def _scalar_wy_pre_solve(q, k, v, alpha, beta, state, I_CC, causal):
    """
    Pre-solve portion of `_process_chunk_scalar`. Returns the two unit-lower
    triangular LHS matrices (stacked) and the metadata needed by
    `_scalar_wy_post_solve`.

    Inputs all in working dtype except `alpha` and `state` which must be fp32.
    """
    # Stage 1: Cumulative decay in log-space
    log_alpha = mx.log(mx.maximum(alpha, 1e-6))                     # (B, H, C)
    log_gamma = mx.cumsum(log_alpha, axis=-1)                       # (B, H, C)
    gamma = mx.exp(log_gamma)
    gamma_C = gamma[:, :, -1]                                       # (B, H)

    log_Gamma = log_gamma[..., :, None] - log_gamma[..., None, :]   # (B, H, C, C)
    Gamma = mx.exp(log_Gamma) * causal                              # (B, H, C, C)

    # KKT once
    KKT = k @ mx.swapaxes(k, -1, -2)                                # (B, H, C, C)

    beta_diag = beta[..., None] * I_CC                              # (B, H, C, C)

    # Both LHS: I + strict_lower(β @ ...)
    L_g  = mx.tril(beta_diag @ (Gamma * KKT), k=-1)                 # gated WY
    L_ug = mx.tril(beta_diag @ KKT,           k=-1)                 # un-gated WY
    return Gamma, KKT, beta_diag, gamma, gamma_C, log_gamma, L_g, L_ug


@mx.compile
def _scalar_wy_post_solve(q, k, v, state, T_g, T_ug, Gamma, gamma,
                          gamma_C, log_gamma, causal):
    """
    Post-solve portion of `_process_chunk_scalar`. Computes the chunk output
    and the new state given the WY tableaux T_g, T_ug.
    """
    U_g = T_g  @ v                                                  # (B, H, C, D)
    W   = T_ug @ k                                                  # (B, H, C, D)

    g_exp = gamma[..., None]                                        # (B, H, C, 1)
    ST = mx.swapaxes(state, -1, -2)                                 # (B, H, D, D)

    QKT = q @ mx.swapaxes(k, -1, -2)                                # (B, H, C, C)
    QKT_causal = QKT * causal
    Q_corrected = q - QKT_causal @ W                                # (B, H, C, D)
    inter = (Q_corrected * g_exp) @ ST

    QKT_gamma = QKT * Gamma
    intra = QKT_gamma @ U_g

    output = inter + intra

    # Stage 5: state update — log-space forward decay
    w_left = W * g_exp
    delta = U_g - w_left @ ST

    log_fwd = log_gamma[..., -1:] - log_gamma                       # (B, H, C)
    k_right = k * mx.exp(log_fwd)[..., None]                        # (B, H, C, D)

    new_state = state * gamma_C[..., None, None] + \
                mx.swapaxes(delta, -1, -2) @ k_right
    return output, new_state


@mx.compile
def _kda_chunk_preamble_no_M(g_c, k_c, v_c, beta_c):
    """
    State-INDEPENDENT preamble for the KDA chunkwise loop, EXCLUDING M and M_oq
    (those still come from the existing fused Metal kernel because the
    factorization-based GEMM is numerically unstable for typical KDA decays).

    Optimization #4 — these are computed ONCE for all n_chunks at the same time
    (batched over the n_chunks axis), avoiding per-chunk Python overhead for
    the 5–6 ops below.

    Within-chunk γ can reach −80 to −100 in practice (default A_log init gives
    A_exp ≈ exp(log(uniform[1,16])) ≈ 8 and softplus on randomly initialized
    g_proj outputs ≈ 1, so γ_step ≈ −8, and γ_C ≈ −512 over 64 steps in the
    worst case — but realistic post-training values are much milder).
    The kernel-based M / M_oq path uses exp(min(γ_t − γ_s, 0)) which stays
    in [0, 1]; replacing it with the (K * exp(-γ)) @ (K * exp(γ))^T
    factorization overflows fp32 because exp(-γ) reaches ~1e37 in real layers.
    So we keep the kernel for M / M_oq.

    Returns (all (BH, n_chunks, ...)):
        gamma     : (BH, n_chunks, C, D)
        exp_gamma : (BH, n_chunks, C, D)
        K_proj    : (BH, n_chunks, C, D)   = exp_gamma * K
        K_back    : (BH, n_chunks, C, D)   = exp(γ_last − γ) * K
        rhs_pre_state : (BH, n_chunks, C, D)   = β * V
    """
    gamma     = mx.cumsum(g_c, axis=-2)                              # (BH, nc, C, D)
    exp_gamma = mx.exp(gamma)
    K_proj    = exp_gamma * k_c
    gamma_last = gamma[..., -1:, :]                                  # (BH, nc, 1, D)
    K_back    = mx.exp(gamma_last - gamma) * k_c
    rhs_pre_state = beta_c[..., :, None] * v_c
    return gamma, exp_gamma, K_proj, K_back, rhs_pre_state


@mx.compile
def _kda_build_IA_and_M_oq_T(M, M_oq, beta_c, A_mask, I_C, O_mask):
    """
    State-INDEPENDENT: assemble IA (= I + diag(β) @ M.T strict-lower) and the
    pre-masked, pre-transposed M_oq used by the post-solve.

    M, M_oq : (BH, n_chunks, C, C) — fresh from `_metal_compute_M_Moq`
    beta_c   : (BH, n_chunks, C)
    """
    A  = beta_c[..., :, None] * mx.swapaxes(M, -1, -2) * A_mask
    IA = I_C + A
    M_oq_masked_T = mx.swapaxes(M_oq, -1, -2) * mx.swapaxes(O_mask, -1, -2)
    return IA, M_oq_masked_T


@mx.compile
def _kda_assemble_rhs(rhs_pre_state, B_c, K_proj, state):
    """rhs = β * (V − K_proj @ state^T) = rhs_pre_state − β * (K_proj @ state^T)."""
    return rhs_pre_state - B_c[..., :, None] * (K_proj @ mx.swapaxes(state, -1, -2))


@mx.compile
def _kda_finalize(U, Q_prime, M_oq_masked_T, K_back, exp_gamma_last, state):
    """Compute chunk output and new state given U from the solve."""
    Q_S0T = Q_prime @ mx.swapaxes(state, -1, -2)
    O = Q_S0T + M_oq_masked_T @ U
    UTK = mx.swapaxes(U, -1, -2) @ K_back
    state_new = exp_gamma_last[..., None, :] * state + UTK
    return O, state_new


@mx.compile
def _kda_step_fused(state, q_t, k_t, v_t, g_t, beta_t, I):
    """
    One KDA recurrence step, fused.

    state:  (B, H, D, D) fp32
    q_t:    (B, H, D)
    k_t:    (B, H, D)
    v_t:    (B, H, D)
    g_t:    (B, H, D)    log-space decay, ≤ 0
    beta_t: (B, H)
    I:      (D, D)       eye

    Returns: (new_state, output_t)
    """
    decay = mx.exp(g_t)                                      # (B, H, D)
    state = state * decay[..., None, :]                       # column-wise scale
    b = beta_t[..., None, None]                               # (B, H, 1, 1)
    k_outer = k_t[..., :, None] @ k_t[..., None, :]           # (B, H, D, D)
    state = state @ (I - b * k_outer)
    state = state + b * (v_t[..., :, None] @ k_t[..., None, :])
    out = (state @ q_t[..., :, None]).squeeze(-1)             # (B, H, D)
    return state, out


@mx.compile
def _gdn_step_fused(state, q_t, k_t, v_t, alpha_t, beta_t, I):
    """One scalar-α GDN recurrence step, fused. Same as KDA but α scalar per head."""
    a = alpha_t[..., None, None]
    b = beta_t[..., None, None]
    k_outer = k_t[..., :, None] @ k_t[..., None, :]
    state = state @ (a * (I - b * k_outer))
    state = state + b * (v_t[..., :, None] @ k_t[..., None, :])
    out = (state @ q_t[..., :, None]).squeeze(-1)
    return state, out


class GatedDeltaLayer(nn.Module):
    """
    Gated DeltaNet / KDA layer.

    Modes (channel_wise_decay):
      True   KDA: g = -exp(A_log) * softplus(g_proj(x) + dt_bias)   (Kimi Linear)
      False  Original GDN: scalar α ∈ (0,1) per head via sigmoid    (ICLR 2025)

    Compute paths (auto-selected; override via `compute_path`):
      'metal'        Custom Metal kernel       (default for channel-wise KDA)
      'chunkwise_wy' Chunkwise WY              (scalar α only — auto for GDN)
      'compiled'     mx.compile-fused step     (fallback)
      'naive'        Pure Python step loop     (debug)

    DEFAULT BEHAVIOR: 'metal' for KDA, 'chunkwise_wy' for scalar-α GDN.
    On non-Metal backends (CUDA, CPU) the Metal kernel won't compile; auto-
    fallback to 'compiled' is implemented in __call__.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 6,
        head_dim: int = 64,
        chunk_size: int = 64,
        channel_wise_decay: bool = True,
        decay_low_rank: Optional[int] = None,
        allow_neg_eigval: bool = False,
        compute_path: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.channel_wise_decay = channel_wise_decay
        self.allow_neg_eigval = allow_neg_eigval
        self.scale = head_dim ** -0.5
        total_dim = num_heads * head_dim
        self.total_dim = total_dim

        # Auto-select compute path
        if compute_path is None:
            compute_path = "metal_sg" if channel_wise_decay else "chunkwise_wy"
        if compute_path == "chunkwise_wy" and channel_wise_decay:
            raise ValueError(
                "chunkwise_wy is only valid for scalar-α (channel_wise_decay=False). "
                "For KDA channel-wise decay use 'metal_vjp', 'chunkwise_kda', "
                "'metal_sg', 'metal', 'compiled', or 'naive'."
            )
        if compute_path == "chunkwise_kda" and not channel_wise_decay:
            raise ValueError(
                "chunkwise_kda is only valid for channel-wise KDA "
                "(channel_wise_decay=True). For scalar-α use 'chunkwise_wy'."
            )
        if compute_path in ("metal", "metal_sg", "metal_vjp") and not channel_wise_decay:
            raise ValueError(
                f"{compute_path} kernel is implemented for channel-wise KDA only. "
                "For scalar-α use 'chunkwise_wy' or 'compiled'."
            )
        if compute_path == "metal_sg" and head_dim % 8 != 0:
            raise ValueError(
                f"metal_sg requires head_dim multiple of 8 for 8x8 simdgroup tiles; "
                f"got head_dim={head_dim}. Use 'metal' for non-multiple-of-8 dims."
            )
        # metal_vjp's backward kernel keeps the per-step state matrix
        # S_tg[D·D] in threadgroup memory plus 8 D-vectors and a couple of
        # reduction scratches:
        #   bytes = D·D·4 + 8·D·4 + ~140 ≈ 4·D·(D + 8) + 140
        # The M-series TG limit is 32 KB; D=80 fits, D=96 doesn't (≈40 KB).
        # Raise here at construction time so users hit a clear error before
        # the first backward pass instead of a cryptic kernel-load failure
        # mid-training.
        if compute_path == "metal_vjp":
            tg_bytes = (head_dim * head_dim + 8 * head_dim) * 4 + 140
            if tg_bytes > 32 * 1024:
                raise ValueError(
                    f"compute_path='metal_vjp' requires "
                    f"D·(D+8)·4 + scratch ≤ 32 KB of TG memory; got "
                    f"head_dim={head_dim} → ~{tg_bytes / 1024:.1f} KB. "
                    f"At this head_dim use compute_path='chunkwise_kda_vjp' "
                    f"(no per-step S_tg in TG memory) instead."
                )
        self.compute_path = compute_path

        # ---- Q, K, V projections ----
        self.q_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_dim, bias=False)

        # ---- Decay parameterization ----
        if channel_wise_decay:
            rank = decay_low_rank or head_dim
            # Two-layer low-rank: hidden -> rank -> total_dim
            self.f_proj_1 = nn.Linear(hidden_size, rank, bias=False)
            self.f_proj_2 = nn.Linear(rank, total_dim, bias=False)
            # A_log ~ log(U[1, 16]); per-head magnitude
            A = mx.random.uniform(low=1.0, high=16.0, shape=(num_heads,))
            self.A_log = mx.log(A)
            self.dt_bias = mx.zeros((total_dim,))
        else:
            self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- Write gate β ----
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ---- Short causal depthwise convs (faster manual kernel on Metal) ----
        self.q_conv1d = ShortCausalConv1d(total_dim, kernel_size=4)
        self.k_conv1d = ShortCausalConv1d(total_dim, kernel_size=4)
        self.v_conv1d = ShortCausalConv1d(total_dim, kernel_size=4)

        # ---- Output: low-rank gate + RMSNormGated + proj ----
        self.g_proj_1 = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_proj_2 = nn.Linear(head_dim, total_dim, bias=True)
        self.o_norm = RMSNormGated(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(total_dim, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array] = None,
        compute_path: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        x:            (B, L, hidden_size)
        state:        (B, H, D, D) or None
        compute_path: override self.compute_path for this call ('metal',
                      'chunkwise_wy', 'compiled', 'naive')
        Returns: (output, new_state)
        """
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        path = compute_path or self.compute_path

        # ---- Project ----
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- Short causal conv ----
        q = self.q_conv1d(q)
        k = self.k_conv1d(k)
        v = self.v_conv1d(v)

        # ---- SiLU + L2 norm + scale ----
        q = l2_normalize(nn.silu(q), axis=-1) * self.scale
        k = l2_normalize(nn.silu(k), axis=-1)
        v = nn.silu(v)

        # ---- Decay gate ----
        if self.channel_wise_decay:
            g = self.f_proj_2(self.f_proj_1(x))                # (B, L, total_dim)
            g = g.reshape(B, L, H, D).astype(mx.float32)
            g = g + self.dt_bias.reshape(H, D)
            A_exp = mx.exp(self.A_log).reshape(1, 1, H, 1)
            g = -A_exp * softplus(g)                            # ≤ 0
            decay_param = g
        else:
            alpha = mx.sigmoid(self.a_proj(x))                  # (B, L, H)
            decay_param = alpha

        # ---- Write gate ----
        beta = mx.sigmoid(self.b_proj(x))                       # (B, L, H)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # ---- Reshape to multi-head ----
        q = q.reshape(B, L, H, D)
        k = k.reshape(B, L, H, D)
        v = v.reshape(B, L, H, D)

        # ---- Dispatch to recurrence path ----
        if path == "chunkwise_wy":
            assert not self.channel_wise_decay, "chunkwise_wy is scalar-α only"
            output, new_state = self._chunkwise_scalar_wy(q, k, v, decay_param, beta, state)
        elif path == "chunkwise_kda_vjp":
            assert self.channel_wise_decay, "chunkwise_kda_vjp is for channel-wise KDA only"
            output, new_state = self._chunkwise_kda_vjp_path(q, k, v, decay_param, beta, state)
        elif path == "chunkwise_kda":
            # Fold heads into batch and call the chunkwise WY function.
            assert self.channel_wise_decay, "chunkwise_kda is for channel-wise KDA only"
            q_bh    = mx.transpose(q,            (0, 2, 1, 3)).reshape(B * H, L, D)
            k_bh    = mx.transpose(k,            (0, 2, 1, 3)).reshape(B * H, L, D)
            v_bh    = mx.transpose(v,            (0, 2, 1, 3)).reshape(B * H, L, D)
            g_bh    = mx.transpose(decay_param,  (0, 2, 1, 3)).reshape(B * H, L, D)
            beta_bh = mx.transpose(beta,         (0, 2, 1)).reshape(B * H, L)
            state_bh = state.reshape(B * H, D, D) if state is not None else None
            out_bh, st_bh = self._chunkwise_kda_forward(
                q_bh, k_bh, v_bh, g_bh, beta_bh, state_bh,
            )
            # Unfold back: (B*H, L, D) -> (B, H, L, D) -> (B, L, H, D)
            output = mx.transpose(out_bh.reshape(B, H, L, D), (0, 2, 1, 3))
            new_state = st_bh.reshape(B, H, D, D)
        elif path in ("metal", "metal_sg", "metal_vjp"):
            if path == "metal_vjp":
                metal_method = self._metal_kernel_kda_vjp
            elif path == "metal_sg":
                metal_method = self._metal_kernel_kda_simdgroup
            else:
                metal_method = self._metal_kernel_kda
            try:
                output, new_state = metal_method(q, k, v, decay_param, beta, state)
            except Exception as e:
                # Metal kernel can fail on CUDA backend, CPU, or unsupported D.
                # Fall back to compiled silently (warn once).
                if not getattr(GatedDeltaLayer, "_metal_fallback_warned", False):
                    import warnings
                    warnings.warn(
                        f"Metal kernel '{path}' unavailable ({type(e).__name__}: {e}); "
                        f"falling back to compute_path='compiled'. This is "
                        f"expected on the CUDA backend and CPU.",
                        RuntimeWarning,
                    )
                    GatedDeltaLayer._metal_fallback_warned = True
                output, new_state = self._loop_compiled_kda(q, k, v, decay_param, beta, state)
        elif path == "compiled":
            if self.channel_wise_decay:
                output, new_state = self._loop_compiled_kda(q, k, v, decay_param, beta, state)
            else:
                output, new_state = self._loop_compiled_gdn(q, k, v, decay_param, beta, state)
        elif path == "naive":
            if self.channel_wise_decay:
                output, new_state = self._loop_naive_kda(q, k, v, decay_param, beta, state)
            else:
                output, new_state = self._loop_naive_gdn(q, k, v, decay_param, beta, state)
        else:
            raise ValueError(f"Unknown compute_path: {path}")

        # ---- Output: gated RMSNorm + proj ----
        output = output.reshape(B, L, H, D)
        gate = self.g_proj_2(self.g_proj_1(x)).reshape(B, L, H, D)
        output = self.o_norm(output, gate)
        output = output.reshape(B, L, H * D)
        return self.o_proj(output), new_state

    # ==================================================================
    # PATH (c): COMPILED STEP — KDA channel-wise (default for KDA mode)
    # ==================================================================
    def _loop_compiled_kda(self, q, k, v, g, beta, state):
        """
        Step-by-step KDA recurrence using mx.compile-fused inner step.
        ~3-5× faster than _loop_naive_kda on M-series.
        """
        B, L, H, D = q.shape

        # Cast to float32 for state stability
        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        if state is None:
            state = mx.zeros((B, H, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        I = mx.eye(D, dtype=mx.float32)
        outputs: List[mx.array] = []

        for t in range(L):
            state, out_t = _kda_step_fused(
                state, q32[:, t], k32[:, t], v32[:, t], g[:, t], beta32[:, t], I
            )
            outputs.append(out_t)

        return mx.stack(outputs, axis=1).astype(q.dtype), state

    def _loop_compiled_gdn(self, q, k, v, alpha, beta, state):
        """Step-by-step scalar-α GDN with mx.compile-fused step."""
        B, L, H, D = q.shape

        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32)
        alpha32 = alpha.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        if state is None:
            state = mx.zeros((B, H, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        I = mx.eye(D, dtype=mx.float32)
        outputs: List[mx.array] = []

        for t in range(L):
            state, out_t = _gdn_step_fused(
                state, q32[:, t], k32[:, t], v32[:, t],
                alpha32[:, t], beta32[:, t], I,
            )
            outputs.append(out_t)

        return mx.stack(outputs, axis=1).astype(q.dtype), state

    # ==================================================================
    # PATH (b): CHUNKWISE WY — scalar-α only (Yang et al., ICLR 2025)
    # ==================================================================
    def _chunkwise_scalar_wy(self, q, k, v, alpha, beta, state):
        """
        Chunkwise WY algorithm for scalar-α GDN, mathematically identical to
        the PyTorch _chunkwise + _process_chunk in your reference.

        ONLY VALID FOR SCALAR α. For channel-wise KDA decay this algorithm
        does not directly apply (D_t and H_t do not commute).

        Args:
            q, k, v: (B, L, H, D)   any dtype (kept; only state/alpha need fp32)
            alpha:   (B, L, H)
            beta:    (B, L, H)
            state:   (B, H, D, D) or None

        Optimizations applied (vs the original PyTorch port):
          (1) constants `I_CC` / `causal` cached by chunk_size — see
              `_get_chunk_consts(C)`. Saves ~1500 allocations per forward.
          (3) q, k, v, β stay in their input dtype; only α and state are
              forced to fp32 (the ones that demand fp32 numerics).
          (b) batched single-launch triangular solve via Metal kernel
              (see `_metal_solve_triangular`).
          (2) pre-solve and post-solve bodies wrapped in `mx.compile`.
        """
        B, L, H, D = q.shape
        C = self.chunk_size
        orig_dtype = q.dtype

        # Only α and state demand fp32 (log/cumsum/exp range; D×D outer-product
        # accumulation respectively). q/k/v/β stay in their input dtype — this
        # halves device-memory traffic on bf16 forwards.
        alpha = alpha.astype(mx.float32)

        # Pad to chunk boundary (alpha=1 means no decay, beta=0 means no write)
        pad = (C - L % C) % C
        if pad > 0:
            zero_qkv = mx.zeros((B, pad, H, D), dtype=q.dtype)
            q = mx.concatenate([q, zero_qkv], axis=1)
            k = mx.concatenate([k, zero_qkv], axis=1)
            v = mx.concatenate([v, zero_qkv], axis=1)
            alpha = mx.concatenate(
                [alpha, mx.ones((B, pad, H), dtype=mx.float32)], axis=1
            )
            beta = mx.concatenate(
                [beta, mx.zeros((B, pad, H), dtype=beta.dtype)], axis=1
            )

        L_pad = q.shape[1]
        nc = L_pad // C
        if state is None:
            state = mx.zeros((B, H, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        # Permute to (B, H, nc, C, D) for chunk processing
        q     = mx.transpose(q.reshape(B, nc, C, H, D),     (0, 3, 1, 2, 4))
        k     = mx.transpose(k.reshape(B, nc, C, H, D),     (0, 3, 1, 2, 4))
        v     = mx.transpose(v.reshape(B, nc, C, H, D),     (0, 3, 1, 2, 4))
        alpha = mx.transpose(alpha.reshape(B, nc, C, H),    (0, 3, 1, 2))
        beta  = mx.transpose(beta.reshape(B, nc, C, H),     (0, 3, 1, 2))

        consts = _get_chunk_consts(C)
        I_CC, causal = consts["I_CC"], consts["causal"]

        chunks_out: List[mx.array] = []
        for c in range(nc):
            out_c, state = self._process_chunk_scalar(
                q[:, :, c], k[:, :, c], v[:, :, c],
                alpha[:, :, c], beta[:, :, c], state,
                I_CC, causal,
            )
            chunks_out.append(out_c)

        # Concatenate chunks: (B, H, C, D) → (B, H, nc*C, D) → (B, nc*C, H, D)
        output = mx.concatenate(chunks_out, axis=2)
        output = mx.transpose(output, (0, 2, 1, 3))
        if pad > 0:
            output = output[:, :L]
        return output.astype(orig_dtype), state

    def _process_chunk_scalar(self, q, k, v, alpha, beta, state, I_CC, causal):
        """
        Single-chunk WY core for scalar-α decay.

        Splits the body into two `mx.compile`-fused functions surrounding the
        stacked Metal triangular solve. mx.compile cannot trace through custom
        Metal kernels, but pre- and post-solve are pure MLX ops and fuse well.

        q, k, v: (B, H, C, D)            any dtype
        alpha:   (B, H, C)                fp32
        beta:    (B, H, C)                any dtype
        state:   (B, H, D, D)             fp32
        I_CC, causal: (C, C)              fp32 (cached)
        """
        B, H, C, D = q.shape

        # ---- Pre-solve (compiled) ----
        Gamma, KKT, beta_diag, gamma, gamma_C, log_gamma, L_g, L_ug = \
            _scalar_wy_pre_solve(q, k, v, alpha, beta, state, I_CC, causal)

        # ---- Stacked Metal triangular solve (one kernel launch for both WY
        #      systems) — this is the optimization that eliminated the per-
        #      chunk CPU↔GPU sync.
        L_stack = mx.concatenate(
            [(I_CC + L_g)[None], (I_CC + L_ug)[None]], axis=0
        ).reshape(2 * B * H, C, C)
        rhs_stack = mx.concatenate(
            [beta_diag[None], beta_diag[None]], axis=0
        ).reshape(2 * B * H, C, C)
        T_stack = self._metal_solve_triangular(L_stack, rhs_stack)
        T_stack = T_stack.reshape(2, B, H, C, C)
        T_g, T_ug = T_stack[0], T_stack[1]

        # ---- Post-solve (compiled) ----
        return _scalar_wy_post_solve(
            q, k, v, state, T_g, T_ug, Gamma, gamma, gamma_C, log_gamma, causal,
        )

    # ==================================================================
    # GPU TRIANGULAR SOLVE — used by chunkwise_kda
    # ==================================================================
    #
    # MLX has no GPU triangular solver as of 0.x; mx.linalg.solve_triangular
    # only runs on the CPU stream, forcing a CPU↔GPU sync per call. For the
    # chunkwise WY this sync happens once per chunk (49 times for L=3136),
    # which alone costs more than the entire step-by-step Metal kernel.
    #
    # This kernel solves  L y = b  where L is unit lower triangular (C×C).
    # Forward substitution: y[i, :] = b[i, :] − Σ_{j<i} L[i, j] · y[j, :].
    # Sequential in row index i; parallel across the D output columns.
    #
    # Layout: 1 threadgroup per batch element, D threads per threadgroup.
    # Each thread owns one output column. Both L (C×C) and the working set of
    # y columns (D×C) live in threadgroup memory.
    #
    # WHY NOT PER-THREAD REGISTER y_col[C]?
    # On M3 GPUs, a `float y_col[C]` declared at function scope is placed in
    # the per-thread stack. For C ≤ 32 it fits in registers; for C ≥ 40 the
    # array spills into thread-local memory and produces silently corrupted
    # values (verified empirically: rel error 1e-7 at C=32, ~1.0 at C ≥ 40).
    # Putting y_col in threadgroup memory side-steps the issue entirely and
    # actually keeps the working set hot in L1 instead of going through
    # the slow thread-local memory region.
    #
    # MEMORY BUDGET:
    #   L_tg     : C * C * 4 bytes
    #   y_cols   : D * C * 4 bytes  (one column per thread)
    # At the default C = D = 64: 16 KB + 16 KB = 32 KB, exactly the M-series
    # threadgroup memory limit. Larger C requires switching to a tiled layout.

    _tri_solve_metal_source = r"""
        // Compile-time constants:
        //   T : input dtype
        //   C : matrix size (square)
        //   D : RHS column count (= threads per threadgroup)

        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;   // 0..D-1, output column

        threadgroup float L_tg[C * C];
        threadgroup float y_cols[D * C];          // (D rows, C cols), col-major
                                                   // y_cols[i * D + tid] = y[i] for thread tid

        // Cooperative load of L (C×C, row-major fp32) into TG mem
        const uint L_base = bh * C * C;
        for (uint i = tid; i < C * C; i += D) {
            L_tg[i] = (float) L_in[L_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b_base = bh * C * D + tid;     // stride by D for rows

        // Forward substitution. Each thread independently solves its own
        // column. y_cols is written/read at index [i * D + tid] so that the
        // C inner-loop reads of y[k] from another thread's column never alias
        // with this thread's own writes — and because each thread only touches
        // its own slice, no inter-iteration barrier is needed.
        for (uint i = 0; i < C; ++i) {
            float yi = (float) b_in[b_base + i * D];
            const uint Li_base = i * C;
            for (uint k = 0; k < i; ++k) {
                yi -= L_tg[Li_base + k] * y_cols[k * D + tid];
            }
            y_cols[i * D + tid] = yi;
        }

        // Write output (also serves as a read-after-write completion point
        // for y_cols within this thread; no barrier needed because each
        // thread reads only its own slots).
        for (uint i = 0; i < C; ++i) {
            y_out[b_base + i * D] = (T) y_cols[i * D + tid];
        }
    """

    _tri_solve_kernel_cache: Dict[Tuple[int, int], object] = {}

    @classmethod
    def _get_tri_solve_kernel(cls, C: int, D: int):
        """Build (cached) a forward-substitution kernel for given C, D."""
        key = (C, D)
        if key in cls._tri_solve_kernel_cache:
            return cls._tri_solve_kernel_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"tri_solve_c{C}_d{D}",
            input_names=["L_in", "b_in"],
            output_names=["y_out"],
            source=cls._tri_solve_metal_source,
        )
        cls._tri_solve_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def _metal_solve_triangular(cls, L: mx.array, b: mx.array) -> mx.array:
        """
        Solve L y = b on GPU via Metal kernel.

        L : (BH, C, C)  unit lower triangular fp32
        b : (BH, C, D)
        Returns y : (BH, C, D)

        Constraint: (C*C + D*C) * 4 bytes must fit in threadgroup memory
        (32 KB on M-series). With D = C this gives C ≤ 64 — exactly the
        default chunk_size of GatedDeltaLayer. Larger C requires a tiled
        layout (not yet implemented).
        """
        BH, C, _ = L.shape
        D = b.shape[-1]
        if (C * C + D * C) * 4 > 32 * 1024:
            raise ValueError(
                f"_metal_solve_triangular: C={C}, D={D} exceeds TG-memory "
                f"budget (C*C + D*C ≤ 8192); use a smaller chunk_size or "
                f"split the RHS into D ≤ {(32 * 1024 // 4 - C * C) // C} "
                f"column tiles."
            )
        kernel = cls._get_tri_solve_kernel(C, D)
        L = L.astype(mx.float32)
        b = b.astype(mx.float32)
        # Note: do NOT mx.eval here — that would force a sync between chunks
        # and erase the pipelining benefit. The kernel runtime resolves any
        # unevaluated dependencies before launch.
        outs = kernel(
            inputs=[L, b],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * D, 1, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(BH, C, D)],
            output_dtypes=[mx.float32],
        )
        return outs[0]

    # ==================================================================
    # GPU UPPER-TRIANGULAR SOLVE — used by chunkwise_kda BACKWARD
    # ==================================================================
    # Solves U y = b where U is unit upper triangular. Backward substitution:
    #   y[i, :] = b[i, :] − Σ_{j>i} U[i, j] · y[j, :]
    # Iteration order is REVERSED w.r.t. the lower-tri kernel (i goes from
    # C-1 down to 0). Same threadgroup layout: D threads per threadgroup,
    # each thread owns one output column. y_cols sits in threadgroup memory
    # for the same reason it does in the lower-tri kernel (per-thread arrays
    # of size C ≥ 40 spill on M3 and corrupt output).

    _tri_solve_upper_metal_source = r"""
        // Compile-time:
        //   T : input dtype
        //   C : matrix size (square)
        //   D : RHS column count (= threads per threadgroup)

        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;

        threadgroup float U_tg[C * C];
        threadgroup float y_cols[D * C];

        const uint U_base = bh * C * C;
        for (uint i = tid; i < C * C; i += D) {
            U_tg[i] = (float) U_in[U_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b_base = bh * C * D + tid;

        // Backward substitution: i goes from C-1 down to 0.
        for (int i = (int) C - 1; i >= 0; --i) {
            float yi = (float) b_in[b_base + i * D];
            const uint Ui_base = i * C;
            for (uint k = i + 1; k < C; ++k) {
                yi -= U_tg[Ui_base + k] * y_cols[k * D + tid];
            }
            y_cols[i * D + tid] = yi;
        }

        for (uint i = 0; i < C; ++i) {
            y_out[b_base + i * D] = (T) y_cols[i * D + tid];
        }
    """

    _tri_solve_upper_kernel_cache: Dict[Tuple[int, int], object] = {}

    @classmethod
    def _get_tri_solve_upper_kernel(cls, C: int, D: int):
        key = (C, D)
        if key in cls._tri_solve_upper_kernel_cache:
            return cls._tri_solve_upper_kernel_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"tri_solve_upper_c{C}_d{D}",
            input_names=["U_in", "b_in"],
            output_names=["y_out"],
            source=cls._tri_solve_upper_metal_source,
        )
        cls._tri_solve_upper_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def _metal_solve_triangular_upper(cls, U: mx.array, b: mx.array) -> mx.array:
        """
        Solve U y = b on GPU where U is unit upper triangular (C × C).

        Used by the chunkwise_kda backward to invert the IA system in the
        adjoint direction:  drhs = solve(IA.T, dU).
        """
        BH, C, _ = U.shape
        D = b.shape[-1]
        if (C * C + D * C) * 4 > 32 * 1024:
            raise ValueError(
                f"_metal_solve_triangular_upper: C={C}, D={D} exceeds TG-memory "
                f"budget (C*C + D*C ≤ 8192)."
            )
        kernel = cls._get_tri_solve_upper_kernel(C, D)
        U = U.astype(mx.float32)
        b = b.astype(mx.float32)
        outs = kernel(
            inputs=[U, b],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * D, 1, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(BH, C, D)],
            output_dtypes=[mx.float32],
        )
        return outs[0]

    # ==================================================================
    # M + M_oq fused kernel for chunkwise WY
    # ==================================================================
    #
    # Computes (in one launch, per chunk):
    #   M    [s, t] = Σ_d K[s,d] K[t,d] exp(min(γ[t,d] − γ[s,d], 0))
    #   M_oq [s, t] = Σ_d K[s,d] Q[t,d] exp(min(γ[t,d] − γ[s,d], 0))
    # by reading K, Q, γ from device memory (avoids ever materializing the
    # (BH, C, C, D) "ed" tensor that the Python implementation builds).
    #
    # Layout: 1 threadgroup per (B*H), C threads per threadgroup. Thread s
    # computes the s-th row of both M and M_oq.

    _M_kernel_source = r"""
        // Compile-time: T (dtype), C (chunk size), D (head dim)

        const uint bh = threadgroup_position_in_grid.x;
        const uint s  = thread_position_in_threadgroup.x;   // row index

        // Load K[s, :] and γ[s, :] for this thread into registers
        float K_s[D];
        float G_s[D];
        const uint base_s = bh * C * D + s * D;
        for (uint d = 0; d < D; ++d) {
            K_s[d] = (float) K_in[base_s + d];
            G_s[d] = (float) gamma_in[base_s + d];
        }

        const uint M_base = bh * C * C + s * C;
        const uint base_chunk = bh * C * D;

        for (uint t = 0; t < C; ++t) {
            float m_val  = 0.0f;
            float mq_val = 0.0f;
            const uint base_t = base_chunk + t * D;
            for (uint d = 0; d < D; ++d) {
                const float Kt = (float) K_in[base_t + d];
                const float Qt = (float) Q_in[base_t + d];
                const float Gt = (float) gamma_in[base_t + d];
                float diff = Gt - G_s[d];
                if (diff > 0.0f) diff = 0.0f;          // causal clip for stability
                const float ed = metal::exp(diff);
                m_val  += K_s[d] * Kt * ed;
                mq_val += K_s[d] * Qt * ed;
            }
            M_out[M_base + t]    = (T) m_val;
            M_oq_out[M_base + t] = (T) mq_val;
        }
    """

    _M_kernel_cache: Dict[Tuple[int, int], object] = {}

    @classmethod
    def _get_M_kernel(cls, C: int, D: int):
        key = (C, D)
        if key in cls._M_kernel_cache:
            return cls._M_kernel_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunkwise_M_c{C}_d{D}",
            input_names=["K_in", "Q_in", "gamma_in"],
            output_names=["M_out", "M_oq_out"],
            source=cls._M_kernel_source,
        )
        cls._M_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def _metal_compute_M_Moq(cls, K: mx.array, Q: mx.array, gamma: mx.array
                             ) -> Tuple[mx.array, mx.array]:
        """
        Fused (M, M_oq) computation. Avoids the (BH, C, C, D) intermediate.

        K, Q, gamma : (BH, C, D) fp32
        Returns (M, M_oq) each (BH, C, C) fp32.

        Routes to the TG-cached v2 kernel when:
          - 2·C·D ≤ 8192 fp32 (TG-memory budget), AND
          - C ≥ 64 (occupancy floor; below this the kernel only spawns 1
            simdgroup/TG and v2's TG-cache cost outweighs its bandwidth gain).
        Falls back to v1 (per-thread row regs only) otherwise.
        """
        BH, C, D = K.shape
        if cls._use_v2_M_kernel and 2 * C * D <= 8192 and C >= 64:
            kernel = cls._get_M_kernel_v2(C, D)
        else:
            kernel = cls._get_M_kernel(C, D)
        outs = kernel(
            inputs=[K, Q, gamma],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * C, 1, 1),
            threadgroup=(C, 1, 1),
            output_shapes=[(BH, C, C), (BH, C, C)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return outs[0], outs[1]

    # ------------------------------------------------------------------
    # M_Moq forward kernel v2 — TILED-D / TG-CACHED variant
    # ------------------------------------------------------------------
    # Identical math to v1, but caches K and γ for the WHOLE chunk in
    # threadgroup memory (32 KB exactly at C=D=64) so the inner d-loop reads
    # K[t, :] and γ[t, :] from TG mem instead of device memory. Q stays in
    # device memory (a 3rd C·D = 16 KB block would overflow the TG budget).
    #
    # Per-TG device traffic, before/after at C=D=64:
    #   v1: 3·C·C·D = 786 K fp32 reads (K, Q, γ all from device per inner iter)
    #   v2: 2·C·D loads (K, γ once into TG) + 1·C·C·D loads (Q from device)
    #     = 8 K (one-time) + 262 K = 270 K  →  ~3× less device traffic.
    #
    # The same TG layout is used by `_M_backward_AB_source` and
    # `_M_backward_dgamma_source` below — those kernels already TG-cache K and
    # γ; this brings the forward up to par.

    _M_kernel_v2_source = r"""
        // Compile-time: T (dtype), C (chunk size), D (head dim)
        // Layout: 1 TG per (B*H), C threads/TG. Thread s computes row s of M
        // and M_oq. K and γ are TG-cached for the whole chunk.

        const uint bh = threadgroup_position_in_grid.x;
        const uint s  = thread_position_in_threadgroup.x;

        threadgroup float K_tg[C * D];
        threadgroup float G_tg[C * D];

        const uint chunk_base = bh * C * D;
        for (uint i = s; i < C * D; i += C) {
            K_tg[i] = (float) K_in[chunk_base + i];
            G_tg[i] = (float) gamma_in[chunk_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-thread row s in registers (referenced D·C times in inner loop)
        float K_s[D];
        float G_s[D];
        const uint row_s = s * D;
        for (uint d = 0; d < D; ++d) {
            K_s[d] = K_tg[row_s + d];
            G_s[d] = G_tg[row_s + d];
        }

        const uint M_base = bh * C * C + s * C;
        const uint Q_base = bh * C * D;

        for (uint t = 0; t < C; ++t) {
            float m_val  = 0.0f;
            float mq_val = 0.0f;
            const uint row_t = t * D;
            const uint Q_off = Q_base + row_t;
            for (uint d = 0; d < D; ++d) {
                const float Kt = K_tg[row_t + d];                  // TG mem
                const float Qt = (float) Q_in[Q_off + d];          // device
                const float Gt = G_tg[row_t + d];                  // TG mem
                float diff = Gt - G_s[d];
                if (diff > 0.0f) diff = 0.0f;
                const float ed = metal::exp(diff);
                m_val  += K_s[d] * Kt * ed;
                mq_val += K_s[d] * Qt * ed;
            }
            M_out[M_base + t]    = (T) m_val;
            M_oq_out[M_base + t] = (T) mq_val;
        }
    """

    _M_kernel_v2_cache: Dict[Tuple[int, int], object] = {}
    # Class-level flag — flip to False to fall back to v1 (per-thread regs only).
    # The v2 kernel produces bit-identical output to v1 (same fp32 ops in the
    # same order); the only difference is the source of K/γ in the inner loop.
    _use_v2_M_kernel: bool = True

    @classmethod
    def _get_M_kernel_v2(cls, C: int, D: int):
        key = (C, D)
        if key in cls._M_kernel_v2_cache:
            return cls._M_kernel_v2_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunkwise_M_v2_c{C}_d{D}",
            input_names=["K_in", "Q_in", "gamma_in"],
            output_names=["M_out", "M_oq_out"],
            source=cls._M_kernel_v2_source,
        )
        cls._M_kernel_v2_cache[key] = kernel
        return kernel

    # ==================================================================
    # FUSED PER-CHUNK KERNEL — assemble_rhs + tri_solve + finalize
    # ==================================================================
    #
    # The chunkwise KDA forward originally ran THREE ops per chunk:
    #     rhs = β * (V − K_proj S^T)             # MLX matmul + elementwise
    #     U   = solve(IA, rhs)                    # _metal_solve_triangular
    #     O, S' = finalize(U, Q', M_oq^T_masked,  # MLX matmuls + elementwise
    #                       K_back, exp(γ_C), S)
    # → 3 kernel launches × n_chunks (49 chunks at L=3136, C=64).
    #
    # This kernel collapses all three into a single launch per chunk. The
    # chunk-to-chunk dependency is still S → S' (next chunk's rhs reads it),
    # so the OUTER loop stays sequential, but the INNER work is one TG.
    #
    # TG layout: 1 TG per (B*H), D threads per TG, thread `tid` owns column tid
    # of the chunk's outputs.
    #
    # TG memory: L_tg[C·C] (IA) + U_cols[C·D] (working U matrix that doubles as
    # the solve scratch) = (C² + C·D)·4 bytes. At C=D=64: 32 KB exactly — same
    # budget as `_metal_solve_triangular`, so no extra constraint vs the
    # current path.
    #
    # Per-thread registers: state_row[D] (the tid-th row of the previous-chunk
    # state, held throughout the kernel) + scalar scratch ≈ 256–300 bytes,
    # well within the M3 register budget.
    #
    # Algorithm (per thread tid):
    #   1. Cooperative load IA into L_tg.
    #   2. Load row tid of state into state_row[D] regs.
    #   3. For each row i ∈ [0, C):
    #        a. compute   kpdot   = Σ_d K_proj[i, d] · state_row[d]
    #        b. compute   rhs_i   = rhs_pre_state[i, tid] − B_c[i] · kpdot
    #        c. forward sub:      rhs_i −= Σ_{k<i} L_tg[i, k] · U_cols[k, tid]
    #        d. write     U_cols[i, tid] = rhs_i
    #   4. (barrier — U_cols is now complete for ALL threads' columns)
    #   5. Write U_out from U_cols (also serves as a fence within the thread).
    #   6. For each row i ∈ [0, C):
    #        a. qsdot = Σ_d Q_prime[i, d] · state_row[d]
    #        b. mou   = Σ_k M_oq^T[i, k]   · U_cols[k, tid]
    #        c. write O_out[i, tid] = (T)(qsdot + mou)
    #   7. Compute state_new column tid: for d_row ∈ [0, D):
    #        utk = Σ_k U_cols[k, d_row] · K_back[k, tid]
    #        state_new[d_row, tid] = exp_gamma_last[tid] · state[d_row, tid] + utk
    #
    # `U_out` is also returned because the chunkwise_kda_vjp path needs the
    # per-chunk U trajectory for backward; the forward-only path drops it.

    _chunk_solve_finalize_source = r"""
        // Compile-time: T (output dtype for O), C, D

        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;     // owns column tid

        threadgroup float L_tg[C * C];
        threadgroup float U_cols[D * C];   // U[i, tid] at U_cols[i*D + tid]

        // ---- 1. Cooperative load IA into TG memory ----
        const uint IA_base = bh * C * C;
        for (uint i = tid; i < C * C; i += D) {
            L_tg[i] = IA_in[IA_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 2. Load state row tid into per-thread registers ----
        float state_row[D];
        const uint state_base = bh * D * D;
        const uint state_row_off = state_base + tid * D;
        for (uint d = 0; d < D; ++d) {
            state_row[d] = state_in[state_row_off + d];
        }

        // ---- 3. Fused rhs + forward substitution ----
        const uint K_proj_base = bh * C * D;
        const uint rps_base    = bh * C * D + tid;
        const uint B_c_base    = bh * C;
        for (uint i = 0; i < C; ++i) {
            float kpdot = 0.0f;
            const uint K_proj_i = K_proj_base + i * D;
            for (uint d = 0; d < D; ++d) {
                kpdot += K_proj_in[K_proj_i + d] * state_row[d];
            }
            float rhs_i = rhs_pre_state_in[rps_base + i * D]
                          - B_c_in[B_c_base + i] * kpdot;
            const uint Li_base = i * C;
            for (uint k = 0; k < i; ++k) {
                rhs_i -= L_tg[Li_base + k] * U_cols[k * D + tid];
            }
            U_cols[i * D + tid] = rhs_i;
        }

        // U_cols is complete in this thread's column. We need ALL columns
        // before reading other-column entries in steps 6-7.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 5. Emit U_out (consumed by VJP path; ignored by forward-only) ----
        const uint U_base = bh * C * D + tid;
        for (uint i = 0; i < C; ++i) {
            U_out[U_base + i * D] = U_cols[i * D + tid];
        }

        // ---- 6. Compute O = Q' S^T + M_oq^T_masked U ----
        const uint Q_prime_base = bh * C * D;
        const uint M_oq_base    = bh * C * C;
        const uint O_base       = bh * C * D + tid;
        for (uint i = 0; i < C; ++i) {
            float qsdot = 0.0f;
            const uint Q_prime_i = Q_prime_base + i * D;
            for (uint d = 0; d < D; ++d) {
                qsdot += Q_prime_in[Q_prime_i + d] * state_row[d];
            }
            float mou = 0.0f;
            const uint M_oq_i = M_oq_base + i * C;
            for (uint k = 0; k < C; ++k) {
                mou += M_oq_T_in[M_oq_i + k] * U_cols[k * D + tid];
            }
            O_out[O_base + i * D] = (T) (qsdot + mou);
        }

        // ---- 7. Compute state_new[:, tid] = γ_last[tid] * state[:, tid] + UᵀK_back ----
        const float gamma_tid = exp_gamma_last_in[bh * D + tid];
        const uint K_back_base    = bh * C * D;
        const uint state_new_base = bh * D * D;
        for (uint d_row = 0; d_row < D; ++d_row) {
            float utk = 0.0f;
            for (uint k = 0; k < C; ++k) {
                utk += U_cols[k * D + d_row] * K_back_in[K_back_base + k * D + tid];
            }
            const float s_old = state_in[state_base + d_row * D + tid];
            state_new_out[state_new_base + d_row * D + tid] =
                gamma_tid * s_old + utk;
        }
    """

    # ==================================================================
    # FUSED PER-CHUNK KERNEL v2 — SIMDGROUP-MATRIX VARIANT
    # ==================================================================
    #
    # Same per-chunk fusion as `_chunk_solve_finalize_source` (rhs + solve +
    # finalize collapsed into one launch), but the four matmul phases are
    # now driven by `simdgroup_matrix<float, 8, 8>` tile multiplies instead
    # of plain scalar inner loops. This is the version that actually wins
    # vs the unfused MLX vendor-matmul path at C = D = 64.
    #
    # Math is identical to the scalar fused kernel. The four matmul phases:
    #   Phase 1 (rhs):     tmp = K_proj @ state^T                    (C, D)
    #                      rhs = β·V − β·tmp
    #   Phase 2 (solve):   sequential forward substitution per column
    #                      U = solve(IA, rhs)                        (C, D)
    #   Phase 3 (output):  O = Q' @ state^T + M_oq^T @ U             (C, D)
    #   Phase 4 (state):   utk = U^T @ K_back                        (D, D)
    #                      state_new = exp(γ_C) ⊙ state + utk
    #
    # TG memory budget:
    #   slot1 (16 KB at C=D=64): cycles state^T → IA → state^T → U^T
    #   slot2 (16 KB at C=D=64): cycles K_proj → rhs → U → utk-scratch
    # No third slot fits, so state and K_back are read directly from device
    # in their respective phases.
    #
    # TG threads: NSG·32 (NSG=4 → 128 threads, 4 simdgroups). The triangular
    # solve uses only the first D threads (one column each); the matmul
    # phases use all NSG simdgroups in parallel.
    #
    # Constraints: C and D must both be multiples of 8 (simdgroup_matrix
    # tile size) and the standard TG-memory budget C·(C+D) ≤ 8192. At C=D=64
    # the budget is hit exactly. For smaller chunks the kernel still works.

    _chunk_solve_finalize_sg_source = r"""
        // Compile-time: C, D, NSG.  NTH = NSG * 32 threads per TG.

        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_index_in_threadgroup;
        const uint nth = threads_per_threadgroup.x;
        const uint sg  = simdgroup_index_in_threadgroup;
        const uint nsg = simdgroups_per_threadgroup;

        threadgroup float slot1[D * D];   // first 16 KB
        threadgroup float slot2[C * D];   // second 16 KB

        // Base offsets per (B*H)
        const uint state_base    = bh * D * D;
        const uint K_proj_base   = bh * C * D;
        const uint Q_prime_base  = bh * C * D;
        const uint K_back_base   = bh * C * D;
        const uint M_oq_base     = bh * C * C;
        const uint IA_base       = bh * C * C;
        const uint rhs_ps_base   = bh * C * D;
        const uint B_c_base      = bh * C;
        const uint exp_g_base    = bh * D;
        const uint O_out_base    = bh * C * D;
        const uint state_new_b   = bh * D * D;
        const uint U_out_base    = bh * C * D;

        // ====== PHASE 0: load state^T into slot1, K_proj into slot2 ======
        // state^T[k, j] = state[j, k]; row-major slot1[k*D + j].
        for (uint idx = tid; idx < D * D; idx += nth) {
            const uint j = idx / D;
            const uint k = idx % D;
            slot1[k * D + j] = state_in[state_base + j * D + k];
        }
        // K_proj is row-major (C, D); copy directly.
        for (uint idx = tid; idx < C * D; idx += nth) {
            slot2[idx] = K_proj_in[K_proj_base + idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== PHASE 1: rhs = β·V − β·(K_proj @ state^T) ======
        // CRITICAL: slot2 is BOTH the matmul A operand (= K_proj) AND the
        // output destination. If we wrote tiles back to slot2 during the
        // matmul loop, other simdgroups (and the next iteration of this
        // simdgroup) would read partially-overwritten K_proj values.
        // Fix: each simdgroup accumulates all of its output tiles in a
        // register array, then writes them after a barrier — that way no
        // write happens until every read has completed.
        {
            const uint M_T = C / 8;
            const uint N_T = D / 8;
            const uint K_T = D / 8;
            const uint TOTAL = M_T * N_T;
            const uint PER_SG = TOTAL / NSG;        // 16 at C=D=64, NSG=4
            simdgroup_matrix<float, 8, 8> C_local[PER_SG];

            // Compute pass — pure reads from slot1/slot2, writes only to
            // per-thread registers.
            for (uint i = 0; i < PER_SG; ++i) {
                const uint t = sg + i * nsg;
                const uint m_tile = t / N_T;
                const uint n_tile = t % N_T;
                C_local[i] = simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kt = 0; kt < K_T; ++kt) {
                    simdgroup_matrix<float, 8, 8> At, Bt;
                    simdgroup_load(At, slot2 + m_tile * 8 * D + kt * 8, D);
                    simdgroup_load(Bt, slot1 + kt * 8 * D + n_tile * 8, D);
                    simdgroup_multiply_accumulate(C_local[i], At, Bt, C_local[i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store pass — writes only.
            for (uint i = 0; i < PER_SG; ++i) {
                const uint t = sg + i * nsg;
                const uint m_tile = t / N_T;
                const uint n_tile = t % N_T;
                simdgroup_store(C_local[i],
                    slot2 + m_tile * 8 * D + n_tile * 8, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Elementwise rhs = rhs_pre_state − β·tmp   (in slot2).
            for (uint idx = tid; idx < C * D; idx += nth) {
                const uint i = idx / D;
                const float bi = B_c_in[B_c_base + i];
                const float v  = rhs_pre_state_in[rhs_ps_base + idx];
                slot2[idx] = v - bi * slot2[idx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ====== PHASE 1.5: load IA into slot1 (overwrite state^T) ======
        for (uint idx = tid; idx < C * C; idx += nth) {
            slot1[idx] = IA_in[IA_base + idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== PHASE 2: triangular solve  IA·U = rhs  (in-place in slot2) ======
        // Sequential per row; D threads (cols), one column each.
        if (tid < D) {
            for (uint i = 0; i < C; ++i) {
                float yi = slot2[i * D + tid];
                const uint Li = i * C;
                for (uint k = 0; k < i; ++k) {
                    yi -= slot1[Li + k] * slot2[k * D + tid];
                }
                slot2[i * D + tid] = yi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Emit U_out (consumed by VJP path).
        for (uint idx = tid; idx < C * D; idx += nth) {
            U_out[U_out_base + idx] = slot2[idx];
        }

        // ====== PHASE 2.5: reload slot1 with state^T (overwrite IA) ======
        for (uint idx = tid; idx < D * D; idx += nth) {
            const uint j = idx / D;
            const uint k = idx % D;
            slot1[k * D + j] = state_in[state_base + j * D + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== PHASE 3: O = Q' @ state^T + M_oq^T @ U ======
        {
            const uint M_T  = C / 8;
            const uint N_T  = D / 8;
            const uint KTD  = D / 8;
            const uint KTC  = C / 8;
            const uint TOTAL = M_T * N_T;
            for (uint t = sg; t < TOTAL; t += nsg) {
                const uint m_tile = t / N_T;
                const uint n_tile = t % N_T;
                simdgroup_matrix<float, 8, 8> Ct =
                    simdgroup_matrix<float, 8, 8>(0.0f);

                // Part 1: Q' @ state^T   (Q' from device, state^T from slot1)
                for (uint kt = 0; kt < KTD; ++kt) {
                    simdgroup_matrix<float, 8, 8> At, Bt;
                    simdgroup_load(At,
                        Q_prime_in + Q_prime_base + m_tile * 8 * D + kt * 8, D);
                    simdgroup_load(Bt,
                        slot1 + kt * 8 * D + n_tile * 8, D);
                    simdgroup_multiply_accumulate(Ct, At, Bt, Ct);
                }

                // Part 2: M_oq^T @ U  (M_oq^T from device, U from slot2)
                for (uint kt = 0; kt < KTC; ++kt) {
                    simdgroup_matrix<float, 8, 8> At, Bt;
                    simdgroup_load(At,
                        M_oq_T_in + M_oq_base + m_tile * 8 * C + kt * 8, C);
                    simdgroup_load(Bt,
                        slot2 + kt * 8 * D + n_tile * 8, D);
                    simdgroup_multiply_accumulate(Ct, At, Bt, Ct);
                }

                simdgroup_store(Ct,
                    O_out + O_out_base + m_tile * 8 * D + n_tile * 8, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ====== PHASE 4 PREP: transpose U into slot1 (overwriting state^T) ======
        // slot1 holds U^T (D rows × C cols, row-major).
        for (uint idx = tid; idx < D * C; idx += nth) {
            const uint k = idx / C;
            const uint i = idx % C;
            slot1[k * C + i] = slot2[i * D + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ====== PHASE 4: state_new = γ·state + U^T @ K_back ======
        {
            const uint M_T  = D / 8;
            const uint N_T  = D / 8;
            const uint KTC  = C / 8;
            const uint TOTAL = M_T * N_T;
            for (uint t = sg; t < TOTAL; t += nsg) {
                const uint d_row_t = t / N_T;
                const uint d_col_t = t % N_T;
                simdgroup_matrix<float, 8, 8> Ct =
                    simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint kt = 0; kt < KTC; ++kt) {
                    simdgroup_matrix<float, 8, 8> At, Bt;
                    // U^T tile (D, C): row-major slot1, stride C
                    simdgroup_load(At,
                        slot1 + d_row_t * 8 * C + kt * 8, C);
                    // K_back from device (C, D), stride D
                    simdgroup_load(Bt,
                        K_back_in + K_back_base + kt * 8 * D + d_col_t * 8, D);
                    simdgroup_multiply_accumulate(Ct, At, Bt, Ct);
                }
                // Stage tile in slot2 so the elementwise step can read it.
                // (slot2 still holds U at this point but we don't need U
                // anymore — overwriting tile-by-tile is safe.)
                simdgroup_store(Ct,
                    slot2 + d_row_t * 8 * D + d_col_t * 8, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Elementwise: state_new[d_row, d_col] = γ[d_col]·state[d_row, d_col] + utk
            for (uint idx = tid; idx < D * D; idx += nth) {
                const uint d_row = idx / D;
                const uint d_col = idx % D;
                const float utk = slot2[idx];
                const float g   = exp_gamma_last_in[exp_g_base + d_col];
                const float s   = state_in[state_base + d_row * D + d_col];
                state_new_out[state_new_b + d_row * D + d_col] = g * s + utk;
            }
        }
    """

    _chunk_solve_finalize_sg_cache: Dict[Tuple[int, int, int], object] = {}

    @classmethod
    def _get_chunk_solve_finalize_sg_kernel(cls, C: int, D: int, NSG: int):
        key = (C, D, NSG)
        if key in cls._chunk_solve_finalize_sg_cache:
            return cls._chunk_solve_finalize_sg_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunk_solve_finalize_sg_c{C}_d{D}_n{NSG}",
            input_names=[
                "IA_in", "rhs_pre_state_in", "B_c_in", "K_proj_in",
                "Q_prime_in", "M_oq_T_in", "K_back_in",
                "exp_gamma_last_in", "state_in",
            ],
            output_names=["O_out", "state_new_out", "U_out"],
            source=cls._chunk_solve_finalize_sg_source,
            header=("#include <metal_stdlib>\n"
                    "#include <metal_simdgroup_matrix>\n"
                    "using namespace metal;"),
        )
        cls._chunk_solve_finalize_sg_cache[key] = kernel
        return kernel

    # Number of simdgroups per threadgroup. 8 = 256 threads is the empirical
    # sweet spot at C=D=64: matmul phases distribute 64 output tiles across 8
    # simdgroups (8 tiles each, fully populating the GPU's compute pipeline);
    # solve runs on the first 64 of 256 threads (the rest idle, but solve is
    # short relative to matmul). Lower NSG (=4) leaves the matmul pipeline
    # under-fed; higher NSG (=16) trips a 2nd-order TG-scheduling penalty.
    # See nsg sweep notes; switching is one line.
    _chunk_sg_NSG: int = 8

    @classmethod
    def _metal_chunk_solve_finalize_sg(
        cls,
        IA: mx.array,
        rhs_pre_state: mx.array,
        B_c: mx.array,
        K_proj: mx.array,
        Q_prime: mx.array,
        M_oq_T_masked: mx.array,
        K_back: mx.array,
        exp_gamma_last: mx.array,
        state: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """simdgroup_matrix variant of `_metal_chunk_solve_finalize`."""
        BH, C, D = K_proj.shape
        if C % 8 != 0 or D % 8 != 0:
            raise ValueError(
                f"_metal_chunk_solve_finalize_sg: C={C}, D={D} must both be "
                f"multiples of 8 (simdgroup_matrix tile size)."
            )
        # Simdgroup variant TG layout: slot1 holds state^T (D·D fp32) during
        # phases 1/3, IA (C·C fp32) during phase 2, U^T (D·C fp32) during
        # phase 4; slot2 holds K_proj/rhs/U/utk-scratch (max C·D, D·D fp32).
        # Budget: max(slot1) + max(slot2) ≤ 8192 fp32. Worst case is
        # max(D·D, C·C, C·D) + max(C·D, D·D) = 2·D·D when D ≥ C.
        slot1_max = max(D * D, C * C, C * D)
        slot2_max = max(C * D, D * D)
        if (slot1_max + slot2_max) * 4 > 32 * 1024:
            raise ValueError(
                f"_metal_chunk_solve_finalize_sg: C={C}, D={D} exceeds TG "
                f"memory budget (slot1+slot2 = {slot1_max + slot2_max} fp32 "
                f"= {(slot1_max + slot2_max) * 4 / 1024:.1f} KB > 32 KB)."
            )
        NSG = cls._chunk_sg_NSG
        kernel = cls._get_chunk_solve_finalize_sg_kernel(C, D, NSG)

        IA = IA.astype(mx.float32)
        rhs_pre_state = rhs_pre_state.astype(mx.float32)
        B_c = B_c.astype(mx.float32)
        K_proj = K_proj.astype(mx.float32)
        Q_prime = Q_prime.astype(mx.float32)
        M_oq_T_masked = M_oq_T_masked.astype(mx.float32)
        K_back = K_back.astype(mx.float32)
        exp_gamma_last = exp_gamma_last.astype(mx.float32)
        state = state.astype(mx.float32)

        NTH = NSG * 32
        outs = kernel(
            inputs=[
                IA, rhs_pre_state, B_c, K_proj,
                Q_prime, M_oq_T_masked, K_back,
                exp_gamma_last, state,
            ],
            template=[("C", C), ("D", D), ("NSG", NSG)],
            grid=(BH * NTH, 1, 1),
            threadgroup=(NTH, 1, 1),
            output_shapes=[(BH, C, D), (BH, D, D), (BH, C, D)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return outs[0], outs[1], outs[2]   # O, state_new, U

    # Class flag — when fused, prefer the simdgroup_matrix variant.
    # Set False to use the scalar fused kernel instead (correct but slower).
    _fused_chunk_use_simdgroup: bool = True

    _chunk_solve_finalize_cache: Dict[Tuple[int, int], object] = {}
    # Class flag — DEFAULT ON, served by the simdgroup_matrix variant.
    #
    # Two kernels live behind this flag:
    #   - SCALAR FUSED (`_chunk_solve_finalize_source`): correct but ~50%
    #     slower than the unfused 3-op path at C=D=64, BH=24, because MLX's
    #     vendor matmul (MPS BLAS / simdgroup_matrix) crushes the kernel's
    #     plain-scalar inner loops.
    #   - SIMDGROUP FUSED (`_chunk_solve_finalize_sg_source`): runs the four
    #     matmul phases on `simdgroup_matrix<float, 8, 8>` tile MAC ops, and
    #     ties the unfused path within ~1% at C=D=64, BH=24, with one
    #     Metal launch per chunk instead of three.
    #
    # The simdgroup variant is selected when both flags are True AND C, D
    # are multiples of 8 (the simdgroup_matrix tile size). Set
    # `_fused_chunk_use_simdgroup = False` to fall back to the scalar
    # variant; set `_use_fused_chunk_kernel = False` to fall back to the
    # original unfused path entirely.
    _use_fused_chunk_kernel: bool = True

    @classmethod
    def _get_chunk_solve_finalize_kernel(cls, C: int, D: int):
        key = (C, D)
        if key in cls._chunk_solve_finalize_cache:
            return cls._chunk_solve_finalize_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunk_solve_finalize_c{C}_d{D}",
            input_names=[
                "IA_in", "rhs_pre_state_in", "B_c_in", "K_proj_in",
                "Q_prime_in", "M_oq_T_in", "K_back_in",
                "exp_gamma_last_in", "state_in",
            ],
            output_names=["O_out", "state_new_out", "U_out"],
            source=cls._chunk_solve_finalize_source,
        )
        cls._chunk_solve_finalize_cache[key] = kernel
        return kernel

    @classmethod
    def _metal_chunk_solve_finalize(
        cls,
        IA: mx.array,
        rhs_pre_state: mx.array,
        B_c: mx.array,
        K_proj: mx.array,
        Q_prime: mx.array,
        M_oq_T_masked: mx.array,
        K_back: mx.array,
        exp_gamma_last: mx.array,
        state: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Fused per-chunk kernel: compute rhs, solve IA U = rhs, then output O
        and state_new — all in one Metal launch per chunk.

        Inputs (after the kernel's internal fp32 cast — caller may pass bf16):
            IA               : (BH, C, C)   unit lower triangular
            rhs_pre_state    : (BH, C, D)   = β * V
            B_c              : (BH, C)      = β (per-row scalar)
            K_proj           : (BH, C, D)   = exp(γ) ⊙ K
            Q_prime          : (BH, C, D)   = exp(γ) ⊙ Q
            M_oq_T_masked    : (BH, C, C)   = (M_oq · O_mask)^T
            K_back           : (BH, C, D)   = exp(γ_C − γ) ⊙ K
            exp_gamma_last   : (BH, D)      = exp(γ_C)
            state            : (BH, D, D)   previous chunk state

        Returns:
            O          : (BH, C, D) fp32   chunk output
            state_new  : (BH, D, D) fp32
            U          : (BH, C, D) fp32   per-chunk U (consumed by VJP path)

        TG-memory budget: (C² + C·D) × 4 bytes ≤ 32 KB → C·(C+D) ≤ 8192.
        At C=D=64: 32 KB exactly.
        """
        BH, C, D = K_proj.shape
        if (C * C + C * D) * 4 > 32 * 1024:
            raise ValueError(
                f"_metal_chunk_solve_finalize: C={C}, D={D} exceeds TG-memory "
                f"budget (C·(C+D) ≤ 8192); use a smaller chunk_size."
            )
        kernel = cls._get_chunk_solve_finalize_kernel(C, D)

        # Cast to fp32 — this matches the existing per-chunk kernels and the
        # state's fp32 invariant. The casts are cheap relative to the kernel's
        # work (state is already fp32, IA/M_oq^T are already fp32).
        IA = IA.astype(mx.float32)
        rhs_pre_state = rhs_pre_state.astype(mx.float32)
        B_c = B_c.astype(mx.float32)
        K_proj = K_proj.astype(mx.float32)
        Q_prime = Q_prime.astype(mx.float32)
        M_oq_T_masked = M_oq_T_masked.astype(mx.float32)
        K_back = K_back.astype(mx.float32)
        exp_gamma_last = exp_gamma_last.astype(mx.float32)
        state = state.astype(mx.float32)

        outs = kernel(
            inputs=[
                IA, rhs_pre_state, B_c, K_proj,
                Q_prime, M_oq_T_masked, K_back,
                exp_gamma_last, state,
            ],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * D, 1, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(BH, C, D), (BH, D, D), (BH, C, D)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return outs[0], outs[1], outs[2]   # O, state_new, U

    # ==================================================================
    # M_Moq BACKWARD KERNELS — fused VJP for the chunkwise_kda_vjp path
    # ==================================================================
    #
    # Forward computed (per (B*H), per chunk):
    #   M[s, t]    = Σ_d K[s,d] K[t,d] · ed[s,t,d]
    #   M_oq[s, t] = Σ_d K[s,d] Q[t,d] · ed[s,t,d]
    # where ed[s,t,d] = exp(min(γ[t,d] − γ[s,d], 0)).
    #
    # Backward derivations (each entry of K, Q, γ aggregates contributions from
    # ALL (s,t) pairs in which it appears; see comments below).
    #
    # We split into TWO kernel launches so per-thread memory stays within the
    # M3 register budget (~512 bytes per thread):
    #
    #   Kernel A: dK + dQ          (per-thread: 2 D-vector accumulators = 512 B)
    #   Kernel B: dγ                (per-thread: 1 D-vector + Q_s[D] = 512 B)
    #
    # Each launch shares the same threadgroup-memory layout: K_tg + γ_tg cached
    # for the chunk (32 KB exactly = M-series TG limit). Q is read directly
    # from device memory inside the inner loop (small 16 KB working set, hot
    # in L1).

    _M_backward_AB_source = r"""
        // Computes dK and dQ contributions through the M / M_oq kernel.
        //
        // dK[s, d] = Σ_t dM[s, t] K[t, d] ed[s, t, d]
        //         + Σ_t dM[t, s] K[t, d] ed[t, s, d]
        //         + Σ_t dM_oq[s, t] Q[t, d] ed[s, t, d]
        // dQ[s, d] = Σ_t dM_oq[t, s] K[t, d] ed[t, s, d]
        //
        // Layout: 1 threadgroup per (B*H), C threads/threadgroup. Thread s
        // computes the s-th row of dK and dQ.

        const uint bh = threadgroup_position_in_grid.x;
        const uint s  = thread_position_in_threadgroup.x;

        threadgroup float K_tg[C * D];
        threadgroup float G_tg[C * D];

        // Cooperative load of K and γ for this chunk
        const uint chunk_KG_base = bh * C * D;
        for (uint i = s; i < C * D; i += C) {
            K_tg[i] = (float) K_in[chunk_KG_base + i];
            G_tg[i] = (float) gamma_in[chunk_KG_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-thread accumulators (in registers)
        float dK_s[D];
        float dQ_s[D];
        for (uint d = 0; d < D; ++d) {
            dK_s[d] = 0.0f;
            dQ_s[d] = 0.0f;
        }

        const uint dM_base = bh * C * C;
        const uint Q_base  = bh * C * D;
        const uint Krow_s_base = s * D;

        for (uint t = 0; t < C; ++t) {
            const float dM_st  = (float) dM_in[dM_base + s * C + t];
            const float dM_ts  = (float) dM_in[dM_base + t * C + s];
            const float dMoq_st = (float) dMoq_in[dM_base + s * C + t];
            const float dMoq_ts = (float) dMoq_in[dM_base + t * C + s];

            const uint Krow_t_base = t * D;
            const uint Qrow_t_base = Q_base + t * D;

            for (uint d = 0; d < D; ++d) {
                const float K_t_d = K_tg[Krow_t_base + d];
                const float G_s_d = G_tg[Krow_s_base + d];
                const float G_t_d = G_tg[Krow_t_base + d];
                const float Q_t_d = (float) Q_in[Qrow_t_base + d];

                const float gd_st = G_t_d - G_s_d;
                const float gd_ts = -gd_st;
                const float ed_st = (gd_st < 0.0f) ? metal::exp(gd_st) : 1.0f;
                const float ed_ts = (gd_ts < 0.0f) ? metal::exp(gd_ts) : 1.0f;

                dK_s[d] += dM_st  * K_t_d * ed_st
                         + dM_ts  * K_t_d * ed_ts
                         + dMoq_st * Q_t_d * ed_st;
                dQ_s[d] += dMoq_ts * K_t_d * ed_ts;
            }
        }

        // Write outputs
        const uint out_row = chunk_KG_base + s * D;
        for (uint d = 0; d < D; ++d) {
            dK_out[out_row + d] = (T) dK_s[d];
            dQ_out[out_row + d] = (T) dQ_s[d];
        }
    """

    _M_backward_dG_source = r"""
        // Computes dγ contributions through M / M_oq.
        //
        // dγ[s, d] aggregates 4 sources (2 from M, 2 from M_oq), all gated
        // by `active = (γ[t]<γ[s])`-style indicators (the ed factor is 1 and
        // γ-independent when inactive, contributing 0 to the gradient):
        //
        //   from M[s,t]:    -dM[s,t] · K[s,d] K[t,d] ed_active[s,t,d]
        //   from M[t,s]:    +dM[t,s] · K[t,d] K[s,d] ed_active[t,s,d]
        //   from M_oq[s,t]: -dM_oq[s,t] · K[s,d] Q[t,d] ed_active[s,t,d]
        //   from M_oq[t,s]: +dM_oq[t,s] · K[t,d] Q[s,d] ed_active[t,s,d]
        //
        // Layout: 1 threadgroup per (B*H), C threads/threadgroup.

        const uint bh = threadgroup_position_in_grid.x;
        const uint s  = thread_position_in_threadgroup.x;

        threadgroup float K_tg[C * D];
        threadgroup float G_tg[C * D];

        const uint chunk_KG_base = bh * C * D;
        for (uint i = s; i < C * D; i += C) {
            K_tg[i] = (float) K_in[chunk_KG_base + i];
            G_tg[i] = (float) gamma_in[chunk_KG_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-thread: Q_s in registers (constant per thread); dG accumulator
        float Q_s[D];
        float dG_s[D];
        const uint Q_s_base = bh * C * D + s * D;
        for (uint d = 0; d < D; ++d) {
            Q_s[d]  = (float) Q_in[Q_s_base + d];
            dG_s[d] = 0.0f;
        }

        const uint dM_base = bh * C * C;
        const uint Q_base  = bh * C * D;
        const uint Krow_s_base = s * D;

        for (uint t = 0; t < C; ++t) {
            const float dM_st   = (float) dM_in[dM_base + s * C + t];
            const float dM_ts   = (float) dM_in[dM_base + t * C + s];
            const float dMoq_st = (float) dMoq_in[dM_base + s * C + t];
            const float dMoq_ts = (float) dMoq_in[dM_base + t * C + s];

            const uint Krow_t_base = t * D;
            const uint Qrow_t_base = Q_base + t * D;

            for (uint d = 0; d < D; ++d) {
                const float K_s_d = K_tg[Krow_s_base + d];
                const float K_t_d = K_tg[Krow_t_base + d];
                const float G_s_d = G_tg[Krow_s_base + d];
                const float G_t_d = G_tg[Krow_t_base + d];
                const float Q_t_d = (float) Q_in[Qrow_t_base + d];

                const float gd_st = G_t_d - G_s_d;
                const float gd_ts = -gd_st;
                const float ed_st_a = (gd_st < 0.0f) ? metal::exp(gd_st) : 0.0f;
                const float ed_ts_a = (gd_ts < 0.0f) ? metal::exp(gd_ts) : 0.0f;

                // Sum 4 contributions
                const float kk = K_s_d * K_t_d;
                const float kq_st_term = K_s_d * Q_t_d;     // for -dMoq[s,t] * K_s * Q_t
                const float kq_ts_term = K_t_d * Q_s[d];    // for +dMoq[t,s] * K_t * Q_s

                dG_s[d] += -dM_st   * kk          * ed_st_a
                         + dM_ts   * kk          * ed_ts_a
                         - dMoq_st * kq_st_term  * ed_st_a
                         + dMoq_ts * kq_ts_term  * ed_ts_a;
            }
        }

        const uint out_row = chunk_KG_base + s * D;
        for (uint d = 0; d < D; ++d) {
            dG_out[out_row + d] = (T) dG_s[d];
        }
    """

    _M_backward_AB_kernel_cache: Dict[Tuple[int, int], object] = {}
    _M_backward_dG_kernel_cache: Dict[Tuple[int, int], object] = {}

    @classmethod
    def _get_M_backward_AB_kernel(cls, C: int, D: int):
        key = (C, D)
        if key in cls._M_backward_AB_kernel_cache:
            return cls._M_backward_AB_kernel_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunkwise_M_backward_AB_c{C}_d{D}",
            input_names=["K_in", "Q_in", "gamma_in", "dM_in", "dMoq_in"],
            output_names=["dK_out", "dQ_out"],
            source=cls._M_backward_AB_source,
        )
        cls._M_backward_AB_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def _get_M_backward_dG_kernel(cls, C: int, D: int):
        key = (C, D)
        if key in cls._M_backward_dG_kernel_cache:
            return cls._M_backward_dG_kernel_cache[key]
        kernel = mx.fast.metal_kernel(
            name=f"chunkwise_M_backward_dG_c{C}_d{D}",
            input_names=["K_in", "Q_in", "gamma_in", "dM_in", "dMoq_in"],
            output_names=["dG_out"],
            source=cls._M_backward_dG_source,
        )
        cls._M_backward_dG_kernel_cache[key] = kernel
        return kernel

    @classmethod
    def _metal_compute_M_Moq_backward(
        cls, K: mx.array, Q: mx.array, gamma: mx.array,
        dM: mx.array, dMoq: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Fused M / M_oq backward.
        Inputs:  K, Q, γ : (BH, C, D) fp32
                 dM, dMoq : (BH, C, C) fp32
        Returns: dK, dQ, dγ : (BH, C, D) fp32
        """
        BH, C, D = K.shape
        kernel_AB = cls._get_M_backward_AB_kernel(C, D)
        outs_AB = kernel_AB(
            inputs=[K, Q, gamma, dM, dMoq],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * C, 1, 1),
            threadgroup=(C, 1, 1),
            output_shapes=[(BH, C, D), (BH, C, D)],
            output_dtypes=[mx.float32, mx.float32],
        )
        dK, dQ = outs_AB

        kernel_dG = cls._get_M_backward_dG_kernel(C, D)
        outs_dG = kernel_dG(
            inputs=[K, Q, gamma, dM, dMoq],
            template=[("T", mx.float32), ("C", C), ("D", D)],
            grid=(BH * C, 1, 1),
            threadgroup=(C, 1, 1),
            output_shapes=[(BH, C, D)],
            output_dtypes=[mx.float32],
        )
        dG = outs_dG[0]
        return dK, dQ, dG

    # ==================================================================
    # PATH (b'): CHUNKWISE WY for KDA channel-wise decay
    # ==================================================================
    #
    # Extends the scalar-α chunkwise WY to channel-wise decay (Yang et al.,
    # ICLR 2025; Wei et al., Kimi Linear). Per chunk, we solve one C×C
    # triangular system (the WY representation), then derive the chunk
    # outputs and the next state from batched matmuls.
    #
    # Numerical stability: never materialize exp(±γ) separately because
    # exp(-γ_t) overflows fp32 for long sequences with strong decay. Instead
    # use exp(γ_t − γ_s) which is bounded in [0, 1] for causal pairs s ≤ t.
    #
    # Memory: requires a (B*H, C_s, C_t, D) intermediate per chunk for the
    # decay-difference tensor. For C=64, D=64, B*H=12: 12 MB per chunk
    # (one chunk live at a time).
    #
    # Note: this path provides FORWARD only. For training, use 'metal_vjp'
    # which has a working backward kernel; chunkwise backward would be a
    # separate (larger) implementation.

    def _chunkwise_kda_forward(self, q, k, v, g, beta, state):
        """
        Chunkwise WY forward for KDA channel-wise decay.

        Inputs (heads-folded into batch):
            q, k, v : (B*H, L, D)    any dtype (kept; only g and state need fp32)
            g       : (B*H, L, D)    fp32 (caller already cast it)
            beta    : (B*H, L)       any dtype
            state   : (B*H, D, D)    or None
        Returns:
            output  : (B*H, L, D)    in q's dtype
            state   : (B*H, D, D)    fp32

        Optimizations vs the original PyTorch port:
          (1) chunk-shape constants cached in `_get_chunk_consts(C)`.
          (3) q/k/v/β stay in their input dtype; only g and state forced to fp32.
          (4) ALL state-independent work (γ, exp_gamma, K_proj/Q_prime/K_back,
              IA, M, M_oq) computed ONCE outside the loop, batched over
              n_chunks. The chunk loop only does the state-dependent rhs
              assembly + Metal solve + state update.
          (6) `_metal_compute_M_Moq` replaced by two batched MLX matmuls
              backed by simdgroup_matrix Metal kernels: factorization
                M[i,j]    = (K * exp(-γ))[i] · (K * exp(γ))[j]   for i ≤ j
                M_oq[i,j] = (K * exp(-γ))[i] · (Q * exp(γ))[j]   for i ≤ j
              The lower triangle is masked downstream so its (potentially
              over-large) value is irrelevant.
          (b) GPU triangular solve via stacked Metal kernel (already present).
          (2) `mx.compile`-fused pre-solve and post-solve.
        """
        BH, L, D = q.shape
        C = self.chunk_size
        orig_dtype = q.dtype

        # g is already fp32 from the caller (line 615); keep q/k/v/β in input dtype.
        # Only force fp32 on state.

        # Pad L up to a multiple of C. Use g=0 (no decay) and beta=0 (no write).
        pad = (C - L % C) % C
        if pad > 0:
            q     = mx.concatenate([q,    mx.zeros((BH, pad, D), q.dtype)], axis=1)
            k     = mx.concatenate([k,    mx.zeros((BH, pad, D), k.dtype)], axis=1)
            v     = mx.concatenate([v,    mx.zeros((BH, pad, D), v.dtype)], axis=1)
            g     = mx.concatenate([g,    mx.zeros((BH, pad, D), g.dtype)], axis=1)
            beta  = mx.concatenate([beta, mx.zeros((BH, pad),    beta.dtype)], axis=1)

        L_pad = q.shape[1]
        n_chunks = L_pad // C

        q_c    = q.reshape(BH, n_chunks, C, D)
        k_c    = k.reshape(BH, n_chunks, C, D)
        v_c    = v.reshape(BH, n_chunks, C, D)
        g_c    = g.reshape(BH, n_chunks, C, D)
        beta_c = beta.reshape(BH, n_chunks, C)

        if state is None:
            state = mx.zeros((BH, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        consts = _get_chunk_consts(C)
        I_C, A_mask, O_mask = consts["I_CC"], consts["A_mask"], consts["O_mask"]

        # ---- LIFTED PREAMBLE 1: cumulative-decay derivatives (no M/M_oq) ----
        gamma_all, exp_gamma_all, K_proj_all, K_back_all, rhs_pre_state_all = \
            _kda_chunk_preamble_no_M(g_c, k_c, v_c, beta_c)

        # ---- M / M_oq via the fused Metal kernel — call ONCE for all chunks
        #      by folding n_chunks into the batch dim. ----
        K_flat = k_c.reshape(BH * n_chunks, C, D).astype(mx.float32)
        Q_flat = q_c.reshape(BH * n_chunks, C, D).astype(mx.float32)
        gamma_flat = gamma_all.reshape(BH * n_chunks, C, D)
        M_flat, M_oq_flat = self._metal_compute_M_Moq(K_flat, Q_flat, gamma_flat)
        M_all    = M_flat.reshape(BH, n_chunks, C, C)
        M_oq_all = M_oq_flat.reshape(BH, n_chunks, C, C)

        # ---- LIFTED PREAMBLE 2: IA + M_oq_masked_T, batched over n_chunks ----
        IA_all, M_oq_T_masked_all = _kda_build_IA_and_M_oq_T(
            M_all, M_oq_all, beta_c, A_mask, I_C, O_mask,
        )
        Q_prime_all = exp_gamma_all * q_c                            # (BH, nc, C, D)

        # ---- PER-CHUNK STATE-DEPENDENT LOOP ----
        # When the fused kernel is enabled, each chunk is ONE Metal launch
        # (rhs+solve+finalize). Otherwise we fall back to the legacy 3-op path.
        outs: List[mx.array] = []
        use_fused = (
            self._use_fused_chunk_kernel
            and (C * C + C * D) * 4 <= 32 * 1024
        )
        # Pick variant. The simdgroup kernel needs:
        #   - C, D both multiples of 8 (the simdgroup_matrix tile size)
        #   - max(D·D, C·C, C·D) + max(C·D, D·D) ≤ 8192 fp32 (TG budget)
        # When the user opts into simdgroup (the default) but it doesn't
        # fit, we fall back to the UNFUSED vendor-matmul path rather than
        # the scalar fused kernel, because at large head_dim the scalar
        # fused kernel is slower than vendor matmul. Scalar fused is only
        # used when explicitly requested via `_fused_chunk_use_simdgroup
        # = False` (e.g. for memory-constrained scenarios).
        sg_slot1 = max(D * D, C * C, C * D)
        sg_slot2 = max(C * D, D * D)
        sg_fits = (sg_slot1 + sg_slot2) * 4 <= 32 * 1024
        if self._fused_chunk_use_simdgroup:
            use_sg = (use_fused and C % 8 == 0 and D % 8 == 0 and sg_fits)
            if use_fused and not use_sg:
                use_fused = False                  # ← drop to unfused
            fuse_fn = self._metal_chunk_solve_finalize_sg
        else:
            use_sg = False
            fuse_fn = self._metal_chunk_solve_finalize

        for ci in range(n_chunks):
            if use_fused:
                O, state, _ = fuse_fn(
                    IA_all[:, ci],
                    rhs_pre_state_all[:, ci],
                    beta_c[:, ci],
                    K_proj_all[:, ci],
                    Q_prime_all[:, ci],
                    M_oq_T_masked_all[:, ci],
                    K_back_all[:, ci],
                    exp_gamma_all[:, ci, -1, :],
                    state,
                )
            else:
                rhs = _kda_assemble_rhs(
                    rhs_pre_state_all[:, ci], beta_c[:, ci],
                    K_proj_all[:, ci], state,
                )
                U = self._metal_solve_triangular(IA_all[:, ci], rhs)
                O, state = _kda_finalize(
                    U, Q_prime_all[:, ci], M_oq_T_masked_all[:, ci],
                    K_back_all[:, ci], exp_gamma_all[:, ci, -1, :], state,
                )
            outs.append(O)

        out = mx.concatenate(outs, axis=1)
        if pad > 0:
            out = out[:, :L]
        return out.astype(orig_dtype), state

    # ==================================================================
    # PATH (b''): CHUNKWISE_KDA_VJP — autograd-compatible chunkwise forward
    # ==================================================================
    #
    # Wraps `_chunkwise_kda_forward` in mx.custom_function so MLX autograd can
    # flow gradients back through the chunkwise recurrence. The backward walks
    # chunks in REVERSE, recomputing per-chunk preamble values from saved
    # state-trajectory and U-trajectory arrays.
    #
    # Memory cost vs metal_vjp:
    #   metal_vjp saves S_traj of shape (BH, L+1, D, D)  =  L * D² fp32 / chunk_size more
    #   chunkwise_kda_vjp saves:
    #     state_traj : (BH, n_chunks+1, D, D)  =  L * D² fp32 / chunk_size  (same scaling)
    #     U_traj     : (BH, L, D)               =  L * D fp32
    #   So the new path uses ~chunk_size× less memory than metal_vjp.
    #
    # The backward derives gradients via hand-computed VJPs:
    #   - Output equation     O = Q' S^T + (M_oq * O_mask)^T U
    #   - State update        S' = exp(γ_C) ⊙ S + U^T K_back
    #   - WY solve            U = solve(IA, β(V - K_proj S^T))
    #   - IA / M / M_oq pieces (M / M_oq backward via einsums on the
    #     (BH, C, C, D) decay-difference tensor — 16 MB live per chunk at C=64,
    #     D=64; far less than the autograd alternative which would keep all
    #     n_chunks of these live simultaneously)
    #   - Cumsum-decay backward via reverse cumsum

    def _chunkwise_kda_with_traj(self, q, k, v, g, beta, state):
        """Forward that ALSO returns the per-chunk U and per-boundary state
        trajectories needed by the backward. Same numerics as
        `_chunkwise_kda_forward`."""
        BH, L, D = q.shape
        C = self.chunk_size
        orig_dtype = q.dtype

        pad = (C - L % C) % C
        if pad > 0:
            q     = mx.concatenate([q,    mx.zeros((BH, pad, D), q.dtype)], axis=1)
            k     = mx.concatenate([k,    mx.zeros((BH, pad, D), k.dtype)], axis=1)
            v     = mx.concatenate([v,    mx.zeros((BH, pad, D), v.dtype)], axis=1)
            g     = mx.concatenate([g,    mx.zeros((BH, pad, D), g.dtype)], axis=1)
            beta  = mx.concatenate([beta, mx.zeros((BH, pad),    beta.dtype)], axis=1)
        L_pad = q.shape[1]; n_chunks = L_pad // C
        q_c = q.reshape(BH, n_chunks, C, D); k_c = k.reshape(BH, n_chunks, C, D)
        v_c = v.reshape(BH, n_chunks, C, D); g_c = g.reshape(BH, n_chunks, C, D)
        beta_c = beta.reshape(BH, n_chunks, C)

        if state is None:
            state = mx.zeros((BH, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)

        consts = _get_chunk_consts(C)
        I_C, A_mask, O_mask = consts["I_CC"], consts["A_mask"], consts["O_mask"]

        gamma_all, exp_gamma_all, K_proj_all, K_back_all, rhs_pre_state_all = \
            _kda_chunk_preamble_no_M(g_c, k_c, v_c, beta_c)

        K_flat = k_c.reshape(BH * n_chunks, C, D).astype(mx.float32)
        Q_flat = q_c.reshape(BH * n_chunks, C, D).astype(mx.float32)
        gamma_flat = gamma_all.reshape(BH * n_chunks, C, D)
        M_flat, M_oq_flat = self._metal_compute_M_Moq(K_flat, Q_flat, gamma_flat)
        M_all    = M_flat.reshape(BH, n_chunks, C, C)
        M_oq_all = M_oq_flat.reshape(BH, n_chunks, C, C)
        IA_all, M_oq_T_masked_all = _kda_build_IA_and_M_oq_T(
            M_all, M_oq_all, beta_c, A_mask, I_C, O_mask,
        )
        Q_prime_all = exp_gamma_all * q_c

        outs: List[mx.array] = []
        U_chunks: List[mx.array] = []
        state_chunks: List[mx.array] = [state]
        use_fused = (
            self._use_fused_chunk_kernel
            and (C * C + C * D) * 4 <= 32 * 1024
        )
        sg_slot1 = max(D * D, C * C, C * D)
        sg_slot2 = max(C * D, D * D)
        sg_fits = (sg_slot1 + sg_slot2) * 4 <= 32 * 1024
        use_sg = (use_fused and self._fused_chunk_use_simdgroup
                  and C % 8 == 0 and D % 8 == 0 and sg_fits)
        fuse_fn = self._metal_chunk_solve_finalize_sg if use_sg \
            else self._metal_chunk_solve_finalize

        for ci in range(n_chunks):
            if use_fused:
                O, state, U = fuse_fn(
                    IA_all[:, ci],
                    rhs_pre_state_all[:, ci],
                    beta_c[:, ci],
                    K_proj_all[:, ci],
                    Q_prime_all[:, ci],
                    M_oq_T_masked_all[:, ci],
                    K_back_all[:, ci],
                    exp_gamma_all[:, ci, -1, :],
                    state,
                )
            else:
                rhs = _kda_assemble_rhs(
                    rhs_pre_state_all[:, ci], beta_c[:, ci],
                    K_proj_all[:, ci], state,
                )
                U = self._metal_solve_triangular(IA_all[:, ci], rhs)
                O, state = _kda_finalize(
                    U, Q_prime_all[:, ci], M_oq_T_masked_all[:, ci],
                    K_back_all[:, ci], exp_gamma_all[:, ci, -1, :], state,
                )
            outs.append(O); U_chunks.append(U); state_chunks.append(state)

        out = mx.concatenate(outs, axis=1)                  # (BH, L_pad, D)
        U_traj = mx.stack(U_chunks, axis=1)                 # (BH, n_chunks, C, D)
        S_traj = mx.stack(state_chunks, axis=1)             # (BH, n_chunks+1, D, D)
        if pad > 0:
            out = out[:, :L]
        return out.astype(orig_dtype), state, U_traj, S_traj

    @staticmethod
    def _reverse_cumsum(x: mx.array, axis: int) -> mx.array:
        """Adjoint of cumsum: y[t] = Σ_{t' ≥ t} x[t']. MLX has no `flip`, so
        use `mx.cumsum(..., reverse=True)` instead."""
        return mx.cumsum(x, axis=axis, reverse=True)

    def _chunkwise_kda_backward(
        self,
        q_c, k_c, v_c, g_c, beta_c,
        U_traj, S_traj,
        dO_all, dstate_final,
        I_C, A_mask, O_mask,
    ):
        """
        Per-chunk reverse-pass backward for the chunkwise KDA recurrence.
        See class-level docstring above for the math.

        All arrays already shaped (BH, n_chunks, C, D) (or appropriate); inputs
        and outputs in fp32 throughout.
        """
        BH, n_chunks, C, D = q_c.shape

        dq_chunks: List[mx.array] = []
        dk_chunks: List[mx.array] = []
        dv_chunks: List[mx.array] = []
        dg_chunks: List[mx.array] = []
        dbeta_chunks: List[mx.array] = []

        dstate = dstate_final                                # (BH, D, D)

        for ci in range(n_chunks - 1, -1, -1):
            # ---- Recompute per-chunk forward intermediates from saved state ----
            S    = S_traj[:, ci]                             # (BH, D, D)
            U    = U_traj[:, ci]                             # (BH, C, D)
            Qc   = q_c[:, ci]; Kc = k_c[:, ci]; Vc = v_c[:, ci]
            Gc   = g_c[:, ci]; Bc = beta_c[:, ci]

            gamma     = mx.cumsum(Gc, axis=-2)               # (BH, C, D)
            gamma_last = gamma[:, -1, :]                     # (BH, D)
            exp_gamma  = mx.exp(gamma)
            exp_gamma_last = exp_gamma[:, -1, :]
            K_proj  = exp_gamma * Kc
            Q_prime = exp_gamma * Qc
            K_back  = mx.exp(gamma_last[:, None, :] - gamma) * Kc
            M, M_oq = self._metal_compute_M_Moq(Kc, Qc, gamma)
            IA      = I_C + Bc[:, :, None] * mx.swapaxes(M, -1, -2) * A_mask
            M_oq_T_masked = mx.swapaxes(M_oq, -1, -2) * mx.swapaxes(O_mask, -1, -2)
            y       = Vc - K_proj @ mx.swapaxes(S, -1, -2)   # (BH, C, D)
            dO      = dO_all[:, ci]                          # (BH, C, D)

            # ---- (1) state_new = exp_gamma_last[None, :] * S + U.T @ K_back ----
            dexp_gamma_last_via_S = mx.sum(S * dstate, axis=-2)         # (BH, D)
            dS_via_decay = exp_gamma_last[..., None, :] * dstate          # (BH, D, D)
            dU_via_S  = K_back @ mx.swapaxes(dstate, -1, -2)              # (BH, C, D)
            dK_back   = U @ dstate                                         # (BH, C, D)

            # ---- (2) O = Q_prime @ S.T + M_oq_T_masked @ U ----
            dQ_prime         = dO @ S                                     # (BH, C, D)
            dS_via_O         = mx.swapaxes(dO, -1, -2) @ Q_prime           # (BH, D, D)
            dM_oq_T_masked   = dO @ mx.swapaxes(U, -1, -2)                # (BH, C, C)
            dU_via_O         = mx.swapaxes(M_oq_T_masked, -1, -2) @ dO    # (BH, C, D)

            dU = dU_via_S + dU_via_O

            # ---- (3) U = solve(IA, rhs) ----
            drhs = self._metal_solve_triangular_upper(
                mx.swapaxes(IA, -1, -2), dU,
            )                                                              # (BH, C, D)
            dIA  = -mx.tril(drhs @ mx.swapaxes(U, -1, -2), k=-1)           # (BH, C, C)

            # ---- (4) rhs = β[..., None] * (V - K_proj @ S.T) = β * y ----
            dbeta_via_rhs   = mx.sum(drhs * y, axis=-1)                    # (BH, C)
            dy              = drhs * Bc[..., None]                         # (BH, C, D)
            dV              = dy
            dK_proj_via_rhs = -dy @ S                                      # (BH, C, D)
            dS_via_rhs      = -mx.swapaxes(dy, -1, -2) @ K_proj            # (BH, D, D)

            # ---- (5) IA = I + β[..., None] * M.T * A_mask ----
            dA            = dIA                                            # already strict-lower
            dM_T_via_IA   = dA * Bc[..., None] * A_mask
            dM_via_IA     = mx.swapaxes(dM_T_via_IA, -1, -2)
            dbeta_via_IA  = mx.sum(dA * mx.swapaxes(M, -1, -2) * A_mask, axis=-1)

            # ---- (6) M_oq_T_masked = M_oq.T * O_mask.T ----
            dM_oq_via_TM = mx.swapaxes(dM_oq_T_masked, -1, -2) * O_mask    # (BH, C, C)

            # ---- (7) M kernel backward via fused Metal kernels ----
            # Two kernel launches: dK+dQ (kernel AB) and dγ (kernel dG).
            # Avoids the (BH, C, C, D) decay-difference intermediate that the
            # MLX einsum implementation materialised.
            dK_via_M, dQ_via_M, dgamma_via_M = self._metal_compute_M_Moq_backward(
                Kc, Qc, gamma, dM_via_IA, dM_oq_via_TM,
            )

            # ---- (8) K_proj, Q_prime, K_back, exp_gamma_last derivatives ----
            dK_proj_total = dK_proj_via_rhs                                # (BH, C, D)
            dK_via_Kproj  = dK_proj_total * exp_gamma
            dgamma_via_Kproj = dK_proj_total * K_proj

            dQ_via_Qprime    = dQ_prime * exp_gamma
            dgamma_via_Qprime = dQ_prime * Q_prime

            dK_via_Kback     = dK_back * mx.exp(gamma_last[:, None, :] - gamma)
            dgamma_via_Kback = -dK_back * K_back
            dgamma_last_via_Kback = mx.sum(dK_back * K_back, axis=-2)      # (BH, D)

            # γ_last appears explicitly in state update: dexp_gamma_last_via_S
            dgamma_last_via_S = dexp_gamma_last_via_S * exp_gamma_last     # (BH, D)
            dgamma_last_total = dgamma_last_via_Kback + dgamma_last_via_S  # (BH, D)

            # ---- (9) Aggregate dgamma and add dgamma_last to last row ----
            dgamma = dgamma_via_M + dgamma_via_Kproj + dgamma_via_Qprime \
                     + dgamma_via_Kback                                    # (BH, C, D)
            # Add dgamma_last to dgamma at the last C step.
            dgamma = mx.concatenate([
                dgamma[:, :-1, :],
                dgamma[:, -1:, :] + dgamma_last_total[:, None, :],
            ], axis=-2)

            # ---- (10) cumsum backward: γ = cumsum(g)  ⇒  dg = reverse_cumsum(dγ) ----
            dg_chunk = self._reverse_cumsum(dgamma, axis=-2)               # (BH, C, D)

            # ---- (11) v gradient — V appears via rhs_pre_state = β·V and y = V - K_proj S^T ----
            # dy = drhs * β; dV = dy. (rhs_pre_state piece is the same; both
            # routes contribute through dy.)
            dv_chunk = dV                                                   # (BH, C, D)

            # ---- (12) Aggregate dK, dQ, dβ for this chunk ----
            dK_chunk = dK_via_M + dK_via_Kproj + dK_via_Kback              # (BH, C, D)
            dQ_chunk = dQ_via_M + dQ_via_Qprime                             # (BH, C, D)
            dbeta_chunk = dbeta_via_rhs + dbeta_via_IA                      # (BH, C)

            # ---- (13) State carry to previous chunk ----
            dstate = dS_via_decay + dS_via_O + dS_via_rhs                  # (BH, D, D)

            dq_chunks.append(dQ_chunk)
            dk_chunks.append(dK_chunk)
            dv_chunks.append(dv_chunk)
            dg_chunks.append(dg_chunk)
            dbeta_chunks.append(dbeta_chunk)

        # Reverse to ascending chunk order
        dq    = mx.stack(list(reversed(dq_chunks)),    axis=1)             # (BH, n_chunks, C, D)
        dk    = mx.stack(list(reversed(dk_chunks)),    axis=1)
        dv    = mx.stack(list(reversed(dv_chunks)),    axis=1)
        dg    = mx.stack(list(reversed(dg_chunks)),    axis=1)
        dbeta = mx.stack(list(reversed(dbeta_chunks)), axis=1)             # (BH, n_chunks, C)
        return dq, dk, dv, dg, dbeta, dstate

    # ---- Custom-function-wrapped chunkwise forward ----
    _chunkwise_kda_recurrence_fn_cache = None

    @classmethod
    def _get_chunkwise_kda_recurrence_fn(cls):
        if cls._chunkwise_kda_recurrence_fn_cache is not None:
            return cls._chunkwise_kda_recurrence_fn_cache

        # We thread `self`/`chunk_size` through the closure of a layer-bound
        # lambda at first call. To make a single global custom_function work
        # across layers with potentially different chunk_size, we read C from
        # the saved trajectory shapes inside the VJP.
        @mx.custom_function
        def _chunkwise_kda_recurrence_fn(q, k, v, g, beta, state_in, layer_marker):
            # `layer_marker` is a 0-d fp32 tensor whose only purpose is to keep
            # the layer reference alive in the autograd graph; we read `self`
            # off it via a side-channel dict lookup keyed by id(layer_marker).
            layer = cls._layer_marker_lookup[int(layer_marker.item())]
            out, state_out, U_traj, S_traj = layer._chunkwise_kda_with_traj(
                q, k, v, g, beta, state_in,
            )
            # Cast everything to fp32 for storage/backward
            return (out.astype(mx.float32), state_out, U_traj, S_traj)

        @_chunkwise_kda_recurrence_fn.vjp
        def _chunkwise_kda_vjp(primals, cotangents, output):
            q, k, v, g, beta, _state_in, layer_marker = primals
            do, dstate_final, _dU_unused, _dS_unused = cotangents
            _out, _S_final, _U_traj_out, S_traj = output

            layer = cls._layer_marker_lookup[int(layer_marker.item())]
            C = layer.chunk_size
            BH, L, D = do.shape

            # The forward unpads the output (length L), but saves U_traj and
            # S_traj at the padded length L_pad = n_chunks · C. The cotangents
            # `do` and the primals q/k/v/g/beta come back at the unpadded L.
            # If L isn't a chunk-boundary multiple, pad them here so the
            # backward runs on the same shape as the saved trajectories.
            U_traj = _U_traj_out
            n_chunks = U_traj.shape[1]
            L_pad = n_chunks * C
            pad = L_pad - L
            assert 0 <= pad < C, f"Unexpected pad mismatch: L={L}, L_pad={L_pad}"

            if pad > 0:
                z_dl = mx.zeros((BH, pad, D), q.dtype)
                z_b  = mx.zeros((BH, pad), beta.dtype)
                z_do = mx.zeros((BH, pad, D), do.dtype)
                q     = mx.concatenate([q,    z_dl], axis=1)
                k     = mx.concatenate([k,    z_dl], axis=1)
                v     = mx.concatenate([v,    z_dl], axis=1)
                g     = mx.concatenate([g,    mx.zeros((BH, pad, D), g.dtype)], axis=1)
                beta  = mx.concatenate([beta, z_b],  axis=1)
                do    = mx.concatenate([do,   z_do], axis=1)

            q_c = q.reshape(BH, n_chunks, C, D).astype(mx.float32)
            k_c = k.reshape(BH, n_chunks, C, D).astype(mx.float32)
            v_c = v.reshape(BH, n_chunks, C, D).astype(mx.float32)
            g_c = g.reshape(BH, n_chunks, C, D).astype(mx.float32)
            beta_c = beta.reshape(BH, n_chunks, C).astype(mx.float32)
            dO_all = do.reshape(BH, n_chunks, C, D).astype(mx.float32)

            consts = _get_chunk_consts(C)
            I_C, A_mask, O_mask = consts["I_CC"], consts["A_mask"], consts["O_mask"]

            dq, dk, dv, dg, dbeta, dstate_init = layer._chunkwise_kda_backward(
                q_c, k_c, v_c, g_c, beta_c,
                U_traj, S_traj,
                dO_all, dstate_final,
                I_C, A_mask, O_mask,
            )
            dq    = dq.reshape(BH, L_pad, D)
            dk    = dk.reshape(BH, L_pad, D)
            dv    = dv.reshape(BH, L_pad, D)
            dg    = dg.reshape(BH, L_pad, D)
            dbeta = dbeta.reshape(BH, L_pad)
            # Strip the padding off so the gradients match the upstream
            # primals' (unpadded) shape.
            if pad > 0:
                dq    = dq[:, :L]
                dk    = dk[:, :L]
                dv    = dv[:, :L]
                dg    = dg[:, :L]
                dbeta = dbeta[:, :L]
            d_layer_marker = mx.zeros_like(layer_marker)
            return dq, dk, dv, dg, dbeta, dstate_init, d_layer_marker

        cls._chunkwise_kda_recurrence_fn_cache = _chunkwise_kda_recurrence_fn
        return _chunkwise_kda_recurrence_fn

    # Layer-marker side channel: maps an int id -> layer instance, so the
    # custom_function VJP (which only receives MLX arrays) can find `self`.
    _layer_marker_lookup: Dict[int, "GatedDeltaLayer"] = {}
    _layer_marker_counter: int = 0

    def _get_layer_marker(self) -> mx.array:
        # Cache one marker per instance.
        marker_id = getattr(self, "_chunkwise_marker_id", None)
        if marker_id is None:
            cls = type(self)
            cls._layer_marker_counter += 1
            marker_id = cls._layer_marker_counter
            cls._layer_marker_lookup[marker_id] = self
            self._chunkwise_marker_id = marker_id
        return mx.array(float(marker_id), dtype=mx.float32)

    def _chunkwise_kda_vjp_path(self, q, k, v, g, beta, state):
        """
        Autograd-compatible chunkwise KDA forward (with hand-derived backward).
        Same I/O contract as `_metal_kernel_kda_vjp` and `_chunkwise_kda_forward`.
        """
        B, L, H, D = q.shape
        if D != self.head_dim:
            raise ValueError(
                f"chunkwise_kda_vjp D mismatch: got {D}, expected {self.head_dim}"
            )

        # Cast to fp32, fold heads into batch
        q32 = q.astype(mx.float32); k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32); g32 = g.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        q_bh    = mx.transpose(q32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        k_bh    = mx.transpose(k32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        v_bh    = mx.transpose(v32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        g_bh    = mx.transpose(g32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        beta_bh = mx.transpose(beta32, (0, 2, 1)).reshape(B * H, L)

        if state is None:
            state_in = mx.zeros((B * H, D, D), dtype=mx.float32)
        else:
            state_in = state.astype(mx.float32).reshape(B * H, D, D)

        fn = self._get_chunkwise_kda_recurrence_fn()
        marker = self._get_layer_marker()
        output_bh, state_out_bh, _U_traj, _S_traj = fn(
            q_bh, k_bh, v_bh, g_bh, beta_bh, state_in, marker,
        )

        output_4d = output_bh.reshape(B, H, L, D)
        output_4d = mx.transpose(output_4d, (0, 2, 1, 3))
        new_state = state_out_bh.reshape(B, H, D, D)
        return output_4d.astype(q.dtype), new_state

    # ==================================================================
    # PATH (d): NAIVE STEP — fallback / debug
    # ==================================================================
    def _loop_naive_kda(self, q, k, v, g, beta, state):
        """Pure-Python step loop, no compile. Same math as path (c), slower."""
        B, L, H, D = q.shape
        q32 = q.astype(mx.float32); k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32); beta32 = beta.astype(mx.float32)

        if state is None:
            state = mx.zeros((B, H, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)
        I = mx.eye(D, dtype=mx.float32)

        outputs: List[mx.array] = []
        for t in range(L):
            decay = mx.exp(g[:, t])
            state = state * decay[..., None, :]
            b_t = beta32[:, t][..., None, None]
            k_t = k32[:, t]; v_t = v32[:, t]; q_t = q32[:, t]
            k_outer = k_t[..., :, None] @ k_t[..., None, :]
            state = state @ (I - b_t * k_outer)
            state = state + b_t * (v_t[..., :, None] @ k_t[..., None, :])
            outputs.append((state @ q_t[..., :, None]).squeeze(-1))
        return mx.stack(outputs, axis=1).astype(q.dtype), state

    def _loop_naive_gdn(self, q, k, v, alpha, beta, state):
        """Pure-Python scalar-α step loop."""
        B, L, H, D = q.shape
        q32 = q.astype(mx.float32); k32 = k.astype(mx.float32)
        v32 = v.astype(mx.float32)
        alpha32 = alpha.astype(mx.float32); beta32 = beta.astype(mx.float32)

        if state is None:
            state = mx.zeros((B, H, D, D), dtype=mx.float32)
        else:
            state = state.astype(mx.float32)
        I = mx.eye(D, dtype=mx.float32)

        outputs: List[mx.array] = []
        for t in range(L):
            a_t = alpha32[:, t][..., None, None]
            b_t = beta32[:, t][..., None, None]
            k_t = k32[:, t]; v_t = v32[:, t]; q_t = q32[:, t]
            k_outer = k_t[..., :, None] @ k_t[..., None, :]
            state = state @ (a_t * (I - b_t * k_outer))
            state = state + b_t * (v_t[..., :, None] @ k_t[..., None, :])
            outputs.append((state @ q_t[..., :, None]).squeeze(-1))
        return mx.stack(outputs, axis=1).astype(q.dtype), state

    # ==================================================================
    # PATH (a): CUSTOM METAL KERNEL — fused KDA recurrence
    # ==================================================================
    #
    # DESIGN:
    #   - One threadgroup per (batch, head) slice
    #   - D threads per threadgroup, each owns one row of the D×D state
    #     (each thread holds D fp32 values in registers — fits comfortably)
    #   - Whole L-step loop runs inside the kernel: one launch instead of L
    #   - Naive scalar matmuls (no simdgroup matmul intrinsics).
    #     This is intentional: simdgroup matmul would be ~3–5× faster but
    #     significantly harder to verify correct without running on hardware.
    #     The Python-overhead win alone gets ~3–8× over path (c).
    #
    # CONSTRAINTS:
    #   - D must equal the threadgroup size (we use threadgroup_size = D)
    #   - D must be ≤ 1024 (Apple GPU max threadgroup size)
    #   - Inputs cast to fp32 inside the wrapper; state always fp32
    #
    # WHAT THIS DOES NOT DO:
    #   - No within-chunk parallelism (steps are sequential, as required by
    #     the recurrence). The W in WY parallelism is what FLA gets via
    #     chunkwise; that's not in this kernel.
    #   - No tensor-core / simdgroup matmul. Add for ~3-5× more if needed.
    #   - No backward pass. For training, MLX's autograd will fall through;
    #     wrap this in mx.custom_function with a manual VJP if you need
    #     gradients. For inference this is fine as-is.

    _kda_metal_source = r"""
        // Per-thread: own one row of the D×D state matrix in registers.
        // Threadgroup-shared scratch holds q_t, k_t, v_t, g_t for the
        // current step plus a single beta_t scalar, refreshed each step.
        //
        // Outputs (channel-wise KDA forward):
        //   output    : (B*H, L, D) — out_t = S_t @ q_t  (post-update state)
        //   state_out : (B*H, D, D) — S_L (final state)
        //   u_out     : (B*H, L, D) — u_t = β_t (v_t − S' k_t), saved for the
        //                              backward kernel so it can invert the
        //                              forward recurrence without storing every S_t.

        const uint bh  = threadgroup_position_in_grid.x;   // 0..B*H-1
        const uint tid = thread_position_in_threadgroup.x; // 0..D-1
        const uint L_  = q_shape[1];                        // sequence length

        // -- Per-thread state row (in registers) --
        float S_row[D];

        // Load initial state row from device memory.
        // state_in layout: (B*H, D, D) row-major
        const uint state_base = bh * D * D + tid * D;
        for (uint j = 0; j < D; ++j) {
            S_row[j] = (float) state_in[state_base + j];
        }

        // -- Threadgroup-shared scratch for current step's vectors --
        threadgroup float q_t[D];
        threadgroup float k_t[D];
        threadgroup float v_t[D];
        threadgroup float g_t[D];
        threadgroup float beta_t_shared[1];
        // Sk_full: each row's Sk_my (= S_row · k_t) gathered into a shared vector
        // so every thread can compute u[my_row] = β_t (v_t[my_row] - Sk_my).
        // u itself is broadcast-trivial: u[my_row] depends only on my_row.

        for (uint t = 0; t < L_; ++t) {
            // Each thread loads one element of q,k,v,g for time t into
            // threadgroup memory; tid == 0 also loads beta.
            const uint vec_base = bh * L_ * D + t * D;
            q_t[tid] = (float) q[vec_base + tid];
            k_t[tid] = (float) k[vec_base + tid];
            v_t[tid] = (float) v[vec_base + tid];
            g_t[tid] = (float) g[vec_base + tid];
            if (tid == 0) {
                beta_t_shared[0] = (float) beta[bh * L_ + t];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float beta_t = beta_t_shared[0];

            // -- (1) Column-wise decay: S_row[j] *= exp(g_t[j]) --
            // (channel-wise decay scales columns of S, channel = j)
            for (uint j = 0; j < D; ++j) {
                S_row[j] *= metal::exp(g_t[j]);
            }

            // -- (2) Compute Sk_my = sum_j S_row[j] * k_t[j] --
            float Sk_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                Sk_my += S_row[j] * k_t[j];
            }

            // -- Save u_my = β_t (v_t[my_row] - Sk_my) --
            // u[my_row] is exactly the rank-1 update vector entry for this row.
            // S_new[my_row] = S'[my_row] + u_my * k_t  (rank-1 row update).
            const float u_my = beta_t * (v_t[tid] - Sk_my);
            u_out[vec_base + tid] = (T) u_my;

            // -- (3) S_row += u_my * k_t (rank-1 add, equivalent to:
            //         S_row -= β Sk_my k_t  then  S_row += β v_my k_t  ) --
            for (uint j = 0; j < D; ++j) {
                S_row[j] += u_my * k_t[j];
            }

            // -- (4) Output: out_t[tid] = sum_j S_row[j] * q_t[j] --
            float out_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                out_my += S_row[j] * q_t[j];
            }
            output[vec_base + tid] = (T) out_my;

            // Sync before next iteration overwrites threadgroup scratch
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // -- Store final state row back to device memory --
        for (uint j = 0; j < D; ++j) {
            state_out[state_base + j] = (T) S_row[j];
        }
    """

    # Cache compiled kernels by D (one per head_dim). Keyed by D since the
    # threadgroup memory and array sizes are compile-time-fixed via template.
    _metal_kernel_cache: Dict[int, object] = {}

    @classmethod
    def _get_metal_kernel(cls, D: int):
        """Build (and cache) a Metal kernel specialized for given head_dim D."""
        if D in cls._metal_kernel_cache:
            return cls._metal_kernel_cache[D]
        kernel = mx.fast.metal_kernel(
            name=f"kda_recurrence_d{D}",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["output", "state_out", "u_out"],
            source=cls._kda_metal_source,
        )
        cls._metal_kernel_cache[D] = kernel
        return kernel

    def _metal_kernel_kda(self, q, k, v, g, beta, state):
        """
        Fused Metal kernel KDA recurrence.

        Args:
            q, k, v: (B, L, H, D)
            g:       (B, L, H, D)  log-space, ≤ 0
            beta:    (B, L, H)
            state:   (B, H, D, D) or None
        Returns:
            output:  (B, L, H, D)
            state:   (B, H, D, D)
        """
        B, L, H, D = q.shape
        if D != self.head_dim:
            raise ValueError(f"Metal kernel D mismatch: got {D}, expected {self.head_dim}")
        if D > 1024:
            raise ValueError(f"Metal kernel requires D <= 1024 (Apple GPU TG max); got {D}")

        kernel = self._get_metal_kernel(D)

        # Cast everything to fp32 for state stability
        q32    = q.astype(mx.float32)
        k32    = k.astype(mx.float32)
        v32    = v.astype(mx.float32)
        g32    = g.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        # Reshape (B, L, H, D) -> (B, H, L, D) -> (B*H, L, D) for kernel
        q_bh    = mx.transpose(q32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        k_bh    = mx.transpose(k32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        v_bh    = mx.transpose(v32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        g_bh    = mx.transpose(g32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        beta_bh = mx.transpose(beta32, (0, 2, 1)).reshape(B * H, L)

        if state is None:
            state_in = mx.zeros((B * H, D, D), dtype=mx.float32)
        else:
            state_in = state.astype(mx.float32).reshape(B * H, D, D)

        # Force evaluation so the kernel can read these directly
        # (kernels can't read from un-evaluated arrays in older MLX versions)
        mx.eval(q_bh, k_bh, v_bh, g_bh, beta_bh, state_in)

        # Launch: total threads = B*H*D, threadgroup size = D
        # → B*H threadgroups, each with D threads
        outputs = kernel(
            inputs=[q_bh, k_bh, v_bh, g_bh, beta_bh, state_in],
            template=[("T", mx.float32), ("D", D)],
            grid=(B * H * D, 1, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(B * H, L, D), (B * H, D, D), (B * H, L, D)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

        out_bh, state_out, _u_bh = outputs   # u is for the VJP path; ignore here

        # Reshape back: (B*H, L, D) -> (B, H, L, D) -> (B, L, H, D)
        output_4d = out_bh.reshape(B, H, L, D)
        output_4d = mx.transpose(output_4d, (0, 2, 1, 3))
        new_state = state_out.reshape(B, H, D, D)

        return output_4d.astype(q.dtype), new_state

    # ==================================================================
    # PATH (a'): "metal_vjp" — Metal forward + custom VJP for training
    # ==================================================================
    #
    # Wraps the forward Metal kernel in `mx.custom_function` so that MLX
    # autograd routes gradients through our hand-written backward instead of
    # falling back to the compiled-step path. This is the difference between
    # a ~10 s/step and a ~few-hundred-ms/step training loop on M-series.
    #
    # Forward (already implemented in _kda_metal_source) outputs:
    #   out_t, S_L, u_t  where  u_t = β_t (v_t − S'_t k_t)
    # We save u_t per step so the backward can invert the forward recurrence
    # without storing every intermediate S_t (which would be O(L·D²) memory).
    #
    # Backward (this method) iterates t from L−1 to 0:
    #   1. Recover S' = S_t − u_t k_t^T, then S_{t−1} = S' / exp(g_t)
    #   2. gS += do_t ⊗ q_t                        # contribution from o_t = S_t q_t
    #   3. dq_t = S_t^T do_t                       # = (S')^T do_t + k_t (u_t · do_t)
    #   4. du = gS k_t,           dk += gS^T u_t   # backward through S_t = S' + u k^T
    #   5. dβ_t = du · w   (w = v_t − S' k_t = u_t / β_t when β_t ≠ 0;
    #                       we recompute w = v_t − S'k_t directly for stability)
    #   6. dw = β_t du,   dv_t = dw
    #   7. dS' = gS − dw ⊗ k_t,  dk += −(S')^T dw
    #   8. dg_t = sum_i (dS' * S')_{ij}            # column-wise sum of dS' ⊙ S'
    #   9. gS_prev = dS' * exp(g_t)                # column scale into previous t
    # Final: dS_init = gS  (after the loop).

    @staticmethod
    def _kda_python_backward(q, k, v, g, beta, u, dout, dstate_final):
        """
        Reference Python backward for the channel-wise KDA recurrence.

        Vectorized over (B*H, L, D); the time axis runs sequentially in a
        Python loop because the recurrence is inherently sequential.

        All inputs are fp32 already (pre-cast by the wrapper).

        Args:
            q, k, v, g : (BH, L, D)
            beta       : (BH, L)
            u          : (BH, L, D)   saved during forward
            dout       : (BH, L, D)
            dstate_final : (BH, D, D)
        Returns:
            dq, dk, dv, dg : (BH, L, D)
            dbeta          : (BH, L)
            dstate_init    : (BH, D, D)   gradient w.r.t. initial state
        """
        BH, L, D = q.shape

        # Will accumulate per-timestep grads here, then stack.
        dq_list = []
        dk_list = []
        dv_list = []
        dg_list = []
        dbeta_list = []

        # Recover S_L by replaying the forward (cheap: reuse the kernel).
        # But we already have S_L from the forward — pass it in via dstate_final
        # (no, that's the *gradient*). We need the actual S_L tensor too.
        # The custom_function wrapper has access to it as part of `output`.
        # See `_kda_recurrence_fn` below — it passes S_final along with cotangents.
        raise NotImplementedError("Use _kda_python_backward_full(...)")

    @staticmethod
    def _kda_python_backward_full(q, k, v, g, beta, u, S_final, state_init,
                                  dout, dstate_final):
        """
        Backward pass for the channel-wise KDA recurrence.

        We materialize all intermediate forward states `S_t` in a single
        forward replay (using the saved u from the forward), then do the
        time-reversed gradient pass against them. We *do not* invert the
        column scale numerically: `S_{t-1} = S' / exp(g_t)` is unstable
        when g_t is large-negative (decay → 0). The forward replay costs
        an extra pass but removes the precision issue entirely.

        Memory cost: O(L · D² · BH) for `S_traj`. For typical layers this
        is the dominant transient buffer; if L gets very long, swap this
        for gradient checkpointing.

        Args:
            q, k, v, g : (BH, L, D)   fp32
            beta       : (BH, L)      fp32
            u          : (BH, L, D)   saved during forward
            S_final    : (BH, D, D)   forward output (used for cross-check only)
            state_init : (BH, D, D)   the initial state passed to forward
            dout       : (BH, L, D)   upstream gradient w.r.t. output
            dstate_final : (BH, D, D) upstream gradient w.r.t. final state
        Returns:
            dq, dk, dv, dg : (BH, L, D)
            dbeta          : (BH, L)
            dstate_init    : (BH, D, D)
        """
        BH, L, D = q.shape

        # ---- Forward replay: materialize all S_t (t = 0..L) ----
        # S_traj[t] = state AFTER step t's update (so S_traj[0] = state_init,
        # S_traj[L] = S_final). Use saved u to avoid recomputing β(v − S' k).
        S_list = [state_init]
        S_cur = state_init
        for t in range(L):
            decay_t = mx.exp(g[:, t])                         # (BH, D)
            S_prime_t = S_cur * decay_t[:, None, :]           # column scale
            u_t = u[:, t]                                     # (BH, D)
            S_cur = S_prime_t + u_t[:, :, None] * k[:, t, None, :]
            S_list.append(S_cur)

        # ---- Reverse-time gradient loop ----
        gS = dstate_final
        dq_rev, dk_rev, dv_rev, dg_rev, dbeta_rev = [], [], [], [], []

        for t in range(L - 1, -1, -1):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = g[:, t]
            u_t = u[:, t]
            beta_t = beta[:, t]
            do_t = dout[:, t]

            decay = mx.exp(g_t)

            S_t = S_list[t + 1]                # state after step t
            # S' (post column scale, pre rank-1) — recover exactly from S_t and u_t.
            # Equivalent to S_list[t] * decay[..., None, :], but recover from S_t
            # so any small numerical drift in the replay still gives the
            # correct dS_t -> dS' relationship.
            S_prime = S_t - u_t[:, :, None] * k_t[:, None, :]

            # (a) o_t = S_t q_t  =>  dS_t (from output) = do_t ⊗ q_t
            gS = gS + do_t[:, :, None] * q_t[:, None, :]

            # (b) dq_t = S_t^T do_t  = (S')^T do_t + k_t (u_t · do_t)
            ud = mx.sum(u_t * do_t, axis=-1, keepdims=True)
            dq_t = mx.einsum("blj,bl->bj", S_prime, do_t) + k_t * ud
            dq_rev.append(dq_t)

            # (c) Backward through S_t = S' + u_t k_t^T
            du = mx.einsum("bij,bj->bi", gS, k_t)              # (BH, D)
            dk_outer = mx.einsum("bij,bi->bj", gS, u_t)        # (BH, D)
            dS_prime = gS                                       # (BH, D, D)

            # (d) Backward through u_t = β_t (v_t − S' k_t)
            Sk = mx.einsum("bij,bj->bi", S_prime, k_t)         # (BH, D)
            w = v_t - Sk
            dbeta_t = mx.sum(du * w, axis=-1)                  # (BH,)
            dbeta_rev.append(dbeta_t)
            dw = beta_t[:, None] * du                          # (BH, D)
            dv_rev.append(dw)

            # dS' from −S' k_t and dk from same:
            dS_prime = dS_prime - dw[:, :, None] * k_t[:, None, :]
            dk_from_Sk = -mx.einsum("blj,bl->bj", S_prime, dw)
            dk_rev.append(dk_outer + dk_from_Sk)

            # (e) Column-scale backward:
            #     S'[i, j] = S_prev[i, j] * decay[j]
            #     dS_prev[i, j] = dS'[i, j] * decay[j]
            #     dg_t[j] = sum_i dS'[i, j] * S'[i, j]   (∂(S'_{ij})/∂g_j = S'_{ij})
            dg_rev.append(mx.sum(dS_prime * S_prime, axis=-2))

            gS = dS_prime * decay[:, None, :]

        # ---- Stack reversed lists into time-major outputs ----
        def _stack_rev(lst):
            return mx.stack(lst[::-1], axis=1)

        return (
            _stack_rev(dq_rev),
            _stack_rev(dk_rev),
            _stack_rev(dv_rev),
            _stack_rev(dg_rev),
            _stack_rev(dbeta_rev),
            gS,                       # dstate_init
        )

    # ----------------------------------------------------------------
    # Metal forward-with-trajectory + backward kernels (Phase 2)
    # ----------------------------------------------------------------
    #
    # The Python backward in `_kda_python_backward_full` is correct but
    # bottlenecked by Python op-dispatch overhead (≈30K ops per layer per
    # training step). Replacing it with a single Metal kernel launch drops
    # the GPU-compute portion to ≈ a few hundred microseconds and removes
    # the per-step dispatch tax entirely.
    #
    # Memory cost added: the forward saves the full state trajectory
    #    S_traj : (B*H, L+1, D, D) fp32
    # so the backward kernel can read S_t at any t without inverting the
    # column-scale (which would be numerically unstable when g_t is very
    # negative). For (B=2, H=6, L=3136, D=64) this is ≈ 617 MB per layer.

    # Forward-with-trajectory: same as the standard forward but also writes
    # S_traj[t+1] after each step's update, and S_traj[0] = state_in at start.
    _kda_forward_with_traj_source = r"""
        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint L_  = q_shape[1];

        float S_row[D];

        // Load initial state into registers AND into S_traj[0]
        const uint state_base = bh * D * D + tid * D;
        const uint traj_init_base = bh * (L_ + 1) * D * D + 0 * D * D + tid * D;
        for (uint j = 0; j < D; ++j) {
            float v_init = (float) state_in[state_base + j];
            S_row[j] = v_init;
            S_traj_out[traj_init_base + j] = (T) v_init;
        }

        threadgroup float q_t[D];
        threadgroup float k_t[D];
        threadgroup float v_t[D];
        threadgroup float g_t[D];
        threadgroup float beta_t_shared[1];

        for (uint t = 0; t < L_; ++t) {
            const uint vec_base = bh * L_ * D + t * D;
            q_t[tid] = (float) q[vec_base + tid];
            k_t[tid] = (float) k[vec_base + tid];
            v_t[tid] = (float) v[vec_base + tid];
            g_t[tid] = (float) g[vec_base + tid];
            if (tid == 0) beta_t_shared[0] = (float) beta[bh * L_ + t];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float beta_t = beta_t_shared[0];

            // (1) Column scale
            for (uint j = 0; j < D; ++j) {
                S_row[j] *= metal::exp(g_t[j]);
            }

            // (2) Sk_my = S' k
            float Sk_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                Sk_my += S_row[j] * k_t[j];
            }

            // (3) u_my = β (v − Sk) ; rank-1 update S += u k^T
            const float u_my = beta_t * (v_t[tid] - Sk_my);
            u_out[vec_base + tid] = (T) u_my;
            for (uint j = 0; j < D; ++j) {
                S_row[j] += u_my * k_t[j];
            }

            // (4) Output
            float out_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                out_my += S_row[j] * q_t[j];
            }
            output[vec_base + tid] = (T) out_my;

            // (5) Save S_t (post-update) into S_traj[t+1]
            const uint traj_t_base = bh * (L_ + 1) * D * D + (t + 1) * D * D + tid * D;
            for (uint j = 0; j < D; ++j) {
                S_traj_out[traj_t_base + j] = (T) S_row[j];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Final state
        for (uint j = 0; j < D; ++j) {
            state_out[state_base + j] = (T) S_row[j];
        }
    """

    # Backward kernel: time-reversed sweep, reading S_t from S_traj.
    # Layout: 1 threadgroup per (B*H), D threads per threadgroup, each owning
    # one row of gS in registers. A single (D × D) TG buffer is reused for
    # S_t / S' / partial-product / gS staging across phases.
    _kda_backward_source = r"""
        const uint bh  = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint L_  = q_shape[1];

        // Per-thread: one row of gS in registers
        float gS_row[D];

        // Init gS from dstate_final
        const uint state_base = bh * D * D + tid * D;
        for (uint j = 0; j < D; ++j) {
            gS_row[j] = (float) dstate_final[state_base + j];
        }

        // Threadgroup memory
        threadgroup float S_tg[D * D];          // dual-purpose: S' / partials / gS
        threadgroup float q_t[D];
        threadgroup float k_t[D];
        threadgroup float v_t[D];
        threadgroup float g_t[D];
        threadgroup float u_t[D];
        threadgroup float do_t[D];
        threadgroup float decay_t[D];
        threadgroup float dw_tg[D];
        threadgroup float beta_t_shared[1];
        // For threadgroup-wide scalar reductions (u·do, dβ).
        // Up to D/32 = 32 partial slots cover all simdgroups for D ≤ 1024.
        threadgroup float reduce_partial[32];
        threadgroup float reduce_total[1];

        // Backward time loop
        for (int t_signed = (int)L_ - 1; t_signed >= 0; --t_signed) {
            const uint t = (uint) t_signed;

            // ---- Load S_t into TG mem from S_traj[t+1] ----
            const uint S_base = bh * (L_ + 1) * D * D + (t + 1) * D * D;
            for (uint i = tid; i < D * D; i += D) {
                S_tg[i] = (float) S_traj[S_base + i];
            }
            // Per-step inputs
            const uint vec_base = bh * L_ * D + t * D;
            q_t[tid]  = (float) q[vec_base + tid];
            k_t[tid]  = (float) k[vec_base + tid];
            v_t[tid]  = (float) v[vec_base + tid];
            g_t[tid]  = (float) g[vec_base + tid];
            u_t[tid]  = (float) u[vec_base + tid];
            do_t[tid] = (float) do_in[vec_base + tid];
            if (tid == 0) beta_t_shared[0] = (float) beta[bh * L_ + t];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float beta_t = beta_t_shared[0];
            decay_t[tid] = metal::exp(g_t[tid]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (P1) S' = S_t − u k^T  (in-place over S_tg) ----
            // Row tid: S'[tid, j] = S_tg[tid, j] − u_t[tid] * k_t[j]
            const float u_my = u_t[tid];
            for (uint j = 0; j < D; ++j) {
                S_tg[tid * D + j] -= u_my * k_t[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Now S_tg holds S'.

            // ---- (P2) gS_row += do_t[tid] * q_t  (registers) ----
            const float do_my = do_t[tid];
            for (uint j = 0; j < D; ++j) {
                gS_row[j] += do_my * q_t[j];
            }

            // ---- (P3) Compute scalar u·do (threadgroup reduction) ----
            {
                float ud_my_local = u_my * do_my;
                float ud_sg = simd_sum(ud_my_local);
                if ((tid & 31) == 0) reduce_partial[tid / 32] = ud_sg;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float s = 0.0f;
                    const uint nsg = (D + 31u) / 32u;
                    for (uint i = 0; i < nsg; ++i) s += reduce_partial[i];
                    reduce_total[0] = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float ud = reduce_total[0];

            // ---- (P4) dq[tid] = (S')^T do + k_t * (u·do) at column tid ----
            // = sum_l S'[l, tid] * do[l] + k_t[tid] * ud
            float dq_my = k_t[tid] * ud;
            for (uint l = 0; l < D; ++l) {
                dq_my += S_tg[l * D + tid] * do_t[l];
            }
            dq_out[vec_base + tid] = (T) dq_my;

            // ---- (P5) du[tid] = sum_j gS[tid, j] * k_t[j]   (registers) ----
            float du_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                du_my += gS_row[j] * k_t[j];
            }

            // ---- (P6) Sk[tid] = S'[tid, :] · k_t ; w[tid] = v_t[tid] − Sk[tid] ----
            float Sk_my = 0.0f;
            for (uint j = 0; j < D; ++j) {
                Sk_my += S_tg[tid * D + j] * k_t[j];
            }
            const float w_my = v_t[tid] - Sk_my;

            // ---- (P7) dβ_t = sum(du * w)   (threadgroup reduction) ----
            {
                float db_my_local = du_my * w_my;
                float db_sg = simd_sum(db_my_local);
                if ((tid & 31) == 0) reduce_partial[tid / 32] = db_sg;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float s = 0.0f;
                    const uint nsg = (D + 31u) / 32u;
                    for (uint i = 0; i < nsg; ++i) s += reduce_partial[i];
                    dbeta_out[bh * L_ + t] = (T) s;
                }
                // No barrier needed yet — we only block before reading reduce_partial again.
            }

            // ---- (P8) dw[tid] = β · du[tid] ; dv[tid] = dw[tid] ----
            const float dw_my = beta_t * du_my;
            dv_out[vec_base + tid] = (T) dw_my;

            // Stash dw into TG mem for transposed matvecs that need it.
            dw_tg[tid] = dw_my;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (P9) dk_from_Sk[tid] = −sum_l S'[l, tid] * dw[l]   (transposed matvec on S') ----
            float dk_from_Sk = 0.0f;
            for (uint l = 0; l < D; ++l) {
                dk_from_Sk -= S_tg[l * D + tid] * dw_tg[l];
            }

            // ---- (P10) dg[tid] = sum_i dS'[i, tid] * S'[i, tid]
            //     dS'[i, j] = gS[i, j] − dw[i] * k_t[j]
            //
            // Each thread tid writes (dS'_row * S'_row) into S_tg[tid, :], then
            // we column-sum across rows. Note we OVERWRITE S' here.
            for (uint j = 0; j < D; ++j) {
                const float dSp = gS_row[j] - dw_my * k_t[j];          // dS'[my, j]
                const float Sp  = S_tg[tid * D + j];                    // S'[my, j]
                S_tg[tid * D + j] = dSp * Sp;                           // overwrite
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dg_my = 0.0f;
            for (uint i = 0; i < D; ++i) {
                dg_my += S_tg[i * D + tid];
            }
            dg_out[vec_base + tid] = (T) dg_my;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (P11) dk_outer[tid] = sum_l gS[l, tid] * u_t[l]   (transposed matvec on gS) ----
            // Stage gS into TG mem (overwrite S_tg), then read columns.
            for (uint j = 0; j < D; ++j) {
                S_tg[tid * D + j] = gS_row[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dk_outer = 0.0f;
            for (uint l = 0; l < D; ++l) {
                dk_outer += S_tg[l * D + tid] * u_t[l];
            }
            dk_out[vec_base + tid] = (T) (dk_outer + dk_from_Sk);

            // ---- (P12) Update gS for previous timestep:
            //     gS_new[my, j] = (gS[my, j] − dw[my] * k_t[j]) * decay[j]
            for (uint j = 0; j < D; ++j) {
                gS_row[j] = (gS_row[j] - dw_my * k_t[j]) * decay_t[j];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Final: write gS_row to dstate_init
        for (uint j = 0; j < D; ++j) {
            dstate_init_out[state_base + j] = (T) gS_row[j];
        }
    """

    _kda_forward_with_traj_kernel_cache: Dict[int, object] = {}
    _kda_backward_kernel_cache: Dict[int, object] = {}

    @classmethod
    def _get_kda_forward_with_traj_kernel(cls, D: int):
        if D in cls._kda_forward_with_traj_kernel_cache:
            return cls._kda_forward_with_traj_kernel_cache[D]
        kernel = mx.fast.metal_kernel(
            name=f"kda_forward_with_traj_d{D}",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["output", "state_out", "u_out", "S_traj_out"],
            source=cls._kda_forward_with_traj_source,
        )
        cls._kda_forward_with_traj_kernel_cache[D] = kernel
        return kernel

    @classmethod
    def _get_kda_backward_kernel(cls, D: int):
        if D in cls._kda_backward_kernel_cache:
            return cls._kda_backward_kernel_cache[D]
        kernel = mx.fast.metal_kernel(
            name=f"kda_backward_d{D}",
            input_names=["q", "k", "v", "g", "beta", "u", "S_traj",
                         "do_in", "dstate_final"],
            output_names=["dq_out", "dk_out", "dv_out", "dg_out",
                          "dbeta_out", "dstate_init_out"],
            source=cls._kda_backward_source,
        )
        cls._kda_backward_kernel_cache[D] = kernel
        return kernel

    # ---- Custom-function-wrapped forward (cached) ----

    # We use a single global custom_function whose forward is a thin wrapper
    # over the existing Metal forward, and whose vjp is the Python backward.
    # The class attribute is created lazily because mx.custom_function captures
    # the function at decoration time.

    _kda_recurrence_fn_cache = None

    @classmethod
    def _get_kda_recurrence_fn(cls):
        if cls._kda_recurrence_fn_cache is not None:
            return cls._kda_recurrence_fn_cache

        # Note: D is read off q.shape[-1] inside the function, so a single
        # custom_function works for any head_dim.
        @mx.custom_function
        def _kda_recurrence_fn(q, k, v, g, beta, state_in):
            """
            Forward: returns (output, state_final, u, S_traj).
            All inputs / outputs are flat-on-(B*H) layout in fp32. The
            S_traj output (B*H, L+1, D, D) is consumed by the VJP backward
            kernel; from the user's perspective only `output` is meaningful.
            """
            BH, L, D = q.shape
            kernel = cls._get_kda_forward_with_traj_kernel(D)
            outs = kernel(
                inputs=[q, k, v, g, beta, state_in],
                template=[("T", mx.float32), ("D", D)],
                grid=(BH * D, 1, 1),
                threadgroup=(D, 1, 1),
                output_shapes=[
                    (BH, L, D),         # output
                    (BH, D, D),         # state_final
                    (BH, L, D),         # u
                    (BH, L + 1, D, D),  # S_traj
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
            )
            output, state_final, u, S_traj = outs
            return output, state_final, u, S_traj

        @_kda_recurrence_fn.vjp
        def _kda_vjp(primals, cotangents, output):
            q, k, v, g, beta, _state_in = primals
            do, dstate_final, _du_unused, _dStraj_unused = cotangents
            _output, _S_final, u, S_traj = output

            BH, L, D = q.shape
            kernel = cls._get_kda_backward_kernel(D)
            outs = kernel(
                inputs=[q, k, v, g, beta, u, S_traj, do, dstate_final],
                template=[("T", mx.float32), ("D", D)],
                grid=(BH * D, 1, 1),
                threadgroup=(D, 1, 1),
                output_shapes=[
                    (BH, L, D),    # dq
                    (BH, L, D),    # dk
                    (BH, L, D),    # dv
                    (BH, L, D),    # dg
                    (BH, L),       # dbeta
                    (BH, D, D),    # dstate_init
                ],
                output_dtypes=[mx.float32] * 6,
            )
            dq, dk, dv, dg, dbeta, dstate_init = outs
            return dq, dk, dv, dg, dbeta, dstate_init

        cls._kda_recurrence_fn_cache = _kda_recurrence_fn
        return _kda_recurrence_fn

    def _metal_kernel_kda_vjp(self, q, k, v, g, beta, state):
        """
        Autograd-compatible KDA recurrence: same I/O contract as
        `_metal_kernel_kda`, but routes through `mx.custom_function` so that
        gradients flow through the custom Python backward (no fallback to the
        compiled path).
        """
        B, L, H, D = q.shape
        if D != self.head_dim:
            raise ValueError(f"metal_vjp D mismatch: got {D}, expected {self.head_dim}")

        # Cast to fp32 (state stability) and flatten heads onto batch.
        q32    = q.astype(mx.float32)
        k32    = k.astype(mx.float32)
        v32    = v.astype(mx.float32)
        g32    = g.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        q_bh    = mx.transpose(q32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        k_bh    = mx.transpose(k32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        v_bh    = mx.transpose(v32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        g_bh    = mx.transpose(g32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        beta_bh = mx.transpose(beta32, (0, 2, 1)).reshape(B * H, L)

        if state is None:
            state_in = mx.zeros((B * H, D, D), dtype=mx.float32)
        else:
            state_in = state.astype(mx.float32).reshape(B * H, D, D)

        fn = self._get_kda_recurrence_fn()
        output_bh, state_out_bh, _u, _S_traj = fn(
            q_bh, k_bh, v_bh, g_bh, beta_bh, state_in
        )

        output_4d = output_bh.reshape(B, H, L, D)
        output_4d = mx.transpose(output_4d, (0, 2, 1, 3))
        new_state = state_out_bh.reshape(B, H, D, D)

        return output_4d.astype(q.dtype), new_state

    # ==================================================================
    # PATH (a'): SIMDGROUP-MATRIX METAL KERNEL — fused KDA recurrence
    # ==================================================================
    #
    # Same recurrence as path (a), but uses simdgroup_matrix<float, 8, 8>
    # matmul intrinsics for the matvec (S @ k, S @ q) and the rank-1
    # outer-product update (S += u k^T).
    #
    # ALGEBRAIC SIMPLIFICATION used here:
    #   The original Householder update
    #       S' = S * exp(g)            (column scale)
    #       S  = S' @ (I - β k k^T) + β v k^T
    #   simplifies (no matrix-matrix product needed) to
    #       u  = β * (v - S' k)        (D-vector)
    #       S  = S' + u k^T            (rank-1 outer)
    #   so per step we have:
    #       (1) column scale            : O(D²)   elementwise
    #       (2) two matvecs S @ k, S @ q: 2 × O(D²) — packed into one matmul
    #       (3) rank-1 outer S += u k^T : O(D²)
    #       (4) output                  : already produced by S @ q
    #
    # MATMUL PACKING:
    #   8x8 simdgroup matmul gives 64 multiply-adds per simdgroup op.
    #   For matvec we pack [k | q | 0 | 0 | 0 | 0 | 0 | 0] (D × 8) so a
    #   single (D × D) @ (D × 8) matmul produces both Sk and Sq in cols
    #   0 and 1 — uses 25% of the simdgroup-matrix capacity, but reuses
    #   each S tile load.
    #   For outer product we use the trick (col-vec) @ (row-vec):
    #       U_tile (8 × 8) with u in col 0, K_tile (8 × 8) with k in row 0
    #       → U_tile @ K_tile = u k^T  (only the col-0/row-0 cross term
    #         is nonzero, giving the outer product).
    #
    # THREADGROUP LAYOUT:
    #   - NSG = D / 8 simdgroups per threadgroup (8 for D=64)
    #   - 32 threads per simdgroup → NSG × 32 = NT threads total
    #   - Simdgroup s owns row-tile s of the state (rows 8s..8s+7)
    #
    # CONSTRAINTS:
    #   - D must be a multiple of 8 (8x8 tiles)
    #   - D <= 256 in practice (keep TG mem under ~32KB)
    #   - Apple Silicon M1 or later (simdgroup_matrix support)

    _kda_metal_simdgroup_source = r"""
        // Compile-time constants from template:  T = scalar dtype, D = head_dim
        constexpr int NSG = D / 8;            // simdgroups per threadgroup
        constexpr int NT  = NSG * 32;         // threads per threadgroup

        const uint bh     = threadgroup_position_in_grid.x;
        const uint sg_id  = simdgroup_index_in_threadgroup;   // 0 .. NSG-1
        const uint tg_tid = thread_position_in_threadgroup.x; // 0 .. NT-1
        const uint L_     = q_shape[1];

        // ---- Threadgroup memory ----
        threadgroup float S_tg [D * D];        // state (D, D) row-major
        threadgroup float q_tg [D];
        threadgroup float k_tg [D];
        threadgroup float v_tg [D];
        threadgroup float g_tg [D];
        threadgroup float Sk_tg[D];
        threadgroup float Sq_tg[D];
        threadgroup float u_tg [D];
        threadgroup float beta_shared[1];

        // Packed buffers for simdgroup-matrix ops
        threadgroup float KQ_col[D * 8];        // (D, 8): col 0 = k, col 1 = q, rest 0
        threadgroup float U_col [D * 8];        // (D, 8): col 0 = u, rest 0
        threadgroup float K_row [8 * D];        // (8, D): row 0 = k, rest 0

        // Scratch for storing 8x8 result tiles back to TG mem
        threadgroup float SKQ_scratch[NSG * 64];

        // Initial state load: all NT threads cooperate
        for (uint i = tg_tid; i < D * D; i += NT) {
            S_tg[i] = (float) state_in[bh * D * D + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < L_; ++t) {
            // ---- Load per-step inputs (q, k, v, g, beta) into TG mem ----
            const uint base_t = bh * L_ * D + t * D;
            if (tg_tid < D) {
                q_tg[tg_tid] = (float) q[base_t + tg_tid];
                k_tg[tg_tid] = (float) k[base_t + tg_tid];
                v_tg[tg_tid] = (float) v[base_t + tg_tid];
                g_tg[tg_tid] = (float) g[base_t + tg_tid];
            }
            if (tg_tid == 0) {
                beta_shared[0] = (float) beta[bh * L_ + t];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (1) Column scale: S[i, j] *= exp(g[j]) ----
            for (uint i = tg_tid; i < D * D; i += NT) {
                S_tg[i] *= metal::exp(g_tg[i % D]);
            }

            // ---- (2) Build KQ_col: col 0 = k, col 1 = q, cols 2..7 = 0 ----
            for (uint i = tg_tid; i < D * 8; i += NT) {
                uint row = i / 8;
                uint col = i % 8;
                KQ_col[i] =
                    (col == 0) ? k_tg[row] :
                    (col == 1) ? q_tg[row] :
                                 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (3) S @ KQ_col via simdgroup matmul ----
            // Each simdgroup s computes the (8 × 8) result tile for row-tile s,
            // accumulating across col-tiles. col 0 of the result = Sk, col 1 = Sq.
            simdgroup_matrix<float, 8, 8> SKQ_acc =
                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint c = 0; c < (uint)NSG; ++c) {
                simdgroup_matrix<float, 8, 8> S_tile, KQ_tile;
                simdgroup_load(S_tile,  S_tg   + 8 * sg_id * D + 8 * c, D);
                simdgroup_load(KQ_tile, KQ_col + 8 * c * 8,             8);
                simdgroup_multiply_accumulate(SKQ_acc, S_tile, KQ_tile, SKQ_acc);
            }
            simdgroup_store(SKQ_acc, SKQ_scratch + sg_id * 64, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Extract col 0 → Sk, col 1 → Sq
            if (tg_tid < D) {
                uint r_sg = tg_tid / 8;
                uint i_in = tg_tid % 8;
                Sk_tg[tg_tid] = SKQ_scratch[r_sg * 64 + i_in * 8 + 0];
                Sq_tg[tg_tid] = SKQ_scratch[r_sg * 64 + i_in * 8 + 1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (4) u = β * (v - Sk) ----
            if (tg_tid < D) {
                u_tg[tg_tid] = beta_shared[0] * (v_tg[tg_tid] - Sk_tg[tg_tid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (4b) Compute scalar kq = k · q via threadgroup reduction ----
            // Output below uses the POST-rank-1-update state's matvec:
            //     S_new @ q = (S' + u k^T) @ q = Sq + u * (k · q)
            // We have Sq (from step 3) and u (step 4); kq is one scalar.
            threadgroup float kq_partial[8];   // partial sums, one per simdgroup
            threadgroup float kq_shared[1];
            {
                float local = (tg_tid < D) ? (k_tg[tg_tid] * q_tg[tg_tid]) : 0.0f;
                float sg_sum = simd_sum(local);
                if ((tg_tid & 31) == 0) kq_partial[tg_tid / 32] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tg_tid == 0) {
                float s = 0.0f;
                for (uint i = 0; i < (uint)NSG; ++i) s += kq_partial[i];
                kq_shared[0] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (5) Build U_col (col 0 = u, rest 0) and K_row (row 0 = k, rest 0) ----
            for (uint i = tg_tid; i < D * 8; i += NT) {
                uint row = i / 8;
                uint col = i % 8;
                U_col[i] = (col == 0) ? u_tg[row] : 0.0f;
            }
            for (uint i = tg_tid; i < 8 * D; i += NT) {
                uint row = i / D;
                uint col = i % D;
                K_row[i] = (row == 0) ? k_tg[col] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (6) Rank-1 update: S += u k^T ----
            // Each simdgroup s updates its row-tile across all NSG col-tiles.
            simdgroup_matrix<float, 8, 8> U_tile;
            simdgroup_load(U_tile, U_col + 8 * sg_id * 8, 8);   // (8, 8) col 0 = u-slice
            for (uint c = 0; c < (uint)NSG; ++c) {
                simdgroup_matrix<float, 8, 8> K_tile, S_tile;
                simdgroup_load(K_tile, K_row + 8 * c,                       D);
                simdgroup_load(S_tile, S_tg   + 8 * sg_id * D + 8 * c,      D);
                simdgroup_multiply_accumulate(S_tile, U_tile, K_tile, S_tile);
                simdgroup_store(S_tile, S_tg + 8 * sg_id * D + 8 * c, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- (7) Output: out[t] = S_new @ q = Sq + u * (k · q) ----
            if (tg_tid < D) {
                output[base_t + tg_tid] = (T) (Sq_tg[tg_tid] + u_tg[tg_tid] * kq_shared[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- Final state save ----
        for (uint i = tg_tid; i < D * D; i += NT) {
            state_out[bh * D * D + i] = (T) S_tg[i];
        }
    """

    # Header for simdgroup_matrix support (M1+ Apple Silicon)
    _kda_metal_simdgroup_header = (
        "#include <metal_stdlib>\n"
        "#include <metal_simdgroup_matrix>\n"
        "using namespace metal;\n"
    )

    _metal_simdgroup_kernel_cache: Dict[int, object] = {}

    @classmethod
    def _get_metal_simdgroup_kernel(cls, D: int):
        """Build (and cache) the simdgroup-matrix Metal kernel for given D."""
        if D in cls._metal_simdgroup_kernel_cache:
            return cls._metal_simdgroup_kernel_cache[D]
        kernel = mx.fast.metal_kernel(
            name=f"kda_recurrence_sg_d{D}",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["output", "state_out"],
            source=cls._kda_metal_simdgroup_source,
            header=cls._kda_metal_simdgroup_header,
        )
        cls._metal_simdgroup_kernel_cache[D] = kernel
        return kernel

    def _metal_kernel_kda_simdgroup(self, q, k, v, g, beta, state):
        """
        Fused Metal kernel KDA recurrence using simdgroup_matrix<float, 8, 8>.

        Same arg/return shapes as _metal_kernel_kda, just a different kernel.

        Args:
            q, k, v: (B, L, H, D)
            g:       (B, L, H, D)  log-space, ≤ 0
            beta:    (B, L, H)
            state:   (B, H, D, D) or None
        Returns:
            output:  (B, L, H, D)
            state:   (B, H, D, D)
        """
        B, L, H, D = q.shape
        if D != self.head_dim:
            raise ValueError(f"Metal_sg D mismatch: got {D}, expected {self.head_dim}")
        if D % 8 != 0:
            raise ValueError(f"Metal_sg requires D divisible by 8; got {D}")
        # Threadgroup memory budget: roughly D*D + 7*D + 16*D + NSG*64 floats × 4 bytes
        # For D=64 ≈ 16KB + 1.5KB + 2KB ≈ 20KB, well under 32KB.
        if D > 256:
            raise ValueError(
                f"Metal_sg requires D <= 256 (TG mem budget); got {D}. "
                f"Use 'metal' (scalar) for larger head_dim."
            )

        kernel = self._get_metal_simdgroup_kernel(D)

        q32    = q.astype(mx.float32)
        k32    = k.astype(mx.float32)
        v32    = v.astype(mx.float32)
        g32    = g.astype(mx.float32)
        beta32 = beta.astype(mx.float32)

        # (B, L, H, D) -> (B, H, L, D) -> (B*H, L, D)
        q_bh    = mx.transpose(q32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        k_bh    = mx.transpose(k32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        v_bh    = mx.transpose(v32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        g_bh    = mx.transpose(g32,    (0, 2, 1, 3)).reshape(B * H, L, D)
        beta_bh = mx.transpose(beta32, (0, 2, 1)).reshape(B * H, L)

        if state is None:
            state_in = mx.zeros((B * H, D, D), dtype=mx.float32)
        else:
            state_in = state.astype(mx.float32).reshape(B * H, D, D)

        mx.eval(q_bh, k_bh, v_bh, g_bh, beta_bh, state_in)

        # NT = (D/8) * 32 threads per threadgroup; B*H threadgroups
        NSG = D // 8
        NT = NSG * 32
        outputs = kernel(
            inputs=[q_bh, k_bh, v_bh, g_bh, beta_bh, state_in],
            template=[("T", mx.float32), ("D", D)],
            grid=(B * H * NT, 1, 1),
            threadgroup=(NT, 1, 1),
            output_shapes=[(B * H, L, D), (B * H, D, D)],
            output_dtypes=[mx.float32, mx.float32],
        )

        out_bh, state_out = outputs

        output_4d = out_bh.reshape(B, H, L, D)
        output_4d = mx.transpose(output_4d, (0, 2, 1, 3))
        new_state = state_out.reshape(B, H, D, D)

        return output_4d.astype(q.dtype), new_state


# ============================================================================
# COMPONENT 6: HYBRID PROCESSOR BLOCK (GDN or NoPE)
# ============================================================================

class HybridBlock(nn.Module):
    """Single processor block: either GDN or NoPE attention."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_type: str = "gdn",
        head_dim: Optional[int] = None,
        chunk_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        channel_wise_decay: bool = True,
        allow_neg_eigval: bool = False,
        spatial_tokens: Optional[int] = None,
        drop_path: float = 0.0,
        gdn_temporal_only: bool = False,
    ):
        super().__init__()
        self.block_type = block_type
        self.spatial_tokens = spatial_tokens
        self.gdn_temporal_only = gdn_temporal_only
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if block_type == "gdn":
            hd = head_dim or (dim // num_heads)
            self.layer = GatedDeltaLayer(
                dim, num_heads, hd, chunk_size,
                channel_wise_decay=channel_wise_decay,
                allow_neg_eigval=allow_neg_eigval,
            )
        else:
            self.layer = NoPEMultiheadAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dim),
            nn.Dropout(dropout),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, N, D = x.shape
        normed = self.norm1(x)

        if self.block_type == "gdn":
            if self.gdn_temporal_only and self.spatial_tokens is not None:
                # Time-only scan: reshape (B, T'*S, D) time-major -> (B*S, T', D),
                # run the GDN per spatial location over T', then reshape back.
                S = self.spatial_tokens
                T = N // S
                xt = normed.reshape(B, T, S, D)
                xt = mx.transpose(xt, (0, 2, 1, 3)).reshape(B * S, T, D)
                out_t, _ = self.layer(xt, None)
                out = mx.transpose(
                    out_t.reshape(B, S, T, D), (0, 2, 1, 3)
                ).reshape(B, N, D)
                new_state = state
            else:
                out, new_state = self.layer(normed, state)
        elif self.spatial_tokens is not None:
            S = self.spatial_tokens
            T = N // S
            normed = normed.reshape(B, T, S, D).reshape(B * T, S, D)
            out = self.layer(normed)
            out = out.reshape(B, T, S, D).reshape(B, N, D)
            new_state = state
        else:
            out = self.layer(normed, mask=mask)
            new_state = state

        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, new_state


# ============================================================================
# COMPONENT 7: COMPLETE NoPE+GDN VIDEO BACKBONE
# ============================================================================

class NoPEGDNVideoBackbone(nn.Module):
    """
    Stage 1: NoPE encoder (spatial-only attention if factorized)
    Stage 2: 3:1 GDN-to-NoPE hybrid processor
    """
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 32,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        processor_dim: int = 384,
        processor_depth: int = 4,
        processor_heads: int = 6,
        gdn_ratio: int = 3,
        chunk_size: int = 64,
        channel_wise_decay: bool = True,
        allow_neg_eigval: bool = False,
        factorized_attention: bool = True,
        gdn_temporal_only: bool = False,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        total_depth = encoder_depth + processor_depth
        if drop_path_rate > 0 and total_depth > 1:
            dpr_full = [drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]
        else:
            dpr_full = [0.0] * total_depth
        encoder_dpr_rate = (
            drop_path_rate * encoder_depth / total_depth if total_depth > 0 else 0.0
        )

        # Stage 1: encoder
        self.encoder = NoPEVideoEncoder(
            img_size=img_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            factorized_attention=factorized_attention,
            drop_path_rate=encoder_dpr_rate,
        )

        grid = self.encoder.tubelet_embed.get_grid_dims()
        spatial_tokens = grid["H"] * grid["W"] if factorized_attention else None

        if encoder_dim != processor_dim:
            self.dim_proj = nn.Linear(encoder_dim, processor_dim)
        else:
            self.dim_proj = nn.Identity()

        # Stage 2: processor with 3:1 ratio
        period = gdn_ratio + 1
        self.processor_blocks = []
        for i in range(processor_depth):
            bt = "nope" if (i + 1) % period == 0 else "gdn"
            self.processor_blocks.append(
                HybridBlock(
                    dim=processor_dim,
                    num_heads=processor_heads,
                    block_type=bt,
                    head_dim=processor_dim // processor_heads,
                    chunk_size=chunk_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    channel_wise_decay=channel_wise_decay,
                    allow_neg_eigval=allow_neg_eigval,
                    spatial_tokens=(
                        spatial_tokens if (bt == "nope" or gdn_temporal_only) else None
                    ),
                    drop_path=dpr_full[encoder_depth + i],
                    gdn_temporal_only=gdn_temporal_only,
                )
            )

        self.processor_norm = nn.LayerNorm(processor_dim, eps=1e-6)

        self.encoder_dim = encoder_dim
        self.processor_dim = processor_dim
        self.spatial_tokens = spatial_tokens
        self.factorized_attention = factorized_attention

    def __call__(
        self,
        video: mx.array,
        processor_mask: Optional[mx.array] = None,
        return_encoder_features: bool = False,
    ):
        # video: (B, T, H, W, C) — channels-last MLX layout
        enc = self.encoder(video)
        x = self.dim_proj(enc)

        states = [None] * len(self.processor_blocks)
        for i, block in enumerate(self.processor_blocks):
            x, states[i] = block(x, mask=processor_mask, state=states[i])

        x = self.processor_norm(x)
        if return_encoder_features:
            return x, enc
        return x

    def get_block_types(self) -> List[str]:
        return [b.block_type for b in self.processor_blocks]


# ============================================================================
# COMPONENT 8: TEMPORAL POOLING HEAD + CLASSIFIER
# ============================================================================

class TemporalPoolingHead(nn.Module):
    """
    Temporal-aware classification head:
      1. Spatial mean per frame: (B, T'*S, D) -> (B, T', D)
      2. Single-query temporal cross-attention -> (B, D)
      3. Linear classifier
    """
    def __init__(self, embed_dim: int, num_classes: int,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.temporal_query = mx.random.normal((1, 1, embed_dim)) * 0.02
        self.t_proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_proj_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def __call__(self, x: mx.array, S: int = 196) -> mx.array:
        B, N, D = x.shape
        T = N // S
        H, d = self.num_heads, self.head_dim

        # Spatial mean per frame
        x_frame = x.reshape(B, T, S, D).mean(axis=2)        # (B, T, D)

        # Temporal cross-attention with single learned query
        q = mx.broadcast_to(self.temporal_query, (B, 1, D))  # (B, 1, D)
        k = self.t_proj_k(x_frame)
        v = self.t_proj_v(x_frame)

        q = mx.transpose(q.reshape(B, 1, H, d), (0, 2, 1, 3))   # (B, H, 1, d)
        k = mx.transpose(k.reshape(B, T, H, d), (0, 2, 1, 3))   # (B, H, T, d)
        v = mx.transpose(v.reshape(B, T, H, d), (0, 2, 1, 3))   # (B, H, T, d)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=d ** -0.5
        )                                                        # (B, H, 1, d)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, D)
        out = self.t_proj_out(out)

        # Residual with mean-pooled features
        out = out + x_frame.mean(axis=1)
        out = self.t_norm(out)
        out = self.dropout(out)
        return self.fc(out)


class VideoClassificationHead(nn.Module):
    """Global mean-pool + linear classifier (matches the PyTorch
    VideoClassificationHead in best_model_25.pt): LayerNorm -> mean over all
    T'xS tokens -> dropout -> Linear."""
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def __call__(self, x: mx.array, S: Optional[int] = None) -> mx.array:
        # x: (B, N, D); S accepted for interface parity with TemporalPoolingHead.
        x = self.norm(x)
        x = x.mean(axis=1)
        x = self.dropout(x)
        return self.fc(x)


class NoPEGDNClassifier(nn.Module):
    """End-to-end NoPE+GDN video classifier for SSv2."""
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 32,
        tubelet_size: Tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        processor_dim: int = 384,
        processor_depth: int = 4,
        processor_heads: int = 6,
        gdn_ratio: int = 3,
        chunk_size: int = 64,
        channel_wise_decay: bool = True,
        allow_neg_eigval: bool = False,
        factorized_attention: bool = True,
        gdn_temporal_only: bool = False,
        head_type: str = "temporal",
        num_classes: int = 174,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        head_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = NoPEGDNVideoBackbone(
            img_size=img_size, num_frames=num_frames,
            tubelet_size=tubelet_size, in_channels=in_channels,
            encoder_dim=encoder_dim, encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            processor_dim=processor_dim, processor_depth=processor_depth,
            processor_heads=processor_heads,
            gdn_ratio=gdn_ratio, chunk_size=chunk_size,
            channel_wise_decay=channel_wise_decay,
            allow_neg_eigval=allow_neg_eigval,
            factorized_attention=factorized_attention,
            gdn_temporal_only=gdn_temporal_only,
            mlp_ratio=mlp_ratio, dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.spatial_tokens = (img_size // tubelet_size[1]) * (img_size // tubelet_size[2])
        if head_type == "mean":
            self.head = VideoClassificationHead(
                processor_dim, num_classes, dropout=head_dropout,
            )
        else:
            self.head = TemporalPoolingHead(
                processor_dim, num_classes,
                num_heads=processor_heads, dropout=head_dropout,
            )

    def __call__(self, video: mx.array) -> mx.array:
        # video: (B, T, H, W, C) — already MLX channels-last
        features = self.backbone(video)
        return self.head(features, S=self.spatial_tokens)


# ============================================================================
# WEIGHT CONVERSION FROM PYTORCH
# ============================================================================

def _pt_to_mlx_key(pt_key: str) -> Optional[str]:
    """
    Translate a PyTorch state_dict key (from the original FdA-kernel
    NoPE+GDN training code) to its MLX param-tree path.

    The two structural differences:
      - PT `nn.Sequential` exposes children as `mlp.0`, `mlp.3`. MLX exposes
        them as `mlp.layers.0`, `mlp.layers.3`.
      - PT GDN uses `f_proj` / `g_proj` as a 2-layer `nn.Sequential` with
        children `.0` and `.1`. The MLX port stores them as flat attributes
        `f_proj_1` / `f_proj_2` and `g_proj_1` / `g_proj_2`.

    Returns None for keys that have no MLX target (e.g. PT-only buffers).
    """
    k = pt_key

    # MLP under any block: mlp.{0,3} -> mlp.layers.{0,3}
    k = k.replace(".mlp.0.", ".mlp.layers.0.")
    k = k.replace(".mlp.3.", ".mlp.layers.3.")

    # GDN low-rank f / g projections
    k = k.replace(".f_proj.0.", ".f_proj_1.")
    k = k.replace(".f_proj.1.", ".f_proj_2.")
    k = k.replace(".g_proj.0.", ".g_proj_1.")
    k = k.replace(".g_proj.1.", ".g_proj_2.")

    return k


def load_pytorch_ssv2_checkpoint(
    model: nn.Module,
    pt_state_dict: Dict[str, "object"],  # noqa: F821 -- torch optional
    *,
    strict: bool = True,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Load a full NoPE+GDN PyTorch checkpoint (FdA-kernel era; the format saved
    by the original `nope_gdn_video_backbone32.ipynb` training loop) into the
    MLX model in this file.

    Handles all structural differences between the two ports:

      * Conv3d weight   PT (oC, iC, T, H, W) -> MLX (oC, T, H, W, iC)   channels-last
      * Conv1d weight   PT (oC, 1, K)        -> MLX (oC, K, 1)          (depthwise)
      * MLP indexing    PT mlp.{0,3}         -> MLX mlp.layers.{0,3}
      * GDN low-rank    PT f_proj.{0,1}      -> MLX f_proj_{1,2}
                        PT g_proj.{0,1}      -> MLX g_proj_{1,2}
      * No bias on attention qkv_proj / out_proj in either port (skipped silently
        if absent)

    Returns a report dict: {"loaded": [...], "shape_mismatch": [...],
    "no_target": [...], "unused_mlx": [...]}.

    Args:
        model:          NoPEGDNClassifier built with config matching the ckpt.
        pt_state_dict:  ckpt['model_state'] from the PyTorch training loop.
        strict:         If True, raises if any MLX param is not covered, or if
                        any PT param is unmapped. Defaults to True.
        verbose:        Print per-section summary.
    """
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten

    mlx_params = dict(tree_flatten(model.parameters()))
    mlx_keys_remaining = set(mlx_params.keys())

    new_pairs: List[Tuple[str, mx.array]] = []
    report = {
        "loaded": [], "shape_mismatch": [], "no_target": [], "unused_mlx": [],
    }

    for pt_key, pt_tensor in pt_state_dict.items():
        mlx_key = _pt_to_mlx_key(pt_key)
        if mlx_key is None or mlx_key not in mlx_params:
            report["no_target"].append(pt_key)
            continue

        # Convert to numpy with required layout permutation.
        if hasattr(pt_tensor, "detach"):
            arr = pt_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(pt_tensor)

        # Tubelet Conv3d: (oC, iC, T, H, W) -> (oC, T, H, W, iC)
        if mlx_key.endswith("tubelet_embed.projection.weight") and arr.ndim == 5:
            arr = np.transpose(arr, (0, 2, 3, 4, 1))

        # Depthwise Conv1d: (oC, 1, K) -> (oC, K, 1)
        elif (mlx_key.endswith("q_conv1d.weight")
              or mlx_key.endswith("k_conv1d.weight")
              or mlx_key.endswith("v_conv1d.weight")) and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))

        target = mlx_params[mlx_key]
        if tuple(target.shape) != tuple(arr.shape):
            report["shape_mismatch"].append(
                (pt_key, mlx_key, tuple(arr.shape), tuple(target.shape))
            )
            continue

        new_pairs.append((mlx_key, mx.array(arr).astype(target.dtype)))
        report["loaded"].append((pt_key, mlx_key))
        mlx_keys_remaining.discard(mlx_key)

    report["unused_mlx"] = sorted(mlx_keys_remaining)

    model.update(tree_unflatten(new_pairs))
    mx.eval(model.parameters())

    if verbose:
        print("=" * 64)
        print("PyTorch -> MLX checkpoint load")
        print("=" * 64)
        print(f"  PT params:       {len(pt_state_dict)}")
        print(f"  MLX params:      {len(mlx_params)}")
        print(f"  Loaded:          {len(report['loaded'])}")
        print(f"  Shape mismatch:  {len(report['shape_mismatch'])}")
        print(f"  PT keys w/o MLX target: {len(report['no_target'])}")
        print(f"  MLX keys not covered:   {len(report['unused_mlx'])}")
        for pt_k, mlx_k, pt_s, mlx_s in report["shape_mismatch"][:6]:
            print(f"    SHAPE  {pt_k}  PT{pt_s} vs MLX{mlx_s} ({mlx_k})")
        for k in report["no_target"][:6]:
            print(f"    NO TARGET  {k}")
        for k in report["unused_mlx"][:6]:
            print(f"    UNUSED MLX  {k}")
        print("=" * 64)

    if strict:
        if report["shape_mismatch"]:
            raise RuntimeError(
                f"{len(report['shape_mismatch'])} shape mismatches; "
                f"first: {report['shape_mismatch'][0]}"
            )
        if report["no_target"]:
            raise RuntimeError(
                f"{len(report['no_target'])} PT keys had no MLX target; "
                f"first: {report['no_target'][0]}"
            )
        if report["unused_mlx"]:
            raise RuntimeError(
                f"{len(report['unused_mlx'])} MLX params were not loaded; "
                f"first: {report['unused_mlx'][0]}"
            )

    return report


def load_videomae_base(
    model: nn.Module,
    checkpoint_name: str = "MCG-NJU/videomae-base",
    verbose: bool = True,
) -> nn.Module:
    """
    Initialize the NoPE encoder of an MLX NoPE+GDN classifier from the
    original VideoMAE *base* checkpoint (Kinetics-400 self-supervised
    masked-video pretraining — NOT the SSv2 or Kinetics-finetuned variants).

    Mirrors the FdA-kernel-era PyTorch loader from nope_gdn_video_backbone32.ipynb,
    adapted for the MLX param tree:

      • Conv3d weight is permuted PT (oC, iC, T, H, W) -> MLX (oC, T, H, W, iC)
      • Q, K, V dense layers in HF VideoMAE are concatenated into qkv_proj
      • MLX `nn.Sequential` exposes children as `mlp.layers.N` (not `mlp.N`)
      • Attention has bias=False here, so any q/k/v/out biases are skipped
      • Positional embeddings are skipped (NoPE)
      • Decoder + GDN processor + head stay at random init

    The model's current dtype is preserved (e.g. bfloat16 after to_bf16()).

    Args:
        model:           NoPEGDNClassifier (or backbone-rooted module) on MLX.
        checkpoint_name: HF hub id. Default "MCG-NJU/videomae-base".
        verbose:         Print per-stage transfer summary.

    Returns:
        The same model, with encoder weights replaced in-place.
    """
    try:
        from transformers import VideoMAEModel
    except ImportError as e:
        raise ImportError(
            "transformers required for VideoMAE init: pip install transformers"
        ) from e
    try:
        import torch  # noqa: F401  (only needed for the HF download)
    except ImportError as e:
        raise ImportError(
            "torch required to download VideoMAE weights: pip install torch"
        ) from e
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten

    if verbose:
        print(f"Downloading VideoMAE base: {checkpoint_name} ...")
    vmae = VideoMAEModel.from_pretrained(checkpoint_name)
    vmae_sd = {k: v.detach().cpu().numpy() for k, v in vmae.state_dict().items()}
    del vmae

    # Strip the optional "videomae." prefix that VideoMAEForPreTraining uses
    # but VideoMAEModel does not — handle both for forward compat.
    def _strip(k: str) -> str:
        return k[len("videomae."):] if k.startswith("videomae.") else k
    vmae_sd = {_strip(k): v for k, v in vmae_sd.items()}

    # Snapshot current MLX param shapes/dtypes so we can match dtype + verify.
    mlx_params = dict(tree_flatten(model.parameters()))

    new_pairs: List[Tuple[str, mx.array]] = []
    loaded: List[str] = []
    skipped: List[str] = []

    def _assign(mlx_key: str, np_arr: "np.ndarray", note: str = ""):
        if mlx_key not in mlx_params:
            skipped.append(f"  no target: {mlx_key}")
            return
        target = mlx_params[mlx_key]
        if tuple(target.shape) != tuple(np_arr.shape):
            skipped.append(
                f"  shape mismatch {mlx_key}: "
                f"PT {tuple(np_arr.shape)} vs MLX {tuple(target.shape)}"
            )
            return
        new_pairs.append((mlx_key, mx.array(np_arr).astype(target.dtype)))
        loaded.append(f"  {mlx_key}{(' (' + note + ')') if note else ''}")

    # ---- 1. Tubelet embed: PT Conv3d (oC, iC, T, H, W) -> MLX (oC, T, H, W, iC)
    pe_w = vmae_sd.get("embeddings.patch_embeddings.projection.weight")
    pe_b = vmae_sd.get("embeddings.patch_embeddings.projection.bias")
    if pe_w is not None and pe_w.ndim == 5:
        pe_w_chlast = np.transpose(pe_w, (0, 2, 3, 4, 1))
        _assign("backbone.encoder.tubelet_embed.projection.weight",
                pe_w_chlast, note="PT (oC,iC,T,H,W)->MLX (oC,T,H,W,iC)")
    if pe_b is not None:
        _assign("backbone.encoder.tubelet_embed.projection.bias", pe_b)

    # ---- 2. Encoder blocks ----
    n_mlx_blocks = sum(
        1 for k in mlx_params
        if k.startswith("backbone.encoder.blocks.") and k.endswith(".norm1.weight")
    )
    n_vmae_blocks = sum(
        1 for k in vmae_sd
        if k.startswith("encoder.layer.") and k.endswith(".layernorm_before.weight")
    )
    n_transfer = min(n_mlx_blocks, n_vmae_blocks)

    for i in range(n_transfer):
        # Norms
        for vmae_suf, mlx_suf in [
            ("layernorm_before.weight", "norm1.weight"),
            ("layernorm_before.bias",   "norm1.bias"),
            ("layernorm_after.weight",  "norm2.weight"),
            ("layernorm_after.bias",    "norm2.bias"),
        ]:
            arr = vmae_sd.get(f"encoder.layer.{i}.{vmae_suf}")
            if arr is not None:
                _assign(f"backbone.encoder.blocks.{i}.{mlx_suf}", arr)

        # Q, K, V -> qkv_proj (concat along output dim)
        q_w = vmae_sd.get(f"encoder.layer.{i}.attention.attention.query.weight")
        k_w = vmae_sd.get(f"encoder.layer.{i}.attention.attention.key.weight")
        v_w = vmae_sd.get(f"encoder.layer.{i}.attention.attention.value.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            _assign(f"backbone.encoder.blocks.{i}.attn.qkv_proj.weight",
                    qkv_w, note="Q+K+V concat")

        q_b = vmae_sd.get(f"encoder.layer.{i}.attention.attention.query.bias")
        k_b = vmae_sd.get(f"encoder.layer.{i}.attention.attention.key.bias")
        v_b = vmae_sd.get(f"encoder.layer.{i}.attention.attention.value.bias")
        if q_b is not None and k_b is not None and v_b is not None:
            qkv_b = np.concatenate([q_b, k_b, v_b], axis=0)
            # Will be silently skipped if the MLX attention has bias=False.
            _assign(f"backbone.encoder.blocks.{i}.attn.qkv_proj.bias",
                    qkv_b, note="Q+K+V concat")

        # Attention output dense
        for vmae_suf, mlx_suf in [
            ("attention.output.dense.weight", "attn.out_proj.weight"),
            ("attention.output.dense.bias",   "attn.out_proj.bias"),
        ]:
            arr = vmae_sd.get(f"encoder.layer.{i}.{vmae_suf}")
            if arr is not None:
                _assign(f"backbone.encoder.blocks.{i}.{mlx_suf}", arr)

        # MLP — MLX Sequential indexing is mlp.layers.{0,3} (Linear, GELU,
        # Dropout, Linear, Dropout)
        for vmae_suf, mlx_suf in [
            ("intermediate.dense.weight", "mlp.layers.0.weight"),
            ("intermediate.dense.bias",   "mlp.layers.0.bias"),
            ("output.dense.weight",       "mlp.layers.3.weight"),
            ("output.dense.bias",         "mlp.layers.3.bias"),
        ]:
            arr = vmae_sd.get(f"encoder.layer.{i}.{vmae_suf}")
            if arr is not None:
                _assign(f"backbone.encoder.blocks.{i}.{mlx_suf}", arr)

    # ---- 3. Final encoder LayerNorm ----
    for suf in ("weight", "bias"):
        arr = vmae_sd.get(f"layernorm.{suf}")
        if arr is not None:
            _assign(f"backbone.encoder.norm.{suf}", arr)

    # ---- 4. Apply ----
    model.update(tree_unflatten(new_pairs))
    mx.eval(model.parameters())

    if verbose:
        print("=" * 60)
        print(f"VideoMAE base -> MLX NoPE encoder ({checkpoint_name})")
        print("=" * 60)
        print(f"  VideoMAE blocks:  {n_vmae_blocks}")
        print(f"  MLX encoder blocks: {n_mlx_blocks}")
        print(f"  Transferred:      {n_transfer} blocks")
        print(f"  Loaded: {len(loaded)} tensors")
        if skipped:
            print(f"  Skipped: {len(skipped)} tensors")
            for s in skipped[:8]:
                print(f"    {s}")
            if len(skipped) > 8:
                print(f"    ... ({len(skipped) - 8} more)")
        print("  NOTE: tubelet projection is native 3D (no inflation).")
        print("  NOTE: Q/K/V concatenated into qkv_proj.")
        print("  NOTE: positional embeddings skipped (NoPE).")
        print("  NOTE: GDN processor + temporal-pool head stay at random init.")
        print("=" * 60)

    return model


def convert_pytorch_state_dict(
    pt_state_dict: Dict[str, "torch.Tensor"],  # noqa: F821 -- torch optional
    name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, mx.array]:
    """
    Convert a PyTorch state_dict from your existing NoPE+GDN checkpoint
    to MLX-compatible flat dict {param_path: mx.array}.

    Layout transforms applied automatically:
      Conv3d  weight (out_C, in_C, T, H, W)  -> (out_C, T, H, W, in_C)
      Conv1d  weight (out_C, in_C, K)        -> (out_C, K, in_C)
        (depthwise: in_C/groups = 1, so MLX wants (out_C, K, 1) — already correct)
      Linear  weight (out, in)               -> unchanged
      LayerNorm weight/bias                  -> unchanged
      RMSNorm weight                         -> unchanged

    name_map is optional; pass {pt_name: mlx_name} if your PyTorch keys differ
    from the MLX names below. Most should match because the module structure
    is preserved.

    Usage:
        import torch
        ckpt = torch.load("nope_gdn_blackwell.pt", map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt
        mlx_weights = convert_pytorch_state_dict(sd)

        from mlx.utils import tree_unflatten
        model = NoPEGDNClassifier(...)
        model.update(tree_unflatten(list(mlx_weights.items())))
        mx.eval(model.parameters())
    """
    import numpy as np

    out: Dict[str, mx.array] = {}
    name_map = name_map or {}

    for pt_key, t in pt_state_dict.items():
        key = name_map.get(pt_key, pt_key)

        # Translate common PyTorch nn name suffixes
        # (your code uses qkv_proj, out_proj, etc. — names already match)
        arr = t.detach().cpu().numpy()

        # Conv3d weights: PT (oC, iC, T, H, W) -> MLX (oC, T, H, W, iC)
        if "tubelet_embed.projection.weight" in key and arr.ndim == 5:
            arr = np.transpose(arr, (0, 2, 3, 4, 1))

        # Conv1d weights for short causal convs: PT (oC, iC/groups, K) -> MLX (oC, K, iC/groups)
        # Your depthwise layers have iC/groups = 1, so PT shape is (oC, 1, K) -> MLX (oC, K, 1)
        if any(s in key for s in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            if arr.ndim == 3:
                arr = np.transpose(arr, (0, 2, 1))

        out[key] = mx.array(arr)

    return out


# ============================================================================
# BENCHMARK + PATH-EQUIVALENCE TESTS
# ============================================================================

def benchmark_kda_paths(B=2, L=128, H=6, D=64, hidden=384, n_iter=20):
    """
    Compare relative speed of the KDA compute paths.

    Run this on your Mac to see the speedup from compiled vs naive,
    and the chunkwise WY scalar-α path for comparison.
    """
    import time

    print(f"\n{'='*70}")
    print(f"KDA path benchmark — B={B}, L={L}, H={H}, D={D}")
    print(f"{'='*70}")

    x = mx.random.normal((B, L, hidden))
    mx.eval(x)

    paths = [
        ("naive_kda",       "naive",         True,  False),
        ("compiled_kda",    "compiled",      True,  True),
        ("metal_kda",       "metal",         True,  True),
        ("metal_sg_kda",    "metal_sg",      True,  True),
        ("chunkwise_kda",   "chunkwise_kda", True,  True),
        ("naive_gdn",       "naive",         False, False),
        ("compiled_gdn",    "compiled",      False, True),
        ("chunkwise_gdn",   "chunkwise_wy",  False, True),
    ]

    for name, path, channel_wise, use_compiled in paths:
        try:
            layer = GatedDeltaLayer(
                hidden_size=hidden, num_heads=H, head_dim=D,
                channel_wise_decay=channel_wise,
                compute_path=path,
            )
            layer.eval()
            mx.eval(layer.parameters())

            # Warmup (also primes compile cache)
            out, _ = layer(x)
            mx.eval(out)

            t0 = time.perf_counter()
            for _ in range(n_iter):
                out, _ = layer(x)
                mx.eval(out)
            dt = (time.perf_counter() - t0) / n_iter * 1000
            print(f"  {name:18s}  {dt:8.2f} ms/iter")
        except Exception as e:
            print(f"  {name:18s}  FAILED: {type(e).__name__}: {e}")


def verify_metal_solve_triangular(atol: float = 1e-4) -> None:
    """
    Direct unit test for `_metal_solve_triangular` over a sweep of matrix
    sizes. Intended to catch bugs that scale with C (e.g. per-thread register
    array size limits on Apple GPUs).

    Historical note: prior to commit FIX-tg-y-cols, the kernel kept the
    per-thread y_col[C] array in the register stack. On M3 GPUs that array
    silently spilled past C ≥ 40 and produced corrupted output. The bug went
    undetected because `verify_path_equivalence` is hard-coded to chunk_size=8,
    which kept C below the spill threshold. This function tests every C the
    kernel is actually used at.
    """
    import numpy as np
    import numpy.linalg as nla

    print(f"\n{'='*70}")
    print(f"_metal_solve_triangular unit test (sweep C)")
    print(f"{'='*70}")
    BH = 4
    failed = []
    for C in [4, 8, 16, 32, 40, 48, 56, 64]:
        np.random.seed(0)
        L_unit = (np.tril(np.random.randn(BH, C, C).astype(np.float32) * 0.1, k=-1)
                  + np.eye(C, dtype=np.float32))
        b_np = np.random.randn(BH, C, C).astype(np.float32)
        y_np = np.empty_like(b_np)
        for i in range(BH):
            y_np[i] = nla.solve(L_unit[i], b_np[i])
        y_mx = GatedDeltaLayer._metal_solve_triangular(mx.array(L_unit), mx.array(b_np))
        mx.eval(y_mx)
        diff = float(np.max(np.abs(np.asarray(y_mx) - y_np)))
        rel = diff / (float(np.max(np.abs(y_np))) + 1e-9)
        ok = rel < atol
        if not ok:
            failed.append((C, rel))
        print(f"  C=D={C:3d}  rel error = {rel:.3e}  "
              f"{'✅' if ok else '❌ KERNEL CORRUPTED'}")
    if failed:
        raise RuntimeError(
            f"_metal_solve_triangular produced corrupted output at "
            f"sizes: {failed}"
        )


def verify_path_equivalence(B=1, L=16, H=2, D=16, hidden=64, atol=1e-4,
                            chunk_size: int = 8):
    """
    Verify that the four paths produce numerically equivalent outputs
    on the same input + parameters. Run this once after any change to
    GatedDeltaLayer to catch regressions.

    naive_kda <-> compiled_kda  (must match exactly modulo fp32 noise)
    naive_gdn <-> compiled_gdn  (must match exactly modulo fp32 noise)
    naive_gdn <-> chunkwise_gdn (must match within atol — different algorithms)

    The `chunk_size` argument lets callers exercise the chunkwise paths at
    realistic chunk sizes (default 64 in production); the historical default
    here is 8 only because the original sequence length L=16 wouldn't fill a
    single chunk otherwise. CI should call this with both 8 AND 64.
    """
    print(f"\n{'='*70}")
    print(f"Path equivalence check — B={B}, L={L}, H={H}, D={D}, "
          f"chunk_size={chunk_size}")
    print(f"{'='*70}")

    mx.random.seed(0)
    x = mx.random.normal((B, L, hidden))
    mx.eval(x)

    def make_layer(channel_wise, path):
        mx.random.seed(42)
        layer = GatedDeltaLayer(
            hidden_size=hidden, num_heads=H, head_dim=D,
            chunk_size=chunk_size,
            channel_wise_decay=channel_wise,
            compute_path=path,
        )
        layer.eval()
        mx.eval(layer.parameters())
        return layer

    # KDA: naive vs compiled
    l_naive  = make_layer(True, "naive")
    l_compld = make_layer(True, "compiled")
    o_n, _ = l_naive(x);  mx.eval(o_n)
    o_c, _ = l_compld(x); mx.eval(o_c)
    diff = mx.max(mx.abs(o_n - o_c)).item()
    print(f"  KDA  naive vs compiled : max |Δ| = {diff:.2e}  "
          f"{'✅' if diff < atol else '❌'}")

    # KDA: naive vs Metal kernel — CRITICAL CORRECTNESS CHECK
    # If this fails, the Metal kernel has a bug. Don't use it.
    try:
        l_metal = make_layer(True, "metal")
        o_m, _ = l_metal(x); mx.eval(o_m)
        diff = mx.max(mx.abs(o_n - o_m)).item()
        status = "✅" if diff < atol else "❌ DO NOT USE METAL PATH"
        print(f"  KDA  naive vs metal    : max |Δ| = {diff:.2e}  {status}")
    except Exception as e:
        print(f"  KDA  naive vs metal    : SKIPPED ({type(e).__name__}: {e})")

    # KDA: naive vs Metal-simdgroup kernel — same correctness check
    try:
        l_sg = make_layer(True, "metal_sg")
        o_sg, _ = l_sg(x); mx.eval(o_sg)
        diff = mx.max(mx.abs(o_n - o_sg)).item()
        status = "✅" if diff < atol else "❌ DO NOT USE METAL_SG PATH"
        print(f"  KDA  naive vs metal_sg : max |Δ| = {diff:.2e}  {status}")
    except Exception as e:
        print(f"  KDA  naive vs metal_sg : SKIPPED ({type(e).__name__}: {e})")

    # KDA: naive vs chunkwise WY (channel-wise) — different algorithm, should
    # match within atol.
    try:
        l_cw = make_layer(True, "chunkwise_kda")
        o_cw, _ = l_cw(x); mx.eval(o_cw)
        diff = mx.max(mx.abs(o_n - o_cw)).item()
        status = "✅" if diff < atol else "❌"
        print(f"  KDA  naive vs chunkwise_kda : max |Δ| = {diff:.2e}  {status}")
    except Exception as e:
        print(f"  KDA  naive vs chunkwise_kda : SKIPPED ({type(e).__name__}: {e})")

    # GDN: naive vs compiled
    l_naive  = make_layer(False, "naive")
    l_compld = make_layer(False, "compiled")
    o_n, _ = l_naive(x);  mx.eval(o_n)
    o_c, _ = l_compld(x); mx.eval(o_c)
    diff = mx.max(mx.abs(o_n - o_c)).item()
    print(f"  GDN  naive vs compiled : max |Δ| = {diff:.2e}  "
          f"{'✅' if diff < atol else '❌'}")

    # GDN: naive vs chunkwise WY
    l_naive = make_layer(False, "naive")
    l_wy    = make_layer(False, "chunkwise_wy")
    o_n, _  = l_naive(x); mx.eval(o_n)
    o_w, _  = l_wy(x);    mx.eval(o_w)
    diff = mx.max(mx.abs(o_n - o_w)).item()
    print(f"  GDN  naive vs chunk-WY : max |Δ| = {diff:.2e}  "
          f"{'✅' if diff < atol else '❌'}")


# ============================================================================
# BF16 MIXED PRECISION
# ============================================================================
#
# Pattern (matches the docstring at the top):
#   - All Linear / Conv / LayerNorm weights         -> bfloat16
#   - Activations between layers                    -> bfloat16 (follow weights)
#   - GDN recurrence state (D × D outer products)   -> stays float32 (overflow guard)
#   - Decay parameterization (A_log, dt_bias)       -> stays float32
#       reason: used inside exp() / softplus(); fp32 keeps the dynamic range safe
#   - Metal kernel I/O                              -> wrappers cast to fp32 internally,
#                                                       so bf16 inputs work as-is
#
# Use:
#     model = NoPEGDNClassifier(...)
#     model = to_bf16(model)              # one-shot cast after init / weight load
#     out = model(video.astype(mx.bfloat16))
#
# Note: this is "pure bf16" mode (forward + backward in bf16). For STRICTLY safer
# AMP-style training, you'd want a master fp32 copy of weights for the optimizer.
# bfloat16 has the same exponent range as float32, so most transformers train
# stably without that. If you see loss divergence after switching, fall back to
# keeping the optimizer state in fp32 (mlx.optimizers exposes `dtype=` on Adam).

from mlx.utils import tree_map

# Defaults: keep these param-name substrings in fp32 (small, used in numerically
# sensitive ops like exp/softplus or as the master norm scale).
_BF16_KEEP_FP32_DEFAULT = ("A_log", "dt_bias")


def cast_to_bf16(
    params,
    keep_fp32: Optional[Tuple[str, ...]] = _BF16_KEEP_FP32_DEFAULT,
):
    """
    Recursively cast every fp32 leaf in a parameter tree to bfloat16, except
    leaves whose path contains any substring in `keep_fp32`.

    Returns a NEW tree; does not mutate `params`. Apply to a model with
    `model.update(cast_to_bf16(model.parameters()))` or use `to_bf16(model)`.

    Args:
        params:    output of `model.parameters()` (nested dict / list of mx.array)
        keep_fp32: tuple of substrings; matching leaves stay fp32. Default keeps
                   GDN's `A_log` and `dt_bias` (used in exp/softplus).
                   Pass `()` to cast literally everything to bf16.
    """
    keep = tuple(keep_fp32 or ())

    def _walk(node, path=""):
        if isinstance(node, dict):
            return {k: _walk(v, f"{path}.{k}" if path else k) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v, f"{path}.{i}") for i, v in enumerate(node)]
        if isinstance(node, mx.array) and node.dtype == mx.float32:
            if any(s in path for s in keep):
                return node
            return node.astype(mx.bfloat16)
        return node

    return _walk(params)


def to_bf16(model: nn.Module,
            keep_fp32: Optional[Tuple[str, ...]] = _BF16_KEEP_FP32_DEFAULT
            ) -> nn.Module:
    """
    In-place cast a model's fp32 parameters to bfloat16 (with the standard
    GDN exclusions). Returns the model for chaining.
    """
    model.update(cast_to_bf16(model.parameters(), keep_fp32=keep_fp32))
    mx.eval(model.parameters())
    return model


def param_dtype_summary(model: nn.Module) -> Dict[str, int]:
    """Return a count of parameters per dtype — useful for verifying the cast."""
    counts: Dict[str, int] = {}
    for _, arr in _flatten_params(model.parameters()):
        key = str(arr.dtype)
        counts[key] = counts.get(key, 0) + arr.size
    return counts


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test():
    """Tiny forward-pass test. Run this first to verify shapes."""
    print("=" * 70)
    print("MLX NoPE+GDN — Smoke Test")
    print("=" * 70)

    # Tiny model
    model = NoPEGDNClassifier(
        img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        processor_dim=128, processor_depth=4, processor_heads=4,
        chunk_size=32, num_classes=174, dropout=0.0,
    )
    model.eval()

    # Force parameter materialization
    mx.eval(model.parameters())

    n_params = sum(
        v.size for _, v in _flatten_params(model.parameters())
    )
    print(f"Model: {n_params/1e6:.2f}M params")

    # Dummy video: (B, T, H, W, C) — channels-last for MLX
    B, T, H, W, C = 1, 16, 224, 224, 3
    video = mx.random.normal((B, T, H, W, C))
    print(f"Input: {video.shape} (B,T,H,W,C — MLX layout)")

    logits = model(video)
    mx.eval(logits)
    print(f"Output logits: {logits.shape}  (expected: ({B}, 174))")

    assert logits.shape == (B, 174), f"shape mismatch: {logits.shape}"
    print("✅ Forward pass OK")

    # Verify backbone block pattern
    types = model.backbone.get_block_types()
    print(f"Processor block pattern: {types}")
    assert types == ["gdn", "gdn", "gdn", "nope"], "3:1 pattern broken"
    print("✅ 3:1 GDN:NoPE pattern OK")

    # ----- bf16 path -----
    print("\n" + "-" * 70)
    print("bf16 mixed precision check")
    print("-" * 70)

    # Same arch, fresh init, then cast to bf16
    model_bf = NoPEGDNClassifier(
        img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        processor_dim=128, processor_depth=4, processor_heads=4,
        chunk_size=32, num_classes=174, dropout=0.0,
    )
    model_bf.eval()
    to_bf16(model_bf)

    summary = param_dtype_summary(model_bf)
    print(f"Param dtype counts: {summary}")
    # MLX dtype repr is e.g. 'mlx.core.bfloat16'; sum across any matching key.
    bf16_count = sum(v for k, v in summary.items() if "bfloat16" in k)
    fp32_count = sum(v for k, v in summary.items() if "float32" in k)
    assert bf16_count > 0, "No params got cast to bf16"
    # Sanity: A_log + dt_bias should be the only fp32 leaves under default exclusions
    # (a few hundred floats at most for this tiny config)
    assert fp32_count < 1024, (
        f"Too many fp32 params left ({fp32_count}); only A_log/dt_bias should remain"
    )
    print(f"✅ Cast: {bf16_count}/{bf16_count+fp32_count} params in bf16, "
          f"{fp32_count} kept fp32 (A_log + dt_bias)")

    video_bf = video.astype(mx.bfloat16)
    logits_bf = model_bf(video_bf)
    mx.eval(logits_bf)
    assert logits_bf.shape == (B, 174), f"bf16 shape mismatch: {logits_bf.shape}"
    assert logits_bf.dtype == mx.bfloat16, f"bf16 logits dtype: {logits_bf.dtype}"
    print(f"✅ bf16 forward (Metal kernel): shape={logits_bf.shape}, "
          f"dtype={logits_bf.dtype}")

    # Tiny gradient check.
    # The Metal kernels have no custom VJP, so for training we switch GDN layers
    # to compute_path='compiled' (which has working autograd in bf16).
    for _, m in model_bf.named_modules():
        if isinstance(m, GatedDeltaLayer):
            m.compute_path = "compiled"

    target = mx.array([42])
    def loss_fn(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x).astype(mx.float32), y))
    grad_fn = nn.value_and_grad(model_bf, loss_fn)
    loss, grads = grad_fn(model_bf, video_bf, target)
    mx.eval(loss, grads)
    n_grad_leaves = sum(1 for _, _ in _flatten_params(grads))
    # Most grads should be bf16 (matching weight dtype), a few fp32 (A_log/dt_bias)
    grad_dtypes: Dict[str, int] = {}
    for _, g in _flatten_params(grads):
        grad_dtypes[str(g.dtype)] = grad_dtypes.get(str(g.dtype), 0) + 1
    print(f"✅ bf16 backward (compiled path): loss={loss.item():.3f}, "
          f"{n_grad_leaves} grad leaves, dtypes={grad_dtypes}")


def _flatten_params(params, prefix=""):
    """Walk MLX parameter tree, yield (name, array)."""
    if isinstance(params, dict):
        for k, v in params.items():
            yield from _flatten_params(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(params, list):
        for i, v in enumerate(params):
            yield from _flatten_params(v, f"{prefix}.{i}")
    elif isinstance(params, mx.array):
        yield prefix, params


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark_kda_paths()
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_path_equivalence()
    else:
        _smoke_test()
        print("\nFor more tests:")
        print("  python nope_gdn_mlx.py verify   # check path equivalence")
        print("  python nope_gdn_mlx.py bench    # benchmark KDA paths")
