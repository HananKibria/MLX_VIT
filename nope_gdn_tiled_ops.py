"""FP32 KDA solve and gradient kernels with bounded Metal threadgroup memory.

Tiles split independent RHS/feature columns, retaining the full recurrence.
The triangular matrix still has to fit; this is not an arbitrary chunk-size
extension. All sizes below include statically allocated threadgroup arrays.
"""

from functools import lru_cache

import mlx.core as mx


TG_FLOATS = 32 * 1024 // 4
FEATURE_TILE = 32


def triangular_tile_width(chunk_size: int, head_dim: int) -> int:
    if chunk_size <= 0 or head_dim <= 0:
        raise ValueError("chunk_size and head_dim must be positive")
    available = (TG_FLOATS - chunk_size * chunk_size) // chunk_size
    if available < 1:
        raise ValueError(
            f"chunk_size={chunk_size} leaves no threadgroup memory for the "
            "triangular solve; lower chunk_size (64 is supported)."
        )
    # Pad the last tile rather than changing full-array strides.
    return min(32, available, head_dim)


def validate_chunk_memory(chunk_size: int, head_dim: int) -> None:
    """Validate both the solve and custom backward, before model construction."""
    triangular_tile_width(chunk_size, head_dim)
    if 2 * chunk_size * FEATURE_TILE > TG_FLOATS:
        raise ValueError(
            f"chunk_size={chunk_size} exceeds the tiled gradient kernel's "
            "32 KiB budget; lower chunk_size."
        )


_TRIANGULAR_SOURCE = r"""
    const uint bh = threadgroup_position_in_grid.y;
    const uint col = threadgroup_position_in_grid.x * W
                     + thread_position_in_threadgroup.x;
    const uint lane = thread_position_in_threadgroup.x;
    threadgroup float A[C * C];
    threadgroup float Y[C * W];
    for (uint i = lane; i < C * C; i += W) {
        A[i] = A_in[bh * C * C + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col < D) {
        for (uint step = 0; step < C; ++step) {
            const uint i = UPPER ? C - 1 - step : step;
            float value = b_in[(bh * C + i) * D + col];
            for (uint previous = 0; previous < step; ++previous) {
                const uint k = UPPER ? C - 1 - previous : previous;
                value -= A[i * C + k] * Y[k * W + lane];
            }
            Y[i * W + lane] = value;
            y_out[(bh * C + i) * D + col] = value;
        }
    }
"""


@lru_cache(None)
def _triangular_kernel():
    return mx.fast.metal_kernel(
        name="nope_tiled_triangular",
        input_names=["A_in", "b_in"],
        output_names=["y_out"],
        source=_TRIANGULAR_SOURCE,
    )


def solve_triangular_tiled(matrix, rhs, *, upper=False):
    """Solve a unit triangular system, tiling only its independent RHS columns."""
    bh, c, d = rhs.shape
    if matrix.shape != (bh, c, c):
        raise ValueError("triangular matrix and RHS batch/chunk shapes must agree")
    width = triangular_tile_width(c, d)
    return _triangular_kernel()(
        inputs=[matrix.astype(mx.float32), rhs.astype(mx.float32)],
        template=[("C", c), ("D", d), ("W", width), ("UPPER", upper)],
        grid=(((d + width - 1) // width) * width, bh, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(bh, c, d)],
        output_dtypes=[mx.float32],
    )[0]


_M_BACKWARD_SOURCE = r"""
    // A lane owns one feature; four SIMD groups traverse independent rows.
    // Cache just W features across all C tokens, then reuse them for every s.
    const uint bh = threadgroup_position_in_grid.z;
    const uint feature_start = threadgroup_position_in_grid.x * W;
    const uint lane = thread_position_in_threadgroup.x;
    const uint row_lane = thread_position_in_threadgroup.y;
    const uint tid = row_lane * W + lane;
    const uint d = feature_start + lane;
    threadgroup float K[C * W];
    threadgroup float G[C * W];
    for (uint i = tid; i < C * W; i += W * ROWS) {
        const uint feature = feature_start + i % W;
        const uint token = i / W;
        const uint pos = (bh * C + token) * D + feature;
        K[i] = feature < D ? K_in[pos] : 0.0f;
        G[i] = feature < D ? gamma_in[pos] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < D) {
        for (uint s = row_lane; s < C; s += ROWS) {
            const float ks = K[s * W + lane];
            const float gs = G[s * W + lane];
            const float qs = Q_in[(bh * C + s) * D + d];
            float dk = 0.0f;
            float dq = 0.0f;
            float dg = 0.0f;
            for (uint t = 0; t < C; ++t) {
                const float kt = K[t * W + lane];
                const float gt = G[t * W + lane];
                const float qt = Q_in[(bh * C + t) * D + d];
                const float mst = dM_in[(bh * C + s) * C + t];
                const float mts = dM_in[(bh * C + t) * C + s];
                const float ost = dMoq_in[(bh * C + s) * C + t];
                const float ots = dMoq_in[(bh * C + t) * C + s];
                const float delta = gt - gs;
                const float est = delta < 0.0f ? metal::exp(delta) : 1.0f;
                const float ets = delta > 0.0f ? metal::exp(-delta) : 1.0f;
                dk += mst * kt * est + mts * kt * ets + ost * qt * est;
                dq += ots * kt * ets;
                const float ast = delta < 0.0f ? est : 0.0f;
                const float ats = delta > 0.0f ? ets : 0.0f;
                dg += -mst * ks * kt * ast + mts * kt * ks * ats
                      -ost * ks * qt * ast + ots * kt * qs * ats;
            }
            const uint out = (bh * C + s) * D + d;
            dK_out[out] = dk;
            dQ_out[out] = dq;
            dG_out[out] = dg;
        }
    }
"""


@lru_cache(None)
def _m_backward_kernel():
    return mx.fast.metal_kernel(
        name="nope_tiled_m_backward",
        input_names=["K_in", "Q_in", "gamma_in", "dM_in", "dMoq_in"],
        output_names=["dK_out", "dQ_out", "dG_out"],
        source=_M_BACKWARD_SOURCE,
    )


def m_moq_backward_tiled(k, q, gamma, dm, dmoq):
    """Compute dK, dQ and dgamma with 2*C*32 FP32 values of shared memory."""
    bh, c, d = k.shape
    if 2 * c * FEATURE_TILE > TG_FLOATS:
        raise ValueError("tiled M backward exceeds 32 KiB; lower chunk_size")
    width, rows = FEATURE_TILE, 4
    return tuple(_m_backward_kernel()(
        inputs=[x.astype(mx.float32) for x in (k, q, gamma, dm, dmoq)],
        template=[("C", c), ("D", d), ("W", width), ("ROWS", rows)],
        grid=(((d + width - 1) // width) * width, rows, bh),
        threadgroup=(width, rows, 1),
        output_shapes=[(bh, c, d)] * 3,
        output_dtypes=[mx.float32] * 3,
    ))
