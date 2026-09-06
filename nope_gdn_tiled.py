"""Value-tiled SIMD Metal kernels for NOPE-GDN's FP32 chunk recurrence.

Every threadgroup retains the entire key dimension but owns at most 32 value
channels. Splitting value channels is exact: each column of the triangular
solve, output channel, and row of the recurrent state is independent. The
96-by-96 state therefore remains intact while a 64-token, 96-channel chunk
uses 28 KiB of threadgroup memory instead of 72 KiB.
"""

from functools import lru_cache

import mlx.core as mx


_THREADGROUP_LIMIT_BYTES = 32 * 1024
_SIMDGROUPS = 8


def _tile_config(chunk_size: int, head_dim: int):
    if not (8 <= chunk_size <= 64 and chunk_size % 8 == 0):
        return None
    if not (8 <= head_dim <= 128 and head_dim % 8 == 0):
        return None
    for width in (32, 16, 8):
        width = min(width, head_dim)
        slot1 = max(head_dim * width, chunk_size * chunk_size, chunk_size * width)
        slot2 = max(chunk_size * width, head_dim * width)
        if (slot1 + slot2) * 4 <= _THREADGROUP_LIMIT_BYTES:
            return width, slot1, slot2
    return None


def supports_tiled_chunk(chunk_size: int, head_dim: int) -> bool:
    """Whether the SIMD tile shapes and 32 KiB memory budget are supported."""
    return _tile_config(chunk_size, head_dim) is not None


def tiled_chunk_memory_bytes(chunk_size: int, head_dim: int) -> int:
    """Return the actual static threadgroup allocation for a supported shape."""
    config = _tile_config(chunk_size, head_dim)
    if config is None:
        raise ValueError(f"Unsupported tiled chunk shape C={chunk_size}, D={head_dim}")
    _, slot1, slot2 = config
    return (slot1 + slot2) * 4


_CHUNK_SOLVE_FINALIZE_SOURCE = r"""
    // R is the value tile width; D remains the complete key dimension.
    const uint bh = threadgroup_position_in_grid.x;
    const uint value_base = threadgroup_position_in_grid.y * R;
    const uint tid = thread_index_in_threadgroup;
    const uint nth = threads_per_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = simdgroups_per_threadgroup;

    // slot1: state^T[D,R] -> IA[C,C] -> state^T[D,R] -> U^T[R,C]
    // slot2: rhs[C,R] -> U[C,R] -> state-update scratch[R,D]
    threadgroup float slot1[SLOT1];
    threadgroup float slot2[SLOT2];

    const uint state_base = bh * D * D;
    const uint chunk_base = bh * C * D;
    const uint square_base = bh * C * C;

    // Load the owned state rows transposed. Pad the last value tile with
    // zero when D is not a multiple of R (D is still a multiple of 8).
    for (uint idx = tid; idx < D * R; idx += nth) {
        const uint key = idx / R;
        const uint value = value_base + idx % R;
        slot1[idx] = value < D ? state_in[state_base + value * D + key] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tmp = K_proj @ state^T[:, value_tile]. Stream K_proj from device
    // memory, so slot2 can receive each result as soon as it is computed.
    // Strided tile loops cover every tile even when TOTAL < NSG.
    for (uint tile = sg; tile < (C / 8) * (R / 8); tile += nsg) {
        const uint row = (tile / (R / 8)) * 8;
        const uint col = (tile % (R / 8)) * 8;
        simdgroup_matrix<float, 8, 8> acc(0.0f);
        for (uint key = 0; key < D; key += 8) {
            simdgroup_matrix<float, 8, 8> a, b;
            simdgroup_load(a, K_proj_in + chunk_base + row * D + key, D);
            simdgroup_load(b, slot1 + key * R + col, R);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, slot2 + row * R + col, R);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rhs = beta * V - beta * tmp. The padded columns remain zero.
    for (uint idx = tid; idx < C * R; idx += nth) {
        const uint row = idx / R;
        const uint value = value_base + idx % R;
        slot2[idx] = value < D
            ? rhs_pre_state_in[chunk_base + row * D + value]
                - B_c_in[bh * C + row] * slot2[idx]
            : 0.0f;
    }
    for (uint idx = tid; idx < C * C; idx += nth) {
        slot1[idx] = IA_in[square_base + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Unit-lower-triangular solve. Each thread owns one complete column;
    // there are no cross-thread dependencies inside forward substitution.
    if (tid < R) {
        for (uint row = 0; row < C; ++row) {
            float value = slot2[row * R + tid];
            for (uint k = 0; k < row; ++k) {
                value -= slot1[row * C + k] * slot2[k * R + tid];
            }
            slot2[row * R + tid] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < C * R; idx += nth) {
        const uint value = value_base + idx % R;
        if (value < D) {
            U_out[chunk_base + (idx / R) * D + value] = slot2[idx];
        }
    }

    // Restore the owned state rows for the output projection.
    for (uint idx = tid; idx < D * R; idx += nth) {
        const uint key = idx / R;
        const uint value = value_base + idx % R;
        slot1[idx] = value < D ? state_in[state_base + value * D + key] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O[:, value_tile] = Q_prime @ state^T + M_oq_T @ U.
    for (uint tile = sg; tile < (C / 8) * (R / 8); tile += nsg) {
        const uint row = (tile / (R / 8)) * 8;
        const uint col = (tile % (R / 8)) * 8;
        // Every real value tile has eight columns because D % 8 == 0.
        if (value_base + col < D) {
            simdgroup_matrix<float, 8, 8> acc(0.0f);
            for (uint key = 0; key < D; key += 8) {
                simdgroup_matrix<float, 8, 8> a, b;
                simdgroup_load(a, Q_prime_in + chunk_base + row * D + key, D);
                simdgroup_load(b, slot1 + key * R + col, R);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            for (uint k = 0; k < C; k += 8) {
                simdgroup_matrix<float, 8, 8> a, b;
                simdgroup_load(a, M_oq_T_in + square_base + row * C + k, C);
                simdgroup_load(b, slot2 + k * R + col, R);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc,
                O_out + chunk_base + row * D + value_base + col, D);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transpose the owned U columns before reusing slot2 for U^T K_back.
    for (uint idx = tid; idx < R * C; idx += nth) {
        const uint value = idx / C;
        const uint row = idx % C;
        slot1[idx] = slot2[row * R + value];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Update only the owned state rows, keeping all D key columns.
    for (uint tile = sg; tile < (R / 8) * (D / 8); tile += nsg) {
        const uint row = (tile / (D / 8)) * 8;
        const uint col = (tile % (D / 8)) * 8;
        simdgroup_matrix<float, 8, 8> acc(0.0f);
        for (uint k = 0; k < C; k += 8) {
            simdgroup_matrix<float, 8, 8> a, b;
            simdgroup_load(a, slot1 + row * C + k, C);
            simdgroup_load(b, K_back_in + chunk_base + k * D + col, D);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, slot2 + row * D + col, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < R * D; idx += nth) {
        const uint value = value_base + idx / D;
        const uint key = idx % D;
        if (value < D) {
            const uint offset = state_base + value * D + key;
            state_new_out[offset] = exp_gamma_last_in[bh * D + key]
                * state_in[offset] + slot2[idx];
        }
    }
"""


@lru_cache(maxsize=None)
def _chunk_kernel(chunk_size: int, head_dim: int, width: int, simdgroups: int):
    return mx.fast.metal_kernel(
        name=f"nope_chunk_tiled_c{chunk_size}_d{head_dim}_r{width}_n{simdgroups}",
        input_names=[
            "IA_in", "rhs_pre_state_in", "B_c_in", "K_proj_in",
            "Q_prime_in", "M_oq_T_in", "K_back_in",
            "exp_gamma_last_in", "state_in",
        ],
        output_names=["O_out", "state_new_out", "U_out"],
        source=_CHUNK_SOLVE_FINALIZE_SOURCE,
        header=(
            "#include <metal_stdlib>\n"
            "#include <metal_simdgroup_matrix>\n"
            "using namespace metal;"
        ),
    )


def chunk_solve_finalize_tiled(
    IA: mx.array,
    rhs_pre_state: mx.array,
    B_c: mx.array,
    K_proj: mx.array,
    Q_prime: mx.array,
    M_oq_T_masked: mx.array,
    K_back: mx.array,
    exp_gamma_last: mx.array,
    state: mx.array,
):
    """Return ``(O, state_new, U)`` as FP32 using one tiled Metal launch.

    Inputs follow the original fused kernel: ``K_proj`` has shape (BH,C,D),
    ``state`` is (BH,D,D), and ``IA`` is unit lower triangular (BH,C,C).
    C must be an 8-multiple through 64; D an 8-multiple through 128. A final
    partial value tile is zero padded internally and all writes are masked.
    """
    bh, chunk_size, head_dim = K_proj.shape
    config = _tile_config(chunk_size, head_dim)
    if config is None:
        raise ValueError(
            "chunk_solve_finalize_tiled requires C and D to be multiples of 8 "
            f"with 8 <= C <= 64 and 8 <= D <= 128; got C={chunk_size}, D={head_dim}"
        )
    width, slot1, slot2 = config
    kernel = _chunk_kernel(chunk_size, head_dim, width, _SIMDGROUPS)
    nth = _SIMDGROUPS * 32
    inputs = [
        IA, rhs_pre_state, B_c, K_proj, Q_prime, M_oq_T_masked,
        K_back, exp_gamma_last, state,
    ]
    outputs = kernel(
        inputs=[value.astype(mx.float32) for value in inputs],
        template=[
            ("C", chunk_size), ("D", head_dim), ("R", width),
            ("SLOT1", slot1), ("SLOT2", slot2),
        ],
        grid=(bh * nth, (head_dim + width - 1) // width, 1),
        threadgroup=(nth, 1, 1),
        output_shapes=[
            (bh, chunk_size, head_dim),
            (bh, head_dim, head_dim),
            (bh, chunk_size, head_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return outputs[0], outputs[1], outputs[2]
