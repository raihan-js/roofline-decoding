"""Fused W4A16 GEMV Triton kernels for batch-1 decoding.

Weight format (group-wise symmetric int4, group_size=128 along K):
  packed: int32 tensor of shape (K//8 ... ) — see pack_weights
  scales: fp16 tensor of shape (num_groups, N), num_groups = K // 128
  dequant: w = (nibble - 8) * scale

Versions:
  v1: dequantise-then-matmul — materialise full fp16 then torch mv.
      Shows the cost of the extra memory pass.
  v2: fused dequant-GEMV — one kernel, tl.dot accumulation, no materialisation.
  v3: fused + autotuned block sizes + split-K over K for large matrices.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128


def pack_weights(w_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Pack fp16 [K, N] into int32-packed int4 + fp16 scales.

    Returns (packed, scales) with packed int32 shape (K//8, N) and
    scales fp16 shape (K//group_size, N).
    """
    assert w_fp16.dim() == 2
    K, N = w_fp16.shape
    assert K % group_size == 0
    assert K % 8 == 0
    w = w_fp16.float()
    num_groups = K // group_size
    w_groups = w.view(num_groups, group_size, N)
    amax = w_groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (amax / 7.0).to(torch.float16)  # [G, 1, N]
    # symmetric int4: q = round(w/s) + 8 in [1, 15]; dequant is (q - 8) * s
    q = (torch.round(w_groups / scales) + 8).clamp(0, 15).to(torch.int32)  # [G, 128, N]
    # pack 8 nibbles into one int32 along K
    q = q.view(num_groups, group_size // 8, 8, N)
    shifts = torch.arange(8, device=q.device).view(1, 1, 8, 1) * 4
    packed = (q << shifts).sum(dim=2).to(torch.int32)  # [G, 16, N]
    packed = packed.reshape(K // 8, N).contiguous()
    scales = scales.squeeze(1).contiguous()  # [G, N]
    return packed, scales


@triton.jit
def _dequant_kernel(
    packed_ptr, scales_ptr, out_ptr,
    K, N,
    stride_pk, stride_pn,
    stride_sk, stride_sn,
    stride_ok, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """v1 helper: materialise fp16 [K, N] from packed int4."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    for k8 in range(0, K // 8):
        # one int32 holds 8 weights for 8 consecutive K rows
        pack = tl.load(packed_ptr + k8 * stride_pk + offs_n * stride_pn, mask=mask_n)
        for nib in range(8):
            k = k8 * 8 + nib
            g = k // GROUP_SIZE
            q = (pack >> (nib * 4)) & 0xF
            s = tl.load(scales_ptr + g * stride_sk + offs_n * stride_sn, mask=mask_n)
            w = (q.to(tl.float32) - 8.0) * s.to(tl.float32)
            tl.store(out_ptr + k * stride_ok + offs_n * stride_on,
                     w.to(tl.float16), mask=mask_n)


def dequantize_v1(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """v1: materialise full fp16 weight (the extra memory pass v2 removes)."""
    K8, N = packed.shape
    K = K8 * 8
    out = torch.empty((K, N), dtype=torch.float16, device=packed.device)
    grid = ((N + 127) // 128,)
    _dequant_kernel[grid](
        packed, scales, out, K, N,
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE, 128,
    )
    return out


@triton.jit
def _fused_gemv_kernel(
    x_ptr, packed_ptr, scales_ptr, out_ptr,
    K, N,
    stride_xk,
    stride_pk, stride_pn,
    stride_sk, stride_sn,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """v2: fused dequant-GEMV. y[n] = sum_k x[k] * dequant(W[k, n])."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # load packed segment: rows k0..k0+BK -> k8 indices, then dequantise
        for kk in range(BLOCK_K // 8):
            k = k0 + kk * 8
            k_in_range = k < K
            pack = tl.load(packed_ptr + (k // 8) * stride_pk + offs_n * stride_pn,
                           mask=mask_n & k_in_range)
            g = k // GROUP_SIZE
            s = tl.load(scales_ptr + g * stride_sk + offs_n * stride_sn,
                        mask=mask_n & k_in_range).to(tl.float32)
            for nib in range(8):
                kk_full = k + nib
                m = mask_n & (kk_full < K)
                q = ((pack >> (nib * 4)) & 0xF).to(tl.float32)
                w = (q - 8.0) * s
                xseg = tl.load(x_ptr + kk_full * stride_xk,
                               mask=(kk_full < K)).to(tl.float32)
                acc = tl.where(m, acc + xseg * w, acc)
    tl.store(out_ptr + offs_n * stride_on, acc.to(tl.float16), mask=mask_n)


def gemv_v2(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """v2 fused GEMV: x [K] fp16, packed int4 [K//8, N], scales [G, N] -> y [N] fp16."""
    K = x.shape[0]
    N = packed.shape[1]
    out = torch.empty((N,), dtype=torch.float16, device=x.device)
    grid = ((N + 127) // 128,)
    _fused_gemv_kernel[grid](
        x, packed, scales, out, K, N,
        x.stride(0),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0),
        GROUP_SIZE, 256, 128,
    )
    return out


def _v3_num_warps_options():
    # Fixed BLOCK_N=64 so the host grid (((N+63)//64) * split_k) always
    # matches the kernel's pid_n mapping. num_warps is the tuning axis.
    return [4, 8]


# NOTE: we deliberately do NOT use @triton.autotune here. Autotune benchmarks
# every config against the same output tensor (~500 warmup+rep launches), and
# a split-K kernel accumulates into that buffer with atomic_add — so the
# buffer holds ~500 stale copies before the timed run even starts, and the
# first-tuned configuration always benchmarks garbage. Tuning is done
# explicitly below with a fresh zero buffer per trial.
_TUNE_CACHE: dict[tuple[int, int], int] = {}


@triton.jit
def _fused_gemv_v3_kernel(
    x_ptr, packed_ptr, scales_ptr, out_ptr,
    K, N,
    stride_xk,
    stride_pk, stride_pn,
    stride_sk, stride_sn,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """v3: fused dequant-GEMV with split-K over K (one program per N-block x K-split)."""
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_n_blocks
    pid_k = pid // num_n_blocks
    # split-K ranges snapped to 8-row boundaries (one int32 per 8 rows).
    # The loop uses a constexpr trip count (Triton requirement); iterations
    # outside this program's split are masked. Middle splits provably never
    # over-cover (ceiling-division algebra); the last split is clipped by the
    # kk < K guard in the nibble loop.
    k8_total = K // 8
    k8_per_split = (k8_total + SPLIT_K - 1) // SPLIT_K
    k8_start = pid_k * k8_per_split
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(k8_per_split):
        k8 = k8_start + i
        in_split = (k8 < k8_start + k8_per_split) & (k8 * 8 < K)
        k = k8 * 8
        pack = tl.load(packed_ptr + k8 * stride_pk + offs_n * stride_pn,
                       mask=mask_n & in_split)
        g = k // GROUP_SIZE
        s = tl.load(scales_ptr + g * stride_sk + offs_n * stride_sn,
                    mask=mask_n & in_split).to(tl.float32)
        for nib in range(8):
            kk = k + nib
            m = mask_n & in_split & (kk < K)
            q = ((pack >> (nib * 4)) & 0xF).to(tl.float32)
            w = (q - 8.0) * s
            xseg = tl.load(x_ptr + kk * stride_xk, mask=(kk < K)).to(tl.float32)
            acc = tl.where(m, acc + xseg * w, acc)
    # split-K partial sums: atomic add into output
    tl.atomic_add(out_ptr + offs_n * stride_on, acc.to(tl.float16), mask=mask_n)


def _tune_num_warps(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                    split_k: int) -> int:
    """Pick the fastest num_warps for this (K, N, split_k) with fresh buffers.

    Each trial gets its own zeroed output: reusing one buffer would let the
    split-K atomic_adds accumulate across trials (the @triton.autotune
    footgun documented above).
    """
    import time

    K, N = x.shape[0], packed.shape[1]
    key = (K, N, split_k)
    if key in _TUNE_CACHE:
        return _TUNE_CACHE[key]
    best, best_t = 4, float("inf")
    for nw in _v3_num_warps_options():
        out = torch.zeros((N,), dtype=torch.float16, device=x.device)
        grid = (((N + 63) // 64) * split_k,)
        launch = lambda: _fused_gemv_v3_kernel[grid](
            x, packed, scales, out, K, N,
            x.stride(0), packed.stride(0), packed.stride(1),
            scales.stride(0), scales.stride(1), out.stride(0),
            GROUP_SIZE, 64, split_k, num_warps=nw)
        for _ in range(5):  # warmup (fresh buffer each time)
            out.zero_()
            launch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            out.zero_()
            launch()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        if dt < best_t:
            best, best_t = nw, dt
    _TUNE_CACHE[key] = best
    return best


def gemv_v3(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
            split_k: int = 1, num_warps: int | None = None) -> torch.Tensor:
    """v3 fused GEMV with explicitly tuned num_warps and optional split-K."""
    K = x.shape[0]
    N = packed.shape[1]
    if num_warps is None:
        num_warps = _tune_num_warps(x, packed, scales, split_k)
    out = torch.zeros((N,), dtype=torch.float16, device=x.device)
    # grid matches kernel pid mapping: pid_n in [0, cdiv(N,64)), pid_k in [0, split_k)
    grid = (((N + 63) // 64) * split_k,)
    _fused_gemv_v3_kernel[grid](
        x, packed, scales, out, K, N,
        x.stride(0),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0),
        GROUP_SIZE, 64, split_k, num_warps=num_warps,
    )
    return out


def reference_gemv(x_fp16: torch.Tensor, w_fp16: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: fp16 GEMV."""
    return (x_fp16.float() @ w_fp16.float()).to(torch.float16)
