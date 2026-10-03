#!/usr/bin/env python3
"""Milestone 2: benchmark v1/v2/v3 kernels vs PyTorch reference on real shapes.

Reports per-shape latency, achieved DRAM bandwidth, and % of measured roofline.
Usage:
  PYTHONPATH=src python scripts/bench_kernels.py --bandwidth-gbs 300
"""
import argparse
import json
import time
from pathlib import Path

import torch

from roofline.kernels import dequantize_v1, gemv_v2, gemv_v3, pack_weights
from roofline.roofline import achieved_bandwidth_percent

SHAPES = [
    ("qwen-q", 1536, 1536),
    ("qwen-mlp-gate", 1536, 8960),
    ("qwen-mlp-down", 8960, 1536),
    ("orch-attn", 1024, 1024),
    ("orch-mlp", 1024, 2816),
]


def bench(fn, iters: int = 100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def dram_bytes(K: int, N: int, version: str) -> int:
    """DRAM bytes read per GEMV call (weights + input vector)."""
    packed = (K // 8) * N
    scales = (K // 128) * N * 2
    x = K * 2
    out = N * 2
    if version == "v1":
        # dequantise pass reads packed+scales, writes fp16; matmul reads fp16
        return packed + scales + 2 * K * N + x + out
    return packed + scales + x + out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-gbs", type=float, required=True)
    ap.add_argument("--out", default="data/results/kernels.json")
    args = ap.parse_args()

    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for name, K, N in SHAPES:
        w = torch.randn(K, N, generator=g, device="cuda", dtype=torch.float16)
        w = (w / w.abs().amax(dim=0, keepdim=True).clamp(min=1e-3))
        x = torch.randn(K, generator=g, device="cuda", dtype=torch.float16)
        packed, scales = pack_weights(w)

        t_ref = bench(lambda: x.float() @ w.float())
        t_v1 = bench(lambda: x.float() @ dequantize_v1(packed, scales).float())
        t_v2 = bench(lambda: gemv_v2(x, packed, scales))
        t_v3 = bench(lambda: gemv_v3(x, packed, scales, split_k=1))
        # split-K=2 only helps large N; measure where relevant
        t_v3s2 = bench(lambda: gemv_v3(x, packed, scales, split_k=2)) if N >= 2816 else None

        row = {"shape": name, "K": K, "N": N,
               "ms": {"ref": round(t_ref, 3), "v1": round(t_v1, 3),
                      "v2": round(t_v2, 3), "v3": round(t_v3, 3)}}
        for ver, t in [("v2", t_v2), ("v3", t_v3)]:
            tok_s_per_ms = 1.0 / t  # one GEMV per "token" here; scaled later
            gbs = dram_bytes(K, N, ver) / 1e9 / (t / 1000)
            row[f"{ver}_gbs"] = round(gbs, 1)
            row[f"{ver}_pct"] = round(100 * gbs / args.bandwidth_gbs, 1)
        if t_v3s2 is not None:
            row["ms"]["v3_split2"] = round(t_v3s2, 3)
        rows.append(row)
        print(f"{name:14s} K={K:<5d} N={N:<5d} ref={t_ref:7.3f}ms "
              f"v1={t_v1:7.3f} v2={t_v2:7.3f}({row['v2_pct']:4.1f}%) "
              f"v3={t_v3:7.3f}({row['v3_pct']:4.1f}%)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bandwidth_gbs": args.bandwidth_gbs, "rows": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
