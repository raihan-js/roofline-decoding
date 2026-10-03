"""Roofline analysis for batch-1 LLM decoding.

Batch-1 decoding is memory-bandwidth bound: every generated token reads every
weight byte once (plus a small KV-cache read). So the ceiling on tokens/s is:

    ceiling = measured_bandwidth_bytes_per_s / weight_bytes_per_token

The bandwidth must be MEASURED on the card, not quoted from the spec sheet.
The weight bytes must be COUNTED from the architecture, not guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

# RTX 3060 12GB (GA106, sm_86): spec is 360 GB/s. We measure, never quote.
SPEC_BANDWIDTH_GBS = 360.0

QWEN_15B = {
    "name": "Qwen2.5-1.5B-Instruct",
    "hidden": 1536,
    "layers": 28,
    "heads": 12,
    "kv_heads": 2,
    "intermediate": 8960,
    "vocab": 151936,
    "tied_embeddings": True,
}

ORCH_FUSION = {
    "name": "ORCH-Fusion",
    "hidden": 1024,
    "layers": 24,
    "heads": 16,
    "kv_heads": 4,
    "intermediate": 2816,
    "vocab": 2103,
    "tied_embeddings": False,
}


@dataclass
class RooflineResult:
    model: str
    dtype: str
    weight_bytes: int
    weight_gb: float
    kv_bytes_per_token: int
    measured_bandwidth_gbs: float
    ceiling_tok_s: float


def bytes_per_param(dtype: str) -> float:
    return {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}[dtype]


def count_decoder_params(cfg: dict, include_norm: bool = True) -> dict[str, int]:
    """Exact parameter count per component class for a GQA LLaMA-style decoder."""
    h, L = cfg["hidden"], cfg["layers"]
    kv_dim = h // cfg["heads"] * cfg["kv_heads"]
    per_layer_attn = h * h + 2 * h * kv_dim + h * h
    per_layer_mlp = 2 * h * cfg["intermediate"] + cfg["intermediate"] * h
    per_layer_norm = 2 * h if include_norm else 0
    decoder = L * (per_layer_attn + per_layer_mlp + per_layer_norm)
    embed = cfg["vocab"] * h
    head = 0 if cfg["tied_embeddings"] else cfg["vocab"] * h
    final_norm = h if include_norm else 0
    return {
        "decoder": decoder,
        "embed": embed,
        "head": head,
        "final_norm": final_norm,
        "total": decoder + embed + head + final_norm,
        "non_embedding": decoder + final_norm,
    }


def weight_bytes_per_token(cfg: dict, dtype: str) -> int:
    """Weight bytes read per decoded token (full model pass, batch 1)."""
    counts = count_decoder_params(cfg)
    return int(counts["total"] * bytes_per_param(dtype))


def kv_bytes_per_token(cfg: dict, dtype: str, seq_len: int = 2048) -> int:
    """KV-cache bytes read per decoded token at a given context length."""
    kv_dim = cfg["hidden"] // cfg["heads"] * cfg["kv_heads"]
    # K and V per layer, full context is re-read each step
    return int(2 * cfg["layers"] * kv_dim * seq_len * bytes_per_param(dtype))


def measure_copy_bandwidth_gbs(n_bytes: int = 1 << 30, iters: int = 20,
                               device: str = "cuda") -> float:
    """Measured device-to-device copy bandwidth (the honest roofline)."""
    torch.cuda.synchronize()
    a = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    b = torch.empty_like(a)
    # Warmup
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # copy_ reads n_bytes and writes n_bytes
    return (2 * n_bytes / 1e9) / dt


def decode_ceiling(cfg: dict, dtype: str, bandwidth_gbs: float,
                   seq_len: int = 2048) -> RooflineResult:
    """Batch-1 decode ceiling from measured bandwidth and counted weights."""
    wb = weight_bytes_per_token(cfg, dtype)
    kvb = kv_bytes_per_token(cfg, dtype, seq_len)
    total = wb + kvb
    return RooflineResult(
        model=cfg["name"],
        dtype=dtype,
        weight_bytes=wb,
        weight_gb=wb / 1e9,
        kv_bytes_per_token=kvb,
        measured_bandwidth_gbs=bandwidth_gbs,
        ceiling_tok_s=bandwidth_gbs * 1e9 / total,
    )


def ceilings_table(bandwidth_gbs: float, seq_len: int = 2048) -> list[dict]:
    """Theoretical batch-1 ceilings for both models x three dtypes."""
    rows = []
    for cfg in (QWEN_15B, ORCH_FUSION):
        for dtype in ("bf16", "int8", "int4"):
            r = decode_ceiling(cfg, dtype, bandwidth_gbs, seq_len)
            rows.append({
                "model": r.model,
                "dtype": dtype,
                "weight_gb": round(r.weight_gb, 3),
                "kv_mb_per_token": round(r.kv_bytes_per_token / 1e6, 2),
                "bandwidth_gbs": round(r.measured_bandwidth_gbs, 1),
                "ceiling_tok_s": round(r.ceiling_tok_s, 1),
            })
    return rows


def achieved_bandwidth_percent(measured_tok_s: float, weight_gb: float,
                               kv_mb_per_token: float,
                               peak_bandwidth_gbs: float) -> dict:
    """What fraction of the roofline does a kernel actually achieve?"""
    bytes_per_token = weight_gb * 1e9 + kv_mb_per_token * 1e6
    achieved_gbs = measured_tok_s * bytes_per_token / 1e9
    return {
        "achieved_gbs": round(achieved_gbs, 1),
        "percent_of_roofline": round(100 * achieved_gbs / peak_bandwidth_gbs, 1),
    }
