# AGENTS.md — roofline-decoding

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: Roofline-First Decoding

Write a fused 4-bit-weight dequantise-plus-GEMV Triton kernel for batch-1 decoding, profile it, report what fraction of the RTX 3060's memory bandwidth it achieves, and compare against production kernels. A learning kernel, stated as such: the point is understanding, not beating Marlin.

## Why this project

Batch-1 decoding is memory-bandwidth bound: each token reads all weights, so the ceiling on tokens/s is bandwidth divided by weight bytes — worked out on a whiteboard before any code. Most portfolios call a quantisation library. This starts from the roofline, writes the kernel, and explains with profiler evidence how far each approach is from the physical limit and why.

## Stack

Python, Triton, PyTorch, torch.profiler, CUDA graphs, Qwen2.5-1.5B-Instruct, ORCH-Fusion (own checkpoint + model code).

## Compute

RTX 3060 12GB only — consumer-GPU performance is the point. No rented GPU.

## Key design decisions

- **Measured bandwidth, never spec.** The roofline uses a torch copy benchmark on the card.
- **Counted weights, never guessed.** Exact per-component parameter counts from the architecture.
- **Three kernel versions.** v1 (dequant-then-matmul, shows the materialization cost), v2 (fused), v3 (autotuned + split-K).
- **Correctness first.** Every kernel tested against a PyTorch reference on real model shapes before any benchmark number is reported.
- **Honest baselines.** If Marlin/llama.cpp win, the write-up explains why with profiler evidence.

## Milestones

1. **Roofline and microbenchmarks** (4d) — measured copy bandwidth + theoretical ceilings table
2. **Naive to fused kernel** (9d) — v1/v2/v3, correctness tests, profiler evidence
3. **End-to-end decode** (6d) — decode loop with static KV cache + CUDA graphs, tokens/s, perplexity guard
4. **Baselines, analysis, release** (6d) — production kernel comparison, % of roofline plot, write-up

## Conventions

- Python 3.10+, pytest for roofline math (CPU) and kernel correctness (GPU).
- Every tokens/s number ships with the weight bytes and bandwidth it implies.
- Never claim to beat a production kernel without numbers.

## Development

```bash
cd roofline-decoding
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,dev]"
pytest tests/ -v
PYTHONPATH=src python scripts/measure_roofline.py
PYTHONPATH=src python scripts/bench_kernels.py --bandwidth-gbs <measured>
```

## Current status

- Scaffold + roofline module + 3 Triton kernels + tests written
- Pending: install, measured bandwidth, kernel benchmarks
