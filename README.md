# Roofline-First Decoding

A fused 4-bit-weight dequantise-plus-GEMV Triton kernel for batch-1 LLM decoding, measured as a percentage of the RTX 3060's memory bandwidth.

## The idea

Batch-1 decoding is memory-bandwidth bound: each token reads every weight byte once. So the ceiling on tokens/s is bandwidth divided by weight bytes — worked out before any code. Then: write the kernel, profile it, and explain with evidence how far each approach is from the physical limit and why.

## Status

Scaffold complete. Benchmarks pending (install in progress).

## Structure

- `src/roofline/roofline.py` — exact weight counting, measured bandwidth, ceilings table
- `src/roofline/kernels.py` — v1 (dequant-then-matmul), v2 (fused), v3 (autotuned + split-K)
- `scripts/measure_roofline.py` — milestone 1: ceilings table
- `scripts/bench_kernels.py` — milestone 2: per-shape latency + % of roofline

## License

MIT
