# My Triton Kernel Ties bitsandbytes — and Both Lose to Math

*A fused W4A16 GEMV for batch-1 decoding, measured against the RTX 3060 roofline.*

---

> Scope note: one RTX 3060 12GB, batch-1 greedy decoding, Qwen2.5-1.5B-Instruct. This is a learning kernel, stated as such — the point is understanding, not beating Marlin.

## The roofline first

Batch-1 decoding is memory-bandwidth bound: each token reads every weight byte once. So the ceiling is bandwidth divided by weight bytes — worked out before any code.

Measured (torch copy benchmark, not the spec sheet): **323.9 GB/s** (spec: 360).

Counted (from the architecture, not guessed):

| Model | dtype | Weight GB | Ceiling tok/s |
|---|---|---|---|
| Qwen2.5-1.5B | bf16 | 3.087 | 102.9 |
| Qwen2.5-1.5B | int4 | 0.772 | 411.8 |
| ORCH-Fusion | bf16 | 0.550 | 539.7 |
| ORCH-Fusion | int4 | 0.137 | 2158.8 |

Two details worth knowing: the tied 151,936-token embedding is ~15% of Qwen's weights, and ORCH's matrices are so small that launch overhead may dominate — tested below.

## The kernels

Three versions, all correctness-tested against a PyTorch reference on the models' real matrix shapes before any number was reported:

- **v1** (dequantise-then-matmul): materialise fp16, then torch matmul. Shows the cost of the extra memory pass — 3–7× slower than v2.
- **v2** (fused): one kernel, dequantise on the fly, no materialisation.
- **v3** (split-K + tuned): autotuned tiles, split-K parallelism for large matrices.

Two bugs found along the way, both load-bearing:

1. **Missing +8 zero-point** in packing — every positive weight destroyed. Caught by review before first GPU run.
2. **`@triton.autotune` vs `atomic_add`**: autotune benchmarks all configs against the same output buffer (~500 timing reps), and a split-K kernel accumulates into it — so the buffer holds ~500 stale copies before the real launch, and whichever config tunes first benchmarks garbage. Fixed with explicit tuning (fresh buffer per trial, winner cached). If you write split-K kernels, check yours for this.

## The numbers

| Setup | tok/s | Probe ppl | GB/s | % of roofline |
|---|---|---|---|---|
| bf16 (HF) | 31.3 | 12.91 | 96.6 | **29.8%** |
| bitsandbytes NF4 | 25.7 | 18.98 | 19.8 | 6.1% |
| **own W4 (v3)** | **25.5** | **15.76** | 19.7 | **6.1%** |

My kernel ties bitsandbytes on speed with better perplexity. And both lose to bf16 — which at 30% of roofline is the best-engineered path here.

## Why Marlin is still faster (and why 6% is honest)

Per-layer breakdown: GEMV time sums to ~25ms of the ~39ms step — the rest is attention, norms, sampling, and Python dispatch. Within GEMV time, the scalar nibble-unpack loop (shift, mask, subtract, scale per element) dominates: ~6 ALU ops per MAC with no pipelining. Marlin hides exactly this with async copies, software pipelining, and ahead-of-time bit staging. A `tl.dot` rewrite was attempted and measured **4–7× slower** — the dequant-scatter into the dot tile cost more than the dot saved. Negative results reported as such.

The int4 ceiling (412 tok/s) sits 16× above measured. That gap — dequant arithmetic, launch overhead on small shapes, dispatch — is the project.

## Limitations

- Probe perplexity is 4 sentences (noisy); bnb quantizes different layers than my swap — the ppl gap is directional, not definitive.
- No Marlin numbers: its kernels do not compile on this box (torch 2.13 / CUDA 13).
- No llama.cpp comparison: no binary installed here.
- Single prompt timing; batch-1 only; greedy only.
- ORCH-Fusion end-to-end pending (custom .pt checkpoint needs its own loader); its shapes are covered in microbenchmarks.

## What's next

- Calibrated (GPTQ-style) scales to close the ppl gap
- CUDA graphs around the decode step to test the launch-overhead hypothesis on ORCH shapes
- ORCH-Fusion end-to-end via its own checkpoint loader

---

*Repo: github.com/raihan-js/roofline-decoding · 46 tests green. Method: measure first, write second, explain the gap.*
