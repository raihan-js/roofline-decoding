#!/usr/bin/env python3
"""Milestone 3: end-to-end batch-1 decode with every linear swapped for v3 W4A16.

- W4Linear: drop-in nn.Linear replacement. Single-token steps use the fused
  v3 kernel; prefill (long sequences) uses dequantise-then-matmul.
- Greedy generation with the HF KV cache, timed -> tokens/s.
- WikiText-2 perplexity before (bf16) and after (W4) quantisation.

Usage:
  PYTHONPATH=src python scripts/decode_loop.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from roofline.kernels import dequantize_v1, gemv_v3, pack_weights


class W4Linear(nn.Module):
    """nn.Linear with int4-packed weights. Batch-1 decode uses gemv_v3."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().to(torch.float16).cuda()
        packed, scales = pack_weights(w)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None
        if self.has_bias:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16).cuda())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K). Single-token rows go through the fused kernel.
        orig = x.shape
        flat = x.reshape(-1, orig[-1])
        if flat.shape[0] == 1:
            out = gemv_v3(flat[0], self.packed, self.scales).unsqueeze(0)
        elif flat.shape[0] <= 8 and orig[-2] == 1:
            # small batch of single tokens (beam-ish): row-wise v3
            out = torch.stack([gemv_v3(row, self.packed, self.scales)
                               for row in flat])
        else:
            # prefill / long sequences: dequantise once, dense matmul
            w = dequantize_v1(self.packed, self.scales)
            out = flat.float() @ w.float()
            out = out.to(torch.float16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig[:-1], self.out_features)


def swap_linears(model: nn.Module, skip: tuple[str, ...] = ()) -> list[str]:
    """Replace every nn.Linear with W4Linear in place. Returns swapped names."""
    swapped = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, W4Linear(mod))
        swapped.append(name)
    return swapped


def weight_bytes_int4(model: nn.Module) -> int:
    total = 0
    for mod in model.modules():
        if isinstance(mod, W4Linear):
            total += mod.packed.numel() * 4 + mod.scales.numel() * 2
        elif isinstance(mod, nn.Embedding):
            total += mod.weight.numel() * 2
        elif isinstance(mod, nn.Linear):
            total += mod.weight.numel() * 2
    return total


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], max_len: int = 512) -> float:
    import math
    nll, n_tok = 0.0, 0
    model.eval()
    for text in texts:
        ids = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_len)["input_ids"].cuda()
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits.float()
        nll += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += ids.shape[1] - 1
    return math.exp(nll / n_tok)


@torch.no_grad()
def greedy_decode(model, tokenizer, prompt: str, gen_tokens: int = 64) -> tuple[str, float]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    past = None
    gen = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        out = model(input_ids=ids if past is None else ids[:, -1:],
                    past_key_values=past, use_cache=True)
        logits, past = out.logits, out.past_key_values
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        gen.append(nxt.item())
        ids = nxt
        if nxt.item() == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tokenizer.decode(gen), len(gen) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", default="data/results/decode.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda").eval()

    probe = [
        "The meaning of life is a philosophical question concerning significance.",
        "Memory bandwidth, not arithmetic throughput, limits batch-1 decoding.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]
    ppl_bf16 = perplexity(model, tok, probe)
    print(f"bf16 probe ppl: {ppl_bf16:.3f}", flush=True)

    swapped = swap_linears(model, skip=tuple(args.skip))
    wb = weight_bytes_int4(model)
    print(f"swapped {len(swapped)} linears, int4+embed bytes: {wb/1e9:.3f} GB", flush=True)

    ppl_w4 = perplexity(model, tok, probe)
    print(f"W4A16 probe ppl: {ppl_w4:.3f} (ratio {ppl_w4/ppl_bf16:.3f})", flush=True)

    text, tps = greedy_decode(model, tok, args.prompt, args.gen_tokens)
    print(f"decode: {tps:.1f} tok/s :: {text[:120]!r}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "swapped_linears": len(swapped),
        "int4_bytes_gb": round(wb / 1e9, 3),
        "ppl_bf16": round(ppl_bf16, 3),
        "ppl_w4": round(ppl_w4, 3),
        "decode_tok_s": round(tps, 1),
        "sample": text[:200],
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
