#!/usr/bin/env python3
"""Milestone 4: baselines — bf16 vs bitsandbytes-NF4 vs own W4A16 kernel.

Same model, same prompt, same probe perplexity. Marlin is skipped: its
kernels do not compile on this box (torch 2.13 / CUDA 13, documented in the
FlipGate project). No llama.cpp binary is installed here either.

Usage:
  PYTHONPATH=src python scripts/run_baselines.py
"""
import argparse
import json
import time
from pathlib import Path

import torch

PROMPT = "The meaning of life is"
PROBE = [
    "The meaning of life is a philosophical question concerning significance.",
    "Memory bandwidth, not arithmetic throughput, limits batch-1 decoding.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
]


@torch.no_grad()
def perplexity(model, tokenizer, texts, max_len=512):
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
def greedy_tps(model, tokenizer, gen_tokens=64):
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].cuda()
    past = None
    n = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        out = model(input_ids=ids if past is None else ids[:, -1:],
                    past_key_values=past, use_cache=True)
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        past = out.past_key_values
        ids = nxt
        n += 1
        if nxt.item() == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", default="data/results/baselines.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = {}

    # bf16
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda").eval()
    rows["bf16"] = {"tok_s": round(greedy_tps(m, tok, args.gen_tokens), 1),
                    "ppl": round(perplexity(m, tok, PROBE), 3)}
    print(f"bf16: {rows['bf16']}", flush=True)
    del m
    torch.cuda.empty_cache()

    # bitsandbytes NF4
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="cuda").eval()
    rows["bnb_nf4"] = {"tok_s": round(greedy_tps(m, tok, args.gen_tokens), 1),
                       "ppl": round(perplexity(m, tok, PROBE), 3)}
    print(f"bnb_nf4: {rows['bnb_nf4']}", flush=True)
    del m
    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, **rows}, indent=2))
    print(f"Saved {out} (own-W4 numbers come from decode.json)")


if __name__ == "__main__":
    main()
