#!/usr/bin/env python3
"""Milestone 1: measured copy bandwidth + theoretical decode ceilings table.

Usage:
  PYTHONPATH=src python scripts/measure_roofline.py
"""
import argparse
import json
from pathlib import Path

from roofline.roofline import ceilings_table, measure_copy_bandwidth_gbs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default="data/results/roofline.json")
    args = ap.parse_args()

    bw = measure_copy_bandwidth_gbs()
    print(f"Measured copy bandwidth: {bw:.1f} GB/s (spec 360 GB/s)", flush=True)
    rows = ceilings_table(bw, seq_len=args.seq_len)

    print(f"\n{'model':28s} {'dtype':>6s} {'weight_GB':>10s} {'KV_MB/tok':>10s} {'ceiling_t/s':>11s}")
    for r in rows:
        print(f"{r['model']:28s} {r['dtype']:>6s} {r['weight_gb']:10.3f} "
              f"{r['kv_mb_per_token']:10.2f} {r['ceiling_tok_s']:11.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bandwidth_gbs": round(bw, 1), "seq_len": args.seq_len,
                               "rows": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
