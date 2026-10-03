"""Roofline-first decoding: fused W4A16 GEMV measured against the RTX 3060 roofline."""

from roofline.roofline import (ORCH_FUSION, QWEN_15B, achieved_bandwidth_percent,
                               ceilings_table, decode_ceiling,
                               measure_copy_bandwidth_gbs)

__version__ = "0.1.0"
__all__ = [
    "ORCH_FUSION", "QWEN_15B", "achieved_bandwidth_percent",
    "ceilings_table", "decode_ceiling", "measure_copy_bandwidth_gbs",
]