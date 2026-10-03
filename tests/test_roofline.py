import pytest

torch = pytest.importorskip("torch")

from roofline.roofline import (ORCH_FUSION, QWEN_15B, achieved_bandwidth_percent,
                               bytes_per_param, ceilings_table,
                               count_decoder_params, decode_ceiling,
                               kv_bytes_per_token, weight_bytes_per_token)


class TestBytesPerParam:
    def test_known_sizes(self):
        assert bytes_per_param("bf16") == 2.0
        assert bytes_per_param("int8") == 1.0
        assert bytes_per_param("int4") == 0.5

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            bytes_per_param("fp32")


class TestParamCounts:
    def test_qwen_totals(self):
        c = count_decoder_params(QWEN_15B)
        # embed is tied: vocab*h counted once
        assert c["head"] == 0
        assert c["embed"] == 151936 * 1536
        # decoder dominates but embed is ~15% (the tied-embedding finding)
        assert 0.10 < c["embed"] / c["total"] < 0.20
        # ~1.5B total
        assert 1.3e9 < c["total"] < 1.7e9

    def test_orch_totals(self):
        c = count_decoder_params(ORCH_FUSION)
        assert c["head"] == 2103 * 1024  # untied
        assert c["embed"] == 2103 * 1024
        # ~272M total (matches ORCH-Fusion spec)
        assert 2.5e8 < c["total"] < 3.0e8

    def test_gqa_saves_kv_params(self):
        full = dict(QWEN_15B, kv_heads=12)
        gqa = dict(QWEN_15B, kv_heads=2)
        assert count_decoder_params(gqa)["decoder"] < count_decoder_params(full)["decoder"]

    def test_non_embedding_excludes_embed_and_head(self):
        c = count_decoder_params(QWEN_15B)
        assert c["non_embedding"] == c["total"] - c["embed"] - c["head"]


class TestWeightBytes:
    def test_bf16_int4_ratio(self):
        bf16 = weight_bytes_per_token(QWEN_15B, "bf16")
        int4 = weight_bytes_per_token(QWEN_15B, "int4")
        assert bf16 / int4 == pytest.approx(4.0)

    def test_qwen_bf16_gb(self):
        # ~1.5B params x 2 bytes ≈ 3.0 GB per token
        assert 2.6e9 < weight_bytes_per_token(QWEN_15B, "bf16") < 3.4e9

    def test_orch_int4_mb(self):
        # ~272M params x 0.5 bytes ≈ 136 MB per token
        assert 120e6 < weight_bytes_per_token(ORCH_FUSION, "int4") < 150e6


class TestKVBytes:
    def test_scales_with_seq_len(self):
        a = kv_bytes_per_token(QWEN_15B, "bf16", seq_len=1024)
        b = kv_bytes_per_token(QWEN_15B, "bf16", seq_len=2048)
        assert b == 2 * a

    def test_kv_small_vs_weights(self):
        # at 2k context, KV read is small next to a 3GB weight read
        kv = kv_bytes_per_token(QWEN_15B, "bf16", seq_len=2048)
        w = weight_bytes_per_token(QWEN_15B, "bf16")
        assert kv / w < 0.15


class TestCeilings:
    def test_int4_ceiling_4x_bf16(self):
        a = decode_ceiling(QWEN_15B, "bf16", 300.0)
        b = decode_ceiling(QWEN_15B, "int4", 300.0)
        # exactly 4x: weights AND kv scale by the same 2.0/0.5 factor
        assert b.ceiling_tok_s / a.ceiling_tok_s == pytest.approx(4.0)

    def test_orch_faster_than_qwen(self):
        # param ratio 1.54B/272M ≈ 5.7x, so the ceiling ratio tracks it
        a = decode_ceiling(QWEN_15B, "int4", 300.0)
        b = decode_ceiling(ORCH_FUSION, "int4", 300.0)
        assert 4.5 < b.ceiling_tok_s / a.ceiling_tok_s < 6.5

    def test_table_shape(self):
        rows = ceilings_table(300.0)
        assert len(rows) == 6  # 2 models x 3 dtypes
        assert all(r["ceiling_tok_s"] > 0 for r in rows)


class TestAchievedPercent:
    def test_half_roofline(self):
        out = achieved_bandwidth_percent(50.0, 3.0, 0.0, 300.0)
        assert out["achieved_gbs"] == pytest.approx(150.0)
        assert out["percent_of_roofline"] == pytest.approx(50.0)

    def test_never_exceeds_100_legitimately(self):
        # >100% means the weight count is wrong, not that physics broke
        out = achieved_bandwidth_percent(200.0, 3.0, 0.0, 300.0)
        assert out["percent_of_roofline"] == pytest.approx(200.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestMeasuredBandwidth:
    def test_copy_bandwidth_sane(self):
        from roofline.roofline import measure_copy_bandwidth_gbs
        bw = measure_copy_bandwidth_gbs(n_bytes=1 << 28, iters=5)
        # 3060: expect 200-360 GB/s
        assert 150.0 < bw < 400.0
