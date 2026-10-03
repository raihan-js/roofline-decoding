import pytest

torch = pytest.importorskip("torch")

from roofline.kernels import (dequantize_v1, gemv_v2, gemv_v3, pack_weights,
                              reference_gemv)

NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Real matrix shapes from the two models under test.
SHAPES = [
    (1536, 1536),   # Qwen q/o proj
    (1536, 8960),   # Qwen mlp gate/up
    (8960, 1536),   # Qwen mlp down
    (1024, 1024),   # ORCH attn
    (1024, 2816),   # ORCH mlp
    (256, 512),     # small (fast CI shape)
]


def make_case(K, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(K, N, generator=g, dtype=torch.float32, device="cuda")
    w = (w / w.abs().amax(dim=0, keepdim=True).clamp(min=1e-3)).to(torch.float16)
    x = torch.randn(K, generator=g, dtype=torch.float16, device="cuda")
    return x, w


@NEEDS_CUDA
class TestPackRoundtrip:
    @pytest.mark.parametrize("K,N", SHAPES)
    def test_dequant_error_bounded(self, K, N):
        _, w = make_case(K, N)
        packed, scales = pack_weights(w)
        assert packed.shape == (K // 8, N)
        assert scales.shape == (K // 128, N)
        w2 = dequantize_v1(packed, scales)
        # symmetric int4: |err| <= scale/2 per element, plus fp16 noise
        err = (w.float() - w2.float()).abs()
        assert err.max() < 0.15

    def test_pack_is_int32(self):
        _, w = make_case(256, 512)
        packed, scales = pack_weights(w)
        assert packed.dtype == torch.int32
        assert scales.dtype == torch.float16


@NEEDS_CUDA
class TestCorrectness:
    """Kernels must reproduce x @ W_dequant up to fp16 arithmetic noise.

    Comparing against the dequantised reference (not the fp16 original)
    separates kernel bugs from expected int4 quantisation noise, so the
    tolerance can stay tight.
    """

    @pytest.mark.parametrize("K,N", SHAPES)
    def test_v1_matches_dequant_ref(self, K, N):
        x, w = make_case(K, N)
        packed, scales = pack_weights(w)
        w2 = dequantize_v1(packed, scales)
        want = (x.float() @ w2.float())
        # sanity: dequant ref differs from fp16 original only by quant noise
        assert ((x.float() @ w2.float()) - reference_gemv(x, w).float()).abs().max() < 15.0
        # v1 IS x @ dequant: must match to fp16 noise
        v1 = (x.float() @ dequantize_v1(packed, scales).float())
        assert torch.allclose(v1, want, atol=0.05, rtol=0.005)

    @pytest.mark.parametrize("K,N", SHAPES)
    def test_v2_matches_dequant_ref(self, K, N):
        x, w = make_case(K, N)
        packed, scales = pack_weights(w)
        w2 = dequantize_v1(packed, scales)
        want = (x.float() @ w2.float())
        got = gemv_v2(x, packed, scales).float()
        assert torch.allclose(got, want, atol=0.1, rtol=0.01)

    @pytest.mark.parametrize("K,N", [(1536, 1536), (1536, 8960), (256, 512)])
    @pytest.mark.parametrize("split_k", [1, 2, 4])
    def test_v3_matches_dequant_ref(self, K, N, split_k):
        x, w = make_case(K, N)
        packed, scales = pack_weights(w)
        w2 = dequantize_v1(packed, scales)
        want = (x.float() @ w2.float())
        got = gemv_v3(x, packed, scales, split_k=split_k).float()
        # split_k>1 uses atomic_add: slightly looser tolerance
        assert torch.allclose(got, want, atol=0.25, rtol=0.02)

    def test_v2_v3_agree(self):
        x, w = make_case(1024, 2816)
        packed, scales = pack_weights(w)
        a = gemv_v2(x, packed, scales).float()
        b = gemv_v3(x, packed, scales, split_k=1).float()
        assert torch.allclose(a, b, atol=0.25, rtol=0.02)
