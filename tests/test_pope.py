from __future__ import annotations

from math import pi
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.pope import PoPE
from pymss_core.modules.bs_roformer.transformer import Attention


def small_model(**overrides):
    kwargs = dict(dim=8, depth=2, heads=2, dim_head=4, stereo=True, num_stems=1,
                  time_transformer_depth=1, freq_transformer_depth=1, freqs_per_bands=(4, 5),
                  stft_n_fft=16, stft_hop_length=4, stft_win_length=16, mask_estimator_depth=1,
                  use_pope=True)
    kwargs.update(overrides)
    return BSRoformer(**kwargs)


def test_pope_matches_polar_similarity_and_preserves_gradients():
    torch.manual_seed(17)
    pope = PoPE(dim=4, heads=2).double()
    with torch.no_grad():
        pope.bias.copy_(torch.linspace(-8, 1, 8).reshape(2, 4))
    q = torch.randn(1, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    encoded_q, encoded_k = pope(q, k)
    positions = torch.arange(5, dtype=torch.float64)
    delta = (positions[:, None] - positions[None, :])[None, :, :, None] * pope.inv_freqs
    phases = delta - pope.bias.clamp(-2 * pi, 0)[:, None, None, :]
    expected = (F.softplus(q).unsqueeze(-2) * F.softplus(k).unsqueeze(-3) * phases.cos()).sum(-1)
    actual = encoded_q @ encoded_k.transpose(-1, -2)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    for grad in (q.grad, k.grad, pope.bias.grad):
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


@pytest.mark.parametrize("flash", [False, True])
def test_attention_uses_original_head_dimension_for_scale(flash):
    torch.manual_seed(23)
    pope = PoPE(dim=4, heads=2)
    attention = Attention(dim=8, heads=2, dim_head=4, pope_embed=pope, flash=flash).eval()
    x = torch.randn(2, 5, 8)
    normalized = attention.norm(x)
    q, k, v = attention.to_qkv(normalized).reshape(2, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
    q, k = pope(q, k)
    expected = ((q @ k.transpose(-1, -2)) * 0.5).softmax(-1) @ v
    expected = expected.transpose(1, 2) * attention.to_gates(normalized).sigmoid().unsqueeze(-1)
    expected = attention.to_out(expected.flatten(start_dim=-2))
    torch.testing.assert_close(attention(x), expected, atol=1e-6, rtol=1e-5)


def test_pope_shares_one_embedding_per_axis_and_loads_strictly():
    model = small_model()
    time_attention = model.layers[0][0].layers[0][0]
    freq_attention = model.layers[0][1].layers[0][0]
    assert time_attention.pope_embed is model.layers[1][0].layers[0][0].pope_embed
    assert freq_attention.pope_embed is model.layers[1][1].layers[0][0].pope_embed
    assert time_attention.pope_embed is not freq_attention.pope_embed
    keys = model.state_dict()
    assert "layers.0.0.layers.0.0.pope_embed.bias" in keys
    assert "layers.0.0.layers.0.0.pope_embed.inv_freqs" in keys
    assert not any("rotary_embed" in key for key in keys)
    restored = small_model().eval()
    restored.load_state_dict(keys, strict=True, assign=True)
    with torch.no_grad():
        output = restored(torch.randn(1, 2, 64))
    assert output.shape == (1, 2, 64)
    assert torch.isfinite(output).all()


def test_default_roformer_keeps_rotary_checkpoint_layout():
    model = small_model(use_pope=False)
    keys = model.state_dict()
    assert "layers.0.0.layers.0.0.rotary_embed.freqs" in keys
    assert not any("pope" in key for key in keys)


@pytest.mark.parametrize("backend", ["mlx", "mlx_attention", "mlx_transformer"])
def test_pope_mlx_requests_fall_back_to_torch(backend):
    model = small_model().eval()
    model.set_mps_model_backend("mlx_full", "float32")
    assert model.mps_model_backend == "torch"
    assert not model._use_mlx_full_forward(SimpleNamespace(device=SimpleNamespace(type="mps")))
    for axis in model.layers[0]:
        axis.set_mps_attention_backend(backend)
        assert axis.mps_attention_backend == "torch"
        assert axis.layers[0][0].mps_attention_backend == "torch"


def test_direct_mlx_forward_rejects_pope():
    model = small_model()
    with pytest.raises(NotImplementedError, match="PoPE"):
        model.mlx_forward_mx(None)


def test_pope_is_not_silently_ignored_for_conformer():
    with pytest.raises(ValueError, match="PoPE"):
        small_model(conformer=True)


def test_half_precision_buffers_preserve_adjacent_long_context_positions():
    pope = PoPE(dim=4, heads=2).half()
    q = torch.ones(1, 2, 2050, 4, dtype=torch.float16)
    encoded_q, encoded_k = pope(q, q)
    reference_q, reference_k = pope.float()(q, q)
    torch.testing.assert_close(encoded_q, reference_q, atol=0, rtol=0)
    torch.testing.assert_close(encoded_k, reference_k, atol=0, rtol=0)
    assert not torch.equal(encoded_q[..., 2048, :], encoded_q[..., 2049, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("backend", ["default", "cudnn", "efficient", "math"])
def test_cuda_mixed_precision_forward(backend):
    model = small_model().eval().cuda()
    for axis in model.layers[0]:
        axis.set_cuda_attention_backend(backend)
    x = torch.randn(1, 2, 128, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = model(x)
    assert output.shape == (1, 2, 128)
    assert torch.isfinite(output).all()
