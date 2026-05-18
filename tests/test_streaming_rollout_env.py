import os

import torch
from torch import nn

from verl.models.transformers import rocm_patch_embed
from verl.workers.rollout import streaming_rollout


def _set_torch_runtime(monkeypatch, *, hip, cuda, accelerator=None):
    monkeypatch.setattr(streaming_rollout.torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(streaming_rollout.torch.version, "cuda", cuda, raising=False)
    if accelerator is None:
        monkeypatch.delenv("DS_ACCELERATOR", raising=False)
    else:
        monkeypatch.setenv("DS_ACCELERATOR", accelerator)


def test_streaming_server_env_keeps_cuda_off_amd_flash_path(monkeypatch):
    _set_torch_runtime(monkeypatch, hip=None, cuda="12.8")
    monkeypatch.setenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")

    env = streaming_rollout._build_streaming_server_env("3")

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] == "FALSE"
    assert "HIP_VISIBLE_DEVICES" not in env
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES" not in env
    assert "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES" not in env


def test_set_streaming_visible_devices_clears_rocm_only_env_on_cuda(monkeypatch):
    _set_torch_runtime(monkeypatch, hip=None, cuda="12.8")
    for key in streaming_rollout._ROCM_ONLY_ENV_KEYS:
        monkeypatch.setenv(key, "stale")

    streaming_rollout._set_streaming_visible_devices("5")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    for key in streaming_rollout._ROCM_ONLY_ENV_KEYS:
        assert key not in os.environ


def test_streaming_server_env_sets_rocm_masks_and_amd_flash(monkeypatch):
    _set_torch_runtime(monkeypatch, hip="6.3", cuda=None, accelerator="rocm")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx90a")

    env = streaming_rollout._build_streaming_server_env("2")

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["HIP_VISIBLE_DEVICES"] == "2"
    assert env["ROCR_VISIBLE_DEVICES"] == "2"
    assert env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] == "TRUE"
    assert env["PYTORCH_ROCM_ARCH"] == "gfx90a"
    assert env["RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"] == "1"
    assert env["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] == "1"


class _DummyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.temporal_patch_size = 2
        self.in_channels = 3
        self.embed_dim = 4
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
        )


class _DummyVlModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_embed = _DummyPatchEmbed()


def test_rocm_patch_embed_is_cuda_noop(monkeypatch):
    monkeypatch.setattr(rocm_patch_embed.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(rocm_patch_embed.torch.version, "cuda", "12.8", raising=False)
    monkeypatch.delenv("DS_ACCELERATOR", raising=False)
    model = _DummyVlModel()

    patched = rocm_patch_embed.patch_rocm_vl_patch_embed(model)

    assert not patched
    assert isinstance(model.visual.patch_embed, _DummyPatchEmbed)


def test_rocm_patch_embed_linearized_conv3d_matches_original(monkeypatch):
    monkeypatch.setattr(rocm_patch_embed.torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr(rocm_patch_embed.torch.version, "cuda", None, raising=False)
    model = _DummyVlModel()
    original = model.visual.patch_embed
    hidden_states = torch.randn(
        5,
        original.in_channels * original.temporal_patch_size * original.patch_size * original.patch_size,
    )
    expected = original.proj(
        hidden_states.view(
            -1,
            original.in_channels,
            original.temporal_patch_size,
            original.patch_size,
            original.patch_size,
        )
    ).view(-1, original.embed_dim)

    patched = rocm_patch_embed.patch_rocm_vl_patch_embed(model)
    actual = model.visual.patch_embed(hidden_states)

    assert patched
    assert isinstance(model.visual.patch_embed, rocm_patch_embed.RocmLinearizedConv3dPatchEmbed)
    torch.testing.assert_close(actual, expected)
