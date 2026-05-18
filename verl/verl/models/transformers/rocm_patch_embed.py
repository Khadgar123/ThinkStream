"""ROCm-only VLM patch embedding compatibility helpers."""

from __future__ import annotations

import os

import torch
from torch import nn


def is_rocm_runtime() -> bool:
    if getattr(torch.version, "hip", None):
        return True
    accelerator = str(os.environ.get("DS_ACCELERATOR", "")).lower()
    return accelerator in {"rocm", "hip"} and not getattr(torch.version, "cuda", None)


class RocmLinearizedConv3dPatchEmbed(nn.Module):
    """Run full-patch Conv3d as an equivalent Linear while preserving weights."""

    def __init__(self, patch_embed: nn.Module) -> None:
        super().__init__()
        self.patch_size = int(getattr(patch_embed, "patch_size"))
        self.temporal_patch_size = int(getattr(patch_embed, "temporal_patch_size"))
        self.in_channels = int(getattr(patch_embed, "in_channels"))
        self.embed_dim = int(getattr(patch_embed, "embed_dim"))
        self.proj = getattr(patch_embed, "proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.to(dtype=target_dtype).reshape(hidden_states.shape[0], -1)
        weight = self.proj.weight.reshape(self.proj.out_channels, -1)
        hidden_states = torch.nn.functional.linear(hidden_states, weight, self.proj.bias)
        return hidden_states.view(-1, self.embed_dim)


def _get_visual_module(model: nn.Module) -> nn.Module | None:
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    inner = getattr(model, "model", None)
    return getattr(inner, "visual", None) if inner is not None else None


def patch_rocm_vl_patch_embed(model: nn.Module) -> bool:
    """Replace Qwen-style full-patch Conv3d only when running on ROCm."""

    if not is_rocm_runtime():
        return False
    visual = _get_visual_module(model)
    patch_embed = getattr(visual, "patch_embed", None) if visual is not None else None
    if patch_embed is None or isinstance(patch_embed, RocmLinearizedConv3dPatchEmbed):
        return False
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv3d) or proj.groups != 1:
        return False
    kernel = tuple(int(x) for x in proj.kernel_size)
    stride = tuple(int(x) for x in proj.stride)
    expected = (
        int(getattr(patch_embed, "temporal_patch_size", -1)),
        int(getattr(patch_embed, "patch_size", -1)),
        int(getattr(patch_embed, "patch_size", -1)),
    )
    if kernel != stride or kernel != expected:
        return False
    visual.patch_embed = RocmLinearizedConv3dPatchEmbed(patch_embed)
    return True
