"""Helpers for loading eval processors from SFT checkpoints."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoProcessor


_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def processor_source_for_checkpoint(ckpt: str) -> str:
    """Return the directory that contains processor files for ``ckpt``.

    DeepSpeed/HF checkpoints may save model/tokenizer files inside
    ``checkpoint-*`` but leave vision processor configs only in the run root.
    Eval should still load model weights from the checkpoint while resolving
    processor assets from the nearest compatible directory.
    """

    path = Path(ckpt)
    if not path.is_dir():
        return ckpt
    if any((path / name).exists() for name in _PROCESSOR_FILES):
        return str(path)
    parent = path.parent
    if path.name.startswith("checkpoint-") and any(
        (parent / name).exists() for name in _PROCESSOR_FILES
    ):
        return str(parent)
    return str(path)


def load_processor_for_checkpoint(ckpt: str, **kwargs):
    return AutoProcessor.from_pretrained(
        processor_source_for_checkpoint(ckpt), **kwargs
    )
