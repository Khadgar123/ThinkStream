try:
    from transformers import (
        PretrainedConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )
except Exception:
    # Keep lightweight data-construction imports usable in environments where
    # transformers/tokenizers are not installed or are version-incompatible.
    PretrainedConfig = object  # type: ignore[assignment]
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

MODEL_CLS = {
    name: cls
    for name, cls in {
        "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
        "qwen3vl": Qwen3VLForConditionalGeneration,
    }.items()
    if cls is not None
}

DEFAULT_VIDEO_FLEX_WINDOW_SIZE = 20


def get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text backbone sub-config, handling both flat and nested layouts."""
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config
