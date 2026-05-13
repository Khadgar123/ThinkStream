"""vLLM eval engine wrapper.

Single source of truth for vLLM init + per-sample input preparation.
Matches the canonical pattern in Qwen3-VL/evaluation/VideoMME/run_videomme.py.
"""

import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def default_mm_processor_cache_gb() -> int:
    """Choose a CPU-side vLLM multimodal preprocessor cache size.

    Streaming windows reuse the same decoded/resized frames across adjacent
    chunks. This cache is therefore the primary vLLM speed path for our
    image-pad/video-meta rollout. Environment overrides take precedence.
    """
    for key in (
        "THINKSTREAM_MM_CACHE_GB",
        "VLLM_MM_PROCESSOR_CACHE_GB",
        "MM_CACHE_GB",
    ):
        raw = os.environ.get(key)
        if raw:
            return max(0, int(float(raw)))

    available_gib = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_gib = float(line.split()[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        available_gib = 0.0

    if available_gib >= 1536:
        return 512
    if available_gib >= 768:
        return 256
    if available_gib >= 384:
        return 128
    if available_gib >= 128:
        return 64
    return 16


def init_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 128000,
    max_images_per_prompt: int = 64,
    max_videos_per_prompt: int = 2,
    seed: int = 3407,
    dtype: str = "bfloat16",
    enforce_eager: bool = False,
    enable_prefix_caching: bool = True,
    mm_processor_cache_gb: Optional[int] = None,
    disable_mm_preprocessor_cache: bool = False,
):
    """Init vLLM engine for ThinkStream eval / rollout.

    enable_prefix_caching=True is critical for streaming video — each chunk's
    prompt shares the system+visual_window prefix with previous chunks, so
    block-level KV cache reuse gives 3-10× speedup. DeepEyesV2 verl recipe
    sets this by default (vllm_rollout_spmd.py:167).

    Set enable_prefix_caching=False only when:
      - Debugging non-determinism in cached vs uncached path
      - On vLLM versions where prefix cache + multi_modal_data interact buggy
    """
    from vllm import LLM
    import torch

    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    if mm_processor_cache_gb is None:
        mm_processor_cache_gb = default_mm_processor_cache_gb()

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        # ThinkStream's canonical pre-extracted-frame protocol is
        # timestamped image lists. Keep video as a fallback for legacy raw
        # video baselines, but normal SFT/RL/eval traffic consumes image slots.
        limit_mm_per_prompt={
            "image": max_images_per_prompt,
            "video": max_videos_per_prompt,
        },
        seed=seed,
        dtype=dtype,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        mm_processor_cache_gb=int(mm_processor_cache_gb),
        disable_mm_preprocessor_cache=bool(disable_mm_preprocessor_cache),
    )


def make_sampling_params(
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    stop_token_ids: List[int] = None,
):
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        stop_token_ids=stop_token_ids or [],
        skip_special_tokens=False,
    )


def generate_with_turn_sampling(
    llm,
    inputs: List[Dict[str, Any]],
    turn_kinds: List[str],
    default_sampling_params,
    sampling_params_by_turn: Dict[str, Any] = None,
):
    """Run vLLM generation with per-turn sampling params while preserving order."""
    if len(inputs) != len(turn_kinds):
        raise ValueError("inputs and turn_kinds must have the same length")
    if not inputs:
        return []
    sampling_params_by_turn = sampling_params_by_turn or {}
    if not sampling_params_by_turn:
        return llm.generate(inputs, sampling_params=default_sampling_params)

    params = [
        sampling_params_by_turn.get(kind, default_sampling_params)
        for kind in turn_kinds
    ]
    if all(p is default_sampling_params for p in params):
        return llm.generate(inputs, sampling_params=default_sampling_params)

    # vLLM accepts one SamplingParams object per request. This keeps mixed
    # streaming/compress turns in a single scheduler batch instead of splitting
    # them into small generate calls such as 61+3 or 63+1.
    try:
        return llm.generate(inputs, sampling_params=params)
    except (TypeError, ValueError, AssertionError) as exc:
        msg = str(exc).lower()
        if "sampling" not in msg and "params" not in msg and "list" not in msg:
            raise

    grouped: Dict[str, List[int]] = {}
    for i, kind in enumerate(turn_kinds):
        key = kind if kind in sampling_params_by_turn else "__default__"
        grouped.setdefault(key, []).append(i)

    outputs = [None] * len(inputs)
    for key, idxs in grouped.items():
        params = sampling_params_by_turn.get(key, default_sampling_params)
        batch = [inputs[i] for i in idxs]
        batch_outputs = llm.generate(batch, sampling_params=params)
        for i, out in zip(idxs, batch_outputs):
            outputs[i] = out
    return outputs


def prepare_vllm_input(
    messages: List[Dict], processor, *, tools: List[Dict] = None,
) -> Dict[str, Any]:
    """Convert HF chat messages to a vLLM request dict.

    Returns: {"prompt": str, "multi_modal_data": {...}, "mm_processor_kwargs": {...}}

    tools: optional v12 turn-local tool schema. When provided, the
    chat template renders <tools>...</tools> in the system prompt so the
    model can emit <tool_call>{...}</tool_call>. Pass None for v12 turns
    whose action space has no tools, such as recall-result answer turns.
    """
    from qwen_vl_utils import process_vision_info

    normalized_messages: List[Dict] = []
    explicit_video_metadata: List[Dict] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            normalized_messages.append(msg)
            continue
        new_content = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            item = dict(item)
            if (
                item.get("type") == "video"
                and item.get("visual_carrier") == "image_pad"
                and (item.get("image") or item.get("image_url"))
            ):
                item["type"] = "image"
                item.pop("visual_carrier", None)
            if item.get("type") == "video":
                meta = item.get("video_metadata")
                if isinstance(meta, dict):
                    explicit_video_metadata.append(
                        {k: v for k, v in meta.items()
                         if k != "do_sample_frames"}
                    )
            new_content.append(item)
        normalized_messages.append({**msg, "content": new_content})

    template_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        do_sample_frames=False,
    )
    if tools is not None:
        template_kwargs["tools"] = tools
    if explicit_video_metadata:
        template_kwargs["video_metadata"] = explicit_video_metadata
    text = processor.apply_chat_template(normalized_messages, **template_kwargs)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        normalized_messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is not None and explicit_video_metadata:
        fixed_video_inputs = []
        for i, video_input in enumerate(video_inputs):
            if (
                i < len(explicit_video_metadata)
                and isinstance(video_input, tuple)
                and len(video_input) == 2
            ):
                fixed_video_inputs.append((video_input[0], explicit_video_metadata[i]))
            else:
                fixed_video_inputs.append(video_input)
        video_inputs = fixed_video_inputs

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs or {},
    }
