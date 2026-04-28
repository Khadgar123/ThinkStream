"""vLLM eval engine wrapper.

Single source of truth for vLLM init + per-sample input preparation.
Matches the canonical pattern in Qwen3-VL/evaluation/VideoMME/run_videomme.py.
"""

import os
from typing import Any, Dict, List

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def init_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 128000,
    max_videos_per_prompt: int = 2,
    seed: int = 3407,
    dtype: str = "bfloat16",
    enforce_eager: bool = False,
):
    from vllm import LLM
    import torch

    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        # video=2 covers visual_window + recalled_frames in agent mode;
        # raise it if you ever stack more video tracks per prompt.
        limit_mm_per_prompt={"video": max_videos_per_prompt},
        seed=seed,
        dtype=dtype,
        enforce_eager=enforce_eager,
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
    )


def prepare_vllm_input(messages: List[Dict], processor) -> Dict[str, Any]:
    """Convert HF chat messages to a vLLM request dict.

    Returns: {"prompt": str, "multi_modal_data": {...}, "mm_processor_kwargs": {...}}
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

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
