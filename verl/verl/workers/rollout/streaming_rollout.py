# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""True-KV HF streaming rollout backend for ThinkStream.

This backend intentionally does not expose an OpenAI/vLLM-compatible HTTP
surface. It is a Ray actor that owns one local HuggingFace Qwen-VL model and a
``StreamingWindowInferenceEngine``. A trajectory leases one server for its full
agent-loop lifetime, so the server can keep physical KV state without another
trajectory interleaving into the same cache.

The first reliable scaling target is one server per GPU (``GEN_TP=1``): an
8-GPU node runs eight independent true-KV trajectory streams in parallel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_resource_name, get_visible_devices_keyword
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


_ROCM_ONLY_ENV_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
    "PYTORCH_ROCM_ARCH",
    "ROCM_HOME",
    "HIP_HOME",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
)


def _is_rocm_runtime() -> bool:
    from verl.models.transformers.rocm_patch_embed import is_rocm_runtime

    return is_rocm_runtime()


def _set_streaming_visible_devices(cuda_visible_devices: str) -> None:
    visible = str(cuda_visible_devices)
    os.environ[get_visible_devices_keyword()] = visible
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if _is_rocm_runtime():
        # On ROCm, Ray/verl reports CUDA_VISIBLE_DEVICES as the generic device
        # key, while PyTorch/flash-attn also respect HIP_VISIBLE_DEVICES.
        os.environ["HIP_VISIBLE_DEVICES"] = visible
        os.environ["ROCR_VISIBLE_DEVICES"] = visible
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
        if os.path.isdir("/opt/rocm"):
            os.environ.setdefault("ROCM_HOME", "/opt/rocm")
            os.environ.setdefault("HIP_HOME", "/opt/rocm")
    else:
        for key in _ROCM_ONLY_ENV_KEYS:
            os.environ.pop(key, None)


def _build_streaming_server_env(visible: str) -> dict[str, str]:
    visible = str(visible)
    env_vars = {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_VISIBLE_DEVICES": visible,
        get_visible_devices_keyword(): visible,
    }
    rocm_runtime = _is_rocm_runtime()
    if rocm_runtime:
        env_vars.update({
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            "HIP_VISIBLE_DEVICES": visible,
            "ROCR_VISIBLE_DEVICES": visible,
            "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
            "PYTORCH_ROCM_ARCH": "gfx942",
        })
        if os.path.isdir("/opt/rocm"):
            env_vars.setdefault("ROCM_HOME", "/opt/rocm")
            env_vars.setdefault("HIP_HOME", "/opt/rocm")
    else:
        # Stale ROCm settings route flash-attn through Triton AMD kernels on
        # NVIDIA nodes and fail before rollout generation starts.
        env_vars["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"
    propagate_keys = []
    if rocm_runtime:
        propagate_keys.extend([
            "LD_LIBRARY_PATH",
            "PYTORCH_ROCM_ARCH",
            "ROCM_HOME",
            "HIP_HOME",
        ])
    for key in propagate_keys:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


def _dtype_from_config(dtype_name: str) -> torch.dtype:
    value = str(dtype_name or "bfloat16").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported streaming rollout dtype: {dtype_name!r}")


def _normalize_model_type(model_type: str | None, architectures: Optional[list[str]] = None) -> str:
    value = str(model_type or "").lower().replace("_", "").replace("-", "")
    arch = "".join(architectures or []).lower().replace("_", "").replace("-", "")
    if "qwen25vl" in value or "qwen25vl" in arch or "qwen2.5vl" in value:
        return "qwen2.5vl"
    if "qwen3vl" in value or "qwen3vl" in arch:
        return "qwen3vl"
    # ThinkStream v12 is Qwen3-VL only. Keep the fallback explicit so older
    # checkpoints with sparse config metadata still run.
    return "qwen3vl"


def _get_text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def _set_video_pixels(processor: Any) -> None:
    min_pixels = int(
        os.environ.get("IMAGE_MIN_PIXELS")
        or os.environ.get("MIN_PIXELS")
        or 256 * 28 * 28
    )
    max_pixels = int(
        os.environ.get("IMAGE_MAX_PIXELS")
        or os.environ.get("MAX_PIXELS")
        or 512 * 28 * 28
    )
    vp = getattr(processor, "video_processor", None)
    if vp is None:
        return
    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        try:
            setattr(vp, name, value)
        except Exception:
            pass
    size = getattr(vp, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = min_pixels
        size["longest_edge"] = max_pixels


def _inspect_streaming_delta_text(
    text: str,
    *,
    turn_kind: str,
    after_post_recall: bool = False,
) -> list[str]:
    """Return true-KV delta contamination flags for debug/strict checks."""
    value = str(text or "")
    kind = str(turn_kind or "").strip().lower()
    flags: list[str] = []
    if not value:
        return flags

    if "<|im_start|>system" in value:
        flags.append("repeated_system_prompt")
    if "# Tools" in value or "<tools>" in value or "</tools>" in value:
        flags.append("repeated_tool_schema")
    if value.count("<visual_window>") > 1:
        flags.append("multiple_visual_windows")
    if len(re.findall(r"<t=", value)) > 1:
        flags.append("multiple_current_timestamps")

    if kind != "post_recall":
        if "<recalled_frames>" in value or "</recalled_frames>" in value:
            flags.append("recall_frames_in_non_post_recall_delta")
        if "<recall_result>" in value or "</recall_result>" in value:
            flags.append("recall_result_in_non_post_recall_delta")
        if "<tool_response>" in value or "</tool_response>" in value:
            flags.append("tool_response_in_non_post_recall_delta")
        if "<tool_call>" in value or "</tool_call>" in value:
            flags.append("tool_call_in_non_post_recall_delta")

    if after_post_recall and kind == "streaming":
        if (
            "<recalled_frames>" in value
            or "<recall_result>" in value
            or "<tool_response>" in value
            or "</tool_response>" in value
            or "The recall tool returned" in value
        ):
            flags.append("post_recall_evidence_leaked_to_next_streaming_delta")
        if "<tool_call>" in value or "</tool_call>" in value:
            flags.append("post_recall_tool_call_leaked_to_next_streaming_delta")
    return flags


class ServerAdapter(BaseRollout):
    """Trainer-side adapter used by verl's async weight update path."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        if rollout_world_size != 1:
            raise ValueError(
                "ThinkStream streaming rollout is one local HF model per GPU; "
                "set actor_rollout_ref.rollout.tensor_model_parallel_size=1, "
                "data_parallel_size=1 and pipeline_model_parallel_size=1."
            )
        rank = int(os.environ.get("RANK", "0"))
        self.replica_rank = rank if replica_rank == -1 else int(replica_rank)
        self.node_rank = 0
        self.server_handle: ray.actor.ActorHandle | None = None

    def _server_name(self) -> str:
        return f"streaming_server_{self.replica_rank}_{self.node_rank}"

    def _get_server(self) -> ray.actor.ActorHandle:
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._server_name())
        return self.server_handle

    async def resume(self, tags: list[str]):
        if self.config.free_cache_engine:
            await self._get_server().wake_up.remote(tags=tags)

    async def release(self):
        if self.config.free_cache_engine:
            await self._get_server().sleep.remote()

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs: Any,
    ):
        start = time.time()
        state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if tensor is None:
                continue
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            state_dict[str(name)] = tensor.detach().to("cpu", non_blocking=False).contiguous()
        result = await self._get_server().update_weights_from_state_dict.remote(
            state_dict,
            global_steps=global_steps,
            peft_config=kwargs.get("peft_config"),
            base_sync_done=kwargs.get("base_sync_done", True),
        )
        if self.replica_rank == 0:
            logger.info(
                "streaming update_weights loaded %s tensors in %.2fs: %s",
                len(state_dict),
                time.time() - start,
                result,
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError("ThinkStream streaming rollout only supports async AgentLoop generation.")


class StreamingRolloutServer:
    """Single-GPU true-KV rollout actor."""

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ray.actor.ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        _set_streaming_visible_devices(cuda_visible_devices)
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = int(replica_rank)
        self.node_rank = int(node_rank)
        self.gpus_per_node = int(gpus_per_node)
        self.nnodes = int(nnodes)
        self.cuda_visible_devices = str(cuda_visible_devices)

        self._lock = asyncio.Lock()
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._engine = None
        self._slots_per_gpu = max(
            1,
            int(os.environ.get("THINKSTREAM_STREAMING_SLOTS_PER_GPU", "1") or 1),
        )
        self._request_to_slot: dict[str, int] = {}
        self._slot_to_request: list[str | None] = [None] * self._slots_per_gpu
        self._last_turn_was_post_recall: list[bool] = [False] * self._slots_per_gpu
        self._pending_turns: list[dict[str, Any]] = []
        self._pending_flush_task: asyncio.Task | None = None
        self._state_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()
        self._global_steps: int | None = None
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._dtype = _dtype_from_config(self.config.dtype)
        self._model_type = _normalize_model_type(
            getattr(getattr(self.model_config, "hf_config", None), "model_type", None),
            getattr(self.model_config, "architectures", None),
        )
        if self.config.tensor_model_parallel_size != 1 or self.config.data_parallel_size != 1:
            raise ValueError("streaming rollout server requires TP=1 and DP=1 per replica.")
        self._load_model()

    def _trace_jsonl(self, payload: dict[str, Any]) -> None:
        path = str(os.environ.get("THINKSTREAM_ROLLOUT_TRACE_JSONL", "") or "").strip()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("failed to write streaming rollout trace %s: %s", path, exc)

    def _timing_jsonl(self, payload: dict[str, Any]) -> None:
        path = str(os.environ.get("THINKSTREAM_ROLLOUT_TIMING_JSONL", "") or "").strip()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("failed to write streaming rollout timing %s: %s", path, exc)

    @staticmethod
    def _tensor_list(value: Any) -> Any:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().to("cpu")
            if hasattr(value, "tolist"):
                return value.tolist()
        except Exception:
            return None
        return value

    @staticmethod
    def _video_payload_summary(video_data: Optional[list[Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(video_data or []):
            video = item[0] if isinstance(item, (list, tuple)) and item else item
            meta = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else None
            if isinstance(video, (list, tuple)):
                sample = [str(x) for x in list(video)[:4]]
                video_desc: dict[str, Any] = {"type": type(video).__name__, "len": len(video), "sample": sample}
            else:
                video_desc = {"type": type(video).__name__, "repr": str(video)[:256]}
            out.append({"index": idx, "video": video_desc, "metadata": meta})
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoProcessor

        from thinkstream.models import MODEL_CLS

        model_path = (
            getattr(self.model_config, "local_path", None)
            or getattr(self.model_config, "path", None)
        )
        if not model_path:
            raise ValueError("model_config.path is required for streaming rollout")
        if self._model_type not in MODEL_CLS:
            raise ValueError(
                f"Unsupported ThinkStream streaming model_type={self._model_type!r}; "
                f"available={sorted(MODEL_CLS)}"
            )

        model_cls = MODEL_CLS[self._model_type]
        self._model = model_cls.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        from verl.models.transformers.rocm_patch_embed import patch_rocm_vl_patch_embed

        if patch_rocm_vl_patch_embed(self._model):
            logger.info("Patched ROCm VLM patch_embed Conv3d to linear projection")
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=getattr(self.model_config, "trust_remote_code", False),
        )
        self._tokenizer = getattr(self._processor, "tokenizer", None) or getattr(
            self.model_config, "tokenizer", None
        )
        if self._tokenizer is None:
            raise ValueError("Streaming rollout requires a tokenizer")
        legacy_tokens = str(
            os.environ.get("THINKSTREAM_ALLOW_LEGACY_AGENT_TOKENS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if legacy_tokens:
            logger.warning(
                "Streaming rollout is using legacy checkpoint agent tokens without "
                "runtime token registration. Use only for checkpoints trained with "
                "old <response>/<silent> tags."
            )
        else:
            from thinkstream.data.agent_protocol import (
                ensure_agent_special_tokens,
                validate_agent_special_tokens,
            )

            added = ensure_agent_special_tokens(self._tokenizer, model=self._model)
            validate_agent_special_tokens(self._tokenizer)
            if added:
                logger.warning(
                    "Streaming rollout added %d missing agent special tokens at load "
                    "time. Prefer checkpoints saved after SFT token registration.",
                    added,
                )
        custom_template = getattr(self.model_config, "custom_chat_template", None)
        if custom_template:
            self._processor.chat_template = custom_template
            self._tokenizer.chat_template = custom_template
        _set_video_pixels(self._processor)
        text_config = _get_text_config(self._model.config)
        if hasattr(self._model.config, "text_config"):
            self._model.config.text_config._attn_implementation = "flash_attention_2_infer"
        else:
            self._model.config._attn_implementation = "flash_attention_2_infer"
        self._model.eval()
        self._model.to(self._device)
        logger.info("Streaming rollout server %s loaded %s on %s", self.replica_rank, model_path, self._device)

    def _build_engine(self):
        if self._engine is not None:
            return self._engine
        from thinkstream.models import DEFAULT_VIDEO_FLEX_WINDOW_SIZE
        from thinkstream.models.inference import StreamingWindowInferenceEngine

        self._model.to(self._device)
        self._model.eval()
        text_config = _get_text_config(self._model.config)
        num_heads = int(getattr(text_config, "num_attention_heads"))
        head_dim = int(getattr(text_config, "head_dim", getattr(text_config, "hidden_size") // num_heads))
        eos_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id < 0:
            eos_id = self._tokenizer.eos_token_id
        video_token_id = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
        if video_token_id is None or video_token_id < 0:
            raise ValueError("Tokenizer does not define <|video_pad|>; use THINKSTREAM_FRAME_PROTOCOL=video_meta")
        max_len = int(self.config.max_model_len or (self.config.prompt_length + self.config.response_length))
        self._engine = StreamingWindowInferenceEngine(
            model=self._model,
            batch_size=self._slots_per_gpu,
            max_len=max_len,
            num_hidden_layers=int(getattr(text_config, "num_hidden_layers")),
            num_key_value_heads=int(getattr(text_config, "num_key_value_heads", num_heads)),
            head_dim=head_dim,
            vocab_size=int(
                getattr(text_config, "vocab_size", None)
                or getattr(self._model.config, "vocab_size", None)
                or len(self._tokenizer)
            ),
            pad_token_id=int(self._tokenizer.pad_token_id or eos_id),
            eos_token_ids=[int(eos_id)],
            video_token_id=int(video_token_id),
            video_flex_window_size=int(
                os.environ.get(
                    "THINKSTREAM_VISUAL_WINDOW_CHUNKS",
                    getattr(self._model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE),
                )
            ),
            dtype=self._dtype,
            device=self._device,
        )
        return self._engine

    async def wake_up(self, tags: Optional[list[str]] = None):
        del tags
        if self._model is not None:
            self._model.to(self._device)
        return True

    async def sleep(self):
        async with self._lock:
            if self._engine is not None:
                self._engine.reset()
                self._engine = None
            self._request_to_slot.clear()
            self._slot_to_request = [None] * self._slots_per_gpu
            self._last_turn_was_post_recall = [False] * self._slots_per_gpu
            if self._model is not None:
                self._model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True

    async def clear_kv_cache(self):
        async with self._lock:
            if self._engine is not None:
                self._engine.reset()
            self._request_to_slot.clear()
            self._slot_to_request = [None] * self._slots_per_gpu
            self._last_turn_was_post_recall = [False] * self._slots_per_gpu
        return True

    async def update_weights_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        global_steps: int = None,
        peft_config: Any = None,
        base_sync_done: bool = True,
    ) -> dict[str, Any]:
        del peft_config, base_sync_done
        async with self._lock:
            target_device = next(self._model.parameters()).device
            casted = {
                name: tensor.to(device=target_device, dtype=self._dtype, non_blocking=False)
                if torch.is_floating_point(tensor)
                else tensor.to(device=target_device, non_blocking=False)
                for name, tensor in state_dict.items()
            }
            incompatible = self._model.load_state_dict(casted, strict=False)
            self._global_steps = global_steps
            if self._engine is not None:
                self._engine.reset()
            self._request_to_slot.clear()
            self._slot_to_request = [None] * self._slots_per_gpu
            self._last_turn_was_post_recall = [False] * self._slots_per_gpu
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            return {
                "loaded": len(casted),
                "missing": len(missing),
                "unexpected": len(unexpected),
                "global_steps": global_steps,
            }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        token_ids: list[int],
        *,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        turn_kind: str = "",
        chunk_idx: Optional[int] = None,
    ) -> dict[str, torch.Tensor | None]:
        if image_data:
            raise RuntimeError(
                "ThinkStream true-KV streaming rollout supports video_meta frames only; "
                "set THINKSTREAM_FRAME_PROTOCOL=video_meta."
            )
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        if video_data:
            videos, video_metadatas = zip(*video_data, strict=False)
            text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            video_token = getattr(self._processor, "video_token", None)
            if video_token and text.count(video_token) != len(videos):
                # `token_ids` already contains Qwen's expanded video placeholders.
                # Feeding the decoded prompt back to the processor would make it
                # expect one metadata entry per expanded token instead of one per
                # logical video block. The processor output ids are replaced by
                # `token_ids` below; this text is only needed to build video
                # tensors/grids and mrope metadata.
                text = "\n".join([video_token] * len(videos))
            encoded = self._processor(
                text=[text],
                videos=list(videos),
                video_metadata=list(video_metadatas),
                return_tensors="pt",
                do_sample_frames=False,
            )
            encoded = dict(encoded)
            encoded["input_ids"] = input_ids
            encoded["attention_mask"] = attention_mask
            video_token_id = getattr(self._processor, "video_token_id", None)
            pixel_values = encoded.get("pixel_values_videos")
            video_grid = encoded.get("video_grid_thw")
            merge_size = int(getattr(self._processor.video_processor, "merge_size", 2))
            if (
                video_token_id is not None
                and pixel_values is not None
                and video_grid is not None
            ):
                expected_features = int((input_ids == int(video_token_id)).sum().item())
                actual_features = int(
                    (video_grid.to(torch.long).prod(dim=1) // (merge_size**2)).sum().item()
                )
                if expected_features > 0 and actual_features != expected_features:
                    if len(video_grid) != 1:
                        raise ValueError(
                            "Streaming rollout video-token/grid mismatch for "
                            f"turn_kind={turn_kind!r} chunk_idx={chunk_idx!r}: "
                            f"expected_features={expected_features} "
                            f"actual_features={actual_features} "
                            f"video_blocks={len(videos)} "
                            f"grid_rows={len(video_grid)} "
                            f"token_len={len(token_ids)}. This usually means "
                            "the incremental prompt slice cut into a multi-block "
                            "recall tool response."
                        )
                    logger.warning(
                        "streaming rollout single-grid video feature mismatch patched: "
                        "turn_kind=%s chunk_idx=%s expected_features=%s actual_features=%s "
                        "patches=%s grid=%s token_len=%s",
                        turn_kind,
                        chunk_idx,
                        expected_features,
                        actual_features,
                        int(pixel_values.shape[0]),
                        video_grid.detach().cpu().tolist(),
                        len(token_ids),
                    )
                    # The agent loop's prompt ids are authoritative for the KV
                    # stream. Keep the rollout alive if the local processor
                    # re-materializes a slightly different video grid for the
                    # same frame payload.
                    target_patches = expected_features * (merge_size**2)
                    current_patches = int(pixel_values.shape[0])
                    if current_patches < target_patches:
                        pad = pixel_values[-1:].expand(target_patches - current_patches, -1)
                        encoded["pixel_values_videos"] = torch.cat([pixel_values, pad], dim=0)
                    elif current_patches > target_patches:
                        encoded["pixel_values_videos"] = pixel_values[:target_patches]
                    if len(video_grid) == 1:
                        t, h, w = [int(x) for x in video_grid[0].tolist()]
                        if t > 0 and w > 0 and target_patches % (t * w) == 0:
                            h = target_patches // (t * w)
                        elif t > 0 and h > 0 and target_patches % (t * h) == 0:
                            w = target_patches // (t * h)
                        else:
                            t, h, w = 1, merge_size, target_patches // merge_size
                        encoded["video_grid_thw"] = torch.tensor(
                            [[t, h, w]],
                            dtype=video_grid.dtype,
                            device=video_grid.device,
                        )

        from thinkstream.data.stream_data_processor import compute_position_ids

        rope_inputs = dict(encoded)
        rope_inputs["video_chunk_size"] = 1
        position_ids = compute_position_ids(rope_inputs, self._processor, self._model_type)

        out: dict[str, torch.Tensor | None] = {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
            "position_ids": position_ids.to(self._device),
            "pixel_values_videos": None,
            "video_grid_thw": None,
        }
        if "pixel_values_videos" in encoded:
            out["pixel_values_videos"] = encoded["pixel_values_videos"].to(self._device)
        if "video_grid_thw" in encoded:
            out["video_grid_thw"] = encoded["video_grid_thw"].to(self._device)
        return out

    def _pad_prepared_inputs(
        self,
        per_slot_inputs: dict[int, dict[str, torch.Tensor | None]],
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Build a strict batch for the fixed-size streaming engine.

        Rows not present in ``per_slot_inputs`` are inactive dummy rows. Their
        attention mask is all-zero and ``active_mask`` is False, so the engine
        preserves their KV/cache metadata.
        """
        slots = self._slots_per_gpu
        active_mask = torch.zeros(slots, dtype=torch.bool, device=self._device)
        max_len = 1
        pos_ndim = 3
        for slot, item in per_slot_inputs.items():
            active_mask[int(slot)] = True
            max_len = max(max_len, int(item["input_ids"].shape[1]))
            pos = item["position_ids"]
            if pos is not None:
                pos_ndim = int(pos.ndim)

        pad_id = int(self._tokenizer.pad_token_id or 0)
        input_ids = torch.full(
            (slots, max_len),
            pad_id,
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros(
            (slots, max_len),
            dtype=torch.long,
            device=self._device,
        )
        if pos_ndim == 3:
            position_ids = torch.zeros(
                (3, slots, max_len),
                dtype=torch.long,
                device=self._device,
            )
        else:
            position_ids = torch.zeros(
                (slots, max_len),
                dtype=torch.long,
                device=self._device,
            )

        pixel_chunks: list[torch.Tensor] = []
        grid_chunks: list[torch.Tensor] = []
        # Qwen's batched video processor consumes video features in the same
        # order as video tokens appear when scanning the batch rows. The rows
        # below are indexed by streaming slot, so multimodal payloads must be
        # concatenated in ascending slot order, not in async arrival order.
        for slot in sorted(per_slot_inputs):
            item = per_slot_inputs[slot]
            slot = int(slot)
            ids = item["input_ids"].to(self._device)
            attn = item["attention_mask"].to(self._device)
            pos = item["position_ids"].to(self._device)
            length = int(ids.shape[1])
            input_ids[slot, :length] = ids[0]
            attention_mask[slot, :length] = attn[0]
            if pos.ndim == 3:
                position_ids[:, slot : slot + 1, :length] = pos
            else:
                position_ids[slot : slot + 1, :length] = pos
            if item.get("pixel_values_videos") is not None:
                pixel_chunks.append(item["pixel_values_videos"].to(self._device))
            if item.get("video_grid_thw") is not None:
                grid_chunks.append(item["video_grid_thw"].to(self._device))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values_videos": (
                torch.cat(pixel_chunks, dim=0) if pixel_chunks else None
            ),
            "video_grid_thw": (
                torch.cat(grid_chunks, dim=0) if grid_chunks else None
            ),
        }, active_mask

    async def _acquire_slot(
        self,
        request_id: str,
        *,
        reset: bool,
        isolated: bool,
    ) -> tuple[int, bool]:
        async with self._state_lock:
            slot = self._request_to_slot.get(request_id)
            new_session = slot is None
            if slot is None:
                free = [i for i, rid in enumerate(self._slot_to_request) if rid is None]
                if not free:
                    raise RuntimeError(
                        "StreamingRolloutServer has no free KV slot; "
                        "increase load-balancer capacity only with matching slots_per_gpu."
                    )
                slot = free[0]
                self._request_to_slot[request_id] = slot
                self._slot_to_request[slot] = request_id
            if reset or isolated or new_session:
                engine = self._build_engine()
                engine.reset_slots([slot])
                self._last_turn_was_post_recall[slot] = False
            return slot, new_session

    async def _release_slot(self, request_id: str) -> None:
        async with self._state_lock:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is None:
                return
            self._slot_to_request[slot] = None
            self._last_turn_was_post_recall[slot] = False
            if self._engine is not None:
                self._engine.reset_slots([slot])

    async def _schedule_turn(self, payload: dict[str, Any]) -> TokenOutput:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        payload["future"] = fut
        async with self._state_lock:
            self._pending_turns.append(payload)
            if self._pending_flush_task is None or self._pending_flush_task.done():
                self._pending_flush_task = asyncio.create_task(self._flush_pending_turns())
        return await fut

    async def _flush_pending_turns(self) -> None:
        delay = float(os.environ.get("THINKSTREAM_STREAMING_BATCH_DELAY_MS", "2") or 2) / 1000.0
        if delay > 0:
            await asyncio.sleep(delay)
        async with self._state_lock:
            if not self._pending_turns:
                return
            used_slots: set[int] = set()
            batch: list[dict[str, Any]] = []
            remaining: list[dict[str, Any]] = []
            batch_key = None
            for item in self._pending_turns:
                slot = int(item["slot"])
                sp = dict(item.get("sampling_params") or {})
                item_key = (
                    str(item.get("turn_kind") or ""),
                    int(sp.get("max_tokens") or sp.get("max_new_tokens") or self.config.response_length),
                    str(sp.get("recall_kv_policy") or ""),
                    bool(sp.get("delete_previous_recall_toolcall_kv", False)),
                )
                if batch_key is None:
                    batch_key = item_key
                if slot in used_slots or len(batch) >= self._slots_per_gpu or item_key != batch_key:
                    remaining.append(item)
                    continue
                used_slots.add(slot)
                batch.append(item)
            self._pending_turns = remaining
            if self._pending_turns:
                self._pending_flush_task = asyncio.create_task(self._flush_pending_turns())
            else:
                self._pending_flush_task = None
        await self._run_turn_batch(batch)

    async def _run_turn_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        async with self._generate_lock:
            try:
                await self._run_turn_batch_inner(batch)
            except Exception as exc:
                for item in batch:
                    fut = item.get("future")
                    if fut is not None and not fut.done():
                        fut.set_exception(exc)

    async def _run_turn_batch_inner(self, batch: list[dict[str, Any]]) -> None:
            turn_t0 = time.perf_counter()
            t0 = time.perf_counter()
            engine = self._build_engine()
            build_engine_sec = time.perf_counter() - t0
            per_slot_inputs: dict[int, dict[str, torch.Tensor | None]] = {}
            prepare_start = time.perf_counter()
            for item in batch:
                per_slot_inputs[int(item["slot"])] = self._prepare_inputs(
                    item["token_ids"],
                    image_data=item["image_data"],
                    video_data=item["video_data"],
                    turn_kind=item["turn_kind"],
                    chunk_idx=item["chunk_idx"],
                )
            inputs, active_mask = self._pad_prepared_inputs(per_slot_inputs)
            prepare_inputs_sec = time.perf_counter() - prepare_start

            sp0 = dict(batch[0]["sampling_params"] or {})
            max_tokens = int(sp0.get("max_tokens") or sp0.get("max_new_tokens") or self.config.response_length)
            top_k_value = sp0.get("top_k", self.config.top_k)
            raw_top_k = int(self.config.top_k if top_k_value is None else top_k_value)
            if raw_top_k <= 0:
                raw_top_k = int(os.environ.get("THINKSTREAM_ROLLOUT_DEFAULT_TOP_K", "50") or 50)
            safe_top_k = max(1, raw_top_k)
            t0 = time.perf_counter()
            result = engine.generate(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values_videos=inputs["pixel_values_videos"],
                video_grid_thw=inputs["video_grid_thw"],
                num_generations=1,
                max_new_tokens=max_tokens,
                top_k=safe_top_k,
                top_p=float(sp0.get("top_p", self.config.top_p)),
                temperature=float(sp0.get("temperature", self.config.temperature)),
                repetition_penalty=float(sp0.get("repetition_penalty", 1.0)),
                return_log_probs=True,
                turn_kind=batch[0]["turn_kind"],
                recall_kv_policy=sp0.get("recall_kv_policy"),
                delete_previous_assistant_kv=bool(sp0.get("delete_previous_recall_toolcall_kv", False)),
                active_mask=active_mask,
            )
            generate_sec = time.perf_counter() - t0
            tokens_list, log_probs_list = result
            post_start = time.perf_counter()
            total_sec = time.perf_counter() - turn_t0

            for item in batch:
                slot = int(item["slot"])
                token_tensor = tokens_list[slot].detach().to("cpu")
                log_prob_tensor = log_probs_list[slot].detach().to("cpu")
                token_out = token_tensor.tolist()
                stop_reason = "eos"
                if not token_out:
                    stop_reason = "empty"
                elif len(token_out) >= max_tokens and token_out[-1] != int(engine.primary_eos_token_id):
                    stop_reason = "length"
                elif token_out[-1] != int(engine.primary_eos_token_id):
                    stop_reason = "other"
                timing_payload = {
                    "event": "streaming_turn_timing",
                    "request_id": item["request_id"],
                    "replica_rank": self.replica_rank,
                    "slot": slot,
                    "batch_active": len(batch),
                    "turn_kind": item["turn_kind"],
                    "chunk_idx": item["chunk_idx"],
                    "stream_reset_before": item["stream_reset_before"],
                    "stream_isolated_turn": item["stream_isolated_turn"],
                    "new_session": item["new_session"],
                    "token_len": len(item["token_ids"]),
                    "new_prompt_len": len(item["new_prompt_ids"] or []),
                    "video_blocks": len(item["video_data"] or []),
                    "generated_tokens": len(token_out),
                    "stop_reason": stop_reason,
                    "max_tokens": max_tokens,
                    "build_engine_sec": build_engine_sec,
                    "prepare_inputs_sec": prepare_inputs_sec,
                    "generate_sec": generate_sec,
                    "postprocess_sec": time.perf_counter() - post_start,
                    "total_sec": total_sec,
                }
                self._timing_jsonl(timing_payload)
                if item["trace_enabled"]:
                    try:
                        cache_len_after = int(engine.decoder.cache_seqlens[0, slot].item())
                    except Exception:
                        cache_len_after = None
                    self._trace_jsonl({
                        "event": "streaming_turn",
                        "request_id": item["request_id"],
                        "replica_rank": self.replica_rank,
                        "slot": slot,
                        "cuda_visible_devices": self.cuda_visible_devices,
                        "turn_kind": item["turn_kind"],
                        "chunk_idx": item["chunk_idx"],
                        "stream_reset_before": item["stream_reset_before"],
                        "stream_isolated_turn": item["stream_isolated_turn"],
                        "new_session": item["new_session"],
                        "token_len": len(item["token_ids"]),
                        "new_prompt_len": len(item["new_prompt_ids"] or []),
                        "delta_flags": item["delta_flags"],
                        "video_tokens": int((per_slot_inputs[slot]["input_ids"] == int(getattr(self._processor, "video_token_id", -1))).sum().item()),
                        "video_blocks": len(item["video_data"] or []),
                        "video_payload": self._video_payload_summary(item["video_data"]),
                        "cache_len_before": item["cache_len_before"],
                        "cache_len_after": cache_len_after,
                        "prompt_delta_text": item["decoded_delta_for_debug"][:12000],
                        "output_text": self._tokenizer.decode(token_out, skip_special_tokens=False),
                        "stop_reason": stop_reason,
                        "max_tokens": max_tokens,
                        "timing": timing_payload,
                    })
                self._last_turn_was_post_recall[slot] = (
                    str(item["turn_kind"] or "").strip().lower() == "post_recall"
                )
                if item["stream_isolated_turn"]:
                    engine.reset_slots([slot])
                    await self._release_slot(item["request_id"])
                item["future"].set_result(TokenOutput(
                    token_ids=token_out,
                    log_probs=log_prob_tensor.tolist(),
                    stop_reason=stop_reason,
                    num_preempted=0,
                    extra_fields={"global_steps": self._global_steps},
                ))

    async def generate_streaming_turn(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        new_prompt_ids: Optional[list[int]] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        turn_kind: str = "",
        chunk_idx: Optional[int] = None,
        stream_reset_before: bool = False,
        stream_isolated_turn: bool = False,
        **kwargs: Any,
    ) -> TokenOutput:
        del kwargs
        slot, new_session = await self._acquire_slot(
            request_id,
            reset=bool(stream_reset_before),
            isolated=bool(stream_isolated_turn),
        )
        token_ids = (
            list(prompt_ids)
            if new_session or stream_reset_before or stream_isolated_turn
            else list(new_prompt_ids or prompt_ids)
        )
        sp = dict(sampling_params or {})
        debug_mode = str(os.environ.get("THINKSTREAM_ROLLOUT_DEBUG", "")).strip().lower()
        trace_enabled = bool(os.environ.get("THINKSTREAM_ROLLOUT_TRACE_JSONL")) or debug_mode in {
            "trace",
            "verbose",
            "2",
        }
        decoded_delta_for_debug = ""
        delta_flags: list[str] = []
        strict_delta = str(
            os.environ.get("THINKSTREAM_ROLLOUT_STRICT_DELTA", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if trace_enabled or debug_mode in {"1", "true", "yes", "on"} or strict_delta:
            try:
                decoded_delta_for_debug = self._tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                )
            except Exception:
                decoded_delta_for_debug = ""
            if not (new_session or stream_reset_before or stream_isolated_turn):
                delta_flags = _inspect_streaming_delta_text(
                    decoded_delta_for_debug,
                    turn_kind=turn_kind,
                    after_post_recall=self._last_turn_was_post_recall[slot],
                )
            if delta_flags:
                message = (
                    "streaming rollout delta contamination: "
                    f"request={request_id} slot={slot} kind={turn_kind} "
                    f"chunk={chunk_idx} flags={delta_flags}"
                )
                if strict_delta:
                    raise RuntimeError(message)
                logger.warning(message)
        cache_len_before = None
        if trace_enabled and self._engine is not None:
            try:
                cache_len_before = int(self._engine.decoder.cache_seqlens[0, slot].item())
            except Exception:
                cache_len_before = None
        return await self._schedule_turn({
            "request_id": request_id,
            "slot": slot,
            "prompt_ids": list(prompt_ids),
            "new_prompt_ids": list(new_prompt_ids or []),
            "token_ids": token_ids,
            "sampling_params": sp,
            "image_data": image_data,
            "video_data": video_data,
            "turn_kind": turn_kind,
            "chunk_idx": chunk_idx,
            "stream_reset_before": bool(stream_reset_before),
            "stream_isolated_turn": bool(stream_isolated_turn),
            "new_session": bool(new_session),
            "trace_enabled": bool(trace_enabled),
            "decoded_delta_for_debug": decoded_delta_for_debug,
            "delta_flags": delta_flags,
            "cache_len_before": cache_len_before,
        })

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs):
        return await self.generate_streaming_turn(
            request_id=request_id,
            prompt_ids=prompt_ids,
            new_prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            stream_reset_before=True,
            **kwargs,
        )

    async def release_streaming_session(self, request_id: str):
        await self._release_slot(request_id)
        return True

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        del reset_prefix_cache
        await self.clear_kv_cache()
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        return True

    async def wait_for_requests_to_drain(self):
        return True

    async def start_profile(self, **kwargs):
        del kwargs
        return True

    async def stop_profile(self):
        return True

    async def set_global_steps(self, global_steps: int):
        self._global_steps = int(global_steps)
        return True


class StreamingReplica(RolloutReplica):
    """One true-KV HF rollout server per GPU."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model)
        if self.world_size != 1:
            raise ValueError(
                "StreamingReplica requires exactly one GPU per replica. "
                "Set rollout.tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=1."
            )
        self.server_class = ray.remote(StreamingRolloutServer)

    async def launch_servers(self):
        assert len(self.workers) == 1, f"streaming rollout expects one worker, got {len(self.workers)}"
        worker_info = await self.workers[0].__ray_call__.remote(
            lambda self: (
                ray.get_runtime_context().get_node_id(),
                ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
            )
        )
        node_id, cuda_visible_device = worker_info
        visible = str(cuda_visible_device)
        env_vars = _build_streaming_server_env(visible)
        name = f"streaming_server_{self.replica_rank}_0"
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=name,
            max_concurrency=32,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=1,
            nnodes=1,
            cuda_visible_devices=visible,
        )
        self.servers.append(server)
        self._server_handle = server
        self._server_address = name

    async def sleep(self):
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])
        return {"aborted_count": sum(r.get("aborted_count", 0) for r in results), "request_ids": []}

    async def resume_generation(self):
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])
