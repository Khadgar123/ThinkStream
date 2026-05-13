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
import logging
import os
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
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
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
        self._active_request_id: str | None = None
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
            batch_size=1,
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
            self._active_request_id = None
            if self._model is not None:
                self._model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True

    async def clear_kv_cache(self):
        async with self._lock:
            if self._engine is not None:
                self._engine.reset()
            self._active_request_id = None
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
            self._active_request_id = None
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
        del chunk_idx, kwargs
        async with self._lock:
            engine = self._build_engine()
            new_session = self._active_request_id != request_id
            if new_session or stream_reset_before or stream_isolated_turn:
                engine.reset()
                self._active_request_id = request_id
                token_ids = list(prompt_ids)
            else:
                token_ids = list(new_prompt_ids or prompt_ids)

            inputs = self._prepare_inputs(
                token_ids,
                image_data=image_data,
                video_data=video_data,
            )
            sp = dict(sampling_params or {})
            max_tokens = int(sp.get("max_tokens") or sp.get("max_new_tokens") or self.config.response_length)
            result = engine.generate(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values_videos=inputs["pixel_values_videos"],
                video_grid_thw=inputs["video_grid_thw"],
                num_generations=1,
                max_new_tokens=max_tokens,
                top_k=max(0, int(sp.get("top_k", self.config.top_k))),
                top_p=float(sp.get("top_p", self.config.top_p)),
                temperature=float(sp.get("temperature", self.config.temperature)),
                repetition_penalty=float(sp.get("repetition_penalty", 1.0)),
                return_log_probs=True,
                turn_kind=turn_kind,
                recall_kv_policy=sp.get("recall_kv_policy"),
                delete_previous_assistant_kv=bool(sp.get("delete_previous_recall_toolcall_kv", False)),
            )
            tokens_list, log_probs_list = result
            token_tensor = tokens_list[0].detach().to("cpu")
            log_prob_tensor = log_probs_list[0].detach().to("cpu")
            token_out = token_tensor.tolist()
            stop_reason = "eos"
            if not token_out:
                stop_reason = "empty"
            elif len(token_out) >= max_tokens and token_out[-1] != int(engine.primary_eos_token_id):
                stop_reason = "length"
            elif token_out[-1] != int(engine.primary_eos_token_id):
                stop_reason = "other"

            if stream_isolated_turn:
                engine.reset()
                self._active_request_id = None

            return TokenOutput(
                token_ids=token_out,
                log_probs=log_prob_tensor.tolist(),
                stop_reason=stop_reason,
                num_preempted=0,
                extra_fields={"global_steps": self._global_steps},
            )

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
        async with self._lock:
            if self._active_request_id == request_id and self._engine is not None:
                self._engine.reset()
            if self._active_request_id == request_id:
                self._active_request_id = None
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
        name = f"streaming_server_{self.replica_rank}_0"
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                }
            },
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
            cuda_visible_devices=str(cuda_visible_device),
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
