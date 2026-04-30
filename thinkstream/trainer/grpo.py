import json
import os
import re
import types
import math
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
import torch

# deepspeed / transformers / slyme are only required for the training nodes
# (rollout / loss / model loading). The pure-tensor reward + advantage helpers
# (`_gdpo_per_reward_group_norm`, `_compute_*_reward`) must remain importable
# without these heavy deps so unit tests can exercise them on CPU-only envs.
try:
    import deepspeed                                       # noqa: F401
    from transformers import PreTrainedModel               # noqa: F401
    from slyme.context import Context, Ref                 # noqa: F401
    from slyme.node import Node, node, wrapper, Auto, expression  # noqa: F401
    from deepslyme.utils.accelerator import empty_cache    # noqa: F401
    _SLYME_AVAILABLE = True
except ImportError:
    _SLYME_AVAILABLE = False
    # Stubs so module-level @node / @wrapper / @expression decorators don't
    # blow up at import time. These will fail loudly if anyone tries to
    # actually invoke a training node without slyme installed.
    PreTrainedModel = object  # type: ignore[assignment,misc]
    Context = object          # type: ignore[assignment,misc]
    Node = object             # type: ignore[assignment,misc]

    def _identity_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap

    node = wrapper = expression = _identity_decorator  # type: ignore[assignment]

    def Ref(*_args, **_kwargs):  # type: ignore[no-redef]
        return None

    class _AutoMeta(type):
        def __getitem__(cls, _item):
            return cls
    class Auto(metaclass=_AutoMeta):  # type: ignore[no-redef]
        pass

    def empty_cache():  # type: ignore[no-redef]
        pass

# Import thinkstream specifics
from thinkstream.model.inference import (
    StreamingWindowInferenceEngine,
    streaming_video_chat,
    think_budget_sample,
)
from thinkstream.data.stream_data_processor import (
    QWEN_TEMPLATE_WO_SYSTEM,
    _make_abs_paths,
    build_video_meta,
    process_messages_to_model_inputs,
    pad_and_cat,
    find_assistant_spans,
    compute_position_ids,
    make_raw_data_module,
)
from thinkstream.data.agent_protocol import SYSTEM_PROMPT_V12
from thinkstream.model.patch import build_video_block_mask
from thinkstream.model import MODEL_CLS, get_text_config, DEFAULT_VIDEO_FLEX_WINDOW_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit logging (env-controlled, no slyme node-signature changes)
# ---------------------------------------------------------------------------
# Enable by setting THINKSTREAM_AUDIT_DIR=<path>; falls back to
# THINKSTREAM_OUTPUT_DIR/audit when only the output dir is exported.
from thinkstream.trainer.audit import AuditWriter, resolve_audit_dir

_GRPO_STEP_WRITER: Optional[AuditWriter] = None
_GRPO_SAMPLE_WRITER: Optional[AuditWriter] = None
_GRPO_STEP_COUNTER = 0


def _grpo_audit_writers():
    """Lazy-init audit writers from env vars."""
    global _GRPO_STEP_WRITER, _GRPO_SAMPLE_WRITER
    if _GRPO_STEP_WRITER is not None or _GRPO_SAMPLE_WRITER is not None:
        return _GRPO_STEP_WRITER, _GRPO_SAMPLE_WRITER
    audit_dir = resolve_audit_dir(
        os.environ.get("THINKSTREAM_AUDIT_DIR"),
        os.environ.get("THINKSTREAM_OUTPUT_DIR"),
    )
    if audit_dir is None:
        return None, None
    _GRPO_STEP_WRITER = AuditWriter(audit_dir / "grpo_step.jsonl")
    _GRPO_SAMPLE_WRITER = AuditWriter(audit_dir / "grpo_sample.jsonl")
    return _GRPO_STEP_WRITER, _GRPO_SAMPLE_WRITER

def _collect_think_lengths(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> List[int]:
    """Collect token lengths of <think>...</think> spans for one (chunk_results, gen_idx).
    One length per chunk that contains a think block.

    Diagnostics/audit only — does not feed any reward.
    """
    lengths: List[int] = []
    for cr in chunk_results:
        gen_tokens_list = cr.get("generated_tokens", [])
        if gen_idx >= len(gen_tokens_list):
            continue
        gen_tokens = gen_tokens_list[gen_idx]
        if isinstance(gen_tokens, torch.Tensor):
            gen_tokens = gen_tokens.tolist()
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            think_part = "<think>" + m.group(1) + "</think>"
            think_ids = tokenizer.encode(think_part, add_special_tokens=False)
            lengths.append(len(think_ids))
    return lengths


# ---------------------------------------------------------------------------
# Nodes for GRPO adapted for DeepSlyme
# ---------------------------------------------------------------------------


@node
def load_grpo_models(
    ctx: Context,
    /,
    *,
    model_name_or_path: Auto[str],
    model_cache_dir: Auto[str],
    bf16: Auto[bool],
    reference_model: Ref[PreTrainedModel],
    model: Ref[PreTrainedModel],
    model_type: Auto[str],
    model_for_generation: Ref[Any],
    deepspeed_config: Auto[dict],
) -> Context:
    """
    Load the policy model with DeepSpeed config, and clean CPU models for generation and reference.
    """
    from transformers.integrations.deepspeed import (
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )
    from transformers.integrations import HfDeepSpeedConfig

    if model_type not in MODEL_CLS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    cls = MODEL_CLS[model_type]

    dtype = torch.bfloat16 if bf16 else None
    attn_implementation = "flash_attention_2"
    vision_attn_implementation = "flash_attention_2"

    # 1. Load Policy Model with DeepSpeed context
    hf_ds_config = HfDeepSpeedConfig(deepspeed_config)
    set_hf_deepspeed_config(hf_ds_config)
    try:
        policy_model = cls.from_pretrained(
            model_name_or_path,
            cache_dir=model_cache_dir,
            attn_implementation="streaming_attention",
            dtype=dtype,
        )
        policy_model.config.vision_config._attn_implementation = (
            vision_attn_implementation
        )
    finally:
        unset_hf_deepspeed_config()
        if "HF_DEEPSPEED_CONFIG" in os.environ:
            del os.environ["HF_DEEPSPEED_CONFIG"]

    # 2. Load Generation Model (clean, CPU)
    logger.info("Loading model_for_generation (clean, CPU)...")
    gen_model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    gen_model.config.vision_config._attn_implementation = vision_attn_implementation
    # flash_attention_2_infer requires attn_cache_seqlens which Qwen3-VL forward
    # does not pass in standard transformers generate. Fall back to regular
    # flash_attention_2 for CPU-based generation model used in GRPO rollout.
    gen_model.config.text_config._attn_implementation = "flash_attention_2"
    gen_model.eval()
    gen_model.requires_grad_(False)
    gen_model.to("cpu")

    # 3. Load Reference Model (clean, frozen, CPU)
    logger.info("Loading reference_model (clean, frozen, CPU)...")
    ref_model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    ref_model.config.vision_config._attn_implementation = vision_attn_implementation
    ref_model.config.text_config._attn_implementation = "streaming_attention"
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to("cpu")

    return ctx.update(
        {
            model: policy_model,
            model_for_generation: gen_model,
            reference_model: ref_model,
        }
    )


@wrapper
def unwrap_model_for_generation(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    model_for_training: Auto[Any],
    inference_engine: Ref[Any],
    model_for_generation: Auto[Any],
    device: Auto[torch.device],
    state_global_step: Auto[int],
    rollout_last_sync_step: Ref[int],
    rollout_sync_per_step: Auto[int] = 1,
) -> Context:
    """Sync weights from ZeRO-3 model_for_training to CPU model_for_generation before rollout."""
    import deepspeed  # heavy dep; imported here so unit tests don't need it
    # With raw DeepSpeed, unwrapped model is accessed via .module
    unwrapped_model = (
        model_for_training.module
        if hasattr(model_for_training, "module")
        else model_for_training
    )

    is_zero3 = (
        hasattr(model_for_training, "zero_optimization_stage")
        and model_for_training.zero_optimization_stage() == 3
    )
    model_for_generation.to(device)
    rollout_last_sync_step_ = ctx.get(rollout_last_sync_step, None)
    if (
        rollout_last_sync_step_ is None
        or state_global_step - rollout_last_sync_step_ >= rollout_sync_per_step
    ):

        def _sync_params():
            train_params = dict(unwrapped_model.named_parameters())
            train_buffers = dict(unwrapped_model.named_buffers())
            with torch.no_grad():
                for name, gen_p in model_for_generation.named_parameters():
                    if name in train_params:
                        gen_p.data.copy_(train_params[name].data)
                    else:
                        logger.warning(
                            "Parameter %s not found in training model.", name
                        )
                for name, gen_b in model_for_generation.named_buffers():
                    if name in train_buffers:
                        gen_b.data.copy_(train_buffers[name].data)

        # NOTE: sync params
        if is_zero3:
            with deepspeed.zero.GatheredParameters(list(unwrapped_model.parameters())):
                _sync_params()
        else:
            _sync_params()
        ctx = ctx.set(rollout_last_sync_step, state_global_step)

    try:
        ctx = call_next(ctx)
    finally:
        model_for_generation.to("cpu")
        ctx = ctx.set(inference_engine, None)
        empty_cache()
    return ctx


@node
def rollout(
    ctx: Context,
    /,
    *,
    step_inputs: Auto[Dict[str, Any]],
    model_for_generation: Auto[Any],
    processor: Auto[Any],
    tokenizer: Auto[Any],
    group_size: Auto[int],
    rollout_data: Ref[Dict[str, Any]],
    inference_engine: Ref[Any],
    model_type: Auto[str],
    rollout_max_new_tokens: Auto[int],
    rollout_max_think_tokens: Auto[int],
    rollout_temperature: Auto[float],
    rollout_top_k: Auto[int],
    rollout_top_p: Auto[float],
    rollout_fpc: Auto[float],
    rollout_max_chunks: Auto[int],
    rollout_min_pixels: Auto[int],
    rollout_max_pixels: Auto[int],
    rollout_extra_chunks: Auto[int] = 5,
    use_vllm_rollout: Auto[bool] = False,
    vllm_rollout_frames_root: Auto[Optional[str]] = None,
    vllm_rollout_video_root: Auto[Optional[str]] = None,
) -> Context:
    """
    GRPO rollout using streaming video inference.

    For each raw sample in the batch, generates G=group_size completions
    chunk-by-chunk and stores per-sample results (generated tokens, chunk
    metadata, raw sample) in ``rollout_data`` for downstream reward
    computation and loss calculation.

    Two backends share an identical output contract — reward / advantage /
    loss code below is unchanged regardless of which is used:

      use_vllm_rollout=False (default, legacy):
        StreamingAgentLoop + HF model.generate per chunk per gen.
        N×G sequential generates; safe baseline used in audit logs.

      use_vllm_rollout=True (v11.3, --use_vllm_rollout):
        streaming_vllm_rollout — chunk-lockstep cross-(sample×gen) batch
        through one vLLM call per chunk. Same MemoryState class, same
        build_single_step_messages, so the prompt format is byte-identical
        to SFT and to the legacy backend. Expected 5-10× rollout speedup
        on N×G ≥ 16 batches; on small batches the speedup is smaller but
        never negative.

    Per-chunk shape returned (both backends):
      {chunk_idx, window_start, window_end,
       generated_tokens: List[Tensor]  # len = G,
       memory_token_count: List[int]   # len = G,
       compress_budget:   List[int]    # len = G,
       recall_returned_chunks: List[List[int]]  # len = G}

    NOTE: This node should be wrapped with ``unwrap_model_for_generation``
    which handles ZeRO-3 parameter gathering and inference engine cleanup.
    """
    from thinkstream.model.agent_loop import StreamingAgentLoop

    all_rollout_results: List[Dict[str, Any]] = []
    model_for_generation.eval()

    # ─── v11.3 vLLM rollout backend ───
    # Same prompt format (build_single_step_messages), same MemoryState
    # advancement, same per-chunk early-stop semantics — only the inference
    # engine differs. Reward / advantage / loss code below sees an identical
    # contract.
    if use_vllm_rollout:
        try:
            from thinkstream.eval.streaming_vllm import streaming_vllm_rollout
        except ImportError as e:
            raise RuntimeError(
                "use_vllm_rollout=True but streaming_vllm not importable: "
                f"{e}. Install vllm + qwen_vl_utils or fall back to "
                "use_vllm_rollout=False."
            ) from e
        # `inference_engine` is the vLLM LLM handle owned by the trainer
        # (unwrap_model_for_generation injects it on ZeRO-3 unwrap).
        all_rollout_results = streaming_vllm_rollout(
            step_inputs,
            llm=inference_engine,
            processor=processor,
            tokenizer=tokenizer,
            group_size=group_size,
            max_new_tokens=rollout_max_new_tokens,
            rollout_max_chunks=rollout_max_chunks,
            rollout_extra_chunks=rollout_extra_chunks,
            min_pixels=rollout_min_pixels,
            max_pixels=rollout_max_pixels,
            temperature=rollout_temperature,
            top_p=rollout_top_p,
            top_k=rollout_top_k,
            frames_root=vllm_rollout_frames_root,
            video_root=vllm_rollout_video_root,
        )
        model_for_generation.train()
        return ctx.set(rollout_data, all_rollout_results)

    # ─── Legacy HF rollout backend (default, kept for parity / fallback) ───

    def _generate_fn(messages, processor, max_new_tokens=256, **kwargs):
        """Wrap model generation for StreamingAgentLoop."""
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(model_for_generation.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model_for_generation.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=rollout_temperature,
                top_k=rollout_top_k,
                top_p=rollout_top_p,
                do_sample=True,
            )
        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)

    for raw_sample in step_inputs:
        data_path = raw_sample.get("data_path", "")
        video_path = raw_sample.get("video_path", "")
        abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

        # v12.4 trajectory format detection (from pass4 *_trajectories.jsonl).
        # When the row carries `questions` (one entry per card) and
        # `gold_action_per_chunk` (placement plan), it's a multi-question
        # trajectory rollout — extract a representative ask_chunk for
        # rollout-length budgeting and inject each question at its own
        # ask_chunk during the loop.step() calls.
        _is_traj_sample = (
            isinstance(raw_sample.get("questions"), list)
            and isinstance(raw_sample.get("gold_action_per_chunk"), dict)
        )

        # Extract task info
        metadata = raw_sample.get("metadata", {})

        if _is_traj_sample:
            # Build a chunk → user_question map from each card's first ask_chunk.
            # Multi-response cards (F7/M1, ~8.9% of questions) repeat the same
            # question at each of their ask_chunks — we honor that by emitting
            # the same string at every chunk in the card's ask_chunks list.
            _question_at_chunk: Dict[int, str] = {}
            _all_ask_chunks: List[int] = []
            for q in raw_sample["questions"]:
                q_text = q.get("question") or q.get("gold_answer", "")
                # Note: pass3a stores the question in raw_sample.input.user_input
                # for SFT, but for trajectory rows we synthesize from the card.
                # Real ingestion via streaming_vllm uses metadata so this
                # placeholder is fine for token-counting / message construction.
                for ac in q.get("ask_chunks") or []:
                    _question_at_chunk[int(ac)] = q_text
                    _all_ask_chunks.append(int(ac))
            ask_chunk = max(_all_ask_chunks) if _all_ask_chunks else (
                rollout_max_chunks - 1
            )
            # No single user_question for trajectory rollouts; per-chunk.
            user_question = None
        else:
            ask_chunk = raw_sample.get("chunk_idx", rollout_max_chunks - 1)
            _question_at_chunk = {}

            # Extract user question (new format: input.user_input;
            # legacy: messages/conversations)
            user_question = None
            if "input" in raw_sample and raw_sample["input"].get("user_input"):
                user_question = raw_sample["input"]["user_input"]
            elif "messages" in raw_sample:
                for msg in raw_sample["messages"]:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text = item.get("text", "")
                                    if "<user_input>" in text:
                                        m = re.search(r'<user_input>(.*?)</user_input>', text)
                                        if m:
                                            user_question = m.group(1)
            elif "conversations" in raw_sample:
                user_convs = [c for c in raw_sample["conversations"] if c.get("role") == "user"]
                if user_convs:
                    user_question = user_convs[0].get("content", "")
            if user_question:
                _question_at_chunk[int(ask_chunk)] = user_question

        # Run G independent rollouts
        per_gen_results: List[List[Dict]] = []
        for g in range(group_size):
            loop = StreamingAgentLoop(
                generate_fn=_generate_fn,
                tokenizer=tokenizer,
                processor=processor,
                model_type=model_type,
                min_pixels=rollout_min_pixels,
                max_pixels=rollout_max_pixels,
                max_new_tokens=rollout_max_new_tokens,
                # v12.5 fix (2026-04-30): plumb frames_root/video_root so the
                # HF rollout backend uses pre-extracted JPEG frames instead of
                # in-line video decode (read_video_torchcodec). Same args
                # already passed to vLLM backend at line 382-383; this fixes
                # the 713s/step rollout slowness reported in user's audit.
                # When frames_root is None (config not set), falls back to
                # video-decode path for backward compat.
                frames_root=vllm_rollout_frames_root,
                video_root=vllm_rollout_video_root,
            )

            chunk_results_g: List[Dict[str, Any]] = []
            num_chunks = min(ask_chunk + 5, rollout_max_chunks)  # run past last ask
            for chunk_idx in range(num_chunks):
                # v12.4 trajectory: question may fire at any chunk that's in
                # _question_at_chunk (one card → ≥1 ask_chunks). Single-question
                # path falls through with question only at the lone ask_chunk.
                q = _question_at_chunk.get(chunk_idx)
                if q is None and not _is_traj_sample and chunk_idx == ask_chunk:
                    q = user_question
                result = loop.step(
                    chunk_idx=chunk_idx,
                    video_path=abs_video_path,
                    user_question=q,
                )
                # Store result with generated tokens for reward/loss computation
                chunk_results_g.append({
                    "chunk_idx": chunk_idx,
                    "action": result.get("action", "unknown"),
                    "think": result.get("think", ""),
                    "payload": result.get("payload", {}),
                    "raw_output": result.get("raw_output", ""),
                    "generated_tokens": tokenizer.encode(
                        result.get("raw_output", ""), add_special_tokens=False,
                    ),
                    # Post-step memory bookkeeping (used by overflow_pen reward).
                    "memory_token_count": int(result.get("memory_token_count", 0)),
                    "compress_budget": int(
                        result.get("compress_budget", 0)
                    ),
                    # Retriever output (used by recall_quality hit-rate reward).
                    # Empty when the rollout didn't recall this chunk.
                    "recall_returned_chunks": list(
                        result.get("recall_returned_chunks") or []
                    ),
                    "window_start": chunk_idx * 2,
                    "window_end": (chunk_idx + 1) * 2,
                })
                # Early stop if model responded — but NOT for trajectory
                # rollouts where multiple questions are asked at different
                # ask_chunks (we must keep rolling so later questions get
                # their own answer chunk).
                if (result.get("action") == "response"
                        and chunk_idx >= ask_chunk
                        and not _is_traj_sample):
                    break
            per_gen_results.append(chunk_results_g)

        # Merge into the expected format: chunk_results with generated_tokens[G]
        # and per-gen memory_token_count list (used by overflow_pen reward).
        max_chunks_seen = max(len(g) for g in per_gen_results)
        merged_chunk_results = []
        for ci in range(max_chunks_seen):
            merged = {
                "chunk_idx": ci,
                "window_start": ci * 2,
                "window_end": (ci + 1) * 2,
                "generated_tokens": [],
                "memory_token_count": [],   # per-gen post-step memory size
                "compress_budget": [],      # per-gen budget (constant within a run, kept per-gen for symmetry)
                "recall_returned_chunks": [],  # per-gen retriever output
            }
            for g in range(group_size):
                if ci < len(per_gen_results[g]):
                    cr_g = per_gen_results[g][ci]
                    merged["generated_tokens"].append(
                        torch.tensor(cr_g["generated_tokens"])
                    )
                    merged["memory_token_count"].append(cr_g.get("memory_token_count", 0))
                    merged["compress_budget"].append(cr_g.get("compress_budget", 0))
                    merged["recall_returned_chunks"].append(
                        list(cr_g.get("recall_returned_chunks") or [])
                    )
                else:
                    # Pad with empty if this gen finished early
                    merged["generated_tokens"].append(torch.tensor([]))
                    merged["memory_token_count"].append(0)
                    merged["compress_budget"].append(0)
                    merged["recall_returned_chunks"].append([])
            merged_chunk_results.append(merged)

        all_rollout_results.append({
            "raw_sample": raw_sample,
            "chunk_results": merged_chunk_results,
        })

    model_for_generation.train()
    return ctx.set(rollout_data, all_rollout_results)


# v11 GDPO-style design — see docs/streaming_position_encoding.md (RL section)
# and /Users/hzh/.claude/plans/fuzzy-plotting-valiant.md for rationale.
#
# Reward keys + weights + the pure-tensor aggregation algorithm live in
# ``gdpo_advantage.py`` so unit tests can import them without dragging in
# transformers / deepspeed / slyme. We re-export here for backward compat.
from thinkstream.trainer.gdpo_advantage import (
    V12_REWARD_DICT_KEYS,
    V12_DEFAULT_REWARD_WEIGHTS,
    V12_ADVANTAGE_MIX_ALPHA,
    per_reward_group_norm as _gdpo_per_reward_group_norm,
    aggregate_gdpo as _gdpo_aggregate,
    aggregate_grpo as _grpo_aggregate,
    aggregate_advantages as _aggregate_advantages,
)

# v12.0 reward components — pure helpers in trainer/v12_rewards.py so unit
# tests can run on CPU without the model stack. Re-exported here for the
# rollout / GRPO caller convenience.
from thinkstream.trainer.v12_rewards import (
    compute_outcome_reward_v12 as _compute_outcome_reward_v12,
    compute_timing_reward_v12 as _compute_timing_reward_v12,
    compute_format_reward_v12 as _compute_format_reward_v12,
    compute_spam_score_v12 as _compute_spam_score_v12,
    compute_compress_quality_v12 as _compute_compress_quality_v12,
    compute_recall_quality_v12 as _compute_recall_quality_v12,
    compute_silent_quality_v12 as _compute_silent_quality_v12,
    aggregate_v12_advantages as _aggregate_v12_advantages,
    # v12.4 — multi-question trajectory + per-chunk silent_quality
    compute_trajectory_outcome_v12 as _compute_trajectory_outcome_v12,
    compute_per_chunk_silent_quality_v12 as _compute_per_chunk_silent_quality_v12,
)

# v12.2 chunk-level rollout (MemAgent recurrent pattern + ReMemR1 mixed
# advantage). The `aggregate_v12_advantages` re-export above is the v1
# implementation kept for backward compatibility (singleton groups → adv=0);
# `compute_mixed_advantage_v12` here is the line-by-line ReMemR1 port
# (singleton groups → preserve raw signal). Trainer can opt into v2 once
# the streaming rollout is wired through ChunkLevelRolloutLoop.
from thinkstream.trainer.v12_rollout import (
    ChunkLevelRolloutConfig as _V12RolloutConfig,
    ChunkLevelRolloutLoop as _V12ChunkLevelRolloutLoop,
    VideoTrajectoryState as _V12VideoTrajectoryState,
    compute_1d_grpo_advantage as _compute_1d_grpo_advantage_remem,
    compute_mixed_advantage_v12 as _compute_mixed_advantage_v12_remem,
    chunk_results_from_loop_result as _chunk_results_from_loop_result,
    default_v12_update_state as _default_v12_update_state,
)


# ===========================================================================
# v12.0 reward components: see thinkstream/trainer/v12_rewards.py
# Re-imported above for in-grpo callers. Functions defined inline here are
# v11 components only.
# ===========================================================================


def _calc_rewards_v12(
    rollout_data: List[Dict[str, Any]],
    *,
    group_size: int,
    tokenizer: Any,
    time_reward_window: int,
    time_reward_slack: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """v12.0 reward computation — 5 components per V12_REWARD_DICT_KEYS.

    Mirrors v11's calc_rewards interface so the trainer can dispatch on
    protocol_version. Returns (rewards [B], rewards_dict {key: [B]},
    rewards_masks [B, n_rewards]) with column order matching
    V12_REWARD_DICT_KEYS.

    KEY DIFFERENCES from v11 calc_rewards:
      - Parses <answer>/<tool_call> instead of <action>/<response>
        (v12 protocol — see thinkstream.data.agent_protocol)
      - Outcome uses compute_outcome_reward_v12 (anti-hack length cap +
        answer_form-aware strict matching when 'binary'/'multiple_choice'/
        'number'/'short_exact')
      - Timing uses compute_timing_reward_v12 bucketed scheme
        (-1 early hallucination / +1 on-time / +0.5 late_partial / -0.5
        missed) — first such reward in any released streaming-video work
      - Spam is ADDITIVE (negative weight × positive score), NOT
        multiplicative (DeepEyesV2 pattern bug fix)
      - Compress turns score summary quality (range_iou + text_match)
        instead of binary triggered/not — system-triggered means timing
        is not a model decision

    Multi-level GRPO advantage aggregation happens downstream in
    aggregate_v12_advantages, not here.
    """
    from thinkstream.data.agent_protocol import parse_agent_output_v12

    weights = V12_DEFAULT_REWARD_WEIGHTS
    keys = list(V12_REWARD_DICT_KEYS)
    all_rewards = {k: [] for k in keys}
    all_masks = {k: [] for k in keys}

    for sample_data in rollout_data:
        raw_sample = sample_data["raw_sample"]
        chunk_results: List[Dict[str, Any]] = sample_data["chunk_results"]
        metadata = raw_sample.get("metadata", {})
        gt_answer = metadata.get("gold_answer", "")
        answer_form = metadata.get("answer_form", "")
        gold_action = (
            metadata.get("gold_action")
            or raw_sample.get("action")
            or ""
        )
        gold_compress_chunks = metadata.get("gold_compress_chunks", [])
        gold_summary_text = metadata.get("gold_summary_text", "")
        gt_chunk_idx = raw_sample.get("chunk_idx")
        # v12.2 — for recall_quality, we need the per-card support_chunks
        # (gold evidence positions, annotated by pass3a) and the retriever's
        # per-chunk returned_chunks for this rollout.
        support_chunks = list(metadata.get("support_chunks") or [])

        for g in range(group_size):
            # Reconstruct each chunk's text + collect tool/answer info
            chunk_texts: List[str] = []
            answer_chunk = None
            n_recall = 0
            n_compress = 0
            final_answer = None
            compress_summary_text = None
            compress_summary_range = None
            # v12.2 recall_quality bookkeeping
            recall_returned_per_call: List[List[int]] = []
            recall_query_text: Optional[str] = None

            for cr in chunk_results:
                gen_tokens = cr.get("generated_tokens", [])
                if g >= len(gen_tokens):
                    continue
                tokens = gen_tokens[g]
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                chunk_texts.append(text)

                parsed = parse_agent_output_v12(text)
                if parsed["kind"] == "answer" and parsed["answer_text"]:
                    if answer_chunk is None:
                        answer_chunk = cr["chunk_idx"]
                        final_answer = parsed["answer_text"]
                elif parsed["kind"] == "recall":
                    n_recall += 1
                    # v12.2: capture query text and retriever output for this
                    # recall event (mirrors v11 grpo.py:1329-1334 logic)
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    if recall_query_text is None:
                        recall_query_text = args.get("query") or ""
                    returned_lists = cr.get("recall_returned_chunks") or []
                    if isinstance(returned_lists, list) and g < len(returned_lists):
                        rl = returned_lists[g]
                        if isinstance(rl, list):
                            recall_returned_per_call.append([int(c) for c in rl])
                elif parsed["kind"] == "compress":
                    n_compress += 1
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    compress_summary_text = args.get("text")
                    compress_summary_range = args.get("time_range")

            # ── outcome ──
            outcome = _compute_outcome_reward_v12(
                final_answer, gt_answer, answer_form=answer_form,
            )
            all_rewards["outcome"].append(outcome)
            # Mask outcome=0 when sample has no gt (e.g., pure silent)
            all_masks["outcome"].append(1.0 if gt_answer else 0.0)

            # ── timing ──
            timing = _compute_timing_reward_v12(
                answer_chunk=answer_chunk,
                visible_start_chunk=gt_chunk_idx,
                visible_end_chunk=(
                    (gt_chunk_idx + time_reward_window)
                    if gt_chunk_idx is not None else None
                ),
            )
            all_rewards["timing"].append(timing)
            all_masks["timing"].append(1.0 if gt_chunk_idx is not None else 0.0)

            # ── format ──
            fmt = _compute_format_reward_v12(chunk_texts)
            all_rewards["format"].append(fmt)
            all_masks["format"].append(1.0)

            # ── spam ──
            spam = _compute_spam_score_v12(
                n_recall_calls=n_recall, n_compress_calls=n_compress,
            )
            all_rewards["spam"].append(spam)
            all_masks["spam"].append(1.0)

            # ── compress_quality / recall_quality DROPPED in v12.3 ──
            # DeepEyesV2 (arXiv:2511.05271) and 2026 NeurIPS/ICLR consensus:
            # tool-specific quality rewards add complexity without matching
            # the generalization gain that pure outcome + GRPO group-norm
            # delivers. support_chunks-based hit_rate also dies on families
            # that lack annotation (CR3/CR6/CR7). The functions remain in
            # v12_rewards.py for legacy callers; we just don't aggregate
            # them. Tool credit flows naturally via outcome propagation.
            #
            # We still capture compress / recall metadata above for telemetry.
            _ = compress_summary_text       # silence linter — future telemetry
            _ = compress_summary_range      # silence linter — future telemetry
            _ = recall_returned_per_call    # silence linter — future telemetry
            _ = recall_query_text           # silence linter — future telemetry
            _ = support_chunks              # silence linter — future telemetry
            _ = gold_summary_text           # silence linter — future telemetry
            _ = gold_compress_chunks        # silence linter — future telemetry

            # ── silent_quality (v12.2 — kept; streaming-specific) ──
            # Closes the two error modes Q3 audit exposed:
            #   silent-when-should-respond  → -0.6
            #   hallucinate-when-should-be-silent → -0.6
            # Always applied (mask=1) — every chunk has a silent/respond
            # decision; the function returns 0.0 for compress/recall_query
            # cases where this signal doesn't apply.
            silent_q = _compute_silent_quality_v12(
                final_answer=final_answer,
                gold_action=gold_action,
                gold_answer=gt_answer,
            )
            all_rewards["silent_quality"].append(silent_q)
            all_masks["silent_quality"].append(1.0)

    # Stack to tensors
    rewards_dict = {
        k: torch.tensor(all_rewards[k], dtype=torch.float)
        for k in keys
    }
    masks_dict = {
        k: torch.tensor(all_masks[k], dtype=torch.float)
        for k in keys
    }

    # Weighted sum (logging-only; downstream multi-level aggregation
    # uses raw + masks)
    rewards = torch.zeros_like(rewards_dict[keys[0]])
    for k in keys:
        rewards = rewards + weights.get(k, 0.0) * rewards_dict[k] * masks_dict[k]

    rewards_masks = torch.stack([masks_dict[k] for k in keys], dim=1)
    return rewards, rewards_dict, rewards_masks


def _calc_rewards_v12_trajectory(
    rollout_data: List[Dict[str, Any]],
    *,
    group_size: int,
    tokenizer: Any,
    answer_window_chunks: int = 5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """v12.4 — trajectory-level reward computation (multi-question + per-chunk).

    Replaces single-question `_calc_rewards_v12` for callers that have
    rolled out a TRAJECTORY (multiple cards, max_per_traj=3 from pass3b).

    Differences from v12.3 _calc_rewards_v12:
      1. **outcome**: averages per-question correctness over ALL cards in
         the trajectory (was: scored ONLY first non-empty answer).
      2. **silent_quality**: scores EACH chunk's silent/respond decision
         against gold_action_per_chunk (was: scored only trajectory-end).
      3. **timing**: averaged over per-question timing scores.
      4. **format / spam**: unchanged (already trajectory-level).
      5. **recall_quality / compress_quality**: still dropped (v12.3 design).

    Input shape:
      rollout_data: list of dicts with:
        {
          "raw_sample": {
            "video_id": str,
            "trajectory_id": str,
            "questions": [{card_id, gold_answer, ask_chunks, ...}, ...],
            "gold_action_per_chunk": {str(chunk_idx): sample_type, ...},
            ...
          },
          "chunk_results": [{"chunk_idx": int, "generated_tokens": [G][...], ...}, ...]
        }

    Output: same shape as `_calc_rewards_v12` —
      (rewards [B], rewards_dict {key: [B]}, rewards_masks [B, num_keys])
      where B = N_trajectories × group_size.
    """
    from thinkstream.data.agent_protocol import parse_agent_output_v12

    weights = V12_DEFAULT_REWARD_WEIGHTS
    keys = list(V12_REWARD_DICT_KEYS)
    all_rewards = {k: [] for k in keys}
    all_masks = {k: [] for k in keys}

    for sample_data in rollout_data:
        raw_sample = sample_data["raw_sample"]
        chunk_results: List[Dict[str, Any]] = sample_data["chunk_results"]
        questions = list(raw_sample.get("questions") or [])
        gold_action_per_chunk = dict(raw_sample.get("gold_action_per_chunk") or {})

        for g in range(group_size):
            # Parse each chunk's output for this rollout group index g.
            chunk_outputs: List[Dict[str, Any]] = []
            chunk_texts: List[str] = []
            n_recall = 0
            n_compress = 0
            for cr in chunk_results:
                gen_tokens = cr.get("generated_tokens", [])
                if g >= len(gen_tokens):
                    continue
                tokens = gen_tokens[g]
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                chunk_texts.append(text)
                parsed = parse_agent_output_v12(text)
                chunk_outputs.append({
                    "chunk_idx": cr.get("chunk_idx"),
                    "kind": parsed.get("kind", "unknown"),
                    "answer_text": parsed.get("answer_text"),
                    "tool_call": parsed.get("tool_call"),
                })
                if parsed.get("kind") == "recall":
                    n_recall += 1
                elif parsed.get("kind") == "compress":
                    n_compress += 1

            # ── outcome (multi-question) ──
            outcome_res = _compute_trajectory_outcome_v12(
                rollout_chunk_outputs=chunk_outputs,
                trajectory_questions=questions,
                answer_window_chunks=answer_window_chunks,
            )
            all_rewards["outcome"].append(outcome_res["outcome"])
            # Mask=0 if trajectory has no questions (base-only)
            all_masks["outcome"].append(1.0 if questions else 0.0)

            # ── timing (per-ask-chunk averaged) ──
            # v12.4 multi-response handling: each ask_chunk gets its own
            # non-overlapping window for timing. Mirrors per-ask scoring
            # in compute_trajectory_outcome_v12 above.
            per_q_timings = []
            by_chunk_idx = {
                out.get("chunk_idx"): out
                for out in chunk_outputs
                if out.get("chunk_idx") is not None
            }
            for q in questions:
                ask_chunks = sorted(q.get("ask_chunks") or [])
                if not ask_chunks:
                    continue
                per_ask_t: List[float] = []
                for i, ask_chunk in enumerate(ask_chunks):
                    if i + 1 < len(ask_chunks):
                        window_end = min(
                            ask_chunks[i + 1] - 1,
                            ask_chunk + answer_window_chunks,
                        )
                    else:
                        window_end = ask_chunk + answer_window_chunks
                    # Find model's first answer in this ask's window
                    model_chunk = None
                    for ci in range(ask_chunk, window_end + 1):
                        out = by_chunk_idx.get(ci)
                        if out is None:
                            continue
                        if (out.get("kind") == "answer"
                                and out.get("answer_text")):
                            model_chunk = ci
                            break
                    t = _compute_timing_reward_v12(
                        answer_chunk=model_chunk,
                        visible_start_chunk=ask_chunk,
                        visible_end_chunk=window_end,
                    )
                    per_ask_t.append(t)
                if per_ask_t:
                    per_q_timings.append(sum(per_ask_t) / len(per_ask_t))
            timing_avg = (
                sum(per_q_timings) / len(per_q_timings)
                if per_q_timings else 0.0
            )
            all_rewards["timing"].append(timing_avg)
            all_masks["timing"].append(1.0 if per_q_timings else 0.0)

            # ── format ──
            fmt = _compute_format_reward_v12(chunk_texts)
            all_rewards["format"].append(fmt)
            all_masks["format"].append(1.0)

            # ── spam ──
            spam = _compute_spam_score_v12(
                n_recall_calls=n_recall, n_compress_calls=n_compress,
            )
            all_rewards["spam"].append(spam)
            all_masks["spam"].append(1.0)

            # ── silent_quality (per-chunk averaged) ──
            silent_res = _compute_per_chunk_silent_quality_v12(
                rollout_chunk_outputs=chunk_outputs,
                gold_action_per_chunk=gold_action_per_chunk,
            )
            all_rewards["silent_quality"].append(silent_res["silent_quality"])
            # Mask=0 if no chunks had non-neutral gold_action (no information)
            all_masks["silent_quality"].append(
                1.0 if silent_res["n_chunks_scored"] > 0 else 0.0
            )

    rewards_dict = {
        k: torch.tensor(all_rewards[k], dtype=torch.float) for k in keys
    }
    masks_dict = {
        k: torch.tensor(all_masks[k], dtype=torch.float) for k in keys
    }
    rewards = torch.zeros_like(rewards_dict[keys[0]])
    for k in keys:
        rewards = rewards + weights.get(k, 0.0) * rewards_dict[k] * masks_dict[k]
    rewards_masks = torch.stack([masks_dict[k] for k in keys], dim=1)
    return rewards, rewards_dict, rewards_masks


@node
def calc_rewards(
    ctx: Context,
    /,
    *,
    rollout_data: Auto[Dict[str, Any]],
    rewards: Ref[torch.Tensor],
    rewards_dict: Ref[Dict[str, torch.Tensor]],
    rewards_masks: Ref[torch.Tensor],
    group_size: Auto[int],
    tokenizer: Auto[Any],
    time_reward_window: Auto[int],
    time_reward_slack: Auto[float],
    rollout_max_think_tokens: Auto[int],
) -> Context:
    """Compute per-trajectory rewards + per-reward applicability masks.

    v12 design (5 components: outcome, timing, format, spam, silent_quality).

    Sub-dispatch: if the sample carries ``questions`` (multi-card trajectory
    format from pass4 ``*_trajectories.jsonl``), use the trajectory-level
    reward function. Otherwise use the single-question path (flat
    ``*_full.jsonl`` / ``*_train_sft.jsonl``).
    """
    _is_traj = False
    if rollout_data:
        first_sample = rollout_data[0].get("raw_sample") or {}
        _is_traj = (
            isinstance(first_sample.get("questions"), list)
            and isinstance(first_sample.get("gold_action_per_chunk"), dict)
        )
    if _is_traj:
        rewards_, rewards_dict_v12, rewards_masks_v12 = _calc_rewards_v12_trajectory(
            rollout_data,
            group_size=group_size,
            tokenizer=tokenizer,
        )
    else:
        rewards_, rewards_dict_v12, rewards_masks_v12 = _calc_rewards_v12(
            rollout_data,
            group_size=group_size,
            tokenizer=tokenizer,
            time_reward_window=time_reward_window,
            time_reward_slack=time_reward_slack,
        )
    return ctx.update({
        rewards: rewards_,
        rewards_dict: rewards_dict_v12,
        rewards_masks: rewards_masks_v12,
    })


# ---------------------------------------------------------------------------
# v11 GDPO-style advantage aggregation
# ---------------------------------------------------------------------------
#
# Replaces the external ``calc_grpo_advantages`` from ``deepslyme.node.rl.grpo``
# (which does single-reward, single group-norm). The new flow:
#
#   1. Per-reward group-norm within G rollouts of the same sample.
#      - Mean-only normalization (ReMemR1 grpo_use_adv=False precedent;
#        avoids std=0 group blow-ups when bimodal).
#      - Masked rows → NaN → ignored by nanmean → contribute 0 advantage.
#   2. Stack [B, num_rewards] → weighted sum → [B].
#   3. Batch-wide whiten ((x - μ) / σ). Critical to keep advantage scale
#      stable when adding/removing reward components.
#   4. Output [B] scalar advantage; downstream ``compute_grpo_loss`` tiles
#      to per-token advantage via the existing completion_mask.

# Stash for grpo_global_metrics — populated each step by the GDPO node.
_LAST_GDPO_DIAG: Dict[str, float] = {}


def _gdpo_per_reward_group_norm(
    reward_col: torch.Tensor,
    mask_col: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Mean-only group-norm with mask handling.

    reward_col, mask_col: [B] = [N_samples * G]
    Returns: [B] per-sample normalized advantage; masked-out rows are 0.

    A "group" is the G rollouts sharing the same sample_id. Empty groups
    (all rows masked) produce all zeros — no gradient signal from that
    sample for this reward, which is the desired behavior.
    """
    B = reward_col.shape[0]
    assert B % group_size == 0, f"reward_col len {B} not divisible by G={group_size}"
    N = B // group_size

    # Mask rewards as NaN so nanmean ignores them
    masked = torch.where(
        mask_col > 0,
        reward_col,
        torch.full_like(reward_col, float("nan")),
    )
    grouped = masked.view(N, group_size)                              # [N, G]
    g_mean = torch.nanmean(grouped, dim=1, keepdim=True)              # [N, 1]
    # Empty groups (all NaN) → nanmean returns NaN; replace with 0 so the
    # subtraction below produces NaN → nan_to_num → 0.
    g_mean = torch.nan_to_num(g_mean, nan=0.0)
    adv = (grouped - g_mean).flatten()                                # [B]
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    return adv


@node
def compute_gdpo_advantages(
    ctx: Context,
    /,
    *,
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    rewards_masks: Auto[torch.Tensor],
    advantages: Ref[torch.Tensor],
    group_size: Auto[int],
    advantage_mode: Auto[str] = "gdpo",
) -> Context:
    """Compute per-rollout advantages from the 8-reward dict + masks.

    v11.4: dispatch to aggregate_gdpo (default) or aggregate_grpo based on
    ``advantage_mode``. Pure-tensor algorithms in ``gdpo_advantage.py``.
    Per-reward / per-component diagnostics are stashed in module-level
    ``_LAST_GDPO_DIAG`` so ``grpo_global_metrics`` can log them without
    adding another slyme Ref.

    advantage_mode:
      "gdpo" — per-reward group-norm → weighted sum → batch-whiten.
               Each component pulls advantage independently; best when
               sparse signals are meaningful but bimodal (the v11 design
               assumption). Default.
      "grpo" — weighted scalar reward first → group z-norm. Standard
               DeepSeekMath formulation; useful as ablation baseline or
               when one outcome reward dominates and you want clean,
               interpretable advantage scaling.
    """
    adv, diag = _aggregate_advantages(
        rewards_dict, rewards_masks, group_size, mode=advantage_mode,
    )
    diag["advantage_mode"] = advantage_mode

    global _LAST_GDPO_DIAG
    _LAST_GDPO_DIAG = diag

    return ctx.set(advantages, adv)


def _extract_questions_at_chunks(raw_sample) -> Dict[int, str]:
    """Build {chunk_idx → user_question_text} from any of the 3 raw_sample
    schemas the trainer accepts.

    BUG FIX (2026-04-30, post pass1 v12.5 audit): the previous implementation
    of _build_rollout_messages only read raw_sample["conversations"], which
    v12.5 trajectory data and v12 flat data DON'T have. Result: questions
    NEVER appeared in the loss-time message reconstruction → policy was
    trained as "answer without seeing the question", a hard distribution
    mismatch from rollout (which DOES inject the question via StreamingAgentLoop).

    Schemas handled:
      A) Trajectory (v12.4+): raw_sample = {"questions": [{"question",
         "ask_chunks": [int, ...], ...}, ...]} — multiple cards, each may fire
         at multiple ask_chunks (multi-response F7/M1).
      B) Flat (v12 SFT): raw_sample = {"input": {"user_input": str}, "chunk_idx": int}
      C) Legacy v11: raw_sample = {"conversations": [{"role":"user","content":...,
         "timestamp": float}, ...]} — kept for backward-compat.

    Returns the same chunk→question map that the rollout path already builds
    (mirrors grpo.rollout's _question_at_chunk construction).
    """
    out: Dict[int, str] = {}

    # Schema A: trajectory
    if (isinstance(raw_sample.get("questions"), list)
            and isinstance(raw_sample.get("gold_action_per_chunk"), dict)):
        for q in raw_sample["questions"]:
            q_text = q.get("question") or q.get("gold_answer", "")
            for ac in q.get("ask_chunks") or []:
                out[int(ac)] = q_text
        return out

    # Schema B: flat input.user_input
    if "input" in raw_sample and raw_sample["input"].get("user_input"):
        ck = int(raw_sample.get("chunk_idx", 0))
        out[ck] = raw_sample["input"]["user_input"]
        return out

    # Schema B'): older messages format with <user_input> tags
    if "messages" in raw_sample:
        ck = int(raw_sample.get("chunk_idx", 0))
        for msg in raw_sample["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            m = re.search(r'<user_input>(.*?)</user_input>', text)
                            if m:
                                out[ck] = m.group(1)
                                return out
        return out

    # Schema C: legacy conversations[]
    if "conversations" in raw_sample:
        for c in raw_sample["conversations"]:
            if c.get("role") == "user":
                ts = float(c.get("timestamp", 0.0))
                # Map timestamp → chunk_idx using AGENT_CHUNK_SEC (= 1 in v12.5)
                ck = int(ts)  # 1s/chunk so timestamp seconds == chunk_idx
                out[ck] = c.get("content", "")
        return out

    return out


def _build_rollout_messages(
    raw_sample, chunk_results, gen_idx, tokenizer, frames_per_chunk
):
    data_path = raw_sample.get("data_path", "")
    video_path = raw_sample.get("video_path", "")
    abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

    # v12.5 fix: build chunk→question map from whichever schema the sample
    # carries (trajectory / flat / legacy). MUST mirror the rollout path's
    # question injection (rollout in this file builds an identical map at
    # line 430-477) so the loss-time reconstruction sees the same conditional
    # context the policy generated under.
    question_at_chunk = _extract_questions_at_chunks(raw_sample)

    num_chunks = len(chunk_results)
    if num_chunks == 0:
        raise ValueError("No chunk results – cannot build messages.")
    video_chunk_size = chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
    total_start = chunk_results[0]["window_start"]
    total_end = chunk_results[-1]["window_end"]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_V12}
    ]

    for cr_idx, cr in enumerate(chunk_results):
        w_start, w_end = cr["window_start"], cr["window_end"]
        is_last = cr_idx == num_chunks - 1
        user_content: List[Dict] = [
            {
                "type": "video",
                "video": abs_video_path,
                "video_start": w_start,
                "video_end": w_end,
            }
        ]

        # v12.5 fix: question fires at the chunk_idx where rollout injected it.
        # chunk_idx = the absolute chunk index of THIS step (cr_idx if rollout
        # ran from chunk 0, but use the explicit field when present).
        cur_chunk_idx = int(cr.get("chunk_idx", cr_idx))
        q_text = question_at_chunk.get(cur_chunk_idx)
        if q_text:
            user_content.append({"type": "text", "text": "\n" + q_text})
        messages.append({"role": "user", "content": user_content})

        gen_text = tokenizer.decode(
            cr["generated_tokens"][gen_idx], skip_special_tokens=False
        )
        for _sp in ("<|im_end|>", "<|endoftext|>"):
            if gen_text.endswith(_sp):
                gen_text = gen_text[: -len(_sp)]
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": gen_text}]}
        )

    video_meta = build_video_meta(
        abs_path=abs_video_path,
        total_start=total_start,
        total_end=total_end,
        num_chunks=num_chunks,
        frames_per_chunk=frames_per_chunk,
    )
    return messages, video_meta, video_chunk_size


@node
def build_grpo_inputs(
    ctx: Context,
    /,
    *,
    step_micro_items: Auto[List],
    step_micro_inputs: Ref[Dict[str, Any]],
    rollout_data: Auto[Dict[str, Any]],
    processor: Auto[Any],
    tokenizer: Auto[Any],
    model_type: Auto[str],
    rollout_fpc: Auto[float],
) -> Context:
    """Convert rollout data + raw sample info into tokenised model inputs.

    ``step_micro_items`` coming in is a list of micro-batch item descriptors
    (``{"sample_idx": int, "gen_idx": int}``).  For each descriptor we:

    1. Reconstruct the full chat messages from the rollout's raw sample and
       generated tokens.
    2. Call ``process_messages_to_model_inputs`` (shared with the SFT pipeline)
       to load video frames and tokenise.
    3. Compute MROPE position IDs.
    4. Collate everything into a single batched dict and write it back to
       ``step_micro_inputs`` so that downstream nodes (``prepare_inputs``,
       ``compute_grpo_loss``) receive the expected tensor format.

    Pixel limits are already baked into the processor via
    ``update_processor_pixels`` (called in ``LazyRawDataset.__init__``).
    """
    micro_items = step_micro_items
    all_items = []
    _preloaded_cache = {}

    for item_desc in micro_items:
        sample_idx, gen_idx = item_desc["sample_idx"], item_desc["gen_idx"]
        sample_data = rollout_data[sample_idx]
        messages, video_meta, video_chunk_size = _build_rollout_messages(
            raw_sample=sample_data["raw_sample"],
            chunk_results=sample_data["chunk_results"],
            gen_idx=gen_idx,
            tokenizer=tokenizer,
            frames_per_chunk=int(rollout_fpc),
        )
        if sample_idx not in _preloaded_cache:
            pv = sample_data.get("_preloaded_video")
            _preloaded_cache[sample_idx] = (
                (pv["split_videos"], pv["video_kwargs"], pv["chunk_metadatas"])
                if pv
                else None
            )

        result = process_messages_to_model_inputs(
            messages=messages,
            video_meta=video_meta,
            video_chunk_size=video_chunk_size,
            processor=processor,
            model_type=model_type,
            add_generation_prompt=False,
            preloaded_frames=_preloaded_cache[sample_idx],
        )
        result["position_ids"] = compute_position_ids(result, processor, model_type)
        all_items.append(result)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"].squeeze(0) for item in all_items],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    video_masks = torch.nn.utils.rnn.pad_sequence(
        [item["video_mask"].squeeze(0) for item in all_items],
        batch_first=True,
        padding_value=0,
    )
    position_ids = pad_and_cat([item["position_ids"] for item in all_items])
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    completion_mask = torch.zeros_like(input_ids)
    for b in range(input_ids.size(0)):
        for start, end in find_assistant_spans(input_ids[b].tolist(), tokenizer):
            completion_mask[b, start:end] = 1

    videos = [
        item["pixel_values_videos"]
        for item in all_items
        if "pixel_values_videos" in item
    ]
    video_grid_thws = [
        item["video_grid_thw"] for item in all_items if "video_grid_thw" in item
    ]

    return ctx.set(
        step_micro_inputs,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "video_mask": video_masks,
            "position_ids": position_ids,
            "pixel_values_videos": torch.cat(videos, dim=0) if videos else None,
            "video_grid_thw": torch.cat(video_grid_thws, dim=0)
            if video_grid_thws
            else None,
        },
    )


@node
def apply_liger_kernel_for_grpo(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
    reference_model: Auto[PreTrainedModel],
    model_type: Auto[str],
) -> Context:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
    from thinkstream.model.patch import GRPO_LCE_FORWARD

    if model_type not in GRPO_LCE_FORWARD:
        raise ValueError(f"Unsupported model_type for GRPO: {model_type}")
    grpo_forward_fn = GRPO_LCE_FORWARD[model_type]

    for m in [model, reference_model]:
        _apply_liger_kernel_to_instance(model=m, fused_linear_cross_entropy=False)
        m.forward = types.MethodType(grpo_forward_fn, m)
    return ctx


@node
def compute_grpo_loss(
    ctx: Context,
    /,
    *,
    step_micro_inputs: Auto[Dict[str, Any]],
    step_advantages: Auto[torch.Tensor],
    model_for_training: Auto[Any],
    reference_model: Auto[PreTrainedModel],
    step_loss: Ref[torch.Tensor],
    beta: Auto[float],
    device: Auto[torch.device],
) -> Context:
    """Compute GRPO loss via the patched model forward.

    The model's ``forward`` has been replaced by the model-type-specific
    GRPO LCE forward from :data:`thinkstream.model.patch.GRPO_LCE_FORWARD`
    (applied by :func:`apply_liger_kernel_for_grpo`).  When the extra kwarg
    ``advantages`` is provided, that forward uses
    :class:`LigerFusedLinearGRPOLoss` to fuse the ``lm_head`` projection and
    the GRPO loss in memory-efficient chunks – the full ``[B, L, V]`` logits
    tensor is **never** materialised.

    Because the call goes through ``model_for_training`` (the DeepSpeed /
    DDP engine) directly, all distributed training features (gradient
    synchronisation, mixed-precision, ZeRO, …) work normally.
    """
    ref_input, ref_weight, ref_bias = None, None, None
    if beta != 0.0:
        reference_model.to(device)
        video_block_mask = build_video_block_mask(
            reference_model,
            step_micro_inputs.get("video_mask"),
            step_micro_inputs.get("attention_mask"),
        )
        ref_backbone_kwargs = dict(
            input_ids=step_micro_inputs["input_ids"],
            attention_mask=step_micro_inputs["attention_mask"],
            position_ids=step_micro_inputs.get("position_ids"),
            pixel_values_videos=step_micro_inputs.get("pixel_values_videos"),
            video_grid_thw=step_micro_inputs.get("video_grid_thw"),
            video_block_mask=video_block_mask,
            use_cache=False,
            return_dict=True,
        )
        with torch.no_grad():
            ref_out = reference_model.model(**ref_backbone_kwargs)
            ref_input = ref_out.last_hidden_state[:, :-1, :].contiguous()
            ref_weight = reference_model.lm_head.weight.detach().clone()
            if reference_model.lm_head.bias is not None:
                ref_bias = reference_model.lm_head.bias.detach().clone()
        reference_model.to("cpu")
        torch.cuda.empty_cache()

    model_kwargs = dict(
        input_ids=step_micro_inputs["input_ids"],
        attention_mask=step_micro_inputs["attention_mask"],
        position_ids=step_micro_inputs.get("position_ids"),
        pixel_values_videos=step_micro_inputs.get("pixel_values_videos"),
        video_grid_thw=step_micro_inputs.get("video_grid_thw"),
        video_mask=step_micro_inputs.get("video_mask"),
        use_cache=False,
        advantages=step_advantages,
        ref_input=ref_input,
        ref_weight=ref_weight,
        ref_bias=ref_bias,
        completion_mask=step_micro_inputs.get("completion_mask"),
        grpo_beta=beta,
    )
    outputs = model_for_training(**model_kwargs)
    return ctx.set(step_loss, outputs.loss)


def _avg_think_len_per_chunk_micro(micro_items, rollout_data_, tokenizer):
    """Diagnostic-only: average think token length across micro-batch.

    Not a reward in v11 (think_len reward removed); kept for visibility.
    """
    all_lengths = []
    for item in micro_items:
        chunk_results = rollout_data_[item["sample_idx"]].get("chunk_results", [])
        all_lengths.extend(
            _collect_think_lengths(chunk_results, item["gen_idx"], tokenizer)
        )
    return sum(all_lengths) / len(all_lengths) if all_lengths else 0.0


@expression
def grpo_micro_metrics(
    ctx: Context,
    /,
    *,
    step_loss: Auto[torch.Tensor],
    step_micro_items: Auto[List],
    rollout_data: Auto[Dict[str, Any]],
    tokenizer: Auto[Any],
) -> dict:
    loss_val = step_loss.detach().float().item()
    avg_think = _avg_think_len_per_chunk_micro(
        step_micro_items, rollout_data, tokenizer
    )

    return {
        "loss": loss_val,
        "avg_think_len": avg_think,
    }


@expression
def grpo_global_metrics(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
    optimizer: Auto[torch.optim.Optimizer],
    rewards: Auto[torch.Tensor],
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    group_size: Auto[int],
) -> dict:
    grad_norm = model_for_training.get_global_grad_norm()
    if hasattr(grad_norm, "item"):
        grad_norm = grad_norm.item()
    lr = optimizer.param_groups[0]["lr"]

    # Global reward mean (raw weighted sum, logging-only)
    reward_mean = rewards.float().mean().item()

    # Intra-group variance, averaged across groups (sanity: should NOT be 0
    # — if it is, all rollouts in each group got identical raw reward and
    # GDPO's per-reward group-norm has nothing to differentiate).
    if rewards.numel() > 1 and group_size > 1:
        grouped_rewards = rewards.float().view(-1, group_size)
        reward_var = grouped_rewards.var(dim=1).mean().item()
    else:
        reward_var = 0.0

    # Component-wise raw reward means
    component_means = {
        f"reward_{k}_mean": v.float().mean().item() for k, v in rewards_dict.items()
    }

    # GDPO per-reward advantage stats + post-whiten total (populated by
    # compute_gdpo_advantages on the same step). Stash → log; harmless if empty.
    return {
        "grad_norm": grad_norm,
        "learning_rate": lr,
        "reward_mean": reward_mean,
        "reward_var": reward_var,
        **component_means,
        **dict(_LAST_GDPO_DIAG),
    }


@node
def prepare_grpo_micro_batches(
    ctx: Context,
    /,
    *,
    advantages: Auto[torch.Tensor],
    rewards: Auto[torch.Tensor],
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    micro_batch_size: Auto[int],
    group_size: Auto[int],
    step_advantages: Ref[torch.Tensor],
    step_micro_rewards: Ref[torch.Tensor],
    step_micro_rewards_dict: Ref[Dict[str, torch.Tensor]],
    step_micro_items: Ref[List],
    step_micro_batches: Ref[list[dict[Ref, Any]]],
) -> Context:
    total_samples = advantages.shape[0]
    num_micro_batches = math.ceil(total_samples / micro_batch_size)
    micro_batches = []

    for mb_idx in range(num_micro_batches):
        start_idx = mb_idx * micro_batch_size
        end_idx = min(start_idx + micro_batch_size, total_samples)
        micro_items = [
            {
                "sample_idx": flat_idx // group_size,
                "gen_idx": flat_idx % group_size,
            }
            for flat_idx in range(start_idx, end_idx)
        ]

        mb_updates = {
            step_advantages: advantages[start_idx:end_idx],
            step_micro_rewards: rewards[start_idx:end_idx],
            step_micro_rewards_dict: {
                k: v[start_idx:end_idx] for k, v in rewards_dict.items()
            },
            step_micro_items: micro_items,
        }
        micro_batches.append(mb_updates)

    return ctx.set(step_micro_batches, micro_batches)


@node
def init_grpo_refs(ctx: Context, /, *, inference_engine: Ref[Any]) -> Context:
    return ctx.set(inference_engine, None)


class DataArgs:
    pass


@node
def init_grpo_dataset(
    ctx: Context,
    /,
    *,
    processor: Auto[Any],
    train_dataset: Ref[Any],
    data_collator: Ref[Any],
    data_dataset_use: Auto[str],
    rollout_min_pixels: Auto[int],
    rollout_max_pixels: Auto[int],
    rollout_fpc: Auto[float],
    rollout_max_chunks: Auto[int],
    model_type: Auto[str],
) -> Context:
    """
    Initialises a raw (unprocessed) dataset for the GRPO pipeline.

    Unlike the SFT ``init_dataset`` which pre-tokenises every sample, this
    node simply loads the JSON annotations so that the rollout stage can
    perform streaming inference on the raw data.

    Like SFT's ``init_dataset``, the processor's pixel limits are updated
    here via ``update_processor_pixels`` to match the rollout configuration.

    Video loading config is forwarded to ``LazyRawDataset`` so that
    ``__getitem__`` can pre-load frames via DataLoader ``num_workers``.
    """
    vp = processor.video_processor
    data_args = DataArgs()
    items = dict(
        dataset_use=data_dataset_use,
        min_pixels=rollout_min_pixels,
        max_pixels=rollout_max_pixels,
        video_min_pixels=rollout_min_pixels,
        video_max_pixels=rollout_max_pixels,
        video_min_frames=getattr(vp, "min_frames", 4),
        video_max_frames=getattr(vp, "max_frames", 768),
        video_fps=getattr(vp, "fps", 2.0),
    )
    for k, v in items.items():
        setattr(data_args, k, v)

    data_module = make_raw_data_module(
        processor,
        data_args,
        frames_per_chunk=int(rollout_fpc),
        max_chunks=rollout_max_chunks,
        model_type=model_type,
    )
    return ctx.update(
        {
            train_dataset: data_module["train_dataset"],
            data_collator: data_module["data_collator"],
        }
    )


@wrapper
def timer(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    name: str = "",
):
    import time

    start = time.time()
    ctx = call_next(ctx)
    end = time.time()
    print(f"{name}: {end - start}s")
    return ctx
