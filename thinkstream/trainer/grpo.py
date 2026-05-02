import json
import os
import re
import types
import math
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
import torch

# v12.11 (2026-05-01): Loss-time batching mode toggle.
#
# "trajectory" (legacy / safe default):
#   1 batch item = 1 (sample_idx, gen_idx) trajectory rollout.
#   _build_rollout_messages concatenates N chunks → single 50K-token sample.
#   Original behavior; reproduces production runs prior to v12.11.
#
# "per_chunk" (MemAgent / ReMemR1 aligned):
#   1 batch item = 1 (sample_idx, gen_idx, chunk_idx) chunk decision.
#   _build_rollout_messages_single_chunk builds short ~6K prompts per chunk.
#   Eliminates OOM + truncation on long-trajectory rollouts.
#
# Switch via env: THINKSTREAM_LOSS_BATCH_MODE=trajectory
# Default flipped to per_chunk in v12.11 — the audit on 2026-05-01 found
# trajectory mode produces 50K+ token concatenated samples on long-trajectory
# rollouts, which OOM under 16K context cap. per_chunk is now the corrected
# default; trajectory remains opt-in for ablation against the legacy v12.10
# behavior. Any value other than "per_chunk"/"trajectory" falls back to per_chunk.
LOSS_BATCH_MODE = os.environ.get("THINKSTREAM_LOSS_BATCH_MODE", "per_chunk").lower()
if LOSS_BATCH_MODE not in ("trajectory", "per_chunk"):
    LOSS_BATCH_MODE = "per_chunk"

# v12.11: ReMemR1-style mixed advantage. When enabled, per-step state
# rewards (format + action) get group-normalized at (uid, step_id) level
# and blended with the trajectory outcome advantage:
#   advantage = α × outcome_adv + (1-α) × state_adv
# Default α matches our docs/v12.0_protocol_migration_design.md §5.7
# (we found α=0.7 worked better than ReMemR1's 0.8 because our per-step
# signal is denser).
USE_STATE_ADVANTAGE = (
    os.environ.get("THINKSTREAM_USE_STATE_ADVANTAGE", "0") == "1"
)
STATE_ADVANTAGE_ALPHA = float(
    os.environ.get("THINKSTREAM_STATE_ADV_ALPHA", "0.7")
)

# v12.11: Dynamic-bsz token packing (MemAgent verl/utils/seqlen_balancing.py).
# When enabled, prepare_grpo_micro_batches groups samples by total token
# count (max_token_len_per_gpu) instead of fixed micro_batch_size, balancing
# per-batch sequence length so OOM is bounded by token budget not item count.
USE_DYNAMIC_BSZ = (
    os.environ.get("THINKSTREAM_USE_DYNAMIC_BSZ", "0") == "1"
)
DYNAMIC_BSZ_MAX_TOKEN_LEN = int(
    os.environ.get("THINKSTREAM_DYNAMIC_BSZ_MAX_TOKEN", "16384")
)

# v12.11: Advantage aggregation mode — the EXISTING gdpo / grpo dispatch
# in `gdpo_advantage.aggregate_advantages` is already switchable via the
# `mode` arg, but until now it was only exposed as a positional kwarg.
# Lift it to env so operators can A/B without touching configs:
#   THINKSTREAM_ADVANTAGE_MODE=gdpo    (default — per-reward group-norm)
#   THINKSTREAM_ADVANTAGE_MODE=grpo    (DeepSeekMath baseline — single scalar)
#   THINKSTREAM_ADVANTAGE_MODE=remem   (gdpo outcome × α + state × (1-α),
#                                       requires USE_STATE_ADVANTAGE=1)
ADVANTAGE_MODE = os.environ.get("THINKSTREAM_ADVANTAGE_MODE", "gdpo").lower()
if ADVANTAGE_MODE not in ("gdpo", "grpo", "remem"):
    ADVANTAGE_MODE = "gdpo"

# v12.11 P1.1 full: state reward computation mode (used when ADVANTAGE_MODE=remem
# AND USE_STATE_ADVANTAGE=1). Four levels of completeness:
#
#   "format_only"   (default, cheapest): per-chunk format reward only.
#                                         Tag well-formedness 0/1 — no teacher
#                                         lookup, parses generated text.
#   "format_action": + per-chunk teacher-action match (silent/response/recall/
#                    compress agreement). Requires gold_action_per_chunk in
#                    raw_sample (already provided by trajectory data).
#   "remem_full"   : full ReMemR1 metric_utils.py port — adds word-level
#                    recall increment for memory-update / recall steps.
#                    Requires ground_truth tokens (raw_sample.gold_answer).
#   "with_silent_q": format_action + per-chunk silent_quality (our
#                    streaming-specific signal — hallucinate/miss penalty).
STATE_REWARD_MODE = os.environ.get(
    "THINKSTREAM_STATE_REWARD_MODE", "format_only"
).lower()
if STATE_REWARD_MODE not in (
    "format_only", "format_action", "remem_full", "with_silent_q",
):
    STATE_REWARD_MODE = "format_only"

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
from thinkstream.data.agent_protocol import (
    SYSTEM_PROMPT_V12,
    AGENT_CHUNK_SEC as AGENT_CHUNK_SEC_RUNTIME,
)
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
        # v12.11 P0.5 (2026-05-01): inference_engine is currently always None
        # because there is no vLLM construction node wired into the trainer
        # init path (init_grpo_refs only sets it to None). Surface this as
        # an explicit error rather than letting streaming_vllm_rollout crash
        # with a less-informative AttributeError on llm.generate().
        # Proper fix requires:
        #   1. Construct vLLM LLM(model=args.model.name_or_path, ...) at init
        #   2. After each train step, sync weights from training model to
        #      vLLM workers (NCCL bridge or LLM.load_weights)
        #   3. Free vLLM cache + restore at rollout boundary
        # Tracked as a follow-up commit; meanwhile, keep use_vllm_rollout=False.
        if inference_engine is None:
            raise RuntimeError(
                "use_vllm_rollout=True but inference_engine is None. The vLLM "
                "construction path is not yet implemented for RL training "
                "(see grpo.init_grpo_refs). Use use_vllm_rollout=False (HF "
                "generate path) or wait for the dedicated vLLM-RL PR."
            )
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
        """Wrap model generation for StreamingAgentLoop.

        v12.6 fix: pass tools=TOOLS_SCHEMA so chat_template auto-renders the
        <tools> block in system prompt — same behavior as SFT data_processor
        (data_processor.py:574). Without tools=, the rollout policy receives
        a different system context than what SFT trained on, breaking
        train/infer parity for tool-call decisions.
        """
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
        video_metadata = []
        has_video_meta = True
        for msg in messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "video":
                    meta = item.get("video_metadata")
                    frames = item.get("video")
                    if isinstance(meta, dict):
                        video_metadata.append(meta)
                    elif isinstance(frames, list) and frames:
                        video_metadata.append({"total_num_frames": len(frames)})
                    else:
                        has_video_meta = False
        template_kwargs = dict(
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, tools=TOOLS_SCHEMA,
            do_sample_frames=False,
        )
        if video_metadata and has_video_meta:
            template_kwargs["video_metadata"] = video_metadata
        inputs = processor.apply_chat_template(messages, **template_kwargs)
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
            # v12.13 fix (P0-1): parallel meta map carries options +
            # answer_form so agent_loop.step → MemoryState.add_query can
            # store them on the query → format_queries_block renders
            # "Options: A) ..." for pending MC at every chunk where the
            # query is still pending. Without this, RL/eval queries lose
            # MC options after the ask chunk passes.
            _question_meta_at_chunk: Dict[int, Dict] = {}
            _all_ask_chunks: List[int] = []
            _all_answer_chunks: List[int] = []
            for q in raw_sample["questions"]:
                q_text = q.get("question") or q.get("gold_answer", "")
                q_meta = {
                    "options": list(q.get("options") or []),
                    "answer_form": q.get("answer_form", ""),
                }
                for ac in q.get("ask_chunks") or []:
                    _question_at_chunk[int(ac)] = q_text
                    _question_meta_at_chunk[int(ac)] = q_meta
                    _all_ask_chunks.append(int(ac))
                for ac in q.get("answer_chunks") or []:
                    _all_answer_chunks.append(int(ac))
            ask_chunk = max(_all_ask_chunks) if _all_ask_chunks else (
                rollout_max_chunks - 1
            )
            # ROLLOUT_END_CHUNK: cap rollout at last answer chunk + slack.
            # This is the upper bound used for `num_chunks = min(end+1, max)`.
            ROLLOUT_SLACK = 2
            _rollout_end_chunk = (
                max(_all_answer_chunks) + ROLLOUT_SLACK
                if _all_answer_chunks
                else ask_chunk + 5  # legacy fallback when no answer_chunks
            )
            # No single user_question for trajectory rollouts; per-chunk.
            user_question = None
        else:
            ask_chunk = raw_sample.get("chunk_idx", rollout_max_chunks - 1)
            _question_at_chunk = {}
            _question_meta_at_chunk = {}
            _rollout_end_chunk = ask_chunk + 5    # flat schema fallback

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
            # v12.13 fix (P0-1): rollout cap extends to last answer_chunk.
            # forward cards have lead 18-32 → ask + 5 would never reach the
            # answer position. _rollout_end_chunk = max(answer_chunks) + slack.
            num_chunks = min(_rollout_end_chunk + 1, rollout_max_chunks)
            for chunk_idx in range(num_chunks):
                # v12.4 trajectory: question may fire at any chunk that's in
                # _question_at_chunk (one card → ≥1 ask_chunks). Single-question
                # path falls through with question only at the lone ask_chunk.
                q = _question_at_chunk.get(chunk_idx)
                if q is None and not _is_traj_sample and chunk_idx == ask_chunk:
                    q = user_question
                # v12.13 fix (P0-1): plumb options/answer_form through to
                # MemoryState.add_query so format_queries_block renders MC
                # Options for pending queries.
                q_meta = _question_meta_at_chunk.get(chunk_idx)
                result = loop.step(
                    chunk_idx=chunk_idx,
                    video_path=abs_video_path,
                    user_question=q,
                    user_question_meta=q_meta,
                )
                # v12.11 P0.6 fix: for recall multi-turn, the FINAL assistant
                # turn that loss-time message reconstruction appends should be
                # the SECOND-pass answer, not the first-pass tool_call. The
                # first-pass tool_call already lives inside step_messages
                # (captured by agent_loop). Storing first-pass here meant:
                #   loss messages = [..., user, assistant(tool_call from step_msgs),
                #                    user(recall_result), assistant(tool_call AGAIN)]
                # → no final answer was ever trained, recall got trained twice.
                final_text = result.get("raw", "") or ""
                if (result.get("action") == "recall"
                        and result.get("recall_step2") is not None):
                    second_pass = result.get("recall_step2_raw_text", "") or ""
                    if second_pass:
                        final_text = second_pass
                # Store result with generated tokens for reward/loss computation
                chunk_results_g.append({
                    "chunk_idx": chunk_idx,
                    "action": result.get("action", "unknown"),
                    "think": result.get("think", ""),
                    "payload": result.get("payload", {}),
                    "raw_output": final_text,
                    "generated_tokens": tokenizer.encode(
                        final_text, add_special_tokens=False,
                    ),
                    # v12.11 P0.6: keep first-pass for diagnostics / format-reward
                    # if needed (recall tool_call format check), but it's NOT
                    # used as the final assistant turn.
                    "recall_first_pass_text": (
                        result.get("raw", "")
                        if result.get("action") == "recall"
                        and result.get("recall_step2") is not None
                        else ""
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
                    # v12.6 fix: chunk_idx × AGENT_CHUNK_SEC (was hardcoded ×2
                    # under v11 2s/chunk; v12.5 uses 1s/chunk).
                    "window_start": chunk_idx * AGENT_CHUNK_SEC_RUNTIME,
                    "window_end": (chunk_idx + 1) * AGENT_CHUNK_SEC_RUNTIME,
                    # v12.6 fix: capture the EXACT prompt messages the
                    # policy conditioned on (full memory/visual_window/
                    # queries/recall context). _build_rollout_messages
                    # below reuses these so loss-time logprobs match
                    # sampling-time logprobs.
                    "step_messages": result.get("step_messages"),
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
                # v12.6 fix: was hardcoded ci*2 (legacy 2s/chunk). Use
                # canonical AGENT_CHUNK_SEC so v12.5's 1s/chunk produces
                # window_start=ci, NOT ci*2.
                "window_start": ci * AGENT_CHUNK_SEC_RUNTIME,
                "window_end": (ci + 1) * AGENT_CHUNK_SEC_RUNTIME,
                "generated_tokens": [],
                "memory_token_count": [],   # per-gen post-step memory size
                "compress_budget": [],      # per-gen budget (constant within a run, kept per-gen for symmetry)
                "recall_returned_chunks": [],  # per-gen retriever output
                # v12.6: per-gen step_messages so _build_rollout_messages
                # can rebuild the exact prompt the policy conditioned on.
                # Loss-time logprobs need the same memory/visual_window/
                # queries context the rollout used.
                "step_messages": [],
                # v12.11 audit-3 fix (2026-05-01): per-gen first-pass recall
                # tool_call text. Reward parser at _calc_rewards_v12_trajectory
                # reads cr["recall_first_pass_text"][gen_idx] to count actual
                # recall calls (n_recall / spam / behavior_recall_used_rate).
                # Without merging this field per-gen, those counters all read
                # zero on HF rollout despite recalls actually happening.
                "recall_first_pass_text": [],
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
                    merged["step_messages"].append(cr_g.get("step_messages"))
                    merged["recall_first_pass_text"].append(
                        cr_g.get("recall_first_pass_text", "")
                    )
                else:
                    # Pad with empty if this gen finished early
                    merged["generated_tokens"].append(torch.tensor([]))
                    merged["memory_token_count"].append(0)
                    merged["compress_budget"].append(0)
                    merged["recall_returned_chunks"].append([])
                    merged["step_messages"].append(None)
                    merged["recall_first_pass_text"].append("")
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

                # v12.11 audit-3 fix (2026-05-01): legacy flat reward path
                # parity with trajectory path. generated_tokens is the
                # SECOND-pass answer for recall chunks (P0.6 fix); the
                # first-pass tool_call lives in recall_first_pass_text.
                # Without re-parsing first pass, n_recall stays zero on
                # actual recall trajectories under DATASET=stream_agent_rl.
                fpt_field = cr.get("recall_first_pass_text", "")
                if isinstance(fpt_field, list):
                    first_pass_text = fpt_field[g] if g < len(fpt_field) else ""
                else:
                    first_pass_text = fpt_field or ""
                first_pass_parsed = (
                    parse_agent_output_v12(first_pass_text)
                    if first_pass_text else None
                )
                is_recall_chunk = (
                    first_pass_parsed is not None
                    and first_pass_parsed.get("kind") == "recall"
                )

                if parsed["kind"] == "answer" and parsed["answer_text"]:
                    if answer_chunk is None:
                        answer_chunk = cr["chunk_idx"]
                        final_answer = parsed["answer_text"]
                if is_recall_chunk:
                    # Capture recall via FIRST-pass parser (which sees the
                    # actual tool_call). Counted regardless of second-pass kind.
                    n_recall += 1
                    args = (first_pass_parsed.get("tool_call") or {}).get("arguments") or {}
                    if recall_query_text is None:
                        recall_query_text = args.get("query") or ""
                    returned_lists = cr.get("recall_returned_chunks") or []
                    if isinstance(returned_lists, list) and g < len(returned_lists):
                        rl = returned_lists[g]
                        if isinstance(rl, list):
                            recall_returned_per_call.append([int(c) for c in rl])
                elif parsed["kind"] == "recall":
                    # Edge case: no recall_first_pass_text (legacy rollout
                    # cache from before the per-gen merge). Fall back to
                    # second-pass parsing.
                    n_recall += 1
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

                # v12.11 audit fix #3 (2026-05-01): generated_tokens for recall
                # chunks now stores ONLY the second-pass answer (P0.6 fix).
                # The first-pass tool_call lives separately on the chunk_result
                # under "recall_first_pass_text". Without re-parsing it here,
                # n_recall / spam / behavior_recall_used_rate would all read
                # zero for actual recall trajectories. Parse first-pass when
                # present, classify the chunk as kind="recall" (its semantic
                # action), and keep the answer_text from second-pass for
                # outcome scoring.
                # v12.11 audit-3: recall_first_pass_text is now a per-gen
                # list (merged in rollout step). Index by g; tolerate the
                # legacy scalar shape for back-compat with cached rollouts.
                fpt_field = cr.get("recall_first_pass_text", "")
                if isinstance(fpt_field, list):
                    first_pass_text = fpt_field[g] if g < len(fpt_field) else ""
                else:
                    first_pass_text = fpt_field or ""
                first_pass_parsed = (
                    parse_agent_output_v12(first_pass_text)
                    if first_pass_text else None
                )
                is_recall_chunk = (
                    first_pass_parsed is not None
                    and first_pass_parsed.get("kind") == "recall"
                )

                # v12.11 audit-5 P0 #2 fix (2026-05-01): for shape-B recall
                # chunks, the SECOND pass produces a real <answer> (the
                # final answer to the question). Downstream rewards
                # (compute_trajectory_outcome_v12 / timing /
                # silent_quality / behavior denominator) all gate on
                # kind == "answer" — if we tag recall chunks as
                # kind="recall", their final answer is invisible to
                # outcome scoring → recall trajectories never get reward
                # credit even when they answer correctly.
                #
                # Fix: keep `kind` from the second-pass parser (= "answer"
                # for shape B's final turn). Track recall via a SEPARATE
                # boolean `is_recall_chunk` so n_recall / spam / behavior
                # counters can still distinguish recall trajectories.
                # Downstream outcome / timing / silent_quality now
                # correctly count the second-pass answer.
                chunk_outputs.append({
                    "chunk_idx": cr.get("chunk_idx"),
                    # Second-pass kind = the FINAL action emitted at this
                    # chunk. For shape-B recall this is "answer"; outcome
                    # scoring needs this.
                    "kind": parsed.get("kind", "unknown"),
                    "answer_text": parsed.get("answer_text"),
                    "tool_call": (
                        first_pass_parsed.get("tool_call") if is_recall_chunk
                        else parsed.get("tool_call")
                    ),
                    # NEW: explicit recall flag for n_recall / spam / behavior
                    # bookkeeping (replaces overloading `kind`).
                    "is_recall_chunk": bool(is_recall_chunk),
                    # v12.11: keep both passes for downstream format-reward audit.
                    "recall_first_pass_kind": (
                        first_pass_parsed.get("kind") if first_pass_parsed
                        else None
                    ),
                    "recall_first_pass_format_error": (
                        bool(first_pass_parsed.get("format_error")) if first_pass_parsed
                        else None
                    ),
                    "format_error": bool(parsed.get("format_error")),
                })
                # Counters: recall counted by first-pass presence; compress by
                # second-pass parser output (compress is single-turn).
                if is_recall_chunk:
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
                # v12.13 fix (P0-2): timing window mirrors outcome reward —
                # uses answer_chunks (where the answer is actually expected),
                # not ask_chunks (where the question is asked). For forward
                # cards the gap is 18-32 chunks; the old `ask + 5` window
                # marked correct late answers as "missed" and ignored the
                # whole forward family.
                #
                # Two modes (matches v12_rewards.compute_trajectory_outcome_v12):
                #   SINGLE: window = [ask_chunk, last_answer_chunk + slack]
                #   MULTI:  per-emit window from per_emit_answers, with
                #           non-overlapping search floor.
                answer_chunks_q = sorted(q.get("answer_chunks") or [])
                ask_chunks = sorted(q.get("ask_chunks") or [])
                ask_chunk_q = q.get("ask_chunk")
                if not isinstance(ask_chunk_q, int):
                    ask_chunk_q = (
                        ask_chunks[0] if ask_chunks else
                        (answer_chunks_q[0] if answer_chunks_q else None)
                    )
                if ask_chunk_q is None:
                    continue
                per_emit_q = q.get("per_emit_answers") or []
                is_multi = (len(answer_chunks_q) > 1 or len(per_emit_q) > 1)
                SLACK = 2

                per_ask_t: List[float] = []
                if is_multi:
                    # Per-emit timing
                    chunk_gold_q = {
                        int(e["chunk"]): str(e.get("value", ""))
                        for e in per_emit_q
                        if isinstance(e, dict) and "chunk" in e
                    }
                    target_chunks = sorted(
                        chunk_gold_q.keys() or answer_chunks_q
                    )
                    next_floor = ask_chunk_q
                    for i, emit_chunk in enumerate(target_chunks):
                        lo = max(next_floor, emit_chunk - SLACK)
                        if i + 1 < len(target_chunks):
                            hi = min(emit_chunk + SLACK,
                                     target_chunks[i + 1] - 1)
                        else:
                            hi = emit_chunk + SLACK
                        if lo > hi:
                            per_ask_t.append(0.0)
                            continue
                        model_chunk = None
                        for ci in range(lo, hi + 1):
                            out = by_chunk_idx.get(ci)
                            if (out and out.get("kind") == "answer"
                                    and out.get("answer_text")):
                                model_chunk = ci
                                break
                        if model_chunk is None:
                            per_ask_t.append(0.0)
                            continue
                        next_floor = model_chunk + 1
                        t = _compute_timing_reward_v12(
                            answer_chunk=model_chunk,
                            visible_start_chunk=max(ask_chunk_q, emit_chunk - SLACK),
                            visible_end_chunk=hi,
                        )
                        per_ask_t.append(t)
                else:
                    # Single-emit: full window from ask to last answer + slack
                    last_emit = (answer_chunks_q[-1]
                                  if answer_chunks_q else
                                  ask_chunk_q + answer_window_chunks)
                    window_end = last_emit + SLACK
                    model_chunk = None
                    for ci in range(ask_chunk_q, window_end + 1):
                        out = by_chunk_idx.get(ci)
                        if (out and out.get("kind") == "answer"
                                and out.get("answer_text")):
                            model_chunk = ci
                            break
                    t = _compute_timing_reward_v12(
                        answer_chunk=model_chunk,
                        visible_start_chunk=ask_chunk_q,
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
            # v12.11 P1.3 (2026-05-01) + audit fix #5: aggregate per-rollout
            # behavior counters with class-specific denominators. n_correct_*
            # use gold-class as denominator, not n_chunks_scored.
            _BEHAVIOR_AGG["n_chunks_scored"] += silent_res.get("n_chunks_scored", 0)
            _BEHAVIOR_AGG["n_correct_silent"] += silent_res.get("n_correct_silent", 0)
            _BEHAVIOR_AGG["n_hallucinate"] += silent_res.get("n_hallucinate", 0)
            _BEHAVIOR_AGG["n_missed"] += silent_res.get("n_missed", 0)
            # Compute gold-class denominators + n_correct_response from
            # gold_action_per_chunk + chunk_outputs. silent_quality scoring
            # didn't expose n_correct_response (its score is "0.0 — outcome
            # handles correctness"); we re-derive here.
            by_chunk = {int(o.get("chunk_idx", -1)): o for o in chunk_outputs}
            for ci_str, gold_action in (gold_action_per_chunk or {}).items():
                try:
                    ci = int(ci_str)
                except Exception:
                    continue
                if ci not in by_chunk:
                    continue
                model_kind = by_chunk[ci].get("kind", "unknown")
                model_ans = (by_chunk[ci].get("answer_text") or "").strip()
                model_silent = (model_kind == "answer" and not model_ans)
                model_response = (model_kind == "answer" and bool(model_ans))
                if gold_action in ("silent", "recall_silent"):
                    _BEHAVIOR_AGG["n_gold_silent"] += 1
                elif gold_action in ("response", "recall_response"):
                    _BEHAVIOR_AGG["n_gold_response"] += 1
                    if model_response:
                        _BEHAVIOR_AGG["n_correct_response"] += 1
            # Recall + compress decision usage rate.
            # v12.11 audit-5 P0 #2: recall is now tracked via the explicit
            # is_recall_chunk flag (kind is now the second-pass action,
            # which is "answer" for shape B's final turn).
            for co in chunk_outputs:
                if co.get("is_recall_chunk"):
                    _BEHAVIOR_AGG["n_recall_emitted"] += 1
                k = co.get("kind", "")
                if k == "compress":
                    if not co.get("format_error"):
                        _BEHAVIOR_AGG["n_compress_well_formed"] += 1
                    _BEHAVIOR_AGG["n_compress_emitted"] += 1
                _BEHAVIOR_AGG["n_chunks_total"] += 1

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

# v12.11 P1.3 (2026-05-01): per-step behavior counters aggregated across
# rollouts in the same training step. Reset by grpo_global_metrics after
# emitting. Read by ablation_runner via behavior_* keys in grpo_step.jsonl.
_BEHAVIOR_AGG: Dict[str, int] = {
    "n_chunks_total": 0,
    "n_chunks_scored": 0,
    # v12.11 audit fix #5 (2026-05-01): split denominators to keep
    # behavior_*_acc semantically clean. Previously response/silent acc
    # both used n_chunks_scored as denominator → "mixed correctness", not
    # per-class accuracy. Now we count gold-class denominators separately.
    "n_gold_silent": 0,        # gold action ∈ {silent, recall_silent}
    "n_gold_response": 0,      # gold action ∈ {response, recall_response}
    "n_correct_silent": 0,     # gold-silent ∧ model emits empty answer
    "n_correct_response": 0,   # gold-response ∧ model emits non-empty answer
    "n_hallucinate": 0,        # gold-silent ∧ model talked
    "n_missed": 0,             # gold-response ∧ model silent
    "n_recall_emitted": 0,
    "n_compress_emitted": 0,
    "n_compress_well_formed": 0,
}


def _drain_behavior_metrics() -> Dict[str, float]:
    """Pop & reset _BEHAVIOR_AGG counters; return the behavior_* metric dict.

    v12.11 audit fix #5: per-class accuracies now use the proper class-specific
    denominator (gold-silent / gold-response), not all-scored-chunks. This is
    what ablation_runner needs to compare A0 vs A1 cleanly.
    """
    global _BEHAVIOR_AGG
    n_total = max(1, _BEHAVIOR_AGG["n_chunks_total"])
    n_gold_silent = max(1, _BEHAVIOR_AGG["n_gold_silent"])
    n_gold_response = max(1, _BEHAVIOR_AGG["n_gold_response"])
    n_compress_chunks = max(1, _BEHAVIOR_AGG["n_compress_emitted"])
    out = {
        "behavior_n_chunks_total": _BEHAVIOR_AGG["n_chunks_total"],
        "behavior_n_gold_silent": _BEHAVIOR_AGG["n_gold_silent"],
        "behavior_n_gold_response": _BEHAVIOR_AGG["n_gold_response"],
        # Class-conditional accuracies (proper denominators).
        "behavior_silent_acc": _BEHAVIOR_AGG["n_correct_silent"] / n_gold_silent,
        "behavior_response_acc": _BEHAVIOR_AGG["n_correct_response"] / n_gold_response,
        # Error rates: hallucinate normalized by gold-silent; missed by gold-response.
        "behavior_hallucinate_rate": _BEHAVIOR_AGG["n_hallucinate"] / n_gold_silent,
        "behavior_missed_rate": _BEHAVIOR_AGG["n_missed"] / n_gold_response,
        # Tool usage rates (across all chunks).
        "behavior_recall_used_rate": _BEHAVIOR_AGG["n_recall_emitted"] / n_total,
        "behavior_compress_format_rate": (
            _BEHAVIOR_AGG["n_compress_well_formed"] / n_compress_chunks
        ),
    }
    _BEHAVIOR_AGG = {k: 0 for k in _BEHAVIOR_AGG}
    return out


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
    advantage_mode: Auto[str] = ADVANTAGE_MODE,
) -> Context:
    """Compute per-rollout advantages from the 8-reward dict + masks.

    v12.11: ``advantage_mode`` now defaults to env-driven ADVANTAGE_MODE
    (set via THINKSTREAM_ADVANTAGE_MODE). Three values:

      "gdpo"  — per-reward group-norm + weighted sum + batch-whiten.
      "grpo"  — DeepSeekMath: weighted scalar reward → group z-norm.
      "remem" — gdpo outcome advantage × α blended with per-step state
                advantage × (1-α); requires THINKSTREAM_USE_STATE_ADVANTAGE=1
                AND per-step state rewards present in rewards_dict.

    Pure-tensor algorithms in ``gdpo_advantage.py``.
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
        # v12.13: question text only — options live in <queries> block.
        # Avoids ~30-tok duplication at ask_chunk where queries_state +
        # user_input would both show A-D options.
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


def _build_rollout_messages_single_chunk(
    raw_sample, chunk_result, gen_idx, tokenizer, frames_per_chunk,
):
    """v12.11 (2026-05-01): MemAgent-style per-chunk message builder.

    Returns messages for ONE chunk decision only (3-turn standard, or 5-turn
    shape B for recall multi-turn). This mirrors MemAgent's
    ``MemoryAgent.action()`` which constructs a fresh prompt each step (see
    /tmp/refs/MemAgent/recurrent/impls/memory.py:175). Memory flows across
    chunks via TEXT inside ``step_messages`` (the policy already saw the
    compressed memory state token at rollout time); we simply replay that
    captured prompt and append the chunk's actual generated assistant turn.

    Loss-time KV is bounded by per-chunk prompt length (~3-6K tokens) instead
    of N × that for the legacy concatenated path. This is the same trick
    ReMemR1 uses (verl/trainer/ppo/ray_trainer.py:1278) where each action
    is its own batch entry, indexed by ``step_uid = uid + str(step_id)``.

    The legacy ``_build_rollout_messages`` (concatenated trajectory) is
    retained for diagnostics; not used in the per-chunk build path.

    Args:
        raw_sample: original sample dict (for video_path / data_path).
        chunk_result: one entry from chunk_results (carries step_messages,
            generated_tokens, window_start/end, chunk_idx).
        gen_idx: which group rollout to replay.
        tokenizer: HF tokenizer (for decoding generated_tokens).
        frames_per_chunk: matches FRAMES_PER_CHUNK (used for video_meta).

    Returns:
        (messages, video_meta, video_chunk_size) — same shape as the
        legacy multi-chunk builder, but for a single-chunk slice.
    """
    data_path = raw_sample.get("data_path", "")
    video_path = raw_sample.get("video_path", "")
    abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

    # Pull the captured prompt the policy actually saw for this chunk + gen.
    sm = chunk_result.get("step_messages")
    step_msgs = None
    if isinstance(sm, list) and gen_idx < len(sm):
        v = sm[gen_idx]
        if isinstance(v, list) and v:
            step_msgs = v
    elif isinstance(sm, list) and sm and isinstance(sm[0], dict):
        # Pre-merge legacy format: single list (gen_idx not split yet).
        step_msgs = sm

    if step_msgs is None:
        # Fallback: legacy reconstruction. Will drop <memory>/<queries>
        # context — train/infer logprobs will diverge. Same warning as the
        # multi-chunk builder.
        logger.warning(
            "_build_rollout_messages_single_chunk: chunk %d missing step_messages "
            "for gen %d — using legacy reconstruction (logprob drift).",
            int(chunk_result.get("chunk_idx", -1)), gen_idx,
        )
        question_at_chunk = _extract_questions_at_chunks(raw_sample)
        cur_chunk_idx = int(chunk_result.get("chunk_idx", 0))
        q_text = question_at_chunk.get(cur_chunk_idx)
        user_content: List[Dict] = [{
            "type": "video",
            "video": abs_video_path,
            "video_start": chunk_result["window_start"],
            "video_end": chunk_result["window_end"],
        }]
        if q_text:
            user_content.append({"type": "text", "text": "\n" + q_text})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V12},
            {"role": "user", "content": user_content},
        ]
    else:
        # Replay the captured prompt verbatim. Its system head stays as the
        # FIRST element (we don't dedup since this is a fresh per-chunk
        # conversation, not a concatenation across chunks).
        messages = list(step_msgs)
        if not messages or messages[0].get("role") != "system":
            # Rare case: captured prompt had no system; prepend ours.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_V12}
            ] + messages

    # Append the chunk's actual generated assistant turn (the model's output).
    gen_tokens_list = chunk_result.get("generated_tokens", [])
    if gen_idx < len(gen_tokens_list):
        gt = gen_tokens_list[gen_idx]
        # Tolerate both torch.Tensor and list[int] shapes.
        if hasattr(gt, "tolist"):
            gt = gt.tolist()
        gen_text = tokenizer.decode(gt, skip_special_tokens=False)
    else:
        gen_text = ""
    for _sp in ("<|im_end|>", "<|endoftext|>"):
        if gen_text.endswith(_sp):
            gen_text = gen_text[: -len(_sp)]
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": gen_text}]}
    )

    # v12.11 (2026-05-01) — TWO-step fix for the per-chunk video reconstruction:
    #
    # Original P0.7 (d50e565): set num_chunks=VISUAL_WINDOW_CHUNKS so loader
    # would produce 16 splits of 2 frames each. CAUGHT in audit (this commit):
    # the captured step_messages user turn contains a SINGLE video item with
    # 32 frames, and Qwen3-VL chat-templated text has only ONE <video_pad>
    # placeholder. The processor needs split_videos length == placeholder
    # count → 16 != 1 → tokenization explodes or silently truncates.
    #
    # Correct fix: num_chunks=1 with frames_per_chunk=32 (= VISUAL_WINDOW_CHUNKS
    # × frames_per_chunk_runtime). Loader produces ONE video tensor of 32
    # frames matching the single placeholder. Range still covers the 16s
    # visual_window so MROPE timestamps align with rollout.
    from thinkstream.data.agent_protocol import VISUAL_WINDOW_CHUNKS
    chunk_idx = int(chunk_result.get("chunk_idx", 0))
    chunk_sec = chunk_result["window_end"] - chunk_result["window_start"]
    visual_window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    visual_window_end_exclusive = chunk_idx + 1
    n_window_chunks = visual_window_end_exclusive - visual_window_start

    video_meta = build_video_meta(
        abs_path=abs_video_path,
        total_start=visual_window_start * chunk_sec,
        total_end=visual_window_end_exclusive * chunk_sec,
        num_chunks=1,                                      # ← one video item
        frames_per_chunk=n_window_chunks * frames_per_chunk,  # ← all frames in it
    )
    video_chunk_size = chunk_sec  # per-RoPE-chunk size unchanged
    return messages, video_meta, video_chunk_size


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

    # v12.6: prefer the EXACT per-gen prompt messages captured at rollout
    # time (chunk_results[i]['step_messages'][gen_idx]). The merge step
    # stores step_messages as a list-of-G per chunk to mirror the per-gen
    # generated_tokens schema. If captured for THIS gen_idx, replay those
    # messages; otherwise fall back to legacy reconstruction (which loses
    # memory/visual_window/queries context).
    def _captured_for_gen(cr, gi):
        sm = cr.get("step_messages")
        if isinstance(sm, list) and gi < len(sm):
            v = sm[gi]
            return v if isinstance(v, list) and v else None
        # Backward-compat: pre-merge format had step_messages as a single list
        if isinstance(sm, list) and sm and isinstance(sm[0], dict):
            return sm
        return None

    has_captured_msgs = all(
        _captured_for_gen(cr, gen_idx) is not None for cr in chunk_results
    )

    if has_captured_msgs:
        # Trajectory shape: keep exactly one system turn + each chunk's
        # captured user content + the assistant's actual generation. For
        # multi-turn recall samples, step_messages already includes the
        # [user(chunk N), assistant(recall), user(tool_result)] tail; we
        # only append the FINAL assistant generation per chunk.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_V12}
        ]
        for cr_idx, cr in enumerate(chunk_results):
            step_msgs = _captured_for_gen(cr, gen_idx)
            # Skip the system head from each step's captured prompt
            # (already at messages[0]); append everything else.
            for m in step_msgs:
                if m.get("role") == "system":
                    continue
                messages.append(m)
            # Append assistant generation from the rollout
            gen_text = tokenizer.decode(
                cr["generated_tokens"][gen_idx], skip_special_tokens=False
            )
            for _sp in ("<|im_end|>", "<|endoftext|>"):
                if gen_text.endswith(_sp):
                    gen_text = gen_text[: -len(_sp)]
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": gen_text}]}
            )
    else:
        # Fallback: legacy video+question reconstruction. WARNING: this
        # drops <memory>/<visual_window>/<queries>, breaking train/infer
        # logprob parity. Used only when step_messages is missing.
        logger.warning(
            "_build_rollout_messages: chunk_results missing step_messages; "
            "using legacy reconstruction — logprobs may drift from rollout."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_V12}
        ]
        for cr_idx, cr in enumerate(chunk_results):
            w_start, w_end = cr["window_start"], cr["window_end"]
            user_content: List[Dict] = [
                {
                    "type": "video",
                    "video": abs_video_path,
                    "video_start": w_start,
                    "video_end": w_end,
                }
            ]
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

    # v12.11 (2026-05-01): mode-aware dispatch.
    # - LOSS_BATCH_MODE="trajectory" → legacy concat builder (item carries no chunk_idx)
    # - LOSS_BATCH_MODE="per_chunk"  → MemAgent-style single-chunk builder
    # Both paths share the same downstream tokenization + collation pipeline.
    use_per_chunk = LOSS_BATCH_MODE == "per_chunk"

    for item_desc in micro_items:
        sample_idx = item_desc["sample_idx"]
        gen_idx = item_desc["gen_idx"]
        sample_data = rollout_data[sample_idx]

        if use_per_chunk:
            chunk_idx = item_desc.get("chunk_idx", 0)
            chunk_results = sample_data.get("chunk_results", [])
            if chunk_idx >= len(chunk_results):
                logger.warning(
                    "build_grpo_inputs[per_chunk]: chunk_idx %d out of range for "
                    "sample %d (only %d chunks); skipping.",
                    chunk_idx, sample_idx, len(chunk_results),
                )
                continue
            messages, video_meta, video_chunk_size = (
                _build_rollout_messages_single_chunk(
                    raw_sample=sample_data["raw_sample"],
                    chunk_result=chunk_results[chunk_idx],
                    gen_idx=gen_idx,
                    tokenizer=tokenizer,
                    frames_per_chunk=int(rollout_fpc),
                )
            )
        else:
            # Legacy: concatenate all chunks into one trajectory message list.
            messages, video_meta, video_chunk_size = _build_rollout_messages(
                raw_sample=sample_data["raw_sample"],
                chunk_results=sample_data["chunk_results"],
                gen_idx=gen_idx,
                tokenizer=tokenizer,
                frames_per_chunk=int(rollout_fpc),
            )

        # v12.11 P0.7 fix (2026-05-01): per-chunk path can NOT reuse the
        # trajectory-level preloaded_frames cache. The cache splits the
        # FULL video into N trajectory chunks (1 chunk each); per-chunk
        # loss needs the 16s visual_window (16 chunks) at this chunk's
        # position. Pass preloaded_frames=None so the loader reads from
        # disk using the corrected video_meta range. Trajectory mode keeps
        # the cache (its split aligns with rollout's per-turn videos).
        if use_per_chunk:
            preloaded_for_call = None
        else:
            if sample_idx not in _preloaded_cache:
                pv = sample_data.get("_preloaded_video")
                _preloaded_cache[sample_idx] = (
                    (pv["split_videos"], pv["video_kwargs"], pv["chunk_metadatas"])
                    if pv
                    else None
                )
            preloaded_for_call = _preloaded_cache[sample_idx]

        result = process_messages_to_model_inputs(
            messages=messages,
            video_meta=video_meta,
            video_chunk_size=video_chunk_size,
            processor=processor,
            model_type=model_type,
            add_generation_prompt=False,
            preloaded_frames=preloaded_for_call,
        )
        result["position_ids"] = compute_position_ids(result, processor, model_type)

        # Length guard. trajectory mode warns at 50K (legacy concat); per_chunk
        # mode warns if anything > 85% (should never happen with 6K samples).
        seq_len = int(result["input_ids"].shape[-1])
        max_len = int(getattr(tokenizer, "model_max_length", 16384) or 16384)
        n_chunks = len(sample_data.get("chunk_results", []))
        if use_per_chunk:
            chunk_idx = item_desc.get("chunk_idx", 0)
            if seq_len > max_len:
                logger.warning(
                    "GRPO per-chunk sample %d/gen %d/chunk %d exceeds "
                    "model_max_length (seq_len=%d > %d). Per-chunk should never "
                    "exceed; check visual/memory config.",
                    sample_idx, gen_idx, chunk_idx, seq_len, max_len,
                )
            elif seq_len > 0.85 * max_len:
                logger.info(
                    "GRPO per-chunk sample %d/gen %d/chunk %d at %.0f%% of cutoff "
                    "(seq_len=%d, max=%d).",
                    sample_idx, gen_idx, chunk_idx,
                    100 * seq_len / max_len, seq_len, max_len,
                )
        else:
            if seq_len > max_len:
                logger.warning(
                    "GRPO trajectory sample exceeds model_max_length "
                    "(seq_len=%d > %d) over %d chunks — collator will truncate, "
                    "completion_mask + ref logprobs on truncated chunks will be "
                    "DROPPED. Lower rollout_max_chunks or switch to "
                    "THINKSTREAM_LOSS_BATCH_MODE=per_chunk.",
                    seq_len, max_len, n_chunks,
                )
            elif seq_len > 0.85 * max_len:
                logger.info(
                    "GRPO trajectory sample at %.0f%% of cutoff (seq_len=%d, "
                    "max=%d, n_chunks=%d) — close to truncation threshold.",
                    100 * seq_len / max_len, seq_len, max_len, n_chunks,
                )

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
    metrics = {
        "grad_norm": grad_norm,
        "learning_rate": lr,
        "reward_mean": reward_mean,
        "reward_var": reward_var,
        **component_means,
        **dict(_LAST_GDPO_DIAG),
        # v12.11 P1.3: drain per-step behavior counters → behavior_* keys
        # consumed by ablation_runner.compute_summary.
        **_drain_behavior_metrics(),
    }

    # v12.6: persist per-step metrics to grpo_step.jsonl (audit writer set
    # by env THINKSTREAM_AUDIT_DIR / THINKSTREAM_OUTPUT_DIR). Without this
    # ablation_runner.py has nothing to read for per-step convergence
    # comparison. Step counter is module-global so we tag each row.
    global _GRPO_STEP_COUNTER
    _GRPO_STEP_COUNTER += 1
    step_writer, _ = _grpo_audit_writers()
    if step_writer is not None:
        step_writer.write({"step": _GRPO_STEP_COUNTER, **metrics})

    return metrics


def _format_score_for_chunk(text: str) -> float:
    """ReMemR1-style format reward (metric_utils.py:86): 1.0 if parsable."""
    from thinkstream.data.agent_protocol import parse_agent_output_v12
    parsed = parse_agent_output_v12(text or "")
    return 0.0 if parsed.get("format_error") else 1.0


def _model_action_kind(text: str) -> str:
    """Parse model output → coarse action label: silent/response/recall/compress/unknown."""
    from thinkstream.data.agent_protocol import parse_agent_output_v12
    p = parse_agent_output_v12(text or "")
    kind = p.get("kind", "unknown")
    if kind == "answer":
        ans = (p.get("answer_text") or "").strip()
        return "response" if ans else "silent"
    if kind in ("recall", "compress"):
        return kind
    return "unknown"


def _action_match_score(model_kind: str, gold_action: str) -> float:
    """Per-chunk teacher-action match reward.

    Aligns with v12_rewards.compute_per_chunk_silent_quality_v12 weights but
    extends to all 4 action types instead of just silent/response.

      both silent    → +1.0   (correct silence)
      both response  → +0.5   (full credit handled by outcome reward)
      both recall    → +0.5   (recall_quality reward handles details)
      both compress  → +0.5
      gold silent / model talked → -0.5  (false positive)
      gold response / model silent → -0.5  (false negative)
      mismatches across action types → 0.0
    """
    if not gold_action:
        return 0.0
    g = gold_action.lower()
    if g in ("silent", "recall_silent"):
        return 1.0 if model_kind == "silent" else -0.5
    if g in ("response", "recall_response"):
        return 0.5 if model_kind == "response" else -0.5
    if g == "recall":
        return 0.5 if model_kind == "recall" else 0.0
    if g == "compress":
        return 0.5 if model_kind == "compress" else 0.0
    return 0.0


def _word_recall_increment(generated: str, ground_truth_words: List[str]) -> float:
    """ReMemR1 metric_utils.py:139 word-level recall: fraction of GT tokens present."""
    if not ground_truth_words:
        return 0.0
    text = (generated or "").lower()
    hits = sum(1 for w in ground_truth_words if w.lower() in text)
    return hits / len(ground_truth_words)


def _gold_action_at(raw_sample: Dict, chunk_idx: int) -> str:
    """Pull teacher's expected action for this chunk from raw_sample.

    Trajectory data carries `gold_action_per_chunk` as a dict {str(chunk_idx): action}.
    Flat data may have it under top-level or under each card. Returns "" if absent.
    """
    gap = raw_sample.get("gold_action_per_chunk")
    if isinstance(gap, dict):
        v = gap.get(str(chunk_idx)) or gap.get(chunk_idx)
        if v:
            return str(v)
    # Fallback: try sample_type field on the matching chunk record.
    return ""


def _estimate_chunk_token_len(chunk_result: Dict, gen_idx: int) -> int:
    """v12.11 P1.2: cheap pre-tokenization estimate for dynamic bsz packing.

    No actual tokenization — just heuristic from step_messages structure.
    Recall multi-turn (shape B with 5 message turns) samples are ~30% longer
    than non-recall (3 turns).

    Tunable estimates calibrated to v12.10 production sampling:
      base       (3-turn): ~5500 tokens (system+tools+visual_window+memory+ans)
      recall     (5-turn): ~8000 tokens (adds <recall_result> + recalled frames)
    """
    sm = chunk_result.get("step_messages")
    is_recall = False
    if isinstance(sm, list) and gen_idx < len(sm):
        v = sm[gen_idx]
        if isinstance(v, list) and len(v) > 3:
            is_recall = True
    return 8000 if is_recall else 5500


def _greedy_pack_by_token_budget(
    seqlens: List[int], max_token_len: int,
) -> List[List[int]]:
    """v12.11 P1.2: greedy bin-pack items into batches by total token budget.

    Approximates MemAgent's `rearrange_micro_batches` (Karmarkar-Karp) with
    a simpler best-fit decreasing greedy. For per-chunk batching with
    relatively uniform item sizes, the greedy gets within 5-10% of optimal.

    Args:
        seqlens: per-item estimated seq length
        max_token_len: total budget per batch (≥ max(seqlens))

    Returns:
        list of batches; each batch is a list of original indices.

    Algorithm (best-fit decreasing):
      1. Sort items by length descending
      2. For each item, place in the LIGHTEST existing batch that still
         has room. If none, open a new batch.
    """
    if not seqlens:
        return []
    if max_token_len < max(seqlens):
        # Single item exceeds budget — treat each item as its own batch.
        return [[i] for i in range(len(seqlens))]
    sorted_idx = sorted(range(len(seqlens)), key=lambda i: -seqlens[i])
    bins: List[Tuple[List[int], int]] = []  # (indices, total_seqlen)
    for idx in sorted_idx:
        s = seqlens[idx]
        # Find lightest bin that fits.
        best = None
        for b_idx in range(len(bins)):
            if bins[b_idx][1] + s <= max_token_len:
                if best is None or bins[b_idx][1] < bins[best][1]:
                    best = b_idx
        if best is not None:
            bins[best] = (bins[best][0] + [idx], bins[best][1] + s)
        else:
            bins.append(([idx], s))
    # Restore original-order indexing within each bin.
    return [sorted(b[0]) for b in bins]


def _per_chunk_state_reward(flat_items, rollout_data, tokenizer, mode: str):
    """v12.11 P1.1 full: per-chunk state reward in 4 selectable modes.

    Mode dispatch:
      "format_only"   — format reward only (ReMemR1 lite). Returns 0/1.
      "format_action" — format + teacher-action match. Returns ∈ [-0.5, 2.0].
      "remem_full"    — format + action + word-level recall increment for
                        recall/response steps. Returns ∈ [-0.5, 3.0].
      "with_silent_q" — format + action + per-chunk silent_quality fold-in
                        (our streaming-specific hallucinate/miss penalty).
                        Returns ∈ [-1.1, 2.3].

    Direct port of /tmp/refs/ReMemR1/verl/trainer/ppo/metric_utils.py:86,134
    with extensions for streaming-video specific signals.

    The mode is read from THINKSTREAM_STATE_REWARD_MODE; this function
    accepts the resolved `mode` string for unit-testability.
    """
    out = []
    for it in flat_items:
        s = it["sample_idx"]; g = it["gen_idx"]; c = it["chunk_idx"]
        # v12.11 P0.3 fix: rollout_data is List, not Dict.
        sample_data = rollout_data[s] if s < len(rollout_data) else {}
        chunks = sample_data.get("chunk_results", []) if sample_data else []
        if c >= len(chunks):
            out.append(0.0); continue
        gt_list = chunks[c].get("generated_tokens", [])
        if g >= len(gt_list):
            out.append(0.0); continue
        gt = gt_list[g]
        if hasattr(gt, "tolist"):
            gt = gt.tolist()
        if not gt:
            out.append(0.0); continue
        text = tokenizer.decode(gt, skip_special_tokens=False)

        # 1. format component (always applied)
        score = _format_score_for_chunk(text)
        if mode == "format_only":
            out.append(score); continue

        # 2. action match component
        raw_sample = sample_data.get("raw_sample", {})
        gold = _gold_action_at(raw_sample, chunks[c].get("chunk_idx", c))
        model_kind = _model_action_kind(text)
        # v12.11 review-fix (2026-05-01): for shape-B recall chunks the
        # generated_tokens text is the SECOND-pass answer (kind="answer"),
        # but the chunk's true semantic action was "recall+answer". The
        # raw text alone can't reveal this; inject the upstream recall
        # signal (recall_first_pass_text on chunk_result) so action_match
        # recognizes recall trajectories instead of treating them as
        # plain "response".
        first_pass_text = chunks[c].get("recall_first_pass_text", "") or ""
        if isinstance(first_pass_text, list):
            first_pass_text = first_pass_text[g] if g < len(first_pass_text) else ""
        if first_pass_text:
            from thinkstream.data.agent_protocol import parse_agent_output_v12
            fp = parse_agent_output_v12(first_pass_text)
            if fp.get("kind") == "recall":
                model_kind = "recall"
        score += _action_match_score(model_kind, gold)

        if mode == "format_action":
            out.append(score); continue

        # 3. word-level recall increment (ReMemR1 full)
        if mode == "remem_full":
            gold_answer = raw_sample.get("gold_answer") or raw_sample.get("answer") or ""
            gt_words = [w for w in str(gold_answer).split() if w.strip()]
            if model_kind in ("response", "recall") and gt_words:
                score += _word_recall_increment(text, gt_words)
            out.append(score); continue

        # 4. silent_quality fold-in (streaming-specific)
        if mode == "with_silent_q":
            from thinkstream.trainer.v12_rewards import (
                compute_per_chunk_silent_quality_v12 as _per_chunk_sq,
            )
            from thinkstream.data.agent_protocol import parse_agent_output_v12
            parsed_out = parse_agent_output_v12(text)
            sq_input = [{
                "chunk_idx": chunks[c].get("chunk_idx", c),
                "kind": parsed_out.get("kind", "unknown"),
                "answer_text": parsed_out.get("answer_text", ""),
            }]
            gap = raw_sample.get("gold_action_per_chunk", {}) or {}
            try:
                sq = _per_chunk_sq(sq_input, gap)
                score += float(sq.get("silent_quality", 0.0))
            except Exception:
                pass
            out.append(score); continue

        # Unknown mode → fall back to format_only score
        out.append(_format_score_for_chunk(text))

    return torch.tensor(out, dtype=torch.float)


@node
def prepare_grpo_micro_batches(
    ctx: Context,
    /,
    *,
    advantages: Auto[torch.Tensor],
    rewards: Auto[torch.Tensor],
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    rollout_data: Auto[Dict[str, Any]],
    tokenizer: Auto[Any],
    micro_batch_size: Auto[int],
    group_size: Auto[int],
    step_advantages: Ref[torch.Tensor],
    step_micro_rewards: Ref[torch.Tensor],
    step_micro_rewards_dict: Ref[Dict[str, torch.Tensor]],
    step_micro_items: Ref[List],
    step_micro_batches: Ref[list[dict[Ref, Any]]],
) -> Context:
    """v12.11 (2026-05-01): switchable micro-batching mode.

    LOSS_BATCH_MODE="per_chunk" (DEFAULT, MemAgent-style):
        1 item = 1 (sample_idx, gen_idx, chunk_idx) chunk decision. N chunks
        become N independent loss-batch entries, each ~6K tokens. Trajectory
        advantage broadcasts to every chunk; ReMemR1 state advantage can be
        added on top via USE_STATE_ADVANTAGE / ADVANTAGE_MODE=remem.
        Eliminates OOM + truncation on long-trajectory rollouts. Default
        flipped from "trajectory" in v12.11 (audit-1 P0.2) once the path's
        bugs were fixed (rollout_data list handling, recall token swap,
        video_meta consistency, recall first-pass merge).

    LOSS_BATCH_MODE="trajectory" (legacy):
        1 item = 1 (sample_idx, gen_idx) trajectory rollout. Original
        v12.10 behavior — produces concatenated N-chunk samples that may
        exceed model_max_length on long trajectories. Opt-in for ablation
        comparison against the legacy baseline.

    Switch via env: THINKSTREAM_LOSS_BATCH_MODE=trajectory (to reproduce
    legacy) or per_chunk (default).

    Operator notes for per_chunk mode:
        - micro_batch_size now counts CHUNKS, not trajectories.
        - For 8 H20 + 8B + 6K avg chunk: micro_batch=4-8 is comfortable.
        - The trainer logs the expansion count.
    """
    total_samples = advantages.shape[0]  # = num_videos × group_size

    if LOSS_BATCH_MODE == "trajectory":
        # ─── LEGACY PATH ─────────────────────────────────────────────────
        # 1 item per (sample_idx, gen_idx). Original v12.10 behavior.
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
        logger.info(
            "GRPO micro-batches mode=trajectory (legacy): %d items → %d batches × %d",
            total_samples, num_micro_batches, micro_batch_size,
        )
        return ctx.set(step_micro_batches, micro_batches)

    # ─── PER-CHUNK PATH (v12.11) ─────────────────────────────────────────
    # Build flat list of per-chunk items by enumerating chunk_results from
    # each (sample_idx, gen_idx) pair. Chunk count varies per trajectory
    # (depends on rollout_max_chunks + early-stop).
    flat_items: List[Dict[str, int]] = []
    flat_advantages: List[torch.Tensor] = []
    flat_rewards: List[torch.Tensor] = []
    flat_rewards_dict: Dict[str, List[torch.Tensor]] = {k: [] for k in rewards_dict}
    # v12.11 P0.3 fix: rollout_data is List[dict] (one entry per video),
    # not a dict — earlier `if sample_idx not in rollout_data` was checking
    # membership against the LIST and silently skipping every chunk.
    for flat_idx in range(total_samples):
        sample_idx = flat_idx // group_size
        gen_idx = flat_idx % group_size
        if sample_idx >= len(rollout_data):
            continue
        sample_data = rollout_data[sample_idx]
        chunk_results = sample_data.get("chunk_results", [])
        active_chunk_count = 0
        for cr in chunk_results:
            gt_list = cr.get("generated_tokens", [])
            if gen_idx < len(gt_list):
                gt = gt_list[gen_idx]
                length = len(gt) if not hasattr(gt, "numel") else int(gt.numel())
                if length > 0:
                    active_chunk_count += 1
        active_chunk_count = max(1, active_chunk_count)
        for chunk_idx in range(active_chunk_count):
            flat_items.append({
                "sample_idx": sample_idx,
                "gen_idx": gen_idx,
                "chunk_idx": chunk_idx,
            })
            flat_advantages.append(advantages[flat_idx:flat_idx + 1])
            flat_rewards.append(rewards[flat_idx:flat_idx + 1])
            for k in rewards_dict:
                flat_rewards_dict[k].append(rewards_dict[k][flat_idx:flat_idx + 1])

    if not flat_items:
        flat_items = [
            {"sample_idx": flat_idx // group_size, "gen_idx": flat_idx % group_size, "chunk_idx": 0}
            for flat_idx in range(total_samples)
        ]
        flat_adv_tensor = advantages
        flat_rew_tensor = rewards
        flat_rd = rewards_dict
    else:
        flat_adv_tensor = torch.cat(flat_advantages, dim=0)
        flat_rew_tensor = torch.cat(flat_rewards, dim=0)
        flat_rd = {k: torch.cat(v, dim=0) for k, v in flat_rewards_dict.items()}

    # ─── v12.11 P1.1: ReMemR1 mixed advantage ─────────────────────────────
    # When ADVANTAGE_MODE=remem AND USE_STATE_ADVANTAGE=1, replace the
    # broadcast trajectory advantage with α·outcome_adv + (1-α)·state_adv
    # (line-by-line port of ReMemR1 ray_trainer.py:1287-1314 via the
    # already-existing compute_mixed_advantage_v12 helper).
    if ADVANTAGE_MODE == "remem" and USE_STATE_ADVANTAGE and flat_items:
        try:
            video_uid_per_row = [str(it["sample_idx"]) for it in flat_items]
            chunk_idx_per_row = [int(it["chunk_idx"]) for it in flat_items]
            # outcome reward = trajectory's broadcast outcome reward.
            # rewards is [num_traj]; sample at flat_idx = sample_idx*group_size + gen_idx
            outcome_rew = torch.stack([
                rewards[it["sample_idx"] * group_size + it["gen_idx"]]
                for it in flat_items
            ])
            state_rew = _per_chunk_state_reward(
                flat_items, rollout_data, tokenizer, mode=STATE_REWARD_MODE,
            )
            # Use the canonical port from v12_rollout (line 393).
            mixed_adv = _compute_mixed_advantage_v12_remem(
                outcome_reward=outcome_rew,
                state_reward=state_rew,
                video_uid_per_row=video_uid_per_row,
                chunk_idx_per_row=chunk_idx_per_row,
                alpha=STATE_ADVANTAGE_ALPHA,
                use_adv=True,
            )
            flat_adv_tensor = mixed_adv
            logger.info(
                "GRPO advantage mode=remem state=%s: α=%.2f outcome + (1-α) "
                "state over %d per-chunk items (state mean=%.3f, std=%.3f).",
                STATE_REWARD_MODE, STATE_ADVANTAGE_ALPHA, len(flat_items),
                float(state_rew.mean()), float(state_rew.std() + 1e-9),
            )
        except Exception as e:
            logger.warning(
                "GRPO ReMemR1 mixed advantage failed (%s); falling back to "
                "broadcast trajectory advantage.", e,
            )

    total_chunk_items = len(flat_items)

    # ─── v12.11 P1.2: dynamic-bsz token packing ─────────────────────────
    # When USE_DYNAMIC_BSZ=1, bin-pack chunk items into batches by total
    # estimated token count instead of fixed item count. Mirrors MemAgent's
    # rearrange_micro_batches (verl/utils/seqlen_balancing.py:216) but at
    # item granularity (pre-tokenization estimate). Eliminates the OOM
    # spikes from variable-length chunks (recall multi-turn ~8K vs base
    # ~5.5K) clumping into the same micro batch.
    if USE_DYNAMIC_BSZ and flat_items:
        seqlens = []
        for it in flat_items:
            # v12.11 P0.3 fix: rollout_data is List, not Dict.
            sample_data = (
                rollout_data[it["sample_idx"]]
                if it["sample_idx"] < len(rollout_data)
                else {}
            )
            chunks = sample_data.get("chunk_results", []) if sample_data else []
            if it["chunk_idx"] < len(chunks):
                seqlens.append(_estimate_chunk_token_len(
                    chunks[it["chunk_idx"]], it["gen_idx"]
                ))
            else:
                seqlens.append(5500)
        partitions = _greedy_pack_by_token_budget(
            seqlens, DYNAMIC_BSZ_MAX_TOKEN_LEN,
        )
        micro_batches = []
        for part_idxs in partitions:
            sel = torch.tensor(part_idxs, dtype=torch.long)
            mb_updates = {
                step_advantages: flat_adv_tensor[sel],
                step_micro_rewards: flat_rew_tensor[sel],
                step_micro_rewards_dict: {
                    k: v[sel] for k, v in flat_rd.items()
                },
                step_micro_items: [flat_items[i] for i in part_idxs],
            }
            micro_batches.append(mb_updates)
        logger.info(
            "GRPO micro-batches mode=per_chunk dynamic-bsz: %d items → "
            "%d batches (max_token_len=%d, sizes=%s, total_tokens=%d)",
            total_chunk_items, len(partitions), DYNAMIC_BSZ_MAX_TOKEN_LEN,
            [len(p) for p in partitions[:8]] + (["..."] if len(partitions) > 8 else []),
            sum(seqlens),
        )
        return ctx.set(step_micro_batches, micro_batches)

    # ─── Fixed micro_batch_size path (legacy / dynamic-bsz off) ─────────
    num_micro_batches = math.ceil(total_chunk_items / micro_batch_size)
    micro_batches = []
    for mb_idx in range(num_micro_batches):
        start_idx = mb_idx * micro_batch_size
        end_idx = min(start_idx + micro_batch_size, total_chunk_items)
        micro_items = flat_items[start_idx:end_idx]
        mb_updates = {
            step_advantages: flat_adv_tensor[start_idx:end_idx],
            step_micro_rewards: flat_rew_tensor[start_idx:end_idx],
            step_micro_rewards_dict: {
                k: v[start_idx:end_idx] for k, v in flat_rd.items()
            },
            step_micro_items: micro_items,
        }
        micro_batches.append(mb_updates)

    logger.info(
        "GRPO micro-batches mode=per_chunk: %d trajectories × group %d = %d "
        "rollouts → %d per-chunk items → %d micro-batches × %d",
        total_samples // group_size, group_size, total_samples,
        total_chunk_items, num_micro_batches, micro_batch_size,
    )
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
    vllm_rollout_frames_root: Auto[Optional[str]] = None,
    vllm_rollout_video_root: Auto[Optional[str]] = None,
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
        frames_root=vllm_rollout_frames_root,
        video_root=vllm_rollout_video_root,
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
