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
    SYSTEM_PROMPT,
    QWEN_TEMPLATE_WO_SYSTEM,
    _make_abs_paths,
    build_video_meta,
    process_messages_to_model_inputs,
    pad_and_cat,
    find_assistant_spans,
    compute_position_ids,
    make_raw_data_module,
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

# ---------------------------------------------------------------------------
# Reward helper functions (Unchanged from original grpo.py)
# ---------------------------------------------------------------------------

_CHUNK_FORMAT_RE = re.compile(
    r"^<think>.*?</think>(?:<response>.*|<silent>)<\|im_end\|>$",
    re.DOTALL,
)

# 4-action agent format regex (per-timestep)
_CHUNK_FORMAT_RE_AGENT = re.compile(
    r"^<think>.*?</think>"
    r"<action>(?:silent|response|recall|compress)</action>"
    r"(?:<response>.*?</response>"
    r"|<query>\{.*?\}</query>"
    r"|<summary>\{.*?\}</summary>)?"
    r"<\|im_end\|>$",
    re.DOTALL,
)

_ACTION_RE = re.compile(r"<action>(silent|response|recall|compress)</action>")

_RESPONSE_RE_AGENT = re.compile(
    r"<action>response</action><response>(.*?)</response>", re.DOTALL
)

_SUMMARY_RE = re.compile(
    r"<action>compress</action><summary>(.*?)</summary>", re.DOTALL
)


def _check_chunk_format(text: str, agent_mode: bool = False) -> bool:
    """Return *True* if a single chunk's generated text matches the format."""
    regex = _CHUNK_FORMAT_RE_AGENT if agent_mode else _CHUNK_FORMAT_RE
    return regex.match(text.strip()) is not None


def _extract_action(text: str) -> str:
    """Extract action type from a generated chunk. Returns 'silent'/'response'/'recall'/'compress'/'unknown'."""
    m = _ACTION_RE.search(text)
    return m.group(1) if m else "unknown"


def _compute_response_reward(
    model_answer: Optional[str],
    gt_answer: str,
    predicted_actions: List[str],
    answer_form: str = "",
) -> float:
    """Primary reward: did the model eventually produce a correct response?

    Unlike the old _compute_action_reward, this does NOT penalize the
    specific action path (recall vs direct response). The model is free
    to choose any path as long as the final answer is correct.

    Scoring:
      correct response: 1.0
      incorrect response: 0.1 (tried but wrong)
      no response (all silent): 0.0
      response when should be silent: -0.2 (over-response penalty)
    """
    has_response = "response" in predicted_actions

    if not gt_answer:
        # No gold answer → this is a silent-only sample
        return -0.2 if has_response else 1.0

    if not has_response:
        return 0.0  # should have responded but didn't

    if not model_answer:
        return 0.1  # responded but couldn't extract answer

    # Compare answer
    gt_clean = _extract_literal_answer(gt_answer)
    if gt_clean and model_answer == gt_clean:
        return 1.0
    # Fuzzy match for descriptive answers
    if answer_form == "descriptive":
        # Keyword overlap as partial credit
        gt_words = set(gt_answer.lower().split())
        ans_words = set(model_answer.lower().split())
        overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
        return min(1.0, overlap * 1.5)  # scale up, cap at 1.0
    return 0.1  # wrong answer


def _compute_recall_hit_rate(
    returned_chunks_per_chunk: List[List[int]],
    support_chunks: List[int],
    *,
    recall_fired: bool = False,
) -> Optional[float]:
    """Recall@K hit-rate over the rollout.

    Returns |∪returned ∩ support| / |support| ∈ [0, 1] when the rollout
    actually recalled and got results AND support is known.

    v11.4 bug fix: previously returned None when `union` was empty,
    silently masking out the most informative failure case ("model
    fired recall but the retriever returned nothing"). That made the
    column mask=0 for the worst-query case and the policy never
    learned to write better queries. Now:
      - support unknown:                    None (genuine "no signal")
      - recall_fired AND union empty:       -0.2 (explicit penalty —
                                             "your query retrieved
                                             nothing useful")
      - !recall_fired AND union empty:      None (sample didn't recall;
                                             nothing to score)
      - non-empty union:                    standard hit-rate

    Union across all recall events in the rollout — the agent can call
    recall multiple times and we credit any match.
    """
    if not support_chunks:
        return None
    union = set()
    for chunks in returned_chunks_per_chunk:
        if chunks:
            union.update(int(c) for c in chunks)
    if not union:
        # v11.4: distinguish "didn't try" (None / mask off) from
        # "tried and failed" (explicit penalty so advantage learns).
        return -0.2 if recall_fired else None
    gold = set(int(c) for c in support_chunks)
    return len(union & gold) / len(gold)


def _compute_recall_quality_reward(
    predicted_actions: List[str],
    chunk_texts: List[str],
    gt_answer: str,
    model_answer: Optional[str],
) -> float:
    """Query format quality (JSON well-formed + no answer leakage).

    Final ∈ [-0.3, 1.0]:
      recall fired → outcome × query_quality
        outcome = 1.0 if response correct, else 0.5
        query_quality = 1.0 (default), -0.3 if no `query`, -0.5 if leak,
                        0.2 if JSON failed
      no recall + correct response: 0.8
      no recall + wrong response  : 0.0
      silent-only sample + recall : -0.3   (unnecessary)
      silent-only sample no recall: 1.0

    Hit-rate and range-tightness are now separate reward columns
    (`recall_hit_rate`, `range_tightness`) so per-reward group-norm can
    normalise each independently.
    """
    used_recall = "recall" in predicted_actions
    if not gt_answer:
        return -0.3 if used_recall else 1.0
    if not used_recall:
        gt_clean = _extract_literal_answer(gt_answer) if gt_answer else None
        if gt_clean and model_answer == gt_clean:
            return 0.8
        return 0.0

    query_quality = 0.5  # default when no <query> tag found at all
    for text in chunk_texts:
        query_match = re.search(r'<query>(.*?)</query>', text, re.DOTALL)
        if query_match:
            try:
                q = json.loads(query_match.group(1))
                has_query = bool(q.get("query", ""))
                query_text = q.get("query", "").lower()
                answer_in_query = gt_answer.lower() in query_text if gt_answer else False
                query_quality = 1.0
                if not has_query:
                    query_quality -= 0.3
                if answer_in_query:
                    query_quality -= 0.5
            except (json.JSONDecodeError, ValueError):
                query_quality = 0.2  # bad JSON

    gt_clean = _extract_literal_answer(gt_answer) if gt_answer else None
    answer_correct = (model_answer == gt_clean) if gt_clean and model_answer else False
    return query_quality if answer_correct else query_quality * 0.5


def _parse_query_time_range(chunk_texts: List[str]):
    """Extract the LAST <query>...</query>'s time_range from rollout texts.

    Returns (t_start, t_end) tuple or None when no query / no time_range /
    malformed. Only the last query is used because earlier ones may belong
    to prior recall events whose results were discarded.
    """
    last = None
    for text in chunk_texts:
        for m in re.finditer(r'<query>(.*?)</query>', text, re.DOTALL):
            try:
                q = json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                continue
            tr = q.get("time_range")
            if not tr:
                continue
            if isinstance(tr, str) and "-" in tr:
                try:
                    a, b = tr.split("-", 1)
                    last = (float(a), float(b))
                except (ValueError, AttributeError):
                    continue
            elif isinstance(tr, (list, tuple)) and len(tr) == 2:
                try:
                    last = (float(tr[0]), float(tr[1]))
                except (TypeError, ValueError):
                    continue
    return last


def _compute_range_tightness_reward(
    chunk_texts: List[str],
    support_chunks: Optional[List[int]],
    video_duration: float,
    chunk_sec: float = 2.0,
) -> Optional[float]:
    """Reward narrow + accurate time_range.

    Only fires when (a) the rollout's last <query> had a time_range, (b)
    we know support_chunks, and (c) the range covers ≥1 support chunk
    (no point rewarding "tight" if it missed). Otherwise returns None
    (mask=0 in the GDPO column).

    score = (1 - range_width / video_duration) × coverage
    where coverage = |support ∩ range| / |support|.

    A range covering all of support but spanning the whole video gets
    score ≈ 0; a range exactly matching support gets close to 1.
    """
    if not support_chunks or video_duration <= 0:
        return None
    tr = _parse_query_time_range(chunk_texts)
    if tr is None:
        return None
    t0, t1 = (min(tr), max(tr))
    range_width = max(0.0, t1 - t0)
    if range_width <= 0:
        return None
    # support coverage by the range
    support_set = set(int(c) for c in support_chunks)
    covered = sum(
        1 for c in support_set
        if (c * chunk_sec + chunk_sec) > t0 and (c * chunk_sec) < t1
    )
    coverage = covered / max(len(support_set), 1)
    if coverage <= 0:
        # v11.4 bug fix: previously returned None and the column was
        # masked out — but coverage=0 is the WORST failure case (model
        # asked for a window that doesn't contain any gold chunk) and
        # is precisely what RL should learn to avoid. Returning a
        # negative score makes the failure visible to advantage.
        return -0.2
    tightness = max(0.0, 1.0 - range_width / video_duration)
    return tightness * coverage


def _compute_format_reward(chunk_texts: List[str], agent_mode: bool = False) -> float:
    """Return format reward in [0, 1]: proportion of chunks that match format."""
    if not chunk_texts:
        return 0.0
    correct_count = sum(1 for t in chunk_texts if _check_chunk_format(t, agent_mode))
    return correct_count / len(chunk_texts)


def _collect_think_lengths(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> List[int]:
    """Collect token lengths of <think>...</think> spans for one (chunk_results, gen_idx).
    One length per chunk that contains a think block.

    Retained for diagnostics/audit only — no longer feeds a reward in v11
    (think_len reward removed: redundant with format check, hard to tune).
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


def _extract_literal_answer(text: str) -> Optional[str]:
    text = text.strip()
    if re.fullmatch(r"[A-E]", text):
        return text
    if re.fullmatch(r"\([A-E]\)", text):
        return text[1]
    if re.fullmatch(r"[A-E]\.", text):
        return text[0]
    if text.lower() in {"yes", "no"}:
        return text.lower()
    if re.fullmatch(r"[0-9]", text):
        return text
    return None


_RESPONSE_RE = re.compile(r"<response>(.*?)(?:<\|im_end\|>|$)", re.DOTALL)


def _scan_responses_for_answer(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> Tuple[Optional[str], Optional[int], int, Optional[int]]:
    """Walk all chunks for one rollout, return both first and last response info.

    v11.4: tuple extended with `last_response_chunk_idx`. Streaming agents
    often emit response then refine in later chunks; the legacy "first
    response only" timing reward penalized this self-correction. The
    caller now uses `last_response_chunk_idx` for timing (the model's
    final committed response) while `first_answer` still gates
    correctness (so a model that keeps spamming responses can't hide a
    wrong-then-right pattern).

    Returns: (first_answer, first_response_chunk_idx, response_count,
              last_response_chunk_idx)
    """
    first_response_chunk_idx: Optional[int] = None
    first_answer: Optional[str] = None
    last_response_chunk_idx: Optional[int] = None
    response_count = 0
    for cr in chunk_results:
        gen_tokens_list = cr.get("generated_tokens", [])
        if gen_idx >= len(gen_tokens_list):
            continue
        gen_tokens = gen_tokens_list[gen_idx]
        if isinstance(gen_tokens, torch.Tensor):
            gen_tokens = gen_tokens.tolist()
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        for m in _RESPONSE_RE.finditer(text):
            response_count += 1
            answer = _extract_literal_answer(m.group(1))
            if answer is not None:
                if first_answer is None:
                    first_answer = answer
                    if first_response_chunk_idx is None:
                        first_response_chunk_idx = cr["chunk_idx"]
                # last_response_chunk_idx tracks the latest chunk that
                # produced ANY parseable response (refinement-aware).
                last_response_chunk_idx = cr["chunk_idx"]
    return (first_answer, first_response_chunk_idx,
            response_count, last_response_chunk_idx)


def _compute_time_reward(
    response_chunk_idx: Optional[int],
    gt_chunk_idx: int,
    window: int,
    slack_window: int = 0,
) -> float:
    if response_chunk_idx is None:
        return 0.0
    diff = abs(response_chunk_idx - gt_chunk_idx)
    if diff <= slack_window:
        return 1.0
    if diff <= slack_window + window:
        return 1.0 - (diff - slack_window) / window
    return 0.0


# NOTE (v9.4 cleanup): the formerly-defined `_compute_correctness_reward`
# (literal-only, returns 0.0 for descriptive) was DEAD CODE — `calc_rewards`
# at L900 calls `_compute_response_reward` instead, which already handles
# descriptive via fuzzy keyword overlap (lines 179-184). The dead helper
# was removed to prevent future drift; if a literal-only correctness signal
# is needed, call `_extract_literal_answer` directly at the call site.


def _compute_num_response_reward(
    num_responses: int, step_window: int = 3, max_responses: int = 10
) -> float:
    """
    Compute a discrete, step-wise reward for the number of responses.
    Exactly 1 response yields 1.0.
    Multiple responses decay in intervals of `step_window`, reaching 0.0
    when exceeding `max_responses`.
    """
    if num_responses == 1:
        return 1.0
    if num_responses <= 0 or num_responses > max_responses:
        return 0.0
    step_window = max(1, step_window)
    step_idx = int((num_responses - 2) // step_window) + 1
    max_steps = int((max_responses - 2) // step_window) + 1
    reward = 1.0 - (float(step_idx) / float(max_steps + 1))
    return max(0.0, reward)


# ---------------------------------------------------------------------------
# New Nodes for GRPO adapted for DeepSlyme
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
    gen_model.config.text_config._attn_implementation = "flash_attention_2_infer"
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

        # Extract task info
        metadata = raw_sample.get("metadata", {})
        ask_chunk = raw_sample.get("chunk_idx", rollout_max_chunks - 1)

        # Extract user question (new format: input.user_input; legacy: messages/conversations)
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
            )

            chunk_results_g: List[Dict[str, Any]] = []
            num_chunks = min(ask_chunk + 5, rollout_max_chunks)  # run a few past ask
            for chunk_idx in range(num_chunks):
                q = user_question if chunk_idx == ask_chunk else None
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
                # Early stop if model responded
                if result.get("action") == "response" and chunk_idx >= ask_chunk:
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
    REWARD_DICT_KEYS,
    DEFAULT_REWARD_WEIGHTS,
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
    aggregate_v12_advantages as _aggregate_v12_advantages,
)


# ===========================================================================
# v12.0 reward components: see thinkstream/trainer/v12_rewards.py
# Re-imported above for in-grpo callers. Functions defined inline here are
# v11 components only.
# ===========================================================================



def _compute_silent_quality_reward(
    predicted_actions: List[str],
    gt_answer: str,
    model_answer: Optional[str],
    answer_form: str = "",
) -> float:
    """Explicit silent supervision (Dispider double-signal).

    Always counted (NOT gated on correctness) so that the model gets a
    direct signal preventing two failure modes:
      1. Silent collapse — model learns to always silent for safety.
      2. Over-response — model spams response on every chunk.

    Scoring (v9.1, scaled to match SFT silent loss weights):
      should-be-silent + silent      → +0.3 (correct silence — comparable to SFT 1.0×)
      should-be-silent + response    → -0.6 (premature/hallucinated response)
      should-respond + silent        → -0.6 (missed event)
      should-respond + response      →  0.0 (correctness reward handles quality)
      HLD-negative (canonical=No) + correct "No" response → +0.5 (refusal bonus)
      HLD-negative + silent          → -0.6 (missed refusal)

    Why the magnitudes: at weight=0.20 the swing is ±0.12 — same order as the
    response-correctness weight ×0.5 (0.25), so silent_quality can no longer be
    drowned out by other reward components. Below this scale RL drifts away
    from the SFT silent prior.
    """
    has_response = "response" in predicted_actions
    is_silent_only = not has_response and "compress" not in predicted_actions
    is_negative = (gt_answer or "").strip().lower() == "no" and answer_form == "binary"

    if is_negative:
        # HLD: model SHOULD respond "No" (not stay silent)
        if has_response and model_answer == "no":
            return 0.5
        if not has_response:
            return -0.6  # silent on HLD = missed refusal
        return 0.0

    if not gt_answer:
        # Silent-only sample: reward correct silence, penalize over-response
        return 0.3 if is_silent_only else -0.6

    # gt_answer present: model should respond
    if is_silent_only:
        return -0.6  # missed: should have responded
    return 0.0       # responded — correctness reward handles quality


def _compute_overflow_penalty(
    chunk_results: List[Dict[str, Any]], gen_idx: int
) -> float:
    """Sparse compress-timing reward (Memex(RL) 2603.04257 soft-trigger pattern).

    Returns -1.0 if any post-step memory_token_count exceeds the agent's
    own compress_budget; 0.0 otherwise. The agent learns "compress before
    overflow" purely from this penalty — no positive reward for "compressed
    at the right time" (SUMER 2511.21726 evidence: search > compress).

    Both fields are populated by ``StreamingAgentLoop.step()``; if missing
    (e.g., legacy chunk_results) the function returns 0.0 (no signal, not
    an error).
    """
    for cr in chunk_results:
        sizes = cr.get("memory_token_count", [])
        budgets = cr.get("compress_budget", [])
        if gen_idx >= len(sizes) or gen_idx >= len(budgets):
            continue
        budget = budgets[gen_idx]
        size = sizes[gen_idx]
        if budget > 0 and size > budget:
            return -1.0
    return 0.0


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

    v11 design (NVIDIA GDPO + Memex(RL) + ReMemR1 hybrid):
      - 6 reward components, each producing a scalar in [-1, 1] per
        (sample, generation) trajectory.
      - For sparse rewards (timing only fires if any rollout responded;
        recall_quality only if any rollout recalled), an applicability
        mask is emitted alongside. Downstream ``compute_gdpo_advantages``
        masks those rows to NaN before per-reward group-norm so masked
        rollouts contribute 0 to that reward's advantage.
      - No hard gates. Hard gates create bimodal advantage distributions
        that batch-whiten cannot recover from (see plan rationale).

    Outputs:
      rewards:       [B] = [N_samples * G] weighted sum of raw rewards
                          (logging-only; the GDPO node recomputes from
                           rewards_dict + rewards_masks).
      rewards_dict:  {key: [B]} per-component raw rewards.
      rewards_masks: [B, num_rewards] applicability mask (1=apply, 0=skip).
                     Column order matches REWARD_DICT_KEYS.
    """
    weights = DEFAULT_REWARD_WEIGHTS
    all_rewards = {k: [] for k in REWARD_DICT_KEYS}
    all_masks = {k: [] for k in REWARD_DICT_KEYS}

    for sample_data in rollout_data:
        raw_sample = sample_data["raw_sample"]
        chunk_results: List[Dict[str, Any]] = sample_data["chunk_results"]

        # Extract ground truth from raw sample (new format: metadata)
        metadata = raw_sample.get("metadata", {})
        gt_answer = metadata.get("gold_answer", "")
        answer_form = metadata.get("answer_form", "")
        support_chunks = list(metadata.get("support_chunks") or [])

        # Timing: use chunk_idx from sample
        gt_chunk_idx = raw_sample.get("chunk_idx")
        time_per_chunk = 2.0

        # Legacy format fallback
        if not gt_answer and "conversations" in raw_sample:
            conversations = raw_sample["conversations"]
            gt_msg = conversations[1] if len(conversations) > 1 else {}
            gt_answer = gt_msg.get("content", "")
            gt_timestamp = float(gt_msg.get("timestamp", 0.0))
            if chunk_results:
                time_per_chunk = chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
                video_start = chunk_results[0]["window_start"]
                gt_chunk_idx = int((gt_timestamp - video_start) / time_per_chunk) if time_per_chunk > 0 else None

        # Sanity: if the sample's gold_action expects a response but no
        # gt_answer is available, correctness/timing/recall_quality masks
        # will be 0 → the trajectory contributes nothing to those signals.
        # Warn loudly so half-supervised samples don't silently pass.
        gold_action = metadata.get("gold_action") or raw_sample.get("action") or ""
        if not gt_answer and gold_action in (
            "response", "recall_query", "recall_response"
        ):
            sid = (
                raw_sample.get("sample_id")
                or raw_sample.get("trajectory_id")
                or raw_sample.get("video_id")
                or "?"
            )
            logger.warning(
                "[grpo.calc_rewards] sample=%s gold_action=%s but gt_answer is "
                "empty (metadata.gold_answer + conversations[1] both missing). "
                "Correctness/timing/recall_quality will be masked-out for this "
                "sample. Re-run Pass4 with the metadata_complete check enabled.",
                sid, gold_action,
            )

        gt_present = bool(gt_answer)

        for g in range(group_size):
            chunk_texts: List[str] = []
            for cr in chunk_results:
                tokens = cr["generated_tokens"][g]
                chunk_texts.append(tokenizer.decode(tokens, skip_special_tokens=False))

            predicted_actions = [_extract_action(t) for t in chunk_texts]
            (model_answer, first_resp_chunk_idx, _resp_count,
             last_resp_chunk_idx) = _scan_responses_for_answer(
                chunk_results, g, tokenizer
            )
            did_respond = "response" in predicted_actions
            did_recall = "recall" in predicted_actions

            # ─── R_format (dense, always applies) ───
            fmt_r = _compute_format_reward(chunk_texts, agent_mode=True)

            # ─── R_correctness (dense, applies only if gold_answer present) ───
            corr_r = _compute_response_reward(
                model_answer, gt_answer, predicted_actions, answer_form
            )

            # ─── R_timing (sparse, applies only if rollout responded AND gt available) ───
            # v11.4 bug fix: use last_resp_chunk_idx (model's final commitment)
            # rather than first_resp_chunk_idx. Streaming agents that respond
            # then refine were being penalized for self-correction.
            if gt_chunk_idx is not None and gt_present:
                slack_window_chunks = (
                    int(time_reward_slack / time_per_chunk)
                    if time_per_chunk > 0
                    else 0
                )
                time_r = _compute_time_reward(
                    last_resp_chunk_idx, gt_chunk_idx,
                    time_reward_window, slack_window_chunks,
                )
            else:
                # Silent-only sample: no timing signal
                time_r = 0.0

            # ─── R_recall_quality / R_recall_hit_rate / R_range_tightness ───
            # Three separate sparse signals — all gated on did_recall.
            # Per-chunk retriever output for this gen — empty list when
            # the chunk's action wasn't recall.
            returned_per_chunk = [
                cr.get("recall_returned_chunks", [[]] * group_size)[g]
                if isinstance(cr.get("recall_returned_chunks"), list)
                else []
                for cr in chunk_results
            ]
            recall_r = _compute_recall_quality_reward(
                predicted_actions, chunk_texts, gt_answer, model_answer
            )
            # v11.4: pass recall_fired so a fired-but-empty recall returns
            # an explicit -0.2 instead of None (which mask=0 hid before).
            hit_rate_r = _compute_recall_hit_rate(
                returned_per_chunk, support_chunks, recall_fired=did_recall,
            )
            # Estimate video_duration from the last chunk window we saw.
            video_duration = (chunk_results[-1].get("window_end", 0.0)
                              if chunk_results else 0.0)
            range_tight_r = _compute_range_tightness_reward(
                chunk_texts, support_chunks, video_duration,
            )

            # ─── R_silent_quality (dense, always applies — streaming-specific) ───
            silent_r = _compute_silent_quality_reward(
                predicted_actions, gt_answer, model_answer, answer_form
            )

            # ─── R_overflow_pen (dense, always applies — Memex soft-trigger pattern) ───
            overflow_r = _compute_overflow_penalty(chunk_results, g)

            all_rewards["format"].append(fmt_r)
            all_rewards["correctness"].append(corr_r)
            all_rewards["timing"].append(time_r)
            all_rewards["silent_quality"].append(silent_r)
            all_rewards["recall_quality"].append(recall_r)
            all_rewards["recall_hit_rate"].append(
                hit_rate_r if hit_rate_r is not None else 0.0
            )
            all_rewards["range_tightness"].append(
                range_tight_r if range_tight_r is not None else 0.0
            )
            all_rewards["overflow_pen"].append(overflow_r)

            # Per-reward applicability masks (downstream GDPO node masks NaN)
            all_masks["format"].append(1.0)
            all_masks["correctness"].append(1.0 if gt_present else 0.0)
            all_masks["timing"].append(
                1.0 if (did_respond and gt_present and gt_chunk_idx is not None) else 0.0
            )
            all_masks["silent_quality"].append(1.0)
            all_masks["recall_quality"].append(1.0 if did_recall else 0.0)
            all_masks["recall_hit_rate"].append(
                1.0 if (did_recall and hit_rate_r is not None) else 0.0
            )
            all_masks["range_tightness"].append(
                1.0 if (did_recall and range_tight_r is not None) else 0.0
            )
            all_masks["overflow_pen"].append(1.0)

    # Logging-only weighted sum (real advantage is computed in
    # compute_gdpo_advantages from rewards_dict + rewards_masks).
    n = len(all_rewards["format"])
    totals: List[float] = []
    for i in range(n):
        s = 0.0
        for k in REWARD_DICT_KEYS:
            if all_masks[k][i] > 0:
                s += weights[k] * all_rewards[k][i]
        totals.append(s)

    total_tensor = torch.tensor(totals, dtype=torch.float32)
    rewards_dict_val = {
        k: torch.tensor(all_rewards[k], dtype=torch.float32) for k in REWARD_DICT_KEYS
    }
    # rewards_masks columns ordered to match REWARD_DICT_KEYS
    masks_tensor = torch.tensor(
        [[all_masks[k][i] for k in REWARD_DICT_KEYS] for i in range(n)],
        dtype=torch.float32,
    )

    # ── Audit log: per-step aggregate + per-(sample, generation) breakdown ──
    try:
        _emit_grpo_audit(
            rollout_data, all_rewards, all_masks, totals, group_size, tokenizer,
        )
    except Exception as e:
        logger.debug("grpo audit log skipped: %s", e)

    return ctx.update({
        rewards: total_tensor,
        rewards_dict: rewards_dict_val,
        rewards_masks: masks_tensor,
    })


def _emit_grpo_audit(
    rollout_data, all_rewards, all_masks, totals, group_size, tokenizer,
) -> None:
    """Write one grpo_step record + one grpo_sample record per (sample, gen)."""
    global _GRPO_STEP_COUNTER
    step_w, sample_w = _grpo_audit_writers()
    if step_w is None and sample_w is None:
        return

    _GRPO_STEP_COUNTER += 1
    n = len(totals)

    if step_w is not None and n > 0:
        def _stats(key):
            xs = all_rewards[key]
            return {
                "mean": float(sum(xs) / len(xs)),
                "min": float(min(xs)),
                "max": float(max(xs)),
            }
        record = {
            "step": _GRPO_STEP_COUNTER,
            "n": n,
            "total": {
                "mean": float(sum(totals) / n),
                "min": float(min(totals)),
                "max": float(max(totals)),
            },
        }
        for k in REWARD_DICT_KEYS:
            record[k] = _stats(k)
            record[f"mask_{k}_rate"] = float(sum(all_masks[k]) / n)
        step_w.write(record)

    if sample_w is not None:
        # Reconstruct (sample, generation) pairing — rewards are flattened in the
        # same order as the calc_rewards loop, group_size per sample.
        i = 0
        for sample_data in rollout_data:
            raw = sample_data["raw_sample"]
            chunk_results = sample_data.get("chunk_results", [])
            metadata = raw.get("metadata", {})
            sid = (
                raw.get("sample_id")
                or raw.get("trajectory_id")
                or raw.get("video_id")
                or "?"
            )
            for g in range(group_size):
                if i >= n:
                    break
                # Re-decode model output for this generation (truncate to keep log small)
                try:
                    chunks = []
                    for cr in chunk_results:
                        toks = cr["generated_tokens"][g]
                        chunks.append(tokenizer.decode(toks, skip_special_tokens=False))
                    actions = [_extract_action(t) for t in chunks]
                    full_out = "".join(chunks)
                    if len(full_out) > 600:
                        full_out = full_out[:600] + "...<truncated>"
                except Exception:
                    actions, full_out = [], ""

                sample_w.write({
                    "step": _GRPO_STEP_COUNTER,
                    "sample_id": sid,
                    "video_id": raw.get("video_id"),
                    "chunk_idx": raw.get("chunk_idx"),
                    "generation": g,
                    "gold_action": metadata.get("gold_action"),
                    "gold_answer": (metadata.get("gold_answer") or "")[:200],
                    "answer_form": metadata.get("answer_form"),
                    "predicted_actions": actions,
                    "model_output": full_out,
                    "rewards": {k: all_rewards[k][i] for k in REWARD_DICT_KEYS},
                    "masks": {k: all_masks[k][i] for k in REWARD_DICT_KEYS},
                    "total": totals[i],
                })
                i += 1


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


def _build_rollout_messages(
    raw_sample, chunk_results, gen_idx, tokenizer, frames_per_chunk
):
    # (Kept identical to original implementation...)
    data_path = raw_sample.get("data_path", "")
    video_path = raw_sample.get("video_path", "")
    abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

    pending_queries = sorted(
        [
            (float(c.get("timestamp", 0.0)), c.get("content", ""))
            for c in raw_sample.get("conversations", [])
            if c.get("role") == "user"
        ],
        key=lambda x: x[0],
    )
    num_chunks = len(chunk_results)
    if num_chunks == 0:
        raise ValueError("No chunk results – cannot build messages.")
    video_chunk_size = chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
    total_start = chunk_results[0]["window_start"]
    total_end = chunk_results[-1]["window_end"]

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        chunk_queries: List[str] = []
        while pending_queries:
            ts = pending_queries[0][0]
            if ts < w_start:
                pending_queries.pop(0)
                continue
            if is_last or ts < w_end:
                chunk_queries.append(pending_queries.pop(0)[1])
            else:
                break
        if chunk_queries:
            user_content.append(
                {"type": "text", "text": "\n" + "\n".join(chunk_queries)}
            )
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
