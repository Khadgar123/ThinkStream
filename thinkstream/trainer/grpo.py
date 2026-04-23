import json
import os
import re
import types
import math
import logging
import deepspeed
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel
from slyme.context import Context, Ref
from slyme.node import Node, node, wrapper, Auto, expression
from deepslyme.utils.accelerator import empty_cache

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


def _compute_action_reward(
    predicted_actions: List[str],
    need_recall: bool,
    wrong_action_penalty: float = 1.0,
    over_recall_penalty: float = 0.3,
) -> float:
    """Evaluate whether the model chose the correct action sequence.

    For need_recall=True samples:
      recall → response = 1.0 (full credit)
      direct response (if correct) = 0.3 (partial)
      all silent = 0.0

    For need_recall=False samples:
      response (no recall) = 1.0
      recall triggered = -over_recall_penalty
      all silent = 0.0
    """
    if need_recall:
        if "recall" in predicted_actions:
            recall_idx = predicted_actions.index("recall")
            if "response" in predicted_actions[recall_idx + 1:]:
                return 1.0  # recall then response
            return 0.5  # recall but no response after
        if "response" in predicted_actions:
            return 0.3  # direct response without recall
        return 0.0  # all silent
    else:
        if "recall" in predicted_actions:
            return max(0.0, 1.0 - over_recall_penalty)
        if "response" in predicted_actions:
            return 1.0
        return 0.0


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


def _avg_think_len_for_generation(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> float:
    """Average number of think tokens (inside <think>...</think>) for one (sample, gen_idx) across chunks."""
    lengths = _collect_think_lengths(chunk_results, gen_idx, tokenizer)
    return sum(lengths) / len(lengths) if lengths else 0.0


def _compute_think_length_factor(
    avg_think_len: float, target_tokens: int, step_window: int = 5
) -> float:
    """
    Compute a discrete, step-wise reward for the thinking token length.
    The reward increases in discrete steps of `step_window`.
    Any length >= (target_tokens - step_window) receives the maximum reward of 1.0.
    """
    if target_tokens <= 0:
        return 1.0
    step_window = max(1, step_window)
    threshold = max(0, target_tokens - step_window)
    if avg_think_len >= threshold:
        return 1.0
    if threshold == 0:
        return 1.0
    step_idx = int(avg_think_len // step_window)
    total_steps = int(threshold // step_window) + 1
    return float(step_idx) / float(total_steps)


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
) -> Tuple[Optional[str], Optional[int], int]:
    first_response_chunk_idx: Optional[int] = None
    first_answer: Optional[str] = None
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
            if first_answer is None:
                answer = _extract_literal_answer(m.group(1))
                if answer is not None:
                    first_answer = answer
                    if first_response_chunk_idx is None:
                        first_response_chunk_idx = cr["chunk_idx"]
    return first_answer, first_response_chunk_idx, response_count


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


def _compute_correctness_reward(model_answer: Optional[str], gt_content: str) -> float:
    if not model_answer:
        return 0.0
    gt_answer = _extract_literal_answer(gt_content)
    if gt_answer is None:
        return 0.0
    return 1.0 if model_answer == gt_answer else 0.0


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
) -> Context:
    """
    GRPO rollout using streaming video inference.

    For each raw sample in the batch, calls ``streaming_video_chat`` to
    generate completions chunk-by-chunk.  Stores per-sample results
    (generated tokens, chunk metadata, raw sample) in ``rollout_data`` for
    downstream reward computation and loss calculation.

    Uses ``StreamingAgentLoop.step()`` for per-timestep re-render rollout,
    matching the SFT training format exactly (explicit memory management,
    <memory>/<visual_window> tags, 4-action protocol).

    For each raw sample, runs the agent loop from chunk 0 to max_chunks
    with G=group_size independent rollouts. Each rollout maintains its own
    memory state.

    NOTE: This node should be wrapped with ``unwrap_model_for_generation``
    which handles ZeRO-3 parameter gathering and inference engine cleanup.
    """
    from thinkstream.model.agent_loop import StreamingAgentLoop

    all_rollout_results: List[Dict[str, Any]] = []
    model_for_generation.eval()

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

        # Extract user question (from messages or conversations)
        user_question = None
        if "messages" in raw_sample:
            for msg in raw_sample["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if "<user_input>" in text:
                                    import re as _re
                                    m = _re.search(r'<user_input>(.*?)</user_input>', text)
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
                    "window_start": chunk_idx * 2,
                    "window_end": (chunk_idx + 1) * 2,
                })
                # Early stop if model responded
                if result.get("action") == "response" and chunk_idx >= ask_chunk:
                    break
            per_gen_results.append(chunk_results_g)

        # Merge into the expected format: chunk_results with generated_tokens[G]
        max_chunks_seen = max(len(g) for g in per_gen_results)
        merged_chunk_results = []
        for ci in range(max_chunks_seen):
            merged = {
                "chunk_idx": ci,
                "window_start": ci * 2,
                "window_end": (ci + 1) * 2,
                "generated_tokens": [],
            }
            for g in range(group_size):
                if ci < len(per_gen_results[g]):
                    merged["generated_tokens"].append(
                        torch.tensor(per_gen_results[g][ci]["generated_tokens"])
                    )
                else:
                    # Pad with empty if this gen finished early
                    merged["generated_tokens"].append(torch.tensor([]))
            merged_chunk_results.append(merged)

        all_rollout_results.append({
            "raw_sample": raw_sample,
            "chunk_results": merged_chunk_results,
        })

    model_for_generation.train()
    return ctx.set(rollout_data, all_rollout_results)


REWARD_DICT_KEYS = ("format", "action", "correctness", "timing", "think_len", "compress")
DEFAULT_REWARD_WEIGHTS = {
    "format": 0.15,
    "action": 0.20,
    "correctness": 0.30,
    "timing": 0.15,
    "think_len": 0.10,
    "compress": 0.10,
}


def _compute_compress_reward(
    chunk_texts: List[str],
    compressed_segments: List[Dict],
) -> float:
    """Evaluate compression quality: entity retention in summary.

    Checks how many entity names from the source compressed segments
    appear in the model's generated summary.
    """
    # Find compress actions in generated text
    summaries = []
    for text in chunk_texts:
        m = _SUMMARY_RE.search(text)
        if m:
            try:
                s = json.loads(m.group(1))
                summaries.append(s.get("text", ""))
            except (json.JSONDecodeError, ValueError):
                pass
    if not summaries:
        return 0.0  # no compress action taken

    # Entity names from compressed segments (ground truth)
    entity_words = set()
    for seg in compressed_segments:
        words = re.findall(r'\b[a-zA-Z_]+\d*\b', seg.get("text", ""))
        entity_words.update(w.lower() for w in words if "_" in w or w[0].isupper())

    if not entity_words:
        return 1.0  # no entities to check, pass

    # Check retention across all summaries
    retained = 0
    for summary_text in summaries:
        summary_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z_]+\d*\b', summary_text))
        retained += len(entity_words & summary_words)
    return min(1.0, retained / len(entity_words))


@node
def calc_rewards(
    ctx: Context,
    /,
    *,
    rollout_data: Auto[Dict[str, Any]],
    rewards: Ref[torch.Tensor],
    rewards_dict: Ref[Dict[str, torch.Tensor]],
    group_size: Auto[int],
    tokenizer: Auto[Any],
    time_reward_window: Auto[int],
    time_reward_slack: Auto[float],
    rollout_max_think_tokens: Auto[int],
) -> Context:
    """Compute per-generation rewards for 4-action per-timestep agent.

    Six reward components (see data_batch1_plan.md §5.3):
    - format: think/action tag structure correct
    - action: chose correct action type vs gold_action
    - correctness: answer matches gold_answer
    - timing: responded at the right chunk
    - think_len: think length near target (40-60 tok)
    - compress: entity retention in compression summaries

    ``rollout_data`` is ``List[Dict]`` of length B (one per sample).
    Sets ``rewards`` (shape [B*G]) and ``rewards_dict``.
    """
    weights = DEFAULT_REWARD_WEIGHTS
    all_rewards = {k: [] for k in REWARD_DICT_KEYS}

    for sample_data in rollout_data:
        raw_sample = sample_data["raw_sample"]
        chunk_results: List[Dict[str, Any]] = sample_data["chunk_results"]

        # Extract ground truth from raw sample
        # Per-timestep format: gold info in metadata
        metadata = raw_sample.get("metadata", {})
        gold_action = metadata.get("gold_action", "")
        gt_content = metadata.get("gold_answer", "")
        need_recall = gold_action == "recall"
        compressed_segments = raw_sample.get("compressed_segments",
                              raw_sample.get("input", {}).get("compressed_segments", []))

        # Timing: use ask_chunk from metadata
        gt_chunk_idx = raw_sample.get("chunk_idx")
        time_per_chunk = 2.0  # AGENT_CHUNK_SEC

        # Legacy format fallback
        if not gold_action and "conversations" in raw_sample:
            conversations = raw_sample["conversations"]
            gt_msg = conversations[1] if len(conversations) > 1 else {}
            gt_content = gt_msg.get("content", "")
            gt_timestamp = float(gt_msg.get("timestamp", 0.0))
            if chunk_results:
                time_per_chunk = chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
                video_start = chunk_results[0]["window_start"]
                gt_chunk_idx = int((gt_timestamp - video_start) / time_per_chunk) if time_per_chunk > 0 else None

        for g in range(group_size):
            chunk_texts: List[str] = []
            for cr in chunk_results:
                tokens = cr["generated_tokens"][g]
                chunk_texts.append(tokenizer.decode(tokens, skip_special_tokens=False))

            predicted_actions = [_extract_action(t) for t in chunk_texts]
            model_answer, response_chunk_idx, num_responses = (
                _scan_responses_for_answer(chunk_results, g, tokenizer)
            )

            # R_format
            fmt_r = _compute_format_reward(chunk_texts, agent_mode=True)

            # R_action
            action_r = _compute_action_reward(predicted_actions, need_recall)

            # R_correctness
            corr_r = _compute_correctness_reward(model_answer, gt_content)

            # R_timing
            if gt_chunk_idx is not None:
                slack_window_chunks = (
                    int(time_reward_slack / time_per_chunk) if time_per_chunk > 0 else 0
                )
                time_r = _compute_time_reward(
                    response_chunk_idx, gt_chunk_idx,
                    time_reward_window, slack_window_chunks,
                )
            else:
                time_r = 0.0

            # R_think_len
            avg_think_len = _avg_think_len_for_generation(chunk_results, g, tokenizer)
            think_r = _compute_think_length_factor(avg_think_len, rollout_max_think_tokens)

            # R_compress
            compress_r = _compute_compress_reward(chunk_texts, compressed_segments)

            total_r = sum(
                weights[k] * v for k, v in zip(
                    REWARD_DICT_KEYS,
                    [fmt_r, action_r, corr_r, time_r, think_r, compress_r],
                )
            )

            all_rewards["format"].append(fmt_r)
            all_rewards["action"].append(action_r)
            all_rewards["correctness"].append(corr_r)
            all_rewards["timing"].append(time_r)
            all_rewards["think_len"].append(think_r)
            all_rewards["compress"].append(compress_r)

    total_tensor = torch.tensor(
        [sum(all_rewards[k][i] * weights[k] for k in REWARD_DICT_KEYS)
         for i in range(len(all_rewards["format"]))],
        dtype=torch.float32,
    )
    rewards_dict_val = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in all_rewards.items()
    }
    return ctx.update({rewards: total_tensor, rewards_dict: rewards_dict_val})


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

    # Global reward mean
    reward_mean = rewards.float().mean().item()

    # Calculate intra-group variance, then inter-group mean
    # rewards shape is [B * G], reshape to [B, G]
    if rewards.numel() > 1 and group_size > 1:
        grouped_rewards = rewards.float().view(-1, group_size)
        # var(dim=1) computes variance within each prompt group
        reward_var = grouped_rewards.var(dim=1).mean().item()
    else:
        reward_var = 0.0

    # Component-wise average rewards
    component_means = {
        f"reward_{k}_mean": v.float().mean().item() for k, v in rewards_dict.items()
    }

    return {
        "grad_norm": grad_norm,
        "learning_rate": lr,
        "reward_mean": reward_mean,
        "reward_var": reward_var,
        **component_means,
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
