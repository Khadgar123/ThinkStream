"""
Render Pass: Convert 3-C samples into SFT-ready training samples.

Takes: 3-C fork+base samples + Pass 2 rollout snapshots
Produces: Complete SFT samples with input + output fields

This is the bridge between data construction (Pass 3) and SFT training.
Each sample gets a full `input` structure that pass5_messages.py renders
into the canonical timestamped-image ShareGPT messages.

Called after Pass 3-C raw sample generation and before Pass 3-E verification.
"""

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

# v12 agentic protocol uses <answer> as the terminal tag.
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

from .config import (
    AGENT_CHUNK_SEC,
    VISUAL_WINDOW_CHUNKS,
    FRAMES_PER_CHUNK,
    compute_visual_window_start,
)
from thinkstream.data.agent_protocol import (
    SYSTEM_PROMPT_V12,
    build_recalled_frames_metadata,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
)

logger = logging.getLogger(__name__)

_OPTION_LABEL_RE = re.compile(r"^\s*([A-D])[\).]\s*(.*)\s*$", re.DOTALL)


def _strip_option_label(text: str) -> str:
    m = _OPTION_LABEL_RE.match(str(text or ""))
    return (m.group(2) if m else str(text or "")).strip()


def _mc_correct_letter_text(card: Dict) -> tuple[str, str]:
    options = list(card.get("options") or [])
    correct = str(card.get("correct_option") or "").strip().upper()
    if correct in {"A", "B", "C", "D"} and len(options) == 4:
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(options):
            return correct, _strip_option_label(options[idx])
    return correct, _strip_option_label(card.get("canonical_answer", ""))


def _semantic_gold_answer(card: Dict, sft_answer: str) -> str:
    """Return the answer used by RL/eval scoring, not necessarily SFT target."""
    answer_form = card.get("answer_form", "")
    canonical = str(card.get("canonical_answer") or "").strip()
    if answer_form == "multiple_choice":
        _letter, text = _mc_correct_letter_text(card)
        return text or canonical
    if card.get("question_type") == "multi_emit" and sft_answer:
        return sft_answer
    return canonical or sft_answer


def _accepted_answers(card: Dict, gold_answer: str) -> List[str]:
    if card.get("answer_form") != "multiple_choice":
        return [gold_answer] if gold_answer else []
    letter, text = _mc_correct_letter_text(card)
    out = []
    if letter:
        out.append(letter)
    if letter and text:
        out.append(f"{letter}) {text}")
    if text:
        out.append(text)
    # Preserve order while de-duplicating.
    seen = set()
    return [x for x in out if x and not (x.lower() in seen or seen.add(x.lower()))]


def _get_system_prompt(prompt_type: str, *, inter_chunk: bool = False) -> str:
    """Return the protocol system prompt for this sample type.

    Ordinary streaming samples use the recall/answer/silent prompt. Compress
    samples use the compression-only prompt; the user input carries only the
    bare legacy trigger marker.
    """
    if inter_chunk or str(prompt_type or "").lower() in {"compress", "system_prompt_compress"}:
        return system_prompt_for_frame_protocol(prompt_kind="compress")
    if str(prompt_type or "").lower() in {"recall_response", "recall_answer", "post_recall"}:
        return system_prompt_for_frame_protocol(prompt_kind="post_recall")
    return SYSTEM_PROMPT_V12


def _build_visual_window(
    chunk_idx: int, num_chunks: int, video_path: str,
    frame_dir: str = None,
) -> Dict:
    """Build visual_window structure from chunk index.

    If frame_dir is provided (pre-extracted frames), includes frame_paths
    for fast training I/O. Otherwise SFT data_processor falls back to
    video_path online decoding.
    """
    window_start = compute_visual_window_start(chunk_idx)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    vw = {
        "video_start": video_start,
        "video_end": video_end,
        "frames": n_frames,
    }

    # If pre-extracted frames exist, add frame_paths for fast I/O
    if frame_dir:
        paths = []
        for ci in range(window_start, chunk_idx + 1):
            for fi in range(FRAMES_PER_CHUNK):
                p = Path(frame_dir) / f"chunk_{ci:04d}_f{fi}.jpg"
                if p.exists():
                    paths.append(str(p))
        if paths:
            vw["frame_paths"] = paths

    return vw


def _build_memory_from_snapshot(snapshot: Dict) -> Dict:
    """Convert rollout snapshot to memory structure for SFT input."""
    memory = {
        "compressed_segments": [],
        "recent_thinks": [],
    }

    # Compressed segments from snapshot
    for seg in snapshot.get("compressed_segments", []):
        out_seg = {
            "time_range": seg["time_range"],
            "text": seg["text"],
        }
        if seg.get("source_chunks"):
            out_seg["source_chunks"] = sorted(int(c) for c in seg.get("source_chunks", []))
        if "merge_level" in seg:
            out_seg["merge_level"] = int(seg.get("merge_level", 0) or 0)
        memory["compressed_segments"].append(out_seg)

    # Recent thinks from snapshot
    for item in snapshot.get("recent_thinks", []):
        if isinstance(item, dict):
            time_str = item.get("time", "")
            text = item.get("text", item.get("obs", ""))
            memory["recent_thinks"].append(f"[{time_str}] {text}")
        elif isinstance(item, str):
            memory["recent_thinks"].append(item)

    return memory


def _build_queries_input(queries_state: List[Dict]) -> List[Dict]:
    """Convert queries_state to the format expected by SFT input.

    v12.12 fix (P0-4): preserve `ask_time` so format_queries_block
    (agent_protocol.py:148) can render Q events at the correct chunk time.
    Without ask_time, the active query collapses to t=0 in the rendered query
    state block — train/infer divergence (runtime has real timestamps).

    v12.13 fix (P0-3): also preserve `options` + `answer_form` so MC pending
    queries can render their choices in the active-query block. forward
    response chunks fire AFTER ask (no fresh user_input), so the model
    only sees the active Q in query state — without options it cannot
    choose A-D meaningfully.
    """
    result = []
    for q in queries_state:
        result.append({
            "card_id": q.get("card_id", ""),
            "question": q.get("question", ""),
            "options": list(q.get("options") or []),
            "correct_option": q.get("correct_option", ""),
            "gold_answer": q.get("gold_answer", ""),
            "correct_answer_text": q.get("correct_answer_text", ""),
            "accepted_answers": list(q.get("accepted_answers") or []),
            "answer_form": q.get("answer_form", ""),
            "answer_style": q.get("answer_style", ""),
            "answer_instruction": q.get("answer_instruction", ""),
            "ask_time": q.get("ask_time", 0),
            "open_until": q.get("open_until", q.get("ask_time", 0)),
            "status": q.get("status", ""),
            "answers": q.get("answers", []),
        })
    return result


def _build_recalled_frames(
    recall_result: Optional[Dict],
    all_frame_paths: Optional[List[str]],
) -> Optional[Dict]:
    """Build the `recalled_frames` input zone for recall_response samples.

    Mirrors the inference-time logic in agent_loop.step (recall branch):
    given returned_chunks from a successful retrieval, derive the
    contiguous time_range, frame count, and per-chunk frame_paths so the
    SFT sample renders <recalled_frames> + actual video frames — not
    text-only. Without this, SFT trains on the recall_result text alone
    and inference's frame injection becomes OOD.

    Returns None only for invalid/empty results. Both recall_response and
    recall_silent should carry frames when retrieval returns historical chunks.
    """
    if not recall_result or recall_result.get("source") != "historical_frames":
        return None
    chunks = select_recall_chunks(recall_result.get("returned_chunks") or [])
    if not chunks:
        return None
    paths = []
    if all_frame_paths:
        from .pass1a_evidence import get_chunk_frame_paths
        for c in chunks:
            paths.extend(get_chunk_frame_paths(all_frame_paths, c))
    return build_recalled_frames_metadata(
        chunks,
        paths,
        chunk_sec=AGENT_CHUNK_SEC,
        frames_per_chunk=FRAMES_PER_CHUNK,
    )


def render_sample(
    sample: Dict,
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Render a 3-C sample into an SFT-ready training sample.

    Combines:
    - sample's output, queries, user_input, recall_result (from 3-C)
    - rollout's snapshot at chunk_idx (from Pass 2)
    - system prompt (from agent_protocol)
    - visual_window structure (computed from chunk_idx)
    - metadata with gold_action/gold_answer (from cards_map)

    `all_frame_paths` is the per-video flat frame list (2fps extracted
    earlier in the pipeline). When provided, recall_response samples get
    `recalled_frames.frame_paths` populated so the SFT loader feeds actual
    historical frames into the model — matching what inference does. If
    omitted, recall_response samples render text-only recall (legacy
    behaviour, retained for back-compat with callers that don't have the
    frame list).

    Returns a complete sample with `input` + `output` + `metadata` fields.
    """
    chunk_idx = sample["chunk_idx"]
    prompt_type = sample.get("prompt_type", "SYSTEM_PROMPT")
    num_chunks = rollout["num_chunks"]

    # Get snapshot for this chunk
    snapshots = rollout["snapshots"]
    snapshot = snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx)) or {}

    # Build input structure
    recall_result = sample.get("recall_result")
    inp = {
        "system": _get_system_prompt(
            prompt_type,
            inter_chunk=sample.get("action") == "compress",
        ),
        "visual_window": _build_visual_window(chunk_idx, num_chunks, video_path),
        "memory": _build_memory_from_snapshot(snapshot),
        "queries": _build_queries_input(sample.get("queries", [])),
        "user_input": sample.get("user_input", ""),
        "recall_result": recall_result,
    }
    rf = _build_recalled_frames(recall_result, all_frame_paths)
    if rf is not None:
        inp["recalled_frames"] = rf

    # For compress samples, remember the gold compressed-chunks set so
    # RL/eval can score the model's <summary> time_range against the
    # teacher's choice. pass3c injects only the boolean
    # ``<compress_trigger/>`` signal into sample.user_input; the gold range
    # lives in the assistant tool_call output and must be derived from memory.
    gold_compress_chunks: List[int] = []
    if sample.get("action") == "compress":
        for event in rollout.get("compression_events", []):
            if event.get("trigger_chunk") == chunk_idx:
                summary = event.get("summary") or {}
                cr = (
                    summary.get("source_chunks")
                    or event.get("compressed_source_chunks")
                    or event.get("compressed_thinks_chunks")
                    or []
                )
                if cr:
                    gold_compress_chunks = sorted(int(c) for c in cr)
                break

    # Build metadata for RL reward computation.
    #
    # Contract (silent failures here ⇒ Pass4 + GRPO get neutered):
    #   - if sample has a card_id, the card MUST be in cards_map.
    #     A missing lookup means cards_map was passed empty/wrong upstream;
    #     fail loudly rather than emit an empty-metadata sample.
    #   - silent-only samples (no card_id) legitimately have empty metadata.
    card_id = sample.get("card_id", "")
    if card_id:
        hardened_card = sample.get("hardened_card")
        if isinstance(hardened_card, dict) and hardened_card.get("card_id") == card_id:
            card = hardened_card
        elif not cards_map or card_id not in cards_map:
            raise KeyError(
                f"[{video_id}] sample chunk={chunk_idx} references card_id="
                f"{card_id!r} but it is missing from cards_map "
                f"(cards_map size={len(cards_map or {})}). Render before Pass4 "
                f"requires fully-populated cards_map."
            )
        else:
            card = cards_map[card_id]
    else:
        card = {}
    # Separate the two answer layers:
    #   - sft_answer: exact target string inside <answer> for this row.
    #   - gold_answer: semantic answer used by RL/eval scoring.
    # For MC, sft_answer may be "A", "A) red apron", or "red apron";
    # gold_answer remains the correct option text so downstream scoring
    # can accept all equivalent formats.
    canonical = card.get("canonical_answer", "")
    # v12: output may live in v12_assistant_turn_2 (multi-turn recall);
    # otherwise it's in sample.output. Both use <answer>...</answer>.
    v12_text = (sample.get("v12_assistant_turn_2")
                or sample.get("output", "") or "")
    m = _ANSWER_RE.search(v12_text)
    sft_answer = m.group(1).strip() if m else ""
    gold_answer = _semantic_gold_answer(card, sft_answer)
    correct_letter, correct_answer_text = _mc_correct_letter_text(card)
    if card.get("answer_form") == "multiple_choice" and correct_answer_text:
        canonical = correct_answer_text

    metadata = {
        "gold_action": sample.get("action", "silent"),
        "gold_answer": gold_answer,
        "sft_answer": sft_answer,
        "correct_answer_text": correct_answer_text,
        "accepted_answers": _accepted_answers(card, gold_answer),
        # Required for post-pass MC option rebalance and audits. Several videos
        # intentionally contain distinct cards with identical question text.
        "card_id": card_id,
        # Keep the card-level canonical separately so GRPO / eval can
        # tell when a sample's gold_answer diverges from the card final
        # answer (signal that it's a multi-probe pre-event chunk).
        "canonical_answer": canonical,
        "answer_form": card.get("answer_form", ""),
        "answer_style": sample.get("answer_style", card.get("answer_style", "")),
        "answer_instruction": sample.get(
            "answer_instruction", card.get("answer_instruction", "")
        ),
        "question_type": card.get("question_type", ""),
        "family": card.get("family", ""),
        "family_name": card.get("family_name", ""),
        "category": card.get("category", ""),
        "skill": card.get("skill", ""),
        "ours_unique": bool(card.get("ours_unique", False)),
        "availability": sample.get("sequence_type", ""),
        "support_chunks": list(card.get("support_chunks")
                               or card.get("grounding_frames") or []),
        # Gold compressed-chunks (for compress samples) — empty list for
        # non-compress samples. Used by streaming-eval / RL to score
        # the model's <summary> time_range vs teacher's policy choice.
        "gold_compress_chunks": gold_compress_chunks,
        # v12.12 fix (P0-1, P0-2): preserve the real placement.ask_chunk
        # and the question schema (text + MC options + correct letter).
        # pass4 was previously inferring ask_chunk from the response
        # sample's chunk_idx, breaking forward/silent_then_response
        # timing. It also dropped question/options/correct_option,
        # forcing eval and RL to fall back to gold_answer (= treating
        # the answer text as the question).
        "ask_chunk": int(sample["ask_chunk"]) if "ask_chunk" in sample else -1,
        "question": card.get("question", ""),
        "options": list(card.get("options") or []),
        "correct_option": card.get("correct_option", correct_letter),
        # v12.13 fix (P0-2): per_emit_answers carries [{chunk, value}, ...]
        # for multi_emit cards. pass4 builds questions[*].per_emit_answers
        # so reward can score multi-emit at each expected answer chunk
        # with its OWN gold (F5 counting: "1","2","3"; PN1 narration: each
        # event's description). Single-emit cards get list of length 1.
        "per_emit_answers": list(sample.get("per_emit_answers") or []),
    }

    # v12.11 audit-4 P0 #1 fix (2026-05-01): merged shape-B recall samples
    # had `output` popped by _merge_recall_pairs_v12 (the canonical text
    # now lives in v12_assistant_turn_2). The previous unconditional read
    # of sample["output"] threw KeyError → render_sample's enclosing
    # try/except dropped the entire merged sample → recall multi-turn
    # vanished from final SFT/RL data.
    # Use the same fallback chain as gold_answer extraction above so
    # both shape-A and shape-B samples render correctly.
    output_text = (
        sample.get("output")
        if sample.get("output") is not None
        else (sample.get("v12_assistant_turn_2") or "")
    )

    # Build complete SFT sample
    rendered = {
        # Core fields for SFT data_processor
        "input": inp,
        "output": output_text,
        "video_path": video_path,
        "video_id": video_id,
        "chunk_idx": chunk_idx,

        # Phase assignment
        "sample_type": sample.get("sample_type", "silent"),
        "action": sample.get("action", "silent"),
        "prompt_type": prompt_type,
        "sequence_type": sample.get("sequence_type", ""),
        "trajectory_id": sample.get("trajectory_id", ""),
        "card_id": card_id,

        # RL reward fields
        "metadata": metadata,
    }

    # Propagate base_role for trajectory-aware loss weighting
    if "base_role" in sample:
        rendered["base_role"] = sample["base_role"]

    # Keep recall tool results at the row top-level as well as inside
    # input. RL/eval utilities may consume rendered flat/trajectory rows
    # directly, while pass5 consumes input.* to inject the actual media.
    if recall_result is not None:
        rendered["recall_result"] = recall_result
    if rf is not None:
        rendered["recalled_frames"] = rf

    # Propagate v12-specific multi-turn fields so pass4 can render them.
    for v12_key in ("v12_assistant_turn_1", "v12_assistant_turn_2", "v12_inter_chunk"):
        if v12_key in sample:
            rendered[v12_key] = sample[v12_key]

    return rendered


def render_trajectory(
    trajectory_samples: List[Dict],
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Render all samples in a trajectory into SFT-ready format."""
    rendered = []
    for sample in trajectory_samples:
        try:
            r = render_sample(sample, rollout, video_path, video_id,
                              cards_map, all_frame_paths=all_frame_paths)
            rendered.append(r)
        except Exception as e:
            logger.warning(
                f"[{video_id}] render failed for chunk {sample.get('chunk_idx')}: {e}"
            )
    return rendered


def render_video_samples(
    all_samples: List[Dict],
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Render all samples for a video, grouped by trajectory."""
    by_traj = {}
    for s in all_samples:
        tid = s.get("trajectory_id", "no_traj")
        by_traj.setdefault(tid, []).append(s)

    rendered = []
    for tid, traj_samples in by_traj.items():
        traj_rendered = render_trajectory(
            traj_samples, rollout, video_path, video_id, cards_map,
            all_frame_paths=all_frame_paths)
        rendered.extend(traj_rendered)

    return rendered
