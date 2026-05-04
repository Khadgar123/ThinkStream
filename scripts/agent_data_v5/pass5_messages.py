"""PASS 5 — Convert single-step samples to LLaMA-Factory ShareGPT messages format.

Reads pass4 outputs and emits one row per sample in the multi-turn messages
format used by LLaMA-Factory / DeepEyesV2 / VST. Each row is a stand-alone
training sample matching fresh-KV-per-chunk inference: every sample's
user.content carries the full state (memory + queries + visual_window +
recalled_frames + user_input) so the model trains under the exact same input
distribution it sees at inference.

Three sample shapes preserved (canonical pass/SFT/RL/eval timestamped-image protocol):
  A. Single-turn       (silent / response / lonely recall / inter-chunk compress)
  B. Multi-turn recall (recall_query → tool turn → final answer, within one chunk)
  C. Inter-chunk compress (system inserts <compress_trigger>, no visual_window)

Self-contained: imports only stdlib + thinkstream.data.agent_protocol (which
itself is stdlib-only). No transformers required.

INPUT
  data/agent_v5/final/{train_sft_full,val,test}.jsonl   (flat single-step rows)
  data/agent_v5/final/{train_sft,...}_trajectories.jsonl (samples nested)

OUTPUT
  data/agent_v5/final/{train_sft,val,test}_messages.jsonl
  data/agent_v5/final/dataset_info.json   (LLaMA-Factory entry stub)

Usage:
  python -m scripts.agent_data_v5.pass5_messages
  python -m scripts.agent_data_v5.pass5_messages --input flat
  python -m scripts.agent_data_v5.pass5_messages --input traj
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC, SYSTEM_PROMPT_V12, format_memory_block, format_queries_block,
    append_timestamped_image_list,
)

logger = logging.getLogger(__name__)

# Project layout:
#   <batch_root>/                         (DEFAULT_DATA_DIR)
#   <batch_root>/final/*.jsonl            (FINAL_DIR)
#   <batch_root>/frames/<vid>/...jpg      (frame paths in samples)
#
# Frame paths inside samples may be absolute or relative to PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from scripts.agent_data_v5.config import DATA_ROOT as DEFAULT_DATA_DIR
except Exception:
    DEFAULT_DATA_DIR = Path(
        os.environ.get("THINKSTREAM_DATA_ROOT")
        or os.environ.get("AGENT_DATA_DIR")
        or str(PROJECT_ROOT / "data" / "agent_v5")
    )
    if not DEFAULT_DATA_DIR.is_absolute():
        DEFAULT_DATA_DIR = PROJECT_ROOT / DEFAULT_DATA_DIR
    if DEFAULT_DATA_DIR.name == "final":
        DEFAULT_DATA_DIR = DEFAULT_DATA_DIR.parent
FINAL_DIR = DEFAULT_DATA_DIR / "final"

try:
    _FRAME_REL_PREFIX = str((DEFAULT_DATA_DIR / "frames").relative_to(PROJECT_ROOT))
except ValueError:
    _FRAME_REL_PREFIX = str(DEFAULT_DATA_DIR / "frames")

SPLITS = [
    ("train_sft_full", "train_sft_trajectories", "train_sft_messages"),
    ("val", "val_trajectories", "val_messages"),
    ("test", "test_trajectories", "test_messages"),
]

# SFT rows are independent fresh-KV snapshots, unlike RL/eval trajectories
# which must keep a complete replay timeline. Keep all high-information
# actions, then downsample low-information patrol silence so SFT still learns
# silence without drowning recall/compress/answer actions.
SFT_SILENT_TO_ACTIVE_RATIO = 1.25
SFT_ACTIVE_SILENT_FRACTION = 0.70
SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO = 0.35


# ---------------------------------------------------------------------------
# Messages construction (self-contained mirror of data_processor logic)
# ---------------------------------------------------------------------------

def _resolve_paths(paths: List[str], base_path: Path) -> List[str]:
    return [str(base_path / p) if not Path(p).is_absolute() else p for p in paths]


def build_messages(sample: Dict, base_path: Path) -> List[Dict]:
    """Produce v12 ShareGPT messages for one sample. Stdlib-only.

    This is the canonical offline renderer. It must stay aligned with
    thinkstream.data.agent_protocol.build_user_content and the verl RL
    prompt builder: memory, queries, visual_window, timestamped images,
    recalled frames, recall_result, then user input.
    """
    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = float(AGENT_CHUNK_SEC)
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )

    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_V12}]}
    ]

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)

    user_content: List[Dict] = []

    # ── Memory block (FIRST — stable monotonic prefix, v12.12) ──────────
    # Placed first so vLLM prefix-cache can reuse [system + memory_at_t-1]
    # as a prefix of [system + memory_at_t]. See agent_protocol.py
    # build_user_content for full ordering rationale.
    memory_text = format_memory_block(inp.get("memory", {}))
    user_content.append({
        "type": "text",
        "text": f"<memory>\n{memory_text}\n</memory>",
    })

    # ── Queries block (past Q&A history; second-stable prefix) ──────────
    queries = inp.get("queries", [])
    if queries and not inter_chunk:
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": f"\n{qt}"})

    # ── Visual window + frames (cache-miss boundary) ────────────────────
    if not inter_chunk:
        vw = inp["visual_window"]
        current_start = chunk_idx * chunk_sec
        current_end = current_start + chunk_sec
        vw_header = json.dumps({
            "start": vw["video_start"],
            "end": vw["video_end"],
            "frames": vw["frames"],
            "current_time": [current_start, current_end],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<visual_window>{vw_header}</visual_window>",
        })

        # Pass4 flat files may omit frame_paths — infer from video_id +
        # chunk_idx offset (NOT just frame_000001..n which would bind every
        # late chunk to video-start frames). Mirrors pass1a get_chunk_frame_paths
        # (chunk_idx × FRAMES_PER_CHUNK) so frame numbers track real video time.
        if "frame_paths" not in vw and "frames" in vw:
            vid = sample.get("video_id", "")
            if vid:
                from thinkstream.data.agent_protocol import (
                    FRAMES_PER_CHUNK as _FPC,
                    VISUAL_WINDOW_CHUNKS as _VWC,
                )
                from scripts.agent_data_v5.config import (
                    compute_visual_window_start as _cvws,
                )
                window_start = _cvws(chunk_idx, _VWC)
                paths: List[str] = []
                for ci in range(window_start, chunk_idx + 1):
                    for fi in range(_FPC):
                        fnum = ci * _FPC + fi + 1
                        paths.append(
                            f"{_FRAME_REL_PREFIX}/{vid}/frame_{fnum:06d}.jpg"
                        )
                vw["frame_paths"] = paths

        if "frame_paths" in vw:
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
                VISUAL_WINDOW_CHUNKS as _VWC,
            )
            from scripts.agent_data_v5.config import (
                compute_visual_window_start as _cvws,
            )
            window_start = _cvws(chunk_idx, _VWC)
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            append_timestamped_image_list(
                user_content,
                _resolve_paths(vw["frame_paths"], base_path),
                fps=float(_FPC / chunk_sec),
                start_frame_index=window_start * _FPC,
                total_num_frames=(chunk_idx + 1) * _FPC,
                latest_start_frame_index=chunk_idx * _FPC,
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
            )
        elif "frame_indices" in vw and video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": vw["video_start"], "video_end": vw["video_end"],
            })
        else:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: visual_window has neither "
                f"frame_paths nor frame_indices."
            )

    # ── Recalled frames (legacy single-turn recall) ────────────────────
    if (
        "recalled_frames" in inp
        and inp["recalled_frames"]
        and not is_recall_multiturn
        and not inter_chunk
    ):
        rf = inp["recalled_frames"]
        rf_header = json.dumps({
            "time_range": rf["time_range"],
            "source": rf.get("source", "historical_frames"),
            "n_frames": rf["n_frames"],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
        })
        if "frame_paths" in rf:
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
            )
            tr0, tr1 = rf["time_range"]
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            append_timestamped_image_list(
                user_content,
                _resolve_paths(rf["frame_paths"], base_path),
                fps=float(_FPC / chunk_sec),
                start_frame_index=int(tr0 * _FPC),
                total_num_frames=int(tr1 * _FPC),
                context_label="recalled frame",
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
            )
        elif video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })

    # ── Legacy single-turn recall_result (text only, no tool turn) ──────
    if inp.get("recall_result") and not is_recall_multiturn and not inter_chunk:
        rr = inp["recall_result"]
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })

    # ── User input (LAST — every step varies) ───────────────────────────
    if inp.get("user_input"):
        user_content.append({
            "type": "text",
            "text": (f"\n<user_input>{inp['user_input']}</user_input>"
                     if not inter_chunk else f"\n{inp['user_input']}"),
        })

    messages.append({"role": "user", "content": user_content})

    # ── Assistant turn(s) ──────────────────────────────────────────────
    if is_recall_multiturn:
        # Shape B: 2 assistant turns sandwiching a tool turn.
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })

        # Tool turn — recall_result + optional historical frames.
        # v12.11 audit-5 P0 #1 fix (2026-05-01): order MUST mirror runtime
        # (agent_loop.py:942-988): <recalled_frames> + video THEN
        # <recall_result>{...}</recall_result>. Previous order put the
        # raw recall_result JSON FIRST, then frames — train/infer drift
        # for shape-B recall second-turn answer training.
        rr = sample.get("recall_result") or inp.get("recall_result") or {}
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        tool_payload: List[Dict] = []

        rf = inp.get("recalled_frames")
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
            })
            tool_payload.append({
                "type": "text",
                "text": f"<recalled_frames>{rf_header}</recalled_frames>",
            })
            if "frame_paths" in rf:
                from thinkstream.data.agent_protocol import (
                    FRAMES_PER_CHUNK as _FPC,
                )
                from scripts.agent_data_v5.config import (
                    AGENT_CHUNK_SEC as _CHUNK_SEC,
                )
                tr_start, tr_end = rf["time_range"]
                try:
                    from scripts.agent_data_v5.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
                tr_start_chunk = int(tr_start / float(_CHUNK_SEC))
                append_timestamped_image_list(
                    tool_payload,
                    _resolve_paths(rf["frame_paths"], base_path),
                    fps=float(_FPC / float(_CHUNK_SEC)),
                    start_frame_index=tr_start_chunk * _FPC,
                    total_num_frames=int(tr_end / float(_CHUNK_SEC)) * _FPC,
                    context_label="recalled frame",
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )
            elif video_path:
                tool_payload.append({
                    "type": "video", "video": video_path,
                    "video_start": rf["time_range"][0],
                    "video_end": rf["time_range"][1],
                })

        # v12.11 audit-5 P0 #1: append <recall_result> AFTER frames so the
        # token stream matches runtime: [<recalled_frames>{...}, video,
        # <recall_result>{...}</recall_result>].
        tool_payload.append({
            "type": "text",
            "text": f"<recall_result>{rr_json}</recall_result>",
        })

        # DeepEyesV2-aligned ShareGPT has no `tool` role — inject as user content.
        # Qwen3-VL chat_template would otherwise nest <tool_response> under
        # <|im_start|>user, so the on-the-wire token stream is identical.
        messages.append({"role": "user", "content": tool_payload})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_2"]}],
        })
    else:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["output"]}],
        })

    return messages


# ---------------------------------------------------------------------------
# IO + driver
# ---------------------------------------------------------------------------

def _iter_flat(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_trajectories(path: Path) -> Iterable[Dict]:
    """Yield each sample inside trajectory rows, propagating top-level fields."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            video_id = traj.get("video_id", "")
            video_path = traj.get("video_path", "")
            traj_id = traj.get("trajectory_id", "")
            for s in traj.get("samples", []):
                s.setdefault("video_id", video_id)
                s.setdefault("video_path", video_path)
                s.setdefault("trajectory_id", traj_id)
                yield s


def _emit_row(sample: Dict, messages: List[Dict]) -> Dict:
    # v12.12 fix (P0-5): propagate verification verdict + metadata so SFT
    # data_processor can filter / downweight failed samples. pipeline.py
    # tags every sample via pass3e with verification.passed/.fail_reasons
    # but keeps all samples in the trajectory; the consumer (SFT loader)
    # is responsible for the actual drop policy.
    verification = sample.get("verification") or {}
    return {
        "trajectory_id": sample.get("trajectory_id", ""),
        "video_id": sample.get("video_id", ""),
        "chunk_idx": sample.get("chunk_idx", -1),
        "sample_type": sample.get("sample_type", ""),
        "sample_id": sample.get("sample_id", ""),
        "v12_inter_chunk": bool(sample.get("v12_inter_chunk", False)),
        "messages": messages,
        "videos": None,
        "verification": {
            "passed": bool(verification.get("passed", True)),
            "fail_reasons": list(verification.get("fail_reasons") or []),
        },
        # Forward metadata so consumers can compute reward / question lookup
        # without re-running render_samples. (pass4 already reads this; SFT
        # filter consults verification.passed.)
        "metadata": sample.get("metadata") or {},
    }


def _sample_rank(sample: Dict, idx: int) -> str:
    key = "|".join([
        str(sample.get("video_id", "")),
        str(sample.get("trajectory_id", "")),
        str(sample.get("sample_id", "")),
        str(sample.get("chunk_idx", "")),
        str(idx),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _is_active_silent(sample: Dict) -> bool:
    if sample.get("sample_type") != "silent" or sample.get("action") != "silent":
        return False
    meta = sample.get("metadata") or {}
    if sample.get("card_id") or meta.get("question"):
        return True
    return sample.get("sequence_type") in {
        "event_watch",
        "multi_response",
        "recall_success",
        "immediate_response",
    }


def _choose_ranked(items: List[tuple[int, Dict]], n: int) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    return sorted(items, key=lambda x: _sample_rank(x[1], x[0]))[:n]


def _is_multi_emit_response(sample: Dict) -> bool:
    if sample.get("sample_type") != "response":
        return False
    meta = sample.get("metadata") or {}
    return (
        meta.get("question_type") == "multi_emit"
        or meta.get("family") in {"F5", "F7", "PN1"}
    )


def _choose_multi_emit_response(
    items: List[tuple[int, Dict]],
    n: int,
) -> List[tuple[int, Dict]]:
    if n <= 0:
        return []
    by_family: Dict[str, List[tuple[int, Dict]]] = {}
    for item in items:
        meta = item[1].get("metadata") or {}
        fam = meta.get("family") or "unknown"
        by_family.setdefault(fam, []).append(item)
    for fam in by_family:
        by_family[fam] = _choose_ranked(by_family[fam], len(by_family[fam]))

    selected: List[tuple[int, Dict]] = []
    cursors = {fam: 0 for fam in by_family}
    families = sorted(by_family)
    while len(selected) < n:
        progressed = False
        for fam in families:
            cur = cursors[fam]
            bucket = by_family[fam]
            if cur >= len(bucket):
                continue
            selected.append(bucket[cur])
            cursors[fam] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def balance_sft_samples(samples: List[Dict]) -> tuple[List[Dict], Dict[str, int]]:
    """Balance only the SFT messages split.

    Policy:
      - keep every recall / compress row;
      - keep ordinary response rows;
      - cap multi-emit response rows so F5/PN1 do not dominate SFT;
      - keep enough silent rows to make silent roughly 55-60% of SFT;
      - prefer active query-bearing silent rows over patrol/background rows.
    """
    indexed = list(enumerate(samples))
    recall_compress = [
        (i, s) for i, s in indexed
        if s.get("sample_type") in {"recall", "compress"}
    ]
    multi_emit_response = [
        (i, s) for i, s in indexed if _is_multi_emit_response(s)
    ]
    ordinary_response = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "response" and not _is_multi_emit_response(s)
    ]
    other_active = [
        (i, s) for i, s in indexed
        if s.get("sample_type") not in {"silent", "response", "recall", "compress"}
    ]
    multi_limit = min(
        len(multi_emit_response),
        max(1, int(len(ordinary_response) * SFT_MULTI_EMIT_TO_OTHER_RESPONSE_RATIO)),
    )
    active = (
        recall_compress
        + ordinary_response
        + other_active
        + _choose_multi_emit_response(multi_emit_response, multi_limit)
    )
    active_silent = [(i, s) for i, s in indexed if _is_active_silent(s)]
    base_silent = [
        (i, s) for i, s in indexed
        if s.get("sample_type") == "silent" and not _is_active_silent(s)
    ]
    if not active:
        return samples, {"before": len(samples), "after": len(samples)}

    target_silent = min(
        len(active_silent) + len(base_silent),
        max(1, int(len(active) * SFT_SILENT_TO_ACTIVE_RATIO)),
    )
    target_active_silent = min(
        len(active_silent),
        int(target_silent * SFT_ACTIVE_SILENT_FRACTION),
    )
    kept_silent = _choose_ranked(active_silent, target_active_silent)
    remaining = target_silent - len(kept_silent)
    if remaining > 0:
        kept_silent.extend(_choose_ranked(base_silent, remaining))
    if len(kept_silent) < target_silent:
        used = {i for i, _s in kept_silent}
        rest = [(i, s) for i, s in active_silent if i not in used]
        kept_silent.extend(_choose_ranked(rest, target_silent - len(kept_silent)))

    selected = active + kept_silent
    selected.sort(key=lambda x: x[0])
    out = [s for _i, s in selected]
    return out, {
        "before": len(samples),
        "after": len(out),
        "active_kept": len(active),
        "ordinary_response_kept": len(ordinary_response),
        "multi_emit_response_before": len(multi_emit_response),
        "multi_emit_response_kept": min(len(multi_emit_response), multi_limit),
        "recall_compress_kept": len(recall_compress),
        "silent_before": len(active_silent) + len(base_silent),
        "silent_kept": len(kept_silent),
        "active_silent_kept": sum(1 for _i, s in kept_silent if _is_active_silent(s)),
        "base_silent_kept": sum(1 for _i, s in kept_silent if not _is_active_silent(s)),
    }


def convert(
    src: Path,
    dst: Path,
    *,
    is_trajectory: bool,
    base_path: Path,
    limit: Optional[int] = None,
    balance_sft: bool = False,
) -> Dict[str, int]:
    iter_fn = _iter_trajectories if is_trajectory else _iter_flat
    counts = {"ok": 0, "failed": 0}
    by_type: Dict[str, int] = {}
    balance_stats: Dict[str, int] = {}
    sample_iter: Iterable[Dict]
    if balance_sft:
        materialized = list(iter_fn(src))
        materialized, balance_stats = balance_sft_samples(materialized)
        sample_iter = materialized
    else:
        sample_iter = iter_fn(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as out:
        for i, sample in enumerate(sample_iter):
            if limit and counts["ok"] >= limit:
                break
            try:
                messages = build_messages(sample, base_path)
            except (KeyError, ValueError) as exc:
                counts["failed"] += 1
                if counts["failed"] <= 5:
                    sid = sample.get("sample_id") or sample.get("trajectory_id") or i
                    logger.warning(f"[{src.name}] sample {sid} skipped: {exc}")
                continue

            row = _emit_row(sample, messages)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["ok"] += 1
            by_type[row["sample_type"]] = by_type.get(row["sample_type"], 0) + 1

    counts["by_type"] = by_type
    if balance_stats:
        counts["balance"] = balance_stats
    return counts


def write_dataset_info(out_dir: Path, splits_done: List[str]) -> None:
    """LLaMA-Factory entry stub. Aligns with DeepEyesV2 (system/user/assistant only)."""
    entries = {}
    for stem in splits_done:
        entries[f"thinkstream_{stem}"] = {
            "file_name": f"{stem}_messages.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", choices=["flat", "traj", "auto"], default="traj",
        help=(
            "'flat' = *_full.jsonl single-step rows, "
            "'traj' = *_trajectories.jsonl (default — matches main pipeline), "
            "'auto' = prefer flat (legacy; can produce data inconsistent with "
            "the main `pipeline.py` invocation which forces --input traj). "
            "v12.11 audit-5 P1 #4 fix (2026-05-01): default flipped from auto "
            "→ traj to match pipeline.py:1266."
        ),
    )
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    parser.add_argument("--base-path", default=str(PROJECT_ROOT),
                        help="Project root for resolving relative video/frame paths. "
                        "Generated samples store frame paths relative to the repo "
                        "or absolute paths under the batch root.")
    parser.add_argument("--limit", type=int, default=0, help="Per-split sample cap (0 = unlimited).")
    parser.add_argument("--no-balance-sft", action="store_true",
                        help="Disable train_sft_messages silent downsampling.")
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    base_path = Path(args.base_path)
    if not final_dir.exists():
        raise SystemExit(f"final dir not found: {final_dir}")

    splits_done: List[str] = []
    for flat_stem, traj_stem, out_stem in SPLITS:
        flat_path = final_dir / f"{flat_stem}.jsonl"
        traj_path = final_dir / f"{traj_stem}.jsonl"

        if args.input == "flat":
            src, is_traj = flat_path, False
        elif args.input == "traj":
            src, is_traj = traj_path, True
        else:
            if flat_path.exists():
                src, is_traj = flat_path, False
            elif traj_path.exists():
                src, is_traj = traj_path, True
            else:
                logger.warning(f"No input found for split {flat_stem}/{traj_stem}, skipping.")
                continue

        if not src.exists():
            logger.warning(f"Input missing: {src}, skipping.")
            continue

        dst = final_dir / f"{out_stem}.jsonl"
        logger.info(f"Converting {src.name} → {dst.name} (is_trajectory={is_traj})")
        balance = out_stem == "train_sft_messages" and not args.no_balance_sft
        counts = convert(src, dst, is_trajectory=is_traj, base_path=base_path,
                         limit=args.limit or None, balance_sft=balance)
        logger.info(
            f"  ok={counts['ok']} failed={counts['failed']} by_type={counts['by_type']}"
        )
        if counts.get("balance"):
            logger.info(f"  SFT balance: {counts['balance']}")
        splits_done.append(out_stem.replace("_messages", ""))

    if splits_done:
        write_dataset_info(final_dir, splits_done)
        logger.info(f"Wrote dataset_info.json → {final_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
