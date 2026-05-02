"""PASS 5 — Convert single-step samples to LLaMA-Factory ShareGPT messages format.

Reads pass4 outputs and emits one row per sample in the multi-turn messages
format used by LLaMA-Factory / DeepEyesV2 / VST. Each row is a stand-alone
training sample matching fresh-KV-per-chunk inference: every sample's
user.content carries the full state (memory + queries + visual_window +
recalled_frames + user_input) so the model trains under the exact same input
distribution it sees at inference.

Three sample shapes preserved (mirrors data_processor.build_per_timestep_messages_v12):
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
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC, SYSTEM_PROMPT_V12, format_memory_block, format_queries_block,
)

logger = logging.getLogger(__name__)

# Project layout:
#   PROJECT_ROOT/data/agent_v5/                         (DEFAULT_DATA_DIR)
#   PROJECT_ROOT/data/agent_v5/final/*.jsonl            (FINAL_DIR)
#   PROJECT_ROOT/data/agent_v5/frames/<vid>/...jpg      (frame paths in samples)
#
# Frame paths inside samples are stored relative to PROJECT_ROOT
# (e.g. "data/agent_v5/frames/<vid>/frame_000001.jpg"), so base_path used
# to resolve them MUST be PROJECT_ROOT — NOT data/. Earlier bug: default
# base_path was DEFAULT_DATA_DIR.parent = .../data/, which produced
# .../data/data/agent_v5/frames/... at resolution time.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = Path(
    os.environ.get("AGENT_DATA_DIR", str(PROJECT_ROOT / "data" / "agent_v5"))
)
FINAL_DIR = DEFAULT_DATA_DIR / "final"

SPLITS = [
    ("train_sft_full", "train_sft_trajectories", "train_sft_messages"),
    ("val", "val_trajectories", "val_messages"),
    ("test", "test_trajectories", "test_messages"),
]


# ---------------------------------------------------------------------------
# Messages construction (self-contained mirror of data_processor logic)
# ---------------------------------------------------------------------------

def _resolve_paths(paths: List[str], base_path: Path) -> List[str]:
    return [str(base_path / p) if not Path(p).is_absolute() else p for p in paths]


def build_messages(sample: Dict, base_path: Path) -> List[Dict]:
    """Produce v12 ShareGPT messages for one sample. Stdlib-only.

    Mirrors thinkstream.sft.data_processor.build_per_timestep_messages_v12,
    so the output is byte-identical (modulo path resolution edge cases) to
    what the SFT data loader synthesizes online.
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
                            f"data/agent_v5/frames/{vid}/frame_{fnum:06d}.jpg"
                        )
                vw["frame_paths"] = paths

        if "frame_paths" in vw:
            # v12.6: attach video_metadata so Qwen3-VL processor renders
            # per-frame `<X.X seconds>` text tokens with REAL video time.
            # Without metadata, processor defaults to fps=24 + indices=0..N
            # → timestamps anchored at sequence start, not real video time.
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
                VISUAL_WINDOW_CHUNKS as _VWC,
            )
            from scripts.agent_data_v5.config import (
                compute_visual_window_start as _cvws,
            )
            n_frames = len(vw["frame_paths"])
            window_start = _cvws(chunk_idx, _VWC)
            # v12.12: runtime mm_processor_kwargs at video item level so
            # qwen-vl-utils.process_vision_info forwards them to vLLM as
            # smart_resize bounds. Matches pass2 / inference / RL rollout.
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            user_content.append({
                "type": "video",
                "video": _resolve_paths(vw["frame_paths"], base_path),
                "min_pixels": _RTKW["min_pixels"],
                "max_pixels": _RTKW["max_pixels"],
                "video_metadata": {
                    "fps": float(_FPC / chunk_sec),
                    "frames_indices": [
                        window_start * _FPC + i for i in range(n_frames)
                    ],
                    "total_num_frames": (chunk_idx + 1) * _FPC,
                },
            })
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
            # v12.6: anchor recalled frames at their REAL video time so
            # Qwen3-VL renders `<X.X seconds>` matching when the frame
            # originally appeared, not where it lands in the sequence.
            from thinkstream.data.agent_protocol import (
                FRAMES_PER_CHUNK as _FPC,
            )
            tr0, tr1 = rf["time_range"]
            n_rf = len(rf["frame_paths"])
            try:
                from scripts.agent_data_v5.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 130_000, "max_pixels": 220_000}
            user_content.append({
                "type": "video",
                "video": _resolve_paths(rf["frame_paths"], base_path),
                "min_pixels": _RTKW["min_pixels"],   # v12.12
                "max_pixels": _RTKW["max_pixels"],
                "video_metadata": {
                    "fps": float(_FPC / chunk_sec),
                    "frames_indices": [
                        int(tr0 * _FPC) + i for i in range(n_rf)
                    ],
                    "total_num_frames": int(tr1 * _FPC),
                },
            })
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
                # v12.11 P1.1 fix (2026-05-01): attach video_metadata so the
                # Qwen3-VL processor renders per-frame `<X.X seconds>` text
                # tokens at the recalled frames' ORIGINAL video time. Without
                # metadata, the processor defaults to fps=24 + indices=0..N-1
                # → recalled frames anchor at "frame 0" instead of their
                # historical timestamps, breaking the design intent of
                # "recall复用原始 MROPE 时间编码" (the model can't tell that
                # these are old frames from time T).
                from thinkstream.data.agent_protocol import (
                    FRAMES_PER_CHUNK as _FPC,
                )
                from scripts.agent_data_v5.config import (
                    AGENT_CHUNK_SEC as _CHUNK_SEC,
                )
                tr_start, tr_end = rf["time_range"]
                n_rf = len(rf["frame_paths"])
                # historical frame indices = (tr_start_chunk * FRAMES_PER_CHUNK +
                # 0..n_rf-1), mirrors the original encoding at recall time.
                tr_start_chunk = int(tr_start / float(_CHUNK_SEC))
                tool_payload.append({
                    "type": "video",
                    "video": _resolve_paths(rf["frame_paths"], base_path),
                    "video_metadata": {
                        "fps": float(_FPC / float(_CHUNK_SEC)),
                        "frames_indices": [
                            tr_start_chunk * _FPC + i for i in range(n_rf)
                        ],
                        # total_num_frames anchors the timestamp scale; use
                        # tr_end_chunk * FRAMES_PER_CHUNK as ceiling.
                        "total_num_frames": int(tr_end / float(_CHUNK_SEC)) * _FPC,
                    },
                })
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


def convert(
    src: Path,
    dst: Path,
    *,
    is_trajectory: bool,
    base_path: Path,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    iter_fn = _iter_trajectories if is_trajectory else _iter_flat
    counts = {"ok": 0, "failed": 0}
    by_type: Dict[str, int] = {}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as out:
        for i, sample in enumerate(iter_fn(src)):
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
                        help="Project root for resolving relative video/frame paths "
                        "(samples store paths like 'data/agent_v5/frames/...'; "
                        "base_path must be PROJECT_ROOT, NOT data/).")
    parser.add_argument("--limit", type=int, default=0, help="Per-split sample cap (0 = unlimited).")
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
        counts = convert(src, dst, is_trajectory=is_traj, base_path=base_path,
                         limit=args.limit or None)
        logger.info(
            f"  ok={counts['ok']} failed={counts['failed']} by_type={counts['by_type']}"
        )
        splits_done.append(out_stem.replace("_messages", ""))

    if splits_done:
        write_dataset_info(final_dir, splits_done)
        logger.info(f"Wrote dataset_info.json → {final_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
