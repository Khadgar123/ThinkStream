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

    # ── Visual window + frames ────────────────────────────────────────
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
            "text": f"<visual_window>{vw_header}</visual_window>",
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
                window_start = max(0, chunk_idx - _VWC + 1)
                paths: List[str] = []
                for ci in range(window_start, chunk_idx + 1):
                    for fi in range(_FPC):
                        fnum = ci * _FPC + fi + 1
                        paths.append(
                            f"data/agent_v5/frames/{vid}/frame_{fnum:06d}.jpg"
                        )
                vw["frame_paths"] = paths

        if "frame_paths" in vw:
            user_content.append({
                "type": "video",
                "video": _resolve_paths(vw["frame_paths"], base_path),
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
            user_content.append({
                "type": "video",
                "video": _resolve_paths(rf["frame_paths"], base_path),
            })
        elif video_path:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })

    # ── Memory block ────────────────────────────────────────────────────
    memory_text = format_memory_block(inp.get("memory", {}))
    user_content.append({
        "type": "text",
        "text": (f"\n<memory>\n{memory_text}\n</memory>" if not inter_chunk
                 else f"<memory>\n{memory_text}\n</memory>"),
    })

    # ── Queries block (past Q&A history) ────────────────────────────────
    queries = inp.get("queries", [])
    if queries and not inter_chunk:
        qt = format_queries_block(queries)
        if qt:
            user_content.append({"type": "text", "text": f"\n{qt}"})

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

    # ── User input (question / compress_trigger / "Continue...") ────────
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
        rr = sample.get("recall_result") or inp.get("recall_result") or {}
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        tool_payload: List[Dict] = [{"type": "text", "text": rr_json}]

        rf = inp.get("recalled_frames")
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
            })
            tool_payload.append({
                "type": "text",
                "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
            })
            if "frame_paths" in rf:
                tool_payload.append({
                    "type": "video",
                    "video": _resolve_paths(rf["frame_paths"], base_path),
                })
            elif video_path:
                tool_payload.append({
                    "type": "video", "video": video_path,
                    "video_start": rf["time_range"][0],
                    "video_end": rf["time_range"][1],
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
    return {
        "trajectory_id": sample.get("trajectory_id", ""),
        "video_id": sample.get("video_id", ""),
        "chunk_idx": sample.get("chunk_idx", -1),
        "sample_type": sample.get("sample_type", ""),
        "sample_id": sample.get("sample_id", ""),
        "v12_inter_chunk": bool(sample.get("v12_inter_chunk", False)),
        "messages": messages,
        "videos": None,
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
        "--input", choices=["flat", "traj", "auto"], default="auto",
        help="'flat' = *_full.jsonl single-step rows, 'traj' = *_trajectories.jsonl, 'auto' = prefer flat.",
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
