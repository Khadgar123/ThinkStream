"""Convert v11 2s/chunk flat data → v12.5 1s/chunk format.

Best-effort conversion rules:
1. chunk_idx stays the same numeric value (sample ordering preserved).
2. All time fields are halved so that 2s-chunk semantics become 1s-chunk
   semantics: video_start, video_end, current_time, time_range, etc.
3. frames count stays the same (2 frames per chunk) — this makes the
   implied fps double from 1→2, matching the new code.
4. System prompt string is rewritten from "2-second" to "1-second".
5. Memory block think timestamps are halved if they are numeric seconds.

Input:  data/agent_v5_current_backup/final/{train_sft_full,val,test}.jsonl
Output: data/agent_v5_current_backup/final/v12_{train_sft_full,val,test}.jsonl
        (then run pass5_messages.py on these to get messages)

Limitations:
- Think/observation text was written for a 2-second interval; after
  conversion it is still the same text but now labelled as 1-second.
  This is an inherent limitation — true 1s granularity requires re-running
  the teacher pipeline (pass1a/pass2).
- Visual window was 12 chunks × 2s = 24s under old config; after halving
  it becomes 12 chunks × 1s = 12s, which is shorter than the new VWC=16s.
  This is acceptable for debug but not for production training.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "agent_v5_current_backup" / "final"

SPLITS = ["train_sft_full", "val", "test"]


def _halve_time_field(val: Any) -> Any:
    """Halve a time value if it is a number or a numeric string."""
    if isinstance(val, (int, float)):
        return val / 2.0
    if isinstance(val, str):
        # Try to parse as number
        try:
            return str(float(val) / 2.0)
        except ValueError:
            pass
        # Handle ranges like "10-30" or "10.0-30.0"
        m = re.match(r"^([\d.]+)-([\d.]+)$", val.strip())
        if m:
            return f"{float(m.group(1))/2.0}-{float(m.group(2))/2.0}"
    return val


def _halve_time_range_list(lst: List) -> List:
    """Halve numeric elements in a [start, end] list."""
    if isinstance(lst, list) and len(lst) == 2:
        return [_halve_time_field(lst[0]), _halve_time_field(lst[1])]
    return lst


def convert_sample(sample: Dict) -> Optional[Dict]:
    s = dict(sample)
    inp = s.get("input")
    if not isinstance(inp, dict):
        return s

    # 1. System prompt text: "2-second" → "1-second", "24s" → "16s"
    system = inp.get("system", "")
    if isinstance(system, str):
        system = system.replace("2-second", "1-second")
        system = system.replace("2 second", "1 second")
        system = system.replace("24s window", "16s window")
        system = system.replace("24s of visual", "16s of visual")
        inp["system"] = system

    # 2. Visual window time fields
    vw = inp.get("visual_window")
    if isinstance(vw, dict):
        for key in ("video_start", "video_end"):
            if key in vw:
                vw[key] = _halve_time_field(vw[key])
        # current_time is usually [start, end]
        if "current_time" in vw:
            vw["current_time"] = _halve_time_range_list(vw["current_time"])

    # 3. Recalled frames time_range
    rf = inp.get("recalled_frames")
    if isinstance(rf, dict):
        if "time_range" in rf:
            rf["time_range"] = _halve_time_range_list(rf["time_range"])

    # 4. Memory block: halve think timestamps if they look like seconds
    memory = inp.get("memory", {})
    if isinstance(memory, dict):
        for key in ("recent_thinks", "compressed_segments"):
            arr = memory.get(key)
            if isinstance(arr, list):
                new_arr = []
                for item in arr:
                    if isinstance(item, dict):
                        item = dict(item)
                        if "time" in item:
                            item["time"] = _halve_time_field(item["time"])
                        if isinstance(item.get("time_range"), list):
                            item["time_range"] = _halve_time_range_list(item["time_range"])
                    new_arr.append(item)
                memory[key] = new_arr

    # 5. Queries: halve time fields
    queries = inp.get("queries")
    if isinstance(queries, list):
        new_queries = []
        for q in queries:
            if isinstance(q, dict):
                q = dict(q)
                if "time" in q:
                    q["time"] = _halve_time_field(q["time"])
                if "time_range" in q:
                    q["time_range"] = _halve_time_field(q["time_range"])
            new_queries.append(q)
        inp["queries"] = new_queries

    # 6. Output text: heuristic replacement of "2s" / "2-second" references
    out = s.get("output")
    if isinstance(out, str):
        out = out.replace("2-second", "1-second")
        out = out.replace("2 second", "1 second")
        s["output"] = out

    # 7. v12_assistant_turns if present
    for turn_key in ("v12_assistant_turn_1", "v12_assistant_turn_2"):
        turn = s.get(turn_key)
        if isinstance(turn, str):
            turn = turn.replace("2-second", "1-second")
            turn = turn.replace("2 second", "1 second")
            s[turn_key] = turn

    return s


def convert_flat_file(src: Path, dst: Path) -> Dict[str, int]:
    counts = {"ok": 0, "skipped": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                counts["skipped"] += 1
                continue
            converted = convert_sample(sample)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["ok"] += 1
    return counts


def convert_trajectory_file(src: Path, dst: Path) -> Dict[str, int]:
    counts = {"ok": 0, "skipped": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                counts["skipped"] += 1
                continue
            traj = dict(traj)
            samples = traj.get("samples")
            if isinstance(samples, list):
                traj["samples"] = [convert_sample(s) for s in samples]
            fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
            counts["ok"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)

    # Flat files
    for stem in SPLITS:
        src = backup_dir / f"{stem}.jsonl"
        dst = backup_dir / f"v12_{stem}.jsonl"
        if not src.exists():
            print(f"SKIP flat: {src} not found")
            continue
        if args.dry_run:
            print(f"DRY-RUN flat: {src.name} → {dst.name}")
            continue
        print(f"Converting flat {src.name} → {dst.name}")
        counts = convert_flat_file(src, dst)
        print(f"  ok={counts['ok']} skipped={counts['skipped']}")

    # Trajectory files (RL)
    traj_splits = ["train_sft", "train_rl", "val", "test"]
    for stem in traj_splits:
        src = backup_dir / f"{stem}_trajectories.jsonl"
        dst = backup_dir / f"v12_{stem}_trajectories.jsonl"
        if not src.exists():
            print(f"SKIP traj: {src} not found")
            continue
        if args.dry_run:
            print(f"DRY-RUN traj: {src.name} → {dst.name}")
            continue
        print(f"Converting traj {src.name} → {dst.name}")
        counts = convert_trajectory_file(src, dst)
        print(f"  ok={counts['ok']} skipped={counts['skipped']}")

    print("\nNext step: swap v12_*.jsonl into place and run pass5_messages.py")


if __name__ == "__main__":
    main()
