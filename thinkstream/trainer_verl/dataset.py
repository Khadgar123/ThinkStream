"""verl dataset adapter for ThinkStream pass5 messages format.

Reads `data/agent_v5/final/train_rl_trajectories.jsonl` (one trajectory
per line) and exposes verl's expected interface:

    {
      "prompt":      List[Dict]   # full chat messages prefix (system + initial user)
      "ground_truth": Dict        # gold_answer + ask_chunks + ... for reward_fn
      "extra_info":  Dict         # video_uid + chunk_idx + ...
    }

The trajectory is materialized as a SEED state (chunk_idx=0, empty memory);
the rollout loop drives it forward turn-by-turn via VerlChunkLevelRollout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Iterator, Optional

from torch.utils.data import Dataset

from thinkstream.data.agent_protocol import (
    TOOLS_SCHEMA,
    normalize_frame_protocol,
    system_prompt_for_frame_protocol,
)
from thinkstream.trainer.v12_rollout import VideoTrajectoryState


class ThinkStreamRLDataset(Dataset):
    """Trajectory-keyed RL dataset.

    One row = one (video, trajectory) seed. RL rollout per row produces G
    × N_chunks completions; reward computed at trajectory end via the
    aggregated outcome from `compute_trajectory_outcome_v12`.
    """

    def __init__(
        self,
        annotation_path: str,
        *,
        max_questions_per_traj: int = 5,
        frame_protocol: str = "ts_image",
    ):
        self.path = Path(annotation_path)
        self.max_questions_per_traj = max_questions_per_traj
        self.frame_protocol = normalize_frame_protocol(frame_protocol)
        self._index: List[Dict] = self._load(self.path)

    def _load(self, path: Path) -> List[Dict]:
        rows: List[Dict] = []
        opener = open
        if str(path).endswith(".gz"):
            import gzip
            opener = gzip.open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        traj = self._index[idx]
        questions = (traj.get("questions") or [])[: self.max_questions_per_traj]

        # Initial seed state (chunk_idx=0, empty memory). The rollout loop
        # advances this; we don't materialize all chunks here — that happens
        # at rollout time.
        seed_state = VideoTrajectoryState(
            video_uid=traj.get("video_id", ""),
            chunk_idx=0,
        )

        # Pre-compute ground_truth bundle expected by reward_fn
        ground_truth = {
            "gold_answer":            "",   # filled per-question at scoring
            "answer_form":            "",
            "ask_chunks":             [],
            "visible_start_chunk":    None,
            "visible_end_chunk":      None,
            "gold_action_per_chunk":  traj.get("gold_action_per_chunk", {}),
            "questions":              questions,
            # support_chunks intentionally NOT exposed to reward_fn — we no
            # longer reward against this gold under the v12.6 minimal scheme.
        }
        if questions:
            q0 = questions[0]
            # v12.13 fix (P1-4): visible_end_chunk extends to last
            # answer_chunk (was max(ask_chunks) — which is identical to
            # ask_chunk for forward cards, missing the entire silent-
            # then-respond window where the answer actually lands).
            answer_chunks = q0.get("answer_chunks") or []
            ask_chunks = q0.get("ask_chunks", [])
            ask_chunk_canonical = q0.get("ask_chunk")
            visible_start = (
                ask_chunk_canonical
                if isinstance(ask_chunk_canonical, int) and ask_chunk_canonical >= 0
                else (min(ask_chunks) if ask_chunks else None)
            )
            visible_end = (
                max(answer_chunks)
                if answer_chunks
                else (max(ask_chunks) if ask_chunks else None)
            )
            ground_truth.update({
                "gold_answer": q0.get("gold_answer", ""),
                "answer_form": q0.get("answer_form", ""),
                "ask_chunks":  ask_chunks,
                "answer_chunks": list(answer_chunks),     # NEW
                "ask_chunk":     ask_chunk_canonical,     # NEW
                "per_emit_answers": list(q0.get("per_emit_answers") or []),  # NEW
                "options":       list(q0.get("options") or []),              # NEW (P0-3)
                "correct_option": q0.get("correct_option", ""),              # NEW (P0-3)
                "visible_start_chunk": visible_start,
                "visible_end_chunk":   visible_end,
            })

        return {
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt_for_frame_protocol(
                        self.frame_protocol
                    ),
                },
            ],
            "ground_truth": ground_truth,
            "extra_info": {
                "video_uid": traj.get("video_id", ""),
                "video_path": traj.get("video_path", ""),
                "trajectory_id": traj.get("trajectory_id", ""),
                "n_chunks_total": traj.get("stats", {}).get("n_chunks_covered", 0),
                "seed_state": seed_state,
                "tools": TOOLS_SCHEMA,
                "frame_protocol": self.frame_protocol,
            },
        }


def build_rl_dataset(
    final_dir: str = "data/agent_v5/final",
    split: str = "train_rl",
    **kwargs,
) -> ThinkStreamRLDataset:
    """Convenience builder. Defaults to pass4-emitted train_rl_trajectories."""
    base = Path(final_dir)
    candidates = [
        base / f"{split}_trajectories.jsonl",
        base / f"{split}_trajectories.jsonl.gz",
    ]
    for p in candidates:
        if p.exists():
            return ThinkStreamRLDataset(str(p), **kwargs)
    raise FileNotFoundError(
        f"No trajectory file found for split={split} under {base}. "
        f"Looked for: {[str(c) for c in candidates]}"
    )
