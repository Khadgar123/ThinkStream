import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data.pre_rl_rollout_audit import (  # noqa: E402
    _load_source_rows,
    _trajectory_max_chunk,
)


class PreRLAuditLoaderTest(unittest.TestCase):
    def test_loads_multi_q_training_parquet_as_trajectory(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest("pandas unavailable") from exc

        row = {
            "video_id": "video_001",
            "video_path": "video_001.mp4",
            "n_chunks": 6,
            "extra_info": {
                "questions": [{
                    "question": "What color is the cup?",
                    "ask_chunks": [1],
                    "answer_chunks": [4],
                    "gold_answer": "red",
                }],
                "gold_action_per_chunk": {"1": "silent", "4": "response"},
                "offline_compress_chunks": [2, 5],
            },
            "reward_model": {
                "ground_truth": json.dumps({
                    "questions": [],
                    "gold_action_per_chunk": {},
                    "offline_compress_chunks": [],
                }),
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train_rl_multi_q.parquet"
            try:
                pd.DataFrame([row]).to_parquet(path)
            except Exception as exc:  # pragma: no cover
                raise unittest.SkipTest(f"parquet writer unavailable: {exc}") from exc

            loaded = _load_source_rows(path)

        self.assertEqual(len(loaded), 1)
        traj = loaded[0]
        self.assertEqual(traj["video_id"], "video_001")
        self.assertEqual(traj["offline_compress_chunks"], [2, 5])
        self.assertEqual(traj["gold_action_per_chunk"], {"1": "silent", "4": "response"})
        self.assertEqual(traj["questions"][0]["answer_chunks"], [4])
        self.assertEqual(_trajectory_max_chunk(traj), 5)


if __name__ == "__main__":
    unittest.main()
