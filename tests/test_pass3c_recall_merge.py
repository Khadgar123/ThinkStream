"""Regression test for pass3c v12.5 recall-pair merge bug.

The bug: `_merge_recall_pairs_v12` was called inside `generate_base_samples`,
which never contains recall pairs (those live in `generate_trajectory_samples`).
Result: 0 multi-turn samples produced despite v12 protocol claiming support
for them. Fixed by moving the merge to `generate_trajectory_samples`'
final return path.

Run: python tests/test_pass3c_recall_merge.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set env BEFORE importing pass3c (PROTOCOL_VERSION reads at module load)
os.environ.setdefault("THINKSTREAM_PROTOCOL", "v12")


def test_merge_pairs_collapses_recall_at_same_chunk():
    """Two samples (recall_query + recall_response) at same (traj, card,
    chunk) → one merged sample with v12_assistant_turn_1/2."""
    from scripts.agent_data_v5.pass3c_samples import _merge_recall_pairs_v12
    samples = [
        {"sample_type": "recall_query", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 5, "output": "<tool_call>...</tool_call>"},
        {"sample_type": "recall_response", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 5, "output": "<answer>red</answer>",
         "recall_result": {"chunks": [3, 4]}},
        {"sample_type": "silent", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 0, "output": "<answer></answer>"},
    ]
    merged = _merge_recall_pairs_v12(samples)
    # 1 silent (untouched) + 1 merged "recall" (was 2)
    assert len(merged) == 2, f"expected 2 samples after merge, got {len(merged)}"
    recall_samples = [s for s in merged if s["sample_type"] == "recall"]
    assert len(recall_samples) == 1
    r = recall_samples[0]
    assert r.get("v12_assistant_turn_1") == "<tool_call>...</tool_call>"
    assert r.get("v12_assistant_turn_2") == "<answer>red</answer>"
    assert r.get("recall_result") == {"chunks": [3, 4]}
    assert "output" not in r  # legacy single-output dropped
    print(f"  PASS merge produces 1 multi-turn 'recall' from rq+rr pair")


def test_merge_handles_recall_silent():
    """recall_query + recall_silent → merged with empty answer."""
    from scripts.agent_data_v5.pass3c_samples import _merge_recall_pairs_v12
    samples = [
        {"sample_type": "recall_query", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 5, "output": "<tool_call>...</tool_call>"},
        {"sample_type": "recall_silent", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 5, "output": "<answer></answer>"},
    ]
    merged = _merge_recall_pairs_v12(samples)
    assert len(merged) == 1
    r = merged[0]
    assert r["sample_type"] == "recall"
    assert r["v12_assistant_turn_2"] == "<answer></answer>"
    assert r.get("v12_post_recall_was_silent") is True
    print("  PASS merge handles recall_silent (post_recall_was_silent=True)")


def test_merge_unpaired_recall_response_promoted():
    """Lonely recall_response (no recall_query at same chunk) → promoted to
    plain response."""
    from scripts.agent_data_v5.pass3c_samples import _merge_recall_pairs_v12
    samples = [
        {"sample_type": "recall_response", "trajectory_id": "t0",
         "card_id": "c1", "chunk_idx": 5, "output": "<answer>red</answer>"},
    ]
    merged = _merge_recall_pairs_v12(samples)
    assert len(merged) == 1
    assert merged[0]["sample_type"] == "response"
    print("  PASS lonely recall_response promoted to 'response'")


def test_call_site_is_in_generate_trajectory_samples():
    """AST: ensure the merge call lives in generate_trajectory_samples,
    NOT in generate_base_samples (which would be a no-op)."""
    import ast
    src = Path(__file__).resolve().parents[1] / "scripts" / "agent_data_v5" / "pass3c_samples.py"
    tree = ast.parse(src.read_text())

    fn_calling_merge = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Name) and f.id == "_merge_recall_pairs_v12":
                    fn_calling_merge.append(node.name)
                    break

    # generate_base_samples MUST NOT call merge (no-op)
    assert "generate_base_samples" not in fn_calling_merge, (
        "REGRESSION: _merge_recall_pairs_v12 still called from "
        "generate_base_samples (where there are no recall pairs)"
    )
    # generate_trajectory_samples MUST call merge
    assert "generate_trajectory_samples" in fn_calling_merge, (
        "merge must be called from generate_trajectory_samples (where "
        "recall pairs actually live)"
    )
    print(f"  PASS merge call site: {fn_calling_merge}")


def main():
    tests = [
        test_merge_pairs_collapses_recall_at_same_chunk,
        test_merge_handles_recall_silent,
        test_merge_unpaired_recall_response_promoted,
        test_call_site_is_in_generate_trajectory_samples,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
