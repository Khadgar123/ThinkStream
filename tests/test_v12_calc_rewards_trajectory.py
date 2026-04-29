"""v12.4 _calc_rewards_v12_trajectory integration test.

Verifies the trajectory-level reward path produces sensible scores for
multi-question rollouts. Uses a stub tokenizer that round-trips text.

Importing thinkstream.trainer.grpo loads transformers, which may fail on
envs with tokenizers/transformers version mismatch. We do an AST verification
of `_calc_rewards_v12_trajectory` first, then attempt the runtime test
(soft-skip on env errors).

Run: python tests/test_v12_calc_rewards_trajectory.py
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _grpo_path():
    return Path(__file__).resolve().parents[1] / "thinkstream" / "trainer" / "grpo.py"


def test_calc_rewards_v12_trajectory_function_exists():
    """AST: _calc_rewards_v12_trajectory must be defined in grpo.py with
    expected signature (rollout_data, *, group_size, tokenizer, ...)."""
    tree = ast.parse(_grpo_path().read_text())
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_calc_rewards_v12_trajectory"):
            found = True
            kw_args = {a.arg for a in node.args.kwonlyargs}
            assert "group_size" in kw_args, "missing group_size kwarg"
            assert "tokenizer" in kw_args, "missing tokenizer kwarg"
            break
    assert found, "_calc_rewards_v12_trajectory not defined in grpo.py"
    print("  PASS _calc_rewards_v12_trajectory defined with expected signature")


def _try_import_grpo():
    """Returns the grpo module or None (with reason printed) on env failure."""
    try:
        from thinkstream.trainer import grpo  # noqa: F401
        return grpo
    except ImportError as e:
        if "tokenizers" in str(e) or "transformers" in str(e):
            print(f"  SKIP runtime tests: env issue ({e})")
            return None
        raise


class _StubTokenizer:
    """Trivial tokenizer: encode = list of ord(c), decode = chr(c) join.
    Wraps tokens as a flat list-of-ints so it round-trips trivially."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(c) for c in tokens)


def _make_rollout_data(questions, gold_action_per_chunk, per_g_outputs):
    """Build a rollout_data list for one trajectory with G rollouts.

    per_g_outputs: list of G strings per chunk; outer list is chunks.
    Returns list-of-1 (one trajectory) suitable for _calc_rewards_v12_trajectory.
    """
    tk = _StubTokenizer()
    chunk_results = []
    for chunk_idx, group_texts in enumerate(per_g_outputs):
        chunk_results.append({
            "chunk_idx": chunk_idx,
            "generated_tokens": [tk.encode(t) for t in group_texts],
        })
    return [{
        "raw_sample": {
            "video_id": "vid_test",
            "trajectory_id": "traj_0",
            "questions": questions,
            "gold_action_per_chunk": gold_action_per_chunk,
        },
        "chunk_results": chunk_results,
    }], tk


def test_trajectory_reward_perfect_run():
    """All questions answered correctly, all silents correct → top score."""
    grpo = _try_import_grpo()
    if grpo is None:
        return
    _calc_rewards_v12_trajectory = grpo._calc_rewards_v12_trajectory

    questions = [
        {"card_id": "c1", "gold_answer": "red apron",
         "answer_form": "literal", "ask_chunks": [3]},
    ]
    gold_action_per_chunk = {
        "0": "silent", "1": "silent", "2": "silent",
        "3": "response",
        "4": "silent", "5": "silent",
    }
    G = 2
    per_g_outputs = [
        ["<think>nothing</think><answer></answer>"] * G,  # chunk 0: silent
        ["<think>nothing</think><answer></answer>"] * G,  # chunk 1: silent
        ["<think>nothing</think><answer></answer>"] * G,  # chunk 2: silent
        ["<think>seen apron</think><answer>red apron</answer>"] * G,  # chunk 3: response
        ["<think>moved on</think><answer></answer>"] * G,  # chunk 4: silent
        ["<think>moved on</think><answer></answer>"] * G,  # chunk 5: silent
    ]
    rollout_data, tk = _make_rollout_data(questions, gold_action_per_chunk, per_g_outputs)
    rewards, rewards_dict, masks = _calc_rewards_v12_trajectory(
        rollout_data, group_size=G, tokenizer=tk,
    )
    # B = 1 trajectory × G = 2
    assert rewards.shape == (G,), rewards.shape
    # Both rollouts perfect — should have high reward
    for g in range(G):
        assert rewards_dict["outcome"][g].item() == 1.0
        assert rewards_dict["timing"][g].item() == 1.0
        assert rewards_dict["silent_quality"][g].item() == 0.3
        # Total (before mask): outcome=1×1.0 + timing=1×0.3 + format=1×0.1 +
        #                      spam=0×(-0.2) + silent_q=0.3×0.2 = 1.46
        assert abs(rewards[g].item() - 1.46) < 1e-4, rewards[g].item()
    print("  PASS perfect run: rewards = 1.46 each")


def test_trajectory_reward_multi_question_partial():
    """3 questions, 1 correct + 1 wrong + 1 silent → outcome=0.33, mixed timing."""
    grpo = _try_import_grpo()
    if grpo is None:
        return
    _calc_rewards_v12_trajectory = grpo._calc_rewards_v12_trajectory

    questions = [
        {"card_id": "c1", "gold_answer": "red",
         "answer_form": "literal", "ask_chunks": [2]},
        {"card_id": "c2", "gold_answer": "blue",
         "answer_form": "literal", "ask_chunks": [5]},
        {"card_id": "c3", "gold_answer": "green",
         "answer_form": "literal", "ask_chunks": [8]},
    ]
    gold_action_per_chunk = {
        "0": "silent", "1": "silent", "2": "response",
        "3": "silent", "4": "silent", "5": "response",
        "6": "silent", "7": "silent", "8": "response",
    }
    G = 1
    # Q1 correct, Q2 wrong, Q3 missed
    per_g_outputs = [
        ["<answer></answer>"], ["<answer></answer>"],
        ["<answer>red</answer>"],     # Q1 correct
        ["<answer></answer>"], ["<answer></answer>"],
        ["<answer>purple</answer>"],  # Q2 wrong
        ["<answer></answer>"], ["<answer></answer>"],
        ["<answer></answer>"],        # Q3 missed
    ]
    rollout_data, tk = _make_rollout_data(questions, gold_action_per_chunk, per_g_outputs)
    rewards, rewards_dict, masks = _calc_rewards_v12_trajectory(
        rollout_data, group_size=G, tokenizer=tk,
    )
    # outcome = 1/3, timing = (+1 + +1 + -0.5)/3 = 0.5
    # silent_quality: 6 silents correct (+0.3 each), 2 responses correct (0.0),
    #                 1 missed response (-0.6) → mean = (6×0.3 + 0 - 0.6) / 9 = 0.133
    assert abs(rewards_dict["outcome"][0].item() - 1/3) < 1e-3
    assert abs(rewards_dict["timing"][0].item() - 0.5) < 1e-3
    sq = rewards_dict["silent_quality"][0].item()
    assert abs(sq - (6 * 0.3 - 0.6) / 9) < 1e-3, sq
    print(f"  PASS multi-q partial: outcome={rewards_dict['outcome'][0]:.3f}, "
          f"timing={rewards_dict['timing'][0]:.3f}, silent_q={sq:.3f}")


def test_trajectory_reward_hallucination_penalized():
    """Hallucinate response when gold says silent → silent_quality drags down."""
    grpo = _try_import_grpo()
    if grpo is None:
        return
    _calc_rewards_v12_trajectory = grpo._calc_rewards_v12_trajectory

    questions = [
        {"card_id": "c1", "gold_answer": "red",
         "answer_form": "literal", "ask_chunks": [3]},
    ]
    gold_action_per_chunk = {
        "0": "silent", "1": "silent", "2": "silent",
        "3": "response", "4": "silent", "5": "silent",
    }
    G = 2
    # Both rollouts answer correctly at chunk 3, but rollout 0 hallucinates
    # at chunk 1 (extra response when gold says silent).
    per_g_outputs = [
        # chunk 0
        ["<answer></answer>"] * G,
        # chunk 1: rollout 0 hallucinates, rollout 1 stays silent
        ["<answer>extra</answer>", "<answer></answer>"],
        # chunk 2
        ["<answer></answer>"] * G,
        # chunk 3: both answer correct
        ["<answer>red</answer>"] * G,
        # chunk 4-5: silent
        ["<answer></answer>"] * G,
        ["<answer></answer>"] * G,
    ]
    rollout_data, tk = _make_rollout_data(questions, gold_action_per_chunk, per_g_outputs)
    rewards, _, _ = _calc_rewards_v12_trajectory(
        rollout_data, group_size=G, tokenizer=tk,
    )
    # rollout_0 (hallucinator) must score LESS than rollout_1 (clean)
    assert rewards[0].item() < rewards[1].item(), (
        f"hallucinator rollout {rewards[0]:.3f} should be < clean rollout {rewards[1]:.3f}"
    )
    diff = rewards[1].item() - rewards[0].item()
    print(f"  PASS hallucination penalized: clean - hallucinator = {diff:.3f}")


def test_trajectory_reward_empty_questions_masked():
    """Trajectory with no questions → outcome mask=0 → no error, no signal."""
    grpo = _try_import_grpo()
    if grpo is None:
        return
    _calc_rewards_v12_trajectory = grpo._calc_rewards_v12_trajectory

    questions = []  # base-only trajectory
    gold_action_per_chunk = {"0": "silent", "1": "silent"}
    G = 1
    per_g_outputs = [
        ["<answer></answer>"], ["<answer></answer>"],
    ]
    rollout_data, tk = _make_rollout_data(questions, gold_action_per_chunk, per_g_outputs)
    rewards, rewards_dict, masks = _calc_rewards_v12_trajectory(
        rollout_data, group_size=G, tokenizer=tk,
    )
    # outcome mask should be 0 since no questions
    # masks tensor: [B, n_keys]; outcome is index 0 in V12_REWARD_DICT_KEYS
    from thinkstream.trainer.gdpo_advantage import V12_REWARD_DICT_KEYS
    outcome_idx = list(V12_REWARD_DICT_KEYS).index("outcome")
    assert masks[0, outcome_idx].item() == 0.0
    print("  PASS empty-questions: outcome masked")


def main():
    tests = [
        test_calc_rewards_v12_trajectory_function_exists,
        test_trajectory_reward_perfect_run,
        test_trajectory_reward_multi_question_partial,
        test_trajectory_reward_hallucination_penalized,
        test_trajectory_reward_empty_questions_masked,
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
