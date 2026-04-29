"""v12.2 chunk-level rollout tests.

Validates:
- compute_1d_grpo_advantage matches ReMemR1 ray_trainer.py:249-288 numerics
  (singleton groups, multi-element, use_adv=True/False, 2D summation)
- compute_mixed_advantage_v12 alpha-mixing of outcome + state advantages
- ChunkLevelRolloutLoop control flow (active mask, termination, state evolution)

Run: python tests/test_v12_rollout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from thinkstream.trainer.v12_rollout import (
    ChunkLevelRolloutConfig,
    ChunkLevelRolloutLoop,
    VideoTrajectoryState,
    chunk_results_from_loop_result,
    compute_1d_grpo_advantage,
    compute_mixed_advantage_v12,
    default_v12_update_state,
    replicate_state_for_group,
)


# =============================================================================
# compute_1d_grpo_advantage — ReMemR1 line-by-line correctness
# =============================================================================


def test_1d_grpo_singleton_use_adv_true():
    """ReMemR1 lines 273-276: singleton group → mean=0, std=1.
    Advantage = (score - 0) / (1 + ε) ≈ score (full raw signal preserved)."""
    rewards = torch.tensor([2.5])
    adv = compute_1d_grpo_advantage(rewards, ["only_group"], use_adv=True)
    # (2.5 - 0) / (1.0 + 1e-6) ≈ 2.5 (within 1e-5)
    assert abs(adv[0].item() - 2.5) < 1e-4, (
        f"singleton group should preserve raw score, got {adv[0].item()}"
    )


def test_1d_grpo_singleton_use_adv_false():
    """Singleton group with use_adv=False: adv = score - 0 = score."""
    rewards = torch.tensor([3.7])
    adv = compute_1d_grpo_advantage(rewards, ["g"], use_adv=False)
    assert abs(adv[0].item() - 3.7) < 1e-6


def test_1d_grpo_multi_use_adv_false():
    """Multi-element group with use_adv=False: subtract group mean only."""
    rewards = torch.tensor([1.0, 3.0, 5.0])
    adv = compute_1d_grpo_advantage(rewards, ["g", "g", "g"], use_adv=False)
    # mean = 3.0, adv = [-2, 0, 2]
    assert torch.allclose(adv, torch.tensor([-2.0, 0.0, 2.0]), atol=1e-6)


def test_1d_grpo_multi_use_adv_true_bessel():
    """Multi-element group with use_adv=True uses unbiased=True (Bessel).
    rewards=[1,3,5]: mean=3, std (unbiased, n-1=2) = sqrt((4+0+4)/2) = 2.0
    adv = (r - 3) / (2 + ε) → [-1, 0, +1]."""
    rewards = torch.tensor([1.0, 3.0, 5.0])
    adv = compute_1d_grpo_advantage(rewards, ["g", "g", "g"], use_adv=True)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(adv, expected, atol=1e-3), (
        f"Bessel std normalization failed; got {adv}, expected {expected}"
    )


def test_1d_grpo_multi_groups():
    """Multiple distinct groups normalize independently."""
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0])
    groups = ["A", "A", "B", "B"]
    adv = compute_1d_grpo_advantage(rewards, groups, use_adv=False)
    # A: mean=2, [1-2, 3-2] = [-1, 1]
    # B: mean=15, [10-15, 20-15] = [-5, 5]
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0, -5.0, 5.0]))


def test_1d_grpo_2d_input_summed():
    """2D [N,T] input: sum over last dim before normalizing.
    Per ReMemR1 line 262: scores = token_level_rewards.sum(dim=-1)."""
    # rewards [3, 2] — each rollout has 2 token-level rewards
    rewards = torch.tensor([
        [0.5, 0.5],   # sum = 1.0
        [1.5, 1.5],   # sum = 3.0
        [2.5, 2.5],   # sum = 5.0
    ])
    adv = compute_1d_grpo_advantage(rewards, ["g", "g", "g"], use_adv=False)
    assert torch.allclose(adv, torch.tensor([-2.0, 0.0, 2.0]))


def test_1d_grpo_mixed_singleton_and_multi():
    """Singleton groups coexist with multi-element groups in the same call."""
    rewards = torch.tensor([7.0, 1.0, 3.0])  # singleton, then 2-elem group
    groups = ["solo", "pair", "pair"]
    adv = compute_1d_grpo_advantage(rewards, groups, use_adv=False)
    # solo: mean=0 → adv=7.0
    # pair: mean=2 → [-1, 1]
    assert torch.allclose(adv, torch.tensor([7.0, -1.0, 1.0]))


def test_1d_grpo_length_mismatch_raises():
    rewards = torch.tensor([1.0, 2.0])
    try:
        compute_1d_grpo_advantage(rewards, ["only_one_group"], use_adv=False)
    except ValueError as e:
        assert "length" in str(e).lower()
        return
    raise AssertionError("expected ValueError for mismatched index length")


# =============================================================================
# compute_mixed_advantage_v12 — alpha-mixed outcome + state
# =============================================================================


def test_mixed_advantage_alpha_one_pure_outcome():
    """alpha=1.0 → pure outcome advantage (state ignored)."""
    outcome = torch.tensor([0.0, 1.0, 2.0, 3.0])
    state = torch.tensor([100.0, 100.0, 100.0, 100.0])  # would dominate if mixed
    video_uids = ["A", "A", "B", "B"]
    chunks = [0, 1, 0, 1]
    mixed = compute_mixed_advantage_v12(
        outcome, state, video_uids, chunks, alpha=1.0, use_adv=False,
    )
    # outcome adv: A → [-0.5, 0.5], B → [-0.5, 0.5]
    expected = torch.tensor([-0.5, 0.5, -0.5, 0.5])
    assert torch.allclose(mixed, expected, atol=1e-5)


def test_mixed_advantage_alpha_zero_pure_state():
    """alpha=0.0 → pure state advantage. Each (uid, chunk) is its own group;
    here all groups are singletons → adv = raw state score (mean=0 baseline)."""
    outcome = torch.tensor([100.0, 100.0])
    state = torch.tensor([0.5, 0.7])
    video_uids = ["X", "Y"]
    chunks = [0, 0]
    mixed = compute_mixed_advantage_v12(
        outcome, state, video_uids, chunks, alpha=0.0, use_adv=False,
    )
    assert torch.allclose(mixed, torch.tensor([0.5, 0.7]))


def test_mixed_advantage_alpha_blend():
    """alpha=0.5: average of outcome and state advantages."""
    outcome = torch.tensor([1.0, 3.0])    # group A: mean 2 → adv [-1, 1]
    state = torch.tensor([10.0, 20.0])    # singletons (uid+chunk unique) → [10, 20]
    video_uids = ["A", "A"]
    chunks = [0, 1]
    mixed = compute_mixed_advantage_v12(
        outcome, state, video_uids, chunks, alpha=0.5, use_adv=False,
    )
    # 0.5 * [-1, 1] + 0.5 * [10, 20] = [4.5, 10.5]
    assert torch.allclose(mixed, torch.tensor([4.5, 10.5]))


def test_mixed_advantage_shape_mismatch_raises():
    try:
        compute_mixed_advantage_v12(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0, 3.0]),  # mismatched
            ["A", "A"], [0, 1],
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


# =============================================================================
# ChunkLevelRolloutLoop — control flow
# =============================================================================


def _make_state(uid="vid_0"):
    return VideoTrajectoryState(
        video_uid=uid,
        chunk_idx=0,
        pending_queries=[{"question": "what color?", "ask_chunk": 0}],
    )


def test_replicate_state_independence():
    """G replicas must be deep-copied — mutating one must not affect others."""
    s = _make_state()
    replicas = replicate_state_for_group(s, group_size=4)
    assert len(replicas) == 4
    replicas[0].chunk_idx = 99
    replicas[0].n_recall_calls = 5
    assert replicas[1].chunk_idx == 0
    assert replicas[1].n_recall_calls == 0
    # Original untouched
    assert s.chunk_idx == 0


def test_default_v12_update_state_silent_answer():
    """Empty answer = silent → state stays active, chunk_idx advances."""
    s = _make_state()
    out = default_v12_update_state(
        s, "<think>nothing yet</think><answer></answer>", chunk_idx=0,
    )
    assert out.is_active is True
    assert out.is_done is False
    assert out.chunk_idx == 1
    assert out.final_answer is None


def test_default_v12_update_state_final_answer():
    """Non-empty answer → mark inactive + record."""
    s = _make_state()
    out = default_v12_update_state(
        s, "<think>I see red</think><answer>red</answer>", chunk_idx=2,
    )
    assert out.is_active is False
    assert out.is_done is True
    assert out.final_answer == "red"
    assert out.final_answer_chunk == 2
    # The pending query should now be in answered_queries
    assert len(out.answered_queries) == 1
    assert out.answered_queries[0]["answer"] == "red"
    assert len(out.pending_queries) == 0


def test_default_v12_update_state_recall_call():
    """recall tool_call increments counter, state stays active."""
    s = _make_state()
    response = (
        '<think>need more context</think>'
        '<tool_call>{"name": "recall", "arguments": {"query": "earlier scene"}}</tool_call>'
    )
    out = default_v12_update_state(s, response, chunk_idx=1)
    assert out.is_active is True
    assert out.n_recall_calls == 1
    assert out.chunk_idx == 2


def test_default_v12_update_state_compress_call():
    """compress tool_call records summary + drops thinks in range."""
    s = _make_state()
    s.recent_thinks = [
        {"chunk": 1, "text": "ann"},
        {"chunk": 5, "text": "bob"},
        {"chunk": 10, "text": "dave"},
    ]
    response = (
        '<think>old context</think>'
        '<tool_call>{"name": "compress", "arguments": '
        '{"time_range": [0, 6], "text": "early scene summary"}}</tool_call>'
    )
    out = default_v12_update_state(s, response, chunk_idx=7)
    assert out.n_compress_calls == 1
    assert len(out.compressed_summaries) == 1
    summary = out.compressed_summaries[0]
    assert summary["time_range"] == [0, 6]
    assert summary["text"] == "early scene summary"
    # Chunk-1 and chunk-5 should be dropped (in [0,6]); chunk-10 retained
    remaining_chunks = [t.get("chunk") for t in out.recent_thinks]
    assert 1 not in remaining_chunks
    assert 5 not in remaining_chunks
    assert 10 in remaining_chunks


def test_chunklevel_rollout_loop_mock_terminates_on_answer():
    """Verify rollout_one_video terminates when all G rollouts emit an answer."""
    cfg = ChunkLevelRolloutConfig(
        group_size=2, max_chunks_per_video=10,
    )

    # Mock generator: emits silent on chunk 0, final answer on chunk 1
    call_counter = {"n": 0}

    def mock_generate(messages_batch, n=1):
        # messages_batch is List[List[Dict]] — one per active rollout
        call_counter["n"] += 1
        # Round 1 (call_counter==1): both silent
        # Round 2 (call_counter==2): both emit answer
        if call_counter["n"] == 1:
            return ["<think>watching</think><answer></answer>"
                    for _ in messages_batch]
        return ["<think>got it</think><answer>red</answer>"
                for _ in messages_batch]

    def mock_build_messages(state, video_meta):
        return [{"role": "user", "content": f"chunk {video_meta['chunk_idx']}"}]

    loop = ChunkLevelRolloutLoop(
        cfg, mock_generate, mock_build_messages, default_v12_update_state,
    )
    initial = _make_state()
    result = loop.rollout_one_video(
        video_meta={"video_uid": "vid_0"}, initial_state=initial,
    )

    assert result["video_uid"] == "vid_0"
    assert len(result["states_per_rollout"]) == 2
    # Both rollouts should have emitted final answer
    for s in result["states_per_rollout"]:
        assert s.is_done is True
        assert s.final_answer == "red"
        assert s.final_answer_chunk == 1
    # responses_per_chunk should have exactly 2 entries (chunk 0 + chunk 1)
    assert len(result["responses_per_chunk"]) == 2


def test_chunklevel_rollout_loop_active_mask_skips_inactive():
    """When one rollout finishes early, only the active one continues generating."""
    cfg = ChunkLevelRolloutConfig(
        group_size=2, max_chunks_per_video=5,
    )

    # Rollout 0 finishes at chunk 0 (answers "red"); rollout 1 stays silent
    call_log = []

    def mock_generate(messages_batch, n=1):
        call_log.append(len(messages_batch))
        if call_log[-1] == 2:
            # First chunk: rollout 0 answers, rollout 1 silent
            return [
                "<think>done</think><answer>red</answer>",
                "<think>still watching</think><answer></answer>",
            ]
        # Subsequent calls: only 1 active rollout (rollout 1)
        return ["<think>still watching</think><answer></answer>"]

    def mock_build_messages(state, video_meta):
        return [{"role": "user", "content": "x"}]

    loop = ChunkLevelRolloutLoop(
        cfg, mock_generate, mock_build_messages, default_v12_update_state,
    )
    result = loop.rollout_one_video(
        video_meta={"video_uid": "vid_x"}, initial_state=_make_state(),
    )

    # Round 1: 2 active rollouts; rounds 2..N: only 1 active rollout
    assert call_log[0] == 2
    for n_active in call_log[1:]:
        assert n_active == 1, (
            f"after rollout 0 finishes, only 1 should be active, got {n_active}"
        )
    # Rollout 0 finished early
    states = result["states_per_rollout"]
    assert states[0].is_done is True
    assert states[0].final_answer == "red"
    assert states[0].final_answer_chunk == 0
    # Rollout 1 hit max_chunks without answering
    assert states[1].is_done is False
    assert states[1].chunk_idx == 5  # advanced through all 5 chunks


def test_chunk_results_adapter_shape():
    """Adapter converts loop result → chunk_results list compatible with
    `_calc_rewards_v12` (which decodes via tokenizer.decode and parses)."""

    class _StubTokenizer:
        """Encodes by splitting on whitespace, returns int ids deterministically.
        Just needs `.encode(text, add_special_tokens=False) -> List[int]`."""
        def encode(self, text, add_special_tokens=False):
            # Map each char to its ord — trivially deterministic for tests.
            return [ord(c) for c in text]

    fake_loop_result = {
        "video_uid": "vid_x",
        "responses_per_chunk": [
            ["abc", "de"],     # chunk 0: 2 rollouts
            ["", "f"],         # chunk 1: rollout 0 inactive (empty)
        ],
    }
    chunk_results = chunk_results_from_loop_result(
        fake_loop_result, _StubTokenizer(),
    )
    assert len(chunk_results) == 2
    assert chunk_results[0]["chunk_idx"] == 0
    assert chunk_results[1]["chunk_idx"] == 1
    # Tokens: "abc" → [97,98,99], "de" → [100,101]
    assert chunk_results[0]["generated_tokens"][0] == [97, 98, 99]
    assert chunk_results[0]["generated_tokens"][1] == [100, 101]
    # Empty rollout → empty token list (not None — caller's len() checks work)
    assert chunk_results[1]["generated_tokens"][0] == []
    assert chunk_results[1]["generated_tokens"][1] == [102]


def test_grpo_reexports_v12_rollout_helpers():
    """Smoke test: grpo.py re-exports v12_rollout helpers without import cycles.

    Skipped on envs where transformers/tokenizers version mismatch prevents
    grpo.py from loading at all — that's an env issue, not a v12.2 bug. We
    AST-verify the re-export block to catch typos without needing the heavy
    stack.
    """
    import ast
    grpo_path = Path(__file__).resolve().parents[1] / "thinkstream" / "trainer" / "grpo.py"
    tree = ast.parse(grpo_path.read_text())

    # Find the `from thinkstream.trainer.v12_rollout import (...)` block.
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "thinkstream.trainer.v12_rollout":
            found = [(alias.name, alias.asname) for alias in node.names]
            break
    assert found, "grpo.py is missing the v12_rollout re-export block"

    expected_names = {
        "ChunkLevelRolloutConfig", "ChunkLevelRolloutLoop",
        "VideoTrajectoryState", "compute_1d_grpo_advantage",
        "compute_mixed_advantage_v12", "chunk_results_from_loop_result",
        "default_v12_update_state",
    }
    actual_names = {orig for orig, _ in found}
    missing = expected_names - actual_names
    assert not missing, f"grpo.py re-export missing: {missing}"

    # Soft-test: try the actual import. If transformers env is healthy, this
    # confirms no circular-import issue. If not, AST already verified the
    # static structure.
    try:
        from thinkstream.trainer import grpo as _grpo
        assert hasattr(_grpo, "_V12ChunkLevelRolloutLoop")
        assert hasattr(_grpo, "_compute_1d_grpo_advantage_remem")
        assert hasattr(_grpo, "_compute_mixed_advantage_v12_remem")
        assert hasattr(_grpo, "_chunk_results_from_loop_result")
    except ImportError as e:
        if "tokenizers" in str(e) or "transformers" in str(e):
            print(f"  (skipped runtime import: env tokenizer/transformers mismatch — {e})")
        else:
            raise


def test_chunklevel_rollout_loop_max_chunks_terminates():
    """All rollouts silent → loop must terminate at max_chunks_per_video."""
    cfg = ChunkLevelRolloutConfig(group_size=3, max_chunks_per_video=4)

    def mock_silent_generate(messages_batch, n=1):
        return ["<think>nope</think><answer></answer>" for _ in messages_batch]

    def mock_build_messages(state, video_meta):
        return [{"role": "user", "content": "x"}]

    loop = ChunkLevelRolloutLoop(
        cfg, mock_silent_generate, mock_build_messages, default_v12_update_state,
    )
    result = loop.rollout_one_video(
        video_meta={"video_uid": "v"}, initial_state=_make_state(),
    )
    assert len(result["responses_per_chunk"]) == 4
    for s in result["states_per_rollout"]:
        assert s.is_active is True   # never inactivated
        assert s.is_done is False
        assert s.chunk_idx == 4


# =============================================================================
# Test runner
# =============================================================================


def main():
    tests = [
        test_1d_grpo_singleton_use_adv_true,
        test_1d_grpo_singleton_use_adv_false,
        test_1d_grpo_multi_use_adv_false,
        test_1d_grpo_multi_use_adv_true_bessel,
        test_1d_grpo_multi_groups,
        test_1d_grpo_2d_input_summed,
        test_1d_grpo_mixed_singleton_and_multi,
        test_1d_grpo_length_mismatch_raises,
        test_mixed_advantage_alpha_one_pure_outcome,
        test_mixed_advantage_alpha_zero_pure_state,
        test_mixed_advantage_alpha_blend,
        test_mixed_advantage_shape_mismatch_raises,
        test_replicate_state_independence,
        test_default_v12_update_state_silent_answer,
        test_default_v12_update_state_final_answer,
        test_default_v12_update_state_recall_call,
        test_default_v12_update_state_compress_call,
        test_chunklevel_rollout_loop_mock_terminates_on_answer,
        test_chunklevel_rollout_loop_active_mask_skips_inactive,
        test_chunk_results_adapter_shape,
        test_grpo_reexports_v12_rollout_helpers,
        test_chunklevel_rollout_loop_max_chunks_terminates,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
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
