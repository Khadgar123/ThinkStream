"""Phase 4 DataProto integration test (real DataProto, not inlined helpers).

Covers what test_phase4_recurrent_advantage.py explicitly disclaimed:
  - real DataProto.repeat(n, interleave=True) (uid duplication semantics)
  - real DataProto __getitem__ + concat (used by pad_dataproto_to_divisor)
  - real swap+reindex flow (the tensor advanced indexing path)
  - real final_batch (uses indexing_proto + reorder via reverse_indices)
  - real pad_dataproto_to_divisor / unpad_dataproto round-trip
  - real compute_1D_grpo_advantage on a DataProto-backed tensor
  - sample_index broadcast on real LongTensor

Usage:
  This test requires the real verl + ray + tensordict + transformers stack.
  Locally on mac, set up a venv (Python ≥ 3.10) and:
    pip install ray tensordict omegaconf packaging hydra-core torch transformers numpy
  Then:
    python scripts/test_rl/test_phase4_dataproto_integration.py

If you just want the math without the dependency stack, run instead:
    python scripts/test_rl/test_phase4_recurrent_advantage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "verl"))

import numpy as np
import torch

try:
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.recurrent.utils import (
        compute_1D_grpo_advantage,
        final_batch,
        reverse_indices,
    )
except ModuleNotFoundError as e:
    print(f"⚠ skipping integration test: {e}")
    print("  Install: pip install ray tensordict omegaconf packaging hydra-core torch transformers numpy")
    raise SystemExit(0)


def _make_original_batch(n_traj: int, prompt_len: int, response_len: int):
    """Build a DataProto representing the un-expanded (B*n) batch.

    Has uid + ground_truth in non_tensor_batch, prompts in batch.
    """
    prompts = torch.zeros(n_traj, prompt_len, dtype=torch.long)
    attention_mask = torch.ones(n_traj, prompt_len, dtype=torch.long)
    uids = np.array([f"p{i // 2}" for i in range(n_traj)], dtype=object)  # n=2 rollouts
    ground_truth = np.array([{"answer": f"a{i}"} for i in range(n_traj)], dtype=object)
    return DataProto.from_single_dict(
        {
            "prompts": prompts,
            "attention_mask": attention_mask,
            "uid": uids,
            "ground_truth": ground_truth,
        }
    )


def _make_gen_batch_output(action_layout, prompt_len: int, response_len: int):
    """Build a DataProto representing the expanded (sum(K_i)) gen_batch_output.

    action_layout: list of (sample_index, is_final, traj_outcome_at_final).
    """
    n_actions = len(action_layout)
    sidx = torch.tensor([a[0] for a in action_layout], dtype=torch.long)
    fmask = torch.tensor([a[1] for a in action_layout], dtype=torch.bool)
    rm_scores = torch.zeros(n_actions, response_len, dtype=torch.float32)
    for i, (_, is_final, outcome) in enumerate(action_layout):
        if is_final:
            rm_scores[i, response_len - 1] = outcome
        else:
            # simulate small per-action format reward (the discarded signal)
            rm_scores[i, 0] = 0.05
    prompts = torch.zeros(n_actions, prompt_len, dtype=torch.long)
    responses = torch.zeros(n_actions, response_len, dtype=torch.long)
    attention_mask = torch.ones(n_actions, prompt_len + response_len, dtype=torch.long)
    response_mask = torch.ones(n_actions, response_len, dtype=torch.long)
    return DataProto.from_single_dict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rm_scores": rm_scores,
            "sample_index": sidx,
            "final_mask": fmask,
        }
    )


def test_real_dataproto_swap_and_reindex():
    """Exercise the swap+reindex path on real DataProto."""
    print("─── Test: real DataProto swap + reindex ───")
    n_traj, prompt_len, response_len = 4, 4, 8
    original_batch = _make_original_batch(n_traj, prompt_len, response_len)
    assert len(original_batch) == 4

    # 4 trajectories, K=[3,1,2,1] actions:
    layout = [
        (0, False, 0.0),
        (0, False, 0.0),
        (0, True, 0.8),   # traj 0 final
        (1, True, 0.2),   # traj 1 final
        (2, False, 0.0),
        (2, True, 1.0),   # traj 2 final
        (3, True, 0.5),   # traj 3 final
    ]
    gen = _make_gen_batch_output(layout, prompt_len, response_len)
    assert len(gen) == 7

    # Reindex non_tensor_batch (mirrors ray_trainer Phase 4 swap)
    sidx_np = gen.batch["sample_index"].cpu().numpy().astype(int)
    for k, v in original_batch.non_tensor_batch.items():
        if k not in gen.non_tensor_batch:
            gen.non_tensor_batch[k] = v[sidx_np]

    # Verify uid was reindexed correctly
    uids_after = gen.non_tensor_batch["uid"]
    expected_uids = np.array(["p0", "p0", "p0", "p0", "p1", "p1", "p1"], dtype=object)
    assert np.array_equal(uids_after, expected_uids), (
        f"uid reindex broken: got {uids_after}, expected {expected_uids}"
    )
    print(f"  ✓ uid reindex via sample_index: {list(uids_after)}")
    print(f"  ✓ ground_truth reindex carried through ({len(gen.non_tensor_batch['ground_truth'])} rows)")


def test_real_final_batch():
    """Exercise verl.recurrent.utils.final_batch on a real DataProto."""
    print()
    print("─── Test: real final_batch (verl.recurrent.utils) ───")
    n_traj, prompt_len, response_len = 4, 4, 8
    original_batch = _make_original_batch(n_traj, prompt_len, response_len)
    layout = [
        (0, False, 0.0),
        (0, False, 0.0),
        (0, True, 0.8),
        (1, True, 0.2),
        (2, False, 0.0),
        (2, True, 1.0),
        (3, True, 0.5),
    ]
    gen = _make_gen_batch_output(layout, prompt_len, response_len)
    sidx_np = gen.batch["sample_index"].cpu().numpy().astype(int)
    for k, v in original_batch.non_tensor_batch.items():
        if k not in gen.non_tensor_batch:
            gen.non_tensor_batch[k] = v[sidx_np]

    final = final_batch(gen, gen.batch["final_mask"], gen.batch["sample_index"])
    assert len(final) == n_traj
    # rm_scores in input order: traj 0=0.8, traj 1=0.2, traj 2=1.0, traj 3=0.5
    per_traj = final.batch["rm_scores"].sum(dim=-1)
    expected = torch.tensor([0.8, 0.2, 1.0, 0.5])
    assert torch.allclose(per_traj, expected, atol=1e-5), (
        f"final_batch rewards out of order: got {per_traj}, expected {expected}"
    )
    print(f"  ✓ final_batch returns {len(final)} rows in input order")
    print(f"  ✓ per-traj reward in input order: {per_traj.tolist()}")


def test_real_pad_unpad_roundtrip():
    """pad_dataproto_to_divisor + unpad_dataproto on the swapped batch."""
    print()
    print("─── Test: pad_dataproto_to_divisor + unpad_dataproto ───")
    n_traj, prompt_len, response_len = 4, 4, 8
    layout = [(0, False, 0), (0, True, 0.8), (1, True, 0.2),
              (2, True, 1.0), (3, True, 0.5)]  # 5 rows
    gen = _make_gen_batch_output(layout, prompt_len, response_len)
    assert len(gen) == 5

    world_size = 4  # simulate FSDP world size of 4
    padded, pad_size = pad_dataproto_to_divisor(gen, world_size)
    assert pad_size == 3, f"5 rows + 3 pad = 8 (next multiple of 4); got pad_size={pad_size}"
    assert len(padded) == 8

    # Verify replicated tail: padded rows are the first pad_size rows of original
    orig_rm0 = gen.batch["rm_scores"][0]
    pad_rm0 = padded.batch["rm_scores"][5]  # first pad row should equal gen[0]
    assert torch.equal(orig_rm0, pad_rm0), "pad replicates from head"
    print(f"  ✓ pad: 5 → 8 (pad_size=3), padded rows replicate from head")

    # Unpad back
    unpadded = unpad_dataproto(padded, pad_size)
    assert len(unpadded) == 5
    assert torch.equal(unpadded.batch["rm_scores"], gen.batch["rm_scores"])
    assert torch.equal(unpadded.batch["sample_index"], gen.batch["sample_index"])
    print(f"  ✓ unpad: 8 → 5, restores all tensor fields")
    print(f"  ✓ round-trip preserves rm_scores + sample_index")


def test_real_e2e_phase4d_flow():
    """End-to-end flow on real DataProto using the CURRENT Phase 4d
    "pad once, mask through" approach (no unpad).

    Verifies:
      - swap + reindex
      - reward_traj_tensor extracted BEFORE pad (length B*n)
      - pad with head-replication for batch + reward_tensor + reward_extra
      - response_mask zero'd for padded rows
      - 1D adv on UNPADDED trajectory rewards
      - sidx is PADDED → adv_per_action has padded shape
      - response_mask=0 multiplies padded rows to 0 in advantages
      - batch stays padded all the way through (would be fed to update_actor)
    """
    print()
    print("─── Test: end-to-end Phase 4d flow on real DataProto (stay-padded) ───")
    n_traj, prompt_len, response_len = 4, 4, 8
    original_batch = _make_original_batch(n_traj, prompt_len, response_len)
    layout = [
        (0, False, 0.0),
        (0, False, 0.0),
        (0, True, 0.8),
        (1, True, 0.2),
        (2, False, 0.0),
        (2, True, 1.0),
        (3, True, 0.5),
    ]
    gen = _make_gen_batch_output(layout, prompt_len, response_len)

    # Phase 4 swap
    sidx_np = gen.batch["sample_index"].cpu().numpy().astype(int)
    sidx_t_for_reindex = torch.from_numpy(sidx_np).long()
    for k, v in original_batch.non_tensor_batch.items():
        if k not in gen.non_tensor_batch:
            gen.non_tensor_batch[k] = v[sidx_np]
    for k, v in original_batch.batch.items():
        if k not in gen.batch:
            gen.batch[k] = v[sidx_t_for_reindex]
    batch = gen
    assert len(batch) == 7

    # Phase 4d: extract trajectory reward BEFORE pad (uses unpadded final_mask)
    fmask = batch.batch["final_mask"]
    sidx = batch.batch["sample_index"]
    rm = batch.batch["rm_scores"]
    final_idx = torch.where(fmask)[0]
    traj_idx_of_finals = sidx[final_idx].long()
    reorder = reverse_indices(traj_idx_of_finals)
    reward_traj_tensor = rm[final_idx][reorder]  # [B*n=4, R]
    assert reward_traj_tensor.shape == (n_traj, response_len)
    per_traj = reward_traj_tensor.sum(dim=-1)
    expected_per_traj = torch.tensor([0.8, 0.2, 1.0, 0.5])
    assert torch.allclose(per_traj, expected_per_traj)

    # Simulate per-action reward_tensor + reward_extra_infos_dict
    reward_tensor = batch.batch["rm_scores"].clone()  # [7, R]
    reward_extra_infos_dict = {
        "raw_outcome": np.array([0.0, 0.0, 0.8, 0.2, 0.0, 1.0, 0.5], dtype=object),
        "n_answered": np.array([0, 0, 3, 1, 0, 2, 1], dtype=object),
    }

    # FSDP pad (mirrors ray_trainer.py)
    world_size = 4
    batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
    assert len(batch) == 8 and pad_size == 1

    # Pad reward_tensor + reward_extra with HEAD replication (matches DataProto)
    reward_tensor = torch.cat([reward_tensor, reward_tensor[:pad_size]], dim=0)
    assert reward_tensor.shape == (8, response_len)
    for k, v in list(reward_extra_infos_dict.items()):
        reward_extra_infos_dict[k] = np.concatenate([v, v[:pad_size]], axis=0)
        assert len(reward_extra_infos_dict[k]) == 8, f"{k} len mismatch after pad"

    # Zero out response_mask for padded rows
    new_resp_mask = batch.batch["response_mask"].clone()
    new_resp_mask[-pad_size:] = 0
    batch.batch["response_mask"] = new_resp_mask
    print(f"  ✓ swap → 7; pad → 8 (pad_size=1); reward_tensor + reward_extra padded; response_mask[-1]=0")

    # Verify that writing reward_extra back to non_tensor_batch doesn't blow up
    for k, v in reward_extra_infos_dict.items():
        batch.non_tensor_batch[k] = v
    assert len(batch.non_tensor_batch["raw_outcome"]) == len(batch), (
        "non_tensor_batch must align with batch length after pad"
    )
    print(f"  ✓ reward_extra writes back into padded batch without length mismatch")

    # 1D advantage on UNPADDED trajectory rewards (uses original_batch.uid B*n=4)
    adv_scalar = compute_1D_grpo_advantage(
        token_level_rewards=reward_traj_tensor,
        index=original_batch.non_tensor_batch["uid"],
        use_adv=True,
    )
    expected_adv = torch.tensor([0.7071, -0.7071, 0.7071, -0.7071])
    assert torch.allclose(adv_scalar, expected_adv, atol=1e-3), (
        f"adv mismatch: got {adv_scalar}, expected {expected_adv}"
    )
    print(f"  ✓ 1D adv on UNPADDED reward_traj_tensor: {adv_scalar.tolist()}")

    # Broadcast via PADDED sample_index
    sidx_padded = batch.batch["sample_index"].long()  # length 8
    adv_per_action = adv_scalar[sidx_padded]
    response_mask = batch.batch["response_mask"]
    advantages = adv_per_action.unsqueeze(-1).tile([1, response_len]) * response_mask
    assert advantages.shape == (8, response_len)

    # Padded row (index 7 = last) must have advantages = 0 (response_mask=0)
    assert torch.equal(advantages[7], torch.zeros(response_len)), (
        "padded row's advantages must be 0 (response_mask=0)"
    )
    # Real rows 0..6 must have non-zero advantages where response_mask=1
    for i in range(7):
        if response_mask[i, 0] > 0:
            assert advantages[i, 0] != 0 or adv_scalar[sidx_padded[i]] == 0
    print(f"  ✓ padded row's advantages = 0 via response_mask=0 (loss-safe)")
    print(f"  ✓ batch stays padded (len={len(batch)}) — would feed update_actor cleanly")


def test_dataproto_repeat_uid_semantics():
    """DataProto.repeat(n, interleave=True) duplicates uids consecutively.

    This is the key invariant Phase 4d relies on: after batch.repeat(n),
    n rollouts of one prompt share the SAME uid → 1D GRPO grouping by
    uid yields per-prompt groups (n rollouts per group).
    """
    print()
    print("─── Test: DataProto.repeat(n) uid semantics ───")
    n_prompts, prompt_len = 3, 4
    prompts = torch.zeros(n_prompts, prompt_len, dtype=torch.long)
    attention_mask = torch.ones(n_prompts, prompt_len, dtype=torch.long)
    uids = np.array(["p0", "p1", "p2"], dtype=object)
    batch = DataProto.from_single_dict(
        {"prompts": prompts, "attention_mask": attention_mask, "uid": uids}
    )
    n = 2
    repeated = batch.repeat(repeat_times=n, interleave=True)
    assert len(repeated) == 6
    expected = np.array(["p0", "p0", "p1", "p1", "p2", "p2"], dtype=object)
    got = repeated.non_tensor_batch["uid"]
    assert np.array_equal(got, expected), f"uid repeat broken: {got=} vs {expected=}"
    print(f"  ✓ repeat(n=2, interleave=True) on uids ['p0','p1','p2'] → {list(got)}")
    print(f"  ✓ n rollouts of one prompt share uid → 1D GRPO groups them per-prompt")


def main() -> int:
    print("═══ Phase 4 DataProto integration test (real verl stack) ═══")
    print()
    test_real_dataproto_swap_and_reindex()
    test_real_final_batch()
    test_real_pad_unpad_roundtrip()
    test_real_e2e_phase4d_flow()
    test_dataproto_repeat_uid_semantics()
    print()
    print("✓ ALL DATAPROTO INTEGRATION TESTS PASS")
    print()
    print("Coverage upgrade vs test_phase4_recurrent_advantage.py:")
    print("  - real DataProto.repeat / __getitem__ / concat")
    print("  - real swap+reindex against actual non_tensor_batch numpy arrays")
    print("  - real verl.recurrent.utils.final_batch (not inlined)")
    print("  - real pad_dataproto_to_divisor + unpad_dataproto round-trip")
    print()
    print("Still NOT covered (would need a multi-GPU ray cluster):")
    print("  - actor/ref worker FSDP dispatch on padded batch")
    print("  - actor loss with broadcasted advantages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
