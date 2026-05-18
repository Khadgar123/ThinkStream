from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "verl"))

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.ray_trainer import (  # noqa: E402
    _attach_action_reward_extras,
    _attach_recurrent_original_fields,
    _compute_recurrent_gdpo_advantages,
    _is_recurrent_rollout_batch,
    _make_action_token_scores,
    _recurrent_actor_update_divisor,
    _zero_padded_response_rows,
)


def _gen_batch() -> DataProto:
    prompt_len = 2
    response_len = 4
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
    ])
    return DataProto.from_single_dict({
        "prompts": torch.zeros(3, prompt_len, dtype=torch.long),
        "responses": torch.zeros(3, response_len, dtype=torch.long),
        "attention_mask": attention_mask,
        "response_mask": attention_mask[:, prompt_len:].clone(),
        "sample_index": torch.tensor([0, 0, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, True], dtype=torch.bool),
    })


def test_recurrent_trainer_helpers_broadcast_final_rewards():
    gen = _gen_batch()
    original = DataProto.from_single_dict({
        "prompts": torch.zeros(2, 2, dtype=torch.long),
        "uid": np.array(["p0", "p1"], dtype=object),
        "reward_model": np.array([{"answer": "a"}, {"answer": "b"}], dtype=object),
    })

    assert _is_recurrent_rollout_batch(gen)
    _attach_recurrent_original_fields(gen, original)
    assert list(gen.non_tensor_batch["uid"]) == ["p0", "p0", "p1"]

    reward_infos = _attach_action_reward_extras(
        gen,
        {"answer_score": np.array([0.8, 0.2], dtype=object)},
    )
    assert list(reward_infos["answer_score"]) == [0.8, 0.8, 0.2]
    assert list(gen.non_tensor_batch["answer_score"]) == [0.8, 0.8, 0.2]

    trajectory_rewards = torch.tensor([
        [0.0, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.2, 0.0],
    ])
    token_scores = _make_action_token_scores(gen, trajectory_rewards)
    expected = torch.tensor([
        [0.0, 0.0, 0.8, 0.0],
        [0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.0],
    ])
    assert torch.allclose(token_scores, expected)

    _zero_padded_response_rows(gen, pad_size=1)
    assert torch.equal(gen.batch["response_mask"][-1], torch.zeros(4, dtype=torch.long))


def test_recurrent_actor_update_divisor_matches_local_minibatch():
    class AttrDict(dict):
        def __getattr__(self, item):
            return self[item]

    config = AttrDict({
        "actor_rollout_ref": AttrDict({
            "actor": AttrDict({"ppo_mini_batch_size": 2}),
            "rollout": AttrDict({"n": 8}),
        })
    })

    divisor = _recurrent_actor_update_divisor(config, world_size=8)
    assert divisor == 16

    expanded_rows = 1816
    pad_size = (divisor - expanded_rows % divisor) % divisor
    local_rows = (expanded_rows + pad_size) // 8
    local_mini_batch = (2 * 8) // 8
    assert local_rows % local_mini_batch == 0


def test_recurrent_gdpo_advantage_keeps_compress_on_compress_rows():
    prompt_len = 2
    response_len = 3
    response_mask = torch.ones(4, response_len, dtype=torch.long)
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": response_mask,
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
        "ts_action_index": np.array([0, 1, 0, 1], dtype=object),
        "ts_action_event_chunk_idx": np.array([0, 8, 0, 8], dtype=object),
        "ts_chunk_turn_kinds": np.array([
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
        ], dtype=object),
        "ts_chunk_asst_texts": np.array([
            ["<think>x</think></Silence>", "<think>m</think>\n<m t=\"0-2\">ok</m>"],
            ["<think>x</think></Silence>", "<think>m</think>\n<m t=\"0-2\">ok</m>"],
            ["<think>x</think></Silence>", "<think>m</think>\n<m t=\"5-6\">old</m>"],
            ["<think>x</think></Silence>", "<think>m</think>\n<m t=\"5-6\">old</m>"],
        ], dtype=object),
        "ts_chunk_action_space_errors": np.array([["", ""], ["", ""], ["", ""], ["", ""]], dtype=object),
        "ts_chunk_hit_max_tokens": np.array([[False, False], [False, False], [False, False], [False, False]], dtype=object),
        "ts_compress_expected_chunks": np.array([
            [[], [0, 1, 2]],
            [[], [0, 1, 2]],
            [[], [0, 1, 2]],
            [[], [0, 1, 2]],
        ], dtype=object),
        "ts_compress_source_texts": np.array([
            [[], '<NEW_CAPTIONS><c t="0">ok</c><c t="1">ok</c><c t="2">ok</c></NEW_CAPTIONS>'],
            [[], '<NEW_CAPTIONS><c t="0">ok</c><c t="1">ok</c><c t="2">ok</c></NEW_CAPTIONS>'],
            [[], '<NEW_CAPTIONS><c t="0">ok</c><c t="1">ok</c><c t="2">ok</c></NEW_CAPTIONS>'],
            [[], '<NEW_CAPTIONS><c t="0">ok</c><c t="1">ok</c><c t="2">ok</c></NEW_CAPTIONS>'],
        ], dtype=object),
    })
    reward_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    reward_extras = {
        "outcome": np.array([1.0, 0.0], dtype=object),
        "answer_decision": np.array([1.0, 0.0], dtype=object),
        "format": np.array([1.0, 1.0], dtype=object),
        "outcome_gate": np.array([1.0, 0.0], dtype=object),
    }

    old_env = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,compress_quality=1"
    )
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "question"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_env is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_env
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode

    # The two streaming rows receive no compress branch signal; only the two
    # compress rows compare [cover=+1, old_only=-1] at the same boundary.
    row_adv = adv[:, 0]
    assert torch.allclose(row_adv[[0, 2]], torch.zeros(2), atol=1e-6)
    assert row_adv[1] > 0
    assert row_adv[3] < 0
    assert metrics["recurrent/gdpo/compress_rows"] == 2.0
    assert metrics["recurrent/gdpo/compress_source_grounding_mean"] > 0.0


def test_recurrent_gdpo_maps_answer_segment_advantage_to_action_rows():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(4, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
        "ts_action_index": np.array([0, 1, 0, 1], dtype=object),
        "ts_action_event_chunk_idx": np.array([1, 9, 1, 9], dtype=object),
        "ts_chunk_turn_kinds": np.array([
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
        ], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
        "segment_starts_json": np.array([
            json.dumps([0, 6]),
            json.dumps([0, 6]),
        ], dtype=object),
        "segment_ends_json": np.array([
            json.dumps([5, 10]),
            json.dumps([5, 10]),
        ], dtype=object),
        "segment_scores_json": np.array([
            json.dumps([0.0, 1.0]),
            json.dumps([1.0, 0.0]),
        ], dtype=object),
        "segment_answer_scores_json": np.array([
            json.dumps([0.0, 1.0]),
            json.dumps([1.0, 0.0]),
        ], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_alpha = os.environ.get("THINKSTREAM_SEGMENT_GLOBAL_ALPHA")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    old_rho = os.environ.get("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,compress_quality=0"
    )
    os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = "0"
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "question"
    os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = "0"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_alpha is None:
            os.environ.pop("THINKSTREAM_SEGMENT_GLOBAL_ALPHA", None)
        else:
            os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = old_alpha
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode
        if old_rho is None:
            os.environ.pop("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO", None)
        else:
            os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = old_rho

    row_adv = adv[:, 0]
    assert row_adv[0] < 0
    assert row_adv[1] > 0
    assert row_adv[2] > 0
    assert row_adv[3] < 0
    assert metrics["recurrent/gdpo/segment_answer_count"] == 4.0
    assert metrics["recurrent/gdpo/segment_answer_rows"] == 2.0
    assert metrics["recurrent/gdpo/segment_compress_local_rows"] == 2.0
    assert metrics["recurrent/gdpo/segment_global_alpha"] == 0.0


def test_recurrent_gdpo_question_mode_keeps_compress_score_off_answer_rows():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(4, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
        "ts_action_index": np.array([0, 1, 0, 1], dtype=object),
        "ts_action_event_chunk_idx": np.array([1, 1, 1, 1], dtype=object),
        "ts_chunk_turn_kinds": np.array([
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
        ], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
        "segment_starts_json": np.array([
            json.dumps([0]),
            json.dumps([0]),
        ], dtype=object),
        "segment_ends_json": np.array([
            json.dumps([5]),
            json.dumps([5]),
        ], dtype=object),
        # Full segment score differs only because of compression.
        "segment_scores_json": np.array([
            json.dumps([1.0]),
            json.dumps([0.0]),
        ], dtype=object),
        # Answer-only score is tied, so non-compress rows must get no local signal.
        "segment_answer_scores_json": np.array([
            json.dumps([0.0]),
            json.dumps([0.0]),
        ], dtype=object),
        "segment_compress_quality_json": np.array([
            json.dumps([1.0]),
            json.dumps([0.0]),
        ], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_alpha = os.environ.get("THINKSTREAM_SEGMENT_GLOBAL_ALPHA")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    old_rho = os.environ.get("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,compress_quality=0"
    )
    os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = "0"
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "question"
    os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = "0.1"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_alpha is None:
            os.environ.pop("THINKSTREAM_SEGMENT_GLOBAL_ALPHA", None)
        else:
            os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = old_alpha
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode
        if old_rho is None:
            os.environ.pop("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO", None)
        else:
            os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = old_rho

    row_adv = adv[:, 0]
    assert torch.allclose(row_adv[[0, 2]], torch.zeros(2), atol=1e-6)
    assert row_adv[1] > 0
    assert row_adv[3] < 0
    assert metrics["recurrent/gdpo/segment_answer_rows"] == 2.0
    assert metrics["recurrent/gdpo/segment_compress_local_rows"] == 2.0
    assert metrics["recurrent/gdpo/segment_compress_local_quality_rho"] == 0.1


def test_recurrent_gdpo_separates_global_grpo_and_global_gdpo_modes():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(2, prompt_len, dtype=torch.long),
        "responses": torch.zeros(2, response_len, dtype=torch.long),
        "attention_mask": torch.ones(2, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(2, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 1], dtype=torch.long),
        "final_mask": torch.tensor([True, True], dtype=torch.bool),
        "uid": np.array(["same", "same"], dtype=object),
    })
    reward_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=1,answer_decision=1,format=1,compress_quality=0"
    )
    try:
        os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "global_grpo"
        grpo_adv, grpo_metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
        os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "global_gdpo"
        gdpo_adv, gdpo_metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode

    assert grpo_adv[0, 0] > 0
    assert grpo_adv[1, 0] < 0
    assert torch.allclose(gdpo_adv[:, 0], torch.zeros(2), atol=1e-6)
    assert grpo_metrics["recurrent/gdpo/credit_mode/global_grpo"] == 1.0
    assert gdpo_metrics["recurrent/gdpo/credit_mode/global_gdpo"] == 1.0


def test_recurrent_gdpo_global_mode_uses_recall_answer_component():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(2, prompt_len, dtype=torch.long),
        "responses": torch.zeros(2, response_len, dtype=torch.long),
        "attention_mask": torch.ones(2, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(2, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 1], dtype=torch.long),
        "final_mask": torch.tensor([True, True], dtype=torch.bool),
        "uid": np.array(["same", "same"], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "recall_answer": np.array([1.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,recall_answer=1,compress_quality=0"
    )
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "global_gdpo"
    try:
        gdpo_adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode

    assert gdpo_adv[0, 0] > 0
    assert gdpo_adv[1, 0] < 0
    assert metrics["recurrent/gdpo/recall_answer_weight"] == 1.0
    assert metrics["recurrent/gdpo/recall_answer_adv_std"] > 0.0


def test_recurrent_gdpo_global_gspo_broadcasts_like_grpo_for_gspo_loss():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(4, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
    })
    reward_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
    }

    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "global_gspo"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode

    row_adv = adv[:, 0]
    assert row_adv[0] > 0
    assert row_adv[1] > 0
    assert row_adv[2] < 0
    assert row_adv[3] < 0
    assert metrics["recurrent/gdpo/credit_mode/global_gspo"] == 1.0
    assert metrics["recurrent/gdpo/global_gspo_rows"] == 4.0
    assert metrics["recurrent/gdpo/global_gspo_final_rows"] == 2.0
    assert metrics["recurrent/gdpo/segment_enabled"] == 0.0


def test_recurrent_gdpo_gates_positive_compress_quality_by_outcome():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(2, prompt_len, dtype=torch.long),
        "responses": torch.zeros(2, response_len, dtype=torch.long),
        "attention_mask": torch.ones(2, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(2, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 1], dtype=torch.long),
        "final_mask": torch.tensor([True, True], dtype=torch.bool),
        "uid": np.array(["same", "same"], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([1.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "recall_answer": np.array([0.0, 0.0], dtype=object),
        "compress_quality": np.array([0.5, 1.0], dtype=object),
        "outcome_gate": np.array([1.0, 0.0], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,recall_answer=0,compress_quality=1"
    )
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "global_gdpo"
    try:
        gdpo_adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode

    assert gdpo_adv[0, 0] > 0
    assert gdpo_adv[1, 0] < 0
    assert metrics["recurrent/gdpo/compress_quality_weight"] == 1.0


def test_recurrent_gdpo_compress_boundary_mode_uses_boundary_segments():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(4, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
        "ts_action_index": np.array([0, 1, 0, 1], dtype=object),
        "ts_action_event_chunk_idx": np.array([1, 9, 1, 9], dtype=object),
        "ts_chunk_turn_kinds": np.array([
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
        ], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
        "compress_segment_starts_json": np.array([
            json.dumps([0, 6]),
            json.dumps([0, 6]),
        ], dtype=object),
        "compress_segment_ends_json": np.array([
            json.dumps([5, 10]),
            json.dumps([5, 10]),
        ], dtype=object),
        "compress_segment_scores_json": np.array([
            json.dumps([0.0, 1.0]),
            json.dumps([1.0, 0.0]),
        ], dtype=object),
        "compress_segment_answer_scores_json": np.array([
            json.dumps([0.0, 1.0]),
            json.dumps([1.0, 0.0]),
        ], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_alpha = os.environ.get("THINKSTREAM_SEGMENT_GLOBAL_ALPHA")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    old_rho = os.environ.get("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,compress_quality=0"
    )
    os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = "0"
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "compress_boundary"
    os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = "0"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_alpha is None:
            os.environ.pop("THINKSTREAM_SEGMENT_GLOBAL_ALPHA", None)
        else:
            os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = old_alpha
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode
        if old_rho is None:
            os.environ.pop("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO", None)
        else:
            os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = old_rho

    row_adv = adv[:, 0]
    assert row_adv[0] < 0
    assert row_adv[1] > 0
    assert row_adv[2] > 0
    assert row_adv[3] < 0
    assert metrics["recurrent/gdpo/credit_mode/compress_boundary"] == 1.0
    assert metrics["recurrent/gdpo/compress_segment_answer_count"] == 4.0
    assert metrics["recurrent/gdpo/compress_segment_local_adv_nonzero_frac"] > 0.0


def test_recurrent_gdpo_question_next_compress_replaces_compress_rows():
    prompt_len = 2
    response_len = 3
    data = DataProto.from_single_dict({
        "prompts": torch.zeros(4, prompt_len, dtype=torch.long),
        "responses": torch.zeros(4, response_len, dtype=torch.long),
        "attention_mask": torch.ones(4, prompt_len + response_len, dtype=torch.long),
        "response_mask": torch.ones(4, response_len, dtype=torch.long),
        "sample_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_mask": torch.tensor([False, True, False, True], dtype=torch.bool),
        "uid": np.array(["same", "same", "same", "same"], dtype=object),
        "ts_action_index": np.array([0, 1, 0, 1], dtype=object),
        "ts_action_event_chunk_idx": np.array([1, 8, 1, 8], dtype=object),
        "ts_chunk_turn_kinds": np.array([
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
            ["streaming", "compress"],
        ], dtype=object),
    })
    reward_tensor = torch.zeros(2, response_len)
    reward_extras = {
        "outcome": np.array([0.0, 0.0], dtype=object),
        "answer_decision": np.array([0.0, 0.0], dtype=object),
        "format": np.array([0.0, 0.0], dtype=object),
        "outcome_gate": np.array([0.0, 0.0], dtype=object),
        "segment_starts_json": np.array([
            json.dumps([0, 6]),
            json.dumps([0, 6]),
        ], dtype=object),
        "segment_ends_json": np.array([
            json.dumps([5, 10]),
            json.dumps([5, 10]),
        ], dtype=object),
        "segment_scores_json": np.array([
            json.dumps([0.0, 0.0]),
            json.dumps([0.0, 0.0]),
        ], dtype=object),
        "segment_answer_scores_json": np.array([
            json.dumps([0.0, 0.0]),
            json.dumps([0.0, 0.0]),
        ], dtype=object),
        "compress_future_segment_starts_json": np.array([
            json.dumps([8]),
            json.dumps([8]),
        ], dtype=object),
        "compress_future_segment_ends_json": np.array([
            json.dumps([10]),
            json.dumps([10]),
        ], dtype=object),
        "compress_future_segment_scores_json": np.array([
            json.dumps([0.0]),
            json.dumps([1.0]),
        ], dtype=object),
        "compress_future_segment_answer_scores_json": np.array([
            json.dumps([0.0]),
            json.dumps([1.0]),
        ], dtype=object),
        "compress_future_segment_compress_quality_json": np.array([
            json.dumps([0.0]),
            json.dumps([0.0]),
        ], dtype=object),
    }

    old_weights = os.environ.get("THINKSTREAM_HDPO_WEIGHTS")
    old_alpha = os.environ.get("THINKSTREAM_SEGMENT_GLOBAL_ALPHA")
    old_mode = os.environ.get("THINKSTREAM_CREDIT_ASSIGNMENT")
    old_rho = os.environ.get("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO")
    os.environ["THINKSTREAM_HDPO_WEIGHTS"] = (
        "outcome=0,answer_decision=0,format=0,compress_quality=0"
    )
    os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = "0"
    os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = "question_next_compress"
    os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = "0"
    try:
        adv, metrics = _compute_recurrent_gdpo_advantages(
            data,
            reward_tensor,
            reward_extras,
            np.array(["same", "same"], dtype=object),
            use_adv=False,
        )
    finally:
        if old_weights is None:
            os.environ.pop("THINKSTREAM_HDPO_WEIGHTS", None)
        else:
            os.environ["THINKSTREAM_HDPO_WEIGHTS"] = old_weights
        if old_alpha is None:
            os.environ.pop("THINKSTREAM_SEGMENT_GLOBAL_ALPHA", None)
        else:
            os.environ["THINKSTREAM_SEGMENT_GLOBAL_ALPHA"] = old_alpha
        if old_mode is None:
            os.environ.pop("THINKSTREAM_CREDIT_ASSIGNMENT", None)
        else:
            os.environ["THINKSTREAM_CREDIT_ASSIGNMENT"] = old_mode
        if old_rho is None:
            os.environ.pop("THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO", None)
        else:
            os.environ["THINKSTREAM_COMPRESS_LOCAL_QUALITY_RHO"] = old_rho

    row_adv = adv[:, 0]
    assert torch.allclose(row_adv[[0, 2]], torch.zeros(2), atol=1e-6)
    assert row_adv[1] < 0
    assert row_adv[3] > 0
    assert metrics["recurrent/gdpo/credit_mode/question_next_compress"] == 1.0
    assert metrics["recurrent/gdpo/compress_future_segment_local_rows"] == 2.0
