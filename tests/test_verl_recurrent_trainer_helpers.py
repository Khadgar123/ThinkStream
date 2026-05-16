from __future__ import annotations

import sys
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
    _is_recurrent_rollout_batch,
    _make_action_token_scores,
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
