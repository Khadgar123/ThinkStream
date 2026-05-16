from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from tensordict import TensorDict  # noqa: E402


class Qwen3VLProcessor:
    image_token_id = 151655
    video_token_id = 151656
    name_or_path = "Qwen3-VL-8B-Instruct"

    def get_rope_index(self, input_ids, attention_mask, **kwargs):
        del kwargs
        batch, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, dtype=input_ids.dtype)
            .view(1, 1, seq_len)
            .expand(3, batch, seq_len)
        )
        return position_ids, torch.zeros((batch, 1), dtype=input_ids.dtype)


class LegacyVLProcessor(Qwen3VLProcessor):
    name_or_path = "Qwen2.5-VL-7B-Instruct"


class _ModelConfig(dict):
    pass


def _worker(processor):
    worker = object.__new__(AgentLoopWorker)
    worker.processor = processor
    worker.tokenizer = type("Tokenizer", (), {"name_or_path": processor.name_or_path})()
    worker.model_config = _ModelConfig(path=processor.name_or_path)
    return worker


def test_qwen3vl_agent_loop_keeps_official_three_channel_mrope():
    worker = _worker(Qwen3VLProcessor())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    position_ids = worker._compute_position_ids(input_ids, attention_mask, {})

    assert tuple(position_ids.shape) == (1, 3, 4)


def test_legacy_vl_agent_loop_keeps_four_channel_compat_path():
    worker = _worker(LegacyVLProcessor())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    position_ids = worker._compute_position_ids(input_ids, attention_mask, {})

    assert tuple(position_ids.shape) == (1, 4, 4)


def test_nested_qwen3vl_position_values_keep_three_channels():
    samples = [
        torch.arange(15, dtype=torch.long).reshape(3, 5),
        torch.arange(21, dtype=torch.long).reshape(3, 7),
    ]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    values = tu.nested_position_ids_values(position_ids)

    assert tuple(values.shape) == (3, 12)


def test_nested_qwen3vl_position_values_repair_equal_length_batches():
    samples = [
        torch.arange(18, dtype=torch.long).reshape(3, 6),
        torch.arange(18, 36, dtype=torch.long).reshape(3, 6),
    ]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    assert tuple(position_ids.values().shape) == (6, 6)
    values = tu.nested_position_ids_values(position_ids)

    assert tuple(values.shape) == (3, 12)


def test_mrope_position_ids_normalize_folded_channels():
    folded = torch.arange(36, dtype=torch.long).reshape(6, 1, 6)

    values = tu.normalize_mrope_position_ids(folded, expected_channels=3)

    assert tuple(values.shape) == (3, 1, 12)
    assert torch.equal(values[:, 0, :6], folded[:3, 0])
    assert torch.equal(values[:, 0, 6:], folded[3:, 0])


def test_maybe_fix_rebuilds_malformed_equal_length_position_nested():
    samples = [
        torch.arange(18, dtype=torch.long).reshape(3, 6),
        torch.arange(18, 36, dtype=torch.long).reshape(3, 6),
    ]
    malformed = torch.nested.as_nested_tensor(samples, layout=torch.jagged)
    data = TensorDict(
        {
            "position_ids": malformed,
            "input_ids": torch.nested.as_nested_tensor(
                [torch.arange(6), torch.arange(6, 12)], layout=torch.jagged
            ),
        },
        batch_size=[2],
    )

    tu.maybe_fix_3d_position_ids(data)
    selected = tu.index_select_tensor_dict(data, torch.tensor([0, 1]))["position_ids"]
    values = tu.nested_position_ids_values(selected)

    assert tuple(data["position_ids"].values().shape) == (3, 12)
    assert tuple(values.shape) == (3, 12)
