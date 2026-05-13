"""Focused SFT mask tests for pass5 trajectory rows."""

from __future__ import annotations

from pathlib import Path

import torch

from thinkstream.data.schema import (
    TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
    TRAJ_TYPE_FROM_COMPRESS,
)
from thinkstream.sft.data_processor import IGNORE_INDEX, preprocess_trajectory_sample
from thinkstream.sft.data_processor import build_recall_video_mask_from_messages


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        if token == "<|video_pad|>":
            return 7
        return -1

    def get_vocab(self):
        return {"<|im_start|>": 1, "assistant": 2, "<|im_end|>": 3}

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant":
            return [1, 2]
        return [42]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def __init__(self, input_ids):
        self._input_ids = torch.tensor([input_ids], dtype=torch.long)
        self.last_kwargs = None

    def apply_chat_template(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return {"input_ids": self._input_ids.clone()}


def test_from_compress_masks_memory_loaded_ack():
    # Two assistant turns:
    #   turn 0 = Memory loaded. ack, should be prompt-only
    #   turn 1 = real streaming answer, should carry loss
    ids = [
        1, 7, 99, 3,          # system
        1, 8, 99, 3,          # user memory prefill
        1, 2, 99, 10, 11, 3,  # assistant ack span = 10,11,3
        1, 8, 99, 3,          # user visual turn
        1, 2, 99, 20, 21, 3,  # assistant loss span = 20,21,3
    ]
    sample = {
        "trajectory_type": TRAJ_TYPE_FROM_COMPRESS,
        "video_id": "vid",
        "data_path": str(Path(".")),
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "<MEM></MEM>"}]},
            {"role": "assistant", "content": "Memory loaded."},
            {"role": "user", "content": [{"type": "text", "text": "<t=30>"}]},
            {"role": "assistant", "content": "<think>x</think><silent>"},
        ],
        "tools": [],
    }
    out = preprocess_trajectory_sample(sample, _FakeProcessor(ids))
    labels = out["labels"][0]

    assert out["eval_meta"]["n_assistant_turns"] == 2
    assert out["eval_meta"]["loss_assistant_turn_indices"] == [1]
    assert labels[12].item() == IGNORE_INDEX
    assert labels[13].item() == IGNORE_INDEX
    assert labels[21].item() == 20
    assert labels[22].item() == 21
    assert labels[23].item() == 3


def test_compact_memory_row_trains_single_mem_assistant():
    ids = [
        1, 7, 99, 3,          # system
        1, 8, 99, 3,          # user
        1, 2, 99, 30, 31, 3,  # assistant <MEM> span
    ]
    sample = {
        "trajectory_type": TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
        "video_id": "vid",
        "data_path": str(Path(".")),
        "messages": [
            {"role": "system", "content": "compact"},
            {"role": "user", "content": [{"type": "text", "text": "OLD_MEMORY"}]},
            {"role": "assistant", "content": "<MEM><m t=\"0-1\">x</m></MEM>"},
        ],
        "tools": [],
    }
    out = preprocess_trajectory_sample(sample, _FakeProcessor(ids))
    labels = out["labels"][0]

    assert out["eval_meta"]["is_compact_memory_update"] is True
    assert out["eval_meta"]["loss_assistant_turn_indices"] == [0]
    assert labels[11].item() == 30
    assert labels[12].item() == 31
    assert labels[13].item() == 3


def test_recall_video_mask_uses_message_kv_scope():
    ids = torch.tensor([[1, 7, 7, 2, 7, 7, 3]], dtype=torch.long)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ["a.jpg"], "kv_scope": "ordinary"},
            {"type": "video", "video": ["b.jpg"], "kv_scope": "recall"},
        ],
    }]
    mask = build_recall_video_mask_from_messages(
        input_ids=ids,
        tokenizer=_FakeTokenizer(),
        messages=messages,
    )
    assert mask is not None
    assert mask.tolist() == [[False, False, False, False, True, True, False]]


def test_post_recall_marks_toolcall_and_tool_response_as_recall_kv():
    # Two assistant turns. The penultimate assistant is the recall tool call;
    # the bridge before the final answer is the tool response plus final
    # assistant header. The final answer content stays ordinary KV.
    ids = [
        1, 9, 50, 3,              # system
        1, 8, 51, 3,              # user chunk
        1, 2, 99, 40, 41, 3,      # assistant recall tool-call span = 40,41,3
        1, 10, 60, 61, 3,         # tool response
        1, 2, 99, 70, 71, 3,      # final assistant answer span = 70,71,3
    ]
    sample = {
        "trajectory_type": TRAJ_TYPE_FROM_COMPRESS,
        "loss_class": "post_recall",
        "loss_assistant_turns": "last",
        "video_id": "vid",
        "data_path": str(Path(".")),
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": "<think>need</think><tool_call>{}</tool_call>"},
            {"role": "tool", "tool_call_id": "recall", "content": [
                {"type": "text", "text": "<recall_result>{}</recall_result>", "kv_scope": "recall"},
            ]},
            {"role": "assistant", "content": "<think>ok</think><answer>yes</answer>"},
        ],
        "tools": [],
    }
    processor = _FakeProcessor(ids)
    out = preprocess_trajectory_sample(sample, processor)
    kv_mask = out["recall_kv_mask"][0].tolist()
    labels = out["labels"][0]

    assert "tools" not in processor.last_kwargs
    expected = [False] * len(ids)
    for i in range(11, 22):
        expected[i] = True
    assert kv_mask == expected
    assert labels[11].item() == IGNORE_INDEX
    assert labels[22].item() == 70
