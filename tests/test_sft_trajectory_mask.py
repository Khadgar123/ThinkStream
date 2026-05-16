"""Focused SFT mask tests for pass5 trajectory rows."""

from __future__ import annotations

from pathlib import Path

import torch

from thinkstream.data.schema import (
    MEMORY_LOAD_ACK,
    TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
    TRAJ_TYPE_FROM_COMPRESS,
)
from thinkstream.sft.data_processor import IGNORE_INDEX, preprocess_trajectory_sample
from thinkstream.sft.data_processor import build_recall_video_mask_from_messages
from thinkstream.sft.data_processor import (
    LOSS_BUCKET_ACTION,
    LOSS_BUCKET_TEXT,
    LOSS_SUBBUCKET_ACT_RECALL,
    LOSS_SUBBUCKET_ACT_RESPONSE,
    LOSS_SUBBUCKET_ACT_SILENT,
    LOSS_SUBBUCKET_TEXT_COMPRESSION,
    LOSS_SUBBUCKET_TEXT_THINK,
    _build_loss_bucket_ids,
)


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


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant":
            return [1, 2]
        return [1000 + ord(ch) for ch in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        out = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(int(i) - 1000) for i in ids)


def _bucketize_text(text: str):
    tok = _CharTokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids + [3]], dtype=torch.long)
    labels = input_ids.clone()
    return _build_loss_bucket_ids(
        input_ids=input_ids,
        labels=labels,
        loss_spans=[(0, len(ids))],
        tokenizer=tok,
    )


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
            {"role": "user", "content": [{"type": "text", "text": '<m t="0-1">x</m>'}]},
            {"role": "assistant", "content": MEMORY_LOAD_ACK},
            {"role": "user", "content": [{"type": "text", "text": "<t=30>"}]},
            {"role": "assistant", "content": "<think>x</think></Silence>"},
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
        1, 2, 99, 30, 31, 3,  # assistant compact-memory span
    ]
    sample = {
        "trajectory_type": TRAJ_TYPE_COMPACT_MEMORY_UPDATE,
        "video_id": "vid",
        "data_path": str(Path(".")),
        "messages": [
            {"role": "system", "content": "compact"},
            {"role": "user", "content": [{"type": "text", "text": "OLD_MEMORY"}]},
            {"role": "assistant", "content": '<m t="0-1">x</m>'},
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


def test_loss_bucket_split_response_answer_from_think():
    text = "<think>visual state</think></Response> A) red cup"
    bucket_ids, subbucket_ids, diag = _bucketize_text(text)
    buckets = bucket_ids[0].tolist()
    subbuckets = subbucket_ids[0].tolist()

    response_pos = text.index("</Response>")
    answer_pos = text.index("A")
    think_body_pos = text.index("visual")

    assert buckets[response_pos] == LOSS_BUCKET_ACTION
    assert subbuckets[response_pos] == LOSS_SUBBUCKET_ACT_RESPONSE
    assert buckets[answer_pos] == LOSS_BUCKET_ACTION
    assert subbuckets[answer_pos] == LOSS_SUBBUCKET_ACT_RESPONSE
    assert buckets[think_body_pos] == LOSS_BUCKET_TEXT
    assert subbuckets[think_body_pos] == LOSS_SUBBUCKET_TEXT_THINK
    assert diag["tokens_by_subbucket"]["act_response"] >= len("</Response> A) red cup")


def test_loss_bucket_split_silent_and_recall_time():
    text = (
        "<think>need old frame</think>"
        '<tool_call>{"name":"recall","arguments":{"start_time":0,"end_time":3}}</tool_call>'
    )
    bucket_ids, subbucket_ids, diag = _bucketize_text(text)
    buckets = bucket_ids[0].tolist()
    subbuckets = subbucket_ids[0].tolist()

    tool_pos = text.index("<tool_call>")
    recall_pos = text.index("recall")
    start_time_pos = text.index('"start_time"')

    assert buckets[tool_pos] == LOSS_BUCKET_ACTION
    assert subbuckets[tool_pos] == LOSS_SUBBUCKET_ACT_RECALL
    assert buckets[recall_pos] == LOSS_BUCKET_ACTION
    assert subbuckets[recall_pos] == LOSS_SUBBUCKET_ACT_RECALL
    assert buckets[start_time_pos] == LOSS_BUCKET_ACTION
    assert subbuckets[start_time_pos] == LOSS_SUBBUCKET_ACT_RECALL
    assert diag["tokens_by_subbucket"]["act_recall"] >= len(
        '<tool_call>{"name":"recall","arguments":{"start_time":0,"end_time":3}}</tool_call>'
    )

    silent_bucket, silent_subbucket, _ = _bucketize_text(
        "<think>waiting</think></Silence>"
    )
    silent_pos = "<think>waiting</think></Silence>".index("</Silence>")
    assert silent_bucket[0, silent_pos].item() == LOSS_BUCKET_ACTION
    assert silent_subbucket[0, silent_pos].item() == LOSS_SUBBUCKET_ACT_SILENT


def test_loss_bucket_split_memory_key_and_compression_body():
    text = '<m t="0-23">A person crosses water.</m>'
    bucket_ids, subbucket_ids, diag = _bucketize_text(text)
    buckets = bucket_ids[0].tolist()
    subbuckets = subbucket_ids[0].tolist()

    open_pos = text.index("<m")
    body_pos = text.index("person")
    close_pos = text.index("</m>")

    assert buckets[open_pos] == LOSS_BUCKET_TEXT
    assert subbuckets[open_pos] == LOSS_SUBBUCKET_TEXT_COMPRESSION
    assert buckets[body_pos] == LOSS_BUCKET_TEXT
    assert subbuckets[body_pos] == LOSS_SUBBUCKET_TEXT_COMPRESSION
    assert buckets[close_pos] == LOSS_BUCKET_TEXT
    assert subbuckets[close_pos] == LOSS_SUBBUCKET_TEXT_COMPRESSION
    assert diag["tokens_by_subbucket"]["text_compression"] >= len(text)


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
                {"type": "text", "text": "The recall tool returned historical video frames for t=0-2.", "kv_scope": "recall"},
            ]},
            {"role": "assistant", "content": "<think>ok</think></Response> yes"},
        ],
        "tools": [],
    }
    processor = _FakeProcessor(ids)
    out = preprocess_trajectory_sample(sample, processor)
    kv_mask = out["recall_kv_mask"][0].tolist()
    query_mask = out["recall_query_mask"][0].tolist()
    labels = out["labels"][0]

    assert "tools" not in processor.last_kwargs
    expected = [False] * len(ids)
    for i in range(11, 22):
        expected[i] = True
    assert kv_mask == expected
    expected_q = [False] * len(ids)
    for i in range(11, 25):
        expected_q[i] = True
    assert query_mask == expected_q
    assert labels[11].item() == IGNORE_INDEX
    assert labels[22].item() == 70
    assert out["eval_meta"]["post_recall_answer_turn_indices"] == [1]


def test_full_trajectory_recall_tool_response_marks_recall_kv():
    # Same shape as a multi-turn trajectory row: recall happens inside a
    # from_compress segment, not in a standalone post_recall row.
    ids = [
        1, 9, 50, 3,              # system
        1, 8, 51, 7, 7, 3,        # user chunk with ordinary video
        1, 2, 99, 40, 41, 3,      # assistant recall tool-call span
        1, 10, 60, 7, 7, 61, 3,   # tool response with recall video
        1, 2, 99, 70, 71, 3,      # final assistant answer span
        1, 8, 52, 7, 7, 3,        # next ordinary user chunk
        1, 2, 99, 80, 81, 3,      # next assistant turn
    ]
    sample = {
        "trajectory_type": TRAJ_TYPE_FROM_COMPRESS,
        "loss_assistant_turns": "all",
        "video_id": "vid",
        "data_path": str(Path(".")),
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [
                {"type": "text", "text": "<t=3>"},
                {"type": "video", "video": ["a.jpg"], "kv_scope": "ordinary"},
            ]},
            {"role": "assistant", "content": "<think>need</think>", "tool_calls": [{
                "id": "rec_3",
                "type": "function",
                "function": {"name": "recall", "arguments": {"start_time": 0, "end_time": 2}},
            }]},
            {"role": "tool", "tool_call_id": "rec_3", "content": [
                {"type": "text", "text": "The recall tool returned historical video frames for t=0-2."},
                {"type": "video", "video": ["b.jpg"]},
            ]},
            {"role": "assistant", "content": "<think>ok</think></Response> yes"},
            {"role": "user", "content": [
                {"type": "text", "text": "<t=4>"},
                {"type": "video", "video": ["c.jpg"], "kv_scope": "ordinary"},
            ]},
            {"role": "assistant", "content": "<think>continue</think></Silence>"},
        ],
        "tools": [],
    }
    out = preprocess_trajectory_sample(sample, _FakeProcessor(ids))
    kv_mask = out["recall_kv_mask"][0].tolist()
    query_mask = out["recall_query_mask"][0].tolist()
    video_mask = out["recall_video_mask"][0].tolist()

    expected_kv = [False] * len(ids)
    for i in range(13, 26):
        expected_kv[i] = True
    assert kv_mask == expected_kv
    assert query_mask[13] is True
    assert query_mask[28] is True
    assert query_mask[29] is False
    assert video_mask[19] is True
    assert video_mask[20] is True
    assert video_mask[32] is False
    assert video_mask[33] is False
    assert out["eval_meta"]["post_recall_answer_turn_indices"] == [1]
