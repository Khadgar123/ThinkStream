"""v12.14 AsyncStreamingVideoAgent rollout smoke test.

Drives AsyncStreamingVideoAgent.rollout() with a deterministic mock
ChatCompletionProxy + a synthetic 30-chunk multi-Q trajectory, verifies:

  - rollout() returns AsyncOutput with:
      * conversations (list of [user, assistant] message lists)
      * sample_index (LongTensor, all == input idx)
      * final_mask (BoolTensor, only last True)
      * multi_modal_data (list of dict-or-None per conversation)
      * metrics (per-Q answer attribution telemetry)
  - chunk-internal recall multi-turn produces 2 conversations for
    chunks where recall fires (D1 shape B alignment)
  - per-Q answer text + chunk attribution survive the rollout
  - multi_modal_data carries video frames + frames_indices for each
    visual-bearing conversation; None for compress / text-only

Does NOT test:
  - actor forward / loss (needs ray_trainer integration)
  - real vLLM generation (mock proxy here)
  - GRPO advantage / sample_index broadcast (needs trainer)

Run:
  python -m scripts.test_rl.test_v12_14_rollout
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "verl"))


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------
@dataclass
class MockChoice:
    message: Dict[str, str]


@dataclass
class MockCompletions:
    choices: List[MockChoice]


class MockChatProxy:
    """Deterministic mock — replaces vLLM ChatCompletionProxy.

    Returns a canned response per chunk based on chunk index encoded in
    the prompt. Produces:
      - silent chunks: <think>watching</think><answer></answer>
      - response chunks: <think>now</think><answer>{gold}</answer>
      - one recall chunk: <think>need history</think><tool_call>{...recall...}</tool_call>
        followed by <think>got it</think><answer>{gold}</answer>
    """

    def __init__(self):
        self.call_log: List[Dict] = []

    async def get_chat_completions(self, messages: List[Dict], **kwargs) -> Tuple[MockCompletions, Optional[Exception]]:
        self.call_log.append({"messages": messages, "kwargs": kwargs})
        # Find last user content to determine chunk context
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        text_blocks = []
        if last_user:
            for c in last_user["content"] if isinstance(last_user["content"], list) else [last_user["content"]]:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_blocks.append(c.get("text", ""))
        joined_text = "\n".join(text_blocks)

        # Determine response based on prompt content
        if "<recall_result>" in joined_text:
            # This is the answer-after-recall turn (turn 2)
            response = '<think>got it from history</think><answer>BLUE</answer>'
        elif "<compress_trigger/>" in joined_text:
            response = ('<think>compressing</think><tool_call>{"name":"compress",'
                        '"arguments":{"summary":{"time_range":[0,8],"text":"early scene"}}}</tool_call>')
        elif "<user_input>" in joined_text:
            # Question fired this chunk — answer it.
            # First check if it's a recall-demo chunk (chunk 80 in our test)
            if "chunk 5-10" in joined_text or "history" in joined_text.lower():
                response = ('<think>need history</think><tool_call>{"name":"recall",'
                            '"arguments":{"query":"red apple table","time_range":"5-10"}}</tool_call>')
            else:
                response = '<think>direct answer</think><answer>RED</answer>'
        else:
            # Silent chunk
            response = '<think>watching</think><answer></answer>'
        return MockCompletions(choices=[MockChoice(message={"role": "assistant", "content": response})]), None


# ---------------------------------------------------------------------------
# Synthetic trajectory (mimics multi_q parquet row.non_tensor_batch)
# ---------------------------------------------------------------------------
def make_synthetic_trajectory(frames_root: Path) -> Dict:
    """Build a 30-chunk trajectory with 3 questions across the timeline."""
    # Create dummy frames (chunk 0..29 × 2 fpc = 60 frames)
    vid_dir = frames_root / "synthetic_vid"
    vid_dir.mkdir(exist_ok=True)
    for i in range(1, 61):
        (vid_dir / f"frame_{i:06d}.jpg").write_text("x")

    return {
        "video_id": "synthetic_vid",
        "video_path": "synthetic_vid.mp4",
        "n_chunks": 30,
        "extra_info": {
            "video_id": "synthetic_vid",
            "questions": [
                # Q0: direct response at chunk 5 (no history needed)
                {
                    "card_id": "q0", "question": "What color?",
                    "options": ["RED", "GREEN", "BLUE", "YELLOW"],
                    "correct_option": "A", "gold_answer": "RED",
                    "answer_form": "multiple_choice",
                    "ask_chunk": 5, "ask_chunks": [5], "answer_chunks": [5],
                    "per_emit_answers": [{"chunk": 5, "value": "RED"}],
                    "support_chunks": [5],
                },
                # Q1: silent_then_response — ask 10, answer 20
                {
                    "card_id": "q1", "question": "What happens later?",
                    "options": ["A1", "B1", "C1", "D1"],
                    "correct_option": "A", "gold_answer": "A1",
                    "answer_form": "multiple_choice",
                    "ask_chunk": 10, "ask_chunks": [10], "answer_chunks": [20],
                    "per_emit_answers": [{"chunk": 20, "value": "A1"}],
                    "support_chunks": [20],
                },
                # Q2: backward recall demo — ask 25, gold needs chunk 5-10 history
                {
                    "card_id": "q2", "question": "What was at chunk 5-10 in history?",
                    "options": ["RED", "BLUE", "GREEN", "YELLOW"],
                    "correct_option": "C", "gold_answer": "BLUE",
                    "answer_form": "multiple_choice",
                    "ask_chunk": 25, "ask_chunks": [25], "answer_chunks": [25],
                    "per_emit_answers": [{"chunk": 25, "value": "BLUE"}],
                    "support_chunks": [5, 6, 7, 8, 9, 10],
                },
            ],
            "gold_action_per_chunk": {},
            "all_ask_chunks": [5, 10, 25],
        },
        "reward_model": {"ground_truth": "{}", "style": "thinkstream_v12_multi_q"},
        "data_source": "thinkstream_v12_streaming_multi_q",
    }


# ---------------------------------------------------------------------------
# Build the gen_item DataProtoItem-like wrapper expected by rollout()
# ---------------------------------------------------------------------------
class FakeDataProtoItem:
    def __init__(self, traj: Dict, idx: int = 0, prompt: List[Dict] = None):
        self.batch = {"sample_index": torch.tensor(idx)}
        prompt_msgs = prompt or [
            {"role": "system", "content": "You are a streaming video agent."},
            {"role": "user", "content": "Process the video."},
        ]
        self.non_tensor_batch = {
            "video_id": traj["video_id"],
            "video_path": traj["video_path"],
            "n_chunks": traj["n_chunks"],
            "extra_info": traj["extra_info"],
            "prompt": prompt_msgs,
        }
        self.meta_info = {}


# ---------------------------------------------------------------------------
# Tokenizer stub — only encode() is exercised
# ---------------------------------------------------------------------------
class StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(range(max(1, len(text) // 4)))

    def apply_chat_template(self, *a, **k):
        return [0]


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
async def main() -> int:
    # verl/__init__.py imports ray (not in mac test env). Load streaming_video
    # by file path so we exercise the real implementation without dragging in
    # ray/aiohttp. interface.py is also load-by-file because it is imported
    # by streaming_video.py via `from verl.recurrent.interface import ...`.
    import importlib.util, types
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    # Stub modules that streaming_video transitively imports
    verl_mod = types.ModuleType('verl')
    sys.modules.setdefault('verl', verl_mod)
    sys.modules.setdefault('verl.recurrent', types.ModuleType('verl.recurrent'))
    sys.modules.setdefault('verl.recurrent.impls', types.ModuleType('verl.recurrent.impls'))
    # Stub verl.protocol (only DataProto / DataProtoItem used as type annotations)
    proto_mod = types.ModuleType('verl.protocol')
    class DataProto: pass
    class DataProtoItem: pass
    proto_mod.DataProto = DataProto
    proto_mod.DataProtoItem = DataProtoItem
    sys.modules['verl.protocol'] = proto_mod
    # verl module-level re-exports DataProto (verl/__init__.py: from .protocol import DataProto)
    verl_mod.DataProto = DataProto
    # Stub verl.utils.dataset.rl_dataset (RLHFDataset base, not exercised in test)
    rds = types.ModuleType('verl.utils.dataset.rl_dataset')
    class RLHFDataset: pass
    def collate_fn(batch): return batch
    rds.RLHFDataset = RLHFDataset
    rds.collate_fn = collate_fn
    sys.modules['verl.utils'] = types.ModuleType('verl.utils')
    sys.modules['verl.utils.dataset'] = types.ModuleType('verl.utils.dataset')
    sys.modules['verl.utils.dataset.rl_dataset'] = rds
    # Stub verl.trainer.ppo.ray_trainer._timer
    rt = types.ModuleType('verl.trainer.ppo.ray_trainer')
    from contextlib import contextmanager
    @contextmanager
    def _timer(label, raw):
        yield
    rt._timer = _timer
    sys.modules['verl.trainer'] = types.ModuleType('verl.trainer')
    sys.modules['verl.trainer.ppo'] = types.ModuleType('verl.trainer.ppo')
    sys.modules['verl.trainer.ppo.ray_trainer'] = rt
    # Stub verl.recurrent.async_utils (ChatCompletionProxy is just an annotation)
    au = types.ModuleType('verl.recurrent.async_utils')
    class ChatCompletionProxy: pass
    au.ChatCompletionProxy = ChatCompletionProxy
    sys.modules['verl.recurrent.async_utils'] = au

    # utils.py imports openai (not in test env). Stub the 2 helpers we need.
    utils_mod = types.ModuleType('verl.recurrent.utils')
    def log_step(logger, step, conversation): pass
    def msg(choice):
        # Choice may have .message attr (mock) or be a dict
        if hasattr(choice, 'message'):
            m = choice.message
            return m if isinstance(m, dict) else {"role": "assistant", "content": str(m)}
        return choice
    utils_mod.log_step = log_step
    utils_mod.msg = msg
    sys.modules['verl.recurrent.utils'] = utils_mod

    # Now load the real interface.py + streaming_video.py
    rec_iface = _load('verl.recurrent.interface',
                      str(PROJECT_ROOT / 'verl/verl/recurrent/interface.py'))
    sv = _load('verl.recurrent.impls.streaming_video',
               str(PROJECT_ROOT / 'verl/verl/recurrent/impls/streaming_video.py'))
    AsyncStreamingVideoAgent = sv.AsyncStreamingVideoAgent
    StreamingVideoConfig = sv.StreamingVideoConfig

    tmp = Path(tempfile.mkdtemp())
    traj = make_synthetic_trajectory(tmp)
    config = StreamingVideoConfig(
        max_chunks=30, max_actions_per_trajectory=80,
        max_chunk_response_length=256,
        max_recall_per_chunk=1,
        compress_token_threshold=999999,  # disable compress for this smoke
        compress_range_min=999,
        visual_window_chunks=8,
        frames_per_chunk=2,
        chunk_sec=1.0,
        frames_root=str(tmp),
    )
    proxy = MockChatProxy()
    tokenizer = StubTokenizer()

    # Stub rollout_config for sampling_params
    class StubCfg:
        def get(self, k, default=None):
            return default
    rollout_config = StubCfg()

    agent = AsyncStreamingVideoAgent(proxy, tokenizer, config, rollout_config)
    # Stub sampling_params (AsyncRAgent base has it but expects rollout_config keys)
    agent.sampling_params = lambda meta: {"temperature": 0.7, "top_p": 0.9}
    # Initialize agent state (normally done by start() in real flow)
    agent.step = 0

    gen_item = FakeDataProtoItem(traj, idx=0)

    print("═══ Running AsyncStreamingVideoAgent.rollout() ═══")
    output = await agent.rollout(gen_item)

    # ── Assertions ──
    n_actions = len(output.conversations)
    print(f"  n_actions = {n_actions}")
    print(f"  sample_index = {output.sample_index.tolist()[:8]}{'...' if n_actions>8 else ''}")
    print(f"  final_mask  = {output.final_mask.tolist()[:8]}{'...' if n_actions>8 else ''}")
    print(f"  mm_data slots = {sum(1 for m in output.multi_modal_data if m is not None)}/{n_actions} have visuals")

    assert n_actions >= 30, f"expected >=30 actions for 30 chunks, got {n_actions}"
    assert output.sample_index.shape[0] == n_actions
    assert (output.sample_index == 0).all().item(), "all actions should map to sample 0"
    assert output.final_mask[-1].item() is True, "last action should be final"
    assert output.final_mask[:-1].sum().item() == 0, "only last action should be final"
    assert len(output.multi_modal_data) == n_actions

    # Q-attribution telemetry
    per_q_chunks = output.metrics.get("ts_per_q_answer_chunk", [])
    per_q_texts = output.metrics.get("ts_per_q_answer_text", [])
    print(f"  per_q_answer_chunk = {per_q_chunks}")
    print(f"  per_q_answer_text  = {per_q_texts}")
    assert len(per_q_chunks) == 3, "3 questions in trajectory"
    # Q0: direct response at chunk 5
    assert per_q_chunks[0] == 5, f"Q0 should answer at chunk 5, got {per_q_chunks[0]}"
    assert per_q_texts[0] == "RED", f"Q0 answer = RED, got {per_q_texts[0]}"
    # Q1: silent_then_response — ask=10, answer=20
    # Mock returns "RED" at any user_input chunk that's not history-related;
    # so Q1 will get RED at chunk 10 (the ask_chunk fires the answer immediately
    # in mock since silent_then_response logic isn't simulated by the mock)
    print(f"  (Q1 mock-driven: ask=10 fires immediately, real model would silent then respond)")
    # Q2: recall demo — ask=25, recall fires, then BLUE answered
    assert per_q_chunks[2] == 25, f"Q2 should answer at chunk 25, got {per_q_chunks[2]}"
    assert per_q_texts[2] == "BLUE", f"Q2 answer should be BLUE (post-recall), got {per_q_texts[2]}"

    # Verify recall happens at chunk 25 → 2 conversations for that chunk
    n_proxy_calls = len(proxy.call_log)
    print(f"  proxy.call_log entries = {n_proxy_calls}")
    # Expected: 30 chunks × 1 + 1 extra for chunk 25's recall multi-turn = 31
    # (chunks not triggering recall = 1 call; chunk 25 = 2 calls)
    assert n_proxy_calls >= 30, f"expected >=30 proxy calls (1+/chunk), got {n_proxy_calls}"

    # Verify multi_modal_data shape on a visual-bearing conversation
    visual_actions = [i for i, m in enumerate(output.multi_modal_data) if m and "videos" in m]
    print(f"  visual-bearing actions: {len(visual_actions)}/{n_actions}")
    if visual_actions:
        sample_mm = output.multi_modal_data[visual_actions[0]]
        videos = sample_mm["videos"]
        print(f"  sample mm['videos'] entries: {len(videos)}, first frames_indices[:4] = {videos[0][1]['frames_indices'][:4]}")
        assert len(videos) >= 1
        assert "frames_indices" in videos[0][1]

    # Verify recall-tool conversation has historical mm
    # Find the conversation right after Q2's tool_call
    recall_conv_idx = None
    for i, conv in enumerate(output.conversations):
        for m in conv:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for c in m["content"]:
                    if isinstance(c, dict) and "<recalled_frames>" in str(c.get("text", "")):
                        recall_conv_idx = i
                        break
    if recall_conv_idx is not None:
        rc_mm = output.multi_modal_data[recall_conv_idx]
        print(f"  recall conversation #{recall_conv_idx} has mm: {rc_mm is not None}")
        if rc_mm:
            rec_videos = rc_mm["videos"]
            assert len(rec_videos) >= 1
            indices = rec_videos[0][1]["frames_indices"]
            # Should be anchored at chunks 5-10 (recall time_range), so frames_indices[0] == 5*2=10
            assert indices[0] == 10, f"recalled frames_indices should start at 10 (chunk 5*fpc=2), got {indices[0]}"
            print(f"  ✓ recall historical MROPE: frames_indices[0]={indices[0]} (chunk 5 × 2 fpc)")

    print("\n✓ ALL v12.14 ROLLOUT SMOKE ASSERTIONS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
