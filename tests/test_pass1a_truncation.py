import json
import asyncio

from scripts.agent_data.pass1a_evidence import run_pass1a
from scripts.agent_data_pipeline.vllm_client import TruncatedCompletionError


class _TruncatingClient:
    def __init__(self):
        self.request_ids = []

    async def _call_one(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        request_id,
        enable_thinking,
        mm_processor_kwargs=None,
        media_io_kwargs=None,
    ):
        self.request_ids.append((request_id, max_tokens, temperature))
        if request_id == "vid_1a_0":
            raise TruncatedCompletionError(
                "vid_1a_0: finish_reason=length max_tokens=16384 "
                "prompt_tokens=10 completion_tokens=16384"
            )
        assert request_id == "vid_1a_0_compact_retry"
        return json.dumps({
            "time": [0, 1],
            "visible_entities": [{
                "desc": "person in a dark shirt",
                "action": "standing",
                "position": "center",
            }],
            "atomic_facts": ["A person stands near the center of the frame."],
            "ocr": [],
            "spatial": "The person is in the center of the frame.",
            "think": (
                "A person in a dark shirt stands near the center of the frame, "
                "with the visible scene focused on their current posture and "
                "surrounding space."
            ),
        })


def test_pass1a_truncation_uses_compact_retry():
    client = _TruncatingClient()

    captions = asyncio.run(
        run_pass1a(
            video_id="vid",
            frame_paths=[],
            num_chunks=1,
            client=client,
        )
    )

    assert captions[0]["parse_success"] is True
    assert captions[0]["_truncation_recovered"] is True
    assert captions[0]["visible_entities"][0]["desc"] == "person in a dark shirt"
    assert client.request_ids == [
        ("vid_1a_0", 16384, 0.3),
        ("vid_1a_0_compact_retry", 2048, 0.2),
    ]
