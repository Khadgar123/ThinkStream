import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.streamingbench.base_vllm import Row, build_image_messages, parse_args  # noqa: E402


def _row(timestamp_sec: float = 80.0) -> Row:
    return Row(
        idx=0,
        csv_name="x.csv",
        question_id="Action_sample_1_q0",
        family="Action",
        sample_id=1,
        task_type="action",
        question="What happens?",
        timestamp_sec=timestamp_sec,
        answer="A",
        options=["A. open", "B. close"],
        temporal_clue_type="",
        frames_required="",
    )


def test_streamingbench_default_window_is_inside_25_45_contract():
    args = parse_args([
        "--endpoints",
        "http://127.0.0.1:18100/v1",
        "--models",
        "dummy",
    ])
    assert 25 <= args.window_sec <= 45


def test_streamingbench_prompt_window_keeps_question_off_start_boundary(tmp_path):
    frame = tmp_path / "frame_000001.jpg"
    frame.write_bytes(b"not-a-real-jpeg")
    messages = build_image_messages(_row(80), [frame], window_sec=32, fps=2)
    text = messages[1]["content"][0]["text"]
    match = re.search(r'start="([0-9.]+)" end="([0-9.]+)"', text)
    assert match
    start = float(match.group(1))
    end = float(match.group(2))
    assert end == 80.0
    assert 25 <= end - start <= 45
    assert start < end
