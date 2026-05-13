from thinkstream.data.agent_protocol import parse_agent_output
from thinkstream.models.agent_loop import _parse_agent_output
from thinkstream.rl.thinkstream import _extract_final_answer
from thinkstream.trainer.rewards import compute_format_reward


def test_unclosed_response_default_strict_but_lenient_recovers_answer():
    text = "<think>answer now</think><response>B"

    strict = parse_agent_output(text)
    assert strict["kind"] == "unknown"
    assert strict["answer_text"] is None
    assert strict["format_error"]

    lenient = parse_agent_output(text, allow_unclosed_response=True)
    assert lenient["kind"] == "answer"
    assert lenient["answer_text"] == "B"
    assert lenient["lenient_unclosed_response"] is True
    assert "missing </response>" in lenient["format_error"]


def test_agent_loop_uses_lenient_response_for_action_but_format_reward_stays_strict():
    text = "<think>answer now</think><response>Yes"

    parsed = _parse_agent_output(text)
    assert parsed["action"] == "response"
    assert parsed["payload"]["response"] == "Yes"
    assert parsed["lenient_unclosed_response"] is True
    assert parsed["format_error"]

    assert compute_format_reward([text]) == 0.0


def test_final_answer_fallback_uses_last_unclosed_response():
    text = (
        "<think>not yet</think><silent>"
        "<think>now answer</think><response>C"
    )
    assert _extract_final_answer(text) == "C"
