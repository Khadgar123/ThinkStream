from thinkstream.data.agent_protocol import parse_agent_output
from thinkstream.models.agent_loop import _parse_agent_output
from thinkstream.rl.thinkstream import _extract_final_answer
from thinkstream.trainer.rewards import compute_format_reward


def test_old_paired_response_tags_are_rejected_even_with_legacy_flag():
    text = "<think>answer now</think><response>B"

    strict = parse_agent_output(text)
    assert strict["kind"] == "unknown"
    assert strict["answer_text"] is None
    assert strict["format_error"]

    parsed = parse_agent_output(text, allow_unclosed_response=True)
    assert parsed["kind"] == "unknown"
    assert parsed["answer_text"] is None
    assert parsed["lenient_unclosed_response"] is False


def test_agent_loop_and_format_reward_reject_old_response_tag():
    text = "<think>answer now</think><response>Yes"

    parsed = _parse_agent_output(text)
    assert parsed["action"] == ""
    assert parsed["payload"] == {}
    assert parsed["lenient_unclosed_response"] is False
    assert parsed["format_error"]

    assert compute_format_reward([text]) == 0.0


def test_final_answer_fallback_uses_current_response_only():
    text = (
        "<think>not yet</think></Silence>"
        "<think>now answer</think><response>C</response>"
    )
    assert _extract_final_answer(text) is None
