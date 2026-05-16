from thinkstream.sft.trainer import (
    _behavior_bucket_for_expected_kind,
    _parse_eval_turn_output,
    _response_detail_flags,
)


def _flags(text):
    parsed = _parse_eval_turn_output(text)
    return _response_detail_flags(text, parsed)


def test_response_detail_flags_strict_response():
    flags = _flags("<think>ok</think></Response> A person enters.")
    assert flags["response_open"] is True
    assert flags["response_close"] is True
    assert flags["response_nonempty"] is True
    assert flags["response_strict"] is True
    assert flags["silent"] is False


def test_response_detail_flags_old_response_tag_is_not_accepted():
    flags = _flags("<think>ok</think><response>A person enters.")
    assert flags["response_open"] is False
    assert flags["response_close"] is False
    assert flags["response_nonempty"] is False
    assert flags["response_strict"] is False
    assert flags["silent"] is False


def test_response_detail_flags_silent_on_expected_response():
    flags = _flags("<think>wait</think></Silence>")
    assert flags["response_open"] is False
    assert flags["response_close"] is False
    assert flags["response_nonempty"] is False
    assert flags["response_strict"] is False
    assert flags["silent"] is True


def test_response_detail_flags_empty_response_not_strict():
    flags = _flags("<think>ok</think></Response>   ")
    assert flags["response_open"] is True
    assert flags["response_close"] is True
    assert flags["response_nonempty"] is False
    assert flags["response_strict"] is False
    assert flags["silent"] is False


def test_post_recall_response_uses_dedicated_eval_bucket():
    assert (
        _behavior_bucket_for_expected_kind(
            "answer_nonempty",
            "streaming_trajectory",
            post_recall_answer=True,
        )
        == "response_post_recall"
    )
    assert (
        _behavior_bucket_for_expected_kind(
            "answer_nonempty",
            "streaming_trajectory",
            post_recall_answer=False,
        )
        == "response"
    )
