from thinkstream.sft.trainer import (
    _parse_eval_turn_output,
    _response_detail_flags,
)


def _flags(text):
    parsed = _parse_eval_turn_output(text)
    return _response_detail_flags(text, parsed)


def test_response_detail_flags_strict_response():
    flags = _flags("<think>ok</think><response>A person enters.</response>")
    assert flags["response_open"] is True
    assert flags["response_close"] is True
    assert flags["response_nonempty"] is True
    assert flags["response_strict"] is True
    assert flags["silent"] is False


def test_response_detail_flags_missing_close_keeps_nonempty_signal():
    flags = _flags("<think>ok</think><response>A person enters.")
    assert flags["response_open"] is True
    assert flags["response_close"] is False
    assert flags["response_nonempty"] is True
    assert flags["response_strict"] is False
    assert flags["silent"] is False


def test_response_detail_flags_silent_on_expected_response():
    flags = _flags("<think>wait</think><silent>")
    assert flags["response_open"] is False
    assert flags["response_close"] is False
    assert flags["response_nonempty"] is False
    assert flags["response_strict"] is False
    assert flags["silent"] is True


def test_response_detail_flags_empty_response_not_strict():
    flags = _flags("<think>ok</think><response>   </response>")
    assert flags["response_open"] is True
    assert flags["response_close"] is True
    assert flags["response_nonempty"] is False
    assert flags["response_strict"] is False
    assert flags["silent"] is False
