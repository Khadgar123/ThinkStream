"""Shared prompt/answer contract helpers for eval paths.

SFT/RL streaming prompts keep the user's question text separate from the
structured active-query block. Eval code should follow the same contract when
it drives ``StreamingAgentLoop``: pass a bare ``user_question`` and put MCQ
options / answer format in ``user_question_meta`` so
``format_queries_block()`` renders them exactly once.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from thinkstream.data.agent_protocol import answer_format_instruction


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_OPTION_LABEL_RE = re.compile(r"^\s*(?:\([A-Z]\)|[A-Z][\).:])\s*")


def strip_option_label(option: Any) -> str:
    return _OPTION_LABEL_RE.sub("", str(option or "")).strip()


def label_mc_options(
    options: Sequence[Any],
    *,
    style: str = "paren",
    max_options: int = 26,
) -> List[str]:
    """Return canonical labeled options, stripping any existing A./A) prefix."""

    sep = ")" if style == "paren" else "."
    out: List[str] = []
    for i, opt in enumerate(list(options or [])[:max_options]):
        out.append(f"{LETTERS[i]}{sep} {strip_option_label(opt)}")
    return out


def build_plain_mcq_prompt(
    question: str,
    options: Sequence[Any],
    *,
    option_style: str = "dot",
    instruction: str = "Answer with a single letter.",
    include_options_header: bool = False,
) -> str:
    """Plain VLM/base prompt with options inline.

    Use this only for baseline/offline prompts that do not also render
    ``<active_query>``. Streaming-agent eval should instead use
    ``build_streaming_query_meta`` and pass the bare question text.
    """

    lines = [str(question or "")]
    if options and include_options_header:
        lines.append("Options:")
    lines.extend(label_mc_options(options, style=option_style))
    if instruction:
        lines.append(instruction)
    return "\n".join(line for line in lines if line)


def infer_answer_form(item: Dict[str, Any]) -> str:
    if item.get("options"):
        return "multiple_choice"
    answer = str(item.get("answer") or item.get("gold_answer") or "").strip()
    if answer.lower() in {"yes", "no"}:
        return "binary"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
        return "number"
    return str(item.get("answer_form") or "")


def build_streaming_query_meta(
    item: Dict[str, Any],
    *,
    answer_form: str | None = None,
    answer_style: str = "letter_plus_text",
) -> Dict[str, Any]:
    """Structured metadata for ``MemoryState.add_query`` / active_query.

    For MCQ, options are canonicalized as ``A) ...`` because that is the SFT/RL
    active-query style. ``correct_option`` and accepted answers are included for
    audits and future scoring even though MemoryState currently ignores them.
    """

    form = answer_form or infer_answer_form(item)
    meta: Dict[str, Any] = {"answer_form": form}
    answer_chunks = item.get("answer_chunks") or item.get("expected_answer_chunks")
    if answer_chunks is not None:
        try:
            meta["answer_chunks"] = [int(x) for x in answer_chunks]
        except TypeError:
            try:
                meta["answer_chunks"] = [int(answer_chunks)]
            except (TypeError, ValueError):
                pass
        except ValueError:
            pass
    per_emit = item.get("per_emit_answers")
    if per_emit:
        meta["per_emit_answers"] = list(per_emit)
    if meta.get("answer_chunks") and item.get("open_until") is not None:
        meta["open_until"] = item.get("open_until")
    if form == "multiple_choice":
        opts = label_mc_options(item.get("options") or [], style="paren")
        correct_raw = item.get("correct_option", "")
        correct = ""
        if isinstance(correct_raw, int) and 0 <= correct_raw < len(LETTERS):
            correct = LETTERS[correct_raw]
        elif isinstance(correct_raw, str):
            co = correct_raw.strip().upper()
            if len(co) == 1 and co in LETTERS:
                correct = co
            else:
                try:
                    idx = int(co)
                    correct = LETTERS[idx] if 0 <= idx < len(LETTERS) else ""
                except ValueError:
                    correct = ""
        if not correct and item.get("gt") is not None:
            try:
                correct = LETTERS[int(item["gt"])]
            except (TypeError, ValueError, IndexError):
                correct = ""
        answer = str(item.get("answer") or item.get("gold_answer") or "").strip()
        if not correct and len(answer) == 1 and answer.upper() in LETTERS:
            correct = answer.upper()
        meta.update({
            "options": opts,
            "answer_style": answer_style,
            "answer_instruction": answer_format_instruction(
                "multiple_choice", answer_style=answer_style, options=opts,
            ),
            "correct_option": correct,
            "gold_answer": item.get("answer") or item.get("gold_answer", ""),
        })
    else:
        instruction = answer_format_instruction(form)
        if instruction:
            meta["answer_instruction"] = instruction
    return meta
