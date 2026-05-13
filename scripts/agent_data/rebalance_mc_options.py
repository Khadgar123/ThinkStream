"""Rebalance multiple-choice option letters in final ThinkStream datasets.

The generator emits open-ended answer text, but MC prompts still expose option
letters. If the correct letter is skewed, models can learn a position prior.
This script deterministically relabels MC options after final split creation
while preserving the correct answer text and all timing metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TRAJ_FILES = {
    "train_sft": "train_sft_trajectories.jsonl",
    "train_rl": "train_rl_trajectories.jsonl",
    "val": "val_trajectories.jsonl",
    "test": "test_trajectories.jsonl",
}
ALL_JSONL_FILES = [
    "train.jsonl",
    "train_sft.jsonl",
    "train_rl.jsonl",
    "val.jsonl",
    "test.jsonl",
    "phase1_train.jsonl",
    "phase2_train.jsonl",
    "phase5_train.jsonl",
    "c1_train.jsonl",
    "train_sft_full.jsonl",
    *TRAJ_FILES.values(),
    "train_sft_messages.jsonl",
    "val_messages.jsonl",
    "test_messages.jsonl",
]

OPTION_RE = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[\).:])\s*(.*)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"<(answer|response)>(.*?)</\1>", re.DOTALL)
QUERY_BLOCK_RE = re.compile(
    r"(?P<qline>\[[^\]\n]+s\]\s+Q:\s+(?P<question>.*?)\n)"
    r"(?P<oline>\[[^\]\n]+s\]\s+Options:\s+)(?P<options>.*?)(?=\n\[|\n</active_query>|\n</queries>|$)",
    re.DOTALL,
)
QUERY_INSTRUCTION_RE = re.compile(
    r"(?P<qline>\[[^\]\n]+s\]\s+Q:\s+(?P<question>.*?)\n)"
    r"(?P<body>.*?)(?P<iprefix>\[[^\]\n]+s\]\s+Answer format: one letter only \()"
    r"(?P<letters>[^)]*)(?P<suffix>\)\.)",
    re.DOTALL,
)
RESPONSE_HISTORY_RE = re.compile(
    r"(<response_history>\s*)(.*?)(\s*</response_history>)",
    re.DOTALL,
)
RESPONSE_HISTORY_LINE_RE = re.compile(r"^(\s*\[[^\]\n]+s\]\s+A:\s*)(.*?)\s*$")


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _strip_label(option: str) -> str:
    m = OPTION_RE.match(str(option or ""))
    return (m.group(3) if m else str(option or "")).strip()


def _label_options(options: List[str]) -> List[str]:
    return [
        f"{letter}) {_strip_label(opt)}"
        for letter, opt in zip(LETTERS[:len(options)], options)
    ]


def _correct_text(options: List[str], correct_option: str) -> str | None:
    correct_option = str(correct_option or "").strip().upper()
    valid = LETTERS[:len(options)]
    if correct_option not in valid:
        return None
    idx = valid.index(correct_option)
    if idx >= len(options):
        return None
    return _strip_label(options[idx])


def _accepted_answers(options: List[str], correct_option: str) -> List[str]:
    correct_option = str(correct_option or "").strip().upper()
    text = _correct_text(options, correct_option) or ""
    out = []
    if correct_option:
        out.append(correct_option)
    if correct_option and text:
        out.append(f"{correct_option}) {text}")
    if text:
        out.append(text)
    seen = set()
    return [x for x in out if x and not (x.lower() in seen or seen.add(x.lower()))]


def _target_for_style(style: str, options: List[str], correct_option: str) -> str | None:
    correct_option = str(correct_option or "").strip().upper()
    text = _correct_text(options, correct_option) or ""
    if style == "letter_only":
        return correct_option
    if style == "letter_plus_text":
        return f"{correct_option}) {text}" if text else correct_option
    if style == "text_only":
        return text
    return None


def _style_for_obj(obj: Dict[str, Any]) -> str:
    return "letter_only"


def _letter_list(options: List[str]) -> str:
    labels = LETTERS[:max(2, min(len(options), 26))]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _instruction_for_style(style: str, options: List[str]) -> str:
    if style == "text_only":
        return "Answer format: answer text only, no option letter."
    if style == "letter_plus_text":
        return "Answer format: letter plus option text, e.g. A) option text."
    return f"Answer format: one letter only ({_letter_list(options)})."


def _patch_per_emit_answers(obj: Dict[str, Any], target: str | None) -> bool:
    """Keep chunk-level MC gold answers aligned after option relabeling."""
    if not target:
        return False
    emits = obj.get("per_emit_answers")
    if not isinstance(emits, list):
        return False
    changed = False
    for emit in emits:
        if not isinstance(emit, dict):
            continue
        current = str(emit.get("value") or "").strip()
        if current and current != target:
            emit["value"] = target
            changed = True
    return changed


def _patch_gold_emits(obj: Dict[str, Any], correct_option: str) -> bool:
    """Keep raw card gold emits aligned with the new correct letter."""
    emits = obj.get("gold_emits")
    if not isinstance(emits, list):
        return False
    changed = False
    for emit in emits:
        if not isinstance(emit, dict):
            continue
        current = str(emit.get("value") or "").strip()
        if current and current != correct_option:
            emit["value"] = correct_option
            changed = True
    return changed


def _patch_query_answers(obj: Dict[str, Any], target: str | None) -> bool:
    """Keep rendered response_history source answers aligned with MC relabeling."""
    if not target:
        return False
    answers = obj.get("answers")
    if not isinstance(answers, list):
        return False
    changed = False
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        current = str(answer.get("text") or "").strip()
        if current and current != target:
            answer["text"] = target
            changed = True
    return changed


def _patch_answer_payload(text: str, target: str | None) -> Tuple[str, bool]:
    if (
        not target
        or not isinstance(text, str)
        or not ("<answer>" in text or "<response>" in text)
    ):
        return text, False

    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        tag = match.group(1)
        current = match.group(2)
        if not current.strip():
            return match.group(0)
        if current.strip() == target:
            return match.group(0)
        changed = True
        return f"<{tag}>{target}</{tag}>"

    return ANSWER_RE.sub(repl, text), changed


def _patch_response_history_payload(text: str, target: str | None) -> Tuple[str, bool]:
    if not target or not isinstance(text, str) or "<response_history>" not in text:
        return text, False
    changed = False

    def repl_block(match: re.Match[str]) -> str:
        nonlocal changed
        body = match.group(2)
        lines = []
        for line in body.splitlines():
            m = RESPONSE_HISTORY_LINE_RE.match(line)
            if not m:
                lines.append(line)
                continue
            current = m.group(2).strip()
            if current and current != target:
                line = f"{m.group(1)}{target}"
                changed = True
            lines.append(line)
        return f"{match.group(1)}{chr(10).join(lines)}{match.group(3)}"

    return RESPONSE_HISTORY_RE.sub(repl_block, text), changed


def _rebalance_options(
    options: List[str], old_correct: str, new_correct: str
) -> Tuple[List[str], str] | None:
    old_correct = str(old_correct or "").strip().upper()
    new_correct = str(new_correct or "").strip().upper()
    valid = LETTERS[:len(options)]
    if len(options) < 2 or old_correct not in valid or new_correct not in valid:
        return None
    correct = _correct_text(options, old_correct)
    if not correct:
        return None
    distractors = [
        _strip_label(opt)
        for i, opt in enumerate(options)
        if i != valid.index(old_correct)
    ]
    new_plain: List[str] = []
    d_iter = iter(distractors)
    for letter in valid:
        if letter == new_correct:
            new_plain.append(correct)
        else:
            new_plain.append(next(d_iter))
    return _label_options(new_plain), new_correct


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _question_key(video_id: str, card_id: str, question: str) -> Tuple[str, str, str]:
    return str(video_id or ""), str(card_id or ""), str(question or "")


def build_mapping(final_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Assign balanced correct letters per split and return per-question mapping."""
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for split, filename in TRAJ_FILES.items():
        path = final_dir / filename
        if not path.exists():
            continue
        records = []
        for row in _iter_jsonl(path):
            video_id = row.get("video_id", "")
            for q in row.get("questions") or []:
                if q.get("answer_form") != "multiple_choice":
                    continue
                options = list(q.get("options") or [])
                old_correct = q.get("correct_option", "")
                key = _question_key(video_id, q.get("card_id", ""), q.get("question", ""))
                if key in mapping:
                    continue
                if _correct_text(options, old_correct) is None:
                    continue
                records.append((key, options, old_correct))

        records.sort(key=lambda r: _stable_hash("|".join(r[0])))
        for i, (key, options, old_correct) in enumerate(records):
            valid = LETTERS[:len(options)]
            new_correct = valid[i % len(valid)]
            reb = _rebalance_options(options, old_correct, new_correct)
            if not reb:
                continue
            new_options, corrected = reb
            mapping[key] = {
                "split": split,
                "old_correct_option": old_correct,
                "correct_option": corrected,
                "options": new_options,
                "answer_style": _style_for_obj(q),
            }
    return mapping


def _lookup(
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
    video_id: str,
    obj: Dict[str, Any],
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]] | None = None,
):
    card_id = str(obj.get("card_id") or "")
    question = str(obj.get("question", ""))
    if card_id:
        key = _question_key(video_id, card_id, question)
        hit = mapping.get(key)
        if hit:
            return hit

    # Some legacy nested query states do not carry card_id. Fall back to the
    # question text only when it is unique for this video; otherwise different
    # cards with identical wording can receive each other's option order.
    q = str(obj.get("question", ""))
    if by_video_question is not None:
        return by_video_question.get((str(video_id or ""), q))
    matches = [
        value
        for (vid, _cid, mapped_question), value in mapping.items()
        if vid == str(video_id or "") and mapped_question == q
    ]
    return matches[0] if len(matches) == 1 else None


def _patch_question(
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
    video_id: str,
    obj: Dict[str, Any],
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]] | None = None,
) -> bool:
    hit = _lookup(mapping, video_id, obj, by_video_question)
    if not hit:
        return False
    obj["options"] = list(hit["options"])
    obj["correct_option"] = hit["correct_option"]
    correct_text = _correct_text(hit["options"], hit["correct_option"])
    changed = True
    if correct_text:
        obj["canonical_answer"] = correct_text
        obj["gold_answer"] = correct_text
        obj["correct_answer_text"] = correct_text
        obj["accepted_answers"] = _accepted_answers(hit["options"], hit["correct_option"])
    style = _style_for_obj(obj)
    target = _target_for_style(style, hit["options"], hit["correct_option"])
    if obj.get("answer_style"):
        obj["answer_style"] = style
    if obj.get("answer_instruction"):
        obj["answer_instruction"] = _instruction_for_style(style, hit["options"])
    if target:
        obj["sft_answer"] = target
    changed = _patch_per_emit_answers(obj, target) or changed
    changed = _patch_gold_emits(obj, hit["correct_option"]) or changed
    changed = _patch_query_answers(obj, target) or changed
    return changed


def _patch_query_text(text: str, video_id: str, by_video_question: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    def repl(match: re.Match[str]) -> str:
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if not hit:
            return match.group(0)
        return f"{match.group('qline')}{match.group('oline')}{' '.join(hit['options'])}"

    def repl_instruction(match: re.Match[str]) -> str:
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if not hit:
            return match.group(0)
        return (
            f"{match.group('qline')}{match.group('body')}"
            f"{match.group('iprefix')}{_letter_list(hit['options'])}{match.group('suffix')}"
        )

    return QUERY_INSTRUCTION_RE.sub(
        repl_instruction,
        QUERY_BLOCK_RE.sub(repl, text),
    )


def _target_from_query_text(
    text: str,
    video_id: str,
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]],
) -> str | None:
    if not isinstance(text, str):
        return None
    for match in QUERY_BLOCK_RE.finditer(text):
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if hit:
            return _target_for_style(
                str(hit.get("answer_style") or "letter_only"),
                hit["options"],
                hit["correct_option"],
            )
    return None


def _answer_values(text: str) -> List[str]:
    if not isinstance(text, str) or not ("<answer>" in text or "<response>" in text):
        return []
    return [m.group(2).strip() for m in ANSWER_RE.finditer(text) if m.group(2).strip()]


def _query_text_errors(
    text: str,
    video_id: str,
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]],
    location: str,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(text, str):
        return errors
    for match in QUERY_BLOCK_RE.finditer(text):
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if not hit:
            continue
        expected = " ".join(hit["options"])
        actual = " ".join(str(match.group("options") or "").split())
        if actual != expected:
            errors.append(
                f"{location}: active-query options mismatch for {question!r}: "
                f"{actual!r} != {expected!r}"
            )
    for match in QUERY_INSTRUCTION_RE.finditer(text):
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if not hit:
            continue
        expected = _letter_list(hit["options"])
        actual = " ".join(str(match.group("letters") or "").split())
        if actual != expected:
            errors.append(
                f"{location}: answer-format letters mismatch for {question!r}: "
                f"{actual!r} != {expected!r}"
            )
    return errors


def _validate_question_obj(
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]],
    video_id: str,
    obj: Dict[str, Any],
    location: str,
) -> List[str]:
    hit = _lookup(mapping, video_id, obj, by_video_question)
    if not hit:
        return []
    errors: List[str] = []
    options = list(obj.get("options") or [])
    correct_option = str(obj.get("correct_option") or "").strip().upper()
    expected_options = list(hit["options"])
    expected_correct = str(hit["correct_option"])
    correct_text = _correct_text(expected_options, expected_correct) or ""
    style = _style_for_obj(obj)
    target = _target_for_style(style, expected_options, expected_correct)

    if options != expected_options:
        errors.append(f"{location}: options not patched for MC card")
    if correct_option != expected_correct:
        errors.append(
            f"{location}: correct_option={correct_option!r}, expected {expected_correct!r}"
        )
    for key in ("canonical_answer", "gold_answer", "correct_answer_text"):
        if key in obj and str(obj.get(key) or "").strip() != correct_text:
            errors.append(
                f"{location}: {key}={obj.get(key)!r}, expected {correct_text!r}"
            )
    accepted = [str(x).strip() for x in obj.get("accepted_answers") or []]
    expected_accepted = _accepted_answers(expected_options, expected_correct)
    if accepted and accepted != expected_accepted:
        errors.append(
            f"{location}: accepted_answers={accepted!r}, expected {expected_accepted!r}"
        )
    if target and "sft_answer" in obj and str(obj.get("sft_answer") or "").strip() != target:
        errors.append(f"{location}: sft_answer={obj.get('sft_answer')!r}, expected {target!r}")

    emits = obj.get("per_emit_answers")
    if target and isinstance(emits, list):
        for i, emit in enumerate(emits):
            if not isinstance(emit, dict):
                continue
            value = str(emit.get("value") or "").strip()
            if value and value != target:
                errors.append(
                    f"{location}: per_emit_answers[{i}]={value!r}, expected {target!r}"
                )
    gold = obj.get("gold_emits")
    if isinstance(gold, list):
        for i, emit in enumerate(gold):
            if not isinstance(emit, dict):
                continue
            value = str(emit.get("value") or "").strip()
            if value and value != expected_correct:
                errors.append(
                    f"{location}: gold_emits[{i}]={value!r}, expected {expected_correct!r}"
                )
    answers = obj.get("answers")
    if target and isinstance(answers, list):
        for i, answer in enumerate(answers):
            if not isinstance(answer, dict):
                continue
            value = str(answer.get("text") or "").strip()
            if value and value != target:
                errors.append(
                    f"{location}: answers[{i}]={value!r}, expected {target!r}"
                )
    return errors


def _validate_row(
    row: Dict[str, Any],
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]],
    location: str,
    parent_video_id: str = "",
) -> List[str]:
    errors: List[str] = []
    video_id = str(row.get("video_id") or parent_video_id or "")
    row_hit = None
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        row_hit = _lookup(mapping, video_id, metadata, by_video_question)
        errors.extend(_validate_question_obj(mapping, by_video_question, video_id, metadata, f"{location}.metadata"))

    for i, q in enumerate(row.get("questions") or []):
        if isinstance(q, dict):
            errors.extend(_validate_question_obj(mapping, by_video_question, video_id, q, f"{location}.questions[{i}]"))

    input_obj = row.get("input")
    if isinstance(input_obj, dict):
        for i, q in enumerate(input_obj.get("queries") or []):
            if isinstance(q, dict):
                errors.extend(_validate_question_obj(mapping, by_video_question, video_id, q, f"{location}.input.queries[{i}]"))

    if row_hit:
        target = _target_for_style(
            _style_for_obj(metadata or {}),
            row_hit["options"],
            row_hit["correct_option"],
        )
        if target:
            for key in ("output", "v12_assistant_turn_2"):
                for value in _answer_values(row.get(key, "")):
                    if value != target:
                        errors.append(
                            f"{location}.{key}: answer={value!r}, expected {target!r}"
                        )

    messages = row.get("messages")
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for pi, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                text = part.get("text", "")
                msg_loc = f"{location}.messages[{mi}].content[{pi}]"
                errors.extend(_query_text_errors(text, video_id, by_video_question, msg_loc))
                history_target = _target_from_query_text(text, video_id, by_video_question)
                if row_hit:
                    target = _target_for_style(
                        _style_for_obj(metadata or {}),
                        row_hit["options"],
                        row_hit["correct_option"],
                    )
                    history_target = history_target or target
                    if target:
                        for value in _answer_values(text):
                            if value != target:
                                errors.append(
                                    f"{msg_loc}: answer={value!r}, expected {target!r}"
                                )
                if history_target:
                    for block in RESPONSE_HISTORY_RE.finditer(text):
                        for line in block.group(2).splitlines():
                            m = RESPONSE_HISTORY_LINE_RE.match(line)
                            if not m:
                                continue
                            value = m.group(2).strip()
                            if value and value != history_target:
                                errors.append(
                                    f"{msg_loc}: response_history={value!r}, "
                                    f"expected {history_target!r}"
                                )

    for i, sample in enumerate(row.get("samples") or []):
        if isinstance(sample, dict):
            errors.extend(
                _validate_row(
                    sample,
                    mapping,
                    by_video_question,
                    f"{location}.samples[{i}]",
                    parent_video_id=video_id,
                )
            )
    return errors


def patch_row(
    row: Dict[str, Any],
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]],
    parent_video_id: str = "",
) -> int:
    changed = 0
    video_id = str(row.get("video_id") or parent_video_id or "")
    row_hit = None

    if isinstance(row.get("questions"), list):
        for q in row.get("questions") or []:
            changed += int(_patch_question(mapping, video_id, q, by_video_question))

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        if row.get("card_id") and not metadata.get("card_id"):
            metadata["card_id"] = row.get("card_id")
            changed += 1
        row_hit = _lookup(mapping, video_id, metadata, by_video_question)
        changed += int(_patch_question(mapping, video_id, metadata, by_video_question))
        if row_hit:
            target = _target_for_style(
                _style_for_obj(metadata),
                row_hit["options"],
                row_hit["correct_option"],
            )
            for key in ("output", "v12_assistant_turn_2"):
                if isinstance(row.get(key), str):
                    new_text, did = _patch_answer_payload(row[key], target)
                    if did:
                        row[key] = new_text
                        changed += 1

    input_obj = row.get("input")
    if isinstance(input_obj, dict):
        for q in input_obj.get("queries") or []:
            if isinstance(q, dict):
                changed += int(_patch_question(mapping, video_id, q, by_video_question))

    if isinstance(row.get("samples"), list):
        for sample in row.get("samples") or []:
            if isinstance(sample, dict):
                changed += patch_row(
                    sample, mapping, by_video_question, parent_video_id=video_id
                )

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                old = part.get("text", "")
                new = _patch_query_text(old, video_id, by_video_question)
                if row_hit:
                    target = _target_for_style(
                        _style_for_obj(metadata or {}),
                        row_hit["options"],
                        row_hit["correct_option"],
                    )
                    new, did_answer = _patch_answer_payload(new, target)
                    changed += int(did_answer)
                else:
                    target = _target_from_query_text(new, video_id, by_video_question)
                new, did_history = _patch_response_history_payload(new, target)
                changed += int(did_history)
                if new != old:
                    part["text"] = new
                    changed += 1
    return changed


def _unique_by_video_question(
    mapping: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_video_question_candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (video_id, _card_id, question), value in mapping.items():
        by_video_question_candidates[(video_id, question)].append(value)
    return {
        key: values[0]
        for key, values in by_video_question_candidates.items()
        if len(values) == 1
    }


def apply_mapping(final_dir: Path, mapping: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    by_video_question = _unique_by_video_question(mapping)
    changed_by_file = {}
    for filename in ALL_JSONL_FILES:
        path = final_dir / filename
        if not path.exists():
            continue
        rows = []
        changed = 0
        for row in _iter_jsonl(path):
            changed += patch_row(row, mapping, by_video_question)
            rows.append(row)
        if changed:
            _write_jsonl(path, rows)
        changed_by_file[filename] = changed
    return changed_by_file


def validate_mapping(final_dir: Path, mapping: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    by_video_question = _unique_by_video_question(mapping)
    errors: List[str] = []
    rows_checked = 0
    for filename in ALL_JSONL_FILES:
        path = final_dir / filename
        if not path.exists():
            continue
        for line_no, row in enumerate(_iter_jsonl(path), start=1):
            rows_checked += 1
            errors.extend(
                _validate_row(
                    row,
                    mapping,
                    by_video_question,
                    f"{filename}:{line_no}",
                )
            )
            if len(errors) > 200:
                break
        if len(errors) > 200:
            break
    return {
        "rows_checked": rows_checked,
        "errors": errors,
        "n_errors": len(errors),
    }


def summarize_mapping(mapping: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    old_by_split: Dict[str, Counter] = defaultdict(Counter)
    new_by_split: Dict[str, Counter] = defaultdict(Counter)
    for value in mapping.values():
        split = value["split"]
        old_by_split[split][value["old_correct_option"]] += 1
        new_by_split[split][value["correct_option"]] += 1
    return {
        "n_questions": len(mapping),
        "old_correct_option_by_split": {
            split: {k: old_by_split[split].get(k, 0) for k in LETTERS[:5]}
            for split in sorted(old_by_split)
        },
        "new_correct_option_by_split": {
            split: {k: new_by_split[split].get(k, 0) for k in LETTERS[:5]}
            for split in sorted(new_by_split)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="data/agent_v5/final")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    mapping = build_mapping(final_dir)
    changed_by_file = apply_mapping(final_dir, mapping)
    validation = validate_mapping(final_dir, mapping)
    report = summarize_mapping(mapping)
    report["changed_by_file"] = changed_by_file
    report["validation"] = {
        "rows_checked": validation["rows_checked"],
        "n_errors": validation["n_errors"],
        "errors_preview": validation["errors"][:20],
    }
    out = Path(args.report) if args.report else final_dir / "mc_rebalance_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if validation["errors"]:
        raise SystemExit(
            "MC rebalance validation failed:\n"
            + "\n".join(validation["errors"][:20])
        )


if __name__ == "__main__":
    main()
