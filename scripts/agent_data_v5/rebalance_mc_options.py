"""Rebalance multiple-choice option letters in final ThinkStream datasets.

The generator emits open-ended answer text, but MC prompts still expose A-D
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


LETTERS = ["A", "B", "C", "D"]
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

OPTION_RE = re.compile(r"^\s*([A-D])[\).]\s*(.*)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
QUERY_BLOCK_RE = re.compile(
    r"(?P<qline>\[[^\]\n]+s\]\s+Q:\s+(?P<question>.*?)\n)"
    r"(?P<oline>\[[^\]\n]+s\]\s+Options:\s+)(?P<options>.*?)(?=\n\[|\n</active_query>|\n</queries>|$)",
    re.DOTALL,
)


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _strip_label(option: str) -> str:
    m = OPTION_RE.match(str(option or ""))
    return (m.group(2) if m else str(option or "")).strip()


def _label_options(options: List[str]) -> List[str]:
    return [f"{letter}) {_strip_label(opt)}" for letter, opt in zip(LETTERS, options)]


def _correct_text(options: List[str], correct_option: str) -> str | None:
    if correct_option not in LETTERS or len(options) != 4:
        return None
    idx = LETTERS.index(correct_option)
    if idx >= len(options):
        return None
    return _strip_label(options[idx])


def _accepted_answers(options: List[str], correct_option: str) -> List[str]:
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
    text = _correct_text(options, correct_option) or ""
    if style == "letter_only":
        return correct_option
    if style == "letter_plus_text":
        return f"{correct_option}) {text}" if text else correct_option
    if style == "text_only":
        return text
    return None


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


def _patch_answer_payload(text: str, target: str | None) -> Tuple[str, bool]:
    if not target or not isinstance(text, str) or "<answer>" not in text:
        return text, False

    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        current = match.group(1)
        if not current.strip():
            return match.group(0)
        if current.strip() == target:
            return match.group(0)
        changed = True
        return f"<answer>{target}</answer>"

    return ANSWER_RE.sub(repl, text), changed


def _rebalance_options(
    options: List[str], old_correct: str, new_correct: str
) -> Tuple[List[str], str] | None:
    if len(options) != 4 or old_correct not in LETTERS or new_correct not in LETTERS:
        return None
    correct = _correct_text(options, old_correct)
    if not correct:
        return None
    distractors = [
        _strip_label(opt)
        for i, opt in enumerate(options)
        if i != LETTERS.index(old_correct)
    ]
    new_plain: List[str] = []
    d_iter = iter(distractors)
    for letter in LETTERS:
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
            new_correct = LETTERS[i % len(LETTERS)]
            reb = _rebalance_options(options, old_correct, new_correct)
            if not reb:
                continue
            new_options, corrected = reb
            mapping[key] = {
                "split": split,
                "old_correct_option": old_correct,
                "correct_option": corrected,
                "options": new_options,
            }
    return mapping


def _lookup(mapping: Dict[Tuple[str, str, str], Dict[str, Any]], video_id: str, obj: Dict[str, Any]):
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
    matches = [
        value
        for (vid, _cid, mapped_question), value in mapping.items()
        if vid == str(video_id or "") and mapped_question == q
    ]
    return matches[0] if len(matches) == 1 else None


def _patch_question(mapping: Dict[Tuple[str, str, str], Dict[str, Any]], video_id: str, obj: Dict[str, Any]) -> bool:
    hit = _lookup(mapping, video_id, obj)
    if not hit:
        return False
    obj["options"] = list(hit["options"])
    obj["correct_option"] = hit["correct_option"]
    correct_text = _correct_text(hit["options"], hit["correct_option"])
    if correct_text:
        if obj.get("answer_form") == "multiple_choice" or "gold_answer" in obj:
            obj["gold_answer"] = correct_text
        obj["correct_answer_text"] = correct_text
        obj["accepted_answers"] = _accepted_answers(hit["options"], hit["correct_option"])
    target = _target_for_style(
        obj.get("answer_style", ""),
        hit["options"],
        hit["correct_option"],
    )
    _patch_per_emit_answers(obj, target)
    return True


def _patch_query_text(text: str, video_id: str, by_video_question: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    def repl(match: re.Match[str]) -> str:
        question = match.group("question").strip()
        hit = by_video_question.get((str(video_id or ""), question))
        if not hit:
            return match.group(0)
        return f"{match.group('qline')}{match.group('oline')}{' '.join(hit['options'])}"

    return QUERY_BLOCK_RE.sub(repl, text)


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
            changed += int(_patch_question(mapping, video_id, q))

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        if row.get("card_id") and not metadata.get("card_id"):
            metadata["card_id"] = row.get("card_id")
            changed += 1
        row_hit = _lookup(mapping, video_id, metadata)
        changed += int(_patch_question(mapping, video_id, metadata))
        if row_hit:
            target = _target_for_style(
                metadata.get("answer_style", ""),
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
                changed += int(_patch_question(mapping, video_id, q))

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
                        (metadata or {}).get("answer_style", ""),
                        row_hit["options"],
                        row_hit["correct_option"],
                    )
                    new, did_answer = _patch_answer_payload(new, target)
                    changed += int(did_answer)
                if new != old:
                    part["text"] = new
                    changed += 1
    return changed


def apply_mapping(final_dir: Path, mapping: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    by_video_question_candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (video_id, _card_id, question), value in mapping.items():
        by_video_question_candidates[(video_id, question)].append(value)
    by_video_question: Dict[Tuple[str, str], Dict[str, Any]] = {
        key: values[0]
        for key, values in by_video_question_candidates.items()
        if len(values) == 1
    }

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
            split: {k: old_by_split[split].get(k, 0) for k in LETTERS}
            for split in sorted(old_by_split)
        },
        "new_correct_option_by_split": {
            split: {k: new_by_split[split].get(k, 0) for k in LETTERS}
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
    report = summarize_mapping(mapping)
    report["changed_by_file"] = changed_by_file

    out = Path(args.report) if args.report else final_dir / "mc_rebalance_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
