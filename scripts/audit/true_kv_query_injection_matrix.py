#!/usr/bin/env python3
"""Compare active-query injection policies under local true-KV streaming.

This is a lightweight HF audit path, intentionally independent of RL/verl. It
uses ``StreamingWindowInferenceEngine`` directly, advances rendered pass5
trajectory rows turn by turn, and mutates only the current user turn to compare
query injection frequency/position.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.local_hf_kv_visual_probe import (  # noqa: E402
    load_model_for_stream,
    make_engine,
    normalize_video_inputs,
)
from thinkstream.data.agent_protocol import (  # noqa: E402
    append_query_answer_with_timing,
    canonical_answer_instruction,
    format_queries_block,
    query_is_complete,
)
from thinkstream.data.stream_data_processor import compute_position_ids  # noqa: E402
from thinkstream.models.agent_loop import _parse_agent_output  # noqa: E402


ACTIVE_RE = re.compile(r"\n?<active_query>\s*.*?\s*</active_query>", re.DOTALL)
HISTORY_RE = re.compile(r"\n?<response_history>\s*.*?\s*</response_history>", re.DOTALL)
LEGACY_QUERY_RE = re.compile(r"\n?<active_query>\s*.*?\s*</active_query>", re.DOTALL)
T_RE = re.compile(r"<t=(\d+)>")


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _resolve_msg(msg: Dict[str, Any], *, frames_root: Path, video_id: str) -> Dict[str, Any]:
    msg = copy.deepcopy(msg)
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    for item in content:
        if not (isinstance(item, dict) and item.get("type") == "video"):
            continue
        video = item.get("video")
        if not isinstance(video, list):
            continue
        fixed = []
        for frame in video:
            frame_path = Path(str(frame))
            if frame_path.is_absolute():
                fixed.append(str(frame_path))
            elif str(frame).startswith("data/"):
                fixed.append(str((REPO_ROOT / frame_path).resolve()))
            else:
                direct = frames_root / frame_path
                if direct.exists():
                    fixed.append(str(direct.resolve()))
                else:
                    fixed.append(str((frames_root / video_id / frame_path).resolve()))
        item["video"] = fixed
    return msg


def _assistant_pairs(messages: List[Dict[str, Any]], limit: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "system":
            i += 1
            continue
        if (
            messages[i].get("role") == "user"
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "assistant"
        ):
            pairs.append((i, i + 1))
            i += 2
            if len(pairs) >= limit:
                break
        else:
            i += 1
    return pairs


def _user_chunk(message: Dict[str, Any], default: int) -> int:
    match = T_RE.search(_text_from_content(message.get("content")))
    return int(match.group(1)) if match else default


def _gold_action(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        name = (((tool_calls[0] or {}).get("function") or {}).get("name") or "").strip()
        return "recall" if name == "recall" else (name or "tool_call")
    parsed = _parse_agent_output(_text_from_content(message.get("content")))
    return str(parsed.get("action") or "")


def _parse_pred_action(text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse current canonical tags, then fall back to old Streamo tags.

    The fallback is used only for this audit so we can compare query-injection
    policies even when an older SFT checkpoint still emits ``<silent>`` /
    ``<response>``. The ``old_tag`` flag is reported separately.
    """
    parsed = _parse_agent_output(text)
    if parsed.get("action"):
        return parsed, False
    cleaned = re.sub(r"<\|im_end\|>\s*$", "", str(text or "").strip())
    if re.search(r"<silent>\s*$", cleaned, re.IGNORECASE):
        parsed = dict(parsed)
        parsed["action"] = "silent"
        parsed["payload"] = {}
        return parsed, True
    m = re.search(r"<response>\s*(.*?)\s*$", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        parsed = dict(parsed)
        parsed["action"] = "response"
        parsed["payload"] = {"response": m.group(1).strip()}
        return parsed, True
    return parsed, False


def _load_trajectories(path: Path, *, batch_size: int, turns: int, skip: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matched = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("trajectory_type") not in {None, "from_start"}:
                continue
            if not row.get("questions_in_segment"):
                continue
            pairs = _assistant_pairs(row.get("messages") or [], turns)
            if len(pairs) < turns:
                continue
            if matched < skip:
                matched += 1
                continue
            row["_assistant_pairs"] = pairs
            rows.append(row)
            if len(rows) >= batch_size:
                break
    if len(rows) < batch_size:
        raise RuntimeError(
            f"only found {len(rows)} trajectories with >= {turns} turns and questions in {path}"
        )
    return rows


def _query_from_question(q: Dict[str, Any]) -> Dict[str, Any]:
    answer_chunks = []
    for value in q.get("answer_chunks") or q.get("expected_answer_chunks") or []:
        try:
            answer_chunks.append(int(value))
        except (TypeError, ValueError):
            pass
    ask = int(q.get("ask_chunk", q.get("ask_time", 0)) or 0)
    return {
        "question": str(q.get("question") or ""),
        "ask_time": ask,
        "answer_chunks": sorted(set(answer_chunks)) or [ask],
        "per_emit_answers": list(q.get("per_emit_answers") or []),
        "options": list(q.get("options") or []),
        "answer_form": str(q.get("answer_form") or ""),
        "answer_style": str(q.get("answer_style") or ""),
        "answer_instruction": canonical_answer_instruction(q),
        "status": "open",
        "answers": [],
        "_card_id": q.get("card_id", ""),
        "_family": q.get("family", ""),
    }


def _make_query_text(q: Dict[str, Any], *, as_user_input: bool = False) -> str:
    if as_user_input:
        prefix = f"[{int(float(q.get('ask_time', 0)))}s]"
        lines = [f"{prefix} Q: {q.get('question', '')}"]
        if q.get("options"):
            lines.append(f"{prefix} Options: " + " ".join(str(x) for x in q.get("options") or []))
        inst = canonical_answer_instruction(q)
        if inst:
            lines.append(f"{prefix} {inst}")
        return "<user_input>\n" + "\n".join(lines) + "\n</user_input>"
    return format_queries_block([q])


def _strip_query_blocks(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in content:
        if not (isinstance(item, dict) and item.get("type") == "text"):
            out.append(copy.deepcopy(item))
            continue
        text = str(item.get("text", ""))
        text = ACTIVE_RE.sub("", text)
        text = HISTORY_RE.sub("", text)
        text = LEGACY_QUERY_RE.sub("", text)
        if text.strip():
            new_item = copy.deepcopy(item)
            new_item["text"] = text
            out.append(new_item)
    return out


def _insert_query_block(
    content: List[Dict[str, Any]],
    block: str,
    *,
    position: str,
) -> List[Dict[str, Any]]:
    if not block:
        return content
    item = {"type": "text", "text": "\n" + block}
    if position == "after_visual":
        last_video = -1
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "video":
                last_video = i
        if last_video >= 0:
            return content[: last_video + 1] + [item] + content[last_video + 1 :]
        return content + [item]
    if position == "before_visual":
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "video":
                return content[:i] + [item] + content[i:]
        return [item] + content
    if position == "front":
        front = dict(item)
        front["text"] = block
        return [front] + content
    return content + [item]


def _strategy_parts(strategy: str) -> Tuple[str, str, bool]:
    if strategy == "ask_only_after_visual":
        return "original", "after_visual", False
    if strategy == "ask_only_before_visual":
        return "ask_only", "before_visual", False
    if strategy == "every_step_after_visual":
        return "every_step", "after_visual", False
    if strategy == "every_step_before_visual":
        return "every_step", "before_visual", False
    if strategy == "window8_after_visual":
        return "window8", "after_visual", False
    if strategy == "user_input_every_step_front":
        return "every_step", "front", True
    raise ValueError(f"unknown strategy: {strategy}")


def _select_prompt_query(query_log: List[Dict[str, Any]], chunk: int, mode: str) -> Optional[Dict[str, Any]]:
    open_queries = [q for q in query_log if str(q.get("status", "open")) == "open"]
    if not open_queries:
        return None
    q = max(open_queries, key=lambda x: float(x.get("ask_time", 0)))
    ask = int(float(q.get("ask_time", 0)))
    answer_chunks = [int(c) for c in q.get("answer_chunks") or []]
    final = max(answer_chunks or [ask])
    if chunk < ask or chunk > final:
        return None
    if mode == "ask_only" and chunk != ask:
        return None
    if mode == "window8" and chunk != ask and chunk > ask + 8:
        return None
    if mode == "every_step":
        return q
    if mode == "ask_only":
        return q
    if mode == "window8":
        return q
    return None


def _prepare_inputs(
    *,
    processor: Any,
    row: Dict[str, Any],
    user_idx: int,
    turn_no: int,
    frames_root: Path,
    strategy: str,
    query_log: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], int, int]:
    messages = row["messages"]
    video_id = str(row.get("video_id") or "")
    base_user = _resolve_msg(messages[user_idx], frames_root=frames_root, video_id=video_id)
    chunk = _user_chunk(base_user, turn_no)
    mode, position, as_user_input = _strategy_parts(strategy)
    injected = 0
    if mode != "original":
        content = _strip_query_blocks(list(base_user.get("content") or []))
        q = _select_prompt_query(query_log, chunk, mode)
        if q is not None:
            content = _insert_query_block(
                content,
                _make_query_text(q, as_user_input=as_user_input),
                position=position,
            )
            injected = 1
        base_user["content"] = content

    if turn_no == 0 and messages and messages[0].get("role") == "system":
        turn_messages = [
            _resolve_msg(messages[0], frames_root=frames_root, video_id=video_id),
            base_user,
        ]
    else:
        turn_messages = [base_user]

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    tools = row.get("tools") or None
    if tools:
        template_kwargs["tools"] = tools
    text = processor.apply_chat_template(turn_messages, **template_kwargs)
    _, video_inputs, video_kwargs = process_vision_info(
        turn_messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadata = normalize_video_inputs(video_inputs)
    proc_kwargs: Dict[str, Any] = {
        "text": [text],
        "videos": videos,
        "return_tensors": "pt",
        **(video_kwargs or {}),
    }
    if video_metadata is not None:
        proc_kwargs["video_metadata"] = video_metadata
    inputs = processor(**proc_kwargs)
    inputs_for_rope = dict(inputs)
    inputs_for_rope["video_chunk_size"] = 1.0
    inputs["position_ids"] = compute_position_ids(inputs_for_rope, processor, "qwen3vl")
    return text, inputs, chunk, injected


def _to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: (inputs.get(key).to(device) if inputs.get(key) is not None else None)
        for key in [
            "input_ids",
            "attention_mask",
            "position_ids",
            "pixel_values_videos",
            "video_grid_thw",
        ]
    }


def _register_new_queries(row: Dict[str, Any], query_log: List[Dict[str, Any]], chunk: int) -> None:
    seen = {q.get("_card_id") for q in query_log}
    for raw in row.get("questions_in_segment") or []:
        ask_value = raw.get("ask_chunk", raw.get("ask_time", -1))
        if ask_value is None:
            ask_value = -1
        if int(ask_value) != int(chunk):
            continue
        q = _query_from_question(raw)
        if q.get("_card_id") in seen:
            continue
        for old in query_log:
            if not query_is_complete(old):
                old["status"] = "replaced"
        query_log.append(q)
        seen.add(q.get("_card_id"))


def _record_answer(query_log: List[Dict[str, Any]], answer: str, chunk: int) -> None:
    if not answer:
        return
    for q in reversed(query_log):
        if str(q.get("status", "open")) == "open":
            append_query_answer_with_timing(q, answer, chunk)
            q["status"] = "answered" if query_is_complete(q) else "open"
            return


def run_strategy(args: argparse.Namespace, model: Any, processor: Any, strategy: str) -> Dict[str, Any]:
    device = f"cuda:{args.gpu}"
    load_args = SimpleNamespace(
        model=args.model,
        model_type=args.model_type,
        pixel_profile=args.pixel_profile,
        max_len=args.max_len,
        kv_window=args.kv_window,
        batch_size=1,
    )
    engine = make_engine(load_args, model, processor, device)
    rows = _load_trajectories(
        Path(args.jsonl),
        batch_size=args.num_trajectories,
        turns=args.turns,
        skip=args.skip_trajectories,
    )
    frames_root = Path(args.frames_root)
    all_rows: List[Dict[str, Any]] = []
    counters = Counter()
    by_gold = defaultdict(Counter)
    prompt_tokens: List[int] = []
    cache_lens: List[int] = []
    injected_count = 0
    total_decode_sec = 0.0
    total_generated = 0

    for slot, row in enumerate(rows):
        engine.reset()
        query_log: List[Dict[str, Any]] = []
        for turn_no, (user_idx, assistant_idx) in enumerate(row["_assistant_pairs"][: args.turns]):
            chunk = _user_chunk(row["messages"][user_idx], turn_no)
            _register_new_queries(row, query_log, chunk)
            _text, inputs, chunk, injected = _prepare_inputs(
                processor=processor,
                row=row,
                user_idx=user_idx,
                turn_no=turn_no,
                frames_root=frames_root,
                strategy=strategy,
                query_log=query_log,
            )
            injected_count += injected
            keep = _to_device(inputs, device)
            t0 = time.time()
            generated = engine.generate(
                **keep,
                max_new_tokens=args.max_new_tokens,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                repetition_penalty=1.0,
            )[0]
            latency = time.time() - t0
            total_decode_sec += latency
            total_generated += int(generated.numel())
            pred_text = processor.tokenizer.decode(
                generated.tolist(),
                skip_special_tokens=False,
            ).strip()
            pred, old_tag = _parse_pred_action(pred_text)
            gold_action = _gold_action(row["messages"][assistant_idx])
            pred_action = str(pred.get("action") or "")
            if pred_action == "response":
                _record_answer(query_log, str((pred.get("payload") or {}).get("response", "")), chunk)
            counters["total"] += 1
            counters["action_match"] += int(pred_action == gold_action)
            counters[f"gold:{gold_action}"] += 1
            counters[f"pred:{pred_action}"] += 1
            counters["format_error"] += int(bool(pred.get("format_error")))
            counters["old_tag_fallback"] += int(old_tag)
            counters["response_tp"] += int(pred_action == "response" and gold_action == "response")
            counters["response_fp"] += int(pred_action == "response" and gold_action != "response")
            counters["response_fn"] += int(pred_action != "response" and gold_action == "response")
            counters["silent_tp"] += int(pred_action == "silent" and gold_action == "silent")
            by_gold[gold_action][pred_action] += 1
            prompt_tokens.append(int(inputs["attention_mask"][0].sum().item()))
            cache_lens.append(int(engine.decoder.cache.cache_seqlens[0, 0].item()))
            if args.keep_examples and len(all_rows) < args.keep_examples:
                all_rows.append({
                    "slot": slot,
                    "video_id": row.get("video_id"),
                    "turn": turn_no,
                    "chunk": chunk,
                    "gold_action": gold_action,
                    "pred_action": pred_action,
                    "format_error": pred.get("format_error", ""),
                    "old_tag_fallback": old_tag,
                    "prompt_tokens": prompt_tokens[-1],
                    "cache_len": cache_lens[-1],
                    "injected": injected,
                    "pred_text": pred_text[:600],
                })

    tp = counters["response_tp"]
    fp = counters["response_fp"]
    fn = counters["response_fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "strategy": strategy,
        "model": args.model,
        "source": args.jsonl,
        "num_trajectories": args.num_trajectories,
        "turns_per_trajectory": args.turns,
        "max_new_tokens": args.max_new_tokens,
        "counts": dict(counters),
        "by_gold": {k: dict(v) for k, v in by_gold.items()},
        "action_match_rate": counters["action_match"] / max(1, counters["total"]),
        "response_precision": precision,
        "response_recall": recall,
        "response_f1": f1,
        "format_error_rate": counters["format_error"] / max(1, counters["total"]),
        "query_injected_turns": injected_count,
        "avg_prompt_tokens": sum(prompt_tokens) / max(1, len(prompt_tokens)),
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "avg_cache_len": sum(cache_lens) / max(1, len(cache_lens)),
        "max_cache_len": max(cache_lens) if cache_lens else 0,
        "total_generated_tokens": total_generated,
        "total_decode_sec": total_decode_sec,
        "tokens_per_sec": total_generated / max(total_decode_sec, 1e-6),
        "examples": all_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategies", default="ask_only_after_visual,every_step_after_visual,every_step_before_visual,window8_after_visual,user_input_every_step_front")
    ap.add_argument("--num-trajectories", type=int, default=4)
    ap.add_argument("--turns", type=int, default=32)
    ap.add_argument("--skip-trajectories", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-type", default="qwen3vl")
    ap.add_argument("--pixel-profile", default="low")
    ap.add_argument("--max-len", type=int, default=49152)
    ap.add_argument("--kv-window", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--keep-examples", type=int, default=16)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    load_args = SimpleNamespace(
        model=args.model,
        model_type=args.model_type,
        pixel_profile=args.pixel_profile,
        max_len=args.max_len,
        kv_window=args.kv_window,
        batch_size=1,
    )
    model, processor = load_model_for_stream(load_args, device)
    processor.tokenizer.padding_side = "right"

    results = []
    for strategy in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        with torch.inference_mode():
            result = run_strategy(args, model, processor, strategy)
        results.append(result)
        print(json.dumps({
            "strategy": result["strategy"],
            "action_match_rate": round(result["action_match_rate"], 4),
            "response_f1": round(result["response_f1"], 4),
            "format_error_rate": round(result["format_error_rate"], 4),
            "query_injected_turns": result["query_injected_turns"],
            "avg_prompt_tokens": round(result["avg_prompt_tokens"], 1),
            "max_cache_len": result["max_cache_len"],
        }, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mode": "true_kv_query_injection_matrix",
        "model": args.model,
        "jsonl": args.jsonl,
        "frames_root": args.frames_root,
        "gpu": args.gpu,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_results": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
