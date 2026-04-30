"""Eval-dataset adapters: convert benchmark formats into ThinkStream
streaming-eval inputs and score model outputs against ground truth.

ThinkStream's eval pipeline assumes one item per video, processed
chunk-by-chunk. Each adapter implements two methods:

    build_user_input(item, chunk_idx) -> str
        Returns the user_input text to inject at chunk_idx. Returns ""
        for chunks where no question is asked (silent expected).

    score(item, model_outputs) -> dict
        Inspects all chunk outputs and returns:
            {
                "correct": bool,
                "answer_chunk": int | None,
                "answer_text": str | None,
                "delay_chunks": int | None,
                "task": str (optional, for per-task aggregation),
            }

Currently supported:
    OVOBenchAdapter — multiple-choice + realtime field
    OurOpenEndedAdapter — our val.jsonl (any answer_form)
    MCNoTimingAdapter — MC datasets without realtime (e.g., VideoMME)

The eval entry script (scripts/eval/v12_ovo_eval.py) selects the
adapter by --dataset-format flag and routes the rest through vLLM.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


# Match a single-letter answer like "A", "C", or "C." (with trailing period
# or whitespace). Anchored to the start of stripped text so rejects answers
# like "I think it is C" — we want STRICT MC adherence per OVO eval rules.
_LETTER_ANSWER_RE = re.compile(r"^([A-D])\b", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer_text(model_output: str) -> Optional[str]:
    """Extract <answer>...</answer> content. Returns None if missing/empty."""
    m = _ANSWER_TAG_RE.search(model_output or "")
    if not m:
        return None
    txt = m.group(1).strip()
    return txt or None


def _normalize_letter(answer: str) -> Optional[str]:
    """Coerce 'C', 'C.', 'c', ' C ' → 'C'. None if not a clean letter."""
    if not answer:
        return None
    m = _LETTER_ANSWER_RE.match(answer.strip())
    return m.group(1).upper() if m else None


# ===========================================================================
# OVO-Bench (multiple-choice + realtime)
# ===========================================================================


class OVOBenchAdapter:
    """OVO-Bench: 1640 items, 12 tasks, all MC with `realtime` field.

    Schema (from /Users/hzh/Downloads/ovo_bench_new.json):
        {
          "id": int,
          "task": "EPM" | "HLD" | "OJR" | ...,
          "video": "...mp4",
          "realtime": float,             # seconds when question is asked
          "question": "...",
          "options": [4 strings],
          "answer": "..." (option text),
          "gt": int (0-3, index into options),
        }

    Special HLD case: one option is "Unable to answer" — gt may equal it
    when the asked information isn't visible. Correct model behavior is
    to pick that option (NOT silent — silent at the question chunk would
    miss the eval window).
    """

    NAME = "ovo_bench"
    from thinkstream.data.agent_protocol import AGENT_CHUNK_SEC  # canonical (v12.5: 1s/chunk)
    @staticmethod
    def num_chunks_for(item: Dict) -> int:
        """How many chunks to stream before/at the question.

        OVO question is asked at `realtime`. We stream up through 2 chunks
        AFTER (allow late answer = realtime + 4 sec) before scoring as miss.
        Caller may stream further; we score only the first answer >= question
        chunk.
        """
        rt = float(item["realtime"])
        question_chunk = int(rt // OVOBenchAdapter.AGENT_CHUNK_SEC)
        return question_chunk + 3  # +2 chunks late window + 1 inclusive

    @staticmethod
    def question_chunk(item: Dict) -> int:
        return int(float(item["realtime"]) // OVOBenchAdapter.AGENT_CHUNK_SEC)

    @staticmethod
    def build_user_input(item: Dict, chunk_idx: int) -> str:
        """Inject question + options ONLY at the question chunk.

        Format follows the convention pass3c uses for MC training samples:
        question first, then options as 'A. ... B. ... C. ... D. ...'
        on separate lines, then a strict answer instruction. The trailing
        instruction matches what Qwen3-VL pretrain has seen (Hermes/MMLU
        eval style) so model emits a single letter.
        """
        if chunk_idx != OVOBenchAdapter.question_chunk(item):
            return ""

        opts = item.get("options") or []
        if not opts:
            return f"Q: {item['question']}"

        letters = ["A", "B", "C", "D", "E"][: len(opts)]
        opts_block = "\n".join(f"{l}. {t}" for l, t in zip(letters, opts))
        instr = f"Answer with one letter ({'/'.join(letters)})."
        return f"Q: {item['question']}\nOptions:\n{opts_block}\n{instr}"

    @staticmethod
    def score(item: Dict, model_outputs: List[Dict]) -> Dict:
        """Score the first non-empty <answer> at-or-after the question chunk.

        model_outputs: list of {"chunk_idx": int, "text": str} ordered
        ascending by chunk_idx.
        """
        gt_idx = item.get("gt", -1)
        gt_letter = "ABCDE"[gt_idx] if 0 <= gt_idx < 5 else None
        question_chunk = OVOBenchAdapter.question_chunk(item)
        late_window = 2  # chunks past question_chunk that still count

        for out in model_outputs:
            if out["chunk_idx"] < question_chunk:
                continue
            ans = _extract_answer_text(out["text"])
            if ans is None:
                continue  # silent at this chunk
            letter = _normalize_letter(ans)
            delay = out["chunk_idx"] - question_chunk
            in_window = delay <= late_window

            if letter is None:
                # Model emitted free text instead of a letter. Do best-effort
                # substring match against option texts.
                opts = item.get("options") or []
                ans_low = ans.lower()
                matched_idx = None
                for i, opt in enumerate(opts):
                    if opt.lower() in ans_low or ans_low in opt.lower():
                        matched_idx = i
                        break
                correct = (matched_idx == gt_idx) if matched_idx is not None else False
                return {
                    "correct": correct and in_window,
                    "answer_chunk": out["chunk_idx"],
                    "answer_text": ans,
                    "delay_chunks": delay,
                    "fmt": "free_text_substring",
                    "task": item.get("task"),
                    "in_window": in_window,
                }

            return {
                "correct": (letter == gt_letter) and in_window,
                "answer_chunk": out["chunk_idx"],
                "answer_text": ans,
                "delay_chunks": delay,
                "fmt": "letter",
                "task": item.get("task"),
                "in_window": in_window,
            }

        return {
            "correct": False,
            "answer_chunk": None,
            "answer_text": None,
            "delay_chunks": None,
            "fmt": "no_answer",
            "task": item.get("task"),
            "in_window": False,
        }


# ===========================================================================
# Our val.jsonl — open-ended with answer_form metadata
# ===========================================================================


class OurOpenEndedAdapter:
    """Our SFT/RL eval set with metadata.answer_form per sample.

    Routes scoring through compute_outcome_reward_v12 (matches RL reward
    so eval and RL-time signal stay aligned). For descriptive answers,
    falls back to fuzzy substring match — caller can pass judge_fn for
    LLM-as-judge.
    """

    NAME = "our_val"
    from thinkstream.data.agent_protocol import AGENT_CHUNK_SEC  # canonical (v12.5: 1s/chunk)
    @staticmethod
    def question_chunk(item: Dict) -> int:
        # Our val items have an `ask_chunk` (or `chunk_idx` for single-step)
        return int(item.get("ask_chunk", item.get("chunk_idx", 0)))

    @staticmethod
    def build_user_input(item: Dict, chunk_idx: int) -> str:
        if chunk_idx != OurOpenEndedAdapter.question_chunk(item):
            return ""
        q = item.get("question", "")
        # If the item already has options (rare in our val, but possible),
        # reuse OVO formatter for consistency.
        opts = item.get("options") or []
        if opts:
            letters = ["A", "B", "C", "D", "E"][: len(opts)]
            opts_block = "\n".join(f"{l}. {t}" for l, t in zip(letters, opts))
            return f"Q: {q}\nOptions:\n{opts_block}\nAnswer with one letter ({'/'.join(letters)})."
        return f"Q: {q}"

    @staticmethod
    def score(item: Dict, model_outputs: List[Dict], *, judge_fn=None) -> Dict:
        from thinkstream.trainer.v12_rewards import compute_outcome_reward_v12

        question_chunk = OurOpenEndedAdapter.question_chunk(item)
        gold = item.get("gold_answer") or item.get("answer", "")
        answer_form = item.get("answer_form", "")

        for out in model_outputs:
            if out["chunk_idx"] < question_chunk:
                continue
            ans = _extract_answer_text(out["text"])
            if ans is None:
                continue
            score = compute_outcome_reward_v12(
                ans, gold, judge_fn=judge_fn, answer_form=answer_form,
            )
            return {
                "correct": score >= 0.5,
                "answer_chunk": out["chunk_idx"],
                "answer_text": ans,
                "delay_chunks": out["chunk_idx"] - question_chunk,
                "outcome_score": score,
                "answer_form": answer_form,
            }

        return {
            "correct": False,
            "answer_chunk": None,
            "answer_text": None,
            "delay_chunks": None,
            "outcome_score": 0.0,
            "answer_form": answer_form,
        }


# ===========================================================================
# MC datasets without timing (e.g., VideoMME, EgoSchema)
# ===========================================================================


class MCNoTimingAdapter:
    """For MC datasets where the question is asked at the END of the video.

    We treat the FINAL chunk as the question chunk. Model is expected to
    have processed the whole video and emit the answer letter at the end.
    """

    NAME = "mc_no_timing"
    from thinkstream.data.agent_protocol import AGENT_CHUNK_SEC  # canonical (v12.5: 1s/chunk)
    @staticmethod
    def question_chunk_for(item: Dict, total_chunks: int) -> int:
        return total_chunks - 1

    @staticmethod
    def build_user_input(item: Dict, chunk_idx: int, total_chunks: int) -> str:
        if chunk_idx != total_chunks - 1:
            return ""
        # Reuse OVO formatter
        return OVOBenchAdapter.build_user_input(
            {**item, "realtime": (total_chunks - 1) * MCNoTimingAdapter.AGENT_CHUNK_SEC},
            chunk_idx,
        )

    @staticmethod
    def score(item: Dict, model_outputs: List[Dict], total_chunks: int) -> Dict:
        # Re-frame as OVO with realtime = last chunk
        synth_item = {
            **item,
            "realtime": (total_chunks - 1) * MCNoTimingAdapter.AGENT_CHUNK_SEC,
        }
        return OVOBenchAdapter.score(synth_item, model_outputs)


# ===========================================================================
# Adapter registry
# ===========================================================================


ADAPTER_REGISTRY = {
    "ovo_bench": OVOBenchAdapter,
    "our_val": OurOpenEndedAdapter,
    "mc_no_timing": MCNoTimingAdapter,
}


def get_adapter(name: str):
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter: {name!r}. "
            f"Available: {list(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name]
