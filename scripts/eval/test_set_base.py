#!/usr/bin/env python
"""Fair base-model eval on our test.jsonl — three video-context modes.

Extracts the user question and gold answer from each test sample,
asks the base model the question with visual context, and scores the
output via OVO-style prefix/substring match.

Modes (--mode):
  offline    — full video [0, video_end] uniformly sampled to 64 frames.
               This is the base-VLM ceiling: "what's the best a strong
               offline VLM can do given full information?"
  streaming  — only the visual_window slice the streaming agent sees,
               i.e. the last visual_window_chunks*agent_chunk_sec
               seconds before the decision (default 12*2=24 sec, 24
               frames @ 1fps). This is the apples-to-apples baseline:
               "given the SAME visual context our agent has at decision
               time, what does base produce?" — the relevant comparison
               for measuring what the agent protocol adds.
  probe      — difficulty-tagging mode. For every sample, runs THREE
               probes and reports per-sample (p_visual, p_text,
               p_recall_oracle) plus a category label:
                 A. visual:        visual_window only          (= streaming)
                 B. text:          no video, question only     (text-leak detector)
                 C. recall_oracle: visual_window + teacher-chosen
                                   recall_result.frame_paths    (recall ceiling)
               Each probe runs --g samples (default 4) with do_sample=True.
               Categories: trivially_visual / text_leak /
               recall_required / too_hard / unknown.  Use this to filter
               training data into the GRPO Goldilocks band (~30-70%).

Only scores Yes/No, integer, and single-letter responses (587 of 1,600
test samples after filtering response/recall_response). Descriptive
gold responses (323 samples) are skipped because reliable scoring
requires an LLM judge — out of scope for a quick base baseline.

Single-GPU usage:
    python scripts/eval/test_set_base.py \\
        --ckpt Qwen/Qwen3-VL-8B-Instruct \\
        --test_jsonl data/agent_v5/final/test.jsonl \\
        --mode streaming \\
        --n 200
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


GOLD_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL)


def detect_model_class(ckpt: str):
    name = ckpt.lower()
    basename = Path(ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls
    if "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls


def extract_gold(sample):
    out = sample.get("output", "")
    if not out and "messages" in sample:
        last = sample["messages"][-1]
        c = last.get("content", "")
        if isinstance(c, list):
            out = (c[0] if c else {}).get("text", "")
        else:
            out = c
    m = GOLD_RE.search(out)
    return m.group(1).strip() if m else None


def gold_kind(gold):
    if gold is None:
        return None
    g = gold.strip()
    if g in ("Yes", "No"):
        return "yes_no"
    if g.isdigit():
        return "int"
    if len(g) == 1 and g.isalpha():
        return "letter"
    return "descriptive"


def extract_question(sample):
    """Pull the user-facing question relevant to this chunk.

    Priority: input.user_input on this chunk → most recent unanswered
    query → most recent query of any kind. Returns None if no question
    is found (then the sample is skipped).

    v9.4.2: reject whitespace-only fields ("   " is truthy in Python).
    """
    inp = sample.get("input", {})
    ui = inp.get("user_input")
    if ui and ui.strip():
        return ui
    queries = inp.get("queries", []) or []
    for q in reversed(queries):
        if not q.get("answers"):
            qtext = q.get("question") or ""
            if qtext.strip():
                return qtext
    if queries:
        qtext = queries[-1].get("question") or ""
        if qtext.strip():
            return qtext
    return None


def score(pred_text, gold, kind, scoring="lenient"):
    """Score base-VLM response against gold.

    scoring="lenient" (default — what previous versions did):
      - yes_no: first occurrence of "yes" / "no" in any case wins
      - int:    first \\d+ run anywhere in the response
      - letter: first character (uppercased) compared to gold
    scoring="strict": stripped response must EXACTLY equal gold token. Useful
      for measuring whether the model NATURALLY outputs the expected format
      without relying on extraction heuristics.
    """
    if not pred_text or not gold:
        return False
    pred = pred_text.strip()
    if scoring == "strict":
        # Strict = the model NATIVELY produced the canonical token (case +
        # exact match). Useful for measuring format adherence; lenient mode
        # below covers semantic correctness regardless of casing.
        if kind == "yes_no":
            return pred in ("Yes", "No") and pred == gold
        if kind == "int":
            return pred.isdigit() and pred == gold
        if kind == "letter":
            return len(pred) == 1 and pred == gold
        return False
    # lenient (default)
    if kind == "yes_no":
        lower = pred.lower()
        i_yes, i_no = lower.find("yes"), lower.find("no")
        i_yes = i_yes if i_yes >= 0 else 10**9
        i_no = i_no if i_no >= 0 else 10**9
        if i_yes == i_no == 10**9:
            return False
        return ("Yes" if i_yes < i_no else "No") == gold
    if kind == "int":
        m = re.search(r"\d+", pred)
        return bool(m) and m.group() == gold
    if kind == "letter":
        return pred[:1].upper() == gold.upper()
    return False  # descriptive: not scored


def resolve_video_path(sample, video_root):
    vp = sample.get("video_path", "")
    if not vp:
        return None
    p = Path(vp)
    if p.is_absolute():
        return str(p)
    if video_root:
        return str(Path(video_root) / vp)
    base = Path(sample.get("data_path") or ".")
    return str(base / vp)


SYSTEM_PROMPT = (
    "You are a helpful video understanding assistant. Watch the "
    "video carefully and answer questions based on what you observe. "
    "If the question is yes/no, answer with Yes or No. If the "
    "question asks for a count, answer with the integer."
)


def build_messages(question, frames=None, recall_frames=None):
    """Build messages for base-model eval.

    - frames=None, recall_frames=None: text-only (probe B: text-leak).
    - frames set, recall_frames=None : visual context (probe A / streaming/offline).
    - frames set, recall_frames set  : visual + recall (probe C: recall ceiling).

    `frames` and `recall_frames` are lists of pre-extracted frame paths.
    """
    user_content = []
    if frames:
        user_content.append({"type": "video", "video": list(frames)})
    if recall_frames:
        user_content.append({"type": "video", "video": list(recall_frames)})
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def run_inference(
    model, processor, messages, *,
    max_new_tokens, num_samples=1, temperature=0.7, top_p=0.95, pad_id=None,
):
    """Run a single batched generate. Returns list of decoded strings (len = num_samples)."""
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = num_samples > 1
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            num_return_sequences=num_samples,
            pad_token_id=pad_id,
        )
    texts = []
    for i in range(gen.shape[0]):
        texts.append(
            processor.tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True)
        )
    return texts


def categorize_difficulty(p_A, p_B, p_C, *,
                          high=0.7, low=0.3):
    """Bucket a sample by (p_visual, p_text, p_recall_oracle).

    - text_leak        : p_B high — solvable from question alone.
    - trivially_visual : p_A high (and p_B not high) — visual window suffices.
    - recall_required  : p_A low, p_C high — needs the recalled frames.
    - too_hard         : p_C low (or visual high then C unknown) — even oracle can't solve.
    - unknown          : otherwise (mid-range; needs more samples).
    """
    if p_B is not None and p_B >= high:
        return "text_leak"
    if p_A is not None and p_A >= high:
        return "trivially_visual"
    if p_C is not None and p_A is not None and p_A < low and p_C >= high:
        return "recall_required"
    if p_C is not None and p_C < low:
        return "too_hard"
    return "unknown"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument(
        "--mode",
        default="streaming",
        choices=["offline", "streaming", "probe"],
        help="offline = full video; streaming = visual_window only; "
        "probe = run all 3 difficulty probes (visual / text / "
        "recall_oracle) per sample for difficulty tagging.",
    )
    p.add_argument(
        "--g", type=int, default=4,
        help="probe mode: samples per probe (G≥2 enables sampled p̂; G=1 = greedy).",
    )
    p.add_argument(
        "--probe_temp", type=float, default=0.7,
        help="probe mode: sampling temperature for p̂ estimate.",
    )
    p.add_argument(
        "--probe_max_recall_frames", type=int, default=8,
        help="probe mode: cap on recall_result.frame_paths used for probe C.",
    )
    p.add_argument(
        "--probe_high", type=float, default=0.7,
        help="probe mode: p̂ threshold for 'easy' (text_leak / trivially_visual).",
    )
    p.add_argument(
        "--probe_low", type=float, default=0.3,
        help="probe mode: p̂ threshold for 'hard' (too_hard).",
    )
    p.add_argument("--n", type=int, default=200, help="Max samples to evaluate")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Override frame count. Defaults: offline=64, streaming=24",
    )
    p.add_argument(
        "--visual_window_chunks",
        type=int,
        default=12,
        help="Window length for --mode streaming (chunks of agent_chunk_sec).",
    )
    p.add_argument("--agent_chunk_sec", type=float, default=2.0)
    p.add_argument("--video_root", default=None)
    p.add_argument("--scoring", default="lenient", choices=["lenient", "strict"],
                   help="lenient (default): first-matching-token wins (yes/no, "
                        "int, or A-D letter anywhere in response). strict: "
                        "response must exactly equal gold (tests whether the "
                        "model natively outputs the right format).")
    p.add_argument("--profile", default="16k", choices=["16k", "32k"],
                   help="Eval context profile (parity with agent eval). For "
                        "base eval the only effect is on max_new_tokens "
                        "default and output JSON metadata — base VLM doesn't "
                        "use queries/recall caps. Stamping the profile in the "
                        "output enables fair comparison with agent runs.")
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    args = p.parse_args()

    if args.max_frames is None:
        args.max_frames = 64 if args.mode == "offline" else (
            args.visual_window_chunks * 2
        )

    Cls = detect_model_class(args.ckpt)
    print(f"[mode={args.mode}] Loading {Cls.__name__} from {args.ckpt} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()

    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # Filter test.jsonl to scorable response samples
    samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))

    scorable = []
    for s in samples:
        if s.get("sample_type") not in ("response", "recall_response"):
            continue
        gold = extract_gold(s)
        kind = gold_kind(gold)
        if kind in ("yes_no", "int", "letter"):
            scorable.append((s, gold, kind))

    if args.n and 0 < args.n < len(scorable):
        scorable = scorable[: args.n]
    print(f"Filtered {len(scorable)} scorable samples (Yes/No, integer, letter)")

    results = []
    skipped = 0
    t0 = time.time()
    root_path = Path(args.test_jsonl).resolve().parent.parent.parent.parent  # .../ThinkStream

    def _abs_frames(rel_paths):
        out = []
        for fp in rel_paths or []:
            pp = Path(fp)
            out.append(str(pp if pp.is_absolute() else root_path / pp))
        return out

    def _sampled_visual_frames(s):
        """Resolve the streaming visual_window frame paths for sample s."""
        inp = s.get("input", {})
        vw = inp.get("visual_window", {})
        return _abs_frames(vw.get("frame_paths", []))

    def _sampled_recall_frames(s, max_frames):
        """Resolve teacher-chosen recall_result.frame_paths (probe C)."""
        inp = s.get("input", {})
        rr = inp.get("recall_result") or {}
        paths = _abs_frames(rr.get("frame_paths", []))
        if max_frames and len(paths) > max_frames:
            paths = paths[:max_frames]
        return paths

    def _estimate_p(messages):
        """Run G samples → return (p_correct or None, n_total). None if messages is None."""
        if messages is None:
            return None, 0
        try:
            texts = run_inference(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                num_samples=max(1, args.g),
                temperature=args.probe_temp,
                pad_id=pad_id,
            )
        except Exception as e:
            print(f"  probe inference failed: {type(e).__name__}: {e}")
            return None, 0
        n_correct = sum(int(score(t, gold, kind, scoring=args.scoring)) for t in texts)
        return (n_correct / len(texts) if texts else None), len(texts)

    for i, (s, gold, kind) in enumerate(scorable):
        try:
            chunk_idx = int(s.get("chunk_idx", 0))
            decision_end = (chunk_idx + 1) * args.agent_chunk_sec
            question = extract_question(s)
            if not question:
                skipped += 1
                continue

            # ---- PROBE MODE ----------------------------------------
            if args.mode == "probe":
                visual_frames = _sampled_visual_frames(s)
                recall_frames = _sampled_recall_frames(s, args.probe_max_recall_frames)

                msgs_A = build_messages(question, frames=visual_frames) if visual_frames else None
                msgs_B = build_messages(question)
                msgs_C = (build_messages(question, frames=visual_frames, recall_frames=recall_frames)
                          if visual_frames and recall_frames else None)

                p_A, n_A = _estimate_p(msgs_A)
                p_B, n_B = _estimate_p(msgs_B)
                p_C, n_C = _estimate_p(msgs_C)
                category = categorize_difficulty(p_A, p_B, p_C,
                                                 high=args.probe_high, low=args.probe_low)

                results.append({
                    "idx": i,
                    "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                    "sample_type": s.get("sample_type"),
                    "kind": kind,
                    "gold": gold,
                    "p_visual": p_A, "n_visual": n_A,
                    "p_text": p_B, "n_text": n_B,
                    "p_recall_oracle": p_C, "n_recall_oracle": n_C,
                    "category": category,
                })
                if (i + 1) % 20 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[{i+1}/{len(scorable)}] {rate:.2f} samples/sec")
                continue

            # ---- OFFLINE / STREAMING MODES -------------------------
            if args.mode == "offline":
                video_path = resolve_video_path(s, args.video_root)
                if not video_path:
                    skipped += 1
                    continue
                video_basename = Path(video_path).stem
                frame_dir = root_path / "data" / "agent_v5" / "frames" / video_basename
                if not frame_dir.exists():
                    inp = s.get("input", {})
                    vw = inp.get("visual_window", {})
                    frame_paths = vw.get("frame_paths", [])
                    if frame_paths:
                        frame_dir = root_path / Path(frame_paths[0]).parent
                    else:
                        skipped += 1
                        continue
                all_frames = sorted(frame_dir.glob("*.jpg"))
                if not all_frames:
                    skipped += 1
                    continue
                n_frames = len(all_frames)
                if n_frames > args.max_frames:
                    indices = [int(i * (n_frames - 1) / (args.max_frames - 1)) for i in range(args.max_frames)]
                    sampled = [str(all_frames[i]) for i in indices]
                else:
                    sampled = [str(f) for f in all_frames]
                messages = build_messages(question, frames=sampled)
            else:  # streaming: use pre-extracted frame_paths from visual_window
                abs_paths = _sampled_visual_frames(s)
                if not abs_paths:
                    skipped += 1
                    continue
                messages = build_messages(question, frames=abs_paths)

            text = run_inference(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                num_samples=1,
                pad_id=pad_id,
            )[0]

            correct = score(text, gold, kind, scoring=args.scoring)
            if args.mode == "offline":
                vw_start, vw_end = 0.0, float(decision_end)
            else:
                vw_start = float(decision_end - args.visual_window_chunks * args.agent_chunk_sec)
                vw_end = float(decision_end)
            results.append({
                "idx": i,
                "sample_id": s.get("sample_id") or s.get("trajectory_id"),
                "sample_type": s.get("sample_type"),
                "kind": kind,
                "gold": gold,
                "video_window": [vw_start, vw_end],
                "pred": text[:300],
                "correct": correct,
            })
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[{i+1}/{len(scorable)}] {rate:.2f} samples/sec")
        except Exception as e:
            print(f"Sample {i} failed: {type(e).__name__}: {e}")
            skipped += 1
            continue

    if not results:
        print("No successful runs.")
        return

    summary_payload = {}
    if args.mode == "probe":
        # Per-probe mean p̂, per-kind, per-category distribution.
        from collections import Counter
        cat_counts = Counter(r["category"] for r in results)
        kind_p = defaultdict(lambda: {"n": 0, "p_A": 0.0, "p_B": 0.0, "p_C": 0.0,
                                       "n_A": 0, "n_B": 0, "n_C": 0})
        for r in results:
            for k in (r["kind"], "_all"):
                v = kind_p[k]
                v["n"] += 1
                if r["p_visual"] is not None:
                    v["p_A"] += r["p_visual"]; v["n_A"] += 1
                if r["p_text"] is not None:
                    v["p_B"] += r["p_text"]; v["n_B"] += 1
                if r["p_recall_oracle"] is not None:
                    v["p_C"] += r["p_recall_oracle"]; v["n_C"] += 1
        print()
        print(f"{'kind':<12} {'n':>5} {'p_visual':>10} {'p_text':>10} {'p_recall':>10}")
        print("-" * 53)
        for k in sorted(kind_p.keys()):
            v = kind_p[k]
            pa = v["p_A"]/v["n_A"] if v["n_A"] else float("nan")
            pb = v["p_B"]/v["n_B"] if v["n_B"] else float("nan")
            pc = v["p_C"]/v["n_C"] if v["n_C"] else float("nan")
            print(f"{k:<12} {v['n']:>5} {pa:>10.3f} {pb:>10.3f} {pc:>10.3f}")
        print()
        print("Difficulty category distribution:")
        for cat, n in cat_counts.most_common():
            print(f"  {cat:<22} {n:>5} ({n/len(results)*100:.1f}%)")
        print()
        print(f"Recommendation: trivially_visual + text_leak should be dropped or hardened;")
        print(f"recall_required is the GRPO golden band (advantage > 0).")
        summary_payload = {
            "by_kind": {k: dict(v) for k, v in kind_p.items()},
            "categories": dict(cat_counts),
            "thresholds": {"high": args.probe_high, "low": args.probe_low},
            "g": args.g,
            "probe_temp": args.probe_temp,
        }
    else:
        by = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in results:
            for k in (r["kind"], "_all"):
                by[k]["n"] += 1
                by[k]["correct"] += int(r["correct"])
        print()
        header = f"{'kind':<15}  {'n':>5}  {'accuracy':>10}"
        print(header)
        print("-" * len(header))
        for k in sorted(by.keys()):
            v = by[k]
            if v["n"] == 0:
                continue
            print(f"{k:<15}  {v['n']:>5}  {v['correct']/v['n']:>10.3f}")
        summary_payload = {"by_kind": {k: dict(v) for k, v in by.items()}}

    print()
    print(f"Skipped: {skipped} (missing video / missing question / errors)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "mode": args.mode,
                "scoring": args.scoring,
                "profile": args.profile,
                "test_jsonl": args.test_jsonl,
                "n_samples": len(results),
                "n_skipped": skipped,
                **summary_payload,
                "samples": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
