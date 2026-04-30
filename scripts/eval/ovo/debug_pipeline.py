"""Debug script to verify agent pipeline correctness step-by-step.

Checks:
1. Input/output format per chunk
2. Truncation (max_new_tokens)
3. Action correctness (silent/response/recall/compress)
4. Compress behavior when system triggers
5. Recall behavior and retrieval quality
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor, AutoTokenizer

from thinkstream.model.agent_loop import (
    StreamingAgentLoop,
    make_generate_fn,
    AGENT_CHUNK_SEC,
)
from thinkstream.model.retrieval import make_retriever
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import (
    update_processor_pixels,
)


def detect_model_class(ckpt: str):
    name = ckpt.lower()
    basename = Path(ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls, "qwen3vl"
    if "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls, "qwen2.5vl"
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls, "qwen3vl"


def resolve_video_path(video_field, video_root):
    p = Path(video_field)
    return str(p) if p.is_absolute() else str(Path(video_root) / video_field)


def build_mcq_question(sample):
    options = sample.get("options", [])
    lines = [sample["question"]]
    for i, opt in enumerate(options):
        lines.append(f"{chr(65+i)}. {opt}")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def build_crr_question(sample):
    return (
        f"{sample['question']}\n"
        f'Return "Yes" if the action described has happened in the visible '
        f"video so far; otherwise return \"No\"."
    )


def build_rec_question(sample):
    activity = sample.get("activity", "perform the action")
    return (
        f"You're watching a video where people may perform a certain action "
        f"repetitively. The performer is referred to as 'they'.\n"
        f"How many times have they {activity} so far?\n"
        f"Your response type should be INT, for example, 0/1/2/3."
    )


def build_ssr_question(step_text):
    return (
        f"You're watching a tutorial video which contains a sequence of steps. "
        f"The following is one step from the procedure:\n\n{step_text}\n\n"
        f"Your task is to decide: Is the person in the video currently "
        f"carrying out this step?\n"
        f'Return "Yes" if they are; return "No" if not.'
    )


def debug_run_agent(loop, video_path, ask_chunks, max_chunk, retriever):
    """Run agent with verbose per-step logging."""
    per_chunk = {}
    for chunk_idx in range(max_chunk + 1):
        q = ask_chunks.get(chunk_idx)
        t0 = time.time()
        result = loop.step(chunk_idx=chunk_idx, video_path=video_path, user_question=q)
        dt = time.time() - t0

        action = result.get("action", "?")
        payload = result.get("payload") or {}
        think = result.get("think", "")

        # Print step summary
        print(f"\n--- chunk={chunk_idx} (t={chunk_idx*AGENT_CHUNK_SEC}s) dt={dt:.2f}s ---")
        print(f"  user_input={q!r}")
        print(f"  action={action!r}")
        print(f"  think_len={len(think)} think={think[:200]!r}")
        if action == "response":
            print(f"  response={payload.get('response', '')!r}")
        elif action == "recall":
            print(f"  recall_query={payload.get('query', {})!r}")
            print(f"  recall_result_chunks={result.get('recall_returned_chunks', [])}")
        elif action == "compress":
            print(f"  compress_summary={payload.get('summary', {})!r}")

        # Memory state
        mem = loop.memory
        recent_tok = mem.count_recent_tokens()
        should_comp = mem.should_compress()
        print(f"  memory: recent_thinks={len(mem.recent_thinks)} tokens={recent_tok} "
              f"compressed={len(mem.compressed_segments)} should_compress={should_comp}")

        # Compress telemetry
        ct = result.get("compress_telemetry")
        if ct:
            print(f"  COMPRESS_TRIGGERED: thinks_at_trigger={ct['thinks_count_at_trigger']} "
                  f"token_count={ct['thinks_token_count']} compressed_chunks={ct['compressed_chunks']}")

        # Check for truncation signs
        raw = result.get("raw_output", "")
        if raw and ("<think>" in raw and "</think>" not in raw):
            print("  WARNING: truncated think tag!")
        if raw and ("<action>" in raw and "</action>" not in raw):
            print("  WARNING: truncated action tag!")

        # Record
        if action == "recall":
            final_action = result.get("final_action") or "recall_then_silent"
            final_payload = result.get("final_payload") or {}
            per_chunk[chunk_idx] = (final_action, final_payload.get("response", ""))
        elif action == "response":
            per_chunk[chunk_idx] = ("response", payload.get("response", ""))
        elif action == "silent":
            per_chunk[chunk_idx] = ("silent", "")
        elif action == "compress":
            per_chunk[chunk_idx] = ("compress", "")
        else:
            per_chunk[chunk_idx] = (action, "")
    return per_chunk


def eval_one(sample, loop, retriever, video_root, scoring="strict"):
    task = sample.get("task")
    video_path = resolve_video_path(sample["video"], video_root)
    if not Path(video_path).exists():
        print(f"VIDEO NOT FOUND: {video_path}")
        return None

    print(f"\n{'='*60}")
    print(f"TASK={task} id={sample.get('id')} video={sample['video']}")

    if task in {"EPM", "ASI", "HLD", "OCR", "ACR", "ATR", "STU", "FPD", "OJR"}:
        # MCQ
        realtime = float(sample["realtime"])
        ask_chunk = int(realtime / AGENT_CHUNK_SEC)
        extra = 2 if scoring == "strict" else 60
        max_chunk = ask_chunk + extra
        question = build_mcq_question(sample)
        print(f"MCQ ask_chunk={ask_chunk} max_chunk={max_chunk}")
        print(f"Question: {question[:200]}")

        loop.reset()
        retriever.chunk_embeddings.clear() if hasattr(retriever, "chunk_embeddings") else None
        per_chunk = debug_run_agent(loop, video_path, {ask_chunk: question}, max_chunk, retriever)

        pred = None
        for c in range(ask_chunk, max_chunk + 1):
            a, r = per_chunk.get(c, ("missing", ""))
            if a == "response" and r:
                import re
                m = re.search(r"\b([A-Da-d])\b", r)
                pred = (m.group(1).upper() if m else r[:1].upper()) if r else None
                break
        gt = chr(65 + sample["gt"])
        print(f"RESULT: gt={gt} pred={pred} correct={pred==gt}")

    elif task == "REC":
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1
        question = build_rec_question(sample)
        print(f"REC max_chunk={max_chunk}")

        loop.reset()
        retriever.chunk_embeddings.clear() if hasattr(retriever, "chunk_embeddings") else None
        per_chunk = debug_run_agent(loop, video_path, {0: question}, max_chunk, retriever)

    elif task == "SSR":
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1
        ask_chunks = {}
        for probe in test_info:
            c = int(float(probe["realtime"]) / AGENT_CHUNK_SEC)
            ask_chunks[c] = build_ssr_question(probe.get("step", ""))
        print(f"SSR max_chunk={max_chunk} n_probes={len(test_info)}")

        loop.reset()
        retriever.chunk_embeddings.clear() if hasattr(retriever, "chunk_embeddings") else None
        per_chunk = debug_run_agent(loop, video_path, ask_chunks, max_chunk, retriever)

    elif task == "CRR":
        ask_time = float(sample["ask_time"])
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        ask_chunk = int(ask_time / AGENT_CHUNK_SEC)
        max_chunk = int(last_probe / AGENT_CHUNK_SEC) + 1
        question = build_crr_question(sample)
        print(f"CRR ask_chunk={ask_chunk} max_chunk={max_chunk}")

        loop.reset()
        retriever.chunk_embeddings.clear() if hasattr(retriever, "chunk_embeddings") else None
        per_chunk = debug_run_agent(loop, video_path, {ask_chunk: question}, max_chunk, retriever)

    return per_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/agent-sft/checkpoint-100")
    p.add_argument("--benchmark_json", default="/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/ovo_bench_new.json")
    p.add_argument("--video_root", default="/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench")
    p.add_argument("--frames_root", default="/home/tione/notebook/gaozhenkun/hzh/data/OVO-Bench/frames")
    p.add_argument("--task", default="FPD", help="Task to debug")
    p.add_argument("--id", type=int, default=None, help="Specific sample id")
    p.add_argument("--max_chunk", type=int, default=None, help="Override max_chunk for speed")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--compress_mode", default="system", choices=["system", "self"])
    p.add_argument("--retriever", default="bm25")
    args = p.parse_args()

    Cls, model_type = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} ...")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda()
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt, model_max_length=16384,
        padding_side="right", use_fast=False,
    )
    tokenizer.add_tokens(
        [t for t in processor.tokenizer.get_added_vocab().keys()
         if t not in tokenizer.get_vocab()],
        special_tokens=True,
    )

    print("Building retriever ...")
    retriever = make_retriever(
        kind=args.retriever, siglip_path="google/siglip-base-patch16-224",
        alpha=0.5, max_results=4, device="cuda",
    )

    loop = StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=100352,
        max_pixels=150528,
        max_new_tokens=args.max_new_tokens,
        retriever=retriever,
        compress_mode=args.compress_mode,
        frames_root=args.frames_root,
        video_root=args.video_root,
    )

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)

    candidates = [s for s in all_samples if s.get("task") == args.task]
    if args.id is not None:
        candidates = [s for s in candidates if s.get("id") == args.id]

    if not candidates:
        print(f"No samples found for task={args.task}")
        return

    sample = candidates[0]
    task = sample.get("task")
    video_path = resolve_video_path(sample["video"], args.video_root)

    # Build ask_chunks
    if task in {"EPM", "ASI", "HLD", "OCR", "ACR", "ATR", "STU", "FPD", "OJR"}:
        realtime = float(sample["realtime"])
        ask_chunk = int(realtime / AGENT_CHUNK_SEC)
        ask_chunks = {ask_chunk: build_mcq_question(sample)}
        default_max = ask_chunk + 2
    elif task == "REC":
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        default_max = int(last_probe / AGENT_CHUNK_SEC) + 1
        ask_chunks = {0: build_rec_question(sample)}
    elif task == "SSR":
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        default_max = int(last_probe / AGENT_CHUNK_SEC) + 1
        ask_chunks = {}
        for probe in test_info:
            c = int(float(probe["realtime"]) / AGENT_CHUNK_SEC)
            ask_chunks[c] = build_ssr_question(probe.get("step", ""))
    elif task == "CRR":
        ask_time = float(sample["ask_time"])
        test_info = sample["test_info"]
        last_probe = max(float(t["realtime"]) for t in test_info)
        ask_chunk = int(ask_time / AGENT_CHUNK_SEC)
        default_max = int(last_probe / AGENT_CHUNK_SEC) + 1
        ask_chunks = {ask_chunk: build_crr_question(sample)}
    else:
        print(f"Unknown task {task}")
        return

    max_chunk = args.max_chunk if args.max_chunk is not None else default_max
    print(f"\nTASK={task} id={sample.get('id')} video={sample['video']} max_chunk={max_chunk}")
    print(f"Ask chunks: {ask_chunks}")

    loop.reset()
    if hasattr(retriever, "chunk_embeddings"):
        retriever.chunk_embeddings.clear()

    per_chunk = debug_run_agent(loop, video_path, ask_chunks, max_chunk, retriever)
    print("\nDone.")


if __name__ == "__main__":
    main()
