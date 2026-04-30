"""Quick vLLM smoke test on checkpoint-100 with batch1 samples."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

from thinkstream.model.agent_loop import build_single_step_messages
from thinkstream.eval.vllm_engine import init_vllm_engine, prepare_vllm_input
from thinkstream.data.agent_protocol import VISUAL_WINDOW_CHUNKS, parse_agent_output

CKPT = ROOT / "output/agent-sft/checkpoint-100"
DEVICE = "cuda:0"


def load_samples(path, n=5):
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append(row)
            if len(out) >= n:
                break
    return out


def sample_to_messages(sample):
    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    memory = inp["memory"]
    snapshot = {
        "chunk_idx": chunk_idx,
        "compressed_segments": memory.get("compressed_segments", []),
        "recent_thinks": memory.get("recent_thinks", []),
        "visual_window_start": max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1),
    }
    video_path = sample.get("video_path", "")
    user_input = inp.get("user_input", "")
    queries = inp.get("queries", [])

    frame_paths = None
    vw = inp.get("visual_window", {})
    if "frame_paths" in vw:
        base = Path(sample.get("data_path", "."))
        frame_paths = [
            str(base / p) if not Path(p).is_absolute() else p
            for p in vw["frame_paths"]
        ]

    return build_single_step_messages(
        snapshot=snapshot,
        chunk_idx=chunk_idx,
        video_path=video_path,
        user_input=user_input,
        queries=queries,
        frame_paths=frame_paths,
    )


def main():
    print(f"Loading processor from {CKPT}")
    processor = AutoProcessor.from_pretrained(CKPT)
    if hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    print(f"Loading vLLM engine from {CKPT}")
    llm = init_vllm_engine(
        model_path=str(CKPT),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        max_videos_per_prompt=1,
    )

    samples = load_samples(ROOT / "data/batch1_backup/final_batch1_backup/val.jsonl", n=5)
    print(f"Loaded {len(samples)} samples\n")

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id", f"sample_{i}")
        stype = sample.get("sample_type", "?")
        print(f"--- Sample {i+1}: {sid} | type={stype} | chunk={sample['chunk_idx']} ---")

        messages = sample_to_messages(sample)
        vllm_input = prepare_vllm_input(messages, processor)

        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=128)
        outputs = llm.generate(vllm_input, sp)
        text = outputs[0].outputs[0].text if outputs else ""

        print(f"Generated: {text[:300]}")
        print(f"Gold:      {sample['output'][:300]}")

        parsed = parse_agent_output(text)
        if parsed:
            print(f"Parsed action: {parsed.get('action', 'N/A')}")
            if parsed.get('query'):
                print(f"Parsed query:  {parsed['query']}")
            if parsed.get('summary'):
                print(f"Parsed summary: {parsed['summary'][:100]}")
        print()


if __name__ == "__main__":
    main()
