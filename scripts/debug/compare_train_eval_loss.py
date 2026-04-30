"""Compare training loss vs eval-pipeline loss on the same sample.

If build_per_timestep_messages (training) and build_single_step_messages
(eval/inference) produce identical inputs, the forward loss should match
to numerical precision (ignoring dropout).
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinkstream.data.agent_protocol import VISUAL_WINDOW_CHUNKS
from thinkstream.model.agent_loop import build_single_step_messages
from thinkstream.sft.data_processor import preprocess_per_timestep

IGNORE_INDEX = -100


def compute_loss_from_dict(data_dict, model, device="cuda"):
    """Run a single forward pass and return CE loss over assistant span."""
    input_ids = data_dict["input_ids"].to(device)
    labels = data_dict["labels"].to(device)

    kwargs = {
        "input_ids": input_ids,
        "labels": labels,
    }
    if data_dict.get("pixel_values_videos") is not None:
        kwargs["pixel_values_videos"] = data_dict["pixel_values_videos"].to(device)
        kwargs["video_grid_thw"] = data_dict["video_grid_thw"].to(device)
    if data_dict.get("pixel_values") is not None:
        kwargs["pixel_values"] = data_dict["pixel_values"].to(device)
        kwargs["image_grid_thw"] = data_dict["image_grid_thw"].to(device)
    if data_dict.get("position_ids") is not None:
        kwargs["position_ids"] = data_dict["position_ids"].to(device)

    with torch.no_grad():
        outputs = model(**kwargs)

    return float(outputs.loss.item())


def sample_to_eval_messages(sample):
    """Convert a pipeline JSON sample to eval messages via build_single_step_messages."""
    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]

    # Reconstruct snapshot from sample memory block
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

    # Frame paths: training sample stores them in visual_window.frame_paths
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


def eval_messages_to_train_dict(messages, output_text, processor):
    """Tokenize eval messages + output exactly like preprocess_per_timestep."""
    # Append assistant turn
    messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": output_text}]}
    ]
    return preprocess_per_timestep({"messages": messages}, processor)


def main():
    # Use trained checkpoint-100 for consistency check.
    model_path = "/home/tione/notebook/gaozhenkun/hzh/ThinkStream/output/agent-sft/checkpoint-100"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    if hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Load one sample from batch1 backup
    data_path = ROOT / "data" / "batch1_backup" / "final_batch1_backup" / "train_sft.jsonl"
    with open(data_path) as f:
        sample = json.loads(f.readline())

    print(f"Sample id: {sample.get('sample_id')}")
    print(f"Sample type: {sample.get('sample_type')}")
    print(f"Chunk idx: {sample.get('chunk_idx')}")

    # ── Training pipeline loss ──
    print("\n--- Training pipeline ---")
    train_dict = preprocess_per_timestep(sample, processor)
    train_loss = compute_loss_from_dict(train_dict, model, device)
    print(f"Train loss: {train_loss:.6f}")

    # ── Eval pipeline loss ──
    print("\n--- Eval pipeline ---")
    eval_msgs = sample_to_eval_messages(sample)
    output_text = sample["output"]
    eval_dict = eval_messages_to_train_dict(eval_msgs, output_text, processor)
    eval_loss = compute_loss_from_dict(eval_dict, model, device)
    print(f"Eval loss:  {eval_loss:.6f}")

    # ── Diff ──
    diff = abs(train_loss - eval_loss)
    print(f"\n--- Diff ---")
    print(f"Absolute diff: {diff:.8f}")
    if diff < 1e-4:
        print("PASS: losses match to numerical precision.")
    else:
        print(f"MISMATCH: diff = {diff:.6f}")

    # Debug: compare tokenized text
    train_text = processor.tokenizer.decode(train_dict["input_ids"][0].tolist())
    eval_text = processor.tokenizer.decode(eval_dict["input_ids"][0].tolist())
    if train_text == eval_text:
        print("Tokenized text: IDENTICAL")
    else:
        print("Tokenized text: DIFFERENT")
        # Show first divergence
        for i, (a, b) in enumerate(zip(train_text, eval_text)):
            if a != b:
                print(f"  First diff at char {i}: train={train_text[i:i+40]!r} eval={eval_text[i:i+40]!r}")
                break


if __name__ == "__main__":
    main()
