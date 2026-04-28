"""Diagnose why the SFT ckpt omits <action> tags in free generation.

Background:
  The first v11.3 SFT run produced eval/action_acc=0.98 (teacher-forced)
  but free-generation output looks like
      </think>responseThe video shows...
  instead of the expected
      </think><action>response</action><response>The video shows...

  Teacher-forced eval feeds the gold prefix `<action>` and asks the
  model what fills the keyword slot — that's "open-book" and explains
  the 0.98. Free-generation is "closed-book": the model has to *decide*
  to emit `<action>` itself, and apparently it doesn't.

  Common root causes:
    1. <action> not registered as a single token; tokenizer splits it
       into < + action + > so the SPAN_WEIGHTS["action"]=8.0 never
       reaches the bracket characters.
    2. resize_token_embeddings was called but the new-token embeddings
       were initialized in a way that left them near-zero in the
       output projection (model produces them with ~0 logit at gen).
    3. Training-data <action> tags exist but the loss-mask infrastructure
       silently misses them (e.g., chat-template alignment off-by-one
       so the action-tag tokens fall outside ans_start..ans_end).
    4. Data is correct, tokenizer is correct, model just learned to
       skip the structural tokens because some other reward dominates.

This script checks (1), (3), and indirectly (4) on a saved ckpt.

Usage:
  python scripts/audit/diagnose_action_tag_omission.py \
      --ckpt /path/to/saved-checkpoint-200 \
      --sample data/agent_v5/final/train_sft.jsonl \
      --n 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Check 1: <action> tokenizes as a single token
# ---------------------------------------------------------------------------


def check_special_token_singleness(tokenizer) -> Dict[str, Dict]:
    """For each agent special token, verify it tokenizes to exactly 1 token id.

    If <action> is multi-token, the SPAN_WEIGHTS pipeline silently
    falls back to default weight (1.0) — explains why the model never
    learns to emit the bracket characters under post-rebalance ACTION
    pressure.
    """
    tags = [
        "<think>", "</think>",
        "<action>", "</action>",
        "<response>", "</response>",
        "<query>", "</query>",
        "<summary>", "</summary>",
        "<recall_result>", "</recall_result>",
        "<memory>", "</memory>",
        "<visual_window>", "</visual_window>",
        "<recalled_frames>", "</recalled_frames>",
        "<user_input>", "</user_input>",
        "<queries>", "</queries>",
        "<compress_trigger>", "</compress_trigger>",
    ]
    out = {}
    vocab = tokenizer.get_vocab()
    for tag in tags:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        in_vocab_id = vocab.get(tag)
        out[tag] = {
            "in_vocab": in_vocab_id is not None,
            "vocab_id": in_vocab_id,
            "encode_ids": ids,
            "encode_len": len(ids),
            "single_token": len(ids) == 1,
            "encoded_pieces": [tokenizer.decode([i], skip_special_tokens=False) for i in ids],
        }
    return out


# ---------------------------------------------------------------------------
# Check 2: training data <output> field contains <action> literally
# ---------------------------------------------------------------------------


def check_data_contains_action(jsonl_path: str, n: int = 20) -> Dict:
    """Sample the training data and count how many <action> tags appear.

    Confirms (or denies) the upstream pipeline's claim that pass3c
    bakes <action> into output. If this returns 0, the bug is upstream
    of SFT and we re-run pass3c (cheap).
    """
    counts = {
        "total_samples": 0,
        "with_action_open": 0,
        "with_action_close": 0,
        "with_response_open": 0,
        "with_summary_open": 0,
        "with_query_open": 0,
        "samples": [],
    }
    with open(jsonl_path) as f:
        for line in f:
            counts["total_samples"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = row.get("output", "")
            if not output:
                continue
            if "<action>" in output:
                counts["with_action_open"] += 1
            if "</action>" in output:
                counts["with_action_close"] += 1
            if "<response>" in output:
                counts["with_response_open"] += 1
            if "<summary>" in output:
                counts["with_summary_open"] += 1
            if "<query>" in output:
                counts["with_query_open"] += 1
            if len(counts["samples"]) < n:
                counts["samples"].append({
                    "sample_type": row.get("sample_type"),
                    "output_preview": output[:300],
                })
    return counts


# ---------------------------------------------------------------------------
# Check 3: tokenized output preserves <action> as registered token
# ---------------------------------------------------------------------------


def check_tokenized_action_position(
    tokenizer, jsonl_path: str, n: int = 5,
) -> List[Dict]:
    """Tokenize a real sample's output, find the <action> token position,
    and verify it's stored as the registered single-token id.

    If `<action>` is registered but the tokenizer.encode(output) yields
    < + action + > as 3 tokens, you've found the bug — register_special_tokens
    isn't being applied to the same tokenizer instance the trainer used,
    or the token wasn't saved into the ckpt's tokenizer config.
    """
    out = []
    action_open_id = tokenizer.convert_tokens_to_ids("<action>")
    action_close_id = tokenizer.convert_tokens_to_ids("</action>")
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = row.get("output", "")
            if "<action>" not in output:
                continue
            ids = tokenizer.encode(output, add_special_tokens=False)
            decoded_per_id = [
                tokenizer.decode([i], skip_special_tokens=False) for i in ids
            ]
            action_pos = [i for i, tid in enumerate(ids) if tid == action_open_id]
            close_pos = [i for i, tid in enumerate(ids) if tid == action_close_id]
            out.append({
                "sample_type": row.get("sample_type"),
                "n_tokens": len(ids),
                "action_open_id": action_open_id,
                "action_open_positions": action_pos,
                "action_close_positions": close_pos,
                "single_token_action": len(action_pos) > 0,
                "first_30_pieces": decoded_per_id[:30],
                "first_30_ids": ids[:30],
            })
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Check 4: model output projection logit for <action> token
# ---------------------------------------------------------------------------


def check_model_action_logit(model, tokenizer) -> Dict:
    """Inspect the model's output-projection bias/embedding for the
    <action> token. If it's near-zero relative to common tokens (e.g.,
    the word "response"), the model literally can't produce <action> at
    sampling time — the new-token embeddings stayed near init.

    Implementation: feed a dummy `</think>` context and look at the
    logits over all vocab; rank where <action> sits vs "response" and
    other words.
    """
    import torch
    action_id = tokenizer.convert_tokens_to_ids("<action>")
    response_id = tokenizer.convert_tokens_to_ids("response")  # may be multi-token; ignore if -1

    device = next(model.parameters()).device
    # Fake input: just encode some end-of-think context.
    ctx = "<|im_start|>assistant\n<think>The video shows a person.</think>"
    ids = tokenizer.encode(ctx, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0, -1]  # next-token logits

    top_k = 20
    topk_vals, topk_idx = logits.topk(top_k)
    return {
        "action_id": action_id,
        "action_logit": float(logits[action_id].item()) if action_id != tokenizer.unk_token_id else None,
        "topk_tokens": [
            (int(i.item()), tokenizer.decode([int(i.item())]), float(v.item()))
            for v, i in zip(topk_vals, topk_idx)
        ],
        "action_rank_in_vocab": int((logits > logits[action_id]).sum().item()) if action_id is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Saved checkpoint dir (contains tokenizer + model).")
    parser.add_argument("--sample", type=str, required=True,
                        help="Training-data JSONL to sample (e.g. train_sft.jsonl).")
    parser.add_argument("--n", type=int, default=5,
                        help="How many samples to check token-level.")
    parser.add_argument("--n_data_check", type=int, default=20,
                        help="How many sample-output previews to print.")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip Check 4 (model logit inspection) — useful if "
                             "you only want fast tokenizer/data checks.")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)

    print()
    print("=" * 70)
    print("Check 1: special-token tokenization (single-token check)")
    print("=" * 70)
    res1 = check_special_token_singleness(tokenizer)
    bad = []
    for tag, info in res1.items():
        marker = "✓" if info["single_token"] else "✗"
        print(f"  {marker} {tag:<25} in_vocab={info['in_vocab']!s:<5} "
              f"len={info['encode_len']} ids={info['encode_ids']} "
              f"pieces={info['encoded_pieces']}")
        if not info["single_token"]:
            bad.append(tag)
    if bad:
        print()
        print(f"  ⚠ {len(bad)} tags are NOT single-token: {bad}")
        print(f"    These won't be span-weighted by data_processor.py:_resolve_span_token_ids")
        print(f"    → ROOT CAUSE candidate. Re-run register_special_tokens before SFT.")
    else:
        print()
        print(f"  ✓ All agent special tokens tokenize as single tokens.")

    print()
    print("=" * 70)
    print(f"Check 2: training data — <action> presence (n={args.n_data_check})")
    print("=" * 70)
    res2 = check_data_contains_action(args.sample, n=args.n_data_check)
    print(f"  Total samples scanned: {res2['total_samples']}")
    print(f"  with <action>:  {res2['with_action_open']} "
          f"({res2['with_action_open']/max(res2['total_samples'],1):.1%})")
    print(f"  with </action>: {res2['with_action_close']}")
    print(f"  with <response>:{res2['with_response_open']}")
    print(f"  with <summary>: {res2['with_summary_open']}")
    print(f"  with <query>:   {res2['with_query_open']}")
    if res2["with_action_open"] / max(res2["total_samples"], 1) < 0.6:
        print(f"  ⚠ Less than 60% of samples have <action> — pipeline issue?")
    print()
    print(f"  First {min(3, len(res2['samples']))} sample previews:")
    for s in res2["samples"][:3]:
        print(f"    [{s['sample_type']}] {s['output_preview'][:200]}")

    print()
    print("=" * 70)
    print(f"Check 3: tokenized <action> position in real samples (n={args.n})")
    print("=" * 70)
    res3 = check_tokenized_action_position(tokenizer, args.sample, n=args.n)
    if not res3:
        print("  ✗ no samples had <action> in output — re-check pass3c output.")
    else:
        for i, s in enumerate(res3):
            print(f"  Sample {i+1} [{s['sample_type']}]:")
            print(f"    n_tokens = {s['n_tokens']}, action_open_id = {s['action_open_id']}")
            print(f"    found <action> at positions: {s['action_open_positions']} "
                  f"(single_token: {s['single_token_action']})")
            print(f"    first 30 pieces: {s['first_30_pieces']}")
            if not s["single_token_action"]:
                print(f"    ⚠ <action> not found as single token — encode broke the tag")

    if not args.skip_model:
        print()
        print("=" * 70)
        print("Check 4: model output-logit ranking for <action>")
        print("=" * 70)
        try:
            from transformers import AutoModelForCausalLM
            print(f"  Loading model from {args.ckpt} (this may take a minute)...")
            model = AutoModelForCausalLM.from_pretrained(
                args.ckpt, trust_remote_code=True, torch_dtype="auto"
            )
            model.eval()
            res4 = check_model_action_logit(model, tokenizer)
            print(f"  <action> token id = {res4['action_id']}")
            print(f"  <action> logit at next-token after </think>: {res4['action_logit']}")
            print(f"  <action> rank in vocab (lower=more likely): {res4['action_rank_in_vocab']}")
            print(f"  Top-20 tokens model wants to emit after </think>:")
            for tid, piece, logit in res4["topk_tokens"]:
                marker = "  *" if tid == res4["action_id"] else "   "
                print(f"    {marker} id={tid:<6} logit={logit:.3f} piece={piece!r}")
            if res4["action_rank_in_vocab"] is not None and res4["action_rank_in_vocab"] > 50:
                print()
                print(f"  ⚠ <action> ranks > 50 in next-token logits — model has "
                      f"learned to skip it. ROOT CAUSE confirmed.")
        except Exception as e:
            print(f"  Model check skipped: {e}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  bad_tag_count       = {len(bad)}")
    print(f"  data_<action>_rate  = {res2['with_action_open']/max(res2['total_samples'],1):.1%}")
    if res3:
        rate = sum(1 for s in res3 if s["single_token_action"]) / len(res3)
        print(f"  tokenize_single_rate= {rate:.1%}")
    if not args.skip_model and 'res4' in dir():
        print(f"  <action> logit rank = {res4.get('action_rank_in_vocab')}")


if __name__ == "__main__":
    main()
