"""Run retrieval audit across multiple oracle margins and schemes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/tione/notebook/gaozhenkun/hzh/envs/thinkstream/bin/python"
AUDIT = ROOT / "scripts/audit/recall_retrieval_audit.py"

SAMPLES = ROOT / "data/batch1_backup/final_batch1_backup/val.jsonl"
FRAMES = ROOT / "data/agent_v5/frames"
ROLLOUT = ROOT / "data/agent_v5/rollout"
CACHE = ROOT / "output/audit_cache"

margins = [0, 2, 4, 6, 8, 10, 12, 16, 20, None]
# None = no oracle (original broad time_range)

schemes_configs = [
    ("bm25_keyword", "bm25_keyword", {}),
    ("bm25_time", "bm25_time", {}),
    ("dense_text", "dense_text", {}),
    ("hybrid_0.0", "hybrid", {"alpha": "0.0"}),
    ("hybrid_0.25", "hybrid", {"alpha": "0.25"}),
    ("hybrid_0.5", "hybrid", {"alpha": "0.5"}),
    ("hybrid_0.75", "hybrid", {"alpha": "0.75"}),
    ("hybrid_1.0", "hybrid", {"alpha": "1.0"}),
]


def run_one(margin, scheme_name, scheme_arg, extra_args):
    out_dir = ROOT / f"output/audit_sweep/margin_{margin}_scheme_{scheme_name}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(AUDIT),
        "--samples", str(SAMPLES),
        "--frames_root", str(FRAMES),
        "--rollout_root", str(ROLLOUT),
        "--cache_dir", str(CACHE),
        "--out_dir", str(out_dir),
        "--schemes", scheme_arg,
        "--top_k", "1,3,5",
        "--device", "cuda",
    ]
    if margin is not None:
        cmd += ["--oracle_time_margin", str(margin)]
    for k, v in extra_args.items():
        cmd += [f"--{k}", v]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL margin={margin} scheme={scheme_name}")
        print(result.stderr[-500:])
        return None

    summary = json.loads((out_dir / "recall_audit_summary.json").read_text())
    data = summary["schemes"][scheme_arg]
    return {
        "hit@1": data["hit@1"],
        "hit@3": data["hit@3"],
        "hit@5": data["hit@5"],
        "mrr": data["mrr"],
    }


def main():
    results = {}
    total = len(margins) * len(schemes_configs)
    done = 0
    for margin in margins:
        for name, scheme_arg, extras in schemes_configs:
            done += 1
            print(f"[{done}/{total}] margin={margin} scheme={name} ...", end=" ", flush=True)
            r = run_one(margin, name, scheme_arg, extras)
            if r:
                results[(margin, name)] = r
                print(f"hit@1={r['hit@1']:.2f}")
            else:
                print("FAILED")

    # Output table
    print("\n" + "=" * 100)
    print(f"{'margin':>8} | {'scheme':>14} | {'hit@1':>6} | {'hit@3':>6} | {'hit@5':>6} | {'MRR':>6}")
    print("-" * 100)
    for margin in margins:
        for name, _, _ in schemes_configs:
            r = results.get((margin, name))
            if r:
                mstr = "broad" if margin is None else str(margin)
                print(f"{mstr:>8} | {name:>14} | {r['hit@1']:>6.2f} | {r['hit@3']:>6.2f} | {r['hit@5']:>6.2f} | {r['mrr']:>6.3f}")
    print("=" * 100)

    # Save JSON
    out_json = ROOT / "output/audit_sweep/summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {f"{m}_{n}": v for (m, n), v in results.items()}
    out_json.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
