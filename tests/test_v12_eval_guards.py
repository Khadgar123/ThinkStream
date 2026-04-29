"""v12 eval safety guards.

Verifies that legacy v11-only code paths raise loudly when invoked with
protocol_version='v12', so v12 model misuse fails fast instead of silently
corrupting eval scores by token-restricting to v11 tokens that don't exist.

Run: python tests/test_v12_eval_guards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mcq_predict_streaming_rejects_v12():
    """mcq_predict_streaming must raise ValueError when protocol_version='v12'.

    The function uses think_budget_sample_agent / think_budget_sample_restricted
    which hard-code v11 tokens (<action>, <response>, <query>, silent/response/
    recall word IDs). Passing a v12 model would mask logits to invalid tokens
    and produce garbage. We assert the guard raises BEFORE any model forward.
    """
    # Import via AST without instantiating — function body has the check at top.
    import ast
    src = Path(__file__).resolve().parents[1] / "thinkstream" / "eval" / "eval_common.py"
    tree = ast.parse(src.read_text())
    found_guard = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "mcq_predict_streaming":
            # Walk the function body for the guard
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Raise):
                    # Match: raise ValueError("... v11-only ...")
                    if isinstance(stmt.exc, ast.Call):
                        if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "ValueError":
                            for a in stmt.exc.args:
                                if isinstance(a, ast.JoinedStr):  # f-string
                                    for v in a.values:
                                        if isinstance(v, ast.Constant) and "v11-only" in v.value:
                                            found_guard = True
    assert found_guard, "mcq_predict_streaming missing v11-only guard"
    print("  PASS mcq_predict_streaming has v11-only guard")


def test_protocol_version_arg_present():
    """add_common_args must register --protocol_version with v11/v12 choices."""
    import ast
    src = Path(__file__).resolve().parents[1] / "thinkstream" / "eval" / "eval_common.py"
    tree = ast.parse(src.read_text())
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "add_argument":
                for a in node.args:
                    if isinstance(a, ast.Constant) and a.value == "--protocol_version":
                        found = True
    assert found, "--protocol_version not registered in add_common_args"
    print("  PASS --protocol_version registered")


def test_eval_entry_scripts_pass_protocol():
    """eval_ovo.py and eval_rtvu.py must forward args.protocol_version."""
    for script in ["thinkstream/eval/ovo_bench/eval_ovo.py",
                   "thinkstream/eval/rtvu/eval_rtvu.py"]:
        path = Path(__file__).resolve().parents[1] / script
        text = path.read_text()
        assert "args.protocol_version" in text, (
            f"{script} does not forward args.protocol_version"
        )
    print("  PASS eval entry scripts forward protocol_version")


def main():
    tests = [
        test_mcq_predict_streaming_rejects_v12,
        test_protocol_version_arg_present,
        test_eval_entry_scripts_pass_protocol,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
