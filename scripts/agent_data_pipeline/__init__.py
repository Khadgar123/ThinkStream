# Agent Data Construction Pipeline
# See docs/data_construction.md for the full design.
#
# Core scripts:
#   generate_data.py  — Main pipeline (Step 1-6)
#   vllm_client.py    — Async vLLM client with concurrency control
#   config.py         — Shared constants and prompts
#   utils.py          — Utility functions
