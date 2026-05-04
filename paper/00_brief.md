# EMNLP 2026 Brief

Target: EMNLP 2026 main conference / ARR long paper.

Working title: ChronoStream: Agentic Temporal Reasoning for Streaming Video Understanding.

Core problem: Existing video reasoning systems usually consume the full video before answering. This batch paradigm creates latency and growing context cost, which is poorly matched to interactive streaming video assistants.

Core idea: Formulate streaming video understanding as a Watch-Think-Speak agent process. At each timestep, the model observes a sliding video window, updates or compresses reasoning memory, optionally recalls historical evidence, and decides whether to respond.

Primary contributions:

1. A streaming video reasoning formulation with explicit per-timestep actions.
2. A memory mechanism that preserves reasoning state while bounding visual/token context.
3. A data and training pipeline that makes per-timestep SFT samples match inference prompts.
4. RL-style training and reward design for response correctness, timing, recall, silence, and format.
5. Evaluation on online video benchmarks with accuracy, latency, and memory/efficiency analysis.

Submission constraints to track:

- Review long paper: 8 pages of main content plus unlimited references.
- Final long paper: typically 9 pages of main content after acceptance.
- Abstract: at most 200 words.
- Limitations section required and placed after conclusion, before references.
- Ethical considerations are encouraged and should also appear after conclusion.
- Paper size: A4.

Sources:

- Official EMNLP 2026 website: https://2026.emnlp.org/
- ACLPUB formatting guide: https://acl-org.github.io/ACLPUB/formatting.html
- ACL style files: https://github.com/acl-org/acl-style-files

Template decision:

- EMNLP 2026 main conference submissions go through ARR.
- ARR currently points authors to the official ACL style template.
- The project should therefore start from the official ACL style zip, not a third-party EMNLP mirror.
- Local template source notes: `paper/submission/TEMPLATE_SOURCE.md`.
