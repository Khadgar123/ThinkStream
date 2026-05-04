# Copyright 2026 ThinkStream contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# v12.14 STREAMING-VIDEO RECURRENT AGENT
# =======================================
#
# Recurrent counterpart to verl/recipe_thinkstream/streaming_agent_loop.py.
# Each chunk's interaction becomes ITS OWN training sample, eliminating
# the stitched-trajectory OOM ceiling.
#
# Status: rollout() implemented end-to-end. ray_trainer recurrent path
# integration still needed (v12.14 step 4 — see
# docs/v12.14_recurrent_design.md).
#
# Architecture mapping streaming_agent_loop.py → this file:
#
#   Outer chunk loop (chunk_idx 0..N)         → for-loop in rollout()
#   D1 chunk-internal recall multi-turn       → 2 conversations within one chunk
#   visual sliding window per chunk           → injected per-conversation mm
#   compress trigger (system event)           → action with no visual_window
#   per-Q answer attribution                  → tracked across conversations
#   stitched response_ids/response_mask        → list of independent conversations
#
# Each conversation = ONE assistant turn = ONE training sample. Across a
# trajectory: typical 1 conversation/chunk for silent/answer/compress;
# 2 conversations/chunk when a recall round fires (turn1=tool_call,
# turn2=answer-after-tool).
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

from verl.recurrent.async_utils import ChatCompletionProxy
from verl.recurrent.interface import (
    AsyncRAgent, AsyncOutput, RConfig, RDataset, RRegister,
)
from verl.recurrent.utils import log_step, msg
from verl.protocol import DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer
from thinkstream.data.agent_protocol import append_timestamped_image_list

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class StreamingVideoConfig(RConfig):
    """Per-action / per-chunk budgets for streaming-video recurrent rollout.

    `max_actions_per_trajectory` is in ACTIONS not chunks. With D1 a
    chunk that triggers recall produces 2 actions. Plan
    max_actions ≈ max_chunks × 1.5-2× for safety.
    """
    # Action / chunk caps
    max_chunks: int = 360                  # upper bound on video chunks
    max_actions_per_trajectory: int = 600  # actions = chunks × ~1.5 (D1 recall)
    max_chunk_prompt_length: int = 8192    # per-action prompt cap
    max_chunk_response_length: int = 512   # per-action response cap

    # Recall (D1)
    max_recall_per_chunk: int = 1
    max_recall_response_length: int = 512

    # Memory / compress (mirror config.py)
    compress_token_threshold: int = 3200
    compress_range_min: int = 8

    # Visual
    visual_window_chunks: int = 16
    frames_per_chunk: int = 2
    chunk_sec: float = 1.0
    visual_window_mode: str = "sliding"

    # Frames root (None → text-only)
    frames_root: str = ""

    # Final aggregation turn
    enable_final_turn: bool = False
    max_final_response_length: int = 1024


# ===========================================================================
# Dataset
# ===========================================================================
class StreamingVideoDataset(RDataset):
    """Multi-Q parquet adapter for recurrent rollout.

    Reads the same parquet shape as CustomRLHFDataset
    (recipe_thinkstream/thinkstream.py:178). Defers field normalisation
    to the agent's rollout() since each chunk rebuilds its own prompt
    from scratch (no benefit pre-tokenising context here).
    """

    def __init__(
        self,
        recurrent_config: StreamingVideoConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )
        self.recurrent_config = recurrent_config

    @override
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        row_dict["sample_uuid"] = str(uuid4())
        row_dict["index"] = row_dict.get("video_id", "")
        # Placeholder — agent rollout() builds chunks from extra_info, not
        # from a pre-tokenised context. Keep a 1-token tensor to satisfy
        # RDataset's contract.
        row_dict["context_ids"] = torch.zeros(1, dtype=torch.long)
        row_dict["context_length"] = torch.tensor(1, dtype=torch.long)
        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
        return (
            ["context_ids", "context_length"],
            ["video_id", "video_path", "n_chunks", "extra_info",
             "reward_model", "data_source", "sample_uuid"],
        )


# ===========================================================================
# Agent — main rollout() implementation
# ===========================================================================
class AsyncStreamingVideoAgent(AsyncRAgent):
    """Streaming-video recurrent agent.

    rollout(gen_item) returns AsyncOutput with N independent conversations
    (one per assistant turn) + per-conversation multi_modal_data.
    """

    def __init__(
        self,
        proxy: ChatCompletionProxy,
        tokenizer: PreTrainedTokenizer,
        config: StreamingVideoConfig,
        rollout_config: DictConfig,
    ):
        super().__init__(proxy, tokenizer, config, rollout_config)
        self.config: StreamingVideoConfig = config

    # ──────────────────────────────────────────────────────────────────────
    # Helpers — ported from streaming_agent_loop.py and shared via
    # the recipe path. Kept self-contained here to avoid circular imports.
    # ──────────────────────────────────────────────────────────────────────
    def _frames_for_chunk(self, video_path: str, chunk_idx: int) -> List[str]:
        """1-indexed JPEG frame paths for chunk_idx. Mirrors
        streaming_agent_loop._chunk_frame_paths."""
        if not video_path or not self.config.frames_root:
            return []
        root = Path(self.config.frames_root)
        if not root.exists():
            return []
        vp = Path(video_path)
        for cand in (root / vp.stem, root / vp.with_suffix("").name):
            if cand.exists() and cand.is_dir():
                root = cand
                break
        out: List[str] = []
        for fi in range(self.config.frames_per_chunk):
            idx_one = chunk_idx * self.config.frames_per_chunk + fi + 1
            for ext_path in (root / f"frame_{idx_one:06d}.jpg",
                             root / f"frame_{idx_one:05d}.jpg",
                             root / f"frame_{idx_one - 1:06d}.jpg"):
                if ext_path.exists():
                    out.append(str(ext_path))
                    break
            else:
                return []
        return out

    def _window_start(self, chunk_idx: int) -> int:
        if self.config.visual_window_mode == "expanding":
            seg = max(1, self.config.visual_window_chunks)
            return (chunk_idx // seg) * seg
        return max(0, chunk_idx - self.config.visual_window_chunks + 1)

    def _build_visual_window(self, video_path: str, chunk_idx: int):
        """Return (frame_paths, window_start_chunk)."""
        ws = self._window_start(chunk_idx)
        flat: List[str] = []
        for c in range(ws, chunk_idx + 1):
            cf = self._frames_for_chunk(video_path, c)
            if not cf:
                return [], ws
            flat.extend(cf)
        return flat, ws

    def _count_recent_thinks_tokens(self, recent_thinks: List[Dict]) -> int:
        total = 0
        for t in recent_thinks:
            text = t.get("text", "") if isinstance(t, dict) else str(t)
            if text:
                try:
                    total += len(self.tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    total += len(text) // 4
        return total

    def _check_compress_trigger(self, recent_thinks: List[Dict]) -> Optional[Tuple[int, int]]:
        if len(recent_thinks) < self.config.compress_range_min:
            return None
        if self._count_recent_thinks_tokens(recent_thinks) < self.config.compress_token_threshold:
            return None
        chunks = [int(t.get("chunk", -1)) for t in recent_thinks
                  if isinstance(t, dict) and t.get("chunk", -1) >= 0]
        if not chunks:
            return None
        return (min(chunks), max(chunks))

    # ──────────────────────────────────────────────────────────────────────
    # Per-action prompt builders — mirror streaming_agent_loop's
    # _build_chunk_user_content + _build_recall_tool_message
    # ──────────────────────────────────────────────────────────────────────
    def _build_chunk_user_message(
        self,
        *,
        memory_text: str,
        chunk_idx: int,
        window_paths: List[str],
        window_start: int,
        triggered_questions: List[Dict],
        compress_trigger_range: Optional[Tuple[int, int]],
    ) -> Tuple[Dict, Optional[Dict]]:
        """Returns (user_message, mm_payload). mm_payload is the
        per-action multi_modal_data entry (or None if text-only)."""
        content: List[Dict] = []
        # memory first (matches SFT layout)
        content.append({
            "type": "text",
            "text": f"<memory>\n{memory_text}\n</memory>",
        })
        mm_payload = None
        if compress_trigger_range is None:
            # visual_window header + timestamped image list
            vw_header = json.dumps({
                "start": window_start * self.config.chunk_sec,
                "end": (chunk_idx + 1) * self.config.chunk_sec,
                "frames": len(window_paths),
                "current_time": [
                    chunk_idx * self.config.chunk_sec,
                    (chunk_idx + 1) * self.config.chunk_sec,
                ],
            })
            content.append({
                "type": "text",
                "text": f"\n<visual_window>{vw_header}</visual_window>",
            })
            if window_paths:
                append_timestamped_image_list(
                    content,
                    window_paths,
                    fps=float(self.config.frames_per_chunk) / float(self.config.chunk_sec),
                    start_frame_index=window_start * self.config.frames_per_chunk,
                    total_num_frames=(chunk_idx + 1) * self.config.frames_per_chunk,
                    latest_start_frame_index=chunk_idx * self.config.frames_per_chunk,
                )
                mm_payload = {
                    "images": list(window_paths),
                }
            # user_input — render triggered questions
            if triggered_questions:
                blocks = []
                for q in triggered_questions:
                    qtxt = q.get("question", "") or ""
                    opts = q.get("options") or []
                    if opts:
                        opt_lines = "\n".join(
                            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(opts)
                        )
                        blocks.append(f"{qtxt}\n{opt_lines}")
                    else:
                        blocks.append(qtxt)
                joined = "\n---\n".join(blocks)
                content.append({
                    "type": "text",
                    "text": f"\n<user_input>{joined}</user_input>",
                })
        else:
            content.append({"type": "text", "text": "\n<compress_trigger/>"})
        return {"role": "user", "content": content}, mm_payload

    def _build_recall_tool_message(
        self, recall_result: Dict, recalled_frames: Optional[Dict],
    ) -> Tuple[Dict, Optional[Dict]]:
        """Returns (tool_message, mm_payload)."""
        content: List[Dict] = []
        mm_payload = None
        if recalled_frames:
            rf_header = json.dumps({
                "time_range": recalled_frames["time_range"],
                "source": recalled_frames.get("source", "historical_frames"),
                "n_frames": recalled_frames["n_frames"],
            })
            content.append({
                "type": "text",
                "text": f"<recalled_frames>{rf_header}</recalled_frames>",
            })
            tr_start, tr_end = recalled_frames["time_range"]
            append_timestamped_image_list(
                content,
                recalled_frames["frame_paths"],
                fps=float(self.config.frames_per_chunk) / float(self.config.chunk_sec),
                start_frame_index=int(tr_start) * self.config.frames_per_chunk,
                total_num_frames=int(tr_end + 1) * self.config.frames_per_chunk,
                context_label="recalled frame",
            )
            mm_payload = {
                "images": list(recalled_frames["frame_paths"]),
            }
        rr_json = json.dumps({
            "source": recall_result.get("source", "failure"),
            "time": recall_result.get("time", ""),
            "text": recall_result.get("text_content", recall_result.get("text", "")),
        }, ensure_ascii=False)
        content.append({
            "type": "text",
            "text": (
                f"\n<recall_result>{rr_json}</recall_result>"
                if recalled_frames else
                f"<recall_result>{rr_json}</recall_result>"
            ),
        })
        return {"role": "user", "content": content}, mm_payload

    # ──────────────────────────────────────────────────────────────────────
    # AsyncRAgent abstract methods
    # ──────────────────────────────────────────────────────────────────────
    @override
    def start(self, gen_batch, timing_raw):
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0
        self.sample_index_list: List[torch.Tensor] = []
        self.final_mask_list: List[torch.Tensor] = []

    @override
    def update(self, gen_output):
        return gen_output

    @override
    def done(self) -> bool:
        return self.step >= self.config.max_actions_per_trajectory

    @override
    def end(self):
        del self.gen_batch, self.timing_raw
        self.step = 0
        sample_index = torch.cat(self.sample_index_list) if self.sample_index_list else torch.zeros(0, dtype=torch.long)
        final_mask = torch.cat(self.final_mask_list) if self.final_mask_list else torch.zeros(0, dtype=torch.bool)
        del self.sample_index_list, self.final_mask_list
        return final_mask, sample_index

    # ──────────────────────────────────────────────────────────────────────
    # Main rollout
    # ──────────────────────────────────────────────────────────────────────
    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """One trajectory → N chunk conversations + final_mask."""
        timing_raw: Dict[str, float] = {}
        sample_idx_int = int(gen_item.batch.get("sample_index", torch.tensor(0)).item())

        nb = gen_item.non_tensor_batch
        video_id = str(nb.get("video_id", ""))
        video_path = str(nb.get("video_path", ""))
        n_chunks_dataset = int(nb.get("n_chunks", 0))
        n_chunks = min(self.config.max_chunks, n_chunks_dataset) if n_chunks_dataset > 0 else self.config.max_chunks

        # Multi-Q extraction
        extra = nb.get("extra_info") or {}
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        questions_raw = extra.get("questions") if isinstance(extra, dict) else None
        if hasattr(questions_raw, "tolist"):
            questions_raw = questions_raw.tolist()
        questions: List[Dict] = []
        for q in (questions_raw or []):
            if hasattr(q, "tolist"):
                q = q.tolist()
            if isinstance(q, dict):
                questions.append({k: (v.tolist() if hasattr(v, "tolist") else v)
                                  for k, v in q.items()})

        # Pre-compute ask_at_chunk
        ask_at_chunk: Dict[int, List[int]] = {}
        for q_idx, q in enumerate(questions):
            aks = q.get("ask_chunks") or []
            if not aks and int(q.get("ask_chunk", -1)) >= 0:
                aks = [int(q["ask_chunk"])]
            for ck in aks:
                try:
                    ask_at_chunk.setdefault(int(ck), []).append(q_idx)
                except (TypeError, ValueError):
                    pass

        # Initial system + user conversation prefix (rebuilt per chunk)
        # Use the dataset's `prompt` column if present, else minimal default.
        prompt_msgs = nb.get("prompt") or []
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        initial_messages = [
            {"role": m.get("role"), "content": m.get("content")}
            if isinstance(m, dict) else m
            for m in prompt_msgs
        ]

        # Trajectory-level state (mirrors VideoTrajectoryState)
        compressed_summaries: List[Dict] = []
        recent_thinks: List[Dict] = []
        per_q_answer_chunk: List[int] = [-1] * len(questions)
        per_q_answer_text: List[str] = [""] * len(questions)
        pending_q: List[int] = []

        conversations: List[List[Dict]] = []
        mm_data_list: List[Optional[Dict]] = []

        chunk_idx = 0
        kwargs_base = self.sampling_params(gen_item.meta_info)
        kwargs_base["max_completion_tokens"] = self.config.max_chunk_response_length

        while chunk_idx < n_chunks and self.step < self.config.max_actions_per_trajectory:
            with _timer("mt_mics", timing_raw):
                compress_range = self._check_compress_trigger(recent_thinks)
                inter_chunk = compress_range is not None

                # Visual window
                window_paths, window_start = ([], chunk_idx)
                if not inter_chunk:
                    window_paths, window_start = self._build_visual_window(
                        video_path, chunk_idx,
                    )

                # Triggered Qs at this chunk
                triggered: List[Dict] = []
                if not inter_chunk:
                    for qi in ask_at_chunk.get(chunk_idx, []):
                        triggered.append(questions[qi])
                        pending_q.append(qi)

                # Memory text — compress + recent thinks
                mem_lines = []
                for s in compressed_summaries:
                    mem_lines.append(f"[{s.get('time_range','')}] {s.get('text','')}")
                for t in recent_thinks:
                    mem_lines.append(f"<{t.get('chunk','')}> {t.get('text','')}")
                memory_text = "\n".join(mem_lines) if mem_lines else "(empty)"

                # Build chunk user message + mm payload
                user_msg, mm_payload = self._build_chunk_user_message(
                    memory_text=memory_text,
                    chunk_idx=chunk_idx,
                    window_paths=window_paths,
                    window_start=window_start,
                    triggered_questions=triggered,
                    compress_trigger_range=compress_range,
                )

                conversation = list(initial_messages) + [user_msg]

            # ── Round 1: chunk's primary turn ──
            with _timer("mt_async_gen", timing_raw):
                completions, err = await self.proxy.get_chat_completions(
                    messages=conversation,
                    **kwargs_base,
                )
                if err:
                    raise err

            with _timer("mt_mics", timing_raw):
                choice = completions.choices[0]
                conversation.append(msg(choice))
                conversations.append(conversation)
                mm_data_list.append(mm_payload)
                self.step += 1
                response_text = conversation[-1]["content"]

            # Parse turn1 to detect recall
            recall_round = 0
            from thinkstream.data.agent_protocol import parse_agent_output_v12
            parsed = parse_agent_output_v12(response_text)
            kind = parsed.get("kind", "unknown")

            # ── Round 2+ : chunk-internal recall multi-turn (D1) ──
            while (
                kind == "recall"
                and not inter_chunk
                and recall_round < self.config.max_recall_per_chunk
                and self.step < self.config.max_actions_per_trajectory
            ):
                with _timer("mt_mics", timing_raw):
                    args = (parsed.get("tool_call") or {}).get("arguments") or {}
                    recall_payload = self._execute_recall_for_action(
                        args, video_path, compressed_summaries, recent_thinks,
                    )
                    tool_msg, tool_mm = self._build_recall_tool_message(
                        recall_payload["recall_result"],
                        recall_payload["recalled_frames"],
                    )
                    conversation = conversation + [tool_msg]

                with _timer("mt_async_gen", timing_raw):
                    completions, err = await self.proxy.get_chat_completions(
                        messages=conversation,
                        **kwargs_base,
                    )
                    if err:
                        raise err

                with _timer("mt_mics", timing_raw):
                    choice = completions.choices[0]
                    conversation.append(msg(choice))
                    conversations.append(conversation)
                    mm_data_list.append(tool_mm)
                    self.step += 1
                    recall_round += 1
                    response_text = conversation[-1]["content"]
                    parsed = parse_agent_output_v12(response_text)
                    kind = parsed.get("kind", "unknown")

            # Per-Q answer attribution (FIFO under SFT-aligned behavior)
            if pending_q and not inter_chunk:
                m = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                if m:
                    answer_str = m.group(1).strip()
                    # 3-stage: window match → LIFO → FIFO floor
                    chosen = None
                    for pos, qi in enumerate(pending_q):
                        ach = questions[qi].get("answer_chunks") or []
                        try:
                            ach_int = [int(x) for x in ach]
                            if ach_int and min(ach_int) <= chunk_idx <= max(ach_int):
                                chosen = pos
                                break
                        except (TypeError, ValueError):
                            pass
                    if chosen is None:
                        chosen = len(pending_q) - 1
                    qi_pop = pending_q.pop(chosen)
                    per_q_answer_chunk[qi_pop] = chunk_idx
                    per_q_answer_text[qi_pop] = answer_str

            # State evolution
            think_text = parsed.get("think") or ""
            if not inter_chunk and think_text and kind in ("answer", "recall", "unknown"):
                recent_thinks.append({"chunk": chunk_idx, "text": think_text})

            # Compress state update (only on compress turn)
            if inter_chunk and kind == "compress":
                summary = (parsed.get("tool_call") or {}).get("arguments", {}).get("summary", {})
                if summary and isinstance(summary, dict):
                    tr = summary.get("time_range", [])
                    if isinstance(tr, list) and len(tr) == 2:
                        lo, hi = int(tr[0]), int(tr[1])
                        compressed_summaries.append({
                            "time_range": [lo, hi],
                            "text": summary.get("text", ""),
                        })
                        # Drop covered recent_thinks
                        recent_thinks = [
                            t for t in recent_thinks
                            if not (lo <= int(t.get("chunk", -1)) <= hi)
                        ]

            # Advance chunk_idx (compress turn doesn't consume chunk)
            if not inter_chunk:
                chunk_idx += 1

        # ── Build sample_index + final_mask ──
        n_actions = len(conversations)
        if n_actions == 0:
            # Pathological case: no rollout happened (shouldn't reach here
            # in practice because n_chunks >= 1 always). Emit one empty
            # final action so downstream tensorisation has something.
            conversations.append(list(initial_messages))
            mm_data_list.append(None)
            n_actions = 1

        # Surface per-Q telemetry on the final conversation's metadata
        # (the trainer can pull this from AsyncOutput.metrics).
        metrics = {
            "ts_video_id": video_id,
            "ts_n_actions": float(n_actions),
            "ts_n_chunks_processed": float(chunk_idx),
            "ts_per_q_answer_chunk": list(per_q_answer_chunk),
            "ts_per_q_answer_text": list(per_q_answer_text),
            "ts_n_questions": float(len(questions)),
        }

        sample_index = torch.full((n_actions,), sample_idx_int, dtype=torch.long)
        final_mask = torch.zeros(n_actions, dtype=torch.bool)
        final_mask[-1] = True

        return AsyncOutput(
            conversations=conversations,
            sample_index=sample_index,
            final_mask=final_mask,
            timing_raw=timing_raw,
            metrics=metrics,
            multi_modal_data=mm_data_list,
        )

    def _execute_recall_for_action(
        self, args: Dict, video_path: str,
        compressed_summaries: List[Dict], recent_thinks: List[Dict],
    ) -> Dict:
        """Synchronous recall — text retrieval + historical frame extraction.
        Mirrors streaming_agent_loop._execute_recall."""
        query = (args.get("query") or args.get("keywords") or args.get("text") or "")
        time_range_raw = args.get("time_range")
        tr_tuple: Optional[Tuple[float, float]] = None
        if isinstance(time_range_raw, str) and time_range_raw.strip():
            m = re.match(r"\s*([\d.]+)\s*-\s*([\d.]+)\s*", time_range_raw)
            if m:
                try:
                    tr_tuple = (float(m.group(1)), float(m.group(2)))
                except ValueError:
                    pass
        elif isinstance(time_range_raw, (list, tuple)) and len(time_range_raw) >= 2:
            try:
                tr_tuple = (float(time_range_raw[0]), float(time_range_raw[1]))
            except (TypeError, ValueError):
                pass

        # Text — keyword overlap (matches streaming_agent_loop's
        # _retrieve_from_memory minus the candidates ranking).
        text_hit = ""
        if query:
            kws = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 3]
            for entry in (compressed_summaries or []) + (recent_thinks or []):
                text = entry.get("text", "") if isinstance(entry, dict) else ""
                if text and any(kw in text.lower() for kw in kws):
                    text_hit = text
                    break

        # Frames
        recalled_frames = None
        recalled_paths: List[str] = []
        tr_start_chunk = tr_end_chunk = -1
        if tr_tuple and self.config.frames_root and video_path:
            try:
                tr_start_chunk = int(tr_tuple[0] / float(self.config.chunk_sec))
                tr_end_chunk = int(tr_tuple[1] / float(self.config.chunk_sec))
            except (TypeError, ValueError, ZeroDivisionError):
                tr_start_chunk = tr_end_chunk = -1
            if tr_start_chunk >= 0 and tr_end_chunk >= tr_start_chunk:
                for ci in range(tr_start_chunk, tr_end_chunk + 1):
                    cf = self._frames_for_chunk(video_path, ci)
                    if cf:
                        recalled_paths.extend(cf)
                if recalled_paths:
                    recalled_frames = {
                        "time_range": [tr_start_chunk, tr_end_chunk],
                        "source": "historical_frames",
                        "n_frames": len(recalled_paths),
                        "frame_paths": recalled_paths,
                    }
        success = bool(recalled_paths) or bool(text_hit)
        recall_result = {
            "source": "historical_frames" if recalled_paths else (
                "memory" if success else "failure"
            ),
            "text_content": text_hit if success else "No matching results found.",
            "text": text_hit if success else "No matching results found.",
            "returned_chunks": list(range(tr_start_chunk, tr_end_chunk + 1)) if recalled_paths else [],
            "time": (f"{tr_start_chunk}-{tr_end_chunk}" if tr_start_chunk >= 0 else ""),
        }
        return {"recall_result": recall_result, "recalled_frames": recalled_frames}


# ===========================================================================
# Registration entry point
# ===========================================================================
REGISTER = RRegister(
    config_cls=StreamingVideoConfig,
    dataset_cls=StreamingVideoDataset,
    agent_cls=AsyncStreamingVideoAgent,
)
