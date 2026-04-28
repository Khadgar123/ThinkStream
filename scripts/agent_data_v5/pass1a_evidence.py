"""
Pass 1-A: Independent Per-Chunk Annotation

Each 2s chunk annotated independently with 2 frames.
No sliding window, no prior context, fully parallel.

Output: Per-chunk JSON with visible_entities(desc), atomic_facts, ocr, spatial.
        No state_changes (derived in 1-B), no entity IDs (assigned in 1-B).

Saved to: data/agent_v5/evidence_1a/{video_id}.json
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    EVIDENCE_1A_DIR,
    EVIDENCE_GRAPH_PROMPT,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Parse
# ---------------------------------------------------------------------------


def build_evidence_request(
    chunk_idx: int,
    frame_paths: List[str],
    video_id: str,
) -> Dict:
    """Build evidence request for one chunk (independent, 2 frames only)."""
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC

    prompt = EVIDENCE_GRAPH_PROMPT.format(start=int(start), end=int(end))
    chunk_frame_paths = get_chunk_frame_paths(frame_paths, chunk_idx)

    return {
        "messages": [{"role": "user", "content": build_vision_content(prompt, chunk_frame_paths)}],
        "max_tokens": PASS_CONFIG["pass1a"]["max_tokens"],
        "temperature": PASS_CONFIG["pass1a"]["temperature"],
        "id": f"{video_id}_1a_{chunk_idx}",
        "_meta": {"video_id": video_id, "chunk_idx": chunk_idx, "time": [start, end]},
    }


def _walker_rescue(s: str) -> Optional[Dict]:
    """Element-by-element walker for truncated JSON. Returns a dict with the
    expected fields when ANY array yielded a closed element, else None.

    Rationale: when max_tokens cuts a response mid-element, json.loads and
    brace-balance both fail to close the outer object. But the prefix of
    `visible_entities` / `atomic_facts` is intact and recoverable.
    """
    out: Dict = {"visible_entities": [], "atomic_facts": [], "ocr": [], "spatial": ""}

    def _walk_objects(start: int) -> List[Dict]:
        items: List[Dict] = []
        i = start
        n = len(s)
        while i < n:
            while i < n and s[i] in " \t\n\r,":
                i += 1
            if i >= n or s[i] == "]":
                break
            if s[i] != "{":
                break
            depth = 0
            j = i
            in_str = False
            esc = False
            closed = False
            while j < n:
                c = s[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif in_str:
                    if c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                items.append(json.loads(s[i : j + 1]))
                            except (json.JSONDecodeError, ValueError):
                                pass
                            closed = True
                            j += 1
                            break
                j += 1
            if not closed:
                break
            i = j
        return items

    def _walk_strings(start: int) -> List[str]:
        items: List[str] = []
        i = start
        n = len(s)
        while i < n:
            while i < n and s[i] in " \t\n\r,":
                i += 1
            if i >= n or s[i] == "]":
                break
            if s[i] != '"':
                break
            j = i + 1
            esc = False
            closed = False
            while j < n:
                if esc:
                    esc = False
                elif s[j] == "\\":
                    esc = True
                elif s[j] == '"':
                    closed = True
                    break
                j += 1
            if not closed:
                break
            try:
                items.append(json.loads(s[i : j + 1]))
            except (json.JSONDecodeError, ValueError):
                pass
            i = j + 1
        return items

    m = re.search(r'"visible_entities"\s*:\s*\[', s)
    if m:
        out["visible_entities"] = _walk_objects(m.end())
    m = re.search(r'"atomic_facts"\s*:\s*\[', s)
    if m:
        objs = _walk_objects(m.end())
        out["atomic_facts"] = objs if objs else _walk_strings(m.end())
    m = re.search(r'"ocr"\s*:\s*\[', s)
    if m:
        out["ocr"] = _walk_strings(m.end())
    m = re.search(r'"spatial"\s*:\s*"', s)
    if m:
        i = m.end() - 1
        j = i + 1
        esc = False
        while j < len(s):
            if esc:
                esc = False
            elif s[j] == "\\":
                esc = True
            elif s[j] == '"':
                try:
                    out["spatial"] = json.loads(s[i : j + 1])
                except (json.JSONDecodeError, ValueError):
                    pass
                break
            j += 1

    if (
        out["visible_entities"]
        or out["atomic_facts"]
        or out["ocr"]
        or out["spatial"].strip()
    ):
        return out
    return None


def parse_evidence_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse 397B evidence output. No state_changes expected (comes from 1-B).

    parse_success contract (v9.5):
      - True  iff JSON parsed AND at least one of
              (visible_entities, atomic_facts, ocr, spatial) is non-empty
      - False otherwise (raw retained as _raw for retry / debug)

    Old code (pre-v9.5) treated any json-decodable string as success even when
    every field was empty. Audit on batch1 showed 46.4% of chunks were
    silent-empty under that contract — videos with rich visible content but
    the model returned `{"time":[t,t]}` and nothing else. The strict
    contract here lets `run_pass1a` detect those and retry.
    """
    default = {
        "time": meta["time"],
        "visible_entities": [],
        "atomic_facts": [],
        "ocr": [],
        "spatial": "",
        "parse_success": False,
    }

    if raw is None:
        return default

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Strip markdown code-block wrappers so json.loads can parse directly.
    raw = raw.removeprefix("```json").removeprefix("```").strip()
    raw = raw.removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback 1: brace-balance scan, EARLIEST { first. The earliest is
        # the top-level object; using `reversed(positions)` would prefer an
        # inner entity dict and silently lose visible_entities/atomic_facts.
        positions = [i for i, c in enumerate(raw) if c == '{']
        parsed = None
        for start_idx in positions:
            depth = 0
            for i in range(start_idx, len(raw)):
                if raw[i] == '{':
                    depth += 1
                elif raw[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            cand = json.loads(raw[start_idx:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            cand = None
                        if isinstance(cand, dict) and (
                            cand.get("visible_entities")
                            or cand.get("atomic_facts")
                            or cand.get("ocr")
                            or cand.get("spatial")
                        ):
                            parsed = cand
                        break
            if parsed is not None:
                break
        # Fallback 2: element walker — survives mid-element truncation.
        if parsed is None:
            parsed = _walker_rescue(raw)
        if parsed is None:
            default["_raw"] = raw[:4000]
            return default

    raw_facts = parsed.get("atomic_facts", [])
    raw_entities = parsed.get("visible_entities", [])
    ocr = parsed.get("ocr", []) or []
    spatial = parsed.get("spatial", "") or ""

    # Empty-content guard: a JSON-valid but content-empty response is a
    # silent failure (~46% baseline). Mark it so the retry path can pick it
    # up; keep _raw so we can post-mortem the failure mode.
    has_content = bool(raw_entities or raw_facts or ocr or
                       (isinstance(spatial, str) and spatial.strip()))
    return {
        "time": parsed.get("time", meta["time"]),
        "visible_entities": [_normalize_entity(e) for e in raw_entities],
        "atomic_facts": [_normalize_atomic_fact(f) for f in raw_facts],
        "ocr": ocr,
        "spatial": spatial,
        "parse_success": has_content,
        **({} if has_content else {"_raw": raw[:4000], "_silent_empty": True}),
    }


def _normalize_atomic_fact(fact) -> dict:
    """v9.5: prompt now emits atomic_facts as list[str]. Old prompts emitted
    list[dict] with confidence/target_resolution_visible. Normalize either
    shape to the dict form so downstream pass3 code (which reads
    cap['atomic_facts'][i]['fact']) keeps working without churn.

    Confidence is set to 1.0 for backward-compat with pass3a's
    `confidence >= 0.7` filter (which is now effectively a no-op since the
    new prompt drops the field — 99.7% of v9.4 facts already had >=0.7).
    """
    if isinstance(fact, str):
        return {"fact": fact, "confidence": 1.0}
    if not isinstance(fact, dict):
        return {"fact": str(fact), "confidence": 0.0, "parse_repaired": True}
    # Legacy dict shape — keep its confidence if present
    fact.setdefault("confidence", 1.0)
    return fact


def _normalize_entity(entity) -> dict:
    if isinstance(entity, str):
        return {"desc": entity, "action": "", "position": ""}
    if not isinstance(entity, dict):
        return {"desc": str(entity), "action": "", "position": ""}
    if "desc" not in entity and "id" in entity:
        entity["desc"] = entity["id"]
    entity.setdefault("desc", "unknown")
    return entity


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_pass1a(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Annotate all chunks in parallel. Returns sorted captions list.

    Retry policy (v9.5): silent-empty responses (json-valid but no
    entities/facts/ocr/spatial) are retried once at higher temperature
    (0.7) to break the deterministic "model decided nothing was worth
    reporting" failure mode. Audit on batch1 showed ~46% of chunks
    failed silently under temperature=0.3 + thinking=True; the retry
    path recovers most of them with no concurrency impact (the retry
    re-uses the same client semaphore).
    """

    # v9.5: enable_thinking=False routes through raw httpx so the
    # chat_template_kwargs actually reaches vLLM (SDK extra_body is
    # dropped on this server — see vllm_client._call_one_raw).
    enable_thinking = bool(PASS_CONFIG["pass1a"].get("thinking", True))

    async def _call(messages, max_tokens, temperature, request_id):
        if semaphore:
            async with semaphore:
                return await client._call_one(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, request_id=request_id,
                    enable_thinking=enable_thinking,
                )
        return await client._call_one(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, request_id=request_id,
            enable_thinking=enable_thinking,
        )

    async def annotate_chunk(chunk_idx):
        request = build_evidence_request(chunk_idx, frame_paths, video_id)
        result = await _call(
            request["messages"], request["max_tokens"],
            request["temperature"], request["id"],
        )
        caption = parse_evidence_result(result, request["_meta"])

        # Retry once if silent-empty. Higher temperature breaks the
        # deterministic "I see nothing worth reporting" path.
        if caption.get("_silent_empty") or not caption.get("parse_success"):
            retry_result = await _call(
                request["messages"], request["max_tokens"],
                0.7, f'{request["id"]}_retry',
            )
            retry_caption = parse_evidence_result(retry_result, request["_meta"])
            if retry_caption.get("parse_success"):
                caption = retry_caption
            else:
                # Both attempts failed — keep flag for downstream to skip
                caption["_retry_failed"] = True

        caption["chunk_idx"] = chunk_idx
        caption["video_id"] = video_id
        return caption

    tasks = [annotate_chunk(i) for i in range(num_chunks)]
    captions = await asyncio.gather(*tasks)
    captions = sorted(captions, key=lambda c: c["chunk_idx"])

    n_ok = sum(1 for c in captions if c.get("parse_success"))
    logger.info(f"  [{video_id}] 1-A done: {n_ok}/{num_chunks} parsed ok")
    return captions


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_1a(video_id: str, captions: List[Dict], output_dir: Path = EVIDENCE_1A_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def load_1a(video_id: str, evidence_dir: Path = EVIDENCE_1A_DIR) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("1a"):
        return None
    path = evidence_dir / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers (used by other passes too)
# ---------------------------------------------------------------------------


def get_chunk_frame_paths(all_frame_paths: List[str], chunk_idx: int) -> List[str]:
    start = chunk_idx * FRAMES_PER_CHUNK
    end = start + FRAMES_PER_CHUNK
    return all_frame_paths[start:end] if start < len(all_frame_paths) else []


def build_vision_content(text: str, image_paths: List[str]) -> list:
    from scripts.agent_data_pipeline.vllm_client import encode_image_base64

    content = []
    for img_path in image_paths:
        if Path(img_path).exists():
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_base64(img_path)},
            })
    content.append({"type": "text", "text": text})
    return content
