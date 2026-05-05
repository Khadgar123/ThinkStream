"""Current-only think rendering from pass1 evidence.

Pass1a now asks the teacher to emit a ``think`` observation-note field for
each independent chunk. This is supervised text content, not Qwen/vLLM
``enable_thinking`` reasoning. Downstream pass2 uses that field directly. The
deterministic renderer below is only a compatibility fallback for old pass1
caches or rare teacher outputs that parse evidence but miss the field.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


PREFERRED_THINK_FIELDS = (
    "think",
    "think_note",
    "observation_think",
    "caption",
    "dense_caption",
)


def clean_text(value: object, *, max_chars: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("fact") or value.get("text") or value.get("desc") or ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    # Frame/protocol tags are routing metadata, never target text.
    text = text.replace("<frame", "frame").replace("/>", "")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def _join_limited(items: Iterable[str], *, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _entity_phrase(entity: object) -> str:
    if not isinstance(entity, dict):
        return clean_text(entity)
    desc = clean_text(entity.get("desc") or entity.get("id") or "", max_chars=220)
    action = clean_text(entity.get("action") or "", max_chars=100)
    pos = clean_text(entity.get("position") or "", max_chars=80)
    if not desc:
        return ""
    phrase = desc
    if action and action.lower() != "static":
        phrase = f"{phrase} {action}"
    if pos:
        phrase = f"{phrase} ({pos})"
    return phrase


def _fact_text(fact: object) -> str:
    if isinstance(fact, dict):
        return clean_text(fact.get("fact") or fact.get("text") or fact)
    return clean_text(fact)


def _cap_words(text: str, *, max_words: int = 95) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words]).rstrip(" ,;:")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


def _sentence_case(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] if text[0].islower() else text


def build_think_from_pass1_evidence(cap: Dict) -> str:
    """Convert one pass1 evidence chunk into a current-only think note."""
    for field in PREFERRED_THINK_FIELDS:
        text = clean_text(cap.get(field), max_chars=900)
        if text:
            return _cap_words(text)

    entities = _join_limited(
        (_entity_phrase(e) for e in cap.get("visible_entities") or []),
        limit=4,
    )
    facts = _join_limited(
        (_fact_text(f) for f in cap.get("atomic_facts") or []),
        limit=3,
    )
    changes = _join_limited((cap.get("state_changes") or []), limit=2)
    ocr = _join_limited((cap.get("ocr") or []), limit=3)
    spatial = clean_text(cap.get("spatial") or "", max_chars=260)

    sentences: List[str] = []
    if entities:
        sentences.append("The current frames show " + "; ".join(entities) + ".")
    if facts:
        for fact in facts:
            fact_sentence = _sentence_case(fact).rstrip(" .")
            if fact_sentence:
                sentences.append(fact_sentence + ".")
    if changes:
        sentences.append("A visible state change is " + "; ".join(changes) + ".")
    if ocr:
        sentences.append("Visible text reads " + "; ".join(ocr) + ".")
    if spatial:
        if spatial[-1] not in ".!?":
            spatial += "."
        sentences.append(spatial)

    if not sentences:
        return "The current chunk has little discernible visual content."
    return _cap_words(" ".join(sentences))


def think_source_for_evidence(cap: Dict) -> str:
    """Return a diagnostic source label for a pass2 think built from evidence."""
    for field in PREFERRED_THINK_FIELDS:
        if clean_text(cap.get(field), max_chars=900):
            return "pass1_observation_note"
    return "pass1_evidence_fallback"
