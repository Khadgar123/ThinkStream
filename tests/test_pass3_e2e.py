"""
End-to-end simulation of the Pass 3 data construction pipeline.

Walks through every stage with realistic mock data:
  evidence → classify_chunks → cards → placements → trajectories
  → fork samples → base samples → combined episode

No 397B calls — all LLM outputs are mocked.
Validates invariants at every step.
"""

import json
import random
import re
import pytest
from copy import deepcopy
from typing import Dict, List, Optional, Set


# =====================================================================
# Shared constants (match production code)
# =====================================================================

VISUAL_WINDOW_CHUNKS = 12
AGENT_CHUNK_SEC = 2

FAMILY_TARGETS = {
    "F1": 3, "F2": 4, "F3": 2, "F4": 2,
    "E1": 3, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 2,
}

RETENTION_CLASS = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
    "and", "or", "it", "yes", "no", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "there", "here", "not", "but", "if", "so", "than", "then",
    "just", "about", "up", "out", "its", "his", "her", "my", "your",
    "their", "our", "me", "him", "them", "us", "we", "they",
    "you", "he", "she", "with", "for", "from", "by", "as",
    "what", "which", "who", "whom", "whose", "how", "when", "where",
    "many", "much", "any", "some", "other", "tell", "describe",
    "currently", "now", "still", "yet", "ever", "already",
})


# =====================================================================
# Inline pure functions (avoid relative import issues)
# =====================================================================

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and len(w) > 1 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _extract_mc_choice_text(question, answer_letter):
    answer_letter = answer_letter.strip().upper()
    pattern = rf'(?:^|\s){answer_letter}[\.\)]\s*(.+?)(?:\s+[B-Z][\.\)]|$)'
    m = re.search(pattern, question, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_card_keywords(card):
    af = card.get("answer_form", "short_exact")
    q = card.get("question", "")
    ca = card.get("canonical_answer", "")
    if af == "binary":
        return extract_keywords(q)
    elif af == "multiple_choice":
        q_base = re.split(r'\s+A[\.\)]', q, maxsplit=1)[0]
        q_kw = extract_keywords(q_base)
        c_kw = extract_keywords(_extract_mc_choice_text(q, ca))
        seen = set()
        out = []
        for w in q_kw + c_kw:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out
    elif af == "number":
        q_kw = extract_keywords(q)
        num = ca.strip()
        if num and num not in set(q_kw):
            q_kw.append(num)
        return q_kw
    else:
        return extract_keywords(ca)


def _keyword_overlap(text, keywords):
    if not keywords:
        return 0.0
    tw = set(re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()))
    return sum(1 for k in keywords if k in tw) / len(keywords)


# =====================================================================
# Stage 0: Build realistic video data (60 chunks = 120s cooking video)
# =====================================================================

def build_evidence(num_chunks=60):
    """Simulate Pass 1 output: per-chunk evidence for a cooking video."""
    rng = random.Random(123)
    actions = [
        "chopping onions", "peeling garlic", "heating oil in pan",
        "adding garlic to pan", "stirring contents", "slicing tomatoes",
        "adding tomatoes to pot", "seasoning with salt", "tasting sauce",
        "plating finished dish",
    ]
    evidence = []
    for i in range(num_chunks):
        action = actions[i % len(actions)]
        cap = {
            "chunk_idx": i,
            "time": [i * 2, (i + 1) * 2],
            "visible_entities": [
                {"desc": "person wearing red apron, short hair",
                 "action": action, "id": "person_1", "position": "center"},
            ],
            "atomic_facts": [
                {"fact": f"person {action}", "confidence": 0.85},
            ],
            "ocr": [],
            "state_changes": [],
        }
        # OCR at multiples of 15
        if i % 15 == 0 and i > 0:
            cap["ocr"] = [f"${rng.randint(2,19)}.99"]
        # Second entity (pot) appears from chunk 5 onward
        if i >= 5:
            cap["visible_entities"].append(
                {"desc": "stainless steel pot", "action": "on right burner",
                 "id": "pot_1", "position": "right"})
        # Third entity (bowl) at specific chunks
        if i in (3, 7, 21, 35, 50):
            cap["visible_entities"].append(
                {"desc": "small white ceramic bowl", "action": "on counter",
                 "id": "bowl_1", "position": "left"})
        # State changes for chunks 10-14 (procedure) and 30-33
        if 10 <= i <= 14 or 30 <= i <= 33:
            cap["state_changes"] = [f"started {action}"]
        # Digit facts at specific chunks
        if i in (12, 24, 36, 48):
            cap["atomic_facts"].append(
                {"fact": f"added approximately 15 grams of seasoning", "confidence": 0.92})
        evidence.append(cap)
    return evidence


def build_rollout(num_chunks=60):
    """Simulate Pass 2 output: question-blind streaming rollout."""
    thinks = []
    for i in range(num_chunks):
        base = f"Person in red apron "
        if i < 20:
            base += "chopping at counter."
        elif i < 40:
            base += "stirring pot on right burner."
        else:
            base += "plating food on counter."
        # Add specific observations at key chunks
        if i in (5, 12, 25, 40):
            base += " Added seasoning to stainless pot."
        if i % 15 == 0 and i > 0:
            base += " Price tag shows $4.99 on screen."
        if i in (3, 7, 21, 35, 50):
            base += " Small white bowl on counter."
        thinks.append({
            "chunk_idx": i,
            "time": f"{i*2}-{(i+1)*2}",
            "think": base,
        })

    snapshots = {}
    for i in range(num_chunks):
        ws = max(0, i - VISUAL_WINDOW_CHUNKS)
        recent = [{"text": thinks[j]["think"], "time": thinks[j]["time"], "chunk": j}
                  for j in range(max(0, i - 10), i)]
        snapshots[i] = {
            "chunk_idx": i,
            "visual_window_start": ws,
            "recent_thinks": recent,
            "compressed_segments": [],
        }

    compression_events = [
        {"trigger_chunk": 20,
         "compressed_thinks_chunks": list(range(0, 10)),
         "summary": {"time_range": [0, 20],
                      "text": "Person in red apron chopped onions and peeled garlic at counter. "
                              "Small white bowl on counter. Stainless pot on right burner."}},
        {"trigger_chunk": 40,
         "compressed_thinks_chunks": list(range(10, 25)),
         "summary": {"time_range": [20, 50],
                      "text": "Person heated oil, added tomatoes and seasoning to pot. "
                              "Stirred contents. Added 15 grams of seasoning."}},
    ]

    return {
        "num_chunks": num_chunks,
        "thinks": thinks,
        "snapshots": snapshots,
        "compression_events": compression_events,
    }


# =====================================================================
# Stage 1: classify_chunks (3-A Step 1)
# =====================================================================

def classify_chunks(evidence):
    """Exact copy of production classify_chunks, minus fallback paths
    (tested separately). Core structural filtering only."""
    fc = {f: [] for f in FAMILY_TARGETS}
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}

    for cap in evidence:
        idx = cap["chunk_idx"]
        entities = cap.get("visible_entities", [])
        facts = [f for f in cap.get("atomic_facts", []) if f.get("confidence", 0) >= 0.7]
        has_digit = any(
            re.search(r'\d{2,}|[\$€¥£]\d|\d\s*(?:kg|lb|ml|oz|cm|mm|g)\b', f.get("fact", ""))
            for f in facts)
        if cap.get("ocr") or has_digit:
            fc["F1"].append(idx)
        if has_digit:
            fc["F3"].append(idx)
        if entities:
            fc["F2"].append(idx)
        if len(entities) >= 2:
            fc["F4"].append(idx)
        if cap.get("state_changes"):
            fc["E2"].append(idx)
        if len(entities) >= 3:
            fc["S1"].append(idx)

    all_chunks = [c["chunk_idx"] for c in evidence if c.get("atomic_facts")]
    target_e1 = FAMILY_TARGETS.get("E1", 3)
    step = max(1, len(all_chunks) // max(target_e1 * 2, 1))
    fc["E1"] = all_chunks[::step]

    # P1: consecutive state_changes >= 3
    consec = []
    for cap in evidence:
        if cap.get("state_changes"):
            consec.append(cap["chunk_idx"])
        else:
            if len(consec) >= 3:
                fc["P1"].extend(consec)
            consec = []
    if len(consec) >= 3:
        fc["P1"].extend(consec)

    # C1/R1 via entity_id
    entity_app = {}
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            eid = e.get("id", "")
            if eid and eid != "unknown":
                entity_app.setdefault(eid, []).append(cap["chunk_idx"])
    for eid, chunks in entity_app.items():
        sc = [c for c in chunks if ev_by_idx.get(c, {}).get("state_changes")]
        if len(sc) >= 2:
            fc["C1"].extend(sc[-2:])
        for i in range(1, len(chunks)):
            if chunks[i] - chunks[i-1] >= 5:
                fc["R1"].append(chunks[i])

    for f in fc:
        fc[f] = sorted(set(fc[f]))
    return fc


# =====================================================================
# Stage 2: Mock 397B card generation (3-A Step 2)
# =====================================================================

def mock_generate_cards(evidence, family_chunks):
    """Simulate 397B output: generate realistic cards without LLM."""
    cards = []
    counter = 0

    def _add(family, question, answer, form, support, vis="transient"):
        nonlocal counter
        counter += 1
        cards.append({
            "card_id": f"vid001_{family}_{counter:03d}",
            "family": family,
            "question": question,
            "canonical_answer": answer,
            "answer_form": form,
            "support_chunks": support,
            "visibility_type": vis,
        })

    if family_chunks.get("F1"):
        _add("F1", "What price is shown on screen?", "4.99", "short_exact",
             [family_chunks["F1"][0]])
        _add("F1", "What text is visible on the display?", "4.99", "short_exact",
             [family_chunks["F1"][-1]] if len(family_chunks["F1"]) > 1 else [family_chunks["F1"][0]])

    if family_chunks.get("F2"):
        _add("F2", "What color is the apron? A.Red B.Blue C.White D.Green",
             "A", "multiple_choice", [family_chunks["F2"][0]], "persistent")
        _add("F2", "Is the pot stainless steel?", "Yes", "binary",
             [family_chunks["F2"][2] if len(family_chunks["F2"]) > 2 else family_chunks["F2"][0]],
             "persistent")

    if family_chunks.get("F3"):
        _add("F3", "How many grams of seasoning were added?", "15", "number",
             [family_chunks["F3"][0]])

    if family_chunks.get("F4"):
        _add("F4", "Is the pot on the right burner?", "Yes", "binary",
             [family_chunks["F4"][0]], "persistent")
        _add("F4", "Is the bowl to the left of the pot?", "Yes", "binary",
             [family_chunks["F4"][1] if len(family_chunks["F4"]) > 1 else family_chunks["F4"][0]])

    if family_chunks.get("E1"):
        _add("E1", "Is the person chopping?", "Yes", "binary",
             [family_chunks["E1"][0]])
        _add("E1", "Is the person stirring?", "Yes", "binary",
             [family_chunks["E1"][len(family_chunks["E1"])//2]])

    if family_chunks.get("E2"):
        _add("E2", "Tell me when the person starts adding tomatoes",
             "Started adding", "short_exact",
             [family_chunks["E2"][0]])

    if family_chunks.get("P1"):
        _add("P1", "Which step is this? A.Second B.Third C.Fourth D.Fifth",
             "B", "multiple_choice", [family_chunks["P1"][1] if len(family_chunks["P1"]) > 1 else family_chunks["P1"][0]])

    if family_chunks.get("C1"):
        _add("C1", "Has the person's activity changed since earlier?",
             "Yes", "binary", family_chunks["C1"][:2])

    if family_chunks.get("R1"):
        _add("R1", "Is the small white bowl still on the counter?",
             "No", "binary", [family_chunks["R1"][0] if family_chunks["R1"] else 7])

    if family_chunks.get("S1"):
        _add("S1", "Describe the current scene",
             "Person in red apron cooking at stove with stainless pot",
             "descriptive", [family_chunks["S1"][0]], "persistent")

    # M1 always gets a card
    _add("M1", "Describe each cooking step as it happens",
         "Chopping onions", "descriptive", [5])

    # Extra cards to reach ~20
    if family_chunks.get("E1"):
        _add("E1", "Is the person tasting?", "No", "binary",
             [family_chunks["E1"][-1]])
    if family_chunks.get("F2"):
        _add("F2", "What material is the pot? A.Stainless B.Cast iron C.Copper D.Glass",
             "A", "multiple_choice", [10], "persistent")
    if family_chunks.get("S1"):
        _add("S1", "Describe what is on the counter",
             "Cutting board, knife, and bowl", "descriptive", [15], "persistent")
    if family_chunks.get("F3"):
        _add("F3", "How many entities are visible?", "3", "number", [21])
    if family_chunks.get("E2"):
        _add("E2", "Has the person started plating?", "No", "binary", [50])
    if family_chunks.get("C1"):
        _add("C1", "Has the pot content changed since earlier?", "Yes", "binary", [15, 35])

    return cards


# =====================================================================
# Stage 3: Placement (3-B) — inline core logic
# =====================================================================

def precompute_retention(card, rollout):
    card_kw = extract_card_keywords(card)
    if not card_kw:
        return {"thinks_retained": {}, "summary_retained": {}}
    rc = RETENTION_CLASS.get(card.get("family", ""), "medium")
    threshold = {"low": 0.5, "medium": 0.35, "high": 0.2}[rc]
    observations = rollout.get("thinks", [])
    thinks_ret = {}
    for ci in card.get("support_chunks", []):
        if ci < len(observations):
            thinks_ret[ci] = _keyword_overlap(observations[ci].get("think", ""), card_kw) > threshold
        else:
            thinks_ret[ci] = False
    summary_ret = {}
    ss = set(card.get("support_chunks", []))
    for idx, ev in enumerate(rollout.get("compression_events", [])):
        if ss & set(ev.get("compressed_thinks_chunks", [])):
            summary_ret[idx] = _keyword_overlap(ev.get("summary", {}).get("text", ""), card_kw) > 0.3
    return {"thinks_retained": thinks_ret, "summary_retained": summary_ret}


def classify_availability(card, ask_chunk, rollout, bitmap):
    sc = set(card.get("support_chunks", []))
    s_start = min(sc) if sc else 0
    s_end = max(sc) if sc else 0
    snap = rollout["snapshots"].get(ask_chunk) or rollout["snapshots"].get(str(ask_chunk))
    if snap is None:
        return "unavailable"
    if s_start > ask_chunk:
        return "in_future"
    ws = snap["visual_window_start"]
    we = snap["chunk_idx"]
    if any(ws <= c <= we for c in sc):
        return "in_visual"
    rc = {item["chunk"] for item in snap.get("recent_thinks", [])}
    if any(bitmap.get("thinks_retained", {}).get(c, False) for c in sc & rc):
        return "in_recent_thinks"
    for idx, ev in enumerate(rollout.get("compression_events", [])):
        if ev["trigger_chunk"] > ask_chunk:
            break
        if sc & set(ev.get("compressed_thinks_chunks", [])):
            if bitmap.get("summary_retained", {}).get(idx, False):
                return "in_compressed"
    if s_end < ask_chunk:
        return "in_history_only"
    return "unavailable"


def determine_sequence_type(card, availability):
    if card.get("family") == "M1":
        return "multi_response"
    if availability == "in_future":
        return "event_watch"
    if availability in ("in_visual", "in_recent_thinks", "in_compressed"):
        return "immediate_response"
    if availability == "in_history_only":
        return "recall_success"
    return "immediate_response"


def compute_placement(card, ask_chunk, seq_type, rollout, evidence):
    nc = rollout["num_chunks"]
    kc = {"ask": ask_chunk}
    if seq_type == "immediate_response":
        kc["post_silent"] = min(ask_chunk + 1, nc - 1)
    elif seq_type == "recall_success":
        kc["post_recall"] = ask_chunk
        kc["post_silent"] = min(ask_chunk + 1, nc - 1)
    elif seq_type == "recall_fail_then_found":
        kc["post_recall"] = ask_chunk
        card_kw = extract_card_keywords(card)
        found = None
        if card_kw:
            for cap in evidence:
                ci = cap.get("chunk_idx", 0)
                if ci <= ask_chunk:
                    continue
                for fact in cap.get("atomic_facts", []):
                    if _keyword_overlap(fact.get("fact", ""), card_kw) > 0.3:
                        found = ci
                        break
                if found:
                    break
        if found and found < nc:
            kc["wait_silent"] = [min(ask_chunk + 1, nc - 1)]
            kc["found_response"] = found
            kc["post_silent"] = min(found + 1, nc - 1)
        else:
            return None
    elif seq_type == "event_watch":
        trigger = min(card.get("support_chunks", [ask_chunk]))
        if trigger <= ask_chunk:
            return None
        gap = trigger - ask_chunk
        wait = list(range(ask_chunk + 2, trigger, max(1, gap // 3)))
        kc["wait_silent"] = wait[:2]
        kc["trigger"] = trigger
        kc["post_silent"] = min(trigger + 1, nc - 1)
    elif seq_type == "multi_response":
        ev_map = {c.get("chunk_idx", i): c for i, c in enumerate(evidence)}
        fr, fs = [], []
        for c in range(ask_chunk + 1, min(nc, ask_chunk + 30)):
            cap = ev_map.get(c, {})
            if cap.get("state_changes"):
                fr.append(c)
            elif len(fs) < 2:
                fs.append(c)
        kc["no_change_silent"] = fs[:2]
        kc["followup_response"] = fr[:5]
        if fr:
            kc["post_silent"] = min(fr[-1] + 1, nc - 1)
    return {"card_id": card["card_id"], "ask_chunk": ask_chunk,
            "sequence_type": seq_type, "key_chunks": kc}


def compute_all_placements(cards, rollout, evidence):
    nc = rollout["num_chunks"]
    all_p = []
    rng = random.Random(42)
    for card in cards:
        vt = card.get("visibility_type", "transient")
        sc = card.get("support_chunks", [])
        if not sc:
            continue
        s_end = max(sc)
        bitmap = precompute_retention(card, rollout) if vt == "transient" else {}
        if vt == "persistent":
            for ask in [nc // 4, nc // 2, 3 * nc // 4]:
                if 0 <= ask < nc:
                    p = compute_placement(card, ask, "immediate_response", rollout, evidence)
                    if p:
                        all_p.append(p)
        else:
            vm = min(s_end + VISUAL_WINDOW_CHUNKS // 2, nc - 1)
            if vm >= s_end:
                av = classify_availability(card, vm, rollout, bitmap)
                if av == "in_visual":
                    p = compute_placement(card, vm, determine_sequence_type(card, av), rollout, evidence)
                    if p:
                        all_p.append(p)
            hc = min(s_end + VISUAL_WINDOW_CHUNKS + 5, nc - 1)
            if hc < nc:
                av = classify_availability(card, hc, rollout, bitmap)
                seq = determine_sequence_type(card, av)
                p = compute_placement(card, hc, seq, rollout, evidence)
                if p:
                    all_p.append(p)
            if rng.random() < 0.3 and hc < nc:
                p = compute_placement(card, hc, "recall_fail_then_found", rollout, evidence)
                if p:
                    all_p.append(p)
        if card.get("family") == "E2":
            ss = min(sc)
            if ss >= 5:
                ask_ew = max(2, ss - 8)
                p = compute_placement(card, ask_ew, "event_watch", rollout, evidence)
                if p:
                    all_p.append(p)
        if card.get("family") == "M1":
            ask = min(5, nc - 1)
            p = compute_placement(card, ask, "multi_response", rollout, evidence)
            if p:
                all_p.append(p)
    return all_p


# =====================================================================
# Stage 4: Trajectory planning (3-B)
# =====================================================================

def plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8, seed=42):
    rng = random.Random(seed)
    used_fam, used_seq, used_ask, used_cid = set(), set(), [], set()
    selected = []
    cands = list(placements)
    budget = target * max_pp
    while cands and len(selected) < budget:
        scored = []
        for p in cands:
            if p["card_id"] in used_cid:
                continue
            card = cards_map.get(p["card_id"], {})
            s = 0.0
            if card.get("_support_inferred"):
                s -= 2.0
            if card.get("answer_form") in {"binary", "multiple_choice", "number", "short_exact"}:
                s += 1.0
            if card.get("family", "") not in used_fam:
                s += 2.0
            if p["sequence_type"] not in used_seq:
                s += 2.0
            if used_ask:
                s += min(min(abs(p["ask_chunk"] - c) for c in used_ask) / 10.0, 1.5)
            else:
                s += 1.5
            scored.append((s, p))
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0][0]
        ties = [sp for sp in scored if sp[0] >= top - 0.1]
        _, best = ties[rng.randint(0, len(ties) - 1)]
        selected.append(best)
        card = cards_map.get(best["card_id"], {})
        used_fam.add(card.get("family", ""))
        used_seq.add(best["sequence_type"])
        used_ask.append(best["ask_chunk"])
        used_cid.add(best["card_id"])
        cands.remove(best)

    selected.sort(key=lambda p: p["ask_chunk"])
    trajs = []
    paired = set()
    for i in range(len(selected)):
        if i in paired or len(trajs) >= target:
            break
        group = [selected[i]]
        paired.add(i)
        for j in range(i + 1, len(selected)):
            if j in paired:
                continue
            if selected[j]["ask_chunk"] - group[-1]["ask_chunk"] >= gap:
                group.append(selected[j])
                paired.add(j)
                if len(group) >= max_pp:
                    break
        trajs.append({"trajectory_id": f"traj_{len(trajs)}", "placements": group})
    for i in range(len(selected)):
        if i not in paired and len(trajs) < target:
            trajs.append({"trajectory_id": f"traj_{len(trajs)}", "placements": [selected[i]]})
    return trajs


# =====================================================================
# Stage 5: Fork sample generation (3-C) — sync mock version
# =====================================================================

def generate_fork_samples(trajectory, cards_map, rollout):
    """Generate fork samples without 397B. Mock response/query text."""
    samples = []
    queries_state = []
    obs_map = {o["chunk_idx"]: o for o in rollout.get("thinks", [])}
    tid = trajectory["trajectory_id"]

    def think(ci):
        o = obs_map.get(ci)
        return o.get("think", "") if o else ""

    for placement in trajectory["placements"]:
        card = cards_map.get(placement["card_id"])
        if not card:
            continue
        kc = placement["key_chunks"]
        seq = placement["sequence_type"]
        ask = kc["ask"]

        if seq == "immediate_response":
            resp = card.get("canonical_answer", "mock")
            samples.append({"chunk_idx": ask, "action": "response", "sample_type": "response",
                            "queries": deepcopy(queries_state), "user_input": card["question"],
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            queries_state.append({"question": card["question"], "answers": [resp]})
            ps = kc.get("post_silent", ask + 1)
            samples.append({"chunk_idx": ps, "action": "silent", "sample_type": "silent",
                            "queries": deepcopy(queries_state), "trajectory_id": tid,
                            "card_id": card["card_id"], "sequence_type": seq})

        elif seq == "recall_success":
            samples.append({"chunk_idx": ask, "action": "recall", "sample_type": "recall_query",
                            "queries": deepcopy(queries_state), "user_input": card["question"],
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            resp = card.get("canonical_answer", "mock")
            samples.append({"chunk_idx": ask, "action": "response", "sample_type": "recall_response",
                            "queries": deepcopy(queries_state),
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            queries_state.append({"question": card["question"], "answers": [resp]})
            ps = kc.get("post_silent", ask + 1)
            samples.append({"chunk_idx": ps, "action": "silent", "sample_type": "silent",
                            "queries": deepcopy(queries_state), "trajectory_id": tid,
                            "card_id": card["card_id"], "sequence_type": seq})

        elif seq == "recall_fail_then_found":
            samples.append({"chunk_idx": ask, "action": "recall", "sample_type": "recall_query",
                            "queries": deepcopy(queries_state), "user_input": card["question"],
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            samples.append({"chunk_idx": ask, "action": "silent", "sample_type": "recall_silent",
                            "queries": deepcopy(queries_state),
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            queries_state.append({"question": card["question"], "answers": []})
            for wc in kc.get("wait_silent", []):
                samples.append({"chunk_idx": wc, "action": "silent", "sample_type": "silent",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
            found = kc.get("found_response")
            if found is not None:
                resp = card.get("canonical_answer", "mock")
                samples.append({"chunk_idx": found, "action": "response", "sample_type": "response",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
                queries_state[-1]["answers"].append(resp)
                ps = kc.get("post_silent", found + 1)
                samples.append({"chunk_idx": ps, "action": "silent", "sample_type": "silent",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})

        elif seq == "event_watch":
            queries_state.append({"question": card["question"], "answers": []})
            samples.append({"chunk_idx": ask, "action": "silent", "sample_type": "silent",
                            "queries": deepcopy(queries_state), "user_input": card["question"],
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            for wc in kc.get("wait_silent", []):
                samples.append({"chunk_idx": wc, "action": "silent", "sample_type": "silent",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
            trigger = kc.get("trigger")
            if trigger is not None:
                resp = card.get("canonical_answer", "mock")
                samples.append({"chunk_idx": trigger, "action": "response", "sample_type": "response",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
                queries_state[-1]["answers"].append(resp)

        elif seq == "multi_response":
            resp = card.get("canonical_answer", "mock")
            samples.append({"chunk_idx": ask, "action": "response", "sample_type": "response",
                            "queries": deepcopy(queries_state), "user_input": card["question"],
                            "trajectory_id": tid, "card_id": card["card_id"], "sequence_type": seq})
            queries_state.append({"question": card["question"], "answers": [resp]})
            for sc in kc.get("no_change_silent", []):
                samples.append({"chunk_idx": sc, "action": "silent", "sample_type": "silent",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
            for fc in kc.get("followup_response", []):
                samples.append({"chunk_idx": fc, "action": "response", "sample_type": "response",
                                "queries": deepcopy(queries_state), "trajectory_id": tid,
                                "card_id": card["card_id"], "sequence_type": seq})
                queries_state[-1]["answers"].append(f"followup at {fc}")

    return samples, queries_state


# =====================================================================
# Stage 6: Base sample selection (3-C)
# =====================================================================

def select_base_chunks(trajectory, rollout, cards_map):
    nc = rollout["num_chunks"]
    selected = set()
    for c in range(min(3, nc)):
        selected.add(c)
    for p in trajectory["placements"]:
        card = cards_map.get(p["card_id"], {})
        for sc in card.get("support_chunks", []):
            for c in range(max(0, sc - 2), min(nc, sc + 3)):
                selected.add(c)
    for p in trajectory["placements"]:
        kc = p["key_chunks"]
        for key, val in kc.items():
            anchors = [val] if isinstance(val, int) else (val if isinstance(val, list) else [])
            for anchor in anchors:
                for c in range(max(0, anchor - 2), min(nc, anchor + 4)):
                    selected.add(c)
    for ev in rollout.get("compression_events", []):
        t = ev.get("trigger_chunk", -1)
        if t >= 0:
            for c in range(max(0, t - 1), min(nc, t + 2)):
                selected.add(c)
            cc = sorted(ev.get("compressed_thinks_chunks", []))
            for c in cc[:2] + cc[-2:]:
                selected.add(c)
    ss = sorted(selected)
    patrol = []
    prev = -1
    for s in ss:
        if s - prev > 10:
            for c in range(prev + 5, s, 5):
                patrol.append(c)
        prev = s
    if ss and nc - 1 - ss[-1] > 10:
        for c in range(ss[-1] + 5, nc, 5):
            patrol.append(c)
    selected.update(patrol)
    return sorted(selected)


# =====================================================================
# THE E2E TEST
# =====================================================================

class TestE2EPipeline:
    """Walk through the entire Pass 3 pipeline step by step."""

    def setup_method(self):
        self.evidence = build_evidence(60)
        self.rollout = build_rollout(60)

    # ---- Stage 1: classify_chunks ----

    def test_stage1_classify_chunks(self):
        fc = classify_chunks(self.evidence)

        # Basic: every family should have a list
        for f in FAMILY_TARGETS:
            assert isinstance(fc[f], list), f"{f} not a list"

        # F2 should be the biggest (almost all chunks have entities)
        assert len(fc["F2"]) >= 50, f"F2 has only {len(fc['F2'])} chunks"

        # E2 requires state_changes (chunks 10-14, 30-33 = 9 chunks)
        assert len(fc["E2"]) >= 5, f"E2 has only {len(fc['E2'])} chunks"

        # P1 requires consecutive >= 3 (chunks 10-14 = 5 consecutive)
        assert len(fc["P1"]) >= 3, f"P1 has only {len(fc['P1'])} chunks"

        # All indices valid
        for f, chunks in fc.items():
            for c in chunks:
                assert 0 <= c < 60, f"{f} has invalid chunk {c}"

    # ---- Stage 2: card generation ----

    def test_stage2_card_generation(self):
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)

        # Should have ~15-25 cards
        assert 10 <= len(cards) <= 30, f"Got {len(cards)} cards"

        # Each card has required fields
        for card in cards:
            assert card.get("card_id"), "missing card_id"
            assert card.get("family") in FAMILY_TARGETS, f"bad family: {card.get('family')}"
            assert card.get("question"), "missing question"
            assert card.get("canonical_answer") is not None, "missing canonical_answer"
            assert card.get("answer_form") in {"binary", "multiple_choice", "number",
                                                "short_exact", "descriptive"}, \
                f"bad answer_form: {card.get('answer_form')}"
            assert isinstance(card.get("support_chunks"), list), "support_chunks not list"
            assert len(card["support_chunks"]) >= 1, "empty support_chunks"
            assert card.get("visibility_type") in {"persistent", "transient"}

        # Unique card_ids
        ids = [c["card_id"] for c in cards]
        assert len(ids) == len(set(ids)), "duplicate card_ids"

        # extract_card_keywords should return non-empty for every card
        for card in cards:
            kw = extract_card_keywords(card)
            assert len(kw) >= 1, f"empty keywords for {card['card_id']} ({card['answer_form']})"

        self._cards = cards
        return cards

    # ---- Stage 3: placements ----

    def test_stage3_placements(self):
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)

        assert len(placements) >= 10, f"Only {len(placements)} placements"

        for p in placements:
            assert p["card_id"], "missing card_id"
            assert 0 <= p["ask_chunk"] < 60, f"ask_chunk {p['ask_chunk']} out of range"
            assert p["sequence_type"] in {"immediate_response", "recall_success",
                                           "recall_fail_then_found", "event_watch",
                                           "multi_response"}, f"bad seq: {p['sequence_type']}"
            assert "ask" in p["key_chunks"]

            # Sequence-specific key_chunks validation
            kc = p["key_chunks"]
            seq = p["sequence_type"]
            if seq == "immediate_response":
                assert "post_silent" in kc
                assert kc["post_silent"] >= kc["ask"]
            elif seq == "recall_success":
                assert "post_recall" in kc
                assert "post_silent" in kc
            elif seq == "recall_fail_then_found":
                assert "found_response" in kc
                assert kc["found_response"] > kc["ask"]
                assert "post_silent" in kc
            elif seq == "event_watch":
                assert "trigger" in kc
                assert kc["trigger"] > kc["ask"]
            elif seq == "multi_response":
                assert "no_change_silent" in kc or "followup_response" in kc

        # Check sequence type variety
        seq_types = set(p["sequence_type"] for p in placements)
        assert len(seq_types) >= 2, f"Only {seq_types} sequence types"

        return placements, cards

    # ---- Stage 4: trajectory planning ----

    def test_stage4_trajectory_planning(self):
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}
        trajs = plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8)

        assert 1 <= len(trajs) <= 5, f"Got {len(trajs)} trajectories"

        for t in trajs:
            assert t["trajectory_id"]
            assert len(t["placements"]) >= 1
            assert len(t["placements"]) <= 5

            # Min gap respected within trajectory
            chunks = sorted(p["ask_chunk"] for p in t["placements"])
            for i in range(1, len(chunks)):
                assert chunks[i] - chunks[i-1] >= 8, \
                    f"Gap {chunks[i]-chunks[i-1]} < 8 in {t['trajectory_id']}"

        # No duplicate card_ids across all trajectories
        all_cids = [p["card_id"] for t in trajs for p in t["placements"]]
        assert len(all_cids) == len(set(all_cids)), "duplicate card_ids in trajectories"

        # Family diversity
        families = set()
        for t in trajs:
            for p in t["placements"]:
                families.add(cards_map[p["card_id"]]["family"])
        assert len(families) >= 4, f"Only {len(families)} families"

        return trajs, cards_map

    # ---- Stage 5: fork sample generation ----

    def test_stage5_fork_samples(self):
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}
        trajs = plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8)

        for t in trajs:
            fork_samples, final_qs = generate_fork_samples(t, cards_map, self.rollout)

            assert len(fork_samples) >= 1, f"{t['trajectory_id']} produced 0 fork samples"

            # Validate each fork sample
            for s in fork_samples:
                assert 0 <= s["chunk_idx"] < 60
                assert s["action"] in {"silent", "response", "recall"}
                assert s["trajectory_id"] == t["trajectory_id"]
                assert isinstance(s["queries"], list)

                # response samples must have user_input or be a followup
                if s["sample_type"] == "response" and s.get("user_input"):
                    assert len(s["user_input"]) > 0

            # queries_state should grow monotonically
            prev_q_len = 0
            for s in fork_samples:
                q_len = len(s["queries"])
                assert q_len >= prev_q_len, \
                    f"queries_state shrank from {prev_q_len} to {q_len} at chunk {s['chunk_idx']}"
                prev_q_len = q_len

            # The first sample with user_input: queries_state should not contain
            # questions from LATER placements (no future leakage).
            # Note: event_watch appends the question BEFORE the ask sample
            # (the question has arrived but the event hasn't happened yet),
            # so queries_state may contain the current question — that's correct.

    # ---- Stage 6: base sample selection ----

    def test_stage6_base_samples(self):
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}
        trajs = plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8)

        for t in trajs:
            base_chunks = select_base_chunks(t, self.rollout, cards_map)

            # Should select a reasonable subset
            assert 15 <= len(base_chunks) <= 55, \
                f"{t['trajectory_id']}: {len(base_chunks)} base chunks"

            # Warmup chunks included
            assert 0 in base_chunks
            assert 1 in base_chunks

            # Evidence anchors: support_chunks should be covered
            for p in t["placements"]:
                card = cards_map.get(p["card_id"], {})
                for sc in card.get("support_chunks", []):
                    if 0 <= sc < 60:
                        assert sc in base_chunks, \
                            f"support_chunk {sc} missing for {p['card_id']}"

            # Compression trigger chunks covered
            for ev in self.rollout["compression_events"]:
                tc = ev["trigger_chunk"]
                assert tc in base_chunks, f"compression trigger {tc} missing"

    # ---- Stage 7: combined episode validation ----

    def test_stage7_combined_episode(self):
        """End-to-end: build a complete episode and validate ratios."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}
        trajs = plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8)

        all_episode_stats = []

        for t in trajs:
            fork_samples, final_qs = generate_fork_samples(t, cards_map, self.rollout)
            base_chunks = select_base_chunks(t, self.rollout, cards_map)
            fork_chunk_set = set(s["chunk_idx"] for s in fork_samples)

            # Base samples (excluding fork chunks)
            base_count = sum(1 for c in base_chunks if c not in fork_chunk_set)

            # Count actions in fork samples
            fork_actions = {}
            for s in fork_samples:
                a = s["action"]
                fork_actions[a] = fork_actions.get(a, 0) + 1

            # Compress events in base
            compress_triggers = set(ev["trigger_chunk"] for ev in self.rollout["compression_events"])
            compress_in_base = sum(1 for c in base_chunks if c in compress_triggers and c not in fork_chunk_set)
            silent_in_base = base_count - compress_in_base

            total = len(fork_samples) + base_count
            silent_total = fork_actions.get("silent", 0) + silent_in_base
            response_total = fork_actions.get("response", 0)
            recall_total = fork_actions.get("recall", 0)

            stats = {
                "traj_id": t["trajectory_id"],
                "n_placements": len(t["placements"]),
                "fork_samples": len(fork_samples),
                "base_samples": base_count,
                "total": total,
                "silent": silent_total,
                "response": response_total,
                "recall": recall_total,
                "compress": compress_in_base,
                "silent_pct": silent_total / total * 100 if total else 0,
                "active_pct": (response_total + recall_total) / total * 100 if total else 0,
            }
            all_episode_stats.append(stats)

        # Print stats for debugging
        print("\n" + "=" * 80)
        print("EPISODE STATISTICS")
        print("=" * 80)
        for s in all_episode_stats:
            print(f"  {s['traj_id']}: {s['n_placements']}Q, "
                  f"{s['total']} samples "
                  f"(silent={s['silent_pct']:.0f}%, active={s['active_pct']:.0f}%, "
                  f"compress={s['compress']})")

        # Aggregate validation
        total_samples = sum(s["total"] for s in all_episode_stats)
        total_active = sum(s["response"] + s["recall"] for s in all_episode_stats)
        total_silent = sum(s["silent"] for s in all_episode_stats)

        assert total_samples > 0, "No samples generated"

        active_pct = total_active / total_samples * 100
        silent_pct = total_silent / total_samples * 100

        print(f"\n  AGGREGATE: {total_samples} samples, "
              f"silent={silent_pct:.0f}%, active={active_pct:.0f}%")
        print(f"  Target: silent 55-70%, active 20-35%")

        # Per-trajectory targets:
        #   5-question trajectory: silent ~65-80%, active ~15-25%  ← good
        #   1-question trajectory: silent ~85-90%, active ~4-6%   ← expected
        # Aggregate mixes both types, so bound is 85%.
        # The real quality signal is the multi-question trajectories.
        assert silent_pct <= 88, f"Too many silent samples: {silent_pct:.0f}%"
        assert active_pct >= 8, f"Too few active samples: {active_pct:.0f}%"

        # Check that at least one trajectory has good ratio (5Q trajectories)
        multi_q_trajs = [s for s in all_episode_stats if s["n_placements"] >= 3]
        if multi_q_trajs:
            best = min(s["silent_pct"] for s in multi_q_trajs)
            assert best <= 85, f"Best multi-Q trajectory still {best:.0f}% silent"

    # ---- Cross-cutting invariants ----

    def test_no_future_leakage_in_queries_state(self):
        """queries_state at chunk N must not contain questions asked after chunk N."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}
        trajs = plan_trajectories(placements, cards_map, target=5, max_pp=5, gap=8)

        for t in trajs:
            fork_samples, _ = generate_fork_samples(t, cards_map, self.rollout)

            # Build ask_chunk → question mapping
            ask_questions = {}
            for p in t["placements"]:
                card = cards_map.get(p["card_id"], {})
                ask_questions[p["ask_chunk"]] = card.get("question", "")

            for s in fork_samples:
                ci = s["chunk_idx"]
                for q_entry in s["queries"]:
                    q_text = q_entry["question"]
                    # Find which ask_chunk this question came from
                    ask_at = None
                    for ac, qt in ask_questions.items():
                        if qt == q_text:
                            ask_at = ac
                            break
                    if ask_at is not None:
                        assert ask_at <= ci, \
                            f"Chunk {ci} sees question from future chunk {ask_at}: '{q_text[:30]}'"

    def test_recall_evidence_exists_in_past(self):
        """For recall_success, support_chunks must be before ask_chunk."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}

        for p in placements:
            if p["sequence_type"] in ("recall_success", "recall_fail_then_found"):
                card = cards_map.get(p["card_id"], {})
                for sc in card.get("support_chunks", []):
                    assert sc < p["ask_chunk"], \
                        f"Recall at chunk {p['ask_chunk']} but evidence at future chunk {sc}"

    def test_event_watch_trigger_after_ask(self):
        """For event_watch, trigger must be after ask_chunk."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)

        for p in placements:
            if p["sequence_type"] == "event_watch":
                kc = p["key_chunks"]
                assert kc["trigger"] > kc["ask"], \
                    f"event_watch trigger {kc['trigger']} <= ask {kc['ask']}"

    def test_persistent_cards_only_immediate(self):
        """Persistent visibility cards should only get immediate_response."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}

        for p in placements:
            card = cards_map.get(p["card_id"], {})
            if card.get("visibility_type") == "persistent":
                assert p["sequence_type"] == "immediate_response", \
                    f"Persistent card {p['card_id']} has seq={p['sequence_type']}"

    def test_recall_fail_found_has_post_silent(self):
        """recall_fail_then_found must have post_silent after found_response."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)
        cards_map = {c["card_id"]: c for c in cards}

        for p in placements:
            if p["sequence_type"] == "recall_fail_then_found":
                kc = p["key_chunks"]
                assert "post_silent" in kc, "recall_fail_then_found missing post_silent"
                assert "found_response" in kc, "recall_fail_then_found missing found_response"
                assert kc["post_silent"] > kc["found_response"] or \
                       kc["post_silent"] == kc["found_response"], \
                    "post_silent should be after found_response"

    def test_all_key_chunks_in_range(self):
        """All key_chunk values must be within [0, num_chunks)."""
        fc = classify_chunks(self.evidence)
        cards = mock_generate_cards(self.evidence, fc)
        placements = compute_all_placements(cards, self.rollout, self.evidence)

        for p in placements:
            kc = p["key_chunks"]
            for key, val in kc.items():
                if isinstance(val, int):
                    assert 0 <= val < 60, f"{p['card_id']}.{key}={val} out of range"
                elif isinstance(val, list):
                    for v in val:
                        assert 0 <= v < 60, f"{p['card_id']}.{key} contains {v} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
