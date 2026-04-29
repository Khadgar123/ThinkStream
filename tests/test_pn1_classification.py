"""Test PN1 (Proactive Narration) family — v12.5 addition.

PN1 picks chunks with novelty signals (state_changes, first-appearance
visible_entities, first-appearance OCR) for short observation cards.
Adds non-silent training density to lower the 85% silent dominance.

Run: python tests/test_pn1_classification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_data_v5.pass3a_cards import (
    FAMILY_FORCE_ATTEMPT,
    FAMILY_TARGETS,
    RETENTION_CLASS,
    classify_chunks,
)


def _ev(idx, *, entities=(), ocr=(), state_changes=()):
    """Tiny helper to build evidence chunk dicts."""
    return {
        "chunk_idx": idx,
        "time": [idx * 2, idx * 2 + 2],
        "visible_entities": [
            {"id": eid, "desc": eid} for eid in entities
        ],
        "ocr": list(ocr),
        "atomic_facts": [{"fact": f"fact_{idx}", "confidence": 0.9}],
        "state_changes": list(state_changes),
    }


def test_pn1_registered_in_family_targets():
    """PN1 must be in FAMILY_TARGETS with target=5 + in FORCE_ATTEMPT
    + RETENTION_CLASS."""
    assert "PN1" in FAMILY_TARGETS, "PN1 missing from FAMILY_TARGETS"
    assert FAMILY_TARGETS["PN1"] == 5, (
        f"PN1 target should be 5 (LiveCC-style density), got {FAMILY_TARGETS['PN1']}"
    )
    assert "PN1" in FAMILY_FORCE_ATTEMPT, "PN1 should be in FORCE_ATTEMPT"
    assert "PN1" in RETENTION_CLASS, "PN1 missing from RETENTION_CLASS"
    print(f"  PASS PN1 registered: target={FAMILY_TARGETS['PN1']}, "
          f"retention={RETENTION_CLASS['PN1']}")


def test_pn1_picks_state_change_chunks():
    """state_changes is a strong novelty signal → PN1 picks those chunks."""
    evidence = [
        _ev(0, entities=("person_1",)),
        _ev(1, entities=("person_1",)),  # no novelty
        _ev(2, entities=("person_1",), state_changes=["door_opens"]),  # novelty!
        _ev(3, entities=("person_1",)),
        _ev(4, entities=("person_1",), state_changes=["light_on"]),  # novelty!
    ]
    fc = classify_chunks(evidence)
    pn1 = fc.get("PN1", [])
    # chunks 2 and 4 had state_changes → must be in PN1 picks
    assert 2 in pn1, f"chunk 2 (state_change) should be in PN1, got {pn1}"
    assert 4 in pn1, f"chunk 4 (state_change) should be in PN1, got {pn1}"
    # chunk 0 also has novelty (first appearance of person_1)
    assert 0 in pn1
    print(f"  PASS PN1 state_change picks: {pn1}")


def test_pn1_picks_first_entity_appearances():
    """First time a visible_entity ID appears = novelty signal."""
    evidence = [
        _ev(0, entities=("apple",)),                    # apple first → novel
        _ev(1, entities=("apple",)),                    # not novel
        _ev(2, entities=("apple", "knife")),             # knife first → novel
        _ev(3, entities=("apple", "knife")),             # not novel
        _ev(4, entities=("apple", "knife", "plate")),    # plate first → novel
    ]
    fc = classify_chunks(evidence)
    pn1 = fc.get("PN1", [])
    assert 0 in pn1
    assert 2 in pn1
    assert 4 in pn1
    # No novelty at 1 or 3 (only repeat entities, no state changes)
    assert 1 not in pn1
    assert 3 not in pn1
    print(f"  PASS PN1 entity-first picks: {pn1}")


def test_pn1_picks_first_ocr_appearances():
    """First time OCR text appears = novelty signal.

    Note: novelty filter requires len(text) > 2 to skip noise like '$5'
    or single chars. Test uses ≥3-char texts so the OCR signal fires.
    """
    evidence = [
        _ev(0, entities=("e",), ocr=["Welcome"]),                # novel (entity "e" first too)
        _ev(1, entities=("e",), ocr=["Welcome"]),                # repeat
        _ev(2, entities=("e",), ocr=["Welcome", "Sale Today"]),  # "Sale Today" new
        _ev(3, entities=("e",), ocr=[]),
    ]
    fc = classify_chunks(evidence)
    pn1 = fc.get("PN1", [])
    assert 0 in pn1
    assert 2 in pn1, f"chunk 2 with new OCR should be in PN1, got {pn1}"
    assert 1 not in pn1
    print(f"  PASS PN1 OCR-first picks: {pn1}")


def test_pn1_caps_at_8_candidates():
    """classify_chunks caps PN1 at 8 candidates (FAMILY_TARGETS keeps 5)."""
    # Build 20 chunks each with a unique entity → all are first-appearance
    evidence = [_ev(i, entities=(f"ent_{i}",)) for i in range(20)]
    fc = classify_chunks(evidence)
    pn1 = fc.get("PN1", [])
    assert len(pn1) <= 8, f"PN1 candidate cap exceeded: {len(pn1)}"
    print(f"  PASS PN1 candidate cap: {len(pn1)} ≤ 8")


def test_pn1_static_video_returns_few():
    """A video with no state changes / repeat entities only → few PN1 picks."""
    evidence = [_ev(i, entities=("static_scene",)) for i in range(20)]
    fc = classify_chunks(evidence)
    pn1 = fc.get("PN1", [])
    # Only chunk 0 has novelty (static_scene first appearance)
    assert pn1 == [0], f"static video should yield only chunk 0: got {pn1}"
    print(f"  PASS static video → PN1={pn1}")


def test_total_cards_per_video_increased():
    """v12.5 bump: total target should be ≥40 cards/video (was 26 in v12.4)."""
    total = sum(FAMILY_TARGETS.values())
    assert total >= 40, (
        f"v12.5 should bump cards/video to ≥40 to lower silent ratio, got {total}"
    )
    assert total <= 50, (
        f"too many cards/video ({total}) — pass3b density caps will throttle"
    )
    print(f"  PASS cards/video target: {total}")


def main():
    tests = [
        test_pn1_registered_in_family_targets,
        test_pn1_picks_state_change_chunks,
        test_pn1_picks_first_entity_appearances,
        test_pn1_picks_first_ocr_appearances,
        test_pn1_caps_at_8_candidates,
        test_pn1_static_video_returns_few,
        test_total_cards_per_video_increased,
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
