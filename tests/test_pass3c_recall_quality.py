import asyncio

from thinkstream.models.agent_loop import filter_archive_by_time_range, time_range_retrieve
from scripts.agent_data.pass3c_samples import (
    RECALL_RETURN_CHUNKS,
    _clean_post_recall_think,
    _post_recall_think_for,
    _post_recall_think_prompt,
    _post_recall_think_via_llm,
    _recall_chunks_for_request,
    _recall_query_for,
    _recall_result_for,
    _recall_wait_query_for,
)


def test_recall_args_for_count_are_start_end_only():
    card = {
        "family": "F5",
        "answer_form": "number",
        "question_type": "multi_emit",
        "question": "How many times has the person opened the drawer so far?",
        "canonical_answer": "3",
        "grounding_frames": [2, 4, 6],
    }

    args = _recall_query_for(card, current_chunk=8)

    assert args == {"start_time": 2, "end_time": 6}
    assert "query" not in args


def test_wait_window_excludes_current_visual_window():
    card = {
        "family": "F7",
        "answer_form": "binary",
        "question": "Has the person started pouring water into the glass?",
        "canonical_answer": "Yes",
        "grounding_frames": [20],
    }

    args = _recall_wait_query_for(card, chunk_idx=12)

    assert args == {"start_time": 0, "end_time": 4}


def test_retriever_closed_window_includes_endpoint_chunk():
    archive = [
        {"chunk": i, "text": f"chunk {i} drawer action"}
        for i in range(5)
    ]

    filtered = filter_archive_by_time_range(
        archive,
        {"start_time": 2, "end_time": 3},
        margin_chunks=0,
    )

    assert [item["chunk"] for item in filtered] == [2, 3]


def test_time_range_retriever_uniformly_caps_returned_chunks():
    archive = [
        {"chunk": i, "time": f"{i}-{i + 1}", "text": f"chunk {i}"}
        for i in range(12)
    ]

    result = time_range_retrieve(
        {"start_time": 0, "end_time": 12},
        archive,
        max_results=4,
    )

    assert result["source"] == "historical_frames"
    assert len(result["returned_chunks"]) == 4
    assert result["returned_chunks"] == [0, 4, 7, 11]
    assert result["retrieval_mode"] == "time_range_uniform"


def test_pass3c_recall_result_uses_only_requested_timerange():
    rollout = {
        "num_chunks": 20,
        "thinks": [
            {"chunk_idx": i, "think": f"visible event at chunk {i}"}
            for i in range(20)
        ],
    }
    card = {
        "family": "P1",
        "answer_form": "short_exact",
        "canonical_answer": "red",
        "grounding_frames": [2, 8, 14],
    }
    args = {"start_time": 2, "end_time": 15}

    raw_chunks = _recall_chunks_for_request(rollout, args, current_chunk=18)
    result = _recall_result_for(
        card,
        rollout,
        "noisy",
        current_chunk=18,
        recall_query=args,
    )

    assert result["returned_chunks"] == raw_chunks
    assert len(result["returned_chunks"]) <= RECALL_RETURN_CHUNKS
    assert all(2 <= c <= 15 for c in result["returned_chunks"])


def test_post_recall_think_uses_recalled_visual_hint():
    card = {
        "family": "P1",
        "answer_form": "short_exact",
        "canonical_answer": "red",
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [4, 5],
        "text_content": "[4-5] The person wore a red apron while chopping onions.",
    }

    think = _post_recall_think_for(
        card,
        recall_result,
        "red",
        card_id="card0",
        chunk_idx=12,
    )

    assert "apron" in think or "chopping" in think
    assert "chunk" not in think.lower()
    assert "recalled" not in think.lower()
    assert "frames" not in think.lower()


def test_post_recall_teacher_prompt_asks_for_answer_relevant_recall_fact():
    card = {
        "family": "P1",
        "answer_form": "multiple_choice",
        "question": "Which color was the apron?",
        "canonical_answer": "red",
        "options": ["A) red", "B) blue"],
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [4, 5],
        "text_content": "[4-5] The person wore a red apron while chopping onions.",
    }

    prompt = _post_recall_think_prompt(
        card,
        {"start_time": 4, "end_time": 5},
        recall_result,
        action="response",
    )

    assert "Attached images" in prompt
    assert "source of truth" in prompt
    assert "8-36 words" in prompt
    assert "visual fact that supports that answer" in prompt
    assert "Do not mention the retrieval/tool source" in prompt
    assert "Teacher-only response target" in prompt
    assert "red apron" in prompt.lower()
    assert '"query"' not in prompt


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def _call_one(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_llm_post_recall_think_accepts_short_objective_sentence():
    card = {
        "card_id": "c0",
        "family": "P1",
        "answer_form": "short_exact",
        "question": "What color was the apron?",
        "canonical_answer": "red",
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [4, 5],
        "text_content": "[4-5] The person wore a red apron while chopping onions.",
    }
    client = _FakeClient("The person is wearing a red apron while chopping onions.")

    think = asyncio.run(
        _post_recall_think_via_llm(
            card,
            {"start_time": 4, "end_time": 5},
            recall_result,
            "red",
            client,
            "vid0",
            12,
            action="response",
        )
    )

    assert think == "The person is wearing a red apron while chopping onions."
    assert "recalled" not in think.lower()
    assert "frames" not in think.lower()
    prompt = client.calls[0]["messages"][0]["content"]
    assert "red apron" in prompt.lower()
    assert "recalled chunk" not in prompt.lower()


def test_llm_post_recall_think_does_not_content_filter_absence_by_default():
    card = {
        "card_id": "c0",
        "family": "P1",
        "answer_form": "short_exact",
        "question": "What color was the apron?",
        "canonical_answer": "red",
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [4, 5],
        "text_content": "[4-5] The person wore a red apron while chopping onions.",
    }
    client = _FakeClient("The apron is visible but does not specify its color.")

    think = asyncio.run(
        _post_recall_think_via_llm(
            card,
            {"start_time": 4, "end_time": 5},
            recall_result,
            "red",
            client,
            "vid0",
            12,
            action="response",
        )
    )

    assert think == "The apron is visible but does not specify its color."


def test_llm_post_recall_think_does_not_filter_mc_absence_by_default():
    card = {
        "card_id": "c0",
        "family": "CR4",
        "answer_form": "multiple_choice",
        "question": "Which grooming tool was used next?",
        "canonical_answer": "A blue grooming brush",
        "options": ["A) A blue grooming brush", "B) A black hose"],
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [76, 89],
        "text_content": "[89-90] The groomer uses a blue brush on the dog's back.",
    }
    client = _FakeClient("A black hose is visible, with no other grooming tool visible.")

    think = asyncio.run(
        _post_recall_think_via_llm(
            card,
            {"start_time": 76, "end_time": 89},
            recall_result,
            "A",
            client,
            "vid0",
            149,
            action="response",
        )
    )

    assert think == "A black hose is visible, with no other grooming tool visible."


def test_llm_post_recall_think_does_not_semantically_filter_mc_option_overlap_by_default():
    card = {
        "card_id": "c0",
        "family": "CR4",
        "answer_form": "multiple_choice",
        "question": "Which grooming tool was used next?",
        "canonical_answer": "A blue grooming brush",
        "correct_option": "A",
        "options": ["A) A blue grooming brush", "B) A black vacuum hose"],
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [76, 89],
        "text_content": "[89-90] The groomer uses a blue brush on the dog's back.",
    }
    client = _FakeClient("The groomer is using a black vacuum hose near the dog.")

    think = asyncio.run(
        _post_recall_think_via_llm(
            card,
            {"start_time": 76, "end_time": 89},
            recall_result,
            "A",
            client,
            "vid0",
            149,
            action="response",
        )
    )

    assert think == "The groomer is using a black vacuum hose near the dog."


def test_post_recall_think_allows_correct_option_with_partial_distractor():
    card = {
        "card_id": "c0",
        "family": "CR5",
        "answer_form": "multiple_choice",
        "question": "What text is fully visible on the banner?",
        "canonical_answer": "STEREO",
        "correct_option": "A",
        "options": ["A) STEREO", "B) ST", "C) NERVLING", "D) STE"],
    }

    think = _clean_post_recall_think(
        "The banner behind the performers clearly displays the text 'STEREO'.",
        card,
    )

    assert think == "The banner behind the performers clearly displays the text 'STEREO'."


def test_post_recall_think_allows_correct_option_with_close_distractor():
    card = {
        "card_id": "c0",
        "family": "CR5",
        "answer_form": "multiple_choice",
        "question": "What name and title are displayed in the overlay?",
        "canonical_answer": "Orene Kearn, Fashion Stylist",
        "correct_option": "C",
        "options": [
            "A) Orene Kern, Fashion Stylist",
            "B) The Stylist, eHow",
            "C) Orene Kearn, Fashion Stylist",
            "D) Steven Watkins, Director",
        ],
    }

    think = _clean_post_recall_think(
        "A text overlay identifies the speaker as Orene Kearn, Fashion Stylist.",
        card,
    )

    assert think == "A text overlay identifies the speaker as Orene Kearn, Fashion Stylist."


def test_post_recall_think_allows_correct_direction_with_reversed_distractor():
    card = {
        "card_id": "c0",
        "family": "CR2",
        "answer_form": "multiple_choice",
        "question": "What visual change occurred to the polka dots?",
        "canonical_answer": "The dots changed from orange to gold.",
        "correct_option": "A",
        "options": [
            "A) The dots changed from orange to gold.",
            "B) The dots remained orange throughout.",
            "C) The dots changed from orange to brown.",
            "D) The dots changed from gold to orange.",
        ],
    }

    think = _clean_post_recall_think(
        "The polka dots on the background fabric change color from orange to gold.",
        card,
    )

    assert think == "The polka dots on the background fabric change color from orange to gold."


def test_post_recall_think_allows_question_context_word_matching_distractor():
    card = {
        "card_id": "c0",
        "family": "CR5",
        "answer_form": "multiple_choice",
        "question": (
            "What specific text logo is embossed on the passenger-side "
            "dashboard of the car?"
        ),
        "canonical_answer": "SRS",
        "correct_option": "D",
        "options": ["A) PASSENGER", "B) AIRBAG", "C) SAFETY", "D) SRS"],
    }

    think = _clean_post_recall_think(
        'The passenger-side dashboard features the embossed text logo "SRS" near the glove compartment.',
        card,
    )

    assert think == 'The passenger-side dashboard features the embossed text logo "SRS" near the glove compartment.'


def test_post_recall_think_allows_object_frame_display_wording():
    card = {
        "card_id": "c0",
        "family": "N1",
        "answer_form": "short_exact",
        "question": "What text appears on the trailer frame?",
        "canonical_answer": "NAM",
    }

    think = _clean_post_recall_think(
        "The silver metal trailer frame displays the text 'NAM' near the locking pin as the person pulls it.",
        card,
    )

    assert think == "The silver metal trailer frame displays the text 'NAM' near the locking pin as the person pulls it."


def test_post_recall_think_strips_earlier_frames_suffix_without_rejecting():
    card = {
        "card_id": "c0",
        "family": "CR2",
        "answer_form": "short_exact",
        "question": "What is the wolf doing near the container?",
        "canonical_answer": "sniffing the floor",
    }

    think = _clean_post_recall_think(
        "The wolf is shown sniffing the floor near the perforated metal container in the earlier frames.",
        card,
    )

    assert think == "The wolf is shown sniffing the floor near the perforated metal container."


def test_post_recall_think_strips_safe_source_wrapper_only():
    card = {
        "card_id": "c0",
        "family": "CR2",
        "answer_form": "short_exact",
        "question": "What is the wolf doing near the container?",
        "canonical_answer": "sniffing the floor",
    }

    think = _clean_post_recall_think(
        "In the earlier frames, the wolf is sniffing the floor near the perforated metal container.",
        card,
    )

    assert think == "The wolf is sniffing the floor near the perforated metal container."


def test_post_recall_think_retries_source_subject_framing_instead_of_rewriting():
    card = {
        "card_id": "c0",
        "family": "P1",
        "answer_form": "short_exact",
        "question": "What color was the apron?",
        "canonical_answer": "red",
    }

    think = _clean_post_recall_think(
        "The recalled frames show the person wearing a red apron while chopping onions.",
        card,
    )

    assert think == ""


def test_post_recall_think_salvages_json_wrapped_sentence():
    card = {
        "card_id": "c0",
        "family": "CR5",
        "answer_form": "short_exact",
        "question": "What text is visible?",
        "canonical_answer": "NAM",
    }

    think = _clean_post_recall_think(
        '{"observation": "The metal plate has NAM printed beside the locking pin."}',
        card,
    )

    assert think == "The metal plate has NAM printed beside the locking pin."


def test_post_recall_think_keeps_long_sentence_instead_of_truncating():
    card = {
        "card_id": "c0",
        "family": "P1",
        "answer_form": "short_exact",
        "question": "What is visible?",
        "canonical_answer": "red marker",
    }
    raw = (
        "The person holds a red marker near the cardboard sign while the table "
        "also contains scissors, tape, a notebook, loose paper, small clips, "
        "and several tools arranged around the workspace for the craft."
    )

    think = _clean_post_recall_think(raw, card)

    assert think == raw
    assert "red marker" in think


def test_llm_post_recall_think_rejects_meta_option_decision():
    card = {
        "card_id": "c0",
        "family": "P1",
        "answer_form": "multiple_choice",
        "question": "What color was the apron?",
        "canonical_answer": "red",
        "options": ["A) red", "B) blue"],
    }
    recall_result = {
        "source": "historical_frames",
        "returned_chunks": [4, 5],
        "text_content": "[4-5] The person wore a red apron while chopping onions.",
    }
    client = _FakeClient("The answer is A.")

    think = asyncio.run(
        _post_recall_think_via_llm(
            card,
            {"start_time": 4, "end_time": 5},
            recall_result,
            "A",
            client,
            "vid0",
            12,
            action="response",
        )
    )

    assert think == ""
