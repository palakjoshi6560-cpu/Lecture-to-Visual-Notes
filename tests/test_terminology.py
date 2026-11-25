from lecturenotes.terminology import correct_segments
from lecturenotes.pipeline import process


def _segments(*texts):
    return [
        {"text": text, "start": float(i * 4), "end": float(i * 4 + 3)}
        for i, text in enumerate(texts)
    ]


def test_dbms_acid_asr_variants_are_corrected_only_in_transaction_context():
    segments = _segments(
        "A database transaction must preserve consistency and isolation.",
        "Adomicity means all operations happen or none happen.",
        "Doorability means committed changes persist after a failure.",
    )
    corrected, changes = correct_segments(segments)

    assert "Atomicity" in corrected[1]["text"]
    assert "Durability" in corrected[2]["text"]
    assert {item["rule"] for item in changes} == {"dbms.atomicity", "dbms.durability"}
    assert segments[1]["text"].startswith("Adomicity")  # original input is not mutated
    assert corrected[1]["start"] == segments[1]["start"]
    assert corrected[1]["end"] == segments[1]["end"]


def test_dbms_normal_form_phrase_repair_requires_normalization_context():
    segments = _segments(
        "Database normalization reduces unnecessary duplication.",
        "Second normal form removes partial dependencies.",
        "Third, normal form removes transitive dependencies.",
    )
    corrected, changes = correct_segments(segments)
    assert "Third Normal Form" in corrected[2]["text"]
    assert any(item["rule"] == "dbms.third_normal_form" for item in changes)


def test_similar_sounding_words_are_not_changed_without_dbms_context():
    segments = _segments(
        "The derivability of this mathematical result is important.",
        "Adomicity is a label used in this fictional example.",
        "The third, normal form in this unrelated story is unusual.",
    )
    corrected, changes = correct_segments(segments)
    assert [item["text"] for item in corrected] == [item["text"] for item in segments]
    assert changes == []


def test_pipeline_uses_corrected_audio_text_but_preserves_original_transcript():
    segments = _segments(
        "A database transaction preserves consistency and isolation.",
        "Adomicity means all transaction operations happen or none happen.",
        "Consistency keeps database rules valid after each transaction.",
        "Doorability means committed database changes persist.",
        "Database normalization reduces duplication in relational tables.",
        "Second normal form removes partial dependencies.",
        "Third, normal form removes transitive dependencies.",
    )
    result = process(segments=segments, source="audio")

    transcript = " ".join(item["text"] for item in result["transcript"])
    assert "Atomicity" in transcript
    assert "Durability" in transcript
    assert "Third Normal Form" in transcript
    assert "Adomicity" in result["raw_transcript"]
    assert "Doorability" in result["raw_transcript"]
    assert len(result["meta"]["terminology_corrections"]) == 3


def test_typed_transcript_is_not_silently_rewritten():
    text = (
        "A database transaction preserves consistency and isolation. "
        "Adomicity is the next topic in this transcript. "
        "Doorability is also mentioned. "
        "The database contains relational tables and keys."
    )
    result = process(text=text, source="text")
    assert "Adomicity" in result["raw_transcript"]
    assert not result["meta"]["terminology_corrections"]



def test_os_terms_are_corrected_only_when_operating_system_process_context_is_strong():
    segments = _segments(
        "In an operating system, processes are scheduled by the CPU.",
        "The threats share the processor using context switching.",
        "The PC stores the process state while it is waiting.",
        "The shortage of first selects algorithm chooses the next process.",
    )
    corrected, changes = correct_segments(segments)
    joined = " ".join(s["text"] for s in corrected)
    assert "threads" in joined
    assert "PCB" in joined
    assert "Shortest Job First" in joined
    assert {c["rule"] for c in changes} == {
        "os.threads_as_threats",
        "os.process_control_block_acronym",
        "os.shortest_job_first_phrase",
    }


def test_os_context_switching_phrase_is_corrected_across_segment_boundary():
    segments = _segments(
        "The operating system schedules processes and performs contacts",
        "switching to share the CPU.",
    )
    corrected, changes = correct_segments(segments)
    assert corrected[0]["text"].endswith("Context switching")
    assert corrected[1]["text"] == "to share the CPU."
    assert any(c["rule"] == "os.context_switching_phrase" and c["segment"] == 0 for c in changes)


def test_os_rules_do_not_rewrite_similar_words_outside_os_context():
    segments = _segments(
        "The threats to the network are increasing.",
        "The PC is used for personal computing.",
        "A shortage of first selects is mentioned in a fictional story.",
    )
    corrected, changes = correct_segments(segments)
    assert [s["text"] for s in corrected] == [s["text"] for s in segments]
    assert changes == []
