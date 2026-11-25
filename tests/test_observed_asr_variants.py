from lecturenotes.terminology import correct_segments


def seg(*texts):
    return [{"text": t, "start": i * 2.0, "end": i * 2.0 + 1.0} for i, t in enumerate(texts)]


def test_observed_os_pce_is_corrected_only_in_process_control_block_context():
    source = seg(
        "The operating system stores information in a process control block or PCE.",
        "A process is a program that is currently executing.",
    )
    corrected, changes = correct_segments(source)
    assert "or PCB" in corrected[0]["text"]
    assert any(c["rule"] == "os.process_control_block_acronym" for c in changes)


def test_observed_dbms_atomicity_variant_is_corrected_in_transaction_context():
    source = seg(
        "A database transaction follows consistency and isolation rules.",
        "Ademisity means all operations happen or none happen.",
    )
    corrected, changes = correct_segments(source)
    assert "Atomicity" in corrected[1]["text"]
    assert any(c["rule"] == "dbms.atomicity" for c in changes)


def test_dbms_phrase_repairs_are_context_guarded():
    source = seg(
        "A database management system is software used to create, read, update, and manage at data.",
        "A foreign key refers to a key in a other table.",
        "Third normal form removes transitive dependencies of unkey attributes.",
    )
    corrected, changes = correct_segments(source)
    joined = " ".join(s["text"] for s in corrected)
    assert "manage data" in joined
    assert "another table" in joined
    assert "non-key attributes" in joined
    assert len(changes) == 3


def test_observed_asr_corrections_do_not_apply_without_subject_context():
    source = seg("Ademisity is a fictional term.", "PCE is a random abbreviation.")
    corrected, changes = correct_segments(source)
    assert [s["text"] for s in corrected] == [s["text"] for s in source]
    assert changes == []
