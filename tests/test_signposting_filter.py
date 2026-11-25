from lecturenotes.nlp import analyze


def test_generic_lecture_signposting_is_excluded_from_study_notes_but_kept_in_transcript():
    text = (
        "We will learn about processes, threads, memory and synchronization. "
        "A process is a program that is currently executing. "
        "Today we will cover CPU scheduling and memory management. "
        "CPU scheduling selects a ready process for execution. "
        "In this lecture, we will explore virtual memory. "
        "Virtual memory allows a process to use an address space larger than physical memory."
    )
    sentences = analyze(text)
    assert sentences[0].excluded == "instructional"
    assert sentences[2].excluded == "instructional"
    assert sentences[4].excluded == "instructional"
    assert sentences[1].excluded is None
    assert sentences[3].excluded is None
    assert sentences[5].excluded is None
    assert sentences[0].text in text  # source wording remains present and unchanged


def test_signposting_filter_is_subject_independent():
    text = (
        "We will learn about photosynthesis and cellular respiration. "
        "Photosynthesis converts light energy into chemical energy. "
        "We will discuss the causes of the French Revolution. "
        "The French Revolution began in 1789."
    )
    sentences = analyze(text)
    assert [s.excluded for s in sentences] == ["instructional", None, "instructional", None]


def test_explanatory_questions_and_second_person_sentences_are_preserved():
    text = (
        "What happens when a process requests memory? "
        "You can observe that the operating system allocates memory to each process. "
        "We will learn about virtual memory. "
        "Virtual memory gives each process a logical address space."
    )
    sentences = analyze(text)
    assert sentences[0].excluded is None
    assert sentences[1].excluded is None
    assert sentences[2].excluded == "instructional"
    assert sentences[3].excluded is None


def test_next_let_us_discuss_is_removed_from_notes_input():
    text = (
        "A database stores related information. "
        "Next, let us discuss normalization. "
        "Normalization reduces unnecessary duplication. "
        "First normal form requires each field to contain a single value."
    )
    sentences = analyze(text)
    transition = next(s for s in sentences if "let us discuss" in s.text.lower())
    assert transition.excluded == "instructional"
    assert any("normalization reduces" in s.text.lower() and s.excluded is None for s in sentences)
