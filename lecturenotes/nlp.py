"""Sentence splitting and concept-mention detection.

Uses spaCy (en_core_web_sm) for noun-phrase boundaries when available; otherwise a
rule-based noun-phrase finder is used so the app still runs. Both paths return the
same Sentence/Mention objects. Set LN_BACKEND=rules to force the fallback.
"""
from __future__ import annotations

import os
import re
from bisect import bisect_right
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field

Mention = namedtuple("Mention", "key surface start end det", defaults=(True,))
DETS = {"a", "an", "the", "one", "each", "every", "this", "its", "their", "these", "those", "some", "any", "another", "his", "her", "our", "your",
        "of", "in", "for", "with", "by", "between", "on", "from", "about", "without", "over", "under", "across"}  # words that precede a noun phrase

STOP = set("""a an the and or but if then than that this these those which who whom whose what when where while
of in on at by for with without to from into onto over under about as is are was were be been being am do does did
has have had having it its they them their he she his her we our you your i me my not no nor so such can could
will would shall should may might must also very more most much many some any each every other another both
through during within among along against via per toward towards upon after before around near behind beyond inside outside until just only even still yet instead never always usually often rarely sometimes one there here how why because since although though however thus therefore hence
up down out off again further once all few own same too s t don doesn isn aren won couldn wouldn didn let""".split())

# Words that end a noun phrase in the rule-based fallback (common lecture verbs).
VERBS = set("""use uses used using require requires required depend depends rely relies based consist consists contain
contains include includes cause causes caused lead leads results produce produces improve improves reduce reduces prevent
prevents minimize minimizes maximize maximizes minimise minimises compute computes calculate calculates represent
represents define defines allow allows enable enables help helps make makes take takes give gives show shows find finds
learn learns update updates apply applies perform performs measures stores connect connects send sends receive receives
generate generates transform transforms convert converts splits named known refers means mean become becomes get gets
keeps needs wants see sees looks tries runs works changes moves reaches increase increases decrease decreases choose
chooses select selects remember notice consider assume assumes happens occurs penalizing penalizes adjusting passes
provide provides manage manages create creates map maps translate translates organize organizes allocate allocates assign assigns schedules handle handles coordinate coordinates protect protects isolate isolates execute executes accesses waits match matches run write read replace decide starts stops detect detects monitor monitors disables memorizes made called escape controls determines affects introduces cover covers summarize""".split())

def _forms(v):
    """Inflections of a regular verb: base, -s/-es/-ies, -ed/-d/-ied."""
    third = v[:-1] + "ies" if v.endswith("y") and v[-2] not in "aeiou" else v + ("es" if v.endswith(("s", "x", "z", "ch", "sh")) else "s")
    past = v[:-1] + "ied" if v.endswith("y") and v[-2] not in "aeiou" else v + ("d" if v.endswith("e") else "ed")
    return {v, third, past}


# General-English explanatory verbs (subject-neutral). Words that are also common nouns (form, force, order, place, record, ...) are
# deliberately absent so they can still be part of terms.
GENERAL_VERBS = set()
for _v in """absorb accept achieve acquire adapt analyze appear arise ask attach attain avoid begin belong bind break bring build
capture carry combine compare compose conclude conduct confirm construct continue contribute correspond delay deliver
demonstrate denote derive describe describes detect determine develop differ discover distinguish divide drive eliminate emerge
employ encourage ensure enter establish estimate evaluate examine exceed exist expand expect explain express extend extract
fail fill follow govern guarantee happen hold identify illustrate imply indicate influence inform initiate insert install involve
join lack launch leave locate maintain merge modify obtain offer operate overcome permit predict prefer prepare
preserve proceed promote propose prove pull push raise realize recognize refer reflect reject relate remain remove repeat
resolve respond restore retain retrieve reveal satisfy serve simplify solve specify stay suffer suggest
supply survive terminate tend transfer transmit understand undergo unify utilize validate vary verify enforce wait occur affect hurt watch grow shrink decline benefit damage drop gain spend buy sell pay earn invest lose win store""".split():
    GENERAL_VERBS |= _forms(_v)
VERBS |= GENERAL_VERBS
# Common lecture predicates that are useful for recovering ASR sentence boundaries.
VERBS.update({"group", "groups", "grouped", "grouping", "subtract", "subtracts", "subtracted", "add", "adds", "added"})

# Numerals/ordinals: part of a name when they recur ("three way handshake", "first/second law"), otherwise trimmed.
ORD = set("first second third fourth fifth two three four five six seven eight nine ten".split())
# Evaluative adjectives: trimmed from the edges of a phrase unless the whole phrase recurs in the text (see refine_mentions).
SOFT = set("""common different various simple large small good bad new important main next typical popular general basic
several certain possible likely strongly slow fast cheap low high full poor actual another today whole real true false
great little better best higher lower larger smaller greater fewer bigger faster slower longer shorter stronger weaker earlier later together jointly""".split()) | ORD
# General-English irregular past forms: never nouns, so they end a noun phrase ("technologies came first").
IRREGULAR_PAST = set("""came went made took gave saw got began grew led held lost sent stood told thought found brought built
became drove wrote chose rose fell understood shown seen given taken broken chosen spoke ran""".split())
CONTROL_VERBS = {"stop", "stops", "start", "starts", "begin", "begins", "keep", "keeps", "avoid", "avoids", "finish", "finishes", "continue", "continues", "enjoy", "risk"}
AUX = {"can", "could", "will", "would", "may", "might", "must", "should", "cannot", "did", "does", "do"}
NOUN_CTX = {"a", "an", "the", "one", "each", "every", "this", "these", "those", "its", "their", "some", "any", "another", "our", "your"}
NOMINAL = ("tion", "sion", "ment", "ness", "ity", "ics", "ism", "ogy", "ing", "ance", "ence")

FILLER = re.compile(r"\b(?:um+|uh+|erm|you know|i mean)\b[,.]?\s*", re.I)
ABBR = re.compile(r"\b(?:e\.g|i\.e|etc|vs|fig|eq|dr|mr|mrs|ms|prof|approx|cf|al)\.", re.I)
BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Za-z0-9])|\n\s*\n")
TOKEN = re.compile(r"[A-Za-z]/[A-Za-z]|[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


HEADING = re.compile(r"^[ \t]*(?:={2,}|#{1,6})[ \t]*(.+?)[ \t]*=*[ \t]*$", re.M)
META_FIELDS = (r"subject|topic|course|lecturer|instructor|professor|speaker|duration|approximate duration|date|title|module|unit|chapter|"
               r"week|session|lecture transcript")
META = re.compile(rf"^(?:{META_FIELDS})\s*:|^this (?:is|document is|file is|recording is) (?:an? )?[^.]{{0,80}}\b(?:transcript|recording|lecture|document|file)\b"
                  r"|^the (?:lecturer|instructor|professor|speaker|teacher) (?:explains|discusses|covers|introduces|gives)", re.I)
# Pure lecture signposting: remove statements that announce what the lecture
# will cover, but keep the original transcript untouched. These patterns are
# deliberately subject-neutral and only match sentence openings.
SIGNPOST = re.compile(
    r"^(?:we(?:'ll|\s+will)\s+(?:learn|study|cover|explore|examine|discuss|understand|look\s+at|focus\s+on)\b"
    r"|we\s+are\s+going\s+to\s+(?:learn|study|cover|explore|examine|discuss)\b"
    r"|today\s+we(?:'ll|\s+will)\s+(?:learn|study|cover|explore|examine|discuss)\b"
    r"|in\s+this\s+(?:lecture|session|class),?\s+we(?:'ll|\s+will)\s+(?:learn|study|cover|explore|examine|discuss)\b"
    r"|the\s+(?:lecture|session|class)\s+will\s+(?:cover|discuss|explore|examine)\b)",
    re.I,
)
INSTR = re.compile(r"^(?:good (?:morning|afternoon|evening)|thank(?:s| you)|welcome|let(?:'s| us| me)\b|before we\b|by the end of\b"
                   r"|now,? let(?:\'s| us)\b|next,? let(?:\'s| us)\b|in this (?:lecture|session|class)\b|today,? we\b|we have now covered|we will now|that concludes|i want to\b"
                   r"|now,? look|next,? we\b"
                   r"|you\s+(?:should|could|must|need\s+to|might\s+want\s+to)\s+"
                   r"(?:check|review|revise|practice|try|think|consider|remember|look\s+at|note|"
                   r"draw|answer|discuss|compare|identify|calculate|solve|write|read|explain)\b"
                   r"|(?:draw|ask|pause|revise|practice|try|think)\b)", re.I)
SECOND_PERSON = re.compile(r"\byou(?:r|rs|rself)?\b", re.I)
HEADER_LINE = re.compile(rf"^[ \t]*(?:{META_FIELDS})[ \t]*:.*$|^[ \t]*[A-Z0-9][A-Z0-9 \-:&]{{3,}}$", re.I | re.M)


@dataclass
class Sentence:
    idx: int
    text: str
    time: float | None = None
    mentions: list = field(default_factory=list)
    kept: list = field(default_factory=list)  # [(concept_key, Mention)] filled by extract
    para: int = 0  # paragraph index in the source text
    section: int = -1  # index of the '===== Heading =====' section (-1 = before the first heading)
    heading: str | None = None
    excluded: str | None = None  # 'metadata' | 'instructional': kept in the transcript, hidden from notes/concepts
    resolved: list = field(default_factory=list)  # [(key, Mention)] for ALL mentions (set by extract.build_concepts)
    verbs: frozenset = frozenset()  # verb-like words induced from this lecture
    doc: object = None  # spaCy Doc when the spaCy backend is active


def concept_key(words):
    """Normalize a noun phrase, dropping a leading determiner and singularising its head.

    This prevents surface phrases such as "one candidate key" from becoming
    separate concepts from the canonical term "candidate key".
    """
    ws = [w.lower() for w in words]
    while len(ws) > 1 and ws[0] in DETS:
        ws.pop(0)
    last = ws[-1]
    if len(last) > 3 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 4 and last.endswith(("sses", "xes", "ches", "shes")):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is", "as", "os")):
        last = last[:-1]
    ws[-1] = last
    # spaCy can split hyphenated compounds into multiple tokens (e.g. non, -, key).
    # Collapse whitespace after replacing the hyphen so concept identifiers stay stable.
    return " ".join(" ".join(ws).replace("-", " ").split())


def split_sections(text: str):
    """Split on '===== Heading =====' / '# Heading' lines -> [(title|None, body, offset)]."""
    ms = list(HEADING.finditer(text))
    if not ms:
        return [(None, text, 0)]
    out = [(None, text[:ms[0].start()], 0)] if text[:ms[0].start()].strip() else []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((m.group(1).strip(" =#"), text[m.end():end], m.end()))
    return out


def _missing_punctuation_boundaries(text: str):
    """Yield conservative clause boundaries where ASR omitted sentence punctuation.

    A split is accepted only when a capitalized clause starter is followed by a
    likely finite predicate. This recovers cases such as ``... data The DBMS
    provides ...`` without splitting coordinated noun phrases such as
    ``student ID, name, and department ID`` or all-caps technical tokens.
    """
    finite = VERBS | AUX | {"is", "are", "was", "were", "has", "have", "had"}
    starters = {"for example", "for instance", "to summarize", "in summary",
                "next", "finally", "however", "meanwhile", "overall"}
    word_rx = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
    # ASR frequently lowercases a new explanatory sentence after dropping
    # punctuation (for example, "... non-key attributes for instance ...").
    # Treat discourse markers as boundaries regardless of their casing.
    discourse = re.compile(
        r"\s+(?=(?:for\s+example|for\s+instance|however|meanwhile|finally|"
        r"to\s+summarize|in\s+summary|in\s+conclusion|overall|imagine|suppose|whereas|"
        r"on\s+the\s+other\s+hand|in\s+contrast)\b)", re.I
    )
    for m in discourse.finditer(text):
        left = text[:m.start()].rstrip()
        if len(left.split()) >= 5 and left[-1:] not in ".!?:;":
            yield m.start(), m.end()

    for m in re.finditer(r"\s+(?=[A-Z])", text):
        left = text[:m.start()].rstrip()
        if len(left.split()) < 3 or (left and left[-1] in ".!?:;"):
            continue
        previous = re.search(r"([A-Za-z]+)[,]?$", left)
        if previous and previous.group(1).lower() in {"and", "or", "the", "a", "an", "of", "to"}:
            continue
        tail = text[m.end():]
        words = word_rx.findall(tail)
        if not words:
            continue
        first = words[0]
        # Do not split acronyms / all-caps identifiers (SQL, ORDER BY, ACID).
        if len(first) > 1 and first.isupper():
            continue
        opener = " ".join(words[:2]).lower()
        if opener in starters:
            yield m.start(), m.end()
            continue
        # Avoid turning a title-case proper name into a sentence unless a
        # finite verb follows within a short subject phrase.
        if any(w.lower() in finite for w in words[1:6]):
            yield m.start(), m.end()


def split_sentences(text: str):
    """Return sentence spans, recovering clear missing-punctuation ASR boundaries."""
    prot = ABBR.sub(lambda m: m.group(0).replace(".", "§"), text)
    matches = []
    for match in BOUNDARY.finditer(prot):
        paragraph_break = match.group().count("\n") >= 2
        if not paragraph_break:
            left = prot[:match.start()].rstrip()
            right = prot[match.end():].lstrip()
            next_word = re.match(r"([A-Za-z]+)", right)
            # ASR sometimes inserts a period before a continuation such as
            # "and removes ...". Keep that clause attached to its subject.
            if (re.search(r"[,.]\s*$", left) and next_word
                    and next_word.group(1)[0].islower()
                    and next_word.group(1).lower() in {"and", "or", "but"}):
                continue
        matches.append((match.start(), match.end(), paragraph_break))
    matches.extend((a, b, False) for a, b in _missing_punctuation_boundaries(prot))
    matches.sort(key=lambda item: (item[0], item[1]))

    spans, pos, para = [], 0, 0
    for a, b, paragraph_break in matches:
        if a < pos:
            continue
        spans.append((pos, a, para))
        pos = b
        if paragraph_break:
            para += 1
    spans.append((pos, len(text), para))

    out, seen = [], set()
    for a, b, para in spans:
        s = FILLER.sub("", re.sub(r"\s+", " ", text[a:b])).strip()
        if (len(s.split()) < 3 and not HEADER_LINE.fullmatch(s)) or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append((s[0].upper() + s[1:], a, para))
    return out


def join_segments(segments):
    """Join timed ASR segments into one text and remember each segment's char offset."""
    text, starts, times = "", [], []
    for seg in segments:
        t = seg["text"].strip()
        if not t:
            continue
        starts.append(len(text))
        times.append(float(seg.get("start", 0.0)))
        text += t + " "
    return text, starts, times


def split_audio_segments(segments, starts, text, target_words=18):
    """Fallback sentence-like chunks when ASR emits long unpunctuated runs.

    Whisper's timestamp segments are often phrase-sized but may have no final
    punctuation. If ordinary punctuation-based splitting produces only one
    sentence, group adjacent ASR segments into readable chunks while retaining
    the original character offsets (and therefore timestamps).
    """
    spans = []
    group_start = None
    group_end = None
    word_count = 0

    def flush():
        nonlocal group_start, group_end, word_count
        if group_start is not None and group_end is not None:
            value = text[group_start:group_end].strip()
            if len(value.split()) >= 3:
                spans.append((value, group_start, 0))
        group_start = group_end = None
        word_count = 0

    for i, seg_start in enumerate(starts):
        seg_end = starts[i + 1] - 1 if i + 1 < len(starts) else len(text)
        piece = text[seg_start:seg_end].strip()
        if not piece:
            continue
        if group_start is None:
            group_start = seg_start
        group_end = seg_end
        word_count += len(piece.split())
        ends_sentence = bool(re.search(r"[.!?][\"')\]]*$", piece))
        if word_count >= target_words or (ends_sentence and word_count >= 3):
            flush()

    flush()
    # Avoid losing a tiny trailing ASR fragment; attach it to the previous chunk.
    if spans and len(text[spans[-1][1]:].split()) < 3:
        previous, offset, para = spans[-1]
        spans[-1] = (text[offset:].strip(), offset, para)
    return spans


def isolate_header(text: str) -> str:
    """Leading metadata lines (ALL-CAPS title, 'Subject: ...') become their own paragraphs instead of fusing with the first sentence."""
    out, in_head = [], True
    for ln in text.split("\n"):
        is_meta = bool(ln.strip()) and bool(HEADER_LINE.fullmatch(ln.rstrip()))
        if in_head and is_meta:
            out += [ln, ""]
            continue
        if ln.strip():
            in_head = False
        out.append(ln)
    return "\n".join(out)


def induce_verbs(texts) -> frozenset:
    """Verb-like words in THIS text (no subject lexicon): they follow modals / 'to', or precede a determiner,
    and are never seen after a determiner. Inflected forms (-s, -es, -ed, -d) are added."""
    aux, det_next, noun = Counter(), Counter(), Counter()
    for t in texts:
        toks = [m.group().lower() for m in TOKEN.finditer(t)] + [""]
        for prev, w, nxt in zip([""] + toks, toks, toks[1:] + [""]):
            if len(w) < 4 or not w.isalpha() or w in STOP:
                continue
            if prev in AUX:
                aux[w] += 2
            elif prev == "to":
                aux[w] += 1
            if nxt in NOUN_CTX:
                det_next[w] += 1
            if prev in NOUN_CTX:
                noun[w] += 1
    base = {w for w in set(aux) | set(det_next) if not noun[w] and not w.endswith(NOMINAL) and (aux[w] >= 2 or det_next[w] >= 2)}
    forms = set(base)
    for v in base:
        if not v.endswith("ed"):
            forms |= _forms(v)
    return frozenset(forms)


_NLP, _TRIED = None, False


def get_nlp():
    global _NLP, _TRIED
    if not _TRIED:
        _TRIED = True
        if os.environ.get("LN_BACKEND", "").lower() != "rules":
            try:
                import spacy
                _NLP = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                _NLP = None
    return _NLP


def backend() -> str:
    return "spacy" if get_nlp() else "rules"


def _participle(low):
    return len(low) > 5 and low.endswith("ed") and not low.endswith("eed")


def _rule_mentions(s: str, verbs=frozenset()):
    runs, cur, prev = [], [], ""
    for m in TOKEN.finditer(s):
        t, low = m.group(), m.group().lower()
        ordinal_normal_form = (
            low == "form" and len(cur) >= 3
            and cur[-1][0].lower() == "normal"
            and cur[-2][0].lower() in ORD
        )
        breaker = ((low in STOP or low in VERBS or low in IRREGULAR_PAST or low in verbs)
                   and not ordinal_normal_form) or (len(low) > 4 and low.endswith("ly"))
        breaker = breaker or (len(cur) > 1 and t[0].isalpha() and _participle(low))
        if t[0].isalpha() and len(t) > 1 and not breaker:
            if not cur:
                cur.append(prev)  # token before the phrase, used as a noun-ness hint
            cur.append((t, m.start(), m.end()))
        elif cur:
            runs.append(cur)
            cur = []
        prev = low
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        det, toks = run[0] in DETS or run[1][1] == 0, run[1:]  # preceded by a noun-introducer, or a sentence-initial subject
        if len(toks) == 1 and toks[0][0].lower().endswith("ing") and run[0] in CONTROL_VERBS:
            continue  # 'begins <gerund>', 'keeps <gerund>': a gerund complement, not a noun phrase ('reduces overfitting' is kept)
        while len(toks) > 1 and _participle(toks[-1][0].lower()):
            toks = toks[:-1]  # "production shifted" -> "production"
        if len(toks) == 1 and _participle(toks[0][0].lower()) and not det:
            continue
        toks = toks[-3:] if len(toks) > 4 else toks
        out.append(Mention(concept_key([w for w, _, _ in toks]), s[toks[0][1]:toks[-1][2]], toks[0][1], toks[-1][2], det))
    return out


def refine_mentions(sents):
    """Statistical clean-up of phrase edges using the lecture itself (no word lists beyond SOFT/ORD).

    An evaluative adjective or numeral at a phrase edge is trimmed ('common activation function' -> 'activation function')
    unless the whole phrase recurs (>= 2 times) or it is an enumeration ('first law', 'second law': same tail, different ordinal)."""
    cnt = Counter(m.key for s in sents for m in s.mentions)
    tails = defaultdict(set)
    for s in sents:
        for m in s.mentions:
            w = m.key.split()
            if len(w) > 1 and w[0] in ORD:
                tails[" ".join(w[1:])].add(w[0])
    list_signals = {"property", "properties", "characteristic", "characteristics", "component", "components",
                    "element", "elements", "principle", "principles", "stage", "stages", "phase", "phases",
                    "factor", "factors", "feature", "features"}
    for s in sents:
        out = []
        for m in s.mentions:
            toks = [(t.group(), t.start(), t.end()) for t in TOKEN.finditer(m.surface) if t.group()[0].isalpha()]
            # Split a parser/rule phrase when ASR omitted punctuation before a
            # capitalized member of an explicitly introduced list ("properties Atomicity").
            split_at = []
            for j in range(1, len(toks)):
                word, _, end = toks[j]
                if not word[:1].isupper() or word.isupper():
                    continue
                previous_word = toks[j - 1][0].lower()
                tail = s.text[m.start + end:]
                next_word_match = re.match(r"\s*([A-Za-z]+)", tail)
                next_word = next_word_match.group(1).lower() if next_word_match else ""
                if previous_word in list_signals or next_word in VERBS or next_word in AUX:
                    split_at.append(j)
            if split_at:
                boundaries = [0] + split_at + [len(toks)]
                for left_i, right_i in zip(boundaries[:-1], boundaries[1:]):
                    part = toks[left_i:right_i]
                    if not part:
                        continue
                    a, b = m.start + part[0][1], m.start + part[-1][2]
                    out.append(Mention(concept_key([t[0] for t in part]), s.text[a:b], a, b,
                                       m.det if left_i == 0 else True))
                continue
            n0 = len(toks)
            while len(toks) > 1 and (toks[0][0].lower() in SOFT or toks[-1][0].lower() in SOFT):
                lead = toks[0][0].lower() in SOFT
                rest = toks[1:] if lead else toks[:-1]
                tok = toks[0][0].lower() if lead else toks[-1][0].lower()
                if cnt[concept_key([t[0] for t in toks])] >= 2 or (tok in ORD and len(tails[concept_key([t[0] for t in rest])]) >= 2):
                    break
                toks = rest
            if not toks or all(t[0].lower() in SOFT for t in toks):
                continue
            if len(toks) != n0:
                a, b = m.start + toks[0][1], m.start + toks[-1][2]
                m = m._replace(key=concept_key([t[0] for t in toks]), surface=m.surface[toks[0][1]:toks[-1][2]], start=a, end=b)
            out.append(m)
        s.mentions = out


def _spacy_mentions(doc, s: str):
    bad = {"DET", "PRON", "NUM", "PUNCT", "ADV", "SYM"}
    out = []
    for nc in doc.noun_chunks:
        toks = list(nc)
        while toks and (toks[0].is_stop or toks[0].pos_ in bad) and not (toks[0].lower_ in SOFT or toks[0].text.isalpha() and toks[0].pos_ == "NUM"):
            toks.pop(0)
        while toks and (toks[-1].is_stop or toks[-1].pos_ in {"PUNCT", "SYM"}):
            toks.pop()
        # Some spaCy model/version combinations incorrectly absorb a verb into
        # the preceding noun chunk (e.g. "foreign key refers" in
        # "A foreign key refers to the primary key"). Trim a terminal verbal
        # token before converting the chunk into a concept. The lexical check
        # handles parser mis-tags, but only when the following text supplies a
        # verb-complement preposition, so valid noun phrases such as "access
        # controls" are not discarded.
        while len(toks) > 1:
            tail = toks[-1]
            after_tail = s[tail.idx + len(tail.text):]
            followed_by_complement = bool(re.match(
                r"\s+(?:to|as|for|on|with|by|in|from|at|into|that)\b",
                after_tail, re.I,
            ))
            parser_verb = tail.pos_ in {"VERB", "AUX"}
            mis_tagged_verb = tail.lower_ in VERBS and tail.lower_ != "means" and followed_by_complement
            if parser_verb or mis_tagged_verb:
                toks.pop()
                continue
            break
        if not toks or any(t.pos_ == "PRON" for t in toks) or not any(t.pos_ in {"NOUN", "PROPN", "X"} for t in toks):
            continue
        toks = toks[-3:] if len(toks) > 4 else toks
        a, b = toks[0].idx, toks[-1].idx + len(toks[-1])
        out.append(Mention(concept_key([t.text for t in toks]), s[a:b], a, b))
    return out


def _merge_mentions(primary, supplement):
    """Parser noun chunks win; rule-based phrases are added only where the parser found nothing. A tagger can label a
    gerund-like noun as a verb ('reduces <noun>' as VBG/xcomp), in which case no noun chunk exists for it."""
    taken = [(m.start, m.end) for m in primary]
    extra = [m for m in supplement if not any(m.start < b and a < m.end for a, b in taken)]
    return sorted(primary + extra, key=lambda m: m.start)


def analyze(text: str | None = None, segments=None) -> list[Sentence]:
    """Split into sentences (headings, timestamps for audio), flag metadata/filler, detect concept mentions."""
    starts = times = None
    segment_items = list(segments) if segments is not None else None
    if segment_items:
        text, starts, times = join_segments(segment_items)
        sections = [(None, text, 0)]
    else:
        sections = split_sections(isolate_header(text or ""))
    sents, seen, sec = [], set(), -1
    for title, body, base in sections:
        if title is not None:
            sec += 1
        pieces = split_sentences(body)
        # ASR may return many timestamped phrases without sentence punctuation.
        # In that case, punctuation splitting collapses the full lecture into
        # one sentence; use grouped ASR boundaries as a safe fallback.
        if starts and len(pieces) < 4:
            pieces = split_audio_segments(segment_items, starts, text)
        for s, off, para in pieces:
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            t = times[max(0, bisect_right(starts, base + off) - 1)] if starts else None
            sents.append(Sentence(len(sents), s, t, para=para + 10000 * (sec + 1), section=sec, heading=title))
    headed = any(t is not None for t, _, _ in sections)
    for s in sents:
        if (headed and s.section < 0) or META.search(s.text) or (s.text.isupper() and len(s.text.split()) <= 12):
            s.excluded = "metadata"
        elif SIGNPOST.search(s.text) or INSTR.search(s.text):
            # Exclude explicit lecture signposting/instructions only. Do not
            # discard explanatory sentences merely because they address "you"
            # or are phrased as questions; those can contain examinable content.
            s.excluded = "instructional"
    live = [s for s in sents if not s.excluded]
    verbs = induce_verbs([s.text for s in live])
    for s in sents:
        s.verbs = verbs
    nlp = get_nlp()
    if nlp:
        for sent, doc in zip(live, nlp.pipe([s.text for s in live])):
            sent.mentions = _merge_mentions(_spacy_mentions(doc, sent.text), _rule_mentions(sent.text, verbs))
            sent.doc = doc
    else:
        for sent in live:
            sent.mentions = _rule_mentions(sent.text, verbs)
    refine_mentions(live)
    return sents
