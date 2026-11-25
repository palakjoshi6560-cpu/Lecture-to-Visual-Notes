"""Concept extraction, relationship extraction and importance scoring."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from .nlp import IRREGULAR_PAST, SOFT, STOP, VERBS
from sklearn.feature_extraction.text import TfidfTransformer

GENERIC = set("""example thing way lot kind type case time part number something today lecture idea point step fact reason question
answer situation person people one summary size direction choice result problem technique method algorithm okay system data information user request
resource mechanism operation execution goal term component form need access word version completion concept aspect issue factor role
approach examination exam""".split())  # vague superordinate and assessment nouns: uninformative on their own in most lectures
EMPH = re.compile(r"\b(?:important|main|remember|crucial|essential|fundamental|critical|keep in mind|note that|key (?:point|idea|takeaway|insight))\b", re.I)
DEF_RE = re.compile(
    r"^\s*(?:,?\s*(?:or|also called|also known as)\s+[^,]{2,40},)?\s*(?:\([^)]{1,30}\)\s*)?"
    r"(?P<v>is defined as|is called|is known as|refers to|means|denotes|happens when|occurs when"
    r"|(?:is|are)(?=\s+(?:an?|the|one of)\b)|(?:is|are))\s+(?P<d>.+?)\s*[.!?]*$", re.I)
ACRONYM = re.compile(r"\s*,?\s*(?:or\s+|\()\s*([A-Za-z]{2,8})\s*[),]")

REL = [  # (label, regex) searched in the text between two concept mentions
    ("is a type of", re.compile(r"\b(?:is|are)\s+(?:an?|the)?\s*(?:type|kind|form|class|subset|variant|instance|special case)\s+of\b")),
    ("is a", re.compile(r"^\W*(?:is|are)\s+(?:an?|the)\W*$")),
    ("contains", re.compile(r"\b(?:consists? of|(?:is|are) (?:composed|made up) of|contains?|includes?|comprises?)\b")),
    ("combines", re.compile(r"\bcombines?\b")),
    ("has", re.compile(r"^\W*(?:has|have)\W*(?:an?|the)?\W*$")),
    ("requires", re.compile(r"\b(?:requires?|needs?)\b")),
    ("depends on", re.compile(r"\b(?:depends? on|relies on|(?:is|are) based on|builds? on)\b")),
    ("refers to", re.compile(r"\brefers? to\b")),
    ("selected as", re.compile(r"\b(?:selected|chosen)\s+as\b")),
    ("uses", re.compile(r"\buses?\b")),
    ("applies to", re.compile(r"\bapplies to\b")),
    ("provides", re.compile(r"\b(?:provides?|offers?)\b")),
    ("manages", re.compile(r"\b(?:manages?|handles?|coordinates?|schedules?)\b")),
    ("creates", re.compile(r"\bcreates?\b")),
    ("allocates", re.compile(r"\b(?:allocates?|assigns?)\b")),
    ("maps", re.compile(r"\b(?:maps?|translates?)\b")),
    ("organizes", re.compile(r"\borganizes?\b")),
    ("stores", re.compile(r"\bstores?\b")),
    ("protects", re.compile(r"\b(?:protects?|isolates?)\b")),
    ("waits for", re.compile(r"\bwaits? for\b")),
    ("leads to", re.compile(r"\b(?:causes?|leads? to|results? in|triggers?|produces?)\b")),
    ("reduces", re.compile(r"\b(?:reduces?|minimi[sz]es?|decreases?|prevents?|lowers?|penali[sz]es?)\b")),
    ("improves", re.compile(r"\b(?:improves?|increases?|maximi[sz]es?|boosts?|enhances?)\b")),
    ("computes", re.compile(r"\b(?:computes?|calculates?|measures?|estimates?)\b")),
    ("updates", re.compile(r"\b(?:updates?|adjusts?|modifies)\b")),
    ("enables", re.compile(r"\b(?:enables?|allows?|helps?)\b")),
    ("used for", re.compile(r"\b(?:is|are) used (?:for|to|in)\b")),
    ("controls", re.compile(r"\b(?:controls?|determines?|affects?)\b")),
    ("occurs in", re.compile(r"\b(?:takes? place|occurs?|happens?|(?:is|are) (?:located|found)) in\b")),
    ("part of", re.compile(r"\b(?:is|are) (?:a )?part of\b|\bbelongs? to\b")),
    ("responsible for", re.compile(r"\b(?:is|are) responsible for\b")),
    ("differs from", re.compile(r"\b(?:unlike|differs? from|in contrast to|compared (?:to|with)|versus|vs)\b")),
    ("e.g.", re.compile(r"^\W*(?:such as|for example|including|e\.g\.?)\W*$")),
]
PASSIVE = re.compile(r"\b(?:is|are)\s+(\w+(?:ed|en))\s+by\b")
NEGATION = re.compile(r"\b(?:not|never|cannot)\b")
FUNCTION = STOP | VERBS | SOFT
PREPS = {"in", "of", "on", "for", "with", "by", "from", "to", "at", "between", "within", "into", "under", "over", "among", "through", "during"}
RELATIVE = re.compile(r"^\s*(?:that|which|who)\b", re.I)
DEMO = re.compile(r"\b(?:this|these|those|that)\s+$", re.I)
WEIGHTS = dict(freq=0.25, central=0.20, signal=0.20, distinct=0.15, spread=0.10, early=0.10)
# These are useful supporting nouns, but they are rarely the central learning
# objective when a lecture also names more specific phrases such as "foreign key"
# or "composite key". Keep them extractable; only temper their ranking.
LOW_SPECIFICITY_SINGLETONS = {"table", "row", "column", "key", "record", "field", "attribute", "value"}
LOW_SPECIFICITY_PREFIXES = {"one", "two", "many", "several", "reliable", "related", "existing", "matching", "organized", "organised"}


@dataclass
class Concept:
    key: str
    name: str
    count: int
    sentences: list
    aliases: list = field(default_factory=list)
    definition: str | None = None
    definition_sentence: int | None = None
    emphasized: bool = False
    topic: int = 0
    importance: float = 0.0
    components: dict = field(default_factory=dict)


def _generic(k):
    return k in GENERIC or k == "trade off" or (len(k.split()) == 1 and (len(k) < 3 or k.isdigit()))


def _acronym_aliases(sents):
    alias = {}
    for s in sents:
        for m in s.mentions:
            a = ACRONYM.match(s.text[m.end:])
            words = m.key.split()
            if a and len(words) > 1 and a.group(1).lower() == "".join(w[0] for w in words):
                alias[a.group(1).lower()] = m.key
    return alias


def _suffix_aliases(counts, alias):
    """Merge 'network' into 'neural network' when it is the only longer concept ending with it."""
    out = dict(alias)
    for k in counts:
        if " " in k or k in out:
            continue
        longer = [l for l in counts if l.endswith(" " + k)]
        if len(longer) == 1 and counts[k] <= counts[longer[0]]:
            out[k] = longer[0]
    for k in counts:  # 'X switching' -> 'X switch' (the more frequent form wins)
        w = k.split()
        if w[-1].endswith("ing") and len(w[-1]) > 5 and k not in out:
            stem = w[-1][:-3]
            for cand in (stem, stem + "e", stem[:-1] if stem[-1:] == stem[-2:-1] else stem):
                o = " ".join(w[:-1] + [cand])
                if o != k and o in counts and o not in out:
                    lo, hi = (k, o) if counts[k] <= counts[o] else (o, k)
                    out[lo] = hi
                    break
    return out


def _definitions(sents, alias):
    """Extract definitions/descriptions from explicit term statements.

    A term can be defined as the complement of a phrase ("A table is in
    first normal form when ..."), not only as the first noun phrase in a
    sentence ("Atomicity means ..."). We inspect every detected mention while
    keeping the surrounding source sentence as the evidence.
    """
    out = {}
    complement_prefix = re.compile(r"\b(?:is|are)\s+(?:in|within)\s+$", re.I)
    complement_suffix = re.compile(r"^\s*when\s+(?P<d>.+?)\s*[.!?]*$", re.I)
    action_description = re.compile(
        r"^\s*(?P<v>removes?|reduces?|prevents?|requires?|contains?|includes?|"
        r"organizes?|organises?|represents?|connects?|stores?|controls?|preserves?|maintains?|ensures?|protects?|uses?|identifies?|provides?)\s+"
        r"(?P<d>.+?)\s*[.!?]*$", re.I,
    )
    for s in sents:
        if s.excluded:
            continue
        for m in s.mentions:
            k = alias.get(m.key, m.key)
            if k in out:
                continue
            prefix = s.text[:m.start]
            suffix = s.text[m.end:]

            # A named event can be introduced in a coordinated clause, e.g.
            # "... or a page fault occurs". Preserve the multi-word event as
            # a defined concept even though it is not sentence-initial.
            event = re.match(r"^\s*(?:occurs?|happens?|is\s+triggered|is\s+raised)\b", suffix, re.I)
            event_nouns = {"fault", "failure", "miss", "exception", "error", "interrupt", "timeout"}
            if (event and len(k.split()) >= 2 and k.split()[-1] in event_nouns
                    and re.search(r"\b(?:or|and)\s+$", prefix, re.I)):
                out[k] = (s.idx, event.group(0).strip().lower())
                continue

            # Complement definition: "A table is in first normal form when ..."
            if complement_prefix.search(prefix):
                cm = complement_suffix.match(suffix)
                if cm and len(cm.group("d").split()) >= 3:
                    out[k] = (s.idx, "when " + cm.group("d"))
                    continue

            # Functional definition: "Structured query language, or SQL is used to ...".
            # Keep the explicit purpose clause as the definition instead of
            # filtering it as a generic passive "used" construction.
            if m.start <= 3:
                used_to = re.match(
                    r"^\s*(?:,\s+or\s+[A-Z]{2,})?\s+is\s+used\s+to\s+(?P<d>.+?)\s*[.!?]*$",
                    suffix, re.I,
                )
                if used_to and len(used_to.group("d").split()) >= 3:
                    out[k] = (s.idx, "used to " + used_to.group("d"))
                    continue

                r = DEF_RE.match(suffix)
                if r:
                    d, v = r.group("d"), r.group("v").lower()
                    d = re.sub(r"^that\s+", "", d, flags=re.I)
                    if len(d.split()) >= 3 and not d.lower().startswith(("not ", "also ")):
                        if v not in ("is", "are") or re.match(r"(?:an?|the|one of)\b", d, re.I) or (
                            len(d.split()) >= 5
                            and not re.search(r"\b(?:because|when|if|while|so that|but)\b", d, re.I)
                            and not re.match(r"\w+(?:ed|ing|ly)\b|(?:also|very|often|usually|typically|then|used)\b", d, re.I)
                        ):
                            out[k] = (s.idx, f"{v} {d}" if v.endswith("when") else re.sub(r"^(?:an?|the)\s+", "", d, flags=re.I))
                            continue

                # Concise predicate descriptions can define a term without a
                # copula: "Third normal form removes transitive dependencies...".
                am = action_description.match(suffix)
                if am and len(am.group("d").split()) >= 2:
                    out[k] = (s.idx, f"{am.group('v').lower()} {am.group('d')}")
    return out


def _display(key, items):
    surf = [m.surface for _, m, is_alias in items if not is_alias] or [m.surface for _, m, _ in items]
    forms = []
    for sf in surf:
        keep = sf.isupper() or any(c.isupper() for c in sf[1:]) or sf[:1].islower()
        forms.append(sf if keep else sf.lower())
    exact = [f for f in forms if f.lower() == key]
    return Counter(exact or forms).most_common(1)[0][0]


def build_concepts(sents, min_count=2):
    """Collect candidate concepts, merge acronyms, filter generic/rare ones. Fills Sentence.kept."""
    alias = _acronym_aliases(sents)
    counts = Counter(alias.get(m.key, m.key) for s in sents for m in s.mentions if not _generic(m.key))
    alias = _suffix_aliases(counts, alias)
    occ, acr = defaultdict(list), defaultdict(set)
    for s in sents:
        for m in s.mentions:
            k = alias.get(m.key, m.key)
            if _generic(k):
                continue
            occ[k].append((s.idx, m, k != m.key))
            if k != m.key:
                acr[k].add(m.surface)
    defs = _definitions(sents, alias)
    emph = {s.idx for s in sents if EMPH.search(s.text)}

    for s in sents:
        s.resolved = [(alias.get(m.key, m.key), m) for m in s.mentions]

    def make(minc, only=None):
        out = {}
        for k, items in occ.items():
            sids = sorted({i for i, _, _ in items})
            if only is not None:
                if k not in only:
                    continue
            elif len(sids) < minc and k not in defs:
                continue
            if " " not in k and k not in defs and not any(m.det or m.surface.isupper() for _, m, _ in items):
                continue  # single words never used as a noun phrase (verbs/adjectives in the rule-based fallback)
            c = Concept(k, _display(k, items), len(items), sids, sorted(acr[k]), emphasized=bool(emph & set(sids)))
            if k in defs:
                c.definition_sentence, c.definition = defs[k]
            out[k] = c
        return out

    concepts = make(min_count)
    if len(concepts) < 8:
        concepts = make(1)
    # A term mentioned once is admitted when a stated relationship ties it to an established concept.
    extra = set()
    for r in extract_relations(sents, dict.fromkeys(occ)):
        if r["kind"] != "pattern":
            continue
        source_kept, target_kept = r["source"] in concepts, r["target"] in concepts
        if source_kept != target_kept:
            extra.add(r["target"] if source_kept else r["source"])
        elif not source_kept and not target_kept and r["label"] in {"requires", "depends on", "refers to", "is a type of", "forms part of"}:
            # Preserve both ends of a strong, explicit relation even when each
            # is mentioned only once (e.g. a named prerequisite relationship).
            # Requiring multi-word endpoints avoids promoting isolated generic nouns.
            if all(" " in endpoint for endpoint in (r["source"], r["target"])):
                extra.update((r["source"], r["target"]))
    concepts.update(make(1, only=extra))
    for s in sents:
        s.resolved = [(alias.get(m.key, m.key), m) for m in s.mentions]
        s.kept = [(alias.get(m.key, m.key), m) for m in s.mentions if alias.get(m.key, m.key) in concepts]
    return concepts


RELCLAUSE = re.compile(r"^\W*(?:is|are)\s+(?:an?|the)\s+(?:[\w\-]+\s+){1,4}?(?:that|which|used to|used for|designed to)\s+")


def _label(between, verbs=frozenset()):
    """Relation label for the text between two ADJACENT mentions, or None when the link is not clearly stated.

    A known pattern gives a canonical label; otherwise the verb(s) found between the mentions are used as an open-vocabulary label.
    Anything else between the mentions (an un-extracted noun, a new clause, a negation) rejects the relation."""
    b = RELCLAUSE.sub(" ", between.lower())  # 'X is a Y that minimizes Z' -> 'minimizes Z'
    if len(b) > 70 or ";" in b or "," in b or NEGATION.search(b) or b.lstrip().startswith("to "):
        return None
    if re.search(r"\b(?:more than|less than|rather than|instead of|other than|no longer|without|except|one another|each other|themselves)\b", b):
        return None
        return None
    b = re.sub(r"\b(?:one|two|zero|none) or (?:more|fewer|many|several|less)\b|\band/or\b", " ", b)  # quantifiers are not clause breaks
    if re.search(r"\b(?:and|or|but|while|when|if|because|although|whereas)\b", b):
        return None  # a new clause starts between the mentions: the left concept is not the subject
    words = re.findall(r"[a-z][a-z\-]*", b)
    best, m = None, PASSIVE.search(b)
    if m:
        best = (m.start(), f"{m.group(1)} by", m.span())
    for label, rx in REL:
        for mm in rx.finditer(b):
            if best is None or mm.start() >= best[0]:
                best = (mm.start(), label, mm.span())
    if best is None:
        if re.search(r"\b(?:is|are|be|been|was|were|being)\b", b):
            return None

        acts = [w for w in words if (w in verbs or w in VERBS) and w not in STOP]
        rest = [w for w in words if w not in FUNCTION and w not in verbs]

        if not acts or rest or len(acts) > 2:
            return None

        label = " to ".join(acts) if len(acts) == 2 and "to" in words else " ".join(acts)
        k = max(i for i, w in enumerate(words) if w == acts[-1])

        return label + (" " + words[k + 1] if k + 1 < len(words) and words[k + 1] in PREPS else "")
    rest = re.findall(r"[a-z][a-z\-]*", b[:best[2][0]] + " " + b[best[2][1]:])
    if any(w not in FUNCTION and w not in verbs and len(w) >= 3 for w in rest):
        return None  # an un-extracted word sits between the mentions
    if any((w in VERBS or w in verbs or w in IRREGULAR_PAST) and w not in STOP for w in rest):
        return None  # a second verb ('need to communicate'): the matched verb is not the relation
    return best[1]


def _label_with_context(between, after, verbs=frozenset()):
    """Recognize prerequisite phrasing split around a concept mention.

    Example: "Second normal form when it is in first normal form and ..."
    The phrase "when it is in" states a prerequisite; the following coordinated
    condition confirms that the endpoint is one of the requirements.
    """
    if (re.search(r"\bwhen\s+(?:it|this|that)\s+(?:is|are)\s+in\s*$", between, re.I)
            and re.match(r"\s+and\b", after, re.I)):
        return "requires"
    return _label(between, verbs)


def extract_relations(sents, concepts):
    """Relations between neighbouring mentions (ALL noun phrases, not only kept concepts, so a skipped noun can never be
    mistaken for the object), each with sentence evidence. Both endpoints must be concepts."""
    acc = {}
    for s in sents:
        ms = sorted(s.resolved, key=lambda x: x[1].start)
        pred = set()  # mentions that are predicate nominals ("X is a Y" -> Y)
        for i in range(len(ms) - 1):
            if i in pred:
                continue
            for j in (i + 1, i + 2):
                if j >= len(ms) or (j == i + 2 and (i + 1) not in pred):
                    continue
                (ka, ma), (kb, mb) = ms[i], ms[j]
                wa, wb = set(ka.split()), set(kb.split())
                if ka == kb or wa <= wb or wb <= wa or ma.end > mb.start:
                    continue
                between = s.text[ma.end:mb.start]
                # Definition clauses are captured as concept definitions, not as
                # binary edges to the first noun inside the definition body.
                if re.match(r"^\s*(?:means|is defined as|is called|is known as|denotes)\b", between, re.I):
                    continue
                prefix = s.text[:ma.start]
                lead_in = re.split(r"[,;:.]", prefix)[-1].lower().split()[-3:]  # words just before the subject, within its clause
                # A sentence-initial adjunct such as "Within a technical domain, X ..."
                # does not make X a prepositional object. A noun phrase embedded
                # after another subject ("an attribute in a relation ...") still does.
                initial_adjunct = bool(re.match(
                    r"^\s*(?:within|in|across|among|under|over|through|during|after|before)\b[^,;:.]{1,55}\s+$",
                    prefix, re.I,
                )) or bool(re.search(r"\bexamples?\s+of\s+$", prefix, re.I))
                contextual_prerequisite = bool(re.search(
                    r"\bwhen\s+(?:it|this|that)\s+(?:is|are)\s+in\s*$", between, re.I
                ) and re.match(r"\s+and\b", s.text[mb.end:], re.I))
                if PREPS & set(lead_in) and not (initial_adjunct or contextual_prerequisite) and (RELATIVE.match(between) or ma.start > 0 and not between.strip().startswith(tuple(PREPS))):
                    continue  # a prepositional object is not the subject
                label, kind = _label_with_context(between, s.text[mb.end:], s.verbs), "pattern"
                if label == "forms part of" and re.search(
                    r",\s*.+?\band\b.+?\b(?:(?:may|can|will)\s+)?form\b", between, re.I
                ):
                    continue  # the true subject is the coordinated noun group after the location adjunct
                if label in ("is a", "is a type of") and j == i + 1:
                    pred.add(j)
                if label == "follow" and (
                    "propert" in kb or re.search(r"\b(?:properties|characteristics|components|principles)\b", between + " " + s.text[mb.end:], re.I)
                ):
                    continue  # list membership is represented by contextual edges below
                if label == "provides" and s.text[mb.end:].lower().startswith(" with "):
                    continue  # 'provides EACH PROCESS with a space': the mention is the recipient, the thing provided is not a concept
                if ka not in concepts or kb not in concepts:
                    continue
                if DEMO.search(s.text[max(0, ma.start - 8):ma.start]) or DEMO.search(s.text[max(0, mb.start - 8):mb.start]):
                    continue  # 'this process' is an anaphoric reference, not a named concept
                if label and len(RELCLAUSE.sub(" ", between.lower()).split()) > (12 if j == i + 2 else 6):
                    label = None
                if label is None:
                    if re.search(r"\b(?:one another|each other|themselves)\b", between, re.I):
                        continue  # the predicate already has a pronoun object; the next concept is a new clause
                    if re.fullmatch(r"\s*(?:,\s*)?(?:and|or|&)\s*", between, re.I):
                        continue  # coordinated subjects are not a semantic relation by themselves
                    if j != i + 1 or len(between) > 40 or re.search(r"[;:,]", between):
                        continue
                    label, kind = "related to", "cooccur"
                    ka, kb = sorted((ka, kb))
                r = acc.setdefault((ka, kb, label), dict(source=ka, target=kb, label=label, kind=kind, weight=0.0, evidence=[]))
                r["weight"] += 1.0 if kind == "pattern" else 0.3
                if s.idx not in r["evidence"]:
                    r["evidence"].append(s.idx)

    # Recover direct relations from original sentence text when a parser's
    # noun-chunk boundaries omit or merge one of the endpoints. The concepts
    # dictionary is the whitelist: both phrases must already be extracted.
    for s in sents:
        text = s.text
        spans = []
        for key in concepts:
            words = key.split()
            phrase = r"[\s-]+".join(re.escape(w) for w in words)
            # Allow a plural final token while preserving the normalized key.
            # Keep the optional suffix outside re.escape so '?' stays regex syntax.
            if words and len(words[-1]) > 2 and not words[-1].endswith(("ss", "us", "is")):
                prefix = r"[\s-]+".join(re.escape(w) for w in words[:-1])
                last = re.escape(words[-1]) + r"s?"
                phrase = (prefix + r"[\s-]+" if prefix else "") + last
            rx = re.compile(r"(?<![A-Za-z0-9])" + phrase + r"(?![A-Za-z0-9])", re.I)
            for match in rx.finditer(text):
                spans.append((match.start(), match.end(), key))
        # Resolve overlapping phrases in favour of the longest phrase at each
        # location (e.g. "foreign key" instead of the nested concept "key").
        spans.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        chosen = []
        for span in spans:
            if any(span[0] < end and start < span[1] for start, end, _ in chosen):
                continue
            chosen.append(span)
        chosen.sort(key=lambda item: item[0])

        # A coordinated subject pair can jointly form one concept:
        # "student ID and course ID together may form a composite key".
        # Represent each member as a constituent of the result, rather than
        # incorrectly claiming that the final member alone forms it.
        coordinated_pairs = set()
        form_verb = re.compile(
            r"^\s*(?:(?:together|jointly)\s+)?(?:(?:may|can|will|do|does|did)\s+)?"
            r"(?:form|forms|formed|make|makes|made|create|creates|created|constitute|constitutes)\s+(?:an?|the)?\s*$",
            re.I,
        )
        for target_i in range(2, len(chosen)):
            target = chosen[target_i]
            # Search a short coordinated group immediately before the result.
            for group_size in range(min(4, target_i), 1, -1):
                group = chosen[target_i - group_size:target_i]
                separators = [text[a[1]:b[0]] for a, b in zip(group, group[1:])]
                if not all(re.fullmatch(r"\s*(?:,\s*)?(?:and|or|&)\s*", gap, re.I) or
                           re.fullmatch(r"\s*,\s*", gap) for gap in separators):
                    continue
                # Do not absorb a preceding location/prepositional object
                # into the coordinated subject list ("In an enrollment table,
                # student ID and course ID form ...").
                group_prefix = text[:group[0][0]].rstrip()
                if (re.match(r"^\s*(?:in|within|on|at|from|for|with|of|by|into|across|among|under|over|through|during|after|before)\b", group_prefix, re.I)
                        and "," not in group_prefix):
                    continue
                tail = text[group[-1][1]:target[0]]
                if not form_verb.match(tail):
                    continue
                target_key = target[2]
                for component in group:
                    source_key = component[2]
                    if source_key == target_key:
                        continue
                    coordinated_pairs.add((source_key, target_key))
                    rel_key = (source_key, target_key, "forms part of")
                    rel = acc.setdefault(rel_key, dict(source=source_key, target=target_key, label="forms part of",
                                                       kind="pattern", weight=0.0, evidence=[]))
                    rel["weight"] += 1.0
                    if s.idx not in rel["evidence"]:
                        rel["evidence"].append(s.idx)
                break

        # The parser-based pass above may have treated the last coordinated
        # subject as the sole actor. Remove only that sentence's misleading edge.
        if coordinated_pairs:
            for rel_key, rel in list(acc.items()):
                if (rel["source"], rel["target"]) in coordinated_pairs and rel["label"] != "forms part of" and s.idx in rel["evidence"]:
                    rel["evidence"].remove(s.idx)
                    rel["weight"] = max(0.0, rel["weight"] - 1.0)
                    if not rel["evidence"]:
                        del acc[rel_key]

        for left, right in zip(chosen, chosen[1:]):
            a_start, a_end, ka = left
            b_start, b_end, kb = right
            if ka == kb or b_start < a_end:
                continue
            between = text[a_end:b_start]
            if (ka, kb) in coordinated_pairs:
                continue
            # A definition clause is evidence for the subject's definition,
            # not a binary edge from the subject to the first noun in its body.
            if re.match(r"^\s*(?:means|is defined as|is called|is known as|denotes)\b", between, re.I):
                continue
            label = _label_with_context(between, text[b_end:], getattr(s, "verbs", frozenset()))
            if not label:
                continue
            if label == "forms part of" and re.search(
                r",\s*.+?\band\b.+?\b(?:(?:may|can|will)\s+)?form\b", between, re.I
            ):
                continue  # skip the enclosing table/location; coordinated subjects are linked below
            # Do not resurrect a relation when the source is itself the object
            # of a preceding preposition; allow a sentence-initial scene-setting
            # adjunct such as "Within a technical domain, foreign key refers ...".
            prefix = text[:a_start]
            lead_words = re.split(r"[,;:.]", prefix)[-1].lower().split()[-3:]
            initial_adjunct = bool(re.match(
                r"^\s*(?:within|in|across|among|under|over|through|during|after|before)\b[^,;:.]{1,55}\s+$",
                prefix, re.I,
            )) or bool(re.search(r"\bexamples?\s+of\s+$", prefix, re.I))
            if PREPS & set(lead_words) and not initial_adjunct:
                continue
            key = (ka, kb, label)
            r = acc.setdefault(key, dict(source=ka, target=kb, label=label, kind="pattern", weight=0.0, evidence=[]))
            r["weight"] += 1.0
            if s.idx not in r["evidence"]:
                r["evidence"].append(s.idx)

    # Recover an explicitly introduced list, whether the lecturer states the
    # introduction and enumeration in one ASR chunk or in two adjacent chunks.
    # Crucially, do not treat a later explanatory clause containing commas and
    # "and" as the list itself (e.g. "Consistency means ... rules, and constraints").
    list_signals = re.compile(r"\b(?:properties|characteristics|components|elements|principles|stages|phases|factors|features)\b", re.I)
    list_shape = re.compile(r",.*?\b(?:and|or)\b", re.I)
    predicate_after_list = re.compile(
        r"\b(?:means|is|are|was|were|controls?|requires?|represents?|describes?|refers?|includes?|preserves?|happens?|occur|occurs|remain|remains)\b",
        re.I,
    )

    def add_context_list(parent_items, member_items, evidence_ids):
        if not parent_items or not member_items:
            return
        parent = max(parent_items, key=lambda item: item[1].start)[0]
        # Trim non-essential modifiers when the canonical head concept is
        # already present in the lecture.
        suffix_candidates = [key for key in concepts
                             if parent != key and parent.endswith(" " + key)]
        if suffix_candidates:
            parent = max(suffix_candidates, key=lambda key: len(key.split()))
        else:
            parts = parent.split()
            while len(parts) > 1 and parts[0] in SOFT:
                parts.pop(0)
            simplified = " ".join(parts)
            if simplified in concepts:
                parent = simplified
        members = list(dict.fromkeys(k for k, _ in member_items if k in concepts))
        if not 2 <= len(members) <= 8:
            return
        label = "includes" if "property" in parent.replace("_", " ") else "has property"
        for member in members:
            if member == parent:
                continue
            key = (parent, member, label)
            rel = acc.setdefault(key, dict(source=parent, target=member, label=label,
                                           kind="context", weight=0.0, evidence=[]))
            rel["weight"] += 0.55
            for evidence_id in evidence_ids:
                if evidence_id not in rel["evidence"]:
                    rel["evidence"].append(evidence_id)
            # A list sentence establishes membership; a separate definition
            # sentence for that member supplies independent supporting context.
            # Keep the evidence source-grounded instead of duplicating the list sentence.
            if len(rel["evidence"]) < 2:
                # A second evidence sentence must independently mention both
                # ends of the relation; a member-only definition is not proof
                # that it belongs to this parent concept.
                for candidate in sents:
                    if candidate.idx in rel["evidence"] or candidate.excluded:
                        continue
                    # Compare the literal source sentence as well as the
                    # normalized mention keys: plural forms such as
                    # "plural forms" may not share the exact canonical key.
                    parent_rx = re.compile(r"\b" + re.escape(parent.rstrip("s")) + r"s?\b", re.I)
                    member_rx = re.compile(r"\b" + re.escape(member.rstrip("s")) + r"s?\b", re.I)
                    if parent_rx.search(candidate.text) and member_rx.search(candidate.text):
                        rel["evidence"].append(candidate.idx)
                        break

    for idx, s in enumerate(sents):
        if s.excluded:
            continue
        signal = list_signals.search(s.text)
        shape = list_shape.search(s.text)
        if signal and shape and shape.start() >= signal.end():
            # Same-chunk form: "four important properties Atomicity, ... and durability."
            # Only take mentions after the list signal and before the list conjunction ends.
            parent_items = [(k, m) for k, m in s.resolved if m.end <= signal.start()
                            and (k in concepts or k.split()[-1] in concepts)]
            stop_match = predicate_after_list.search(s.text, shape.end())
            list_end = stop_match.start() if stop_match else len(s.text)
            member_items = [(k, m) for k, m in s.kept
                            if k in concepts and m.start >= signal.end() and m.end <= list_end]
            if not parent_items:
                parent_items = [(k, m) for k, m in s.kept if k in concepts and m.start >= list_end]
            if not parent_items:
                for prior in reversed(sents[max(0, idx - 2):idx]):
                    parent_items = [(k, m) for k, m in prior.kept if k in concepts]
                    if parent_items:
                        break
            add_context_list(parent_items, member_items, [s.idx])
            continue

        if idx == 0:
            continue
        prev = sents[idx - 1]
        prev_signal = list_signals.search(prev.text)
        if prev.excluded or not prev_signal or not shape:
            continue
        # Two-chunk form: the current chunk must be a genuine bare enumeration.
        # If a predicate occurs before its first comma, it is explanatory prose,
        # not a list (prevents false links from "Consistency means ... rules, and ...").
        first_comma = s.text.find(",")
        if first_comma < 0:
            continue
        first_predicate = predicate_after_list.search(s.text)
        if first_predicate and first_predicate.start() < first_comma:
            continue
        if len(s.text[:shape.end()].split()) > 18:
            continue
        parent_items = [(k, m) for k, m in prev.resolved
                        if m.end <= prev_signal.start()
                        and (k in concepts or k.split()[-1] in concepts)]
        stop_match = predicate_after_list.search(s.text, shape.end())
        list_end = stop_match.start() if stop_match else len(s.text)
        member_items = [(k, m) for k, m in s.kept
                        if k in concepts and m.end <= list_end]
        if not parent_items:
            for prior in reversed(sents[max(0, idx - 2):idx - 1]):
                parent_items = [(k, m) for k, m in prior.kept if k in concepts]
                if parent_items:
                    break
        add_context_list(parent_items, member_items, [prev.idx, s.idx])
    # Canonicalize harmless leading quantifiers ("one candidate key" ->
    # "candidate key") when the unmodified concept is present in the lecture.
    quantifiers = {"one", "two", "three", "several", "many", "some", "each"}
    normalized = []
    for rel in acc.values():
        source, target = rel["source"], rel["target"]
        if source.split()[:1] and source.split()[0] in quantifiers:
            candidate = " ".join(source.split()[1:])
            if candidate in concepts:
                source = candidate
        if target.split()[:1] and target.split()[0] in quantifiers:
            candidate = " ".join(target.split()[1:])
            if candidate in concepts:
                target = candidate
        item = dict(rel)
        item["source"], item["target"] = source, target
        # Distinguish an example list from physical containment: an
        # "examples of X include Y" clause expresses an instance relation.
        evidence_text = " ".join(
            sents[i].text for i in item.get("evidence", [])
            if isinstance(i, int) and 0 <= i < len(sents)
        )
        if item.get("label") in {"contains", "includes"} and re.search(
            r"\bexamples?\s+of\b", evidence_text, re.I
        ):
            item["label"] = "has example"
        # Disambiguate a generic descriptor noun in a definition phrase
        # such as "X is the process of selecting Y".
        if (item.get("label") in {"is a", "is a type of"}
                and item.get("target") == "process"
                and re.search(r"\bprocess\s+of\b", evidence_text, re.I)):
            continue
        if source != target:
            normalized.append(item)
    return normalized


def _norm(v):
    mx = max(v) if len(v) else 0
    return [x / mx if mx > 0 else 0.0 for x in v]


def score_concepts(concepts, relations, topic_of):
    """Importance = weighted mix of frequency, graph centrality, definition/emphasis signal,
    topic distinctiveness (TF-IDF over topics), topic spread and early introduction."""
    keys, n_sents, n_topics = list(concepts), len(topic_of), max(topic_of) + 1
    G = nx.Graph()
    G.add_nodes_from(keys)
    for r in relations:
        if r["source"] in concepts and r["target"] in concepts:
            w = G.get_edge_data(r["source"], r["target"], {}).get("weight", 0) + r["weight"]
            G.add_edge(r["source"], r["target"], weight=w)
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {k: 1 / len(keys) for k in keys}
    M = np.zeros((len(keys), n_topics))
    for ci, k in enumerate(keys):
        for sid in concepts[k].sentences:
            M[ci, topic_of[sid]] += 1
    tfidf = TfidfTransformer(norm=None, sublinear_tf=True).fit_transform(M).toarray()
    comp = {
        "freq": _norm([math.log1p(concepts[k].count) for k in keys]),
        "central": _norm([pr[k] for k in keys]),
        "signal": [0.6 * (concepts[k].definition is not None) + 0.4 * concepts[k].emphasized for k in keys],
        "distinct": _norm(tfidf.max(axis=1)),
        "spread": [(M[i] > 0).sum() / n_topics for i in range(len(keys))],
        "early": [1.0 if concepts[k].sentences[0] < 0.2 * n_sents else 0.5 if concepts[k].sentences[0] < 0.5 * n_sents else 0.0 for k in keys],
    }
    for i, k in enumerate(keys):
        c = concepts[k]
        c.components = {name: round(float(v[i]), 3) for name, v in comp.items()}
        base = float(sum(WEIGHTS[n] * v[i] for n, v in comp.items()))
        parts = k.split()
        adjustment = 0.0
        # Multi-word names tend to encode a more precise learning concept than
        # their isolated component nouns. Do not reward list fragments such as
        # "one candidate key" or generic descriptions such as "reliable updates".
        if len(parts) >= 2 and parts[0] not in LOW_SPECIFICITY_PREFIXES:
            adjustment += min(0.18, 0.09 * (len(parts) - 1))
        if len(parts) == 1 and k in LOW_SPECIFICITY_SINGLETONS:
            adjustment -= 0.30
        c.components["specificity_adjustment"] = round(adjustment, 3)
        c.importance = round(min(1.0, max(0.0, base + adjustment)), 3)
        c.topic = int(M[i].argmax())
