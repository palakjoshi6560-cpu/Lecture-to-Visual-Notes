"""Generic, context-aware correction engine for ASR segments.

Subject vocabulary is loaded from external JSON packs so the engine itself
remains independent of any particular course or lecture.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern


@dataclass(frozen=True)
class WordRule:
    rule_id: str
    pattern: Pattern[str]
    replacement: str


@dataclass(frozen=True)
class PhraseRule:
    rule_id: str
    pattern: Pattern[str]
    replacement: str


@dataclass(frozen=True)
class ContextPack:
    name: str
    required_anchor: Pattern[str]
    supporting_anchors: tuple[Pattern[str], ...]
    min_supporting: int
    word_rules: tuple[WordRule, ...] = ()
    phrase_rules: tuple[PhraseRule, ...] = ()


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _load_packs() -> tuple[ContextPack, ...]:
    """Load optional vocabulary packs stored beside this generic engine."""
    pack_dir = Path(__file__).with_name("terminology_packs")
    packs: list[ContextPack] = []
    if not pack_dir.is_dir():
        return ()

    for path in sorted(pack_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            word_rules = tuple(
                WordRule(item["id"], _compile(item["pattern"]), item["replacement"])
                for item in raw.get("word_rules", [])
            )
            phrase_rules = tuple(
                PhraseRule(item["id"], _compile(item["pattern"]), item["replacement"])
                for item in raw.get("phrase_rules", [])
            )
            packs.append(ContextPack(
                name=raw["name"],
                required_anchor=_compile(raw["required_anchor"]),
                supporting_anchors=tuple(_compile(p) for p in raw.get("supporting_anchors", [])),
                min_supporting=int(raw.get("min_supporting", 0)),
                word_rules=word_rules,
                phrase_rules=phrase_rules,
            ))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid terminology pack: {path.name}: {exc}") from exc
    return tuple(packs)


_CONTEXT_PACKS = _load_packs()


def _active_packs(all_text: str) -> list[ContextPack]:
    active = []
    for pack in _CONTEXT_PACKS:
        if not pack.required_anchor.search(all_text):
            continue
        hits = sum(1 for pattern in pack.supporting_anchors if pattern.search(all_text))
        if hits >= pack.min_supporting:
            active.append(pack)
    return active


def _record(corrections, index, match, replacement, rule, pack):
    corrections.append({
        "segment": index,
        "original": match.group(0),
        "corrected": replacement,
        "rule": rule,
        "context": pack,
    })


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.title()
    return replacement


def correct_segments(segments):
    """Correct context-authorized terms, including phrases split across ASR segments.

    Segment timestamps and extra fields are preserved. A phrase crossing a segment
    boundary is stored in the segment where the phrase starts; remaining text stays
    in its original segment. The audit log identifies that starting segment.
    """
    original = [dict(segment) for segment in (segments or [])]
    all_text = " ".join(str(segment.get("text", "")) for segment in original)
    active = _active_packs(all_text)
    if not active:
        return original, []

    corrected = [str(segment.get("text", "")) for segment in original]
    corrections = []

    # First correct single-token variants and phrases wholly contained in a segment.
    for index, text in enumerate(corrected):
        for pack in active:
            for rule in pack.word_rules:
                matches = list(rule.pattern.finditer(text))
                if matches:
                    text = rule.pattern.sub(
                        lambda m, canonical=rule.replacement: _preserve_case(m.group(0), canonical),
                        text,
                    )
                    for match in matches:
                        _record(corrections, index, match, rule.replacement, rule.rule_id, pack.name)
            for rule in pack.phrase_rules:
                matches = list(rule.pattern.finditer(text))
                if matches:
                    text = rule.pattern.sub(rule.replacement, text)
                    for match in matches:
                        _record(corrections, index, match, rule.replacement, rule.rule_id, pack.name)
        corrected[index] = text

    # Then inspect adjacent segment pairs. This catches phrases such as
    # when Whisper splits a technical phrase across adjacent ASR segments.
    for index in range(len(corrected) - 1):
        for pack in active:
            for rule in pack.phrase_rules:
                left, right = corrected[index], corrected[index + 1]
                boundary = len(left)
                combined = left + " " + right
                matches = [
                    m for m in rule.pattern.finditer(combined)
                    if m.start() < boundary and m.end() > boundary + 1
                ]
                # Process right-to-left so earlier offsets remain valid.
                for match in reversed(matches):
                    before = combined[:match.start()]
                    after = combined[match.end():]
                    corrected[index] = before + rule.replacement
                    corrected[index + 1] = after.lstrip()
                    _record(corrections, index, match, rule.replacement, rule.rule_id, pack.name)

    result = []
    for index, segment in enumerate(original):
        updated = dict(segment)
        updated["text"] = corrected[index]
        result.append(updated)
    return result, corrections
