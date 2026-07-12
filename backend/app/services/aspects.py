import re
from collections import Counter
from typing import Any

from app.resources.aspect_dictionary import ASPECT_KEYWORDS


def split_sentences(text: str) -> list[str]:
    """
    Split Japanese and English review text into sentences.
    """
    if not text:
        return []

    sentences = re.split(
        r"(?<=[。！？.!?])\s*|\n+",
        str(text),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

def split_clauses(sentence: str) -> list[str]:
    """
    Split one sentence around contrast expressions.

    Examples:
    - 発色は良いけど、落ちやすい
    - The color is nice, but it fades quickly
    """
    if not sentence:
        return []

    clause_pattern = re.compile(
        r"\s*(?:"
        r"けれども|けれど|だけど|ですけど|けど|"
        r"しかし|でも|ただし|一方で|"
        r"but|however|although|though|yet"
        r")\s*[,、]?\s*",
        flags=re.IGNORECASE,
    )

    clauses = clause_pattern.split(sentence)

    return [
        clause.strip(" ,、")
        for clause in clauses
        if clause.strip(" ,、")
    ]

def split_evidence_units(text: str) -> list[str]:
    """
    Split review text first into sentences and then into
    smaller contrast-based clauses.
    """
    evidence_units = []

    for sentence in split_sentences(text):
        clauses = split_clauses(sentence)

        if clauses:
            evidence_units.extend(clauses)
        else:
            evidence_units.append(sentence)

    return evidence_units


def extract_aspects(review_text: str) -> list[dict[str, Any]]:
    """
    Detect aspects mentioned in one review and return
    supporting evidence sentences.
    """
    evidence_units = split_evidence_units(review_text)
    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        evidence_sentences = []

        for evidence_text in evidence_units:
            normalized_text = evidence_text.lower()

            if any(
                keyword.lower() in normalized_text
                for keyword in keywords
            ):
                evidence_sentences.append(evidence_text)

        if evidence_sentences:
            detected_aspects.append(
                {
                    "aspect": aspect,
                    "evidence": evidence_sentences,
                    "mention_count": len(evidence_sentences),
                }
            )

    return detected_aspects


def summarize_aspects(
    aspect_results: list[list[dict[str, Any]]],
) -> dict[str, int]:
    """
    Count how many reviews mention each aspect.
    """
    counter = Counter()

    for review_aspects in aspect_results:
        for result in review_aspects:
            counter[result["aspect"]] += 1

    return {
        aspect: counter.get(aspect, 0)
        for aspect in ASPECT_KEYWORDS
    }