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
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def extract_aspects(review_text: str) -> list[dict[str, Any]]:
    """
    Detect aspects mentioned in one review and return
    supporting evidence sentences.
    """
    sentences = split_sentences(review_text)
    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        evidence_sentences = []

        for sentence in sentences:
            normalized_sentence = sentence.lower()

            if any(
                keyword.lower() in normalized_sentence
                for keyword in keywords
            ):
                evidence_sentences.append(sentence)

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