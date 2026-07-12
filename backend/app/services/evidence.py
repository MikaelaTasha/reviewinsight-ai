from collections import defaultdict
from typing import Any


def calculate_evidence_score(
    confidence: float,
    text: str,
) -> float:
    """
    Calculate a simple ranking score for one evidence sentence.

    The score combines:
    - model confidence: 70%
    - sentence informativeness based on length: 30%

    This is an explainable baseline and can later be expanded
    with helpful votes, verified purchases, or semantic relevance.
    """
    confidence_score = max(
        0.0,
        min(float(confidence), 1.0),
    )

    # Longer sentences often contain more useful explanation.
    # The length contribution stops increasing after 80 characters.
    length_score = min(len(text.strip()) / 80, 1.0)

    final_score = (
        0.70 * confidence_score
        + 0.30 * length_score
    )

    return round(final_score, 4)


def rank_aspect_evidence(
    aspect_sentiment_results: list[list[dict[str, Any]]],
    top_k: int = 3,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Group and rank evidence by aspect and sentiment.

    Example:
    {
        "Color": {
            "positive": [...],
            "neutral": [...],
            "negative": [...]
        }
    }
    """
    grouped_evidence = defaultdict(
        lambda: {
            "negative": [],
            "neutral": [],
            "positive": [],
        }
    )

    seen_evidence = set()

    for review_aspects in aspect_sentiment_results:
        for aspect_result in review_aspects:
            aspect_name = aspect_result["aspect"]

            for evidence in aspect_result["evidence_sentiments"]:
                text = str(evidence["text"]).strip()
                sentiment = evidence["sentiment"]
                confidence = float(evidence["confidence"])

                # Prevent duplicate evidence from appearing more than once
                # under the same Aspect–Sentiment Pair.
                evidence_key = (
                    aspect_name,
                    sentiment,
                    text,
                )

                if not text or evidence_key in seen_evidence:
                    continue

                seen_evidence.add(evidence_key)

                evidence_score = calculate_evidence_score(
                    confidence=confidence,
                    text=text,
                )

                grouped_evidence[aspect_name][sentiment].append(
                    {
                        "text": text,
                        "confidence": round(confidence, 4),
                        "evidence_score": evidence_score,
                    }
                )

    ranked_evidence = {}

    for aspect_name, sentiment_groups in grouped_evidence.items():
        ranked_evidence[aspect_name] = {}

        for sentiment, evidence_items in sentiment_groups.items():
            sorted_items = sorted(
                evidence_items,
                key=lambda item: item["evidence_score"],
                reverse=True,
            )

            ranked_evidence[aspect_name][sentiment] = sorted_items[
                :top_k
            ]

    return ranked_evidence