from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from app.services.aspect_sentiment import get_dominant_sentiment
from app.services.evidence import calculate_evidence_score

import re


def normalize_variant_name(value: str) -> str:
    """
    Normalize Product Variant names for more consistent grouping.
    """
    normalized = str(value).strip()

    normalized = re.sub(
        r"^color:\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.upper()

def analyze_product_variants(
    analysis_df: pd.DataFrame,
    aspect_sentiment_results: list[list[dict[str, Any]]],
    top_k_evidence: int = 2,
) -> list[dict[str, Any]]:
    """
    Aggregate review and aspect analysis by Product Variant.

    A Product Variant is identified by:
    Product_Name + Color
    """
    if len(analysis_df) != len(aspect_sentiment_results):
        raise ValueError(
            "The number of reviews and aspect results must match."
        )

    variant_groups: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    normalized_df = analysis_df.reset_index(drop=True)

    for row_position, row in normalized_df.iterrows():
        product_name = str(row["Product_Name"])
        color = normalize_variant_name(row["Color"])

        variant_key = (
            product_name,
            color,
        )

        if variant_key not in variant_groups:
            variant_groups[variant_key] = {
                "product_name": product_name,
                "color": color,
                "review_count": 0,
                "rating_total": 0.0,
                "confidence_total": 0.0,
                "needs_review_count": 0,
                "overall_sentiments": Counter(),
                "aspects": defaultdict(
                    lambda: {
                        "mentions": 0,
                        "sentiments": Counter(),
                        "evidence": [],
                    }
                ),
            }

        variant_data = variant_groups[variant_key]

        variant_data["review_count"] += 1
        variant_data["rating_total"] += float(
            row["Star_Rating"]
        )
        variant_data["confidence_total"] += float(
            row["Confidence"]
        )

        if bool(row["Needs_Review"]):
            variant_data["needs_review_count"] += 1

        overall_sentiment = str(
            row["Predicted_Sentiment"]
        )

        variant_data["overall_sentiments"][
            overall_sentiment
        ] += 1

        review_aspects = aspect_sentiment_results[
            row_position
        ]

        for aspect_result in review_aspects:
            aspect_name = str(
                aspect_result["aspect"]
            )

            aspect_data = variant_data["aspects"][
                aspect_name
            ]

            aspect_data["mentions"] += int(
                aspect_result["mention_count"]
            )

            sentiment_distribution = aspect_result[
                "sentiment_distribution"
            ]

            aspect_data["sentiments"].update(
                {
                    "negative": int(
                        sentiment_distribution["negative"]
                    ),
                    "neutral": int(
                        sentiment_distribution["neutral"]
                    ),
                    "positive": int(
                        sentiment_distribution["positive"]
                    ),
                }
            )

            for evidence in aspect_result[
                "evidence_sentiments"
            ]:
                evidence_text = str(
                    evidence["text"]
                ).strip()

                evidence_sentiment = str(
                    evidence["sentiment"]
                )

                evidence_confidence = float(
                    evidence["confidence"]
                )

                if not evidence_text:
                    continue

                evidence_score = calculate_evidence_score(
                    confidence=evidence_confidence,
                    text=evidence_text,
                )

                aspect_data["evidence"].append(
                    {
                        "text": evidence_text,
                        "sentiment": evidence_sentiment,
                        "confidence": round(
                            evidence_confidence,
                            4,
                        ),
                        "evidence_score": evidence_score,
                    }
                )

    variant_results = []

    for variant_data in variant_groups.values():
        review_count = variant_data["review_count"]

        average_rating = round(
            variant_data["rating_total"]
            / review_count,
            2,
        )

        average_confidence = round(
            variant_data["confidence_total"]
            / review_count,
            4,
        )

        overall_sentiment_counts = (
            variant_data["overall_sentiments"]
        )

        overall_sentiment = get_dominant_sentiment(
            overall_sentiment_counts
        )

        aspect_summaries = {}

        for aspect_name, aspect_data in (
            variant_data["aspects"].items()
        ):
            sentiment_counts = aspect_data[
                "sentiments"
            ]

            positive_evidence = [
                evidence
                for evidence in aspect_data["evidence"]
                if evidence["sentiment"] == "positive"
            ]

            negative_evidence = [
                evidence
                for evidence in aspect_data["evidence"]
                if evidence["sentiment"] == "negative"
            ]

            positive_evidence.sort(
                key=lambda evidence: evidence[
                    "evidence_score"
                ],
                reverse=True,
            )

            negative_evidence.sort(
                key=lambda evidence: evidence[
                    "evidence_score"
                ],
                reverse=True,
            )

            aspect_summaries[aspect_name] = {
                "mentions": aspect_data["mentions"],
                "negative": sentiment_counts.get(
                    "negative",
                    0,
                ),
                "neutral": sentiment_counts.get(
                    "neutral",
                    0,
                ),
                "positive": sentiment_counts.get(
                    "positive",
                    0,
                ),
                "dominant_sentiment": (
                    get_dominant_sentiment(
                        sentiment_counts
                    )
                ),
                "top_positive_evidence": (
                    positive_evidence[
                        :top_k_evidence
                    ]
                ),
                "top_negative_evidence": (
                    negative_evidence[
                        :top_k_evidence
                    ]
                ),
            }

        concern_aspects = [
            {
                "aspect": aspect_name,
                "negative_mentions": summary[
                    "negative"
                ],
            }
            for aspect_name, summary in (
                aspect_summaries.items()
            )
            if summary["negative"] > 0
        ]

        concern_aspects.sort(
            key=lambda item: item[
                "negative_mentions"
            ],
            reverse=True,
        )

        variant_results.append(
            {
                "product_name": variant_data[
                    "product_name"
                ],
                "color": variant_data["color"],
                "review_count": review_count,
                "average_rating": average_rating,
                "average_sentiment_confidence": (
                    average_confidence
                ),
                "needs_review_count": variant_data[
                    "needs_review_count"
                ],
                "overall_sentiment": (
                    overall_sentiment
                ),
                "sentiment_distribution": {
                    "negative": (
                        overall_sentiment_counts.get(
                            "negative",
                            0,
                        )
                    ),
                    "neutral": (
                        overall_sentiment_counts.get(
                            "neutral",
                            0,
                        )
                    ),
                    "positive": (
                        overall_sentiment_counts.get(
                            "positive",
                            0,
                        )
                    ),
                },
                "aspects": aspect_summaries,
                "top_concern_aspects": (
                    concern_aspects[:3]
                ),
            }
        )

    variant_results.sort(
        key=lambda variant: (
            variant["review_count"],
            variant["average_rating"],
        ),
        reverse=True,
    )

    return variant_results