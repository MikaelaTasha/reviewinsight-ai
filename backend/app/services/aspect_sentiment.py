from collections import Counter
from typing import Any

from app.services.sentiment import analyze_sentiment


def get_dominant_sentiment(
    sentiment_counts: Counter,
) -> str:
    """
    Return the most frequent sentiment.

    Return 'mixed' when multiple sentiments share
    the highest count.
    """
    if not sentiment_counts:
        return "unknown"

    highest_count = max(sentiment_counts.values())

    top_sentiments = [
        sentiment
        for sentiment, count in sentiment_counts.items()
        if count == highest_count
    ]

    if len(top_sentiments) > 1:
        return "mixed"

    return top_sentiments[0]


def analyze_aspect_sentiments(
    aspect_results: list[list[dict[str, Any]]],
    batch_size: int = 16,
) -> tuple[
    list[list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """
    Analyze sentiment for every evidence sentence attached
    to each detected aspect.

    The same evidence sentence is analyzed only once, even
    when it supports multiple aspects.
    """

    # Collect unique evidence sentences.
    unique_sentences = []

    for review_aspects in aspect_results:
        for aspect_result in review_aspects:
            for sentence in aspect_result["evidence"]:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)

    # Handle the case where no aspects were detected.
    if not unique_sentences:
        return aspect_results, {}

    # Run sentiment analysis on all unique sentences in one batch.
    predictions, _ = analyze_sentiment(
        unique_sentences,
        batch_size=batch_size,
    )

    # Connect each sentence to its prediction.
    prediction_by_sentence = {
        sentence: prediction
        for sentence, prediction in zip(
            unique_sentences,
            predictions,
        )
    }

    enhanced_results = []
    global_aspect_counts = {}

    for review_aspects in aspect_results:
        enhanced_review_aspects = []

        for aspect_result in review_aspects:
            aspect_name = aspect_result["aspect"]
            evidence_sentiments = []

            sentiment_counts = Counter()
            confidence_values = []

            for sentence in aspect_result["evidence"]:
                prediction = prediction_by_sentence[sentence]

                sentiment = prediction["label"]
                confidence = float(prediction["score"])

                evidence_sentiments.append(
                    {
                        "text": sentence,
                        "sentiment": sentiment,
                        "confidence": confidence,
                    }
                )

                sentiment_counts[sentiment] += 1
                confidence_values.append(confidence)

            dominant_sentiment = get_dominant_sentiment(
                sentiment_counts
            )

            average_confidence = (
                round(
                    sum(confidence_values)
                    / len(confidence_values),
                    4,
                )
                if confidence_values
                else 0.0
            )

            enhanced_review_aspects.append(
                {
                    "aspect": aspect_name,
                    "evidence": aspect_result["evidence"],
                    "mention_count": aspect_result["mention_count"],
                    "dominant_sentiment": dominant_sentiment,
                    "average_confidence": average_confidence,
                    "sentiment_distribution": {
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
                    },
                    "evidence_sentiments": evidence_sentiments,
                }
            )

            # Create the overall aspect-level summary.
            if aspect_name not in global_aspect_counts:
                global_aspect_counts[aspect_name] = Counter()

            global_aspect_counts[aspect_name].update(
                sentiment_counts
            )

        enhanced_results.append(enhanced_review_aspects)

    aspect_sentiment_summary = {}

    for aspect_name, sentiment_counts in global_aspect_counts.items():
        total_mentions = sum(sentiment_counts.values())

        aspect_sentiment_summary[aspect_name] = {
            "mentions": total_mentions,
            "negative": sentiment_counts.get("negative", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "positive": sentiment_counts.get("positive", 0),
            "dominant_sentiment": get_dominant_sentiment(
                sentiment_counts
            ),
        }

    return enhanced_results, aspect_sentiment_summary