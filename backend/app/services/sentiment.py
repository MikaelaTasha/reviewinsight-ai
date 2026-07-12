from collections import Counter
from functools import lru_cache
from typing import Any

from transformers import pipeline


MODEL_NAME = (
    "cardiffnlp/"
    "twitter-xlm-roberta-base-sentiment-multilingual"
)


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """
    Load and cache the Hugging Face sentiment model.

    Caching prevents the model from being loaded again
    for every API request.
    """
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def normalize_label(label: str) -> str:
    """
    Convert model labels into readable sentiment names.
    """
    label_mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    if label in label_mapping:
        return label_mapping[label]

    return label.lower()


def analyze_sentiment(
    texts: list[str],
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Predict sentiment for a list of review texts.
    """
    if not texts:
        return [], {
            "negative": 0,
            "neutral": 0,
            "positive": 0,
        }

    sentiment_pipeline = get_sentiment_pipeline()

    predictions = sentiment_pipeline(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )

    formatted_predictions = []

    for prediction in predictions:
        label = normalize_label(prediction["label"])

        formatted_predictions.append(
            {
                "label": label,
                "score": round(float(prediction["score"]), 4),
            }
        )

    counts = Counter(
        prediction["label"]
        for prediction in formatted_predictions
    )

    summary = {
        "negative": counts.get("negative", 0),
        "neutral": counts.get("neutral", 0),
        "positive": counts.get("positive", 0),
    }

    return formatted_predictions, summary


def expected_sentiment_from_rating(rating: float) -> str:
    """
    Convert a star rating into its expected sentiment.

    Ratings 1-2: negative
    Rating 3: neutral
    Ratings 4-5: positive
    """
    if rating >= 4:
        return "positive"

    if rating <= 2:
        return "negative"

    return "neutral"


def rating_sentiment_mismatch(
    rating: float,
    sentiment: str,
) -> bool:
    """
    Return True when the predicted sentiment conflicts
    with the sentiment expected from the star rating.
    """
    expected_sentiment = expected_sentiment_from_rating(rating)

    return sentiment != expected_sentiment


def assess_prediction(
    rating: float,
    sentiment: str,
    confidence: float,
    review_text: str,
) -> dict[str, Any]:
    """
    Assess one model prediction and determine whether
    it should be reviewed manually.
    """
    expected_sentiment = expected_sentiment_from_rating(rating)

    sentiment_mismatch = rating_sentiment_mismatch(
        rating,
        sentiment,
    )

    low_confidence = confidence < 0.60

    needs_review = sentiment_mismatch or low_confidence

    if sentiment_mismatch:
        review_priority = "Medium"
        reason = "Star rating conflicts with predicted sentiment."
        recommendation = (
            "Review this prediction because the star rating "
            "and predicted sentiment disagree."
        )

    elif low_confidence:
        review_priority = "Medium"
        reason = "The sentiment prediction has low confidence."
        recommendation = (
            "Review this prediction manually because the model "
            "confidence is low."
        )

    else:
        review_priority = "Low"
        reason = (
            "The star rating and predicted sentiment are consistent."
        )
        recommendation = (
            "No immediate manual review is required."
        )

    return {
        "expected_sentiment": expected_sentiment,
        "needs_review": needs_review,
        "reason": reason,
        "evidence": [review_text],
        "review_priority": review_priority,
        "recommendation": recommendation,
    }