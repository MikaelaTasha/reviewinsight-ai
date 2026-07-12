from collections import Counter
from typing import Any

from app.services.sentiment import analyze_sentiment

NEGATIVE_PHRASES = [
    # Japanese: longevity
    "落ちやすい",
    "落ちる",
    "色落ち",
    "消える",
    "持たない",
    "長持ちしない",
    "取れやすい",

    # Japanese: texture / skin reaction
    "乾燥する",
    "乾燥します",
    "乾燥しやすい",
    "皮向け",
    "荒れる",
    "荒れた",
    "痒い",
    "かゆい",
    "ヒリヒリ",
    "ベタつく",
    "べたつく",

    # Japanese: scent / preference
    "苦手",
    "嫌い",
    "きつい",
    "臭い",

    # English
    "fades quickly",
    "fade quickly",
    "doesn't last",
    "does not last",
    "not long lasting",
    "too strong",
    "dislike",
    "hate",
    "irritating",
    "drying",
    "sticky",
]


POSITIVE_PHRASES = [
    # Japanese
    "色持ちが良い",
    "色持ち良い",
    "落ちにくい",
    "長持ち",
    "乾燥しない",
    "荒れない",
    "気に入った",
    "気に入っています",
    "好きです",
    "良い香り",
    "いい香り",
    "コスパが良い",
    "安い",
    "問題なく使用できる",
    "問題なく使用できている",
    "何の問題もなく",
    "肌に優しい",
    "肌にも優しい",
    "刺激がない",
    "刺激なし",

    # English
    "long lasting",
    "long-lasting",
    "doesn't fade",
    "does not fade",
    "doesnt fade",
    "love",
    "good scent",
    "smells good",
    "affordable",
]

POSITIVE_SLANG_PHRASES = [
    "ドンピシャ",
    "好みにドンピシャ",
    "好みど真ん中",
    "最高すぎる",
    "可愛すぎる",
    "かわいすぎる",
    "良すぎる",
    "よすぎる",
    "神",
    "リピ確",
]

def detect_positive_slang_context(text: str) -> bool:
    """
    Detect positive Japanese slang only when it appears
    together with clearly positive context.
    """
    normalized_text = str(text).lower()

    has_positive_context = any(
        phrase.lower() in normalized_text
        for phrase in POSITIVE_SLANG_PHRASES
    )

    has_yabai = any(
        expression in normalized_text
        for expression in [
            "ヤバい",
            "やばい",
            "ヤバ",
            "やば",
        ]
    )

    return has_positive_context and has_yabai

def apply_domain_polarity_rules(
    text: str,
    predicted_sentiment: str,
    confidence: float,
) -> tuple[str, float, str | None]:
    """
    Apply conservative domain-specific polarity corrections.

    The model prediction is only overridden when the text
    contains a clear positive or negative phrase.

    Returns:
        corrected sentiment,
        corrected confidence,
        rule name or None
    """
    normalized_text = str(text).lower()

    positive_slang_context = detect_positive_slang_context(
        normalized_text
    )   

    matched_negative = any(
        phrase.lower() in normalized_text
        for phrase in NEGATIVE_PHRASES
    )

    matched_positive = any(
        phrase.lower() in normalized_text
        for phrase in POSITIVE_PHRASES
    )

    if positive_slang_context:
        return (
            "positive",
            max(float(confidence), 0.90),
            "positive_slang_rule",
        )

    mixed_or_negated_markers = [
        "わけじゃなく",
        "わけではなく",
        "完全に落ちるわけではない",
        "完全に落ちてしまうわけじゃなく",
        "not completely",
        "but still",
        "合わないと",
        "何の問題もなく",
        "問題なく使用",
        "肌にも優しい",
        "肌に優しい",
    ]

    has_mixed_or_negated_context = any(
        marker in normalized_text
        for marker in mixed_or_negated_markers
    )     

    # If both positive and negative phrases appear,
    # keep the model output because the clause may be mixed.
    positive_resolution_markers = [
    "何の問題もなく",
        "問題なく使用",
        "肌にも優しい",
        "肌に優しい",
    ]

    has_positive_resolution = any(
        marker in normalized_text
        for marker in positive_resolution_markers
    )
    if has_positive_resolution:
        return (
            "positive",
            max(float(confidence), 0.85),
            "positive_resolution_rule",
        )
    
    if matched_negative and matched_positive:
        return (
            predicted_sentiment,
            confidence,
            "mixed_rule_match",
        )

    if matched_negative and not has_mixed_or_negated_context:
        return (
            "negative",
            max(float(confidence), 0.85),
            "negative_phrase_rule",
        )

    if matched_positive:
        return (
            "positive",
            max(float(confidence), 0.85),
            "positive_phrase_rule",
        )

    return (
        predicted_sentiment,
        confidence,
        None,
    )

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

                model_sentiment = prediction["label"]
                model_confidence = float(prediction["score"])

                (
                    sentiment,
                    confidence,
                    applied_rule,
                ) = apply_domain_polarity_rules(
                    text=sentence,
                    predicted_sentiment=model_sentiment,
                    confidence=model_confidence,
                )

                evidence_sentiments.append(
                    {
                        "text": sentence,
                        "sentiment": sentiment,
                        "confidence": round(confidence, 4),
                        "model_sentiment": model_sentiment,
                        "model_confidence": round(
                            model_confidence, 
                            4,
                        ),
                        "applied_rule": applied_rule,
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