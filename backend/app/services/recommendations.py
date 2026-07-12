from typing import Any


ASPECT_ACTIONS = {
    "Color": {
        "improve": (
            "Review shade accuracy, pigmentation, and color consistency "
            "for this Product Variant."
        ),
        "maintain": (
            "Maintain the current color direction because customer "
            "feedback is predominantly positive."
        ),
    },
    "Undertone": {
        "improve": (
            "Review how well the shade matches different skin undertones "
            "and improve shade guidance for customers."
        ),
        "maintain": (
            "Continue communicating the undertones that this Product "
            "Variant suits well."
        ),
    },
    "Texture": {
        "improve": (
            "Investigate dryness, stickiness, uneven application, or "
            "other texture-related concerns."
        ),
        "maintain": (
            "Maintain the current texture characteristics that customers "
            "describe positively."
        ),
    },
    "Longevity": {
        "improve": (
            "Improve wear time, resistance to fading, and transfer "
            "performance while preserving other strengths."
        ),
        "maintain": (
            "Maintain the current longevity performance because feedback "
            "is predominantly positive."
        ),
    },
    "Finish": {
        "improve": (
            "Review the gloss, matte effect, shine, and final appearance "
            "of this Product Variant."
        ),
        "maintain": (
            "Maintain the current finish because customers generally "
            "respond positively to its appearance."
        ),
    },
    "Packaging": {
        "improve": (
            "Investigate packaging usability, leakage, damage, and "
            "applicator-related issues."
        ),
        "maintain": (
            "Maintain the current packaging design and usability."
        ),
    },
    "Scent": {
        "improve": (
            "Review scent strength and fragrance preference because some "
            "customers may find the scent unpleasant or overpowering."
        ),
        "maintain": (
            "Maintain the current scent profile while continuing to "
            "monitor differences in customer preference."
        ),
    },
    "Price": {
        "improve": (
            "Review pricing and perceived value relative to product "
            "performance and competing products."
        ),
        "maintain": (
            "Maintain the current value proposition because price-related "
            "feedback is predominantly positive."
        ),
    },
    "Skin_Reaction": {
        "improve": (
            "Prioritize investigation of irritation, dryness, itching, "
            "or other skin-reaction concerns."
        ),
        "maintain": (
            "Maintain the current skin-comfort characteristics while "
            "continuing to monitor safety-related feedback."
        ),
    },
}


def calculate_negative_ratio(
    negative: int,
    neutral: int,
    positive: int,
) -> float:
    """
    Calculate the proportion of negative evidence mentions.
    """
    total = negative + neutral + positive

    if total == 0:
        return 0.0

    return negative / total


def determine_recommendation_priority(
    aspect_name: str,
    negative_mentions: int,
    negative_ratio: float,
    total_mentions: int,
) -> str:
    """
    Assign priority using both negative proportion and evidence volume.

    A high negative ratio based on only one sentence should not
    automatically become High priority.
    """

    # Safety concerns remain important, but one isolated report
    # is marked Medium until more evidence appears.
    if aspect_name == "Skin_Reaction":
        if negative_mentions >= 2:
            return "High"

        if negative_mentions == 1:
            return "Medium"

    # High priority requires repeated negative evidence.
    if (
        negative_mentions >= 3
        or (
            negative_mentions >= 2
            and total_mentions >= 3
            and negative_ratio >= 0.50
        )
    ):
        return "High"

    # One negative mention or a meaningful negative ratio
    # receives Medium priority.
    if (
        negative_mentions >= 1
        or (
            total_mentions >= 3
            and negative_ratio >= 0.20
        )
    ):
        return "Medium"

    return "Low"


def generate_aspect_recommendation(
    aspect_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate one explainable recommendation for one aspect.
    """
    negative = int(summary.get("negative", 0))
    neutral = int(summary.get("neutral", 0))
    positive = int(summary.get("positive", 0))
    mentions = int(summary.get("mentions", 0))

    negative_ratio = calculate_negative_ratio(
        negative=negative,
        neutral=neutral,
        positive=positive,
    )

    priority = determine_recommendation_priority(
        aspect_name=aspect_name,
        negative_mentions=negative,
        negative_ratio=negative_ratio,
        total_mentions=mentions,
    )

    action_templates = ASPECT_ACTIONS.get(
        aspect_name,
        {
            "improve": (
                "Investigate the negative evidence associated with this "
                "aspect."
            ),
            "maintain": (
                "Maintain the current performance and continue monitoring "
                "customer feedback."
            ),
        },
    )

    if negative > 0:
        recommendation_type = "improve"
        recommendation = action_templates["improve"]
        reason = (
            f"{negative} of {mentions} evidence mentions were negative "
            f"({negative_ratio:.1%})."
        )
    else:
        recommendation_type = "maintain"
        recommendation = action_templates["maintain"]
        reason = (
            f"No negative evidence was found among {mentions} evidence "
            "mentions."
        )

    return {
        "aspect": aspect_name,
        "recommendation_type": recommendation_type,
        "priority": priority,
        "reason": reason,
        "recommendation": recommendation,
        "metrics": {
            "mentions": mentions,
            "negative": negative,
            "neutral": neutral,
            "positive": positive,
            "negative_ratio": round(negative_ratio, 4),
        },
        "supporting_negative_evidence": summary.get(
            "top_negative_evidence",
            [],
        ),
        "supporting_positive_evidence": summary.get(
            "top_positive_evidence",
            [],
        ),
    }


def generate_product_variant_recommendations(
    product_variant_analysis: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate Decision Support recommendations for each Product Variant.
    """
    recommendation_results = []

    priority_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    for variant in product_variant_analysis:
        aspect_recommendations = []

        for aspect_name, summary in variant.get(
            "aspects",
            {},
        ).items():
            recommendation = generate_aspect_recommendation(
                aspect_name=aspect_name,
                summary=summary,
            )

            aspect_recommendations.append(recommendation)

        aspect_recommendations.sort(
            key=lambda item: (
                priority_order[item["priority"]],
                item["metrics"]["negative"],
                item["metrics"]["negative_ratio"],
            ),
            reverse=True,
        )

        improvement_actions = [
            recommendation
            for recommendation in aspect_recommendations
            if recommendation["recommendation_type"] == "improve"
        ]

        maintenance_actions = [
            recommendation
            for recommendation in aspect_recommendations
            if recommendation["recommendation_type"] == "maintain"
        ]

        overall_priority = (
            improvement_actions[0]["priority"]
            if improvement_actions
            else "Low"
        )

        recommendation_results.append(
            {
                "product_name": variant["product_name"],
                "color": variant["color"],
                "review_count": variant["review_count"],
                "overall_priority": overall_priority,
                "improvement_actions": improvement_actions,
                "maintenance_actions": maintenance_actions,
            }
        )

    recommendation_results.sort(
        key=lambda item: (
            priority_order[item["overall_priority"]],
            item["review_count"],
        ),
        reverse=True,
    )

    return recommendation_results