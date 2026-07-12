import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.aspect_sentiment import (
    analyze_aspect_sentiments,
)
from app.services.aspects import (
    extract_aspects,
    summarize_aspects,
)
from app.services.evidence import rank_aspect_evidence
from app.services.preprocessing import preprocess_reviews
from app.services.product_variant import (
    analyze_product_variants,
)
from app.services.sentiment import (
    analyze_sentiment,
    assess_prediction,
)
from app.utils.validation import (
    validate_columns,
    validate_numeric_columns,
)




router = APIRouter(
    prefix="/reviews",
    tags=["Review Analysis"],
)

logger = logging.getLogger(__name__)


@router.post("/analyze")
async def analyze_reviews(
    file: UploadFile = File(...),
    limit: int = 50,
):
    """
    Analyze uploaded product reviews using a multilingual
    sentiment-analysis model.

    The endpoint validates the CSV, preprocesses review text,
    predicts sentiment, checks rating-sentiment consistency,
    and returns review recommendations.
    """

    # Check whether a file was uploaded.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    # Accept CSV files only.
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    # Prevent extremely large test requests.
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 200.",
        )

    logger.info(
        "Starting sentiment analysis for file=%s limit=%s",
        file.filename,
        limit,
    )

    # Read the uploaded CSV.
    try:
        df = pd.read_csv(file.file)

    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        ) from exc

    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be parsed as a valid CSV.",
        ) from exc

    # Check required columns.
    missing_columns = validate_columns(df)

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "The CSV is missing required columns.",
                "missing_columns": missing_columns,
            },
        )

    # Check numeric values.
    invalid_numeric_values = validate_numeric_columns(
        df,
        ["Price", "Star_Rating"],
    )

    if invalid_numeric_values:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Some numeric columns contain invalid values.",
                "invalid_values": invalid_numeric_values,
            },
        )

    # Clean and prepare review data.
    cleaned_df, preprocessing_summary = preprocess_reviews(df)

    # Analyze only the requested number of reviews.
    analysis_df = cleaned_df.head(limit).copy()

    texts = analysis_df["Clean_Review_Text"].tolist()

    # Run multilingual sentiment analysis.
    predictions, sentiment_summary = analyze_sentiment(texts)

    # Add model outputs to the DataFrame.
    analysis_df["Predicted_Sentiment"] = [
        prediction["label"]
        for prediction in predictions
    ]

    analysis_df["Confidence"] = [
        prediction["score"]
        for prediction in predictions
    ]

    # Assess whether each prediction should be reviewed.
    assessments = [
        assess_prediction(
            rating=float(rating),
            sentiment=prediction["label"],
            confidence=float(prediction["score"]),
            review_text=str(review_text),
        )
        for rating, prediction, review_text in zip(
            analysis_df["Star_Rating"],
            predictions,
            analysis_df["Review_Text"],
        )
    ]

    # Add assessment results to the DataFrame.
    analysis_df["Expected_Sentiment"] = [
        assessment["expected_sentiment"]
        for assessment in assessments
    ]

    analysis_df["Needs_Review"] = [
        assessment["needs_review"]
        for assessment in assessments
    ]

    analysis_df["Reason"] = [
        assessment["reason"]
        for assessment in assessments
    ]

    analysis_df["Evidence"] = [
        assessment["evidence"]
        for assessment in assessments
    ]

    analysis_df["Review_Priority"] = [
    assessment["review_priority"]
    for assessment in assessments
    ]

    analysis_df["Recommendation"] = [
        assessment["recommendation"]
        for assessment in assessments
    ]

    # Extract product aspects and supporting evidence.
    aspect_results = [
        extract_aspects(str(review_text))
        for review_text in analysis_df["Review_Text"]
    ]

    #count how many reviews mention each aspect.
    aspect_summary = summarize_aspects(aspect_results)

    #analyze the sentiment of every aspect eveidence sentence.
    (
        aspect_sentiment_results,
        aspect_sentiment_summary,
    ) = analyze_aspect_sentiments(aspect_results)

    # Rank the strongest evidence for each Aspect–Sentiment Pair.
    ranked_evidence = rank_aspect_evidence(
        aspect_sentiment_results,
        top_k=3,
    )

    # Aggregate results by Product Name and Color.
    product_variant_analysis = analyze_product_variants(
        analysis_df=analysis_df,
        aspect_sentiment_results=aspect_sentiment_results,
        top_k_evidence=2,
    )

    analysis_df["Aspect_Results"] = (
        aspect_sentiment_results
    )

    analysis_df["Detected_Aspects"] = [
        [
            result["aspect"]
            for result in review_aspects
        ]
        for review_aspects in aspect_sentiment_results
    ]

    # Count how many predictions need manual review.
    needs_review_count = int(
        analysis_df["Needs_Review"].sum()
    )

    logger.info(
        "Completed sentiment analysis for file=%s analyzed_reviews=%s "
        "needs_review=%s",
        file.filename,
        len(analysis_df),
        needs_review_count,
    )

    # Return analysis results dictionary.
    return {
        "status": "success",
        "filename": file.filename,
        "analyzed_reviews": len(analysis_df),
        "needs_review_count": needs_review_count,
        "sentiment_summary": sentiment_summary,
        "aspect_summary": aspect_summary,
        "aspect_sentiment_summary": aspect_sentiment_summary,
        "ranked_evidence": ranked_evidence,
        "product_variant_analysis": product_variant_analysis,
        "preprocessing": preprocessing_summary,
        "sample_results": (
            analysis_df[
                [
                    "Product_Name",
                    "Color",
                    "Star_Rating",
                    "Review_Text",
                    "Predicted_Sentiment",
                    "Expected_Sentiment",
                    "Confidence",
                    "Needs_Review",
                    "Reason",
                    "Evidence",
                    "Review_Priority",
                    "Recommendation",
                    "Detected_Aspects",
                    "Aspect_Results",
                ]
            ]
            .head(10)
            .to_dict(orient="records")
        ),
    }