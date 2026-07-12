import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.aspects import (
    extract_aspects,
    summarize_aspects,
)
from app.services.preprocessing import preprocess_reviews
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

    analysis_df["Aspect_Results"] = aspect_results

    analysis_df["Detected_Aspects"] = [
        [
            result["aspect"]
            for result in review_aspects
        ]
        for review_aspects in aspect_results
    ]

    aspect_summary = summarize_aspects(aspect_results)

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

    # Return analysis results.
    return {
        "status": "success",
        "filename": file.filename,
        "analyzed_reviews": len(analysis_df),
        "needs_review_count": needs_review_count,
        "sentiment_summary": sentiment_summary,
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
        "aspect_summary": aspect_summary,
    }