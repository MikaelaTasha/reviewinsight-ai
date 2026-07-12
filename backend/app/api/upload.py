from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd

from app.services.preprocessing import preprocess_reviews
from app.utils.validation import (
    validate_columns,
    validate_numeric_columns,
)

router = APIRouter()


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Check whether a file was provided.
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

    try:
        # Read the uploaded CSV into a Pandas DataFrame.
        df = pd.read_csv(file.file)

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

        # Check whether numeric columns contain invalid text values.
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

        # Clean review text and prepare the dataset.
        cleaned_df, preprocessing_summary = preprocess_reviews(df)

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

    # Return a summary of the cleaned dataset.
    return {
        "status": "success",
        "filename": file.filename,
        "total_reviews": len(cleaned_df),
        "total_brands": int(cleaned_df["Brand"].nunique()),
        "total_products": int(cleaned_df["Product_Name"].nunique()),
        "total_variants": int(
            cleaned_df[["Product_Name", "Color"]]
            .drop_duplicates()
            .shape[0]
        ),
        "average_rating": round(
            float(cleaned_df["Star_Rating"].mean()),
            2,
        ),
        "missing_values": {
            column: int(count)
            for column, count in cleaned_df.isna().sum().items()
        },
        "columns": list(cleaned_df.columns),
        "preprocessing": preprocessing_summary,
    }