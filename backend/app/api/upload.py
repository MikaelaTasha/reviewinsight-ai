from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd

router = APIRouter()

REQUIRED_COLUMNS = [
    "Product_Name",
    "Brand",
    "Color",
    "Price",
    "Star_Rating",
    "Review_Text",
    "Verified_Purchase",
]


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    try:
        df = pd.read_csv(file.file)
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be parsed as a valid CSV.",
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "The CSV is missing required columns.",
                "missing_columns": missing_columns,
            },
        )

    return {
        "status": "success",
        "filename": file.filename,
        "total_reviews": len(df),
        "total_brands": int(df["Brand"].nunique()),
        "total_products": int(df["Product_Name"].nunique()),
        "total_variants": int(
            df[["Product_Name", "Color"]]
            .drop_duplicates()
            .shape[0]
        ),
        "average_rating": round(
            float(df["Star_Rating"].mean()),
            2,
        ),
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "columns": list(df.columns),
    }