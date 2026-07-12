import re
import unicodedata

import pandas as pd


def clean_review_text(value: object) -> str:
    """Normalize one review while preserving meaningful content."""
    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize full-width and half-width Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Replace repeated spaces, tabs, and line breaks with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_reviews(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean review data and report what was changed."""
    cleaned_df = df.copy()

    original_rows = len(cleaned_df)

    cleaned_df["Clean_Review_Text"] = cleaned_df["Review_Text"].apply(
        clean_review_text
    )

    # Remove rows with no usable review text.
    empty_review_mask = cleaned_df["Clean_Review_Text"].eq("")
    empty_reviews_removed = int(empty_review_mask.sum())

    cleaned_df = cleaned_df.loc[~empty_review_mask].copy()

    # Remove only rows that are exact duplicates.
    duplicate_mask = cleaned_df.duplicated(keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())

    cleaned_df = cleaned_df.loc[~duplicate_mask].copy()

    # Give missing variants a readable value.
    cleaned_df["Color"] = (
        cleaned_df["Color"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    cleaned_df = cleaned_df.reset_index(drop=True)

    summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned_df),
        "empty_reviews_removed": empty_reviews_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
    }

    return cleaned_df, summary