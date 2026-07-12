import pandas as pd

REQUIRED_COLUMNS = [
    "Product_Name",
    "Brand",
    "Price",
    "Star_Rating",
    "Review_Text"
]


def validate_columns(df: pd.DataFrame):

    missing_columns = []

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns

def validate_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, int]:
    invalid_counts = {}

    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(converted.isna().sum() - df[column].isna().sum())

        if invalid_count > 0:
            invalid_counts[column] = invalid_count

    return invalid_counts