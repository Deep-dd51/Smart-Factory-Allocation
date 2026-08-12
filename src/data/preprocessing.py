"""
Data preprocessing and cleaning pipeline.

This module:
- Loads the raw Nassau Candy dataset.
- Standardizes column names and categorical values.
- Converts date columns.
- Reconstructs corrupted shipping dates.
- Creates reliable shipping lead time.
- Validates the cleaned dataset.
"""

from pathlib import Path
import logging

import pandas as pd


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

BASE_DATE_OFFSET = 904
YEAR_OFFSET_DAYS = 365


# -------------------------------------------------------------------
# Date processing
# -------------------------------------------------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Order Date and Ship Date to datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with parsed date columns.
    """

    df = df.copy()

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    if df["Order Date"].isna().any():
        raise ValueError("Invalid values found in Order Date.")

    if df["Ship Date"].isna().any():
        raise ValueError("Invalid values found in Ship Date.")

    return df


# -------------------------------------------------------------------
# Categorical cleaning
# -------------------------------------------------------------------

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary whitespace from categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    return df


# -------------------------------------------------------------------
# Lead-time reconstruction
# -------------------------------------------------------------------

def reconstruct_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct realistic shipping lead time.

    The raw Ship Date contains systematic offsets.

    Observed offset groups:

        904 days
        1269 days = 904 + 365
        1634 days = 904 + 730

    The underlying lead time is therefore obtained by
    removing the base 904-day offset and the appropriate
    number of 365-day shifts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Order Date and Ship Date.

    Returns
    -------
    pd.DataFrame
        Dataset with reconstructed lead time.
    """

    df = df.copy()

    raw_lead_time = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    years_shifted = (
        (raw_lead_time - BASE_DATE_OFFSET)
        // YEAR_OFFSET_DAYS
    )

    df["lead_time_days"] = (
        raw_lead_time
        - BASE_DATE_OFFSET
        - YEAR_OFFSET_DAYS * years_shifted
    )

    return df


# -------------------------------------------------------------------
# Reconstruct realistic Ship Date
# -------------------------------------------------------------------

def reconstruct_ship_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a corrected Ship Date from Order Date and lead time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing Order Date and lead_time_days.

    Returns
    -------
    pd.DataFrame
        Dataset with corrected Ship Date.
    """

    df = df.copy()

    df["Original Ship Date"] = df["Ship Date"]

    df["Ship Date"] = (
        df["Order Date"]
        + pd.to_timedelta(
            df["lead_time_days"],
            unit="D",
        )
    )

    return df


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_lead_time(df: pd.DataFrame) -> None:
    """
    Validate reconstructed lead times.

    Raises
    ------
    ValueError
        If invalid lead times are detected.
    """

    if df["lead_time_days"].isna().any():
        raise ValueError(
            "Missing lead_time_days detected."
        )

    if (df["lead_time_days"] < 0).any():
        raise ValueError(
            "Negative lead times detected."
        )

    if (df["lead_time_days"] > 30).any():
        raise ValueError(
            "Lead times greater than 30 days detected."
        )


# -------------------------------------------------------------------
# Complete preprocessing pipeline
# -------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Nassau Candy dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    logger.info("Starting preprocessing.")

    df = clean_categorical_columns(df)

    df = parse_dates(df)

    df = reconstruct_lead_time(df)

    validate_lead_time(df)

    df = reconstruct_ship_date(df)

    logger.info(
        "Preprocessing completed successfully."
    )

    return df