"""
Feature Engineering
-------------------

Creates business-oriented features for the
Smart Factory Allocation project.
"""

from pathlib import Path
import sys

import pandas as pd


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Date features
# -------------------------------------------------------------------

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar features from Order Date.
    """

    df = df.copy()

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        errors="coerce"
    )

    df["order_year"] = (
        df["Order Date"].dt.year
    )

    df["order_month"] = (
        df["Order Date"].dt.month
    )

    df["order_quarter"] = (
        df["Order Date"].dt.quarter
    )

    df["order_day"] = (
        df["Order Date"].dt.day
    )

    df["order_day_of_week"] = (
        df["Order Date"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["order_day_of_week"] >= 5
    ).astype(int)

    return df


# -------------------------------------------------------------------
# Profitability features
# -------------------------------------------------------------------

def create_profitability_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create profitability-related features.
    """

    df = df.copy()

    # Profit margin
    df["profit_margin"] = (
        df["Gross Profit"]
        / df["Sales"]
    )

    # Sales generated per unit
    df["sales_per_unit"] = (
        df["Sales"]
        / df["Units"]
    )

    # Cost per unit
    df["cost_per_unit"] = (
        df["Cost"]
        / df["Units"]
    )

    # Profit per unit
    df["profit_per_unit"] = (
        df["Gross Profit"]
        / df["Units"]
    )

    # Cost as percentage of sales
    df["cost_ratio"] = (
        df["Cost"]
        / df["Sales"]
    )

    return df


# -------------------------------------------------------------------
# Shipping features
# -------------------------------------------------------------------

def create_shipping_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create shipping-performance features.
    """

    df = df.copy()

    # Lead-time category
    df["lead_time_category"] = pd.cut(
        df["lead_time_days"],
        bins=[
            -1,
            0,
            2,
            4,
            7,
            float("inf")
        ],
        labels=[
            "Same Day",
            "Fast",
            "Normal",
            "Slow",
            "Very Slow"
        ]
    )

    # Profit generated per shipping day.
    #
    # Same-day shipments have 0 lead time,
    # so we replace 0 with 1 for this metric
    # to avoid division by zero.
    df["profit_per_shipping_day"] = (
        df["Gross Profit"]
        / df["lead_time_days"].replace(0, 1)
    )

    return df


# -------------------------------------------------------------------
# Product features
# -------------------------------------------------------------------

def create_product_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create product-level features.
    """

    df = df.copy()

    # Product family from Product ID
    df["product_family"] = (
        df["Product ID"]
        .astype(str)
        .str.split("-")
        .str[0]
    )

    return df


# -------------------------------------------------------------------
# Geographic features
# -------------------------------------------------------------------

def create_geographic_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create useful geographic features.
    """

    df = df.copy()

    df["location_key"] = (
        df["State/Province"].astype(str)
        + "_"
        + df["Region"].astype(str)
    )

    return df


# -------------------------------------------------------------------
# Complete feature engineering pipeline
# -------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Returns a dataset containing both:
    - analytical/business features
    - features that may later be selected for modeling
    """

    df = create_date_features(df)

    df = create_profitability_features(df)

    df = create_shipping_features(df)

    df = create_product_features(df)

    df = create_geographic_features(df)

    return df


# -------------------------------------------------------------------
# Model feature selection
# -------------------------------------------------------------------

def get_model_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Select features that are safe to use for
    lead-time prediction.

    Target-derived features are intentionally excluded
    to prevent data leakage.
    """

    model_columns = [

        # -----------------------------------------------------------
        # Temporal features
        # -----------------------------------------------------------

        "order_year",
        "order_month",
        "order_quarter",
        "order_day_of_week",
        "is_weekend",

        # -----------------------------------------------------------
        # Shipping configuration
        # -----------------------------------------------------------

        "Ship Mode",

        # -----------------------------------------------------------
        # Product information
        # -----------------------------------------------------------

        "Product ID",
        "Division",

        # -----------------------------------------------------------
        # Geographic information
        # -----------------------------------------------------------

        "Country/Region",
        "Region",
        "State/Province",
        "City",
        "Postal Code",

        # -----------------------------------------------------------
        # Commercial information
        # -----------------------------------------------------------

        "Sales",
        "Units",
        "Gross Profit",
        "Cost",

        # -----------------------------------------------------------
        # Unit economics
        # -----------------------------------------------------------

        "profit_margin",
        "sales_per_unit",
        "cost_per_unit",
        "profit_per_unit",
        "cost_ratio",

        # -----------------------------------------------------------
        # Geographic grouping
        # -----------------------------------------------------------

        "location_key",
    ]

    # ---------------------------------------------------------------
    # Check that all required columns exist
    # ---------------------------------------------------------------

    missing_columns = [
        column
        for column in model_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------------
    # Return model feature matrix
    # ---------------------------------------------------------------

    return df[model_columns].copy()