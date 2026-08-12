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

    df = df.copy()

    df["profit_margin"] = (
        df["Gross Profit"] / df["Sales"]
    )

    df["sales_per_unit"] = (
        df["Sales"] / df["Units"]
    )

    df["cost_per_unit"] = (
        df["Cost"] / df["Units"]
    )

    df["profit_per_unit"] = (
        df["Gross Profit"] / df["Units"]
    )

    df["cost_ratio"] = (
        df["Cost"] / df["Sales"]
    )

    return df


# -------------------------------------------------------------------
# Shipping features
# -------------------------------------------------------------------

def create_shipping_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

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

    df = df.copy()

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

    Target-derived features are excluded to
    prevent data leakage.
    """

    model_columns = [

        # Temporal
        "order_year",
        "order_month",
        "order_quarter",
        "order_day_of_week",
        "is_weekend",

        # Shipping
        "Ship Mode",

        # Product
        "Product ID",
        "Division",

        # Geography
        "Country/Region",
        "Region",
        "State/Province",
        "City",
        "Postal Code",

        # Commercial
        "Sales",
        "Units",
        "Gross Profit",
        "Cost",

        # Unit economics
        "profit_margin",
        "sales_per_unit",
        "cost_per_unit",
        "profit_per_unit",
        "cost_ratio",

        # Geographic grouping
        "location_key",
    ]

    # Check for missing columns
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

    return df[model_columns].copy()