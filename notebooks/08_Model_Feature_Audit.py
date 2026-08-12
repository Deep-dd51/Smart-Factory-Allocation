"""
Model Feature Audit
-------------------

Checks the features that will eventually be used
for lead-time prediction.
"""

import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from src.data.feature_engineering import (
    engineer_features,
    get_model_features,
)


# -------------------------------------------------------------------
# Load cleaned dataset
# -------------------------------------------------------------------

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_nassau_candy.csv"
)

df = pd.read_csv(DATA_FILE)


# -------------------------------------------------------------------
# Engineer features
# -------------------------------------------------------------------

featured_df = engineer_features(df)


# -------------------------------------------------------------------
# Select model features
# -------------------------------------------------------------------

X = get_model_features(featured_df)

y = featured_df["lead_time_days"]


# -------------------------------------------------------------------
# Basic information
# -------------------------------------------------------------------

print("=" * 70)
print("MODEL FEATURE AUDIT")
print("=" * 70)

print(
    f"\nFeature matrix shape : {X.shape}"
)

print(
    f"Target shape         : {y.shape}"
)


# -------------------------------------------------------------------
# Model features
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL FEATURES")
print("=" * 70)

for number, column in enumerate(
    X.columns,
    start=1
):
    print(
        f"{number:2}. {column}"
    )


# -------------------------------------------------------------------
# Target
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET")
print("=" * 70)

print(
    "Target: lead_time_days"
)

print("\nTarget statistics:")

print(
    y.describe().to_string()
)


# -------------------------------------------------------------------
# Leakage check
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

forbidden_features = [

    # Target itself
    "lead_time_days",

    # Features derived from target
    "lead_time_category",
    "profit_per_shipping_day",

    # Dates created from the target
    "Ship Date",
    "Original Ship Date",
]


leakage_found = [
    column
    for column in forbidden_features
    if column in X.columns
]


if leakage_found:

    print(
        "WARNING - possible leakage:"
    )

    for column in leakage_found:
        print(
            f"- {column}"
        )

else:

    print(
        "PASS - no target-derived features "
        "found in model feature matrix."
    )


# -------------------------------------------------------------------
# Data quality
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print(
    f"Missing values in X: "
    f"{X.isnull().sum().sum()}"
)

print(
    f"Missing values in y: "
    f"{y.isnull().sum()}"
)


# -------------------------------------------------------------------
# Feature data types
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE DATA TYPES")
print("=" * 70)

print(
    X.dtypes.to_string()
)


# -------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL FEATURE AUDIT COMPLETED")
print("=" * 70)