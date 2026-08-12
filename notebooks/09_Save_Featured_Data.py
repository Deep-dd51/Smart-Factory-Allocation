"""
Save Featured Dataset
---------------------

Loads the cleaned Nassau Candy dataset,
runs feature engineering, validates the
result, and saves the featured dataset.
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
    engineer_features
)


# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_nassau_candy.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)


# -------------------------------------------------------------------
# Load cleaned dataset
# -------------------------------------------------------------------

print("=" * 70)
print("FEATURED DATASET CREATION")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Input shape: {df.shape}"
)


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

print("\nRunning feature engineering...")

featured_df = engineer_features(df)

print(
    f"Featured shape: {featured_df.shape}"
)


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

missing_values = (
    featured_df.isnull().sum().sum()
)

duplicate_rows = (
    featured_df.duplicated().sum()
)

print(
    f"Missing values : {missing_values}"
)

print(
    f"Duplicate rows : {duplicate_rows}"
)

print(
    f"Columns        : {len(featured_df.columns)}"
)


# -------------------------------------------------------------------
# Check target
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

if "lead_time_days" in featured_df.columns:

    print(
        "Target column found: lead_time_days"
    )

    print(
        f"Minimum lead time: "
        f"{featured_df['lead_time_days'].min()}"
    )

    print(
        f"Maximum lead time: "
        f"{featured_df['lead_time_days'].max()}"
    )

else:

    raise ValueError(
        "lead_time_days column is missing!"
    )


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING DATASET")
print("=" * 70)

featured_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------------------------
# Confirm
# -------------------------------------------------------------------

print(
    "\nFeatured dataset saved successfully!"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


print("\n" + "=" * 70)
print("FEATURED DATASET CREATION COMPLETED")
print("=" * 70)