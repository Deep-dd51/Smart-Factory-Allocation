"""
Feature Inspection
------------------
Inspect the cleaned dataset before feature engineering.
"""

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_nassau_candy.csv"
)

df = pd.read_csv(DATA_FILE)


print("=" * 70)
print("SMART FACTORY ALLOCATION - FEATURE INSPECTION")
print("=" * 70)


# ---------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------

print("\nDataset shape:")
print(df.shape)


# ---------------------------------------------------------
# Columns
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ---------------------------------------------------------
# Data types
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())


# ---------------------------------------------------------
# Unique values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:

    print(
        f"{column:30} "
        f"{df[column].nunique():6} unique"
    )


# ---------------------------------------------------------
# Sample data
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)

print(
    df.head(10).to_string(index=False)
)


# ---------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    df[numeric_columns]
    .describe()
    .T
    .to_string()
)


# ---------------------------------------------------------
# Categorical columns
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL VALUES")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print(
        df[column]
        .value_counts()
        .head(20)
        .to_string()
    )


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print(
    f"Missing values : {df.isnull().sum().sum()}"
)

print(
    f"Duplicates     : {df.duplicated().sum()}"
)


print("\n" + "=" * 70)
print("FEATURE INSPECTION COMPLETED")
print("=" * 70)