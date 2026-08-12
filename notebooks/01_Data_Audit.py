"""
Data Audit
----------
Initial inspection of the Nassau Candy Distributor dataset.
"""

import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project path
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Import project data loader
# -------------------------------------------------------------------

from src.data.data_loader import load_raw_data


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

print("=" * 70)
print("SMART FACTORY ALLOCATION - DATA AUDIT")
print("=" * 70)

df = load_raw_data()

print(f"\nDataset shape: {df.shape}")


# -------------------------------------------------------------------
# 1. Column names
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. COLUMN NAMES")
print("=" * 70)

for index, column in enumerate(df.columns, start=1):
    print(f"{index:2}. {column}")


# -------------------------------------------------------------------
# 2. Data types
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# -------------------------------------------------------------------
# 3. Dataset information
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DATASET INFORMATION")
print("=" * 70)

df.info()


# -------------------------------------------------------------------
# 4. First five rows
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. FIRST FIVE ROWS")
print("=" * 70)

print(df.head().to_string())


# -------------------------------------------------------------------
# 5. Missing values
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

print(
    missing
    .sort_values("missing_count", ascending=False)
    .to_string()
)


# -------------------------------------------------------------------
# 6. Duplicate records
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("6. DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# -------------------------------------------------------------------
# 7. Unique values
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("7. UNIQUE VALUES")
print("=" * 70)

unique_counts = pd.DataFrame({
    "unique_values": df.nunique(),
    "data_type": df.dtypes.astype(str)
})

print(
    unique_counts
    .sort_values("unique_values", ascending=False)
    .to_string()
)


# -------------------------------------------------------------------
# 8. Categorical columns
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("8. CATEGORICAL COLUMNS")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    print(
        f"\n{column} "
        f"({df[column].nunique()} unique values)"
    )

    print(
        df[column]
        .value_counts()
        .head(10)
        .to_string()
    )


# -------------------------------------------------------------------
# 9. Numerical columns
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("9. NUMERICAL COLUMNS")
print("=" * 70)

numerical_columns = df.select_dtypes(
    include=["number"]
).columns

print(list(numerical_columns))


print("\nNumerical statistics:")

print(
    df[numerical_columns]
    .describe()
    .T
    .to_string()
)


# -------------------------------------------------------------------
# 10. Date columns
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("10. POTENTIAL DATE COLUMNS")
print("=" * 70)

date_candidates = [
    column
    for column in df.columns
    if "date" in column.lower()
]

print(date_candidates)


for column in date_candidates:

    print(f"\n--- {column} ---")

    print(
        df[column]
        .head(10)
        .to_string(index=False)
    )


# -------------------------------------------------------------------
# End
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETED")
print("=" * 70)