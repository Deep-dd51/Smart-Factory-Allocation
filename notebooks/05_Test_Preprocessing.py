"""
Test the production preprocessing pipeline.
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.data.data_loader import load_raw_data
from src.data.preprocessing import preprocess_data


# ---------------------------------------------------------
# Load raw data
# ---------------------------------------------------------

df = load_raw_data()

print("=" * 70)
print("PREPROCESSING TEST")
print("=" * 70)

print(f"\nRaw shape: {df.shape}")


# ---------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------

cleaned_df = preprocess_data(df)


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

print("\nCleaned shape:")
print(cleaned_df.shape)


print("\nLead-time statistics:")

print(
    cleaned_df["lead_time_days"]
    .describe()
)


# ---------------------------------------------------------
# Shipping mode analysis
# ---------------------------------------------------------

print("\nLead time by shipping mode:")

print(
    cleaned_df
    .groupby("Ship Mode")["lead_time_days"]
    .agg(
        count="count",
        minimum="min",
        median="median",
        mean="mean",
        maximum="max",
    )
)


# ---------------------------------------------------------
# Show corrected dates
# ---------------------------------------------------------

print("\nSample corrected dates:")

print(
    cleaned_df[
        [
            "Order Date",
            "Original Ship Date",
            "Ship Date",
            "Ship Mode",
            "lead_time_days",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Final checks
# ---------------------------------------------------------

print("\nMissing values:")
print(
    cleaned_df.isnull().sum().sum()
)


print("\nDuplicate rows:")
print(
    cleaned_df.duplicated().sum()
)


print("\nPreprocessing test completed successfully.")