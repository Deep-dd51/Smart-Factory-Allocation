"""
Shipping Date Offset Analysis
-----------------------------
Investigates whether the corrupted Ship Date
contains a systematic date offset.
"""

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.data.data_loader import load_raw_data


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = load_raw_data()


# ---------------------------------------------------------
# Convert dates
# ---------------------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d-%m-%Y"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="%d-%m-%Y"
)


# ---------------------------------------------------------
# Raw lead time
# ---------------------------------------------------------

df["raw_lead_time"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days


# ---------------------------------------------------------
# Analyze by Ship Mode
# ---------------------------------------------------------

print("=" * 70)
print("LEAD TIME BY SHIPPING MODE")
print("=" * 70)

mode_analysis = (
    df.groupby("Ship Mode")["raw_lead_time"]
    .agg(
        count="count",
        minimum="min",
        median="median",
        mean="mean",
        maximum="max"
    )
    .sort_values("median")
)

print(mode_analysis)


# ---------------------------------------------------------
# Minimum lead time by shipping mode
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MINIMUM RAW LEAD TIME BY SHIPPING MODE")
print("=" * 70)

print(
    df.groupby("Ship Mode")["raw_lead_time"]
    .min()
    .sort_values()
)


# ---------------------------------------------------------
# Most common lead times by shipping mode
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MOST COMMON RAW LEAD TIMES")
print("=" * 70)

for mode in sorted(df["Ship Mode"].unique()):

    print(f"\n--- {mode} ---")

    values = (
        df.loc[
            df["Ship Mode"] == mode,
            "raw_lead_time"
        ]
        .value_counts()
        .head(10)
    )

    print(values)


# ---------------------------------------------------------
# Compare raw lead time against a 904-day offset
# ---------------------------------------------------------

df["adjusted_lead_time_904"] = (
    df["raw_lead_time"] - 904
)


print("\n" + "=" * 70)
print("LEAD TIME AFTER REMOVING 904-DAY OFFSET")
print("=" * 70)

print(
    df["adjusted_lead_time_904"]
    .describe()
)


# ---------------------------------------------------------
# Adjusted lead time by shipping mode
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ADJUSTED LEAD TIME BY SHIPPING MODE")
print("=" * 70)

adjusted_analysis = (
    df.groupby("Ship Mode")["adjusted_lead_time_904"]
    .agg(
        count="count",
        minimum="min",
        median="median",
        mean="mean",
        maximum="max"
    )
)

print(adjusted_analysis)


# ---------------------------------------------------------
# Sample records
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

print(
    df[
        [
            "Order Date",
            "Ship Date",
            "Ship Mode",
            "raw_lead_time",
            "adjusted_lead_time_904"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)