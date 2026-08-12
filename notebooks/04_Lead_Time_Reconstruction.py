"""
Lead Time Reconstruction Analysis
----------------------------------
Identifies systematic date offsets and
tests whether realistic shipping durations
can be reconstructed.
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
# Remove systematic offsets
# ---------------------------------------------------------

df["candidate_lead_time"] = (
    df["raw_lead_time"]
    - 904
    - 365
    * (
        (df["raw_lead_time"] - 904) // 365
    )
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("=" * 70)
print("RECONSTRUCTED LEAD TIME ANALYSIS")
print("=" * 70)

print("\nCandidate lead-time statistics:")

print(
    df["candidate_lead_time"]
    .describe()
)


# ---------------------------------------------------------
# Distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CANDIDATE LEAD-TIME DISTRIBUTION")
print("=" * 70)

print(
    df["candidate_lead_time"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# By shipping mode
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CANDIDATE LEAD TIME BY SHIPPING MODE")
print("=" * 70)

mode_summary = (
    df.groupby("Ship Mode")["candidate_lead_time"]
    .agg(
        count="count",
        minimum="min",
        median="median",
        mean="mean",
        maximum="max"
    )
)

print(mode_summary)


# ---------------------------------------------------------
# Invalid candidate values
# ---------------------------------------------------------

invalid = df[
    (df["candidate_lead_time"] < 0)
    | (df["candidate_lead_time"] > 30)
]

print("\n" + "=" * 70)
print("POTENTIALLY INVALID RECONSTRUCTED VALUES")
print("=" * 70)

print(
    f"Records outside 0-30 days: {len(invalid)}"
)


if len(invalid) > 0:

    print(
        invalid[
            [
                "Order ID",
                "Order Date",
                "Ship Date",
                "Ship Mode",
                "raw_lead_time",
                "candidate_lead_time"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Offset groups
# ---------------------------------------------------------

df["offset_group"] = (
    (df["raw_lead_time"] - df["candidate_lead_time"])
)

print("\n" + "=" * 70)
print("OFFSET GROUPS")
print("=" * 70)

print(
    df["offset_group"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RECONSTRUCTION ANALYSIS COMPLETED")
print("=" * 70)