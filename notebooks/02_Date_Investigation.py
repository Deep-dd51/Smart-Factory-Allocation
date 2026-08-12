"""
Date Integrity Investigation
-----------------------------
Investigates the relationship between Order Date and Ship Date.
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
# Load data
# ---------------------------------------------------------

from src.data.data_loader import load_raw_data

df = load_raw_data()


print("=" * 70)
print("DATE INTEGRITY INVESTIGATION")
print("=" * 70)


# ---------------------------------------------------------
# Convert dates
# ---------------------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


# ---------------------------------------------------------
# Check conversion
# ---------------------------------------------------------

print("\nDate conversion:")
print(
    df[["Order Date", "Ship Date"]]
    .isna()
    .sum()
)


# ---------------------------------------------------------
# Date ranges
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATE RANGES")
print("=" * 70)

print(
    f"Order Date minimum : {df['Order Date'].min()}"
)

print(
    f"Order Date maximum : {df['Order Date'].max()}"
)

print(
    f"Ship Date minimum  : {df['Ship Date'].min()}"
)

print(
    f"Ship Date maximum  : {df['Ship Date'].max()}"
)


# ---------------------------------------------------------
# Calculate raw lead time
# ---------------------------------------------------------

df["raw_lead_time_days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days


# ---------------------------------------------------------
# Lead-time statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RAW LEAD TIME")
print("=" * 70)

print(
    df["raw_lead_time_days"]
    .describe()
)


# ---------------------------------------------------------
# Lead-time distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LEAD TIME VALUE COUNTS")
print("=" * 70)

print(
    df["raw_lead_time_days"]
    .value_counts()
    .sort_index()
    .head(30)
)


# ---------------------------------------------------------
# Extremely large lead times
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LARGE LEAD TIMES")
print("=" * 70)

large_lead_time = df[
    df["raw_lead_time_days"] > 365
]

print(
    f"Rows with lead time > 365 days: "
    f"{len(large_lead_time)}"
)

print(
    large_lead_time[
        [
            "Order ID",
            "Order Date",
            "Ship Date",
            "Ship Mode",
            "raw_lead_time_days"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Same Day orders
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAME DAY SHIPPING")
print("=" * 70)

same_day = df[
    df["Ship Mode"].str.strip().str.lower() == "same day"
]

print(
    f"Same Day records: {len(same_day)}"
)

print(
    same_day[
        [
            "Order Date",
            "Ship Date",
            "Ship Mode",
            "raw_lead_time_days"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Negative lead times
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NEGATIVE LEAD TIMES")
print("=" * 70)

negative = df[
    df["raw_lead_time_days"] < 0
]

print(
    f"Negative lead-time records: {len(negative)}"
)


# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATE INVESTIGATION COMPLETED")
print("=" * 70)