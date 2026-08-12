"""
Factory Allocation Audit
------------------------

Investigates the relationship between:

- Product
- Product family
- Division
- Region
- State
- City
- Location
- Sales
- Profit
- Shipping mode
- Lead time

Purpose:
    Determine what information is available for building
    the factory reallocation recommendation engine.
"""

import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# -------------------------------------------------------------------
# DATA FILE
# -------------------------------------------------------------------

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

print("=" * 70)
print("FACTORY ALLOCATION AUDIT")
print("=" * 70)

print("\nLoading featured dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# -------------------------------------------------------------------
# ALL COLUMNS
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("AVAILABLE COLUMNS")
print("=" * 70)

for number, column in enumerate(
    df.columns,
    start=1
):
    print(
        f"{number:2d}. {column}"
    )


# -------------------------------------------------------------------
# PRODUCT ANALYSIS
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("PRODUCT ANALYSIS")
print("=" * 70)

if "Product ID" in df.columns:

    print(
        f"Unique products: "
        f"{df['Product ID'].nunique()}"
    )

    print("\nTop products by record count:")

    print(
        df["Product ID"]
        .value_counts()
        .head(20)
        .to_string()
    )


# -------------------------------------------------------------------
# PRODUCT FAMILY
# -------------------------------------------------------------------

if "product_family" in df.columns:

    print("\n" + "=" * 70)
    print("PRODUCT FAMILY")
    print("=" * 70)

    print(
        df["product_family"]
        .value_counts()
        .to_string()
    )


# -------------------------------------------------------------------
# LOCATION ANALYSIS
# -------------------------------------------------------------------

if "location_key" in df.columns:

    print("\n" + "=" * 70)
    print("LOCATION ANALYSIS")
    print("=" * 70)

    print(
        f"Unique locations: "
        f"{df['location_key'].nunique()}"
    )

    print("\nLocation distribution:")

    print(
        df["location_key"]
        .value_counts()
        .head(30)
        .to_string()
    )


# -------------------------------------------------------------------
# PRODUCT → LOCATION RELATIONSHIP
# -------------------------------------------------------------------

if (
    "Product ID" in df.columns
    and "location_key" in df.columns
):

    print("\n" + "=" * 70)
    print("PRODUCT → LOCATION RELATIONSHIP")
    print("=" * 70)

    product_location = (
        df.groupby(
            "Product ID"
        )["location_key"]
        .nunique()
    )

    print(
        "\nProducts appearing in multiple locations:"
    )

    multiple_locations = (
        product_location[
            product_location > 1
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        multiple_locations
        .head(30)
        .to_string()
    )

    print(
        f"\nProducts with multiple locations: "
        f"{len(multiple_locations)}"
    )


# -------------------------------------------------------------------
# LOCATION → PRODUCT RELATIONSHIP
# -------------------------------------------------------------------

if (
    "Product ID" in df.columns
    and "location_key" in df.columns
):

    print("\n" + "=" * 70)
    print("LOCATION → PRODUCT RELATIONSHIP")
    print("=" * 70)

    location_products = (
        df.groupby(
            "location_key"
        )["Product ID"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        location_products
        .head(30)
        .to_string()
    )


# -------------------------------------------------------------------
# LEAD TIME BY LOCATION
# -------------------------------------------------------------------

if (
    "location_key" in df.columns
    and "lead_time_days" in df.columns
):

    print("\n" + "=" * 70)
    print("LEAD TIME BY LOCATION")
    print("=" * 70)

    location_lead_time = (
        df.groupby(
            "location_key"
        )["lead_time_days"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )

    print(
        location_lead_time
        .to_string()
    )


# -------------------------------------------------------------------
# PROFIT BY LOCATION
# -------------------------------------------------------------------

if (
    "location_key" in df.columns
    and "Gross Profit" in df.columns
):

    print("\n" + "=" * 70)
    print("PROFIT BY LOCATION")
    print("=" * 70)

    location_profit = (
        df.groupby(
            "location_key"
        )["Gross Profit"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
        .sort_values(
            "sum",
            ascending=False
        )
    )

    print(
        location_profit
        .to_string()
    )


# -------------------------------------------------------------------
# SHIP MODE
# -------------------------------------------------------------------

if "Ship Mode" in df.columns:

    print("\n" + "=" * 70)
    print("SHIPPING MODE DISTRIBUTION")
    print("=" * 70)

    print(
        df["Ship Mode"]
        .value_counts()
        .to_string()
    )


# -------------------------------------------------------------------
# DIVISION
# -------------------------------------------------------------------

if "Division" in df.columns:

    print("\n" + "=" * 70)
    print("DIVISION DISTRIBUTION")
    print("=" * 70)

    print(
        df["Division"]
        .value_counts()
        .to_string()
    )


# -------------------------------------------------------------------
# REGION
# -------------------------------------------------------------------

if "Region" in df.columns:

    print("\n" + "=" * 70)
    print("REGION DISTRIBUTION")
    print("=" * 70)

    print(
        df["Region"]
        .value_counts()
        .to_string()
    )


# -------------------------------------------------------------------
# FACTORY-LIKE COLUMNS
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FACTORY-RELATED COLUMN CHECK")
print("=" * 70)

factory_keywords = [
    "factory",
    "plant",
    "warehouse",
    "facility",
    "location",
    "manufacturing",
    "manufacturer",
]

factory_columns = []

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in factory_keywords
    ):
        factory_columns.append(
            column
        )


if factory_columns:

    print(
        "Potential factory/location columns:"
    )

    for column in factory_columns:
        print(
            f"- {column}"
        )

else:

    print(
        "No explicit factory column found."
    )


# -------------------------------------------------------------------
# COMPLETION
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FACTORY ALLOCATION AUDIT COMPLETED")
print("=" * 70)