"""
Factory / Location Discovery

Purpose:
    Investigate the dataset for information that can support
    factory allocation and reallocation optimization.

This notebook does NOT create artificial factory assignments.
It only discovers what factory/location information actually
exists in the available data.
"""

import sys
from pathlib import Path

import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# DATA PATH
# ================================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("FACTORY / LOCATION DISCOVERY")
print("=" * 70)


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading featured dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ================================================================
# COLUMN INVENTORY
# ================================================================

print("\n" + "=" * 70)
print("COLUMN INVENTORY")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ================================================================
# POSSIBLE FACTORY COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("FACTORY-RELATED COLUMN SEARCH")
print("=" * 70)

factory_keywords = [
    "factory",
    "plant",
    "warehouse",
    "facility",
    "manufactur",
    "source",
    "origin",
    "location",
]

factory_columns = []

for column in df.columns:
    column_lower = column.lower()

    if any(keyword in column_lower for keyword in factory_keywords):
        factory_columns.append(column)


if factory_columns:
    print("\nPotential factory-related columns found:")

    for column in factory_columns:
        print(f"- {column}")
else:
    print("\nNo explicit factory-related columns found.")


# ================================================================
# LOCATION COLUMNS
# ================================================================

print("\n" + "=" * 70)
print("LOCATION-RELATED COLUMNS")
print("=" * 70)

location_keywords = [
    "country",
    "state",
    "province",
    "city",
    "region",
    "postal",
    "location",
]

location_columns = []

for column in df.columns:
    column_lower = column.lower()

    if any(keyword in column_lower for keyword in location_keywords):
        location_columns.append(column)


for column in location_columns:
    unique_count = df[column].nunique(dropna=True)

    print(
        f"- {column:<20} "
        f"Unique values: {unique_count}"
    )


# ================================================================
# PRODUCT / LOCATION RELATIONSHIP
# ================================================================

print("\n" + "=" * 70)
print("PRODUCT / LOCATION RELATIONSHIP")
print("=" * 70)

if "Product ID" in df.columns and "location_key" in df.columns:

    product_location = (
        df.groupby("Product ID")["location_key"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nLocations served by each product:\n")
    print(product_location.to_string())

else:
    print(
        "\nProduct ID or location_key is not available."
    )


# ================================================================
# LOCATION / PRODUCT RELATIONSHIP
# ================================================================

print("\n" + "=" * 70)
print("LOCATION / PRODUCT RELATIONSHIP")
print("=" * 70)

if "Product ID" in df.columns and "location_key" in df.columns:

    location_product = (
        df.groupby("location_key")["Product ID"]
        .nunique()
        .sort_values(ascending=False)
    )

    print("\nProducts appearing in each location:\n")
    print(location_product.to_string())

else:
    print(
        "\nProduct ID or location_key is not available."
    )


# ================================================================
# PRODUCT + LOCATION + PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("PRODUCT / LOCATION PERFORMANCE")
print("=" * 70)

required_columns = [
    "Product ID",
    "location_key",
    "lead_time_days",
    "Sales",
    "Gross Profit",
    "Cost",
]

if all(column in df.columns for column in required_columns):

    performance = (
        df.groupby(
            ["Product ID", "location_key"]
        )
        .agg(
            records=("Product ID", "size"),
            avg_lead_time=("lead_time_days", "mean"),
            median_lead_time=("lead_time_days", "median"),
            total_sales=("Sales", "sum"),
            total_profit=("Gross Profit", "sum"),
            total_cost=("Cost", "sum"),
        )
        .reset_index()
    )

    performance["profit_margin"] = (
        performance["total_profit"]
        / performance["total_sales"]
    )

    performance = performance.sort_values(
        ["Product ID", "avg_lead_time"]
    )

    print("\nSample product-location performance:\n")

    print(
        performance
        .head(30)
        .to_string(index=False)
    )

else:

    print(
        "\nRequired performance columns are not available."
    )


# ================================================================
# LOCATION PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("LOCATION PERFORMANCE SUMMARY")
print("=" * 70)

if "location_key" in df.columns:

    location_performance = (
        df.groupby("location_key")
        .agg(
            records=("location_key", "size"),
            avg_lead_time=("lead_time_days", "mean"),
            median_lead_time=("lead_time_days", "median"),
            total_sales=("Sales", "sum"),
            total_profit=("Gross Profit", "sum"),
        )
        .sort_values(
            "avg_lead_time",
            ascending=True
        )
    )

    print(
        location_performance
        .to_string()
    )


# ================================================================
# CHECK FOR FACTORY ASSIGNMENT
# ================================================================

print("\n" + "=" * 70)
print("FACTORY ASSIGNMENT CONCLUSION")
print("=" * 70)

if factory_columns:

    print(
        "\nExplicit factory-related columns exist."
    )

    print(
        "These columns should be investigated before "
        "building the optimization model."
    )

else:

    print(
        "\nNo explicit factory assignment column was found "
        "in the featured dataset."
    )

    print(
        "\nCurrent interpretation:"
    )

    print(
        "location_key represents the destination/location "
        "associated with each order."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "We should NOT treat location_key as a factory "
        "without additional evidence."
    )

    print(
        "\nThe next step is to inspect the original dataset "
        "and project files for factory information."
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("DISCOVERY SUMMARY")
print("=" * 70)

print(f"\nRows                    : {len(df):,}")
print(
    f"Unique products         : "
    f"{df['Product ID'].nunique() if 'Product ID' in df.columns else 'N/A'}"
)
print(
    f"Unique locations        : "
    f"{df['location_key'].nunique() if 'location_key' in df.columns else 'N/A'}"
)

print(
    f"Explicit factory fields : "
    f"{len(factory_columns)}"
)

print("\nFactory discovery completed.")
print("=" * 70)