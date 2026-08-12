"""
Factory Allocation Candidate Analysis

Purpose:
    Identify reliable product-location combinations that could be
    considered as alternative factory/location assignments.

The analysis:
    1. Loads the featured Nassau Candy dataset.
    2. Validates data quality.
    3. Calculates product-location performance.
    4. Identifies reliable product-location combinations.
    5. Calculates current product-level baselines.
    6. Generates allocation candidates where the alternative
       location has better average lead time.
    7. Ranks and saves the candidates.
"""

from pathlib import Path
import sys

import pandas as pd


# ======================================================================
# PROJECT PATH
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# IMPORTS
# ======================================================================

from src.data.feature_engineering import engineer_features


# ======================================================================
# FILE PATHS
# ======================================================================

FEATURED_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_allocation"
)

PRODUCT_LOCATION_FILE = (
    OUTPUT_DIR
    / "product_location_performance.csv"
)

CANDIDATE_FILE = (
    OUTPUT_DIR
    / "allocation_candidates.csv"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

# Minimum number of historical records required for a
# product-location combination to be considered reliable.
MIN_RECORDS = 20


# ======================================================================
# LOAD DATA
# ======================================================================

print("=" * 70)
print("FACTORY ALLOCATION CANDIDATE ANALYSIS")
print("=" * 70)

print("\nLoading featured dataset...")

if not FEATURED_DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nFeatured dataset not found:\n{FEATURED_DATA_FILE}"
    )

df = pd.read_csv(FEATURED_DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ======================================================================
# DATA QUALITY
# ======================================================================

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

required_columns = [
    "Product ID",
    "location_key",
    "lead_time_days",
    "Sales",
    "Gross Profit",
    "Units",
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"- {column}"
            for column in missing_required
        )
    )

print(f"Rows                 : {len(df):,}")
print(
    f"Unique products      : "
    f"{df['Product ID'].nunique():,}"
)
print(
    f"Unique locations     : "
    f"{df['location_key'].nunique():,}"
)
print(
    f"Missing lead times   : "
    f"{df['lead_time_days'].isna().sum():,}"
)
print(
    f"Missing locations    : "
    f"{df['location_key'].isna().sum():,}"
)


# ======================================================================
# PRODUCT-LOCATION PERFORMANCE
# ======================================================================

print("\n" + "=" * 70)
print("PRODUCT-LOCATION PERFORMANCE")
print("=" * 70)

product_location_performance = (
    df.groupby(
        [
            "Product ID",
            "location_key",
        ],
        dropna=False
    )
    .agg(
        records=(
            "lead_time_days",
            "count"
        ),
        avg_lead_time=(
            "lead_time_days",
            "mean"
        ),
        median_lead_time=(
            "lead_time_days",
            "median"
        ),
        total_sales=(
            "Sales",
            "sum"
        ),
        total_profit=(
            "Gross Profit",
            "sum"
        ),
        total_cost=(
            "Cost",
            "sum"
        ),
        total_units=(
            "Units",
            "sum"
        ),
    )
    .reset_index()
)


# ----------------------------------------------------------------------
# Profit margin
# ----------------------------------------------------------------------

product_location_performance["profit_margin"] = (
    product_location_performance["total_profit"]
    /
    product_location_performance["total_sales"]
).fillna(0)


# ----------------------------------------------------------------------
# Sort by product and lead time
# ----------------------------------------------------------------------

product_location_performance = (
    product_location_performance
    .sort_values(
        [
            "Product ID",
            "avg_lead_time"
        ]
    )
    .reset_index(drop=True)
)


# ----------------------------------------------------------------------
# Create output directory
# ----------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------------------
# Save performance table
# ----------------------------------------------------------------------

product_location_performance.to_csv(
    PRODUCT_LOCATION_FILE,
    index=False
)

print(
    "\nProduct-location performance saved to:"
)

print(PRODUCT_LOCATION_FILE)


# ======================================================================
# RELIABLE PRODUCT-LOCATION COMBINATIONS
# ======================================================================

print("\n" + "=" * 70)
print("RELIABLE PRODUCT-LOCATION COMBINATIONS")
print("=" * 70)

print(
    f"\nMinimum records required: {MIN_RECORDS}"
)


reliable_combinations = (
    product_location_performance[
        product_location_performance["records"]
        >= MIN_RECORDS
    ]
    .copy()
)


print(
    f"Reliable combinations: "
    f"{len(reliable_combinations)}"
)


# ----------------------------------------------------------------------
# Display fastest reliable combinations
# ----------------------------------------------------------------------

fastest_reliable = (
    reliable_combinations
    .sort_values(
        "avg_lead_time"
    )
    .head(20)
)


print(
    "\nFastest reliable product-location combinations:"
)

print(
    fastest_reliable[
        [
            "Product ID",
            "location_key",
            "records",
            "avg_lead_time",
            "median_lead_time",
            "total_sales",
            "total_profit",
        ]
    ]
    .to_string(index=False)
)


# ======================================================================
# CURRENT PRODUCT BASELINES
# ======================================================================

print("\n" + "=" * 70)
print("CURRENT PRODUCT BASELINES")
print("=" * 70)


current_baselines = (
    df.groupby(
        "Product ID",
        dropna=False
    )
    .agg(
        product_records=(
            "lead_time_days",
            "count"
        ),
        current_avg_lead_time=(
            "lead_time_days",
            "mean"
        ),
        current_median_lead_time=(
            "lead_time_days",
            "median"
        ),
        current_total_sales=(
            "Sales",
            "sum"
        ),
        current_total_profit=(
            "Gross Profit",
            "sum"
        ),
        current_total_units=(
            "Units",
            "sum"
        ),
    )
    .reset_index()
)


current_baselines = (
    current_baselines
    .sort_values(
        "current_avg_lead_time"
    )
    .reset_index(drop=True)
)


print(
    current_baselines[
        [
            "Product ID",
            "product_records",
            "current_avg_lead_time",
            "current_median_lead_time",
            "current_total_sales",
            "current_total_profit",
            "current_total_units",
        ]
    ]
    .to_string(index=False)
)


# ======================================================================
# GENERATE ALLOCATION CANDIDATES
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING ALLOCATION CANDIDATES")
print("=" * 70)


candidates = []


# ----------------------------------------------------------------------
# Evaluate every reliable product-location combination
# ----------------------------------------------------------------------

for _, row in reliable_combinations.iterrows():

    product_id = row["Product ID"]
    candidate_location = row["location_key"]

    candidate_avg = row["avg_lead_time"]
    candidate_median = row["median_lead_time"]

    candidate_records = row["records"]

    candidate_sales = row["total_sales"]
    candidate_profit = row["total_profit"]

    # --------------------------------------------------------------
    # Find current product baseline
    # --------------------------------------------------------------

    baseline = current_baselines[
        current_baselines["Product ID"]
        == product_id
    ]

    if baseline.empty:
        continue

    baseline_row = baseline.iloc[0]

    current_avg = baseline_row[
        "current_avg_lead_time"
    ]

    current_median = baseline_row[
        "current_median_lead_time"
    ]

    current_sales = baseline_row[
        "current_total_sales"
    ]

    current_profit = baseline_row[
        "current_total_profit"
    ]

    # --------------------------------------------------------------
    # Only consider alternatives that improve lead time
    # --------------------------------------------------------------

    if candidate_avg >= current_avg:
        continue

    # --------------------------------------------------------------
    # Lead-time improvement
    # --------------------------------------------------------------

    lead_time_improvement_days = (
        current_avg
        -
        candidate_avg
    )

    if current_avg != 0:

        lead_time_improvement_pct = (
            lead_time_improvement_days
            /
            current_avg
        ) * 100

    else:

        lead_time_improvement_pct = 0.0

    # --------------------------------------------------------------
    # Profit comparison
    #
    # NOTE:
    # The historical profit of the candidate location is shown
    # for comparison. It does NOT mean that profit will definitely
    # change after reassignment.
    # --------------------------------------------------------------

    profit_difference = (
        candidate_profit
        -
        current_profit
    )

    # --------------------------------------------------------------
    # Create candidate record
    # --------------------------------------------------------------

    candidates.append(
        {
            "Product ID": product_id,

            "candidate_location":
                candidate_location,

            "current_avg_lead_time":
                current_avg,

            "candidate_avg_lead_time":
                candidate_avg,

            "current_median_lead_time":
                current_median,

            "candidate_median_lead_time":
                candidate_median,

            "lead_time_improvement_days":
                lead_time_improvement_days,

            "lead_time_improvement_pct":
                lead_time_improvement_pct,

            "candidate_records":
                candidate_records,

            "candidate_total_sales":
                candidate_sales,

            "candidate_total_profit":
                candidate_profit,

            "current_total_sales":
                current_sales,

            "current_total_profit":
                current_profit,

            "profit_difference":
                profit_difference,
        }
    )


# ======================================================================
# CREATE CANDIDATE DATAFRAME
# ======================================================================

candidates_df = pd.DataFrame(
    candidates
)


print(
    f"\nAllocation candidates generated: "
    f"{len(candidates_df)}"
)


# ======================================================================
# RANK CANDIDATES
# ======================================================================

if not candidates_df.empty:

    # --------------------------------------------------------------
    # Reliability score
    #
    # More records = more reliable historical evidence.
    # Log transformation prevents very large record counts
    # from dominating the score.
    # --------------------------------------------------------------

    import numpy as np

    candidates_df[
        "reliability_score"
    ] = np.log1p(
        candidates_df[
            "candidate_records"
        ]
    )


    # --------------------------------------------------------------
    # Candidate score
    #
    # Prioritizes:
    #   1. Lead-time improvement
    #   2. Historical reliability
    #
    # This is NOT the final optimization score.
    # --------------------------------------------------------------

    candidates_df[
        "candidate_score"
    ] = (
        candidates_df[
            "lead_time_improvement_pct"
        ]
        *
        candidates_df[
            "reliability_score"
        ]
    )


    # --------------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------------

    candidates_df = (
        candidates_df
        .sort_values(
            [
                "candidate_score",
                "lead_time_improvement_pct",
                "candidate_records",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------------
    # Add rank
    # --------------------------------------------------------------

    candidates_df[
        "recommendation_rank"
    ] = (
        candidates_df.index
        + 1
    )


    # ==================================================================
    # DISPLAY TOP CANDIDATES
    # ==================================================================

    print("\n" + "=" * 70)
    print("TOP ALLOCATION CANDIDATES")
    print("=" * 70)

    display_columns = [
        "recommendation_rank",
        "Product ID",
        "candidate_location",
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
        "candidate_total_profit",
        "profit_difference",
        "candidate_score",
    ]


    print(
        candidates_df[
            display_columns
        ]
        .head(30)
        .to_string(index=False)
    )


else:

    print(
        "\nNo allocation candidates found."
    )


# ======================================================================
# SAVE CANDIDATES
# ======================================================================

candidates_df.to_csv(
    CANDIDATE_FILE,
    index=False
)


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("CANDIDATE ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"Total products                  : "
    f"{df['Product ID'].nunique():,}"
)

print(
    f"Total locations                 : "
    f"{df['location_key'].nunique():,}"
)

print(
    f"Reliable product-location pairs : "
    f"{len(reliable_combinations):,}"
)

print(
    f"Allocation candidates           : "
    f"{len(candidates_df):,}"
)


if not candidates_df.empty:

    print(
        f"Best lead-time improvement      : "
        f"{candidates_df['lead_time_improvement_pct'].max():.2f}%"
    )

    print(
        f"Average improvement             : "
        f"{candidates_df['lead_time_improvement_pct'].mean():.2f}%"
    )

    best_candidate = candidates_df.iloc[0]

    print("\nBest candidate:")

    print(
        f"Product           : "
        f"{best_candidate['Product ID']}"
    )

    print(
        f"Location          : "
        f"{best_candidate['candidate_location']}"
    )

    print(
        f"Current lead time : "
        f"{best_candidate['current_avg_lead_time']:.2f} days"
    )

    print(
        f"Candidate lead time: "
        f"{best_candidate['candidate_avg_lead_time']:.2f} days"
    )

    print(
        f"Improvement       : "
        f"{best_candidate['lead_time_improvement_pct']:.2f}%"
    )

    print(
        f"Historical records: "
        f"{int(best_candidate['candidate_records'])}"
    )


# ======================================================================
# OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"\nProduct-location performance:\n"
    f"{PRODUCT_LOCATION_FILE}"
)

print(
    f"\nAllocation candidates:\n"
    f"{CANDIDATE_FILE}"
)

print("\n" + "=" * 70)
print("FACTORY ALLOCATION CANDIDATE ANALYSIS COMPLETED")
print("=" * 70)