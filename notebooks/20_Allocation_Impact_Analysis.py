"""
20_Allocation_Impact_Analysis.py

Factory Reallocation & Shipping Optimization
Allocation Impact Analysis

Purpose:
    Evaluate the potential impact of reallocating products to candidate
    factory/location combinations.

Inputs:
    data/processed/featured/_nassau_candy.csv
    data/processed/factory_allocation/allocation_candidates.csv
    data/processed/factory_allocation/product_location_performance.csv

Outputs:
    data/processed/factory_allocation/allocation_impact_analysis.csv
    data/processed/factory_allocation/product_impact_summary.csv
    data/processed/factory_allocation/location_impact_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURED_DIR = PROCESSED_DIR / "featured"
FACTORY_ALLOCATION_DIR = PROCESSED_DIR / "factory_allocation"

OUTPUT_DIR = FACTORY_ALLOCATION_DIR


# ======================================================================
# CONFIGURATION
# ======================================================================

MIN_CANDIDATE_RECORDS = 20

# Only consider candidates that provide a meaningful improvement.
MIN_IMPROVEMENT_DAYS = 0.10
MIN_IMPROVEMENT_PCT = 2.0

# Maximum number of recommendations retained per product.
MAX_CANDIDATES_PER_PRODUCT = 5


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# DATASET DISCOVERY
# ======================================================================

def find_featured_dataset():
    """
    Locate the featured dataset.

    The project currently uses:

        data/processed/featured/_nassau_candy.csv

    The function also supports alternative filenames so the pipeline
    does not break if the filename changes later.
    """

    candidates = [
        FEATURED_DIR / "_nassau_candy.csv",
        FEATURED_DIR / "nassau_candy.csv",
        FEATURED_DIR / "featured_nassau_candy.csv",
        PROCESSED_DIR / "featured_nassau_candy.csv",
        PROCESSED_DIR / "featured_data.csv",
        PROCESSED_DIR / "featured_dataset.csv",
    ]

    # --------------------------------------------------------------
    # Check known locations
    # --------------------------------------------------------------

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    # --------------------------------------------------------------
    # Search recursively inside featured directory
    # --------------------------------------------------------------

    if FEATURED_DIR.exists():
        csv_files = list(FEATURED_DIR.rglob("*.csv"))

        if len(csv_files) == 1:
            return csv_files[0]

        featured_files = [
            path
            for path in csv_files
            if "featured" in path.name.lower()
        ]

        if featured_files:
            return featured_files[0]

        # If multiple files exist, use the first CSV.
        if csv_files:
            return csv_files[0]

    raise FileNotFoundError(
        "\nCould not find featured dataset.\n\n"
        f"Checked directory:\n{FEATURED_DIR}\n\n"
        "Expected a CSV such as:\n"
        f"{FEATURED_DIR / '_nassau_candy.csv'}"
    )


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():

    print_header("LOADING DATA")

    featured_path = find_featured_dataset()

    candidates_path = (
        FACTORY_ALLOCATION_DIR /
        "allocation_candidates.csv"
    )

    performance_path = (
        FACTORY_ALLOCATION_DIR /
        "product_location_performance.csv"
    )

    print("Featured dataset:")
    print(featured_path)

    print()
    print("Allocation candidates:")
    print(candidates_path)

    print()
    print("Product-location performance:")
    print(performance_path)

    # --------------------------------------------------------------
    # Validate files
    # --------------------------------------------------------------

    if not featured_path.exists():
        raise FileNotFoundError(
            f"\nFeatured dataset not found:\n{featured_path}"
        )

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"\nAllocation candidates not found:\n{candidates_path}"
        )

    if not performance_path.exists():
        raise FileNotFoundError(
            f"\nProduct-location performance not found:\n"
            f"{performance_path}"
        )

    # --------------------------------------------------------------
    # Load CSV files
    # --------------------------------------------------------------

    df = pd.read_csv(featured_path)

    candidates = pd.read_csv(candidates_path)

    performance = pd.read_csv(performance_path)

    print()
    print(f"Featured dataset shape     : {df.shape}")
    print(f"Candidate dataset shape    : {candidates.shape}")
    print(f"Performance dataset shape  : {performance.shape}")

    return df, candidates, performance


# ======================================================================
# VALIDATE DATA
# ======================================================================

def validate_columns(df, candidates, performance):

    print_header("DATA VALIDATION")

    required_df_columns = [
        "Product ID",
        "location_key",
        "lead_time_days",
        "Sales",
        "Gross Profit",
        "Units",
    ]

    required_candidate_columns = [
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

    required_performance_columns = [
        "Product ID",
        "location_key",
        "records",
        "avg_lead_time",
        "median_lead_time",
        "total_sales",
        "total_profit",
    ]

    missing_df = [
        col for col in required_df_columns
        if col not in df.columns
    ]

    missing_candidates = [
        col for col in required_candidate_columns
        if col not in candidates.columns
    ]

    missing_performance = [
        col for col in required_performance_columns
        if col not in performance.columns
    ]

    if missing_df:
        raise ValueError(
            "Featured dataset is missing columns:\n"
            + "\n".join(f"- {c}" for c in missing_df)
        )

    if missing_candidates:
        raise ValueError(
            "Allocation candidate dataset is missing columns:\n"
            + "\n".join(f"- {c}" for c in missing_candidates)
        )

    if missing_performance:
        raise ValueError(
            "Product-location performance dataset is missing columns:\n"
            + "\n".join(f"- {c}" for c in missing_performance)
        )

    print("Featured dataset columns       : OK")
    print("Candidate dataset columns      : OK")
    print("Performance dataset columns    : OK")


# ======================================================================
# CURRENT PRODUCT BASELINE
# ======================================================================

def calculate_product_baselines(df):

    print_header("CURRENT PRODUCT BASELINES")

    baseline = (
        df.groupby("Product ID")
        .agg(
            product_records=("Product ID", "size"),
            current_avg_lead_time=("lead_time_days", "mean"),
            current_median_lead_time=("lead_time_days", "median"),
            current_total_sales=("Sales", "sum"),
            current_total_profit=("Gross Profit", "sum"),
            current_total_units=("Units", "sum"),
        )
        .reset_index()
    )

    return baseline


# ======================================================================
# PREPARE CANDIDATES
# ======================================================================

def prepare_candidates(candidates, baselines):

    print_header("PREPARING ALLOCATION CANDIDATES")

    result = candidates.copy()

    # --------------------------------------------------------------
    # Merge baseline values if candidate file doesn't contain them.
    # --------------------------------------------------------------

    baseline_columns = [
        "Product ID",
        "product_records",
        "current_avg_lead_time",
        "current_median_lead_time",
        "current_total_sales",
        "current_total_profit",
        "current_total_units",
    ]

    available_baseline_columns = [
        col
        for col in baseline_columns
        if col in baselines.columns
    ]

    baseline_small = baselines[available_baseline_columns].copy()

    # Avoid duplicate columns during merge.
    columns_to_add = [
        col
        for col in available_baseline_columns
        if col != "Product ID"
        and col not in result.columns
    ]

    if columns_to_add:
        result = result.merge(
            baseline_small[
                ["Product ID"] + columns_to_add
            ],
            on="Product ID",
            how="left",
        )

    # --------------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------------

    numeric_columns = [
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
        "candidate_total_profit",
        "profit_difference",
        "candidate_score",
    ]

    for col in numeric_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            )

    # --------------------------------------------------------------
    # Calculate missing improvement values
    # --------------------------------------------------------------

    if "lead_time_improvement_days" not in result.columns:
        result["lead_time_improvement_days"] = (
            result["current_avg_lead_time"]
            - result["candidate_avg_lead_time"]
        )

    result["lead_time_improvement_days"] = (
        result["current_avg_lead_time"]
        - result["candidate_avg_lead_time"]
    )

    result["lead_time_improvement_pct"] = np.where(
        result["current_avg_lead_time"] != 0,
        (
            result["lead_time_improvement_days"]
            / result["current_avg_lead_time"]
        ) * 100,
        0.0,
    )

    # --------------------------------------------------------------
    # Candidate records
    # --------------------------------------------------------------

    result["candidate_records"] = (
        result["candidate_records"]
        .fillna(0)
    )

    # --------------------------------------------------------------
    # Historical profit impact
    # --------------------------------------------------------------

    if "current_total_profit" in result.columns:

        result["estimated_profit_change"] = (
            result["candidate_total_profit"]
            - result["current_total_profit"]
        )

    else:

        result["estimated_profit_change"] = (
            result["profit_difference"]
        )

    # --------------------------------------------------------------
    # Estimated improvement per historical record
    # --------------------------------------------------------------

    result["lead_time_improvement_per_record"] = np.where(
        result["candidate_records"] > 0,
        result["lead_time_improvement_days"]
        / result["candidate_records"],
        0.0,
    )

    # --------------------------------------------------------------
    # Risk level
    # --------------------------------------------------------------

    result["risk_level"] = np.select(
        [
            result["candidate_records"] < 20,
            result["candidate_records"].between(20, 49),
            result["candidate_records"].between(50, 99),
            result["candidate_records"] >= 100,
        ],
        [
            "High",
            "Medium",
            "Low",
            "Very Low",
        ],
        default="Unknown",
    )

    return result


# ======================================================================
# IMPACT CALCULATION
# ======================================================================

def calculate_impact(candidates):

    print_header("CALCULATING ALLOCATION IMPACT")

    result = candidates.copy()

    # --------------------------------------------------------------
    # Estimate annual/portfolio impact using historical records.
    #
    # Since the dataset does not provide an explicit factory capacity
    # or future demand forecast, we use historical product volume as
    # the reference scale.
    # --------------------------------------------------------------

    if "current_total_units" in result.columns:

        result["estimated_units_affected"] = (
            result["current_total_units"]
        )

    else:

        result["estimated_units_affected"] = (
            result["candidate_records"]
        )

    # --------------------------------------------------------------
    # Estimated lead-time days saved
    # --------------------------------------------------------------

    result["estimated_lead_time_days_saved"] = (
        result["lead_time_improvement_days"]
        * result["candidate_records"]
    )

    # --------------------------------------------------------------
    # Estimated percentage of product records affected
    # --------------------------------------------------------------

    if "product_records" in result.columns:

        result["record_coverage_pct"] = np.where(
            result["product_records"] > 0,
            (
                result["candidate_records"]
                / result["product_records"]
            ) * 100,
            0.0,
        )

    else:

        result["record_coverage_pct"] = 0.0

    # --------------------------------------------------------------
    # Operational benefit score
    #
    # Combines:
    #   - lead-time improvement
    #   - historical reliability
    #   - candidate score
    # --------------------------------------------------------------

    candidate_score = (
        result["candidate_score"]
        .fillna(0)
    )

    improvement = (
        result["lead_time_improvement_pct"]
        .clip(lower=0)
        .fillna(0)
    )

    records = (
        result["candidate_records"]
        .clip(lower=0)
        .fillna(0)
    )

    reliability = np.log1p(records)

    result["operational_benefit_score"] = (
        improvement
        * (1 + reliability)
    )

    # --------------------------------------------------------------
    # Combined decision score
    # --------------------------------------------------------------

    result["decision_score"] = (
        0.50 * result["operational_benefit_score"]
        + 0.30 * candidate_score
        + 0.20 * result["record_coverage_pct"]
    )

    # --------------------------------------------------------------
    # Recommendation type
    # --------------------------------------------------------------

    result["recommendation_type"] = np.select(
        [
            result["lead_time_improvement_pct"] >= 15,
            result["lead_time_improvement_pct"].between(8, 14.999),
            result["lead_time_improvement_pct"].between(3, 7.999),
            result["lead_time_improvement_pct"] > 0,
        ],
        [
            "Strong Reallocation Candidate",
            "High Potential",
            "Moderate Potential",
            "Low Potential",
        ],
        default="No Improvement",
    )

    return result


# ======================================================================
# FILTER CANDIDATES
# ======================================================================

def filter_candidates(candidates):

    print_header("FILTERING CANDIDATES")

    result = candidates.copy()

    before = len(result)

    # Reliable historical evidence.
    result = result[
        result["candidate_records"]
        >= MIN_CANDIDATE_RECORDS
    ].copy()

    after_records = len(result)

    # Meaningful improvement.
    result = result[
        (
            result["lead_time_improvement_days"]
            >= MIN_IMPROVEMENT_DAYS
        )
        &
        (
            result["lead_time_improvement_pct"]
            >= MIN_IMPROVEMENT_PCT
        )
    ].copy()

    after_improvement = len(result)

    print(f"Initial candidates       : {before}")
    print(f"Reliable candidates      : {after_records}")
    print(f"Meaningful candidates    : {after_improvement}")

    return result


# ======================================================================
# RANK CANDIDATES
# ======================================================================

def rank_candidates(candidates):

    print_header("RANKING ALLOCATION CANDIDATES")

    result = candidates.copy()

    result = result.sort_values(
        by=[
            "decision_score",
            "lead_time_improvement_pct",
            "candidate_records",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    result["recommendation_rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ======================================================================
# LIMIT PER PRODUCT
# ======================================================================

def limit_candidates_per_product(candidates):

    if candidates.empty:
        return candidates

    result = (
        candidates
        .sort_values(
            [
                "Product ID",
                "decision_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "Product ID",
            group_keys=False,
        )
        .head(MAX_CANDIDATES_PER_PRODUCT)
        .copy()
    )

    result = result.sort_values(
        "decision_score",
        ascending=False,
    ).reset_index(drop=True)

    result["recommendation_rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ======================================================================
# PRODUCT IMPACT SUMMARY
# ======================================================================

def create_product_summary(candidates, baselines):

    if candidates.empty:
        return pd.DataFrame()

    summary = (
        candidates
        .groupby("Product ID")
        .agg(
            candidate_count=(
                "candidate_location",
                "count",
            ),
            best_candidate_location=(
                "candidate_location",
                "first",
            ),
            current_avg_lead_time=(
                "current_avg_lead_time",
                "first",
            ),
            best_candidate_lead_time=(
                "candidate_avg_lead_time",
                "min",
            ),
            best_lead_time_improvement_days=(
                "lead_time_improvement_days",
                "max",
            ),
            best_lead_time_improvement_pct=(
                "lead_time_improvement_pct",
                "max",
            ),
            best_candidate_records=(
                "candidate_records",
                "first",
            ),
            best_candidate_score=(
                "decision_score",
                "max",
            ),
        )
        .reset_index()
    )

    summary["product_recommendation"] = np.select(
        [
            summary["best_lead_time_improvement_pct"] >= 15,
            summary["best_lead_time_improvement_pct"].between(
                8,
                14.999,
            ),
            summary["best_lead_time_improvement_pct"].between(
                3,
                7.999,
            ),
            summary["best_lead_time_improvement_pct"] > 0,
        ],
        [
            "Strong Candidate",
            "High Potential",
            "Moderate Potential",
            "Low Potential",
        ],
        default="No Recommendation",
    )

    return summary.sort_values(
        "best_lead_time_improvement_pct",
        ascending=False,
    )


# ======================================================================
# LOCATION IMPACT SUMMARY
# ======================================================================

def create_location_summary(candidates):

    if candidates.empty:
        return pd.DataFrame()

    summary = (
        candidates
        .groupby("candidate_location")
        .agg(
            candidate_count=(
                "Product ID",
                "count",
            ),
            products=(
                "Product ID",
                "nunique",
            ),
            avg_lead_time_improvement_days=(
                "lead_time_improvement_days",
                "mean",
            ),
            avg_lead_time_improvement_pct=(
                "lead_time_improvement_pct",
                "mean",
            ),
            max_lead_time_improvement_pct=(
                "lead_time_improvement_pct",
                "max",
            ),
            total_candidate_records=(
                "candidate_records",
                "sum",
            ),
            avg_decision_score=(
                "decision_score",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "avg_lead_time_improvement_pct",
            "products",
        ],
        ascending=[
            False,
            False,
        ],
    )


# ======================================================================
# PRINT TOP CANDIDATES
# ======================================================================

def print_top_candidates(candidates, n=20):

    print_header("TOP ALLOCATION IMPACT CANDIDATES")

    if candidates.empty:
        print("No qualifying allocation candidates found.")
        return

    columns = [
        "recommendation_rank",
        "Product ID",
        "candidate_location",
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
        "estimated_lead_time_days_saved",
        "record_coverage_pct",
        "decision_score",
        "recommendation_type",
    ]

    available = [
        col
        for col in columns
        if col in candidates.columns
    ]

    display_df = candidates[available].head(n).copy()

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ======================================================================
# PRINT PRODUCT SUMMARY
# ======================================================================

def print_product_summary(summary):

    print_header("PRODUCT IMPACT SUMMARY")

    if summary.empty:
        print("No product recommendations available.")
        return

    columns = [
        "Product ID",
        "candidate_count",
        "best_candidate_location",
        "current_avg_lead_time",
        "best_candidate_lead_time",
        "best_lead_time_improvement_days",
        "best_lead_time_improvement_pct",
        "best_candidate_records",
        "product_recommendation",
    ]

    available = [
        col
        for col in columns
        if col in summary.columns
    ]

    print(
        summary[available]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ======================================================================
# PRINT LOCATION SUMMARY
# ======================================================================

def print_location_summary(summary):

    print_header("LOCATION IMPACT SUMMARY")

    if summary.empty:
        print("No location recommendations available.")
        return

    print(
        summary.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ======================================================================
# OVERALL SUMMARY
# ======================================================================

def print_overall_summary(
    original_candidates,
    filtered_candidates,
    product_summary,
    location_summary,
):

    print_header("ALLOCATION IMPACT SUMMARY")

    print(
        f"Original allocation candidates : "
        f"{len(original_candidates):,}"
    )

    print(
        f"Qualifying candidates          : "
        f"{len(filtered_candidates):,}"
    )

    if filtered_candidates.empty:
        print()
        print("No qualifying candidates were found.")
        return

    best = filtered_candidates.iloc[0]

    print(
        f"Products with recommendations  : "
        f"{filtered_candidates['Product ID'].nunique():,}"
    )

    print(
        f"Locations involved             : "
        f"{filtered_candidates['candidate_location'].nunique():,}"
    )

    print(
        f"Best lead-time improvement      : "
        f"{best['lead_time_improvement_pct']:.2f}%"
    )

    print(
        f"Best improvement in days        : "
        f"{best['lead_time_improvement_days']:.2f} days"
    )

    print(
        f"Average improvement             : "
        f"{filtered_candidates['lead_time_improvement_pct'].mean():.2f}%"
    )

    print(
        f"Average improvement in days    : "
        f"{filtered_candidates['lead_time_improvement_days'].mean():.2f} days"
    )

    print(
        f"Total estimated days saved     : "
        f"{filtered_candidates['estimated_lead_time_days_saved'].sum():.2f}"
    )

    print()
    print("BEST ALLOCATION RECOMMENDATION")
    print("-" * 40)

    print(
        f"Product              : "
        f"{best['Product ID']}"
    )

    print(
        f"Candidate location   : "
        f"{best['candidate_location']}"
    )

    print(
        f"Current lead time    : "
        f"{best['current_avg_lead_time']:.2f} days"
    )

    print(
        f"Candidate lead time  : "
        f"{best['candidate_avg_lead_time']:.2f} days"
    )

    print(
        f"Improvement          : "
        f"{best['lead_time_improvement_days']:.2f} days"
    )

    print(
        f"Improvement          : "
        f"{best['lead_time_improvement_pct']:.2f}%"
    )

    print(
        f"Historical records   : "
        f"{int(best['candidate_records'])}"
    )

    print(
        f"Decision score       : "
        f"{best['decision_score']:.2f}"
    )


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    candidates,
    product_summary,
    location_summary,
):

    print_header("SAVING RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    impact_path = (
        OUTPUT_DIR /
        "allocation_impact_analysis.csv"
    )

    product_path = (
        OUTPUT_DIR /
        "product_impact_summary.csv"
    )

    location_path = (
        OUTPUT_DIR /
        "location_impact_summary.csv"
    )

    candidates.to_csv(
        impact_path,
        index=False,
    )

    product_summary.to_csv(
        product_path,
        index=False,
    )

    location_summary.to_csv(
        location_path,
        index=False,
    )

    print("Allocation impact analysis:")
    print(impact_path)

    print()
    print("Product impact summary:")
    print(product_path)

    print()
    print("Location impact summary:")
    print(location_path)

    return (
        impact_path,
        product_path,
        location_path,
    )


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():

    print_header(
        "ALLOCATION IMPACT ANALYSIS"
    )

    # --------------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------------

    df, candidates, performance = load_data()

    # --------------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------------

    validate_columns(
        df,
        candidates,
        performance,
    )

    # --------------------------------------------------------------
    # 3. Basic dataset information
    # --------------------------------------------------------------

    print_header("DATASET INFORMATION")

    print(
        f"Featured dataset rows       : "
        f"{len(df):,}"
    )

    print(
        f"Unique products             : "
        f"{df['Product ID'].nunique():,}"
    )

    print(
        f"Unique locations            : "
        f"{df['location_key'].nunique():,}"
    )

    print(
        f"Original candidates         : "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------------
    # 4. Current product baselines
    # --------------------------------------------------------------

    baselines = calculate_product_baselines(
        df
    )

    print(
        baselines.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # --------------------------------------------------------------
    # 5. Prepare candidates
    # --------------------------------------------------------------

    prepared_candidates = prepare_candidates(
        candidates,
        baselines,
    )

    # --------------------------------------------------------------
    # 6. Calculate impact
    # --------------------------------------------------------------

    impacted_candidates = calculate_impact(
        prepared_candidates
    )

    # --------------------------------------------------------------
    # 7. Filter
    # --------------------------------------------------------------

    qualifying_candidates = filter_candidates(
        impacted_candidates
    )

    # --------------------------------------------------------------
    # 8. Rank
    # --------------------------------------------------------------

    ranked_candidates = rank_candidates(
        qualifying_candidates
    )

    # --------------------------------------------------------------
    # 9. Keep best candidates per product
    # --------------------------------------------------------------

    final_candidates = limit_candidates_per_product(
        ranked_candidates
    )

    # --------------------------------------------------------------
    # 10. Re-rank after limiting
    # --------------------------------------------------------------

    if not final_candidates.empty:

        final_candidates = final_candidates.sort_values(
            "decision_score",
            ascending=False,
        ).reset_index(drop=True)

        final_candidates["recommendation_rank"] = (
            np.arange(len(final_candidates)) + 1
        )

    # --------------------------------------------------------------
    # 11. Product summary
    # --------------------------------------------------------------

    product_summary = create_product_summary(
        final_candidates,
        baselines,
    )

    # --------------------------------------------------------------
    # 12. Location summary
    # --------------------------------------------------------------

    location_summary = create_location_summary(
        final_candidates
    )

    # --------------------------------------------------------------
    # 13. Display results
    # --------------------------------------------------------------

    print_top_candidates(
        final_candidates,
        n=20,
    )

    print_product_summary(
        product_summary
    )

    print_location_summary(
        location_summary
    )

    # --------------------------------------------------------------
    # 14. Overall summary
    # --------------------------------------------------------------

    print_overall_summary(
        candidates,
        final_candidates,
        product_summary,
        location_summary,
    )

    # --------------------------------------------------------------
    # 15. Save
    # --------------------------------------------------------------

    save_results(
        final_candidates,
        product_summary,
        location_summary,
    )

    # --------------------------------------------------------------
    # Complete
    # --------------------------------------------------------------

    print_header(
        "ALLOCATION IMPACT ANALYSIS COMPLETED"
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()