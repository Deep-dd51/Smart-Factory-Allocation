"""
22_Recommendation_Decision_Report.py

FACTORY REALLOCATION & SHIPPING OPTIMIZATION
Recommendation & Decision Report

Purpose:
    Convert Step 21 optimization results into management-ready
    recommendations and decision-support outputs.

Input:
    data/processed/factory_allocation/allocation_optimization/
        final_allocation_recommendations.csv

Outputs:
    data/processed/factory_allocation/recommendations/
        final_recommendations.csv
        product_recommendations.csv
        location_recommendations.csv
        executive_summary.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACTORY_ALLOCATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_allocation"
)

OPTIMIZATION_DIR = (
    FACTORY_ALLOCATION_DIR
    / "allocation_optimization"
)

OUTPUT_DIR = (
    FACTORY_ALLOCATION_DIR
    / "recommendations"
)

FINAL_RECOMMENDATIONS_PATH = (
    OPTIMIZATION_DIR
    / "final_allocation_recommendations.csv"
)

# Possible fallback locations in case the project structure differs
FALLBACK_INPUTS = [
    FACTORY_ALLOCATION_DIR / "allocation_optimization.csv",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_allocation"
    / "allocation_impact"
    / "allocation_impact_analysis.csv",
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

WIDTH = 70


def print_header(title):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def print_subheader(title):
    print()
    print("-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def safe_round(value, digits=2):
    try:
        if pd.isna(value):
            return 0.0
        return round(float(value), digits)
    except Exception:
        return 0.0


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_input_file():
    """
    Locate Step 21 final recommendation file.
    """

    candidates = [
        FINAL_RECOMMENDATIONS_PATH,
        *FALLBACK_INPUTS,
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    # Last-resort recursive search
    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT,
    ]

    possible_names = {
        "final_allocation_recommendations.csv",
        "allocation_optimization.csv",
        "allocation_impact_analysis.csv",
    }

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            if path.name in possible_names:
                return path

    raise FileNotFoundError(
        "\nCould not find Step 21 optimization output.\n"
        "\nExpected:\n"
        f"{FINAL_RECOMMENDATIONS_PATH}\n"
        "\nPlease run:\n"
        "python notebooks/21_Allocation_Optimization.py"
    )


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):
    """
    Normalize column names so the script is tolerant of
    small naming differences between pipeline versions.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    aliases = {
        "product": "product_id",
        "productid": "product_id",
        "product_code": "product_id",

        "location": "candidate_location",
        "recommended_location": "candidate_location",
        "candidate_factory": "candidate_location",

        "current_lead_time": "current_avg_lead_time",
        "optimized_lead_time": "candidate_avg_lead_time",
        "candidate_lead_time": "candidate_avg_lead_time",

        "improvement_days": "lead_time_improvement_days",
        "lead_time_saved_days": "lead_time_improvement_days",

        "improvement_percentage": "lead_time_improvement_pct",
        "improvement_percent": "lead_time_improvement_pct",

        "records": "candidate_records",

        "score": "optimization_score",
        "decision_score": "optimization_score",

        "recommendation": "recommendation_type",
    }

    for old_name, new_name in aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_optimization_data():

    print_header("RECOMMENDATION & DECISION REPORT")

    print("\nLoading Step 21 optimization data...")

    input_path = find_input_file()

    print(f"\nInput dataset:")
    print(input_path)

    df = pd.read_csv(input_path)

    df = normalize_columns(df)

    print(f"\nDataset shape: {df.shape}")

    return df, input_path


# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):

    print_header("DATA VALIDATION")

    required_columns = [
        "product_id",
        "candidate_location",
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print("\nMissing required columns:")

        for col in missing:
            print(f" - {col}")

        print("\nAvailable columns:")

        for col in df.columns:
            print(f" - {col}")

        raise ValueError(
            "\nThe Step 21 output does not contain the required "
            "optimization columns."
        )

    print("Required columns : OK")

    # Numeric conversion
    numeric_columns = [
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
    ]

    if "optimization_score" in df.columns:
        numeric_columns.append("optimization_score")

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid rows
    before = len(df)

    df = df.dropna(
        subset=[
            "product_id",
            "candidate_location",
            "current_avg_lead_time",
            "candidate_avg_lead_time",
            "lead_time_improvement_days",
            "lead_time_improvement_pct",
        ]
    ).copy()

    removed = before - len(df)

    if removed > 0:
        print(f"Invalid rows removed: {removed}")

    print(f"Valid rows          : {len(df)}")

    return df


# ============================================================
# RECOMMENDATION PRIORITY
# ============================================================

def assign_priority(row):

    improvement = float(
        row["lead_time_improvement_pct"]
    )

    records = float(
        row["candidate_records"]
    )

    score = float(
        row.get("optimization_score", 0)
    )

    # Strong recommendation
    if (
        improvement >= 15
        and records >= 20
        and score >= 60
    ):
        return "STRONG PRIORITY"

    # High priority
    if (
        improvement >= 10
        and records >= 20
        and score >= 35
    ):
        return "HIGH PRIORITY"

    # Moderate priority
    if (
        improvement >= 5
        and records >= 20
    ):
        return "MODERATE PRIORITY"

    # Low priority
    return "LOW PRIORITY"


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def assign_confidence(row):

    records = float(
        row["candidate_records"]
    )

    improvement = float(
        row["lead_time_improvement_pct"]
    )

    if records >= 100 and improvement >= 5:
        return "HIGH"

    if records >= 50 and improvement >= 5:
        return "MEDIUM-HIGH"

    if records >= 20:
        return "MEDIUM"

    return "LOW"


# ============================================================
# ACTION
# ============================================================

def assign_action(row):

    priority = row["priority"]
    confidence = row["confidence"]

    if priority == "STRONG PRIORITY":
        return "Pilot Reallocation"

    if priority == "HIGH PRIORITY":
        return "Prioritize for Operational Review"

    if priority == "MODERATE PRIORITY":
        return "Evaluate as Secondary Option"

    if confidence == "LOW":
        return "Collect More Evidence"

    return "Monitor"


# ============================================================
# BUSINESS IMPACT
# ============================================================

def calculate_business_impact(row):

    improvement = float(
        row["lead_time_improvement_days"]
    )

    records = float(
        row["candidate_records"]
    )

    return improvement * records


# ============================================================
# CREATE FINAL RECOMMENDATIONS
# ============================================================

def create_recommendations(df):

    print_header("CREATING MANAGEMENT RECOMMENDATIONS")

    result = df.copy()

    # --------------------------------------------------------
    # Recommendation priority
    # --------------------------------------------------------

    result["priority"] = result.apply(
        assign_priority,
        axis=1
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    result["confidence"] = result.apply(
        assign_confidence,
        axis=1
    )

    # --------------------------------------------------------
    # Recommended action
    # --------------------------------------------------------

    result["recommended_action"] = result.apply(
        assign_action,
        axis=1
    )

    # --------------------------------------------------------
    # Estimated historical lead-time savings
    # --------------------------------------------------------

    result["estimated_days_saved"] = (
        result["lead_time_improvement_days"]
        * result["candidate_records"]
    )

    # --------------------------------------------------------
    # Business impact proxy
    # --------------------------------------------------------

    result["historical_impact_score"] = result.apply(
        calculate_business_impact,
        axis=1
    )

    # --------------------------------------------------------
    # Recommendation explanation
    # --------------------------------------------------------

    def explanation(row):

        return (
            f"Reallocate {row['product_id']} toward "
            f"{row['candidate_location']} based on an estimated "
            f"{row['lead_time_improvement_days']:.2f}-day reduction "
            f"({row['lead_time_improvement_pct']:.2f}%) in historical "
            f"average lead time, supported by "
            f"{int(row['candidate_records'])} historical records."
        )

    result["decision_explanation"] = result.apply(
        explanation,
        axis=1
    )

    # --------------------------------------------------------
    # Sort by priority and score
    # --------------------------------------------------------

    priority_order = {
        "STRONG PRIORITY": 1,
        "HIGH PRIORITY": 2,
        "MODERATE PRIORITY": 3,
        "LOW PRIORITY": 4,
    }

    result["_priority_order"] = (
        result["priority"]
        .map(priority_order)
        .fillna(99)
    )

    score_column = (
        "optimization_score"
        if "optimization_score" in result.columns
        else "lead_time_improvement_pct"
    )

    result = result.sort_values(
        [
            "_priority_order",
            score_column,
            "lead_time_improvement_pct",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    result["recommendation_rank"] = (
        range(1, len(result) + 1)
    )

    result = result.drop(
        columns=["_priority_order"]
    )

    return result


# ============================================================
# PRODUCT SUMMARY
# ============================================================

def create_product_summary(df):

    print_header("PRODUCT RECOMMENDATION SUMMARY")

    rows = []

    for product_id, group in df.groupby(
        "product_id",
        sort=False
    ):

        group = group.sort_values(
            "lead_time_improvement_pct",
            ascending=False
        )

        best = group.iloc[0]

        rows.append({
            "product_id": product_id,
            "candidate_count": len(group),
            "best_location": best["candidate_location"],
            "current_avg_lead_time": best[
                "current_avg_lead_time"
            ],
            "recommended_avg_lead_time": best[
                "candidate_avg_lead_time"
            ],
            "lead_time_improvement_days": best[
                "lead_time_improvement_days"
            ],
            "lead_time_improvement_pct": best[
                "lead_time_improvement_pct"
            ],
            "candidate_records": best[
                "candidate_records"
            ],
            "priority": best["priority"],
            "confidence": best["confidence"],
            "recommended_action": best[
                "recommended_action"
            ],
            "optimization_score": best.get(
                "optimization_score",
                np.nan
            ),
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "lead_time_improvement_pct",
        ascending=False
    )

    return summary


# ============================================================
# LOCATION SUMMARY
# ============================================================

def create_location_summary(df):

    print_header("LOCATION RECOMMENDATION SUMMARY")

    summary = (
        df.groupby("candidate_location")
        .agg(
            candidate_count=(
                "product_id",
                "count"
            ),
            products=(
                "product_id",
                "nunique"
            ),
            avg_lead_time_improvement_days=(
                "lead_time_improvement_days",
                "mean"
            ),
            avg_lead_time_improvement_pct=(
                "lead_time_improvement_pct",
                "mean"
            ),
            max_lead_time_improvement_pct=(
                "lead_time_improvement_pct",
                "max"
            ),
            total_candidate_records=(
                "candidate_records",
                "sum"
            ),
            total_estimated_days_saved=(
                "estimated_days_saved",
                "sum"
            ),
        )
        .reset_index()
    )

    if "optimization_score" in df.columns:

        score_summary = (
            df.groupby("candidate_location")
            ["optimization_score"]
            .mean()
            .reset_index(
                name="avg_optimization_score"
            )
        )

        summary = summary.merge(
            score_summary,
            on="candidate_location",
            how="left"
        )

    summary = summary.sort_values(
        "avg_lead_time_improvement_pct",
        ascending=False
    )

    return summary


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def create_executive_summary(
    recommendations,
    product_summary,
    location_summary
):

    print_header("EXECUTIVE SUMMARY")

    total_candidates = len(recommendations)

    products = recommendations[
        "product_id"
    ].nunique()

    locations = recommendations[
        "candidate_location"
    ].nunique()

    strong = (
        recommendations["priority"]
        == "STRONG PRIORITY"
    ).sum()

    high = (
        recommendations["priority"]
        == "HIGH PRIORITY"
    ).sum()

    avg_improvement = (
        recommendations[
            "lead_time_improvement_pct"
        ].mean()
    )

    avg_days = (
        recommendations[
            "lead_time_improvement_days"
        ].mean()
    )

    total_days_saved = (
        recommendations[
            "estimated_days_saved"
        ].sum()
    )

    best = recommendations.iloc[0]

    text = f"""
FACTORY REALLOCATION & SHIPPING OPTIMIZATION
EXECUTIVE DECISION SUMMARY
============================================================

OVERVIEW
------------------------------------------------------------
Optimization candidates evaluated : {total_candidates}
Products with recommendations     : {products}
Locations involved                : {locations}

RECOMMENDATION PRIORITY
------------------------------------------------------------
Strong priority candidates        : {strong}
High priority candidates          : {high}

AVERAGE IMPACT
------------------------------------------------------------
Average lead-time improvement      : {avg_improvement:.2f}%
Average lead-time reduction        : {avg_days:.2f} days
Estimated historical days saved    : {total_days_saved:.2f}

BEST RECOMMENDATION
------------------------------------------------------------
Product                            : {best['product_id']}
Recommended location               : {best['candidate_location']}
Current lead time                  : {best['current_avg_lead_time']:.2f} days
Recommended lead time              : {best['candidate_avg_lead_time']:.2f} days
Lead-time reduction                : {best['lead_time_improvement_days']:.2f} days
Percentage improvement              : {best['lead_time_improvement_pct']:.2f}%
Historical records                  : {int(best['candidate_records'])}
Priority                            : {best['priority']}
Confidence                          : {best['confidence']}
Action                              : {best['recommended_action']}

MANAGEMENT INTERPRETATION
------------------------------------------------------------
The optimization identifies product-location combinations
where historical shipping performance suggests a potential
reduction in shipping lead time.

The recommendations should be treated as decision-support
signals rather than guaranteed future outcomes. Historical
performance, operational capacity, factory constraints,
inventory availability, and implementation costs should be
validated before executing a physical reallocation.

RECOMMENDED NEXT STEP
------------------------------------------------------------
Prioritize the strongest recommendations for operational
pilot testing and validate the expected lead-time improvement
against actual factory capacity and shipping constraints.

============================================================
END OF EXECUTIVE SUMMARY
============================================================
"""

    print(text)

    return text


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    recommendations,
    product_summary,
    location_summary,
    executive_summary
):

    print_header("SAVING RECOMMENDATION OUTPUTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Final recommendations
    # --------------------------------------------------------

    final_path = (
        OUTPUT_DIR
        / "final_recommendations.csv"
    )

    recommendations.to_csv(
        final_path,
        index=False
    )

    # --------------------------------------------------------
    # Product recommendations
    # --------------------------------------------------------

    product_path = (
        OUTPUT_DIR
        / "product_recommendations.csv"
    )

    product_summary.to_csv(
        product_path,
        index=False
    )

    # --------------------------------------------------------
    # Location recommendations
    # --------------------------------------------------------

    location_path = (
        OUTPUT_DIR
        / "location_recommendations.csv"
    )

    location_summary.to_csv(
        location_path,
        index=False
    )

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "executive_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            executive_summary.strip()
            + "\n"
        )

    print("\nFiles created:")

    print(
        f"\nFinal recommendations:\n"
        f"{final_path}"
    )

    print(
        f"\nProduct recommendations:\n"
        f"{product_path}"
    )

    print(
        f"\nLocation recommendations:\n"
        f"{location_path}"
    )

    print(
        f"\nExecutive summary:\n"
        f"{summary_path}"
    )

    return (
        final_path,
        product_path,
        location_path,
        summary_path,
    )


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

def display_results(
    recommendations,
    product_summary,
    location_summary
):

    print_header("TOP MANAGEMENT RECOMMENDATIONS")

    display_columns = [
        "recommendation_rank",
        "product_id",
        "candidate_location",
        "current_avg_lead_time",
        "candidate_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
        "priority",
        "confidence",
        "recommended_action",
    ]

    available_columns = [
        col
        for col in display_columns
        if col in recommendations.columns
    ]

    print(
        recommendations[
            available_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    print_header("PRODUCT-LEVEL RECOMMENDATIONS")

    product_columns = [
        "product_id",
        "best_location",
        "current_avg_lead_time",
        "recommended_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "candidate_records",
        "priority",
        "confidence",
        "recommended_action",
    ]

    available_product_columns = [
        col
        for col in product_columns
        if col in product_summary.columns
    ]

    print(
        product_summary[
            available_product_columns
        ]
        .to_string(index=False)
    )

    print_header("LOCATION-LEVEL SUMMARY")

    print(
        location_summary
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

def display_final_statistics(df):

    print_header("FINAL DECISION STATISTICS")

    print(
        f"Recommendations evaluated : {len(df)}"
    )

    print(
        f"Products involved         : "
        f"{df['product_id'].nunique()}"
    )

    print(
        f"Locations involved        : "
        f"{df['candidate_location'].nunique()}"
    )

    print(
        f"Average improvement       : "
        f"{df['lead_time_improvement_pct'].mean():.2f}%"
    )

    print(
        f"Average improvement days   : "
        f"{df['lead_time_improvement_days'].mean():.2f}"
    )

    print(
        f"Maximum improvement       : "
        f"{df['lead_time_improvement_pct'].max():.2f}%"
    )

    print(
        f"Total estimated days saved: "
        f"{df['estimated_days_saved'].sum():.2f}"
    )

    print("\nPriority distribution:")

    print(
        df["priority"]
        .value_counts()
        .to_string()
    )

    best = df.iloc[0]

    print_header("BEST MANAGEMENT RECOMMENDATION")

    print(
        f"Product              : "
        f"{best['product_id']}"
    )

    print(
        f"Recommended location : "
        f"{best['candidate_location']}"
    )

    print(
        f"Current lead time    : "
        f"{best['current_avg_lead_time']:.2f} days"
    )

    print(
        f"Expected lead time   : "
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
        f"Priority             : "
        f"{best['priority']}"
    )

    print(
        f"Confidence           : "
        f"{best['confidence']}"
    )

    print(
        f"Recommended action   : "
        f"{best['recommended_action']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, input_path = load_optimization_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    df = validate_data(df)

    # --------------------------------------------------------
    # Create recommendations
    # --------------------------------------------------------

    recommendations = create_recommendations(df)

    # --------------------------------------------------------
    # Product summary
    # --------------------------------------------------------

    product_summary = create_product_summary(
        recommendations
    )

    # --------------------------------------------------------
    # Location summary
    # --------------------------------------------------------

    location_summary = create_location_summary(
        recommendations
    )

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    executive_summary = create_executive_summary(
        recommendations,
        product_summary,
        location_summary
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        recommendations,
        product_summary,
        location_summary
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    display_final_statistics(
        recommendations
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        recommendations,
        product_summary,
        location_summary,
        executive_summary
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print_header(
        "RECOMMENDATION & DECISION REPORT COMPLETED"
    )

    print(
        "\nStep 22 completed successfully."
    )

    print(
        "\nNext step:"
    )

    print(
        "python notebooks/23_Allocation_Visualization.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()