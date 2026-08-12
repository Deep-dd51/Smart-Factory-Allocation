from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

# app/app.py -> Smart-Factory-Allocation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECOMMENDATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_allocation"
    / "recommendations"
    / "final_recommendations.csv"
)

VISUALIZATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "factory_allocation"
    / "visualizations"
)

VISUALIZATION_DATASET = (
    VISUALIZATION_DIR / "visualization_dataset.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Factory Allocation",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .dashboard-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .dashboard-subtitle {
            font-size: 1.05rem;
            color: #666666;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 650;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .recommendation-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #dddddd;
            background-color: #fafafa;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_recommendations():
    """Load final recommendation dataset."""

    if not RECOMMENDATIONS_PATH.exists():
        return None

    try:
        return pd.read_csv(RECOMMENDATIONS_PATH)
    except Exception as e:
        st.error(f"Could not read recommendations file: {e}")
        return None


@st.cache_data
def load_visualization_dataset():
    """Load visualization dataset if available."""

    if not VISUALIZATION_DATASET.exists():
        return None

    try:
        return pd.read_csv(VISUALIZATION_DATASET)
    except Exception:
        return None


def safe_numeric(df, column):
    """Convert a dataframe column to numeric safely."""

    if column not in df.columns:
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# LOAD DATA
# ============================================================

recommendations = load_recommendations()
visualization_data = load_visualization_dataset()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">🏭 Smart Factory Allocation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-subtitle">
        Factory Reallocation & Shipping Optimization Recommendation Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Dashboard")

    page = st.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Recommendations",
            "Lead Time Analysis",
            "Impact Analysis",
            "Data Explorer",
        ],
    )

    st.divider()

    st.subheader("Data Status")

    if recommendations is not None:
        st.success("Recommendations loaded")
    else:
        st.error("Recommendations not found")

    if visualization_data is not None:
        st.success("Visualization data loaded")
    else:
        st.warning("Visualization dataset not found")

    st.divider()

    st.caption("Project Root")

    st.code(
        str(PROJECT_ROOT),
        language="text",
    )


# ============================================================
# DATA NOT FOUND
# ============================================================

if recommendations is None:

    st.error(
        "Final recommendations dataset was not found."
    )

    st.markdown("### Expected file location")

    st.code(
        str(RECOMMENDATIONS_PATH),
        language="text",
    )

    st.markdown("### Expected project structure")

    st.code(
        """
Smart-Factory-Allocation/
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       └── factory_allocation/
│           ├── recommendations/
│           │   └── final_recommendations.csv
│           │
│           └── visualizations/
│               ├── visualization_dataset.csv
│               ├── 01_lead_time_comparison.png
│               ├── 02_improvement_percentage.png
│               ├── 03_decision_score.png
│               ├── 04_recommendation_priority.png
│               ├── 05_location_impact.png
│               ├── 06_product_impact.png
│               ├── 07_days_saved.png
│               └── 08_executive_dashboard.png
        """,
        language="text",
    )

    st.stop()


# ============================================================
# NORMALIZE IMPORTANT COLUMNS
# ============================================================

df = recommendations.copy()


# ------------------------------------------------------------
# Recommended lead time
# ------------------------------------------------------------

if "recommended_avg_lead_time" not in df.columns:

    if "candidate_avg_lead_time" in df.columns:

        df["recommended_avg_lead_time"] = safe_numeric(
            df,
            "candidate_avg_lead_time",
        )

    else:

        df["recommended_avg_lead_time"] = 0.0


# ------------------------------------------------------------
# Current lead time
# ------------------------------------------------------------

if "current_avg_lead_time" not in df.columns:

    df["current_avg_lead_time"] = 0.0

else:

    df["current_avg_lead_time"] = safe_numeric(
        df,
        "current_avg_lead_time",
    )


# ------------------------------------------------------------
# Lead-time improvement
# ------------------------------------------------------------

if "lead_time_improvement_days" not in df.columns:

    df["lead_time_improvement_days"] = (
        safe_numeric(
            df,
            "current_avg_lead_time",
        )
        - safe_numeric(
            df,
            "recommended_avg_lead_time",
        )
    )


# ------------------------------------------------------------
# Lead-time improvement percentage
# ------------------------------------------------------------

if "lead_time_improvement_pct" not in df.columns:

    current = safe_numeric(
        df,
        "current_avg_lead_time",
    )

    improvement = safe_numeric(
        df,
        "lead_time_improvement_days",
    )

    df["lead_time_improvement_pct"] = 0.0

    valid_current = current > 0

    df.loc[valid_current, "lead_time_improvement_pct"] = (
        improvement[valid_current]
        / current[valid_current]
        * 100
    )


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True,
    )

    total_recommendations = len(df)

    products = (
        df["product_id"].nunique()
        if "product_id" in df.columns
        else 0
    )

    locations = (
        df["candidate_location"].nunique()
        if "candidate_location" in df.columns
        else 0
    )

    avg_improvement = safe_numeric(
        df,
        "lead_time_improvement_pct",
    ).mean()

    avg_days_reduction = safe_numeric(
        df,
        "lead_time_improvement_days",
    ).mean()

    total_days_saved = safe_numeric(
        df,
        "estimated_lead_time_days_saved",
    ).sum()

    if total_days_saved == 0:

        total_days_saved = safe_numeric(
            df,
            "estimated_days_saved",
        ).sum()


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Recommendations",
            f"{total_recommendations}",
        )

    with col2:

        st.metric(
            "Products",
            f"{products}",
        )

    with col3:

        st.metric(
            "Locations",
            f"{locations}",
        )

    with col4:

        st.metric(
            "Avg Improvement",
            f"{avg_improvement:.2f}%",
        )

    with col5:

        st.metric(
            "Avg Days Reduced",
            f"{avg_days_reduction:.2f}",
        )


    st.divider()


    # --------------------------------------------------------
    # TOTAL DAYS SAVED
    # --------------------------------------------------------

    st.subheader("Estimated Operational Impact")

    impact_col1, impact_col2 = st.columns(2)

    with impact_col1:

        st.metric(
            "Estimated Total Days Saved",
            f"{total_days_saved:.2f}",
        )

    with impact_col2:

        max_improvement = safe_numeric(
            df,
            "lead_time_improvement_pct",
        ).max()

        max_reduction = safe_numeric(
            df,
            "lead_time_improvement_days",
        ).max()

        st.metric(
            "Maximum Improvement",
            f"{max_improvement:.2f}% / {max_reduction:.2f} days",
        )


    # --------------------------------------------------------
    # BEST RECOMMENDATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Best Recommendation</div>',
        unsafe_allow_html=True,
    )

    if len(df) > 0:

        best_index = (
            safe_numeric(
                df,
                "lead_time_improvement_pct",
            ).idxmax()
        )

        best = df.loc[best_index]

        product = best.get(
            "product_id",
            "N/A",
        )

        location = best.get(
            "candidate_location",
            "N/A",
        )

        current_lt = float(
            pd.to_numeric(
                best.get(
                    "current_avg_lead_time",
                    0,
                ),
                errors="coerce",
            ) or 0
        )

        recommended_lt = float(
            pd.to_numeric(
                best.get(
                    "recommended_avg_lead_time",
                    0,
                ),
                errors="coerce",
            ) or 0
        )

        improvement_days = float(
            pd.to_numeric(
                best.get(
                    "lead_time_improvement_days",
                    0,
                ),
                errors="coerce",
            ) or 0
        )

        improvement_pct = float(
            pd.to_numeric(
                best.get(
                    "lead_time_improvement_pct",
                    0,
                ),
                errors="coerce",
            ) or 0
        )

        priority = best.get(
            "priority",
            "N/A",
        )

        confidence = best.get(
            "confidence",
            "N/A",
        )

        action = best.get(
            "recommended_action",
            "N/A",
        )

        st.success(
            f"""
**Product:** {product}

**Recommended Location:** {location}

**Current Lead Time:** {current_lt:.2f} days

**Recommended Lead Time:** {recommended_lt:.2f} days

**Improvement:** {improvement_days:.2f} days ({improvement_pct:.2f}%)

**Priority:** {priority}

**Confidence:** {confidence}

**Recommended Action:** {action}
"""
        )


    # --------------------------------------------------------
    # EXECUTIVE DASHBOARD IMAGE
    # --------------------------------------------------------

    executive_image = (
        VISUALIZATION_DIR
        / "08_executive_dashboard.png"
    )

    if executive_image.exists():

        st.markdown(
            '<div class="section-title">Executive Visualization</div>',
            unsafe_allow_html=True,
        )

        st.image(
            str(executive_image),
            use_container_width=True,
        )


# ============================================================
# RECOMMENDATIONS PAGE
# ============================================================

elif page == "Recommendations":

    st.markdown(
        '<div class="section-title">Factory Reallocation Recommendations</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "product_id",
        "candidate_location",
        "current_avg_lead_time",
        "recommended_avg_lead_time",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "decision_score",
        "priority",
        "confidence",
        "recommended_action",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    display_df = df[available_columns].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Recommendations CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="final_recommendations.csv",
        mime="text/csv",
    )


# ============================================================
# LEAD TIME ANALYSIS
# ============================================================

elif page == "Lead Time Analysis":

    st.markdown(
        '<div class="section-title">Lead Time Analysis</div>',
        unsafe_allow_html=True,
    )

    chart_columns = [
        "product_id",
        "current_avg_lead_time",
        "recommended_avg_lead_time",
    ]

    available = [
        column
        for column in chart_columns
        if column in df.columns
    ]

    if len(available) >= 3:

        chart_df = df[available].copy()

        chart_df = chart_df.set_index(
            "product_id"
        )

        st.bar_chart(
            chart_df[
                [
                    "current_avg_lead_time",
                    "recommended_avg_lead_time",
                ]
            ]
        )

    else:

        st.warning(
            "Required lead-time columns are not available."
        )


    lead_time_image = (
        VISUALIZATION_DIR
        / "01_lead_time_comparison.png"
    )

    if lead_time_image.exists():

        st.subheader("Lead Time Comparison")

        st.image(
            str(lead_time_image),
            use_container_width=True,
        )


    improvement_image = (
        VISUALIZATION_DIR
        / "02_improvement_percentage.png"
    )

    if improvement_image.exists():

        st.subheader("Improvement Percentage")

        st.image(
            str(improvement_image),
            use_container_width=True,
        )


# ============================================================
# IMPACT ANALYSIS
# ============================================================

elif page == "Impact Analysis":

    st.markdown(
        '<div class="section-title">Operational Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    impact_columns = [
        "product_id",
        "candidate_location",
        "lead_time_improvement_days",
        "lead_time_improvement_pct",
        "estimated_lead_time_days_saved",
        "estimated_profit_change",
        "decision_score",
        "risk_level",
    ]

    available = [
        column
        for column in impact_columns
        if column in df.columns
    ]

    if available:

        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # Decision Score
    # --------------------------------------------------------

    decision_image = (
        VISUALIZATION_DIR
        / "03_decision_score.png"
    )

    if decision_image.exists():

        st.subheader("Decision Score")

        st.image(
            str(decision_image),
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    priority_image = (
        VISUALIZATION_DIR
        / "04_recommendation_priority.png"
    )

    if priority_image.exists():

        st.subheader("Recommendation Priority")

        st.image(
            str(priority_image),
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Location Impact
    # --------------------------------------------------------

    location_image = (
        VISUALIZATION_DIR
        / "05_location_impact.png"
    )

    if location_image.exists():

        st.subheader("Location Impact")

        st.image(
            str(location_image),
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Product Impact
    # --------------------------------------------------------

    product_image = (
        VISUALIZATION_DIR
        / "06_product_impact.png"
    )

    if product_image.exists():

        st.subheader("Product Impact")

        st.image(
            str(product_image),
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Days Saved
    # --------------------------------------------------------

    days_image = (
        VISUALIZATION_DIR
        / "07_days_saved.png"
    )

    if days_image.exists():

        st.subheader("Estimated Days Saved")

        st.image(
            str(days_image),
            use_container_width=True,
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="section-title">Data Explorer</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**"
    )

    st.write("Available columns:")

    st.code(
        "\n".join(df.columns.tolist()),
        language="text",
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Full Recommendation Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="final_recommendations.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Factory Allocation • Factory Reallocation & Shipping Optimization"
)