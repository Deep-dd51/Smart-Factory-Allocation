"""
Test the saved lead-time prediction service.
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
# IMPORT
# -------------------------------------------------------------------

from src.prediction.predictor import (
    load_model,
    predict_lead_time,
)


# -------------------------------------------------------------------
# DATA
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
print("PREDICTION SERVICE TEST")
print("=" * 70)

print("\nLoading featured dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# -------------------------------------------------------------------
# SAME FEATURES USED DURING TRAINING
# -------------------------------------------------------------------

TARGET = "lead_time_days"

excluded_columns = [
    TARGET,
    "lead_time_category",
    "profit_per_shipping_day",
    "Ship Date",
    "Original Ship Date",
    "Row ID",
    "Order ID",
    "Customer ID",
    "Order Date",
    "Product Name",
    "Postal Code",
]


feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


X = df[
    feature_columns
].copy()


# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------

print("\nLoading saved model...")

model = load_model()

print(
    "Saved model loaded successfully!"
)


# -------------------------------------------------------------------
# PREDICT SAMPLE
# -------------------------------------------------------------------

print("\nGenerating predictions...")

sample = X.head(
    10
)

predictions = predict_lead_time(
    model,
    sample,
)


# -------------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------------

results = pd.DataFrame(
    {
        "Predicted Lead Time": predictions
    }
)

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    f"Minimum prediction: "
    f"{predictions.min():.4f} days"
)

print(
    f"Maximum prediction: "
    f"{predictions.max():.4f} days"
)

print(
    f"Average prediction: "
    f"{predictions.mean():.4f} days"
)


# -------------------------------------------------------------------
# COMPLETION
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION SERVICE TEST COMPLETED")
print("=" * 70)