"""
Gradient Boosting Lead-Time Prediction Model
--------------------------------------------

Model:
    Gradient Boosting Regressor

Target:
    lead_time_days
"""

import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# File
# -------------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

TARGET = "lead_time_days"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

print("=" * 70)
print("GRADIENT BOOSTING LEAD-TIME MODEL")
print("=" * 70)

print("\nLoading featured dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# -------------------------------------------------------------------
# Remove target and leakage columns
# -------------------------------------------------------------------

excluded_columns = [
    TARGET,

    # Target-derived
    "lead_time_category",
    "profit_per_shipping_day",

    # Outcome information
    "Ship Date",
    "Original Ship Date",

    # Identifiers
    "Row ID",
    "Order ID",
    "Customer ID",

    # Redundant date
    "Order Date",

    # High-cardinality / redundant
    "Product Name",
    "Postal Code",
]


feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


X = df[feature_columns].copy()

y = df[TARGET].copy()


# -------------------------------------------------------------------
# Feature types
# -------------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32",
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "string",
        "category",
    ]
).columns.tolist()


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    f"Total features      : {len(X.columns)}"
)

print(
    f"Numerical features  : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# -------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)


# -------------------------------------------------------------------
# Numerical preprocessing
# -------------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)


# -------------------------------------------------------------------
# Categorical preprocessing
# -------------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        )
    ]
)


# -------------------------------------------------------------------
# Preprocessor
# -------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        )
    ]
)


# -------------------------------------------------------------------
# Gradient Boosting
# -------------------------------------------------------------------

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
)


# -------------------------------------------------------------------
# Complete pipeline
# -------------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING GRADIENT BOOSTING")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train,
)

print(
    "Gradient Boosting training completed."
)


# -------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred,
)

rmse = mean_squared_error(
    y_test,
    y_pred,
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred,
)


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("GRADIENT BOOSTING RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} days"
)

print(
    f"RMSE : {rmse:.4f} days"
)

print(
    f"R²   : {r2:.4f}"
)


# -------------------------------------------------------------------
# Prediction range
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION RANGE")
print("=" * 70)

print(
    f"Minimum prediction: "
    f"{y_pred.min():.4f}"
)

print(
    f"Maximum prediction: "
    f"{y_pred.max():.4f}"
)


# -------------------------------------------------------------------
# Sample predictions
# -------------------------------------------------------------------

results = pd.DataFrame(
    {
        "Actual Lead Time": y_test.values,
        "Predicted Lead Time": y_pred,
    }
)


print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print(
    results
    .head(15)
    .to_string(index=False)
)


# -------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("GRADIENT BOOSTING MODEL COMPLETED")
print("=" * 70)