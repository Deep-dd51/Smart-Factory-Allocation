"""
Train and save the final Random Forest model.

Final selected model:
    Random Forest Regressor

Target:
    lead_time_days
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest_lead_time.pkl"
)


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

TARGET = "lead_time_days"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# -------------------------------------------------------------------
# CREATE MODEL DIRECTORY
# -------------------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------

print("=" * 70)
print("FINAL RANDOM FOREST MODEL")
print("=" * 70)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

print("\nLoading featured dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# -------------------------------------------------------------------
# REMOVE TARGET / LEAKAGE
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


X = df[
    feature_columns
].copy()

y = df[
    TARGET
].copy()


# -------------------------------------------------------------------
# FEATURE TYPES
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
    f"Total features      : "
    f"{len(X.columns)}"
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
# TRAIN / TEST SPLIT
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
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
# NUMERICAL PIPELINE
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
# CATEGORICAL PIPELINE
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
        ),
    ]
)


# -------------------------------------------------------------------
# PREPROCESSOR
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
        ),
    ]
)


# -------------------------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# -------------------------------------------------------------------
# COMPLETE PIPELINE
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
# TRAIN
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train,
)

print(
    "Final Random Forest training completed."
)


# -------------------------------------------------------------------
# EVALUATE
# -------------------------------------------------------------------

print("\nGenerating validation predictions...")

y_pred = pipeline.predict(
    X_test
)


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
# RESULTS
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
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
# SAVE MODEL
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    "Model saved successfully!"
)

print(
    f"\nSaved to:\n{MODEL_FILE}"
)


# -------------------------------------------------------------------
# MODEL SIZE
# -------------------------------------------------------------------

model_size_mb = (
    MODEL_FILE.stat().st_size
    / (1024 * 1024)
)

print(
    f"\nModel size: "
    f"{model_size_mb:.2f} MB"
)


# -------------------------------------------------------------------
# COMPLETION
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 70)