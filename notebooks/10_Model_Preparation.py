"""
Model Preparation
-----------------

Prepares the engineered Nassau Candy dataset
for machine-learning models.

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


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_nassau_candy.csv"
)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

print("=" * 70)
print("MODEL PREPARATION")
print("=" * 70)

print("\nLoading featured dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# -------------------------------------------------------------------
# Target
# -------------------------------------------------------------------

TARGET = "lead_time_days"


if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# -------------------------------------------------------------------
# Remove target/leakage columns
# -------------------------------------------------------------------

excluded_columns = [
    # Target
    TARGET,

    # Target-derived features
    "lead_time_category",
    "profit_per_shipping_day",

    # Future/outcome information
    "Ship Date",
    "Original Ship Date",

    # Identifier columns
    "Row ID",
    "Order ID",
    "Customer ID",

    # Redundant date representation
    "Order Date",

    # High-cardinality/redundant descriptive fields
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
# Identify feature types
# -------------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


# -------------------------------------------------------------------
# Print feature information
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    f"\nTotal features      : {len(X.columns)}"
)

print(
    f"Numerical features  : {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)


print("\nNumerical features:")

for feature in numeric_features:
    print(f"- {feature}")


print("\nCategorical features:")

for feature in categorical_features:
    print(f"- {feature}")


# -------------------------------------------------------------------
# Train / test split
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=None
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
            )
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
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# -------------------------------------------------------------------
# Combined preprocessing
# -------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# -------------------------------------------------------------------
# Fit preprocessing on training data only
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FITTING PREPROCESSOR")
print("=" * 70)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape : "
    f"{X_test_processed.shape}"
)


# -------------------------------------------------------------------
# Final validation
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"Training target mean: "
    f"{y_train.mean():.4f}"
)

print(
    f"Testing target mean : "
    f"{y_test.mean():.4f}"
)

print(
    f"\nMissing values in X_train: "
    f"{X_train.isnull().sum().sum()}"
)

print(
    f"Missing values in X_test: "
    f"{X_test.isnull().sum().sum()}"
)

print(
    f"Missing values in y_train: "
    f"{y_train.isnull().sum()}"
)

print(
    f"Missing values in y_test: "
    f"{y_test.isnull().sum()}"
)


print("\n" + "=" * 70)
print("MODEL PREPARATION COMPLETED")
print("=" * 70)