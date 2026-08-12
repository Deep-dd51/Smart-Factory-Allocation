"""
Lead-Time Prediction Service

Loads the trained Random Forest pipeline and
provides prediction functionality.
"""

from pathlib import Path

import joblib
import pandas as pd


# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


# -------------------------------------------------------------------
# MODEL PATH
# -------------------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_lead_time.pkl"
)


# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------

def load_model():
    """
    Load the trained Random Forest pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# -------------------------------------------------------------------
# PREDICT
# -------------------------------------------------------------------

def predict_lead_time(
    model,
    data: pd.DataFrame,
):
    """
    Predict lead time for one or more records.

    Parameters
    ----------
    model:
        Trained model pipeline.

    data:
        DataFrame containing the model features.

    Returns
    -------
    predictions:
        Predicted lead time in days.
    """

    predictions = model.predict(
        data
    )

    return predictions


# -------------------------------------------------------------------
# MAIN TEST
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("LEAD-TIME PREDICTION SERVICE")
    print("=" * 70)

    print("\nLoading trained model...")

    model = load_model()

    print(
        "Model loaded successfully!"
    )

    print(
        f"\nModel path:\n{MODEL_PATH}"
    )

    print("\nPrediction service is ready.")