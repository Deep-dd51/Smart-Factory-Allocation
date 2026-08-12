"""
Central configuration for the Smart Factory Allocation project.
"""

from pathlib import Path


# -------------------------------------------------------------------
# Project directories
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"


# -------------------------------------------------------------------
# Dataset paths
# -------------------------------------------------------------------

RAW_DATA_FILE = RAW_DATA_DIR / "Nassau Candy Distributor.csv"

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR / "cleaned_nassau_candy.csv"
)

FEATURED_DATA_FILE = (
    PROCESSED_DATA_DIR / "featured_nassau_candy.csv"
)


# -------------------------------------------------------------------
# Machine learning configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20


# -------------------------------------------------------------------
# Project metadata
# -------------------------------------------------------------------

PROJECT_NAME = "Smart Factory Allocation & Logistics Intelligence System"
VERSION = "1.0.0"