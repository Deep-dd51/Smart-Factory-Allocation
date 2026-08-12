"""
Production data processing pipeline.

Loads the raw Nassau Candy dataset, applies preprocessing,
validates the result, and saves the cleaned dataset.
"""

import sys
import logging
from pathlib import Path


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from config.settings import CLEANED_DATA_FILE
from src.data.data_loader import load_raw_data
from src.data.preprocessing import preprocess_data


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main() -> None:
    """Run the complete data cleaning pipeline."""

    logger.info("Starting data processing pipeline.")

    # Load raw dataset
    df = load_raw_data()

    logger.info(
        "Raw dataset shape: %s",
        df.shape,
    )

    # Preprocess
    cleaned_df = preprocess_data(df)

    logger.info(
        "Cleaned dataset shape: %s",
        cleaned_df.shape,
    )

    # Create output directory
    CLEANED_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save cleaned dataset
    cleaned_df.to_csv(
        CLEANED_DATA_FILE,
        index=False,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("DATA PROCESSING COMPLETED")
    print("=" * 70)

    print(f"Rows              : {len(cleaned_df):,}")
    print(f"Columns           : {len(cleaned_df.columns)}")

    print(
        f"Missing values    : "
        f"{cleaned_df.isnull().sum().sum():,}"
    )

    print(
        f"Duplicate rows    : "
        f"{cleaned_df.duplicated().sum():,}"
    )

    print(
        f"Lead time range   : "
        f"{cleaned_df['lead_time_days'].min()} - "
        f"{cleaned_df['lead_time_days'].max()} days"
    )

    print(
        f"\nSaved to:\n{CLEANED_DATA_FILE}"
    )


if __name__ == "__main__":
    main()