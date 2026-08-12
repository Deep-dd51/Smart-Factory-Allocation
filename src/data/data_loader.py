"""
Data loading utilities for the Smart Factory Allocation project.
"""

from pathlib import Path
import logging

import pandas as pd

from config.settings import RAW_DATA_FILE


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data loading functions
# -------------------------------------------------------------------

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is empty or cannot be loaded as a CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {file_path.suffix}"
        )

    try:
        df = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The CSV file is empty: {file_path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Unable to parse CSV file: {file_path}"
        ) from exc

    logger.info(
        "Dataset loaded successfully: %s rows, %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load the project's original Nassau Candy dataset.

    Returns
    -------
    pd.DataFrame
        Raw Nassau Candy dataset.
    """

    return load_csv(RAW_DATA_FILE)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a basic summary of a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize.

    Returns
    -------
    dict
        Dataset summary information.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            2,
        ),
    }