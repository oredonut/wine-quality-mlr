"""Data loading utilities for the medical cost regression project."""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Parameters:
        path: File system path to the CSV file.

    Returns:
        A pandas DataFrame containing the raw insurance data.

    Side effects:
        Reads from the local filesystem.
    """
    # Read the CSV into a DataFrame for downstream preprocessing.
    return pd.read_csv(path)
