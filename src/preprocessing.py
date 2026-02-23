"""Preprocessing steps for preparing insurance data for modeling."""

import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Encode categorical variables and split the dataset into train/test sets.

    Parameters:
        df: Raw insurance dataset as a pandas DataFrame.

    Returns:
        A 4-tuple containing:
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training target values.
        - y_test: Testing target values.

    Side effects:
        None. The input DataFrame is not modified in place.
    """
    # One-hot encode categorical columns so the linear model can use them.
    # drop_first avoids perfect multicollinearity (dummy variable trap).
    encoded_df = pd.get_dummies(df, drop_first=True, dtype=int)

    # Separate features from the target (charges).
    features = encoded_df.drop("charges", axis=1)
    target = encoded_df["charges"]

    # Split into train/test for unbiased evaluation.
    return train_test_split(features, target, test_size=0.2, random_state=42)
