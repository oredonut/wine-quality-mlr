"""Model training utilities for the medical cost regression project."""

from sklearn.linear_model import LinearRegression

def train_model(x_train: object, y_train: object) -> LinearRegression:
    """Train a linear regression model on the provided training data.

    Parameters:
        x_train: Feature matrix for training.
        y_train: Target vector for training.

    Returns:
        A fitted scikit-learn LinearRegression model.

    Side effects:
        Fits a model in memory (no disk I/O).
    """
    # Create and fit a simple linear regression model.
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
