"""Model evaluation helpers for regression metrics."""

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model: object, x_test: object, y_test: object) -> tuple[float, float]:
    """Evaluate a trained model using MSE and R^2 metrics.

    Parameters:
        model: A fitted regression model with a predict method.
        x_test: Feature matrix for testing.
        y_test: True target values for testing.

    Returns:
        A 2-tuple (mse, r2) with mean squared error and R^2 score.

    Side effects:
        None.
    """
    # Generate predictions on the held-out test set.
    predictions = model.predict(x_test)

    # Compute regression metrics for model quality.
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mse, r2
