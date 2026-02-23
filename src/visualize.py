"""Visualization utilities for model diagnostics."""

import matplotlib.pyplot as plt

def plot_residuals(y_test: object, y_pred: object) -> None:
    """Plot residuals to visualize prediction errors.

    Parameters:
        y_test: True target values for testing.
        y_pred: Model predictions for the test set.

    Returns:
        None.

    Side effects:
        Displays a matplotlib figure and saves `residual_plot.png` to disk.
    """
    # Residuals show where predictions differ from actual values.
    residuals = y_test - y_pred

    # Create a scatter plot of predicted values vs residuals.
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)

    # Reference line at zero residual for visual balance.
    plt.axhline(0, color="red", linestyle="--")

    # Add labels and title for clarity.
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()

    # Save the plot for later inspection and show it on screen.
    plt.savefig("residual_plot.png")
    plt.show()
