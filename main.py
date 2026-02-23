"""Entry point for training and evaluating the medical cost regression model."""

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_residuals

def main() -> None:
    """Run the full modeling workflow: load, preprocess, train, evaluate, visualize.

    Parameters:
        None.

    Returns:
        None.

    Side effects:
        Reads data from disk, trains a model in memory, writes a plot image,
        and prints metrics to stdout.
    """
    # Load the raw insurance dataset from disk.
    raw_data = load_data("data/insurance.csv")

    # Preprocess data: encode categories and split into train/test sets.
    x_train, x_test, y_train, y_test = preprocess_data(raw_data)

    # Train the regression model using the training split.
    model = train_model(x_train, y_train)

    # Evaluate model quality on the test split.
    mse, r2 = evaluate_model(model, x_test, y_test)

    # Generate predictions for residual diagnostics.
    y_pred = model.predict(x_test)

    # Visualize residuals to inspect model error patterns.
    plot_residuals(y_test, y_pred)

    # Report metrics for quick feedback.
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    main()

# How to run this file:
# `python main.py`
