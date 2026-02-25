import matplotlib.pyplot as plt

def plot_residuals(y_test: object, y_pred: object) -> None:

    residuals = y_test - y_pred

   
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="blue", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()

    plt.savefig("residual_plot.png")
    plt.show()
