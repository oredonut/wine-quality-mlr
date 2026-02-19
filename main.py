from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_residuals

def main():
    df = load_data("data/insurance.csv")
    x_train, x_test, y_train, y_test = preprocess_data(df)
    model = train_model(x_train, y_train)
    mse, r2 = evaluate_model(model, x_test, y_test)
    y_pred = model.predict(x_test)
    plot_residuals(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    main()
