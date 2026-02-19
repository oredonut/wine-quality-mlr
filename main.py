from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

def main():
    df = load_data("data/insurance.csv")
    x_train, x_test, y_train, y_test= preprocess_data(df)


    print("Mean Squared Error: {mse}")
    print("R^2 Score: {r2}")

    if _names_ == "__main__":
        main()