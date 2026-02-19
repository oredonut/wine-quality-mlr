from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, x_test, y_test):
    predictions = model.predcit(x_test)
    mse = mean_squared_error(y_test, predictions)

    return mse, r2
