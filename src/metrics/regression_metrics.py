from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

def evaluate_regression(y_true, y_pred):
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return {"MSE": mse, "MAE": mae, "R2": r2}
