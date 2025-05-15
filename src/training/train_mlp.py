import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.data.load_dataset import load_dataset
from src.models.mlp_model import MLP
from src.metrics.regression_metrics import evaluate_regression
from src.visualization.plot_metrics import plot_metrics


df = load_dataset()
X = df[['X_int', 'Y_int', 'X_fac', 'Y_fac', 'Ang']]
y = df[['Mean']]
INPUT_DIM = X.shape[1]

HIDDEN_DIM = 128
OUTPUT_DIM = 1
LR = 0.005
EPOCHS = 500
BATCH_SIZE = 64

model_name = f"mlp_{INPUT_DIM}f_{HIDDEN_DIM}h_{EPOCHS}ep"

#  Checking if the Logs folder exists
os.makedirs("results/logs", exist_ok=True)
log_path = os.path.join("results", "logs", f"{model_name}.txt")
log_file = open(log_path, "w")
log_file.write("Epoch\tLoss\n") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Checking for overlaps between the train and test sets
train_indices = set(map(tuple, X_train.numpy()))
test_indices = set(map(tuple, X_test.numpy()))
intersection = train_indices.intersection(test_indices)
print(f"Number of matching rows: {len(intersection)}") 

model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
train_loss_history = []

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()

    train_loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        log_str = f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}"
        print(log_str)
        log_file.write(f"{epoch+1}\t{loss.item():.6f}\n")
        
log_file.close()

# Saving model weights and configuration
weights_path = os.path.join("models", "weights", f"{model_name}.pth")
config_path = os.path.join("models", "configs", f"{model_name}.json")

torch.save(model.state_dict(), weights_path)

config = {
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "lr": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "features": ['X_int', 'Y_int', 'X_fac', 'Y_fac', 'Ang'],
    "targets": ['Mean']
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Model weights saved: {weights_path}")
print(f"Model configuration saved: {config_path}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    metrics = evaluate_regression(y_test, y_pred)

print("Evaluation metrics on test set:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


metrics_path = os.path.join("results", "metrics", f"{model_name}.json")
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

plot_metrics(
    y_test,
    y_pred,
    train_loss_history=train_loss_history,
    val_loss_history=[],  
    model_name=model_name,
    show=True 
)   
