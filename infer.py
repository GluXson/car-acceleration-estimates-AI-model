"""Inferowanie na nowych danych"""
import torch
import pandas as pd
import joblib
from src.model import MLP
import sys

if len(sys.argv) < 2:
    print("Usage: python infer.py <input.csv>")
    sys.exit(1)

input_file = sys.argv[1]

# Wczytaj dane i znormalizuj
df = pd.read_csv(input_file)
scaler = joblib.load('models/scaler.pkl')
X = scaler.transform(df.values)

# Model
model = MLP(input_dim=X.shape[1])
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Predykcja
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    pred = model(X_tensor)

df['predicted_mpg'] = pred.numpy()
df.to_csv('data/examples/output.csv', index=False)
print(f"Predykcje zapisane → data/examples/output.csv")
