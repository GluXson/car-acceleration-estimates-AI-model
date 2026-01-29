"""Trenuje model z bias correction"""
from src.dataset import get_loaders
from src.model import MLP
from src.trainer import Trainer
import torch
import numpy as np

def calculate_bias(model, loader):
    """Oblicza średnie przeszacowanie/niedoszacowanie modelu"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            error = pred - y_batch  # Dodatnie = przeszacowanie, ujemne = niedoszacowanie
            errors.extend(error.numpy().flatten())
    
    bias = np.mean(errors)
    return bias

print("="*50)
print("TRENING MODELU Z AUTO-KOREKCJĄ")
print("="*50 + "\n")

# Dane
train_loader, val_loader, test_loader, input_dim = get_loaders('data/raw/auto_mpg.data')

# Model
model = MLP(input_dim=input_dim)
print(f"Model: {sum(p.numel() for p in model.parameters())} parametrów\n")

# FAZA 1: Podstawowy trening
print("="*50)
print("FAZA 1: Podstawowy trening")
print("="*50 + "\n")
trainer = Trainer(model, train_loader, val_loader, lr=0.001, epochs=200)
trainer.train()

# FAZA 2: Analiza bias
print("\n" + "="*50)
print("FAZA 2: Analiza bias")
print("="*50 + "\n")

model.load_state_dict(torch.load('models/best_model.pth'))
train_bias = calculate_bias(model, train_loader)
val_bias = calculate_bias(model, val_loader)

print(f"Train bias: {train_bias:.4f} MPG")
print(f"Val bias:   {val_bias:.4f} MPG")

if abs(val_bias) > 0.5:
    if val_bias > 0:
        print(f"\n⚠️  Model PRZESZACOWUJE o ~{val_bias:.2f} MPG")
    else:
        print(f"\n⚠️  Model NIEDOSZACOWUJE o ~{abs(val_bias):.2f} MPG")
    
    print("Uruchamiam bias correction...\n")
    
    # FAZA 3: Fine-tuning z correction
    print("="*50)
    print("FAZA 3: Fine-tuning z bias correction")
    print("="*50 + "\n")
    
    # Fine-tune z mniejszym lr
    trainer_ft = Trainer(model, train_loader, val_loader, lr=0.0001, epochs=50)
    trainer_ft.train()
    
    # Sprawdź nowy bias
    model.load_state_dict(torch.load('models/best_model.pth'))
    new_val_bias = calculate_bias(model, val_loader)
    
    print(f"\n✅ Nowy val bias: {new_val_bias:.4f} MPG (poprawa: {abs(val_bias - new_val_bias):.4f})")
else:
    print("\n✅ Bias jest akceptowalny (<0.5 MPG), brak potrzeby korekcji")

print("\n" + "="*50)
print("Model zapisany: models/best_model.pth")
print("Uruchom: python evaluate.py")
print("="*50)
