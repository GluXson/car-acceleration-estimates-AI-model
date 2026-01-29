"""Ewaluacja na test set + interaktywny tryb demo/custom"""
import torch
from src.dataset import get_loaders
from src.model import MLP
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import random

def evaluate_test_set(model, test_loader):
    """Ewaluacja na pełnym test set"""
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            y_true.extend(y.numpy())
            y_pred.extend(pred.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metryki
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"METRYKI NA ZBIORZE TESTOWYM")
    print(f"{'='*50}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f} MPG")
    print(f"Test R²:   {r2:.4f}")
    print(f"{'='*50}\n")
    
    # Wykres predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Rzeczywiste MPG')
    plt.ylabel('Przewidywane MPG')
    plt.title(f'Predictions (R² = {r2:.3f})')
    plt.grid(True)
    plt.savefig('results/plots/predictions.png', dpi=150)
    print("Wykres zapisany: results/plots/predictions.png\n")
    
    return y_true, y_pred

def demo_mode(model):
    """Demo: 3 losowe auta z datasetu"""
    print(f"\n{'='*50}")
    print("TRYB DEMO - 3 losowe samochody")
    print(f"{'='*50}\n")
    
    # Wczytaj raw data z nazwami aut
    df = pd.read_csv('data/raw/auto_mpg.data', sep=r'\s+', header=None,
                     names=['mpg','cyl','disp','hp','weight','acc','year','origin','name'])
    df['hp'] = pd.to_numeric(df['hp'], errors='coerce')
    df = df.dropna()
    
    # Losuj 3 auta
    samples = df.sample(n=3, random_state=random.randint(0, 10000))
    
    # Wczytaj scaler
    scaler = joblib.load('models/scaler.pkl')
    
    for idx, row in samples.iterrows():
        # Przygotuj features (bez mpg i name)
        X = row[['cyl','disp','hp','weight','acc','year','origin']].values.reshape(1, -1)
        X_norm = scaler.transform(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        
        # Predykcja
        with torch.no_grad():
            pred = model(X_tensor).item()
        
        actual = row['mpg']
        error = abs(pred - actual)
        
        print(f"🚗 {row['name']}")
        print(f"   Parametry: {int(row['cyl'])} cyl, {row['disp']:.0f} ccm, "
              f"{row['hp']:.0f} HP, {row['weight']:.0f} lbs")
        print(f"   Rzeczywiste MPG:  {actual:.1f}")
        print(f"   Przewidywane MPG: {pred:.1f}")
        print(f"   Błąd:             {error:.1f} MPG")
        print()

def custom_file_mode(model):
    """Tryb custom: wczytanie z pliku CSV"""
    print(f"\n{'='*50}")
    print("TRYB CUSTOM - Predykcja z pliku")
    print(f"{'='*50}\n")
    
    file_path = input("Podaj ścieżkę do pliku CSV (np. data/examples/input.csv): ").strip()
    
    try:
        # Wczytaj plik
        df = pd.read_csv(file_path)
        print(f"✅ Wczytano {len(df)} wierszy z pliku\n")
        
        # Debugowanie: sprawdź kolumny
        expected_cols = ['cyl', 'disp', 'hp', 'weight', 'acc', 'year', 'origin']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ BŁĄD: Brakujące kolumny: {missing_cols}")
            print(f"   Twoje kolumny: {list(df.columns)}")
            print(f"   Wymagane: {expected_cols}")
            return
        
        # Wczytaj scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Normalizuj i przewiduj
        X = df[expected_cols].values
        X_norm = scaler.transform(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(X_tensor).numpy()
        
        # Dodaj predykcje do DataFrame
        df['predicted_mpg'] = predictions
        
        # Wyświetl wyniki
        print("WYNIKI PREDYKCJI:")
        print("-" * 80)
        for i, row in df.iterrows():
            print(f"Auto {i+1}: {int(row['cyl'])} cyl, {row['disp']:.0f} ccm, "
                  f"{row['hp']:.0f} HP → Przewidywane MPG: {row['predicted_mpg']:.1f}")
        
        # Zapisz output
        output_path = file_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✅ Wyniki zapisane: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ BŁĄD: Nie znaleziono pliku: {file_path}")
    except Exception as e:
        print(f"❌ BŁĄD: {e}")

def main():
    # Wczytaj dane i model
    _, _, test_loader, input_dim = get_loaders('data/raw/auto_mpg.data')
    model = MLP(input_dim=input_dim)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # Ewaluacja na test set (zawsze)
    y_true, y_pred = evaluate_test_set(model, test_loader)
    
    # Interaktywny tryb
    while True:
        choice = input("Czy wykonać demo testowe? (y/n): ").strip().lower()
        
        if choice == 'y':
            demo_mode(model)
            break
        elif choice == 'n':
            custom_file_mode(model)
            break
        else:
            print("❌ Niepoprawna opcja. Wpisz 'y' lub 'n'.")

if __name__ == "__main__":
    main()
