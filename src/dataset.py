import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class FuelDataset(Dataset):
    """Dataset Auto MPG z preprocessing"""
    
    def __init__(self, csv_file, scaler=None):
        # Wczytaj i oczyść dane
        df = pd.read_csv(csv_file, sep=r'\s+', header=None,
                         names=['mpg','cyl','disp','hp','weight','acc','year','origin','name'])
        df['hp'] = pd.to_numeric(df['hp'], errors='coerce')  # '?' → NaN
        df = df.dropna()                                      # Usuń NaN
        df = df.drop('name', axis=1)                         # Usuń nazwę auta
        
        # Rozdziel X i y
        self.y = df['mpg'].values.reshape(-1, 1).astype('float32')
        self.X = df.drop('mpg', axis=1).values.astype('float32')
        
        # Normalizacja: X' = (X - μ) / σ
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_loaders(csv_file, batch_size=32, splits=(0.7, 0.15, 0.15)):
    """Tworzy train/val/test DataLoader"""
    dataset = FuelDataset(csv_file)
    
    # Split danych: 70% train, 15% val, 15% test
    n = len(dataset)
    train_size = int(splits[0] * n)
    val_size = int(splits[1] * n)
    test_size = n - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Zapisz scaler do inference
    joblib.dump(dataset.scaler, 'models/scaler.pkl')
    
    return train_loader, val_loader, test_loader, dataset.X.shape[1]
