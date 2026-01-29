import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

class Trainer:
    """Zarządza treningiem MLP"""
    
    def __init__(self, model, train_loader, val_loader, lr=0.001, epochs=200):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        
        # Loss: MSE = (1/n)·Σ(y - ŷ)²
        self.criterion = nn.MSELoss()
        
        # Optimizer: Adam (adaptive learning rate)
        # θ_t+1 = θ_t - α·m_t/√(v_t + ε)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Historia
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Jedna epoka treningu"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in self.train_loader:
            # Forward: ŷ = f(X; θ)
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            
            # Backward: ∇θ L = ∂L/∂θ
            self.optimizer.zero_grad()  # Wyzeruj gradienty
            loss.backward()              # Oblicz gradienty (backprop)
            self.optimizer.step()        # Update wag: θ = θ - α·∇θ
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Walidacja (bez gradientów)"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():  # Wyłącz obliczanie gradientów
            for X_batch, y_batch in self.val_loader:
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Główna pętla treningowa"""
        print(f"Trening: {self.epochs} epok, lr={self.optimizer.param_groups[0]['lr']}")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Zapisz best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        print(f"\nBest Val Loss: {self.best_val_loss:.4f}")
        self.plot_losses()
    
    def plot_losses(self):
        """Wykres loss curves"""
        Path('results/plots').mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/loss_curves.png', dpi=150)
        print("Wykres zapisany: results/plots/loss_curves.png")
