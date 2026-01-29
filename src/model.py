import torch
import torch.nn as nn

class MLP(nn.Module):
    """MLP dla regresji MPG: 7 → 128 → 64 → 32 → 1"""
    
    def __init__(self, input_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)   # Warstwa 1: 7 → 128 neuronów
        self.fc2 = nn.Linear(128, 64)          # Warstwa 2: 128 → 64
        self.fc3 = nn.Linear(64, 32)           # Warstwa 3: 64 → 32
        self.fc4 = nn.Linear(32, 1)            # Warstwa 4: 32 → 1 (MPG)
        
        self.relu = nn.ReLU()                  # Aktywacja: max(0, x)
        self.dropout = nn.Dropout(0.15)        # Dropout: losowo wyłącza 15% neuronów
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicjalizacja wag (He initialization dla ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        """Forward pass: x → ReLU → Dropout → ... → output"""
        x = self.dropout(self.relu(self.fc1(x)))  # h1 = Dropout(ReLU(W1·x))
        x = self.dropout(self.relu(self.fc2(x)))  # h2 = Dropout(ReLU(W2·h1))
        x = self.relu(self.fc3(x))                # h3 = ReLU(W3·h2)
        x = self.fc4(x)                            # y = W4·h3 (bez aktywacji)
        return x
