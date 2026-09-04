import torch.nn as nn
import torch

"""Modelos implementados"""
class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
                
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        out = self.linear(x)
        return out
    
class FinnancialModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FinnancialModel, self).__init__()
        
        self.net = nn.Sequential(
            # Primeira camada oculta
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            
            # Segunda camada oculta
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            
            # Terceira camada oculta
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Camada de Saída (1 neurônio para prever Weekly_Sales)
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)