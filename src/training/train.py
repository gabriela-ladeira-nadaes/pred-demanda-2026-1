import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Placeholder do treinamento

def train_model(model: nn.Module, epochs: int, train_loader: DataLoader, test_loader: DataLoader) -> nn.Module:
    """Treina o modelo a partir do DataLoader de treino"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Agora iteramos sobre os lotes (batches) entregues pelo DataLoader
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad(): 
         for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            test_loss += loss.item()
            
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train MSELoss: {avg_train_loss:.4f} | Test MSELoss: {avg_test_loss:.4f}")

    return model

class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
                
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        out = self.linear(x)
        return out
