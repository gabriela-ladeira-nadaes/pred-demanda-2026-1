import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# Placeholder do treinamento

def train_model(model: nn.Module, epochs: int, train_loader: DataLoader, test_loader: DataLoader) -> nn.Module:
    """Treina o modelo a partir do DataLoader de treino"""
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        train_mse_loss = 0.0
        train_mae_loss = 0.0
        
        train_predictions = []
        train_targets = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_X)
          
            loss_mse = criterion_mse(predictions, batch_y)
            loss_mae = criterion_mae(predictions, batch_y)
            loss_mse.backward()
            optimizer.step()
            train_mse_loss += loss_mse.item()
            train_mae_loss += loss_mae.item()
            train_predictions.extend(predictions.detach().numpy())
            train_targets.extend(batch_y.detach().numpy())
        
        avg_train_mse = train_mse_loss / len(train_loader)
        avg_train_rmse = avg_train_mse ** 0.5
        avg_train_mae = train_mae_loss / len(train_loader)
        train_r2 = r2_score(train_targets, train_predictions)

        model.eval()
        test_mse_loss = 0.0
        test_mae_loss = 0.0
        
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                mse = criterion_mse(predictions, batch_y)
                mae = criterion_mae(predictions, batch_y)
                
                test_mse_loss += mse.item()
                test_mae_loss += mae.item()
                test_predictions.extend(predictions.numpy())
                test_targets.extend(batch_y.numpy())
                
      
        avg_test_mse = test_mse_loss / len(test_loader)
        avg_test_rmse = avg_test_mse ** 0.5
        avg_test_mae = test_mae_loss / len(test_loader)
        test_r2 = r2_score(test_targets, test_predictions)
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  TRAIN -> RMSE: {avg_train_rmse:.2f} | MAE: {avg_train_mae:.2f} | R2: {train_r2:.4f}")
        print(f"  TEST  -> RMSE: {avg_test_rmse:.2f} | MAE: {avg_test_mae:.2f} | R2: {test_r2:.4f}")
        print("-" * 50)
    return model

class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
                
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        out = self.linear(x)
        return out
