import torch
import torch.nn as nn
import joblib

from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def save_model(model: nn.Module, scaler_y: StandardScaler, standardizer: ColumnTransformer):
    """Salvar o modelo treinado e os transformadores de dados"""
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    model_name = f"pytorch_workflow_model_{timestamp}.pth"
    scaler_y_name = f"scaler_y_{timestamp}.pkl"
    transformer_name = f"column_transformer_{timestamp}.pkl"

    model_save_path = model_path / model_name
    scaler_y_save_path = model_path / scaler_y_name
    transformer_save_path = model_path / transformer_name

    checkpoint = {
        'name': model.__class__.__name__,
        'weights': model.state_dict()
    }

    torch.save(obj=checkpoint, f=model_save_path)    
    joblib.dump(scaler_y, scaler_y_save_path)
    joblib.dump(standardizer, transformer_save_path)

    print(f"Modelo salvo em: {model_save_path}")
    print(f"Scaler Y salvo em: {scaler_y_save_path}")
    print(f"Transformador X salvo em: {transformer_save_path}")


def load_model(model: nn.Module, path: str | Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["weights"] if isinstance(checkpoint, dict) and "weights" in checkpoint else checkpoint
    model.load_state_dict(weights)
    model.eval()
    return model

class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
                
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        out = self.linear(x)
        return out
    
class FinancialModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FinancialModel, self).__init__()
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LSTMModel(nn.Module):
    """Modelo LSTM para capturar dependências temporais de longo prazo nas vendas."""
    
    def __init__(self, input_dim: int,output_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :]) 
        return out
