import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_device() -> torch.device:
    """Retorna a GPU se disponível, senão a CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte arrays NumPy para tensores."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.reshape(-1, 1)
    return X_tensor, y_tensor

def make_dataset(X: torch.Tensor, y: torch.Tensor) -> TensorDataset:
    """Empacota features e target em um dataset de tensores"""
    return TensorDataset(X, y)

def make_dataloader(dataset: TensorDataset, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """Cria dataloader a partir de um TensorDataset"""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader