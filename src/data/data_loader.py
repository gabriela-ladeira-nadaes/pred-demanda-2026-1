from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn


def load_data(path: str | Path) -> pd.DataFrame:
    """Ler o csv para um DataFrame"""
    return pd.read_csv(path)

def validate_data(data: pd.DataFrame) -> bool:
    """Validade se o Dataframe é vazio"""
    return not data.empty

def save_model(model: torch.nn.Module):
    """Salvar o modelo treinado na past models"""
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    timestamp = agora.strftime("%Y%m%d_%H%M%S")
    model_name = f"pytorch_workflow_model_{timestamp}.pth"
    model_save_path = model_path / model_name
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f"Modelo salvo com sucesso em: {model_save_path}")
    
