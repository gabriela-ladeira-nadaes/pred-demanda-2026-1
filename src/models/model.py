from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

def save_model(path: str | Path, model: RandomForestRegressor) -> None:
    """
    Nesta etapa, RandomForestRegressor é utilizado apenas como exemplo de
    tipo na assinatura da função para demonstrar o uso de tipagem em Python.
    A implementação final poderá suportar diferentes tipos de modelos.
    """
    pass
