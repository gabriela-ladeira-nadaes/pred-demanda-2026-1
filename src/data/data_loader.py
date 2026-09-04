from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Ler o csv para um DataFrame"""
    return pd.read_csv(path)

def validate_data(data: pd.DataFrame) -> bool:
    """Validade se o Dataframe é vazio"""
    return not data.empty

    
