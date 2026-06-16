import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def validate_data(data: pd.DataFrame) -> Boolean:
    return true
