from pathlib import Path
import pandas as pd

def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def validate_data(data: pd.DataFrame) -> bool:
    return not data.empty
