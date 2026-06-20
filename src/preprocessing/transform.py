import pandas as pd
import numpy as np

def sort_data(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    sorted_data = data.sort_values(by=date_col)
    return sorted_data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def split_data(
    data: pd.DataFrame,
    target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    X = data.drop(columns=[target_column])
    y = np.array(data[target_column])

    # placeholders
    X_train, X_test = X, X 
    y_train, y_test = y, y

    return X_train, X_test, y_train, y_test

def pick_columns(data: pd.DataFrame, desired_columns:np.ndarray) -> pd.DataFrame:
    dataset_columns = np.array(data[0])
    pass


def describe_array(data: np.ndarray) -> dict[str, object]:
    """
    Retorna informações básicas sobre o array.
    """
    return {
        "shape": data.shape,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }