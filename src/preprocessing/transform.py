import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils.config import (DATE_COLUMN, TARGET_COLUMN, TEST_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURES_WITHOUT_BOOLEAN, FIXED_SEED)

def sort_data(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ordena o df por data"""
    return data.sort_values(by=date_col)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Limpeza do dataset"""
    data = data.copy()
    data = data.drop(columns=['Unnamed: 0'], errors="ignore")
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data = sort_data(data, DATE_COLUMN).reset_index(drop=True)
    return data

def build_features_matrix(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Monta arrays NumPy para a matriz de features (X) e o vetor alvo (y). 
    Ao chamar esta funcao, as features categoricas devem estar encodadas."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES # Necessario ordenar desta forma para a funcao standardize
    X = data[feature_cols].to_numpy(dtype = np.float32)
    y = data[TARGET_COLUMN].to_numpy(dtype = np.float32)
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = TEST_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide dados de treino e teste ordenados cronologicamente para evitar vazamento"""   
    return train_test_split(X, y, test_size=test_size, random_state=FIXED_SEED, shuffle=False)

def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    standardizer = ColumnTransformer(
        transformers=[           
            ('categorics', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES_WITHOUT_BOOLEAN),
            ('numerics', StandardScaler(), NUMERIC_FEATURES),
            ('boolean', 'passthrough', ["IsHoliday"]) 
        ]
    )

    X_train_processed = standardizer.fit_transform(X_train_df)

    X_test_processed = standardizer.transform(X_test_df)
    return X_train_processed, X_test_processed

def describe_array(data: np.ndarray) -> dict[str, object]:
    """
    Retorna informacoes basicas sobre os dados.
    """
    return {
        "shape": data.shape,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }