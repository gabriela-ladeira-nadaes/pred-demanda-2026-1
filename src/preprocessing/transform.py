import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import (DATE_COLUMN, TARGET_COLUMN, TEST_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEMPORAL_FEATURES,RANDOM_SEED)

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
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalizacao z-score utilizando NumPy.
    Calcula media e desvio padrao somente no conjunto de treino e aplica aos conjuntos de treino e teste para evitar vazamento.
    Colunas categoricas nao sao alteradas. Alteramos apenas colunas com valores continuos."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Selecao de features continuas (nao-categoricas)
    n_continuous = len(NUMERIC_FEATURES) #+ len(TEMPORAL_FEATURES) # As primeiras n colunas sao continuas conforme definido em build_features_matrix
    cont_train = X_train[:, :n_continuous].astype(np.float64) # float 64 porque, para calcular mean e std, a soma dos anos estouraria o limite de inteiros exatos do float32

    # Metricas e normalizacao
    mean = cont_train.mean(axis = 0)
    std = cont_train.std(axis = 0)
    std[std == 0] = 1.0 # evitar divisao por zero caso a coluna seja constante

    out_dtype = X_train.dtype
    X_train[:, :n_continuous] = ((X_train[:, :n_continuous].astype(np.float64) - mean) / std).astype(out_dtype)
    X_test[:, :n_continuous]  = ((X_test[:, :n_continuous].astype(np.float64)  - mean) / std).astype(out_dtype)
    
    return X_train, X_test

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