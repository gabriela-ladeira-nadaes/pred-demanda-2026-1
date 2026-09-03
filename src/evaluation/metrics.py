import numpy as np

def wmae_score_numpy(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    """Calcula o Erro Absoluto Médio Ponderado (WMAE) usando NumPy"""
    pesos = np.where(is_holiday == 1, 5, 1)
    erro_absoluto = np.abs(y_true - y_pred)
    return np.sum(pesos * erro_absoluto) / np.sum(pesos)
