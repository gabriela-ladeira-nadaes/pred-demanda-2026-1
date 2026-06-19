import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def evaluate_metrics(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    return {}
