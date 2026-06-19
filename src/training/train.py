import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor()
    # model.fit(X_train, y_train) a ser implementado
    return model
    