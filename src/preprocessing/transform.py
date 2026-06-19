import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def split_data(
    data: pd.DataFrame,
    target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # placeholders
    X_train, X_test = X, X 
    y_train, y_test = y, y

    return X_train, X_test, y_train, y_test
