import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data

def split_data(
    data: pd.DataFrame,
    target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    return x_train, y_train, x_test, y_test
