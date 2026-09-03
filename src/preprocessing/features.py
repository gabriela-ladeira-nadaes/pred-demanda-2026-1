import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Separa as Features temporais"""
    data = data.copy()
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
    return data
