import os
import torch
import joblib
import pandas as pd

from pathlib import Path
from datetime import timedelta
from models.model import FinancialModel, LinearRegression,LSTMModel
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_latest_model_and_scalers(models_dir: str = "models"):
    """Busca o modelo e os transformadores mais recentes baseados no timestamp."""
    models_path = Path(models_dir)
    
    # 1. Encontra todos os arquivos de modelo e pega o mais recente
    arquivos_modelos = list(models_path.glob("pytorch_workflow_model_*.pth"))
    if not arquivos_modelos:
        raise FileNotFoundError("Nenhum modelo encontrado na pasta especificada.")
    
    last_model_path = max(arquivos_modelos, key=os.path.getctime)    
    timestamp = last_model_path.stem.replace("pytorch_workflow_model_", "")
    
    checkpoint = torch.load(last_model_path, weights_only=False)
    class_name = checkpoint['name'] 
    weights = checkpoint['weights']

    first_key = list(weights.keys())[0]
    dynamic_input_dim = weights[first_key].shape[1]

    model_maps = {
        "FinancialModel": FinancialModel(dynamic_input_dim, 1),
        "LinearRegression": LinearRegression(dynamic_input_dim, 1),
        "LSTM" : LSTMModel(dynamic_input_dim,1)
    }
    
    if class_name not in model_maps:
        raise ValueError(f"Arquitetura '{class_name}' desconhecida pelo predictor.")
    
    # 3. Instancia o modelo certo e injeta os pesos
    model = model_maps[class_name]
    model.load_state_dict(weights)
    model.eval()
    
    print(f"Modelo carregado: {class_name} (Arquivo: {last_model_path.name})")
    
    scaler_y = joblib.load(models_path / f"scaler_y_{timestamp}.pkl")
    transformer = joblib.load(models_path / f"column_transformer_{timestamp}.pkl")
    
    return model, scaler_y, transformer

def generate_forecast(model: torch.nn.Module, scaler_y: StandardScaler, transformer: ColumnTransformer, start_date: str, df_historic: pd.DataFrame, store_id: int = None, dept_id: int = None, weeks_ahead: int = 26) -> pd.DataFrame:
    """Gera projeções para uma loja/dept específico ou para toda a rede Walmart."""
    
    unique_combos = df_historic[['Store', 'Dept', 'Size', 'Type']].drop_duplicates()
    
    if store_id is not None:
        unique_combos = unique_combos[unique_combos['Store'] == store_id]
    if dept_id is not None:
        unique_combos = unique_combos[unique_combos['Dept'] == dept_id]
        
    if unique_combos.empty:
        raise ValueError("Combinação de Loja e Departamento não encontrada.")

    df_historic_sorted = df_historic.sort_values('Date')
    latest_macro = df_historic_sorted.groupby('Store').tail(1)[['Store', 'CPI', 'Unemployment', 'Fuel_Price']]

    df_historic_copy = df_historic.copy()
    df_historic_copy['Month'] = pd.to_datetime(df_historic_copy['Date']).dt.month
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df_historic_copy[markdown_cols] = df_historic_copy[markdown_cols].fillna(0)
    
    seasonal_cols = ['Temperature'] + markdown_cols
    seasonal_stats = df_historic_copy.groupby(['Store', 'Month'])[seasonal_cols].mean().reset_index()

    start_datetime = pd.to_datetime(start_date)
    next_dates = pd.date_range(start=start_datetime + timedelta(days=7), periods=weeks_ahead, freq='W-FRI')
    df_dates = pd.DataFrame({'Date': next_dates})
    
    df_future = unique_combos.merge(df_dates, how='cross')
    
    df_future['Year'] = df_future['Date'].dt.year
    df_future['Month'] = df_future['Date'].dt.month
    df_future['WeekOfYear'] = df_future['Date'].dt.isocalendar().week
    df_future['IsHoliday'] = df_future['WeekOfYear'].isin([6, 36, 47, 52])
    
    df_future = df_future.merge(latest_macro, on='Store', how='left')
    df_future = df_future.merge(seasonal_stats, on=['Store', 'Month'], how='left')
    
    df_future.fillna(0, inplace=True)
    
    df_transformer = df_future.drop(columns=['Date']) if 'Date' not in transformer.feature_names_in_ else df_future
    
    X_processed = transformer.transform(df_transformer)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()
        
    preds_real = scaler_y.inverse_transform(preds_scaled)
    df_future['Projected_Sales'] = preds_real
    
    return df_future