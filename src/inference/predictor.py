import os
import torch
import joblib
import pandas as pd

from pathlib import Path
from datetime import timedelta
from models.model import FinnancialModel, LinearRegression
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
        "FinnancialModel": FinnancialModel(dynamic_input_dim, 1),
        "LinearRegression": LinearRegression(dynamic_input_dim, 1)
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

def generate_forecast(model: torch.nn.Module, scaler_y: StandardScaler, transformer: ColumnTransformer, store_id: int, dept_id: int, start_date: str, df_historic: pd.DataFrame, weeks_ahead: int = 26) -> pd.DataFrame:
    """Gera um DataFrame com projeções de vendas para as próximas semanas."""
    
    # Extrai o tamanho e o tipo da loja baseando-se no histórico real
    store_info = df_historic[df_historic['Store'] == store_id].iloc[0]
    store_size = store_info['Size']
    store_type = store_info['Type']
        
    start_datetime = pd.to_datetime(start_date)
    next_dates = pd.date_range(start=start_datetime + timedelta(days=7), periods=weeks_ahead, freq='W-FRI')

    df_future = pd.DataFrame({
        'Store': store_id,
        'Dept': dept_id,
        'Date': next_dates,
        'Type': store_type,      
        'Size': store_size,  
        'Temperature': 60.0, 
        'Fuel_Price': 3.5,
        'CPI': 220.0,
        'Unemployment': 7.0,
        'MarkDown1': 0.0, 
        'MarkDown2': 0.0, 
        'MarkDown3': 0.0, 
        'MarkDown4': 0.0, 
        'MarkDown5': 0.0
    })
   
    df_future['Year'] = df_future['Date'].dt.year
    df_future['Month'] = df_future['Date'].dt.month
    
    # Renomeado de 'Week' para 'WeekOfYear' para bater com o padrão de treino
    df_future['WeekOfYear'] = df_future['Date'].dt.isocalendar().week
    df_future['IsHoliday'] = df_future['WeekOfYear'].isin([6, 36, 47, 52])
    
    # Remove a coluna Date apenas se ela não foi usada no fit do ColumnTransformer
    df_transformer = df_future.drop(columns=['Date']) if 'Date' not in transformer.feature_names_in_ else df_future
    
    # Pipeline de Inferência
    X_processed = transformer.transform(df_transformer)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()
        
    preds_real = scaler_y.inverse_transform(preds_scaled)
   
    df_future['Projected_Sales'] = preds_real
    return df_future