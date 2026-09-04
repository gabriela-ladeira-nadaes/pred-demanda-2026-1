import pandas as pd
from inference.predictor import load_latest_model_and_scalers, generate_forecast
from inference.visualization import plot_financial_projection, plot_financial_acumulate_projection
from data.data_loader import load_data
from utils.config import DATA_PATH


def run_predictions():
    """Roda a predição e faz a exbição"""    
    df_historic = load_data(DATA_PATH)
    latest_date = df_historic['Date'].max()
   
    models_path = "models"
    model, scaler_y, transformer = load_latest_model_and_scalers(models_path)
        
    df_future = generate_forecast(
        model=model, 
        scaler_y=scaler_y, 
        transformer=transformer, 
        store_id=None, 
        dept_id=None,
        start_date=latest_date,
        df_historic=df_historic,
        weeks_ahead=52
    )
    
    plot_financial_projection(
        df_historic, 
        df_future, 
        store_id=None, 
        dept_id=None,
    )

    plot_financial_acumulate_projection(
            df_historic, 
            df_future, 
            store_id=None, 
            dept_id=None,
        )

if __name__ == "__main__":
    run_predictions()