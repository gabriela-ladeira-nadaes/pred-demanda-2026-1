from pathlib import Path
from data.data_loader import load_data, validate_data
from preprocessing.transform import clean_data, split_data
from training.train import train_model
from evaluation.metrics import evaluate_metrics
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

DATA_PATH = Path("data/walmart_dataset_sales.csv")

def main() -> None:
    data: pd.DataFrame = load_data(DATA_PATH)
    if not validate_data(data):
        print("Base de dados nao validado")
        return
        
    data = clean_data(data)
    
    X_train, X_test, y_train, y_test = split_data(data, "Weekly_Sales")

    trained : RandomForestRegressor = train_model(X_train, y_train)

    evaluated : pd.Dataframe = evaluate_metrics(trained, X_test, y_test)

    print("Base de dados avaliada com sucesso")

if __name__ == "__main__":
    main()
