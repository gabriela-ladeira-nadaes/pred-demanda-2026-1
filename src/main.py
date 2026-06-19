import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data.data_loader import load_data, validate_data
from preprocessing.transform import clean_data, split_data
from training.train import train_model
from evaluation.metrics import evaluate_metrics
from utils.config import DATA_PATH, TARGET_COLUMN

def main() -> None:
    data: pd.DataFrame = load_data(DATA_PATH)
    if not validate_data(data):
        print("Base de dados nao validadA")
        return
        
    data = clean_data(data)
    
    X_train, X_test, y_train, y_test = split_data(data, TARGET_COLUMN)

    model: RandomForestRegressor = train_model(X_train, y_train)

    metrics: dict[str, float] = evaluate_metrics(model, X_test, y_test)

    print("Base de dados avaliada com sucesso")
    print(metrics)

if __name__ == "__main__":
    main()
