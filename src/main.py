from pathlib import Path
from data.data_loader import load_data, validate_data
from preprocessing.transform import clean_data, split_data
import pandas as pd

DATA_PATH = Path("data/walmart_dataset_sales.csv")

def main() -> None:
    data: pd.DataFrame = load_data(DATA_PATH)
    if not validate_data(data):
        print("Base de daos nao validade")
        return
        
    data = clean_data(data)
    
    X_treino, X_teste, y_treino, y_teste = split_data(data)

if __name__ == "__main__":
    main()
