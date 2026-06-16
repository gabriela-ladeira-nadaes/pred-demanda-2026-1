<<<<<<< HEAD
from pathlib import Path
from data_loader import load_data
from transform import clean_data, split_data

DATA_PATH = Path("data/walmart_dataset_sales.csv")

def main() -> None:
    data = load_data(DATA_PATH)
    data = clean_data(data)
    
    X_treino, X_teste, y_treino, y_teste = split_data(dados)

if __name__ == "__main__":
    main()