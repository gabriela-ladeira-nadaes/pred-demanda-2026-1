from pathlib import pathlib

BASE_DIR: Path = Path(__file__).resolve().parents[2] # raiz do projeto 3 níveis acima desse arquivo
DATA_PATH: Path = BASE_DIR/"data"/"walmart_dataset_sales.csv" # caminho do dataset
TARGET_COLUMN: str = "Weekly_Sales" # coluna alvo da predição

# parametros do experimento
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2