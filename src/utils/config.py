from pathlib import Path

# Caminhos
BASE_DIR: Path = Path(__file__).resolve().parents[2] # raiz do projeto 3 níveis acima desse arquivo
DATA_PATH: Path = BASE_DIR/"data"/"walmart_dataset_sales.csv" # caminho do dataset

# Parametros do experimento
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
BATCH_SIZE: int = 64

# Coluna alvo
TARGET_COLUMN: str = "Weekly_Sales"

# Coluna temporal
DATE_COLUMN: str = "Date"

# Features categoricos
CATEGORICAL_FEATURES: list[str] = ["Store", "Dept", "Type", "IsHoliday"]

# Features numericos
NUMERIC_FEATURES: list[str] = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]

# Features temporais numericos derivados da coluna Date em no pre-processamento
TEMPORAL_FEATURES: list[str] = ["Year", "Month", "WeekOfYear"]

