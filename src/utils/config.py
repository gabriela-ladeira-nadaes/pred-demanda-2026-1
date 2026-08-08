from pathlib import Path
from datetime import datetime

# Caminhos
BASE_DIR: Path = Path(__file__).resolve().parents[2] # raiz do projeto 3 níveis acima desse arquivo
DATA_PATH: Path = BASE_DIR/"data"/"walmart_dataset_sales.csv" # caminho do dataset

# Parametros do experimento
FIXED_SEED: int = 42
TEST_SIZE: float = 0.2
BATCH_SIZE: int = 64
EPOCHS: int = 50
CUTOFF_DATE: datetime = '2012-01-01'

# Coluna alvo
TARGET_COLUMN: str = "Weekly_Sales"

# Coluna temporal
DATE_COLUMN: str = "Date"
SORT_COLUMNS: list[str] = ['Store', 'Dept', 'Date']

# Features categoricos
CATEGORICAL_FEATURES: list[str] = ["Store", "Dept", "Type", "IsHoliday"]

CATEGORICAL_FEATURES_WITHOUT_BOOLEAN : list[str] = ["Store", "Dept", "Type"]

# Features numericos
NUMERIC_FEATURES: list[str] = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Features temporais numericos derivados da coluna Date em no pre-processamento
TEMPORAL_FEATURES: list[str] = ["Year", "Month", "WeekOfYear"]
TEMPORAL_FEATURES_CATEGORICAL: list[str] = ["Month", "WeekOfYear"]
TEMPORAL_FEATURE_YEAR: list[str] = ["Year"]

