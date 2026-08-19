"""Fixtures e utilitários compartilhados pelos testes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Caminho de `src/` para que os pacotes (data, preprocessing, ...) sejam importáveis.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def setup_src_path() -> None:
    """Garante que os pacotes de `src/` estejam no sys.path."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def make_sample_dataframe(n_rows: int = 120) -> pd.DataFrame:
    """Gera um DataFrame sintético com o mesmo schema do dataset do Walmart.

    As datas cobrem 2010 a 2012 (frequência semanal), cruzando a data de corte
    (2012-01-01) para que treino e teste fiquem ambos não vazios e o ano varie
    dentro do treino (necessário para a padronização do atributo Year).
    """
    rng = np.random.default_rng(42)

    dates = pd.date_range(start="2010-01-08", periods=n_rows, freq="W-FRI")

    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n_rows),
            "Store": rng.integers(1, 4, n_rows),
            "Date": dates.strftime("%Y-%m-%d"),
            "IsHoliday": rng.integers(0, 2, n_rows),
            "Dept": rng.integers(1, 5, n_rows).astype(float),
            "Weekly_Sales": rng.normal(15_000.0, 5_000.0, n_rows),
            "Temperature": rng.normal(50.0, 15.0, n_rows),
            "Fuel_Price": rng.normal(3.0, 0.5, n_rows),
            "MarkDown1": rng.normal(0.0, 1_000.0, n_rows),
            "MarkDown2": rng.normal(0.0, 1_000.0, n_rows),
            "MarkDown3": rng.normal(0.0, 1_000.0, n_rows),
            "MarkDown4": rng.normal(0.0, 1_000.0, n_rows),
            "MarkDown5": rng.normal(0.0, 1_000.0, n_rows),
            "CPI": rng.normal(200.0, 10.0, n_rows),
            "Unemployment": rng.normal(8.0, 1.0, n_rows),
            "Type": rng.integers(1, 4, n_rows),
            "Size": rng.integers(50_000, 200_000, n_rows).astype(float),
        }
    )
    return df
