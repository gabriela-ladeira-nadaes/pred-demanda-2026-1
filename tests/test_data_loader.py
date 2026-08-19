"""Testes de carregamento dos dados."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TEST_DIR))

from data.data_loader import load_data, validate_data  # noqa: E402
from helpers import make_sample_dataframe  # noqa: E402


class TestDataLoading(unittest.TestCase):
    def test_load_data_reads_csv(self):
        df = make_sample_dataframe()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            df.to_csv(path, index=False)
            loaded = load_data(path)

        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(len(loaded), len(df))
        self.assertEqual(list(loaded.columns), list(df.columns))

    def test_load_data_real_dataset(self):
        from utils.config import DATA_PATH

        if not Path(DATA_PATH).exists():
            self.skipTest("Dataset real não encontrado.")

        data = load_data(DATA_PATH)
        self.assertFalse(data.empty)
        self.assertIn("Weekly_Sales", data.columns)
        self.assertGreater(len(data), 0)

    def test_validate_data_empty(self):
        self.assertFalse(validate_data(pd.DataFrame()))

    def test_validate_data_non_empty(self):
        self.assertTrue(validate_data(make_sample_dataframe()))


if __name__ == "__main__":
    unittest.main()
