"""Testes de regras específicas do sistema (invariantes de negócio)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TEST_DIR))

from preprocessing.features import create_features  # noqa: E402
from preprocessing.transform import clean_data, split_data, standardize  # noqa: E402
from utils.config import CUTOFF_DATE, DATE_COLUMN, NUMERIC_FEATURES  # noqa: E402
from helpers import make_sample_dataframe  # noqa: E402


class TestSystemRules(unittest.TestCase):
    def setUp(self):
        self.df = create_features(clean_data(make_sample_dataframe()))

    def test_split_is_chronological(self):
        """Treino = passado (Date < corte) e teste = futuro (Date >= corte)."""
        X_train, _, X_test, _ = split_data(self.df)

        cutoff = pd.to_datetime(CUTOFF_DATE)
        n_train = int((self.df[DATE_COLUMN] < cutoff).sum())
        n_test = int((self.df[DATE_COLUMN] >= cutoff).sum())

        self.assertEqual(X_train.shape[0], n_train)
        self.assertEqual(X_test.shape[0], n_test)
        self.assertEqual(n_train + n_test, len(self.df))
        self.assertGreater(n_train, 0)
        self.assertGreater(n_test, 0)

    def test_standardize_fits_only_on_train(self):
        """A padronização não pode vazar informação do teste para o treino."""
        X_train, _, X_test, _ = split_data(self.df)

        train_a, test_a = standardize(X_train, X_test)

        # Perturba apenas as colunas numéricas do teste (sem criar categorias
        # desconhecidas no OneHotEncoder).
        X_test_alt = X_test.copy()
        X_test_alt[:, : len(NUMERIC_FEATURES)] *= 2.0
        train_b, test_b = standardize(X_train, X_test_alt)

        # A transformação do treino é idêntica independente do teste.
        np.testing.assert_array_equal(train_a, train_b)
        # Mas a transformação do teste muda quando o teste muda.
        self.assertFalse(np.array_equal(test_a, test_b))


if __name__ == "__main__":
    unittest.main()
