"""Testes de pré-processamento (limpeza, features, split e padronização)."""
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
from preprocessing.transform import (  # noqa: E402
    build_features_matrix,
    clean_data,
    describe_array,
    split_data,
    standardize,
)
from utils.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    NUMERIC_FEATURES,
    SORT_COLUMNS,
    TARGET_COLUMN,
    TEMPORAL_FEATURES,
)
from helpers import make_sample_dataframe  # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.raw = make_sample_dataframe()

    def _prepared(self) -> pd.DataFrame:
        return create_features(clean_data(self.raw))

    def test_clean_data_drops_unnamed_and_sorts(self):
        shuffled = self.raw.sample(frac=1.0, random_state=1).reset_index(drop=True)
        out = clean_data(shuffled)

        self.assertNotIn("Unnamed: 0", out.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out[DATE_COLUMN]))

        # Ordenado por Store, Dept e Date e reindexado de 0..n-1.
        expected = out.sort_values(SORT_COLUMNS).reset_index(drop=True)
        pd.testing.assert_frame_equal(out, expected)
        self.assertTrue((out.index == np.arange(len(out))).all())

    def test_create_features_adds_temporal(self):
        cleaned = clean_data(self.raw)
        out = create_features(cleaned)

        for col in TEMPORAL_FEATURES:
            self.assertIn(col, out.columns)

        first_date = cleaned[DATE_COLUMN].iloc[0]
        self.assertEqual(int(out["Year"].iloc[0]), first_date.year)
        self.assertEqual(int(out["Month"].iloc[0]), first_date.month)
        self.assertEqual(int(out["WeekOfYear"].iloc[0]), first_date.isocalendar().week)

    def test_build_features_matrix_shape_and_target(self):
        df = self._prepared()
        X, y = build_features_matrix(df)

        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(TEMPORAL_FEATURES)
        self.assertEqual(X.shape, (len(df), expected_cols))
        self.assertEqual(y.shape, (len(df),))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

        # A primeira coluna segue a ordem fixa NUMERIC -> CATEGORICAL -> TEMPORAL.
        self.assertEqual(NUMERIC_FEATURES[0], "Temperature")
        np.testing.assert_allclose(y, df[TARGET_COLUMN].to_numpy(dtype=np.float32))

    def test_split_data_shapes(self):
        df = self._prepared()
        X_train, y_train, X_test, y_test = split_data(df)

        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(df))
        self.assertEqual(y_train.shape[0], X_train.shape[0])
        self.assertEqual(y_test.shape[0], X_test.shape[0])
        self.assertGreater(X_train.shape[0], 0)
        self.assertGreater(X_test.shape[0], 0)

    def test_standardize_shape(self):
        df = self._prepared()
        X_train, _, X_test, _ = split_data(df)
        Xp_train, Xp_test = standardize(X_train, X_test)

        self.assertEqual(Xp_train.shape[0], X_train.shape[0])
        self.assertEqual(Xp_test.shape[0], X_test.shape[0])
        self.assertEqual(Xp_train.shape[1], Xp_test.shape[1])
        # O one-hot encoding expande a dimensionalidade em relação ao array bruto.
        self.assertGreater(Xp_train.shape[1], X_train.shape[1])

    def test_standardize_mean(self):
        df = self._prepared()
        X_train, _, X_test, _ = split_data(df)
        Xp, _ = standardize(X_train, X_test)

        # Colunas "binárias" (one-hot + IsHoliday) ficam em {0, 1}; as demais
        # são as colunas numéricas padronizadas, que devem ter média 0 e std 1.
        binary = np.all(np.isin(Xp, [0.0, 1.0]), axis=0)
        numeric_cols = Xp[:, ~binary]

        self.assertGreater(numeric_cols.shape[1], 0)
        np.testing.assert_allclose(numeric_cols.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(numeric_cols.std(axis=0), 1.0, atol=1e-6)

    def test_standardize_binary_columns_are_onehot(self):
        df = self._prepared()
        X_train, _, X_test, _ = split_data(df)
        Xp, _ = standardize(X_train, X_test)

        binary = np.all(np.isin(Xp, [0.0, 1.0]), axis=0)
        self.assertGreater(int(binary.sum()), 0)
        self.assertTrue(np.isin(Xp[:, binary], [0.0, 1.0]).all())

    def test_describe_array(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        info = describe_array(a)

        self.assertEqual(info["shape"], (2, 2))
        self.assertAlmostEqual(info["mean"], 2.5)
        self.assertAlmostEqual(info["min"], 1.0)
        self.assertAlmostEqual(info["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
