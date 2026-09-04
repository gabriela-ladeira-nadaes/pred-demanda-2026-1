"""Testes da saída (forward) dos modelos."""
import sys
import unittest
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.model import FinancialModel, LinearRegression  # noqa: E402


class TestModelOutput(unittest.TestCase):
    def test_financial_model_output_shape(self):
        model = FinancialModel(input_dim=10, output_dim=1)
        out = model(torch.randn(8, 10))

        self.assertEqual(tuple(out.shape), (8, 1))
        self.assertEqual(out.dtype, torch.float32)

    def test_financial_model_output_dim(self):
        model = FinancialModel(input_dim=10, output_dim=3)
        out = model(torch.randn(8, 10))

        self.assertEqual(tuple(out.shape), (8, 3))

    def test_linear_regression_output_shape(self):
        model = LinearRegression(input_dim=10, output_dim=1)
        out = model(torch.randn(8, 10))

        self.assertEqual(tuple(out.shape), (8, 1))

    def test_model_output_is_finite(self):
        model = FinancialModel(input_dim=10, output_dim=1)
        out = model(torch.randn(8, 10))

        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
