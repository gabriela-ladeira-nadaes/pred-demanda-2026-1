"""Testes de salvamento e carregamento dos pesos do modelo."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.model import load_model, save_model  # noqa: E402
from models.model import FinancialModel  # noqa: E402


class TestSaveLoad(unittest.TestCase):
    def test_save_model_passes_state_dict_and_filename(self):
        model = FinancialModel(input_dim=4, output_dim=1)
        scaler_y = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
        standardizer = ColumnTransformer(
            [("num", StandardScaler(), [0])]
        ).fit(np.array([[1.0], [2.0], [3.0]]))

        with mock.patch("models.model.torch.save") as mock_save, \
                mock.patch("models.model.joblib.dump") as mock_dump:
            save_model(model, scaler_y, standardizer)

        mock_save.assert_called_once()
        self.assertEqual(mock_dump.call_count, 2)
        kwargs = mock_save.call_args.kwargs

        # O nome segue o padrão pytorch_workflow_model_<timestamp>.pth.
        self.assertIn("f", kwargs)
        self.assertRegex(str(kwargs["f"]), r"pytorch_workflow_model_\d{8}_\d{6}\.pth$")

        # O checkpoint carrega a arquitetura junto dos pesos, para que o
        # predictor saiba qual classe reinstanciar ao carregar.
        saved = kwargs["obj"]
        self.assertEqual(set(saved.keys()), {"name", "weights"})
        self.assertEqual(saved["name"], "FinancialModel")
        self.assertEqual(set(saved["weights"].keys()), set(model.state_dict().keys()))

    def test_state_dict_roundtrip(self):
        model = FinancialModel(input_dim=4, output_dim=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(model.state_dict(), path)
            loaded = torch.load(path, weights_only=True)

        restored = FinancialModel(input_dim=4, output_dim=1)
        restored.load_state_dict(loaded)

        for p1, p2 in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(p1, p2))


    def test_load_model_restores_weights(self):
        model = FinancialModel(input_dim=4, output_dim=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save({"name": "FinancialModel", "weights": model.state_dict()}, path)

            restored = load_model(FinancialModel(input_dim=4, output_dim=1), path)

        for p1, p2 in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main()
