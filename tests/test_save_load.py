"""Testes de salvamento e carregamento dos pesos do modelo."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data.data_loader import save_model  # noqa: E402
from training.train import FinnancialModel  # noqa: E402


class TestSaveLoad(unittest.TestCase):
    def test_save_model_passes_state_dict_and_filename(self):
        model = FinnancialModel(input_dim=4, output_dim=1)

        with mock.patch("data.data_loader.torch.save") as mock_save:
            save_model(model)

        mock_save.assert_called_once()
        kwargs = mock_save.call_args.kwargs

        # O nome segue o padrão pytorch_workflow_model_<timestamp>.pth.
        self.assertIn("f", kwargs)
        self.assertRegex(str(kwargs["f"]), r"pytorch_workflow_model_\d{8}_\d{6}\.pth$")

        # O objeto salvo é o state_dict completo do modelo.
        saved = kwargs["obj"]
        self.assertEqual(set(saved.keys()), set(model.state_dict().keys()))

    def test_state_dict_roundtrip(self):
        model = FinnancialModel(input_dim=4, output_dim=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(model.state_dict(), path)
            loaded = torch.load(path, weights_only=True)

        restored = FinnancialModel(input_dim=4, output_dim=1)
        restored.load_state_dict(loaded)

        for p1, p2 in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
