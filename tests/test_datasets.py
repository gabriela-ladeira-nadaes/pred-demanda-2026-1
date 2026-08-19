"""Testes de shape dos tensores, dataset e dataloader."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data.datasets import get_device, make_dataloader, make_dataset, to_tensors  # noqa: E402


class TestTensorShapes(unittest.TestCase):
    def test_to_tensors_dtype_and_shape(self):
        X = np.random.rand(100, 12).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)

        Xt, yt = to_tensors(X, y)

        self.assertIsInstance(Xt, torch.Tensor)
        self.assertIsInstance(yt, torch.Tensor)
        self.assertEqual(Xt.dtype, torch.float32)
        self.assertEqual(yt.dtype, torch.float32)
        self.assertEqual(tuple(Xt.shape), (100, 12))
        # Regra do sistema: o alvo é remodelado para (N, 1).
        self.assertEqual(tuple(yt.shape), (100, 1))

    def test_make_dataset(self):
        Xt = torch.randn(100, 12)
        yt = torch.randn(100, 1)

        ds = make_dataset(Xt, yt)

        self.assertIsInstance(ds, TensorDataset)
        self.assertEqual(len(ds), 100)
        x0, y0 = ds[0]
        self.assertEqual(tuple(x0.shape), (12,))
        self.assertEqual(tuple(y0.shape), (1,))

    def test_make_dataloader_batch_shapes(self):
        Xt = torch.randn(100, 12)
        yt = torch.randn(100, 1)
        ds = make_dataset(Xt, yt)

        loader = make_dataloader(ds, batch_size=32, shuffle=False)

        self.assertIsInstance(loader, DataLoader)
        batches = list(loader)
        self.assertEqual(len(batches), 4)  # ceil(100 / 32)

        for bx, by in batches[:-1]:
            self.assertEqual(tuple(bx.shape), (32, 12))
            self.assertEqual(tuple(by.shape), (32, 1))

        last_x, last_y = batches[-1]
        self.assertEqual(tuple(last_x.shape), (4, 12))
        self.assertEqual(tuple(last_y.shape), (4, 1))

    def test_get_device(self):
        device = get_device()

        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
