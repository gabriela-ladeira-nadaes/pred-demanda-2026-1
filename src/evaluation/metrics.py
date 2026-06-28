import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_metrics(model: nn.Module, test_loader: DataLoader) -> dict[str, float]:
    """Avalia o modelo no conjunto de teste. Implementacao nas proximas etapas."""
    return {}
