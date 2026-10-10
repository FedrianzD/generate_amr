"""Check that imported GCNConv preserves the previous weighted GCN computation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(graph: bool) -> None:
    raise RuntimeError("Archived checker requires the retired PyG CLI; use the maintained notebook instead.")
    legacy = load_module("dense_legacy", HERE / "train_gcn_dense.py").SentenceScorer(graph=graph)
    pyg = load_module("pyg_model", HERE / "train_gcn_dense.py").SentenceScorer(graph=graph)
    with torch.no_grad():
        pyg.conv1.lin.weight.copy_(legacy.fc1.weight)
        pyg.conv1.bias.copy_(legacy.fc1.bias)
        pyg.conv2.lin.weight.copy_(legacy.fc2.weight)
        pyg.conv2.bias.copy_(legacy.fc2.bias)
        pyg.head.weight.copy_(legacy.head.weight)
        pyg.head.bias.copy_(legacy.head.bias)
    legacy.eval()
    pyg.eval()
    torch.manual_seed(7)
    x = torch.randn(8, 768)
    a = torch.rand(8, 8)
    a = (a + a.T) / 2
    a.fill_diagonal_(0)
    a[a < 0.25] = 0
    first = legacy(x, a)
    second = pyg(x, a)
    torch.testing.assert_close(second, first, rtol=1e-5, atol=1e-5)
    print(f"{'GCN' if graph else 'baseline'} max difference: {(second - first).abs().max().item():.8f}")


if __name__ == "__main__":
    check(graph=True)
    check(graph=False)
