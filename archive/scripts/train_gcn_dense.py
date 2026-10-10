"""Pilot extractive summarization on Liputan6 dev features and SMATCH graphs.

The original dev split is divided by article into pilot train/validation sets.
The original test split is never used for optimization. PyTorch .pt files must
come from a trusted source: torch.load deserializes a pickle container.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "liputan6"
SOURCE = DATA / "source/Liputan6/liputan6/liputan6_data/canonical"


def load_features(path: Path) -> dict[str, torch.Tensor]:
    # weights_only blocks arbitrary pickle globals; the supplied files contain
    # only dictionaries, strings, and plain tensors.
    records = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a list of feature records")
    result = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"id", "embeddings"}:
            raise ValueError(f"{path}: expected records with id and embeddings")
        doc_id = str(record["id"])
        if doc_id in result:
            raise ValueError(f"{path}: duplicate article {doc_id}")
        value = record["embeddings"]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 768:
            raise ValueError(f"{path}: article {doc_id} has invalid embedding shape")
        if not torch.isfinite(value).all():
            raise ValueError(f"{path}: article {doc_id} has nonfinite embeddings")
        result[doc_id] = value.float()
    return result


def load_split(features: Path, split: str, adjacency_root: Path, source_root: Path, max_documents: int | None = None):
    print(f"Loading {split} features from {features}...", flush=True)
    vectors = load_features(features)
    examples = []
    errors = []
    records = list(vectors.items())[:max_documents]
    for doc_id, x in tqdm(records, desc=f"Checking {split} articles", unit="article"):
        adjacency_path = adjacency_root / split / f"{doc_id}.npy"
        source_path = source_root / split / f"{doc_id}.json"
        try:
            if not adjacency_path.is_file() or not source_path.is_file():
                raise ValueError("missing adjacency or source JSON")
            a = np.load(adjacency_path, allow_pickle=False)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            n = len(source["clean_article"])
            gold = source["extractive_summary"]
            if x.shape[0] != n or a.shape != (n, n):
                raise ValueError(f"row mismatch: features={x.shape[0]}, adjacency={a.shape}, source={n}")
            if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-5):
                raise ValueError("adjacency is nonfinite or asymmetric")
            if np.min(a) < -1e-5 or np.max(a) > 1 + 1e-5:
                raise ValueError("SMATCH adjacency outside [0, 1]")
            if not isinstance(gold, list) or any(type(i) is not int or i < 0 or i >= n for i in gold):
                raise ValueError("invalid extractive_summary indices")
            if n == 0:
                raise ValueError("empty article")
            y = torch.zeros(n, dtype=torch.float32)
            y[gold] = 1
            examples.append((doc_id, x, torch.from_numpy(a.astype(np.float32)), y))
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            errors.append((doc_id, str(exc)))
    return examples, errors


class SentenceScorer(nn.Module):
    def __init__(self, input_dim: int = 768, hidden: int = 256, graph: bool = True):
        super().__init__()
        self.graph = graph
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if self.graph:
            a = adjacency + torch.eye(n, device=x.device)
            degree = a.sum(dim=1).clamp_min(1e-8).rsqrt()
            a = degree[:, None] * a * degree[None, :]
        else:
            a = torch.eye(n, device=x.device)
        h = F.relu(self.fc1(a @ x))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(a @ h))
        return self.head(h).squeeze(-1)


@torch.no_grad()
def evaluate(model, examples, device, top_k):
    model.eval()
    tp = predicted = actual = 0
    loss_sum = 0.0
    predictions = []
    for doc_id, x, a, y in examples:
        logits = model(x.to(device), a.to(device))
        loss_sum += F.binary_cross_entropy_with_logits(logits, y.to(device)).item()
        chosen = torch.topk(logits, min(top_k, len(y))).indices.cpu().tolist()
        gold = set(torch.nonzero(y).flatten().tolist())
        tp += len(set(chosen) & gold)
        predicted += len(chosen)
        actual += len(gold)
        predictions.append({"id": doc_id, "selected_sentences": sorted(chosen), "gold_sentences": sorted(gold)})
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    return {"loss": loss_sum / len(examples), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}, predictions


def evaluate_saved_checkpoint(args, device):
    checkpoint_path = args.checkpoint or args.output / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No saved checkpoint at {checkpoint_path}; train first")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"state_dict", "baseline", "top_k"} <= checkpoint.keys():
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    model = SentenceScorer(graph=not checkpoint["baseline"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test, errors = load_split(args.test_features, "test", args.adjacency_root, args.source_root, args.max_documents)
    print(f"test: {len(test)} aligned, {len(errors)} rejected")
    if errors or not test:
        raise ValueError(f"test alignment failed: first 10 errors: {errors[:10]}")
    metrics, predictions = evaluate(model, test, device, checkpoint["top_k"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "test_predictions.jsonl").open("w", encoding="utf-8") as out:
        for row in predictions:
            out.write(json.dumps(row) + "\n")
    print("test:", json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-features", type=Path, default=DATA / "features/dev.pt")
    parser.add_argument("--test-features", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Saved model for test evaluation; defaults to OUTPUT/best.pt")
    parser.add_argument("--adjacency-root", type=Path, default=DATA / "adjacency_matrices")
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gcn-pilot")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=None, help="Small smoke run; never use its scores as final results")
    parser.add_argument("--validate-only", action="store_true", help="Audit feature/adjacency/source alignment without training")
    parser.add_argument("--baseline", action="store_true", help="Use embeddings without graph propagation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.epochs < 1 or args.top_k < 1:
        parser.error("validation-fraction must be in (0,1); epochs and top-k must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.test_features and not args.validate_only:
        evaluate_saved_checkpoint(args, device)
        return
    examples, errors = load_split(args.dev_features, "dev", args.adjacency_root, args.source_root, args.max_documents)
    print(f"dev: {len(examples)} aligned, {len(errors)} rejected")
    if errors:
        print(f"Rejected {len(errors)} mismatched dev articles; first 10: {errors[:10]}")
    if args.validate_only:
        if args.test_features:
            test, test_errors = load_split(args.test_features, "test", args.adjacency_root, args.source_root, args.max_documents)
            print(f"test: {len(test)} aligned, {len(test_errors)} rejected; first 10: {test_errors[:10]}")
            if test_errors:
                raise ValueError("test alignment failed")
        if errors:
            raise ValueError("dev alignment failed")
        return
    if len(examples) < 2:
        parser.error("fewer than two aligned dev articles")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    cutoff = max(1, min(len(examples) - 1, round(len(examples) * (1 - args.validation_fraction))))
    train, validation = examples[:cutoff], examples[cutoff:]
    model = SentenceScorer(graph=not args.baseline).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        total_loss = 0.0
        progress = tqdm(train, desc=f"Epoch {epoch}/{args.epochs}", unit="article")
        for article_index, (_, x, a, y) in enumerate(progress, start=1):
            optimizer.zero_grad()
            logits = model(x.to(device), a.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if article_index % 10 == 0 or article_index == len(train):
                progress.set_postfix(loss=f"{total_loss / article_index:.4f}")
        metrics, _ = evaluate(model, validation, device, args.top_k)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), **metrics})
        print(json.dumps(history[-1]))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"state_dict": model.state_dict(), "baseline": args.baseline,
                        "seed": args.seed, "top_k": args.top_k, "validation_f1": best_f1}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps({"seed": args.seed,
        "train_ids": [v[0] for v in train], "validation_ids": [v[0] for v in validation],
        "rejected": errors}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
