"""Generate the self-contained PyG training notebook from shared script definitions."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "archive/scripts/train_gcn_dense.py"
OUTPUT = ROOT / "notebooks/liputan6/train_gcn.ipynb"


def definition(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            start = min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1
            return "\n".join(source.splitlines()[start:node.end_lineno]) + "\n"
    raise ValueError(f"Missing definition: {name}")


def main() -> None:
    raise RuntimeError(
        "Archived generator: its source uses the dense model, while the "
        "maintained notebook uses PyG. Do not overwrite the notebook with this generator."
    )
    source = SCRIPT.read_text(encoding="utf-8")
    cells = [
        nbf.v4.new_markdown_cell(textwrap.dedent("""\
            # Liputan6 sentence selection with PyG GCNConv

            This notebook trains a **new** two-layer GCN on an article-level split of
            Liputan6 `dev`. Each sentence is a node. Its 768-dimensional feature row
            comes from `dev.pt`; SMATCH scores in `{doc_id}.npy` become weighted edges.
            `extractive_summary` supplies training labels. The original `test` split
            is evaluated only when the final optional cell is enabled.

            Run with the **generate_amr** Conda kernel. The notebook uses the same
            model/data definitions as the historical dense CLI, but has
            independent configuration, training, and evaluation cells. It writes to
            `runs/pyg-gcn-notebook/` so the earlier dense-model results stay intact.

            The previous `runs/gcn-pilot/best.pt` belongs to the dense implementation
            and cannot be loaded into this PyG model. Its original code is preserved
            as `archive/scripts/train_gcn_dense.py`.
        """)),
        nbf.v4.new_code_cell(textwrap.dedent("""\
            import json
            import random
            from pathlib import Path

            import numpy as np
            import torch
            from torch import nn
            from torch.nn import functional as F
            from torch_geometric.nn import GCNConv
            from tqdm.auto import tqdm

            ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                         if (p / "AGENTS.md").is_file()
                         and (p / "notebooks/liputan6/train_gcn.ipynb").is_file()), None)
            if ROOT is None:
                raise RuntimeError("Open this notebook from within the generate_amr project")
            DATA = ROOT / "data/liputan6"
            SOURCE = DATA / "source/Liputan6/liputan6/liputan6_data/canonical"
            print("Project:", ROOT)
            print("PyTorch:", torch.__version__, "PyG GCNConv:", GCNConv.__name__)
        """)),
        nbf.v4.new_markdown_cell("## Data loading and alignment"),
        nbf.v4.new_code_cell(definition(source, "load_features") + "\n" + definition(source, "load_split")),
        nbf.v4.new_markdown_cell("## Model: weighted SMATCH edges → two GCNConv layers → sentence scores"),
        nbf.v4.new_code_cell(definition(source, "SentenceScorer")),
        nbf.v4.new_code_cell(definition(source, "evaluate")),
        nbf.v4.new_markdown_cell("## Configure the pilot run"),
        nbf.v4.new_code_cell(textwrap.dedent("""\
            DEV_FEATURES = DATA / "features/dev.pt"
            TEST_FEATURES = DATA / "features/test.pt"
            ADJACENCY_ROOT = DATA / "adjacency_matrices"
            OUTPUT = ROOT / "runs/pyg-gcn-notebook"
            EPOCHS = 15
            VALIDATION_FRACTION = 0.20
            TOP_K = 2  # Pilot choice; the meeting notes do not specify k=2.
            SEED = 42
            MAX_DOCUMENTS = None  # Set to 20 for a smoke run only.
            BASELINE = False  # True uses only self-loops (embeddings without graph edges).
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print("Device:", DEVICE)
        """)),
        nbf.v4.new_markdown_cell("## Load dev articles and split by article ID"),
        nbf.v4.new_code_cell(textwrap.dedent("""\
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            examples, errors = load_split(DEV_FEATURES, "dev", ADJACENCY_ROOT, SOURCE, MAX_DOCUMENTS)
            print(f"dev: {len(examples)} aligned, {len(errors)} rejected")
            if errors:
                print("First errors:", errors[:10])
            if len(examples) < 2:
                raise ValueError("Need at least two aligned articles")
            rng = random.Random(SEED)
            rng.shuffle(examples)
            cutoff = max(1, min(len(examples) - 1, round(len(examples) * (1 - VALIDATION_FRACTION))))
            train, validation = examples[:cutoff], examples[cutoff:]
            print(f"train={len(train)}, validation={len(validation)}")
        """)),
        nbf.v4.new_markdown_cell("## Train and select checkpoint on validation sentence F1"),
        nbf.v4.new_code_cell(textwrap.dedent("""\
            model = SentenceScorer(graph=not BASELINE).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            best_f1 = -1.0
            history = []
            for epoch in range(1, EPOCHS + 1):
                model.train()
                rng.shuffle(train)
                total_loss = 0.0
                progress = tqdm(train, desc=f"Epoch {epoch}/{EPOCHS}", unit="article")
                for article_index, (_, x, a, y) in enumerate(progress, start=1):
                    optimizer.zero_grad()
                    logits = model(x.to(DEVICE), a.to(DEVICE))
                    loss = F.binary_cross_entropy_with_logits(logits, y.to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    if article_index % 10 == 0 or article_index == len(train):
                        progress.set_postfix(loss=f"{total_loss / article_index:.4f}")
                metrics, _ = evaluate(model, validation, DEVICE, TOP_K)
                history.append({"epoch": epoch, "train_loss": total_loss / len(train), **metrics})
                print(json.dumps(history[-1]))
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    torch.save({"state_dict": model.state_dict(), "architecture": "pyg-gcnconv-v1",
                                "baseline": BASELINE, "seed": SEED, "top_k": TOP_K,
                                "validation_f1": best_f1}, OUTPUT / "best.pt")
            (OUTPUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            (OUTPUT / "split.json").write_text(json.dumps({"seed": SEED,
                "train_ids": [v[0] for v in train], "validation_ids": [v[0] for v in validation],
                "rejected": errors}, indent=2), encoding="utf-8")
            print("Best validation F1:", best_f1)
        """)),
        nbf.v4.new_markdown_cell(textwrap.dedent("""\
            ## Final test evaluation (run only after model choices are fixed)

            Set `RUN_TEST_EVALUATION = True` when ready. This cell loads `best.pt`
            and evaluates it without retraining. It saves sentence indices for
            ROUGE evaluation against `clean_summary` with `evaluate_rouge.py`.
        """)),
        nbf.v4.new_code_cell(textwrap.dedent("""\
            RUN_TEST_EVALUATION = False
            if RUN_TEST_EVALUATION:
                checkpoint = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=True)
                if checkpoint.get("architecture") != "pyg-gcnconv-v1":
                    raise ValueError("Checkpoint is not from this PyG architecture")
                test_model = SentenceScorer(graph=not checkpoint["baseline"]).to(DEVICE)
                test_model.load_state_dict(checkpoint["state_dict"])
                test, test_errors = load_split(TEST_FEATURES, "test", ADJACENCY_ROOT, SOURCE, MAX_DOCUMENTS)
                if test_errors or not test:
                    raise ValueError(f"Test alignment failed: {test_errors[:10]}")
                test_metrics, predictions = evaluate(test_model, test, DEVICE, checkpoint["top_k"])
                (OUTPUT / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
                with (OUTPUT / "test_predictions.jsonl").open("w", encoding="utf-8") as out:
                    for row in predictions:
                        out.write(json.dumps(row) + "\\n")
                print("Test metrics:", test_metrics)
        """)),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python (generate_amr)", "language": "python", "name": "generate_amr"}
    notebook.metadata.language_info = {"name": "python", "version": "3.10"}
    nbf.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
