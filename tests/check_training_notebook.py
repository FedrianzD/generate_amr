"""Execute the self-contained training notebook on synthetic train/dev/test inputs."""

import json
import tempfile
import sqlite3
from contextlib import closing
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/liputan6/train_gcn.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        adjacency = root / "adjacency"
        feature_paths = {}
        ids = {"train": ["1", "2", "3", "4"], "dev": ["5", "6"], "test": ["7", "8"]}
        article = [["alpha", "beta", "gamma"], ["alpha", "beta", "gamma", "duplicate"],
                   ["delta", "epsilon", "zeta"], ["irrelevant", "news"]]
        for split, members in ids.items():
            (source / split).mkdir(parents=True)
            (adjacency / split).mkdir(parents=True)
            records = []
            for doc_id in members:
                content = {"clean_article": article, "clean_summary": [article[0], article[2]],
                           "extractive_summary": [3]}  # Deliberately disagrees with greedy oracle.
                (source / split / f"{doc_id}.json").write_text(json.dumps(content), encoding="utf-8")
                a = np.full((4, 4), 0.5, dtype=np.float32)
                np.fill_diagonal(a, 0)
                np.save(adjacency / split / f"{doc_id}.npy", a)
                records.append({"id": doc_id, "embeddings": torch.randn(4, 768)})
            feature_paths[split] = root / f"{split}.pt"
            torch.save(records, feature_paths[split])
        # Highest-scoring docs are incomplete or have equal-count misaligned indices.
        amr_db = root / "train.sqlite"
        with closing(sqlite3.connect(amr_db)) as db, db:
            db.executescript("""
                CREATE TABLE storage_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO storage_metadata VALUES ('schema_version', 'amr-documents-v1');
                CREATE TABLE graphs (
                    split TEXT, doc_id TEXT, sent_idx INTEGER, sentence_id TEXT, amr_penman TEXT,
                    PRIMARY KEY(split, doc_id, sent_idx), UNIQUE(split, sentence_id));
            """)
            for doc_id, indices in {"1": [0, 1, 2], "2": range(4),
                                    "3": range(4), "4": [0, 1, 2, 4]}.items():
                db.executemany("INSERT INTO graphs VALUES (?, ?, ?, ?, ?)",
                               [("train", doc_id, i, f"{doc_id}_{i}", "(a / alpha)")
                                for i in indices])
        train_records = torch.load(feature_paths["train"], weights_only=True)
        shards = [root / "train_1.pt", root / "train_2.pt"]
        torch.save(train_records[:2], shards[0])
        torch.save(train_records[2:], shards[1])
        feature_paths["train"] = shards
        np.savez_compressed(adjacency / "train.npz", **{
            doc_id: np.load(adjacency / "train" / f"{doc_id}.npy") for doc_id in ids["train"]})
        scores = root / "scores.csv"
        pd.DataFrame({"id": list("12345678"),
                      "split": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
                      "factuality": [1] * 8,
                      "prob_yes_factuality": [0.95, 0.9, 0.9, 0.94, 1.0, 1.0, 1.0, 1.0]}).to_csv(scores, index=False)
        scope = {"__name__": "__notebook_smoke__"}
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            exec(compile(cell.source, f"{NOTEBOOK}:cell-{index}", "exec"), scope)
            if cell.id == "configuration":
                scope.update({"ROOT": root, "FEATURE_PATHS": feature_paths,
                              "SOURCE_ROOT": source, "ADJACENCY_ROOT": adjacency,
                              "FACTUALITY_CSV": scores, "TRAIN_DOCUMENTS": 2, "TRAIN_AMR_DB": amr_db,
                              "OUTPUT": root / "run", "EPOCHS": 1, "BATCH_SIZE": 2,
                              "DROPOUT": 0.0, "DEVICE": "cpu", "RUN_TEST_EVALUATION": True})
        assert scope["train_ids"] == ["2", "3"], "Rank continuous scores, exclude dev/test, break ties by ID"
        exclusions = pd.read_csv(scope["OUTPUT"] / "train_amr_excluded.csv", dtype={"id": str})
        assert exclusions["id"].tolist() == ["1", "4"]
        assert set(exclusions["reason"]) == {"incomplete_or_misaligned_amr"}
        assert scope["manifest"]["factuality_rank"].tolist() == [3, 4]
        assert len(scope["train"]) == 2 and len(scope["validation"]) == 2
        assert scope["train"][0]["gold"] == [0, 2], "Must regenerate oracle, not reuse extractive_summary"
        assert scope["history"][-1]["epoch"] == 1
        assert scope["test_metrics"]["documents"] == 2
        assert scope["best_rouge"] == scope["history"][-1]["rouge1_f1"]
        selected = scope["select_sentences"](torch.tensor([3., 2., 1., 0.]),
                                              [" ".join(s) for s in article], 2, True)
        assert selected == [0, 2], "Must block the repeated trigram"
        chosen = scope["select_sentences"](torch.tensor([-1., -2., -3., -4.]),
                                            [" ".join(s) for s in article], 2, True)
        assert len(chosen) == 2, "Ranked top-K is not positive-only selection"
        assert scope["ORACLE_SETTINGS"]["max_sentences"] == 2
        assert scope["greedy_oracle"](["alpha", "beta", "gamma"], "alpha beta gamma") == [0, 1]
        scope["ORACLE_SETTINGS"]["max_sentences"] = None
        assert scope["cached_oracle"]("train", "budget", ["alpha", "beta", "gamma"],
                                         "alpha beta gamma") == [0, 1, 2]
        scope["ORACLE_SETTINGS"]["max_sentences"] = 2
        assert scope["cached_oracle"]("train", "budget", ["alpha", "beta", "gamma"],
                                         "alpha beta gamma") == [0, 1], "Cap must invalidate unlimited caches"
        assert scope["greedy_oracle"](["alpha", "alpha"], "alpha") == [0]
        assert scope["cached_oracle"]("train", "2", ["alpha", "beta"], "beta") == [1]
        model = scope["model"].eval()
        graphs = [example["graph"] for example in scope["train"]]
        with torch.no_grad():
            torch.testing.assert_close(model(Batch.from_data_list(graphs)),
                                       torch.cat([model(graph) for graph in graphs]), atol=1e-5, rtol=1e-5)
        broken = root / "missing.pt"
        torch.save([{"id": "2", "embeddings": torch.zeros(4, 768)}], broken)
        try:
            scope["load_feature_map"](broken, ["2", "3"])
        except ValueError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("Missing ranked features must fail, not silently replace documents")
        predictions = scope["OUTPUT"] / "test_predictions.jsonl"
        assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
        assert (scope["OUTPUT"] / "best.pt").is_file()
        assert (scope["OUTPUT"] / "last.pt").is_file()
        assert not list(scope["OUTPUT"].rglob("*.tmp"))
        # Final evaluation must also work without preparing/training again.
        standalone = {"__name__": "__evaluation_only__"}
        for cell in notebook.cells:
            if cell.cell_type != "code" or cell.id not in {
                "imports", "configuration", "data-helpers", "model-evaluation", "final-test"
            }:
                continue
            exec(compile(cell.source, f"{NOTEBOOK}:{cell.id}", "exec"), standalone)
            if cell.id == "configuration":
                standalone.update({"FEATURE_PATHS": feature_paths, "SOURCE_ROOT": source,
                                   "ADJACENCY_ROOT": adjacency, "OUTPUT": scope["OUTPUT"],
                                   "DEVICE": "cpu", "RUN_TEST_EVALUATION": True})
        assert "train" not in standalone and standalone["test_metrics"]["documents"] == 2
    print("PASS: ranking/splits, greedy labels/cache, residual minibatch GCN, trigram selection, "
          "ROUGE checkpoint selection, held-out test, missing-input failure, atomic outputs")


if __name__ == "__main__":
    main()
