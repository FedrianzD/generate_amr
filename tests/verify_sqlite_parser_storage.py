"""Exercise actual SQLite storage and dataset code embedded in both parsers."""
import ast
import importlib.util
import json
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import penman

ROOT = Path(__file__).resolve().parents[1]
STORAGE_SOURCE = (ROOT / "common/amr_sqlite.py").read_text(encoding="utf-8")
spec = importlib.util.spec_from_file_location(
    "converter", ROOT / "scripts/liputan6/documents_to_sqlite.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def check_notebook(name, root):
    path = ROOT / f"notebooks/liputan6/amr_parse_id2id_abdi_{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    for cell, source in zip(notebook["cells"], sources):
        if cell["cell_type"] == "code":
            compile("\n".join(line for line in source.splitlines()
                              if not line.startswith("%")), str(path), "exec")
    assert sources[11].startswith(STORAGE_SOURCE)
    previous = root / "previous"
    previous.mkdir()
    for index in (10, 2):
        (previous / f"100_{index}.txt").write_text(
            f"# ::id 100_{index}\n# ::snt sentence {index}\n(b / baik)\n", encoding="utf-8")
    output = root / "output"
    output.mkdir()
    ns = dict(OUTPUT_DB=output / "train.sqlite", SPLIT="train", PREV_DB="", PREV_DIR=str(previous))
    exec(sources[11], ns)
    assert ns["PARSED_IDS"] == {"100_2", "100_10"}
    assert [g.metadata["id"] for g in ns["STORE"].read_document("100")] == ["100_2", "100_10"]
    exec(sources[11], ns)
    ns.update(_logged_truncations={"100_3"}, MAX_TEXT_TOKENS=100)
    exec(sources[15][sources[15].index("def store_graph("):], ns)
    ns["store_graph"](["100_3", "200_0"], ["new text", "other doc"],
                      ["(b / baru)", "(l / lain)"])
    store = ns["STORE"]
    graphs = store.read_document("100")
    assert [g.metadata["id"] for g in graphs] == ["100_2", "100_3", "100_10"]
    assert graphs[1].metadata["input_truncated"] == "true"
    assert graphs[1].metadata["kept_text_tokens"] == "100"
    assert not list(output.glob("*.txt"))
    before = ns["PARSED_IDS"].copy()
    try:
        ns["store_graph"](["300_0", "100_2"], ["new", "conflict"],
                          ["(n / new)", "(c / conflict)"])
    except ValueError as exc:
        assert "Conflicting" in str(exc)
    else:
        raise AssertionError("Conflicting generation batch was accepted")
    assert ns["PARSED_IDS"] == before == store.parsed_ids()
    try:
        ns["store_graph"](["300_0"], [], ["(n / new)"])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched batch lengths were accepted")
    documents = root / "documents"
    documents.mkdir()
    (documents / "400.txt").write_text(
        "# ::id 400_0\n# ::custom preserved\n(f / first)\n", encoding="utf-8")
    converted = root / "converted.sqlite"
    converter.convert(documents, converted)
    resumed = ns["prepare_sqlite_store"](root / "resumed.sqlite", "train", converted)
    assert resumed.parsed_ids() == {"400_0"}
    assert resumed.read_document("400")[0].metadata["custom"] == "preserved"
    store.import_database(converted)
    assert "400_0" in store.parsed_ids()
    # An identical repeated import must avoid parsing/encoding stored graphs.
    with patch.object(penman, "decode", side_effect=AssertionError("Redundant decode")):
        store.import_database(converted)
    conflicting = ns["AMRSQLiteStore"](root / "conflicting.sqlite", "train")
    conflicting.write_graphs([penman.decode(
        "# ::id 400_0\n(c / conflicting)", model=ns["STORAGE_MODEL"])])
    try:
        store.import_database(conflicting.path)
    except ValueError as exc:
        assert "Conflicting" in str(exc)
    else:
        raise AssertionError("Conflicting SQLite import was accepted")
    # Document-level legacy files are also accepted.
    store.import_directory(documents)
    other = ns["AMRSQLiteStore"](ns["OUTPUT_DB"], "test")
    other.write_graphs([penman.decode("# ::id 500_0\n(t / test)", model=ns["STORAGE_MODEL"])])
    assert "500_0" not in store.parsed_ids()
    dataset_node = next(node for node in ast.parse(sources[13]).body
                        if isinstance(node, ast.ClassDef) and node.name == "Liputan6Dataset")
    data = pd.DataFrame([
        dict(id="100_2", doc_id="100", sent_idx=2, split="train", text="parsed"),
        dict(id="100_0", doc_id="100", sent_idx=0, split="train", text="missing"),
        dict(id="500_0", doc_id="500", sent_idx=0, split="train", text="other split only"),
        dict(id="600_0", doc_id="600", sent_idx=0, split="test", text="test"),
    ])
    class TableReader:
        @staticmethod
        def read_parquet(path):
            return data.copy()
    ns.update(Dataset=object, pd=TableReader, DATA_PATH="", DATA_FILE="unused",
              OUTPUT_DIR=str(output), DOC_LIMIT=None, SENT_LIMIT=None, SORT_BY_LENGTH=False)
    exec(ast.get_source_segment(sources[13], dataset_node), ns)
    dataset = ns["Liputan6Dataset"]()
    assert dataset.df["id"].tolist() == ["100_0", "500_0"]
    ns.update(pd=pd, json=json)
    with patch.object(pd, "read_parquet", return_value=data.copy()):
        exec(sources[23], ns)
    statuses = ns["report"].set_index("doc_id")["status"].to_dict()
    assert statuses["100"] == "partial_or_misaligned"
    assert statuses["500"] == "no_graphs"
    assert statuses["400"] == "outside_source"
    exec(sources[24], ns)
    converter.convert(documents, ns["OUTPUT_DB"])
    with closing(ns["readonly_database"](ns["OUTPUT_DB"])) as connection:
        assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
    print(f"PASS: {name}: legacy import, converter compatibility, SQLite resume, "
          "split isolation, partial-document filtering, metadata, batch rollback")


def main():
    with tempfile.TemporaryDirectory() as directory:
        for name in ("kaggle_documents", "gcp_fixed"):
            root = Path(directory) / name
            root.mkdir()
            check_notebook(name, root)


if __name__ == "__main__":
    main()
