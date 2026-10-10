"""Score selected Liputan6 source sentences against clean_summary with ROUGE.

Run after train_gcn.ipynb has produced test_predictions.jsonl. This measures
summary overlap with the human-written reference; it is distinct from oracle
sentence-selection F1. Scores are averaged per article.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/liputan6/source/Liputan6/liputan6/liputan6_data/canonical"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "runs/pyg-gcn-top10000-factuality/test_predictions.jsonl")
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.predictions.with_name("test_rouge.json")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for name in ("rouge1", "rouge2", "rougeL")}
    count = 0
    seen = set()
    with args.predictions.open(encoding="utf-8") as predictions:
        for line in predictions:
            row = json.loads(line)
            doc_id = str(row["id"])
            if doc_id in seen:
                raise ValueError(f"duplicate prediction for article {doc_id}")
            seen.add(doc_id)
            source_path = args.source_root / args.split / f"{doc_id}.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            article = source["clean_article"]
            reference = source["clean_summary"]
            selected = row["selected_sentences"]
            if not isinstance(selected, list) or not selected or any(
                type(index) is not int or index < 0 or index >= len(article) for index in selected
            ) or len(set(selected)) != len(selected):
                raise ValueError(f"invalid selected sentence indices for article {doc_id}")
            if not isinstance(reference, list) or not reference:
                raise ValueError(f"missing clean_summary for article {doc_id}")
            generated_text = " ".join(" ".join(article[index]) for index in sorted(selected))
            reference_text = " ".join(" ".join(sentence) for sentence in reference)
            scores = scorer.score(reference_text, generated_text)
            for name, score in scores.items():
                totals[name]["precision"] += score.precision
                totals[name]["recall"] += score.recall
                totals[name]["f1"] += score.fmeasure
            count += 1
    if count == 0:
        raise ValueError("prediction file is empty")
    result = {"articles": count, "reference": "clean_summary", "selection": "source sentences in original order",
              "aggregation": "macro mean", "use_stemmer": False,
              "scores": {name: {key: value / count for key, value in measures.items()}
                         for name, measures in totals.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
