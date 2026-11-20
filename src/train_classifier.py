"""Train a lightweight TF-IDF + LinearSVC headline category classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.etl import load_news_df

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "sample_news_1k.jsonl"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "category_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                    max_features=20_000,
                ),
            ),
            ("clf", LinearSVC(C=1.0, dual="auto", random_state=42)),
        ]
    )


def _text_column(df) -> list[str]:
    headlines = df["headline"].astype(str)
    descs = df["short_description"].astype(str)
    return (headlines + " " + descs).str.strip().tolist()


def train(
    data_path: str | Path = DEFAULT_DATA,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_news_df(data_path)
    # Keep categories with enough support for a meaningful stratified split.
    counts = df["category"].value_counts()
    keep = counts[counts >= 5].index
    df = df[df["category"].isin(keep)].reset_index(drop=True)

    X = _text_column(df)
    y = df["category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics: dict[str, Any] = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": int(len(set(y))),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "report": classification_report(y_test, preds, zero_division=0, output_dict=True),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_category(
    headline: str,
    description: str = "",
    model: Pipeline | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Predict category with decision-function probabilities (softmax over scores)."""
    pipe = model or load_model()
    text = f"{headline} {description}".strip()
    label = str(pipe.predict([text])[0])

    classes = [str(c) for c in pipe.named_steps["clf"].classes_]
    scores = pipe.decision_function([text])
    scores = np.atleast_2d(scores)[0]
    # Softmax over decision scores for a simple probability-like ranking.
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    ranking = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])[:top_k]
    return {
        "category": str(label),
        "probabilities": {c: float(p) for c, p in ranking},
    }


def main() -> None:
    metrics = train()
    print(
        f"Saved {MODEL_PATH.name} | accuracy={metrics['accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} classes={metrics['n_classes']}"
    )


if __name__ == "__main__":
    main()
