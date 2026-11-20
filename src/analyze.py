"""Single-article analysis used by the Gradio explorer and tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.etl import load_news_df
from src.nlp import analyze_sentiment, extract_entities, keywords_for_text
from src.train_classifier import MODEL_PATH, load_model, predict_category

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "dataset" / "sample_news_1k.jsonl"


@lru_cache(maxsize=1)
def _background_descriptions(limit: int = 200) -> tuple[str, ...]:
    if not SAMPLE_PATH.is_file():
        return ()
    df = load_news_df(SAMPLE_PATH)
    texts = (df["headline"] + " " + df["short_description"]).astype(str).tolist()
    return tuple(texts[:limit])


@lru_cache(maxsize=1)
def _model():
    if MODEL_PATH.is_file():
        return load_model(MODEL_PATH)
    return None


def analyze_article(headline: str, description: str = "") -> dict[str, Any]:
    """Run category, sentiment, NER, and keywords on one headline/article."""
    headline = (headline or "").strip()
    description = (description or "").strip()
    combined = f"{headline}. {description}".strip()

    sentiment = analyze_sentiment(combined or headline)
    entities = extract_entities(combined or headline)
    keywords = keywords_for_text(combined or headline, list(_background_descriptions()))

    prediction: dict[str, Any]
    model = _model()
    if model is not None and headline:
        prediction = predict_category(headline, description, model=model)
    else:
        prediction = {"category": "unknown", "probabilities": {}}

    return {
        "category": prediction["category"],
        "probabilities": prediction.get("probabilities", {}),
        "sentiment": sentiment,
        "entities": entities,
        "keywords": keywords,
    }
