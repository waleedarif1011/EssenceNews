"""Tests for ETL normalization, sentiment bounds, and classifier predictions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.etl import clean_text, load_news_df, normalize_record, to_dataframe
from src.nlp import analyze_sentiment, extract_entities, extract_keywords
from src.train_classifier import MODEL_PATH, predict_category, train

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "sample_news_1k.jsonl"


@pytest.fixture(scope="module")
def sample_df():
    assert SAMPLE.is_file(), "dataset/sample_news_1k.jsonl must ship with the repo"
    return load_news_df(SAMPLE)


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train once on the sample (writes artifacts used by the app)."""
    metrics = train(SAMPLE)
    assert MODEL_PATH.is_file()
    assert metrics["accuracy"] >= 0.15  # stratified multi-class baseline sanity
    return metrics


def test_clean_text_and_normalize():
    raw = {
        "headline": "  Hello, World!!! ",
        "short_description": "A <Great> Story...",
        "category": "POLITICS",
        "link": "https://example.com",
        "authors": "Jane Doe",
        "date": "2022-09-23",
    }
    cleaned = normalize_record(raw)
    assert cleaned["headline"] == "hello, world"
    assert "<" not in cleaned["short_description"]
    assert cleaned["category"] == "politics"
    assert cleaned["link"].startswith("https://")
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_to_dataframe_drops_empty_headlines():
    rows = [
        {"headline": "One", "short_description": "a", "category": "x", "link": "", "authors": "", "date": ""},
        {"headline": "", "short_description": "b", "category": "y", "link": "", "authors": "", "date": ""},
    ]
    df = to_dataframe(rows)
    assert len(df) == 1
    assert df.iloc[0]["headline"] == "one"


def test_load_sample_has_expected_shape(sample_df):
    assert len(sample_df) == 1000
    assert {"headline", "short_description", "category"}.issubset(sample_df.columns)
    assert sample_df["category"].nunique() >= 10


def test_sentiment_output_range():
    pos = analyze_sentiment("This is a wonderful, fantastic, excellent day.")
    neg = analyze_sentiment("This is a terrible, awful, horrible disaster.")
    empty = analyze_sentiment("")
    for result in (pos, neg, empty):
        assert -1.0 <= result["polarity"] <= 1.0
        assert 0.0 <= result["subjectivity"] <= 1.0
    assert pos["polarity"] > neg["polarity"]


def test_entities_api_shape():
    ents = extract_entities("Barack Obama spoke in New York about the United Nations.")
    assert set(ents) == {"ORG", "PERSON", "GPE"}
    for values in ents.values():
        assert isinstance(values, list)


def test_keywords_nonempty_on_corpus(sample_df):
    texts = (sample_df["headline"] + " " + sample_df["short_description"]).head(50).tolist()
    kws = extract_keywords(texts, top_n=5)
    assert len(kws) >= 1


def test_classifier_predicts_on_five_sample_rows(sample_df, trained_model):
    rows = sample_df.head(5)
    for _, row in rows.iterrows():
        result = predict_category(row["headline"], row["short_description"])
        assert isinstance(result["category"], str) and result["category"]
        assert result["probabilities"]
        # Top probability mass should be positive and sum of returned top-k <= 1
        probs = list(result["probabilities"].values())
        assert all(p >= 0 for p in probs)
        assert sum(probs) <= 1.0 + 1e-6
