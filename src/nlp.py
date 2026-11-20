"""Sentiment, NER, and keyword / topic helpers."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

# Labels we surface in the explorer UI.
ENTITY_LABELS = ("ORG", "PERSON", "GPE")


@lru_cache(maxsize=1)
def _nlp():
    """Load spaCy English model; fall back to blank pipeline if unavailable."""
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return spacy.blank("en")
    except ImportError:
        return None


def analyze_sentiment(text: str) -> dict[str, float]:
    """Return polarity [-1, 1] and subjectivity [0, 1] via TextBlob."""
    blob = TextBlob(str(text or ""))
    return {
        "polarity": float(blob.sentiment.polarity),
        "subjectivity": float(blob.sentiment.subjectivity),
    }


def extract_entities(text: str, labels: tuple[str, ...] = ENTITY_LABELS) -> dict[str, list[str]]:
    """Extract named entities for the requested spaCy labels."""
    result = {label: [] for label in labels}
    nlp = _nlp()
    if nlp is None or not str(text or "").strip():
        return result
    # Blank models have no NER pipe — return empty lists cleanly.
    if "ner" not in nlp.pipe_names:
        return result
    doc = nlp(str(text))
    for ent in doc.ents:
        if ent.label_ in result:
            result[ent.label_].append(ent.text)
    return result


def top_entities(
    texts: list[str],
    labels: tuple[str, ...] = ENTITY_LABELS,
    limit: int = 8,
) -> dict[str, list[tuple[str, int]]]:
    """Aggregate top entities across many texts."""
    counters = {label: Counter() for label in labels}
    for text in texts:
        ents = extract_entities(text, labels=labels)
        for label, values in ents.items():
            counters[label].update(values)
    return {label: counters[label].most_common(limit) for label in labels}


def extract_keywords(
    texts: list[str],
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[str]:
    """TF-IDF unigram/bigram keywords across a corpus of texts."""
    corpus = [str(t) for t in texts if str(t).strip()]
    if not corpus:
        return []
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=1 if len(corpus) < 5 else 2,
        max_features=5000,
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    order = scores.argsort()[::-1][:top_n]
    return [features[i] for i in order]


def keywords_for_text(
    text: str,
    background_texts: list[str] | None = None,
    top_n: int = 8,
) -> list[str]:
    """Keywords for one article, optionally contrasted against a background corpus."""
    text = str(text or "").strip()
    if not text:
        return []
    if background_texts:
        corpus = [text] + [str(t) for t in background_texts if str(t).strip()]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return extract_keywords([text], top_n=top_n)
        row = matrix[0].toarray().ravel()
        features = vectorizer.get_feature_names_out()
        order = row.argsort()[::-1][:top_n]
        return [features[i] for i in order if row[i] > 0]
    return extract_keywords([text], top_n=top_n)


def lda_topics(
    texts: list[str],
    n_topics: int = 4,
    n_top_words: int = 7,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Fit a small LDA model and return top words per topic."""
    corpus = [str(t) for t in texts if str(t).strip()]
    if len(corpus) < n_topics:
        return []
    cv = CountVectorizer(stop_words="english", max_df=0.95, min_df=1)
    try:
        X = cv.fit_transform(corpus)
    except ValueError:
        return []
    if X.shape[1] == 0:
        return []
    n_topics = min(n_topics, X.shape[0], X.shape[1])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(X)
    names = cv.get_feature_names_out()
    topics: list[dict[str, Any]] = []
    for idx, component in enumerate(lda.components_):
        words = [names[i] for i in component.argsort()[: -n_top_words - 1 : -1]]
        topics.append({"topic": idx + 1, "words": words})
    return topics
