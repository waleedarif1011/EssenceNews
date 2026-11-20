---
title: Essence News Intelligence Explorer
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Essence News — News Intelligence Explorer

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/waleedarif1011/essencenews)
[![CI](https://github.com/waleedarif1011/essencenews/actions/workflows/ci.yml/badge.svg)](https://github.com/waleedarif1011/essencenews/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

End-to-end NLP portfolio project: ETL → sentiment / NER / TF-IDF keywords → lightweight headline category classifier → interactive **News Intelligence Explorer**.

> **Live demo:** public Space URL is wired above. If the badge 404s, create the Space with the steps in [`docs/hf-space-deploy.md`](docs/hf-space-deploy.md) (no `HF_TOKEN` was available in the agent environment).

## Screenshots

| Analyze tab | Dataset Explorer |
|-------------|------------------|
| ![Analyze headline](assets/analyze-tab.png) | ![Dataset explorer](assets/explorer-tab.png) |

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.train_classifier   # writes artifacts/category_model.joblib
python app.py                    # News Intelligence Explorer (Gradio)
```

Optional — download the full ~210k-row HuffPost JSONL:

```bash
python scripts/download_data.py
```

## Project layout

```
app.py                      # Gradio News Intelligence Explorer
src/
  etl.py                    # load & normalize JSONL
  nlp.py                    # TextBlob sentiment, spaCy NER, TF-IDF / LDA
  train_classifier.py       # TF-IDF + LinearSVC category model
  analyze.py                # single-article orchestration
dataset/sample_news_1k.jsonl
artifacts/category_model.joblib
artifacts/metrics.json
notebooks/news_analysis.ipynb
scripts/download_data.py
tests/test_nlp.py
```

Large intermediates (`News_Category_Dataset_v3.json`, `etl_output.csv`, `structured_output.jsonl`) are gitignored — regenerate locally via the download script + notebook / `src.etl`.

## Classifier metrics

Trained on the stratified 1k sample (80/20 holdout, 42 categories):

| Metric | Value |
|--------|-------|
| Accuracy | 0.535 |
| Macro F1 | 0.358 |
| Weighted F1 | 0.497 |
| Train / test rows | 800 / 200 |

## Pipeline

```mermaid
graph TD;
    A[Raw JSONL Data] --> B[ETL: Clean & Standardize]
    B --> C[Structured records]
    C --> D[EDA & NLP Analysis]
    D --> E[TF-IDF + LinearSVC classifier]
    E --> F[Gradio News Intelligence Explorer]
```

## App features

1. **Analyze Headline / Article** — predicted category (top-k scores), sentiment polarity/subjectivity, named entities (ORG / PERSON / GPE), topic keywords. Four one-click sample headlines included.
2. **Dataset Explorer** — category distribution, mean sentiment by category, and top entities from `sample_news_1k.jsonl`.

## Tests & CI

```bash
python -m pytest -q
```

GitHub Actions runs install → spaCy model download → train → pytest on every push/PR to `main`.

## References

- [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- [spaCy](https://spacy.io/) · [TextBlob](https://textblob.readthedocs.io/) · [scikit-learn](https://scikit-learn.org/) · [Gradio](https://www.gradio.app/)

## Author

**Waleed Arif** — Data Scientist
