"""News Intelligence Explorer — Gradio demo for Essence News NLP."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from src.analyze import analyze_article
from src.etl import load_news_df
from src.nlp import analyze_sentiment, top_entities
from src.train_classifier import METRICS_PATH

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "dataset" / "sample_news_1k.jsonl"

SAMPLE_HEADLINES = [
    {
        "label": "COVID boosters (U.S. News)",
        "headline": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
        "description": (
            "Health experts said it is too early to predict whether demand would match "
            "up with the 171 million doses of the new boosters the U.S. ordered for the fall."
        ),
    },
    {
        "label": "Senate politics",
        "headline": "Senate Passes Landmark Bill to Cap Insulin Costs for Seniors",
        "description": (
            "Lawmakers celebrated a bipartisan win as advocates said the measure could "
            "ease prescription drug prices for millions of Medicare patients."
        ),
    },
    {
        "label": "Hollywood entertainment",
        "headline": "Oscar-Winning Director Announces Surprise Sequel After Fan Campaign",
        "description": (
            "The filmmaker teased a return to the franchise that made her famous, "
            "promising new characters and a winter 2026 release window."
        ),
    },
    {
        "label": "Wellness tip",
        "headline": "Why Morning Walks Beat Afternoon Coffee for Focus, Experts Say",
        "description": (
            "A short outdoor stroll can raise alertness and mood without the crash "
            "that often follows a second espresso, according to sleep researchers."
        ),
    },
]


def _load_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.is_file():
        return pd.DataFrame(columns=["headline", "short_description", "category"])
    return load_news_df(SAMPLE_PATH)


def _metrics_md() -> str:
    if not METRICS_PATH.is_file():
        return "_Train the classifier (`python -m src.train_classifier`) to populate metrics._"
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return (
        "### Classifier metrics (held-out 20% of sample)\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Accuracy | {metrics.get('accuracy', float('nan')):.3f} |\n"
        f"| Macro F1 | {metrics.get('macro_f1', float('nan')):.3f} |\n"
        f"| Weighted F1 | {metrics.get('weighted_f1', float('nan')):.3f} |\n"
        f"| Classes | {metrics.get('n_classes', '—')} |\n"
    )


def _format_analysis(result: dict) -> tuple[str, str, str, str]:
    probs = result.get("probabilities") or {}
    if probs:
        prob_lines = "\n".join(f"- **{cat}**: {p:.1%}" for cat, p in probs.items())
    else:
        prob_lines = "_No model loaded — run `python -m src.train_classifier`._"

    sent = result.get("sentiment") or {}
    polarity = float(sent.get("polarity", 0.0))
    subjectivity = float(sent.get("subjectivity", 0.0))
    tone = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
    sentiment_md = (
        f"**Tone:** {tone}\n\n"
        f"- Polarity: `{polarity:.3f}` (−1 → +1)\n"
        f"- Subjectivity: `{subjectivity:.3f}` (0 → 1)"
    )

    ents = result.get("entities") or {}
    ent_parts = []
    for label in ("ORG", "PERSON", "GPE"):
        values = ents.get(label) or []
        shown = ", ".join(dict.fromkeys(values)) if values else "—"
        ent_parts.append(f"- **{label}:** {shown}")
    entities_md = "\n".join(ent_parts)

    kws = result.get("keywords") or []
    keywords_md = ", ".join(f"`{k}`" for k in kws) if kws else "—"

    category_md = f"### Predicted category: **{result.get('category', 'unknown')}**\n\n{prob_lines}"
    return category_md, sentiment_md, entities_md, keywords_md


def analyze(headline: str, description: str) -> tuple[str, str, str, str]:
    result = analyze_article(headline, description)
    return _format_analysis(result)


def apply_sample(label: str) -> tuple[str, str]:
    sample = next((s for s in SAMPLE_HEADLINES if s["label"] == label), SAMPLE_HEADLINES[0])
    return sample["headline"], sample["description"]


def explorer_summary() -> tuple[str, str, str]:
    df = _load_sample()
    if df.empty:
        empty = "_Sample dataset missing._"
        return empty, empty, empty

    cat_counts = df["category"].value_counts().head(12)
    cat_md = "### Category distribution (top 12)\n\n" + "\n".join(
        f"- **{cat}**: {n}" for cat, n in cat_counts.items()
    )

    # Sentiment by category (top 8 categories)
    top_cats = cat_counts.head(8).index.tolist()
    rows = []
    for cat in top_cats:
        subset = df[df["category"] == cat]
        texts = (subset["headline"] + " " + subset["short_description"]).tolist()
        polarities = [analyze_sentiment(t)["polarity"] for t in texts]
        avg = sum(polarities) / max(len(polarities), 1)
        rows.append(f"- **{cat}**: {avg:+.3f} (n={len(polarities)})")
    sent_md = "### Mean sentiment polarity by category\n\n" + "\n".join(rows)

    sample_texts = (df["headline"] + " " + df["short_description"]).astype(str).tolist()[:300]
    tops = top_entities(sample_texts, limit=6)
    ent_lines = []
    for label, pairs in tops.items():
        if pairs:
            shown = ", ".join(f"{name} ({count})" for name, count in pairs)
        else:
            shown = "—(spaCy model not loaded)"
        ent_lines.append(f"- **{label}:** {shown}")
    ent_md = "### Top entities (first 300 sample articles)\n\n" + "\n".join(ent_lines)

    return cat_md, sent_md, ent_md


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Essence News — News Intelligence Explorer") as demo:
        gr.Markdown(
            "# News Intelligence Explorer\n"
            "Paste a headline (and optional short description) to get category, "
            "sentiment, named entities, and topic keywords — or browse the 1k sample."
        )
        with gr.Tabs():
            with gr.Tab("Analyze Headline / Article"):
                with gr.Row():
                    with gr.Column():
                        preset = gr.Dropdown(
                            choices=[s["label"] for s in SAMPLE_HEADLINES],
                            value=SAMPLE_HEADLINES[0]["label"],
                            label="One-click samples",
                        )
                        headline = gr.Textbox(
                            label="Headline",
                            lines=2,
                            value=SAMPLE_HEADLINES[0]["headline"],
                        )
                        description = gr.Textbox(
                            label="Short description (optional)",
                            lines=4,
                            value=SAMPLE_HEADLINES[0]["description"],
                        )
                        run = gr.Button("Analyze", variant="primary")
                    with gr.Column():
                        category_out = gr.Markdown()
                        sentiment_out = gr.Markdown(label="Sentiment")
                        entities_out = gr.Markdown()
                        keywords_out = gr.Markdown()
                gr.Markdown(_metrics_md())
                preset.change(fn=apply_sample, inputs=[preset], outputs=[headline, description])
                run.click(
                    fn=analyze,
                    inputs=[headline, description],
                    outputs=[category_out, sentiment_out, entities_out, keywords_out],
                )
                demo.load(
                    fn=lambda: analyze(SAMPLE_HEADLINES[0]["headline"], SAMPLE_HEADLINES[0]["description"]),
                    outputs=[category_out, sentiment_out, entities_out, keywords_out],
                )

            with gr.Tab("Dataset Explorer"):
                refresh = gr.Button("Refresh summary", variant="secondary")
                cat_out = gr.Markdown()
                sent_out = gr.Markdown()
                ent_out = gr.Markdown()
                refresh.click(fn=explorer_summary, outputs=[cat_out, sent_out, ent_out])
                demo.load(fn=explorer_summary, outputs=[cat_out, sent_out, ent_out])

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
