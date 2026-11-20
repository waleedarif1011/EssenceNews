# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment or Cursor secrets.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create essencenews --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/essencenews
git push space HEAD:main
```

3. In the Space settings, ensure the SDK is **Gradio** and the app file is `app.py`.
4. Confirm both tabs work (Analyze + Dataset Explorer) and the four sample headlines return categories.
5. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/essencenews`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter for Spaces)
- `src/` (`etl.py`, `nlp.py`, `train_classifier.py`, `analyze.py`)
- `dataset/sample_news_1k.jsonl`
- `artifacts/category_model.joblib`, `artifacts/metrics.json`

On first Space build, either commit the trained artifacts or add a startup step:

```bash
python -m spacy download en_core_web_sm
python -m src.train_classifier
```
