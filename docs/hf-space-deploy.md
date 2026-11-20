# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space is created by running `scripts/deploy_hf_space.sh` with a Hugging Face **write** token exported as `HF_TOKEN` (or via `huggingface-cli login`). Cursor *app* secrets used for `publish-app` are **not** injected into cloud-agent shells — export the token in your local terminal or CI before running the script. The Space was **not** created from the usability PR.

## One-command deploy (recommended)

```bash
# Token: https://huggingface.co/settings/tokens  (Write + Spaces)
export HF_TOKEN=hf_...
./scripts/deploy_hf_space.sh
```

The script is idempotent:

1. Creates `waleedarif1011/essencenews` (Gradio SDK) if it does not exist.
2. Uploads Space runtime files via `huggingface_hub.upload_folder` (skips notebooks, tests, docs, download helpers, large raw dumps).
3. Prints `GET https://huggingface.co/spaces/waleedarif1011/essencenews` status (expect **200** when the build is up).

Overrides: `HF_USER`, `HF_SPACE`.

## Manual CLI steps

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the write token
huggingface-cli repo create essencenews --type space --space_sdk gradio -y
# then either re-run ./scripts/deploy_hf_space.sh  OR:
git remote add space https://huggingface.co/spaces/waleedarif1011/essencenews
git push space main
```

## After deploy

1. Confirm the Space builds and both tabs work (Analyze + Dataset Explorer); four sample headlines should return categories.
2. Keep the Live Demo badge in `README.md` pointing at `https://huggingface.co/spaces/waleedarif1011/essencenews` (update only if the slug differs).
3. Set the GitHub repo **Homepage** to that Space URL (maintainer / `gh repo edit`).

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter for Spaces)
- `src/` (`etl.py`, `nlp.py`, `train_classifier.py`, `analyze.py`)
- `dataset/sample_news_1k.jsonl`
- `artifacts/category_model.joblib`, `artifacts/metrics.json`
- `assets/` (optional; README screenshots)

`requirements.txt` pins `en-core-web-sm` so NER works on cold start without a separate spaCy download step. If artifacts are missing on first build, regenerate with:

```bash
python -m src.train_classifier
```
