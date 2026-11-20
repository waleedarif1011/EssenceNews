#!/usr/bin/env python3
"""Download the full HuffPost News Category Dataset on demand.

The repo ships a 1k stratified sample at dataset/sample_news_1k.jsonl.
Use this script when you need the full ~210k-row JSONL for local experiments.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Public mirror of the Kaggle "News Category Dataset" (JSON Lines).
DEFAULT_URL = (
    "https://raw.githubusercontent.com/rishabhmisra/News-Category-Dataset/"
    "master/News_Category_Dataset_v3.json"
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dataset" / "News_Category_Dataset_v3.json"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"Downloading\n  {url}\n→ {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 — surface clear guidance to users
        print(
            "Download failed. You can also grab the dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/rmisra/news-category-dataset\n"
            f"Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB → {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Source URL for the JSONL file")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Destination path (default: dataset/News_Category_Dataset_v3.json)",
    )
    args = parser.parse_args()
    download(args.url, args.out)


if __name__ == "__main__":
    main()
