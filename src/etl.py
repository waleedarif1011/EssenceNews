"""Load and normalize HuffPost-style news JSONL records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

COLUMNS = ["headline", "short_description", "category", "link", "authors", "date"]


def clean_text(text: Any) -> str:
    """Lowercase, strip, and remove special characters (keep word chars, spaces, . ,)."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    """Return a cleaned, flat record with the standard columns."""
    out: dict[str, str] = {}
    for col in COLUMNS:
        value = record.get(col, "")
        if col in {"headline", "short_description", "category", "authors"}:
            out[col] = clean_text(value)
        else:
            out[col] = "" if value is None else str(value).strip()
    return out


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON Lines into a list of raw dicts (skips malformed lines)."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def iter_normalized(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream normalized records from a JSONL path."""
    for record in load_jsonl(path):
        yield normalize_record(record)


def to_dataframe(records: Iterable[dict[str, Any]], normalize: bool = True) -> pd.DataFrame:
    """Convert records to a DataFrame; optionally normalize and drop empty headlines."""
    rows = [normalize_record(r) if normalize else r for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS]
    df = df.drop_duplicates()
    df = df[df["headline"].astype(str).str.len() > 0].reset_index(drop=True)
    return df


def load_news_df(path: str | Path) -> pd.DataFrame:
    """Convenience: load JSONL → normalized DataFrame."""
    return to_dataframe(load_jsonl(path), normalize=True)


def to_structured(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Nested key–value form used by the notebook / downstream NLP."""
    return {
        "headline": row["headline"],
        "category": row["category"],
        "metadata": {
            "description": row.get("short_description", ""),
            "authors": row.get("authors", ""),
            "date": row.get("date", ""),
            "link": row.get("link", ""),
        },
    }
