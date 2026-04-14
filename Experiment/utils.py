from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime


def make_output_dir(root: str = "outputs", run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"run_{ts}"
    out_dir = Path(root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(obj, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(records, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")