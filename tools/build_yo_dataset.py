"""Build the Russian yo dataset from the permissive source lists.

Reads ``core/data/yo_safe.txt`` (unambiguous, applied at runtime), ``core/data/yo_not_safe.txt``
(ambiguous, published only) and ``core/data/yo_additions.json`` (project words such as the plain
spelling of "text"). Writes:

* ``core/data/yo_runtime.tsv.gz`` - ``ye-form<TAB>yo-form``, the table :mod:`core.yo` loads
* ``core/data/yo_dataset.jsonl.gz`` - one record per word form for other TTS pipelines

No network. Standard library plus :mod:`core.yo`. Does not publish the files anywhere.

    python tools/build_yo_dataset.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import yo  # noqa: E402

DATA = ROOT / "core" / "data"
SAFE = DATA / "yo_safe.txt"
NOT_SAFE = DATA / "yo_not_safe.txt"
ADDITIONS = DATA / "yo_additions.json"
RUNTIME = DATA / "yo_runtime.tsv.gz"
DATASET = DATA / "yo_dataset.jsonl.gz"


def _keep(key: str, table: Dict[str, str]) -> bool:
    """Lowercase forms, plus a capital form only when there is no lowercase twin."""
    if not key or key[0] not in yo._UPPER_SET:
        return True
    low = key[0].lower() + key[1:]
    return low not in table


def _records(table: Dict[str, str], kind: str, source: str) -> Iterable[dict]:
    for key in table:
        if not _keep(key, table):
            continue
        form = table[key]
        row = {
            "word": key,
            "yo_form": form,
            "stress": yo.stress_index(form),
            "type": kind,
            "source": source,
        }
        if kind == "homograph":
            plain = yo.fold(form)
            row["variants"] = [{"form": plain, "hint": ""}, {"form": form, "hint": ""}]
        yield row


def _apply_additions(rows: Dict[str, dict], runtime: Dict[str, str]) -> None:
    additions: List[dict] = json.loads(ADDITIONS.read_text(encoding="utf-8"))
    for item in additions:
        word = item["word"]
        form = item["yo_form"]
        row = {
            "word": word,
            "yo_form": form,
            "stress": item.get("stress", yo.stress_index(form)),
            "type": item.get("type", "unambiguous"),
            "source": "project",
            "hint": item.get("hint", ""),
        }
        rows[word] = row
        if form != word:
            runtime[yo.fold(word)] = form
            if word[0] not in yo._UPPER_SET:
                runtime[yo._capitalise(yo.fold(word))] = yo._capitalise(form)


def build() -> dict:
    """Write the runtime table and the JSONL dataset. Returns byte sizes and record counts."""
    safe = yo.load(SAFE.read_text(encoding="utf-8"))
    ambiguous = yo.load(NOT_SAFE.read_text(encoding="utf-8"))
    runtime = {key: safe[key] for key in safe if safe[key] != key}
    by_word: Dict[str, dict] = {}
    for row in _records(safe, "unambiguous", "eyo-kernel"):
        by_word[row["word"]] = row
    for row in _records(ambiguous, "homograph", "eyo-kernel"):
        if row["word"] in safe:
            continue
        by_word[row["word"]] = row
    _apply_additions(by_word, runtime)
    lines = [json.dumps(by_word[key], ensure_ascii=False, sort_keys=True) for key in sorted(by_word)]
    pairs = [f"{key}\t{runtime[key]}" for key in sorted(runtime)]
    # mtime=0 so a rebuild of the same inputs is byte-identical.
    DATASET.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), mtime=0))
    RUNTIME.write_bytes(gzip.compress(("\n".join(pairs) + "\n").encode("utf-8"), mtime=0))
    return {
        "runtime_bytes": RUNTIME.stat().st_size,
        "dataset_bytes": DATASET.stat().st_size,
        "runtime_rows": len(pairs),
        "dataset_rows": len(lines),
    }


def main() -> int:
    stats = build()
    print(
        f"runtime {stats['runtime_rows']} rows, {stats['runtime_bytes']} bytes; "
        f"dataset {stats['dataset_rows']} rows, {stats['dataset_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
