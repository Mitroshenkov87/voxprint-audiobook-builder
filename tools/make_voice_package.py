"""Build a distributable voice package for the online voices index.

    python tools/make_voice_package.py --adapter <trained voice folder> --spec tools/voice_specs/alexander.json \\
        --out dist_voices --url https://huggingface.co/<user>/<repo>/resolve/main/alexander.zip

Writes ``<out>/<id>/`` (the files the app needs, with the final ``voice.json``), ``<out>/<id>.zip``, and
``<out>/<id>.index-entry.json`` - one object for the ``voices`` list of the repository's ``index.json`` (with SHA-256 and
size).  The spec is a small JSON with the fields to put into ``voice.json`` (names, descriptions, licence, consent ...).
Weights never go into the code repository: upload the zip to a model host / release asset and publish the entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import voice_info  # noqa: E402

PACKAGE_FILES = ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav", "training_meta.json", "preview.wav", "preview.mp3")


def build_voice_json(adapter: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """The adapter's own ``voice.json`` (training facts: duration, epochs, base model) updated with the spec and sanitised."""
    base = voice_info.read_voice_json(adapter) or {}
    merged = {**base, **{k: v for k, v in spec.items() if v not in (None, "")}}
    info = voice_info.normalize_info(merged, fallback_id=str(spec.get("id") or adapter.name))
    info["id"] = str(spec.get("id") or info.get("id") or adapter.name)
    return info


def index_entry(info: Dict[str, Any], zip_path: Path, url: str) -> Dict[str, Any]:
    """The object for the index ``voices`` list."""
    data = zip_path.read_bytes()
    entry = {"id": info["id"], "name": info["name"], "language": info.get("language", ""), "author": info.get("author", ""),
             "license": info["license"], "voice_type": info.get("voice_type", ""), "description": info.get("description", ""),
             "base_model": info.get("base_model", ""), "url": url, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    for key in ("names", "descriptions", "license_url"):
        if info.get(key):
            entry[key] = info[key]
    return entry


def build(adapter: Path, spec: Dict[str, Any], out: Path, url: str) -> Dict[str, Any]:
    """Create the folder, the zip and the entry; returns the entry."""
    adapter, out = Path(adapter), Path(out)
    if not (adapter / "adapter_model.safetensors").is_file() or not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"{adapter} is not a trained voice folder (adapter_model.safetensors / adapter_config.json missing)")
    info = build_voice_json(adapter, spec)
    folder = out / info["id"]
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for name in PACKAGE_FILES:
        if (adapter / name).is_file():
            shutil.copy2(adapter / name, folder / name)
    voice_info.write_voice_json(folder, info)
    zip_path = out / f"{info['id']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f"{info['id']}/{f.name}")
    entry = index_entry(info, zip_path, url)
    (out / f"{info['id']}.index-entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--url", default="https://example.org/voices/VOICE.zip", help="where the zip will be hosted (https)")
    a = ap.parse_args(argv)
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    entry = build(a.adapter, spec, a.out, a.url.replace("VOICE", str(spec.get("id", "voice"))))
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
