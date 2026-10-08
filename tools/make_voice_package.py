"""Build a distributable voice package for the online voices index.

    python tools/make_voice_package.py --adapter <trained voice folder> --spec tools/voice_specs/example-open-voice.json \\
        --out dist_voices --url https://huggingface.co/<user>/<repo>/resolve/main/open-voice.zip

Writes ``<out>/<id>/`` (the files the app needs, with the final ``voice.json``, plus ``README.md`` - a Hugging Face model card
with YAML metadata, ignored by the app's importer), ``<out>/<id>.zip``, and ``<out>/<id>.index-entry.json`` - one object for
the ``voices`` list of the repository's ``index.json`` (with SHA-256 and size).  The spec is a small JSON with the fields to put into ``voice.json`` (names, descriptions, licence, consent ...).
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
    if spec.get("voice_type") and not spec.get("gender") and not spec.get("age_group"):
        # an older spec with only voice_type replaces the adapter's gender / age group (voice_type is derived from them)
        merged["gender"], merged["age_group"] = voice_info.split_voice_type(str(spec["voice_type"]))
    info = voice_info.normalize_info(merged, fallback_id=str(spec.get("id") or adapter.name))
    info["id"] = str(spec.get("id") or info.get("id") or adapter.name)
    return info


def index_entry(info: Dict[str, Any], zip_path: Path, url: str) -> Dict[str, Any]:
    """The object for the index ``voices`` list."""
    data = zip_path.read_bytes()
    entry = {"id": info["id"], "name": info["name"], "language": info.get("language", ""), "author": info.get("author", ""),
             "license": info["license"], "voice_type": info.get("voice_type", ""), "description": info.get("description", ""),
             "base_model": info.get("base_model", ""), "url": url, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    for key in ("names", "descriptions", "license_url") + DETAIL_FIELDS:
        if info.get(key):
            entry[key] = info[key]
    return entry


#: Optional schema-3 details copied into the index entry and the model card when set.
DETAIL_FIELDS = ("gender", "age_group", "speaker", "prepared_by", "organization", "project_url")
#: voice.json licence id -> Hugging Face ``license:`` id; anything else is published as ``other`` with ``license_name``.
HF_LICENSES = {"CC0-1.0": "cc0-1.0", "CC-BY-4.0": "cc-by-4.0", "CC-BY-SA-4.0": "cc-by-sa-4.0", "CC-BY-NC-4.0": "cc-by-nc-4.0",
               "CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0"}


def _md(text: Any) -> str:
    """A value safe inside a markdown table cell (voice.json text is single-line already)."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;")


def model_card(info: Dict[str, Any], url: str = "") -> str:
    """``README.md`` for a Hugging Face repo hosting the package: YAML metadata (licence, language, base model, tags) and
    a table with every voice.json detail that is set.  YAML strings are written as JSON strings, which YAML reads as-is."""
    from core.languages import language_name

    lic = str(info.get("license", ""))
    lang = str(info.get("language", ""))
    meta = [("license", json.dumps(HF_LICENSES.get(lic, "other")))]
    if lic not in HF_LICENSES:
        meta.append(("license_name", json.dumps(lic.replace("/", "-") or "unknown")))
        if info.get("license_url"):
            meta.append(("license_link", json.dumps(info["license_url"])))
    lines = ["---"] + [f"{k}: {v}" for k, v in meta]
    if lang:
        lines += ["language:", f"- {json.dumps(lang.split('-')[0])}"]      # the Hub expects ISO 639-1 codes here
    if info.get("base_model"):
        lines.append(f"base_model: {json.dumps(info['base_model'])}")
    lines += ["library_name: peft", "pipeline_tag: text-to-speech", "tags:", "- voxprint", "- qwen3-tts", "- lora",
              "- voice-clone", "---", "", f"# {_md(info.get('name') or info.get('id', 'voice'))}", ""]
    if info.get("description"):
        lines += [_md(info["description"]), ""]
    rows = [("Language", f"{language_name(lang)} (`{lang}`)" if lang else ""),
            ("Gender", info.get("gender", "")), ("Age group", info.get("age_group", "")),
            ("Speaker", info.get("speaker", "")), ("Prepared by", info.get("prepared_by", "")),
            ("Organization", info.get("organization", "")), ("Author", info.get("author", "")),
            ("Project", info.get("project_url", "")),
            ("License", f"[{_md(lic)}]({info['license_url']})" if info.get("license_url") else _md(lic)),
            ("Commercial use", "yes" if info.get("commercial_use") else "no"),
            ("Base model", info.get("base_model", "")),
            ("Training speech", f"{float(info.get('duration', 0)) / 60:.0f} min" if info.get("duration") else ""),
            ("Epochs", str(info.get("epochs") or "")), ("Created (UTC)", info.get("created", "")),
            ("Package", url)]
    lines += ["| Field | Value |", "|---|---|"]
    lines += [f"| {k} | {v if k in ('License', 'Language') else _md(v)} |" for k, v in rows if v]
    lines += ["", "A LoRA voice for [Voxprint](https://github.com/Mitroshenkov87/voxprint-audiobook-builder): download the zip "
              "and import it in the Voices window. The licence covers the trained model; it does not replace the consent "
              "of the person whose voice it is.", ""]
    return "\n".join(lines)


def build(adapter: Path, spec: Dict[str, Any], out: Path, url: str, typed_names: bool = True) -> Dict[str, Any]:
    """Create the folder, the zip and the entry; returns the entry.

    With ``typed_names`` (default) the folder and the zip carry the voice type: ``<id>_<male|female|child|other|unspecified>``.
    ``voice.json`` inside always has the ``voice_type`` field."""
    adapter, out = Path(adapter), Path(out)
    if not (adapter / "adapter_model.safetensors").is_file() or not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"{adapter} is not a trained voice folder (adapter_model.safetensors / adapter_config.json missing)")
    info = build_voice_json(adapter, spec)
    base = voice_info.with_type_suffix(info["id"], info.get("voice_type", "")) if typed_names else info["id"]
    folder = out / base
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for name in PACKAGE_FILES:
        if (adapter / name).is_file():
            shutil.copy2(adapter / name, folder / name)
    voice_info.write_voice_json(folder, info)
    (folder / "README.md").write_text(model_card(info, url), encoding="utf-8")
    zip_path = out / f"{base}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f"{base}/{f.name}")
    entry = index_entry(info, zip_path, url)
    (out / f"{base}.index-entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def main(argv: Optional[list] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--url", default="https://example.org/voices/VOICE.zip",
                    help="where the zip will be hosted (https); VOICE is replaced by the package name (<id>_<type>)")
    ap.add_argument("--plain-names", action="store_true", help="name the zip <id>.zip without the voice type")
    a = ap.parse_args(argv)
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    typed = not a.plain_names
    name = str(spec.get("id", "voice"))
    if typed:      # the same (derived) type the package folder gets in build()
        name = voice_info.with_type_suffix(name, build_voice_json(a.adapter, spec).get("voice_type", ""))
    entry = build(a.adapter, spec, a.out, a.url.replace("VOICE", name), typed_names=typed)
    print(json.dumps(entry, indent=2))      # ASCII-escaped: safe on any Windows console code page
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
