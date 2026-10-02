"""Разработческий скрипт: скачивает ТЕКСТЫ лицензий компонентов в licenses/ (из репозиториев проектов и SPDX).

Запускается вручную при изменении credits.json; результат (папка licenses/) хранится в репозитории и
попадает в установщик. Для каждого файла проверяется, что он не пустой и (для известных лицензий) содержит
характерную фразу. Использование:  python tools/fetch_licenses.py [--only id1,id2]
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "licenses"
NAMES = ("LICENSE", "LICENSE.APACHE", "LICENSE.BSD", "LICENSE-MIT", "LICENSE-APACHE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "COPYING", "COPYING.txt", "LICENCE")
SPDX = "https://raw.githubusercontent.com/spdx/license-list-data/main/text/{}.txt"

# id файла -> ("repo", "owner/name") | ("url", прямой URL) | ("spdx", SPDX-идентификатор)
SOURCES = {
    "qwen-tts": ("repo", "QwenLM/Qwen3-TTS"),
    "qwen-asr": ("repo", "QwenLM/Qwen3-ASR"),
    "qwen-omni-utils": ("repo", "QwenLM/Qwen2-VL"),
    "pytorch": ("repo", "pytorch/pytorch"),
    "torchaudio": ("repo", "pytorch/audio"),
    "transformers": ("repo", "huggingface/transformers"),
    "peft": ("repo", "huggingface/peft"),
    "accelerate": ("repo", "huggingface/accelerate"),
    "huggingface_hub": ("repo", "huggingface/huggingface_hub"),
    "safetensors": ("repo", "huggingface/safetensors"),
    "bitsandbytes": ("repo", "bitsandbytes-foundation/bitsandbytes"),
    "numpy": ("repo", "numpy/numpy"),
    "scipy": ("repo", "scipy/scipy"),
    "librosa": ("repo", "librosa/librosa"),
    "soundfile": ("repo", "bastibe/python-soundfile"),
    "pydub": ("repo", "jiaaro/pydub"),
    "imageio-ffmpeg": ("repo", "imageio/imageio-ffmpeg"),
    "einops": ("repo", "arogozhnikov/einops"),
    "onnxruntime": ("repo", "microsoft/onnxruntime"),
    "nagisa": ("repo", "taishi-i/nagisa"),
    "sox": ("repo", "rabitt/pysox"),
    "ru-normalizr": ("repo", "NickZaitsev/ru-normalizr"),
    "rutextnorm": ("repo", "shigabeev/russian_tts_normalization"),
    "pymorphy3": ("repo", "no-plagiarism/pymorphy3"),
    "pymorphy3-dicts-ru": ("repo", "no-plagiarism/pymorphy3-dicts"),
    "num2words": ("repo", "savoirfairelinux/num2words"),
    "roman": ("repo", "zopefoundation/roman"),
    "eng_to_ipa": ("repo", "mphilli/English-to-IPA"),
    "packaging-apache": ("url", "https://raw.githubusercontent.com/pypa/packaging/main/LICENSE.APACHE"),
    "packaging-bsd": ("url", "https://raw.githubusercontent.com/pypa/packaging/main/LICENSE.BSD"),
    "tqdm": ("repo", "tqdm/tqdm"),
    "uv-mit": ("url", "https://raw.githubusercontent.com/astral-sh/uv/main/LICENSE-MIT"),
    "uv-apache": ("url", "https://raw.githubusercontent.com/astral-sh/uv/main/LICENSE-APACHE"),
    "pyinstaller": ("repo", "pyinstaller/pyinstaller"),
    "inno-setup": ("url", "https://raw.githubusercontent.com/jrsoftware/issrc/main/license.txt"),
    "tabler-icons": ("repo", "tabler/tabler-icons"),
    "ctc-forced-aligner": ("repo", "MahmoudAshraf97/ctc-forced-aligner"),
    "gpl-3.0": ("spdx", "GPL-3.0-only"),
    "qt-lgpl-3.0": ("spdx", "LGPL-3.0-only"),
    "Apache-2.0": ("spdx", "Apache-2.0"),
}


def get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "voxprint"}), timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None


def fetch(kind: str, ref: str) -> tuple[str | None, str]:
    if kind == "spdx":
        u = SPDX.format(ref)
        return get(u), u
    if kind == "url":
        return get(ref), ref
    for br in ("main", "master"):
        for n in NAMES:
            u = f"https://raw.githubusercontent.com/{ref}/{br}/{n}"
            t = get(u)
            if t and len(t.strip()) > 200:
                return t, u
    return None, ref


def main(argv: list[str]) -> int:
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    OUT.mkdir(exist_ok=True)
    index = {}
    bad = []
    for lid, (kind, ref) in SOURCES.items():
        if only and lid not in only:
            continue
        text, url = fetch(kind, ref)
        if not text:
            bad.append(lid)
            print("FAIL", lid, url)
            continue
        (OUT / f"{lid}.txt").write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        index[lid] = url
        print("ok  ", lid, len(text), url)
    old = {}
    if (OUT / "SOURCES.json").exists():
        old = json.loads((OUT / "SOURCES.json").read_text(encoding="utf-8"))
    old.update(index)
    (OUT / "SOURCES.json").write_text(json.dumps(old, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print("failed:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
