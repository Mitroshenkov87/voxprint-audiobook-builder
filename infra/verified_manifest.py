"""Манифест «проверено Voxprint»: закреплённые версии пакетов и ревизии моделей, с которыми приложение тестировалось.

* Встроенный файл verified_manifest.json лежит рядом с кодом (и в exe).
* Необязательный удалённый манифест (REMOTE_MANIFEST_URL / переменная VOXPRINT_MANIFEST_URL) подхватывается,
  если он корректен и новее встроенного; последняя удачная копия кэшируется в state/.
* Безопасность: из манифеста принимаются только известные пакеты и модели из списков version_manager,
  версии проверяются как PEP 440, ревизии - как 40-значный sha. Ничего другого pip не получит.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from packaging.version import InvalidVersion, Version

from infra import paths

log = logging.getLogger("voxprint.manifest")

SCHEMA = 1
#: TODO: после публикации репозитория сюда можно прописать raw-URL манифеста; пусто = только встроенный.
REMOTE_MANIFEST_URL = ""
BUNDLED_PATH = Path(__file__).with_name("verified_manifest.json")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Manifest:
    name: str = "Voxprint verified set"
    date: str = ""
    note: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    source: str = "bundled"      # bundled | remote | cache

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "name": self.name, "date": self.date, "note": self.note,
                "packages": self.packages, "models": self.models}


def parse_manifest(data: Any, allowed_packages: Optional[List[str]] = None,
                   allowed_models: Optional[List[str]] = None, source: str = "bundled") -> Manifest:
    """Проверяет структуру; бросает ValueError при любой неточности."""
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unsupported manifest schema")
    pk = data.get("packages") or {}
    md = data.get("models") or {}
    if not isinstance(pk, dict) or not isinstance(md, dict):
        raise ValueError("bad manifest sections")
    for name, ver in pk.items():
        if allowed_packages is not None and name not in allowed_packages:
            raise ValueError(f"package not allowed: {name}")
        try:
            Version(str(ver))
        except InvalidVersion as exc:
            raise ValueError(f"bad version for {name}: {ver}") from exc
    for repo, sha in md.items():
        if allowed_models is not None and repo not in allowed_models:
            raise ValueError(f"model not allowed: {repo}")
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise ValueError(f"bad revision for {repo}")
    return Manifest(str(data.get("name", "")), str(data.get("date", "")), str(data.get("note", "")),
                    {k: str(v) for k, v in pk.items()}, dict(md), source)


def _allowed():
    from infra.version_manager import TRACKED_MODELS, TRACKED_PACKAGES

    return list(TRACKED_PACKAGES), list(TRACKED_MODELS)


def load_bundled() -> Manifest:
    pk, md = _allowed()
    return parse_manifest(json.loads(BUNDLED_PATH.read_text(encoding="utf-8")), pk, md, "bundled")


def cache_file() -> Path:
    return paths.state_dir() / "verified_manifest.json"


def load_manifest(fetch_json: Optional[Callable[[str], Any]] = None, url: Optional[str] = None) -> Manifest:
    """Встроенный манифест; удалённый (или кэш) заменяет его, только если корректен и не старее."""
    best = load_bundled()
    pk, md = _allowed()
    url = url if url is not None else os.environ.get("VOXPRINT_MANIFEST_URL", REMOTE_MANIFEST_URL)
    candidate: Optional[Manifest] = None
    if url and fetch_json is not None:
        try:
            candidate = parse_manifest(fetch_json(url), pk, md, "remote")
            cache_file().write_text(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - сеть/формат: тихо остаёмся на том, что есть
            log.warning("remote manifest ignored: %s", exc)
            candidate = None
    if candidate is None:
        try:
            candidate = parse_manifest(json.loads(cache_file().read_text(encoding="utf-8")), pk, md, "cache")
        except (OSError, ValueError):
            candidate = None
    if candidate is not None and candidate.date >= best.date:
        best = candidate
    return best


NODEPS_PREFIXES = ("qwen-",)   # qwen-asr / qwen-tts закрепляют разные transformers -> ставятся с --no-deps


def requirements_lines(m: Optional[Manifest] = None, nodeps: bool = False) -> List[str]:
    m = m or load_bundled()
    return [f"{n}=={v}" for n, v in sorted(m.packages.items()) if n.startswith(NODEPS_PREFIXES) == nodeps]


def main(argv: Optional[List[str]] = None) -> int:
    """python -m infra.verified_manifest [--nodeps]  -> строки requirements из встроенного манифеста."""
    import sys

    nodeps = "--nodeps" in (argv if argv is not None else sys.argv[1:])
    print("# Сгенерировано: python -m infra.verified_manifest" + (" --nodeps" if nodeps else "")
          + "  (версии из infra/verified_manifest.json)")
    if nodeps:
        print("# Ставить так: pip install --no-deps -r requirements-nodeps.txt")
    for line in requirements_lines(nodeps=nodeps):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
