""""Verified by Voxprint" manifest: pinned package versions and model revisions the app was tested with.

* The bundled ``verified_manifest.json`` lives next to the code (and inside the exe).
* An optional remote manifest (``REMOTE_MANIFEST_URL`` / the ``VOXPRINT_MANIFEST_URL`` variable) is picked up if it
  is valid and not older than the bundled one; the last good copy is cached in ``state/``.
* Safety: only known packages and models from the ``version_manager`` lists are accepted, versions must be valid
  PEP 440 and revisions 40-character shas.  pip is never handed anything else.
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
#: TODO: once the repository is published, put the raw URL of the manifest here; empty = bundled manifest only.
REMOTE_MANIFEST_URL = ""
BUNDLED_PATH = Path(__file__).with_name("verified_manifest.json")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Manifest:
    """A parsed manifest: package -> exact version, model repo -> commit sha, plus where it came from (``source``)."""
    name: str = "Voxprint verified set"
    date: str = ""
    note: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    source: str = "bundled"      # bundled | remote | cache

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the on-disk JSON format (used for the cache file)."""
        return {"schema": SCHEMA, "name": self.name, "date": self.date, "note": self.note,
                "packages": self.packages, "models": self.models}


def parse_manifest(data: Any, allowed_packages: Optional[List[str]] = None,
                   allowed_models: Optional[List[str]] = None, source: str = "bundled") -> Manifest:
    """Validate the structure and return a :class:`Manifest`; raises ValueError on any irregularity."""
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
    """The (packages, models) allow-lists from :mod:`infra.version_manager` (imported lazily to avoid a cycle)."""
    from infra.version_manager import TRACKED_MODELS, TRACKED_PACKAGES

    return list(TRACKED_PACKAGES), list(TRACKED_MODELS)


def load_bundled() -> Manifest:
    """Load the manifest shipped with the program."""
    pk, md = _allowed()
    return parse_manifest(json.loads(BUNDLED_PATH.read_text(encoding="utf-8")), pk, md, "bundled")


def cache_file() -> Path:
    """Where the last good remote manifest is cached."""
    return paths.state_dir() / "verified_manifest.json"


def load_manifest(fetch_json: Optional[Callable[[str], Any]] = None, url: Optional[str] = None) -> Manifest:
    """The bundled manifest; a remote one (or the cache) replaces it only if it is valid and not older."""
    best = load_bundled()
    pk, md = _allowed()
    url = url if url is not None else os.environ.get("VOXPRINT_MANIFEST_URL", REMOTE_MANIFEST_URL)
    candidate: Optional[Manifest] = None
    if url and fetch_json is not None:
        try:
            candidate = parse_manifest(fetch_json(url), pk, md, "remote")
            cache_file().write_text(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - network/format problem: quietly keep what we have
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


NODEPS_PREFIXES = ("qwen-",)   # qwen-asr / qwen-tts pin different transformers versions -> they are installed with --no-deps


def requirements_lines(m: Optional[Manifest] = None, nodeps: bool = False) -> List[str]:
    """``name==version`` lines of the manifest; ``nodeps`` selects the packages that must be installed with ``--no-deps``."""
    m = m or load_bundled()
    return [f"{n}=={v}" for n, v in sorted(m.packages.items()) if n.startswith(NODEPS_PREFIXES) == nodeps]


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m infra.verified_manifest [--nodeps]`` prints requirements lines from the bundled manifest."""
    import sys

    nodeps = "--nodeps" in (argv if argv is not None else sys.argv[1:])
    print("# Generated by: python -m infra.verified_manifest" + (" --nodeps" if nodeps else "")
          + "  (versions from infra/verified_manifest.json)")
    if nodeps:
        print("# Install with: pip install --no-deps -r requirements-nodeps.txt")
    for line in requirements_lines(nodeps=nodeps):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
