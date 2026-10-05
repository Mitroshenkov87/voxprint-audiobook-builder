"""The voice library: trained or imported voices stored under ``%LOCALAPPDATA%\\Voxprint\\voices\\<id>\\``.

A voice is a folder with a LoRA adapter (``adapter_model.safetensors`` + ``adapter_config.json``), the reference clip
(``ref_sample.wav``, also the preview sample), ``training_meta.json`` and :mod:`core.voice_info`'s ``voice.json``.

* Training registers its result here automatically (:meth:`VoiceLibrary.add_from_adapter`).
* Voices can be imported from a folder or a ``.zip`` (:meth:`import_folder`, :meth:`import_zip`) - zip extraction is
  hardened: only known file names are extracted, nothing is written outside the voice folder, sizes are capped.
* Only folders **inside the library root** are ever deleted or overwritten.

Pure Python (no Qt, no torch); the root folder is injectable so tests use a temporary directory.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from core import voice_info
from core.errors import VoiceLibraryError
from core.i18n import tr
from infra import paths

log = logging.getLogger("voxprint.voices")

#: Files that make a usable adapter.
REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")
#: Everything that may live in a voice folder; any other file in an imported archive/folder is ignored.
ALLOWED_FILES = REQUIRED_FILES + ("ref_sample.wav", "training_meta.json", "preview.wav", "consent_statement.wav", voice_info.VOICE_FILENAME)
MAX_FILE_BYTES = 1 << 30            # 1 GiB per file
MAX_TOTAL_BYTES = 2 << 30           # 2 GiB per voice
PREVIEW_FILES = ("preview.wav", "ref_sample.wav")
_ID_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def slugify(name: str) -> str:
    """Folder-safe identifier from a voice name (letters/digits/``-``/``_`` kept, Unicode letters allowed)."""
    s = _ID_RE.sub("-", (name or "").strip().lower()).strip("-_")
    return s[:48] or "voice"


def adapter_complete(folder: Path) -> bool:
    """True if ``folder`` contains the files a LoRA adapter needs."""
    return all((Path(folder) / n).is_file() for n in REQUIRED_FILES)


@dataclass
class VoiceRecord:
    """One voice of the library: its folder and the normalized ``voice.json`` content."""
    id: str
    path: Path
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Display name (the entry of ``names`` for the UI language if the voice has one, else ``name``)."""
        return self._localized("names", str(self.info.get("name") or self.id))

    def _localized(self, key: str, default: str) -> str:
        """``info[key][ui language]`` if present, else ``default`` (never raises)."""
        table = self.info.get(key)
        if isinstance(table, dict) and table:
            try:
                from core.i18n import get_language

                hit = table.get(get_language())
                if hit:
                    return str(hit)
            except Exception:  # noqa: BLE001 - a display text must never fail
                pass
        return default

    @property
    def description(self) -> str:
        """Description in the UI language (``descriptions``), else ``description``."""
        return self._localized("descriptions", str(self.info.get("description") or ""))

    @property
    def test_only(self) -> bool:
        """True for test-only voices: test/demo use only, no public release of the audio, no commercial use."""
        return self.license == voice_info.LICENSE_TEST_ONLY

    @property
    def language(self) -> str:
        """Language of the voice (as recorded at training time)."""
        return str(self.info.get("language", ""))

    @property
    def license(self) -> str:
        """SPDX-like licence id."""
        return str(self.info.get("license", voice_info.DEFAULT_LICENSE))

    @property
    def scope(self) -> str:
        """Usage scope: ``commercial`` / ``public_noncommercial`` / ``private_only`` (consent mark, else derived from the licence)."""
        from core import consent

        return consent.scope_of(self.info)

    @property
    def commercial_use(self) -> bool:
        """Derived from the licence: may the narration be used commercially?"""
        return bool(self.info.get("commercial_use", False))

    @property
    def preview_path(self) -> Optional[Path]:
        """A short sample of the voice (``preview.wav`` or the training reference clip), if present."""
        for n in PREVIEW_FILES:
            if (self.path / n).is_file():
                return self.path / n
        return None

    @property
    def base_model(self) -> str:
        """Hugging Face repo id of the base model the adapter was trained on."""
        return str(self.info.get("base_model", ""))

    @property
    def ref_text(self) -> str:
        """Text spoken in ``ref_sample.wav`` (from ``training_meta.json``); needed to clone the voice at synthesis time."""
        import json

        try:
            meta = json.loads((self.path / "training_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(meta.get("ref_sample_text", "")) if isinstance(meta, dict) else ""


class VoiceLibrary:
    """File-system backed registry of voices.  ``root`` defaults to :func:`infra.paths.voices_dir`."""

    def __init__(self, root: Optional[Path] = None) -> None:
        """Use ``root`` (created on demand) as the library folder."""
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        """The library folder (created if missing)."""
        r = self._root if self._root is not None else paths.voices_dir()
        r.mkdir(parents=True, exist_ok=True)
        return r

    # ------------------------------------------------------------------ reading
    def _record(self, folder: Path) -> Optional[VoiceRecord]:
        """Load one voice folder; ``None`` if it is not a usable voice."""
        if not folder.is_dir() or not adapter_complete(folder):
            return None
        info = voice_info.normalize_info(voice_info.read_voice_json(folder) or {}, fallback_id=folder.name)
        info["id"] = folder.name                       # the folder name is the identity
        return VoiceRecord(folder.name, folder, info)

    def list_voices(self) -> List[VoiceRecord]:
        """All usable voices, newest first (then by name)."""
        out = [r for d in sorted(self.root.iterdir()) if not d.name.startswith(".")
               and (r := self._record(d)) is not None]
        out.sort(key=lambda r: r.name.lower())
        out.sort(key=lambda r: r.info.get("created", ""), reverse=True)
        return out

    def get(self, voice_id: str) -> Optional[VoiceRecord]:
        """The voice with this id, or ``None``."""
        folder = self._folder_for(voice_id)
        return self._record(folder) if folder else None

    def is_empty(self) -> bool:
        """True if the library holds no usable voice."""
        return not self.list_voices()

    def _folder_for(self, voice_id: str) -> Optional[Path]:
        """Folder of ``voice_id`` if it is a direct child of the root (no path tricks), else ``None``."""
        if not voice_id or voice_id != Path(voice_id).name or voice_id in (".", ".."):
            return None
        return self.root / voice_id

    def unique_id(self, name: str, voice_type: Optional[str] = None) -> str:
        """A free folder name derived from ``name`` (``anna``, ``anna-2``, ...); with ``voice_type`` it carries the type
        (``anna_male``, ``anna_female``, ``anna_unspecified``, then ``anna_male-2`` ...)."""
        base = slugify(name)
        if voice_type is not None:
            base = slugify(voice_info.with_type_suffix(base, voice_type))
        cand, n = base, 2
        while (self.root / cand).exists():
            cand, n = f"{base}-{n}", n + 1
        return cand

    # ------------------------------------------------------------------ adding
    def add_from_adapter(self, adapter_dir: Path, info: Optional[Dict[str, Any]] = None,
                         name: str = "", typed_id: bool = False) -> VoiceRecord:
        """Copy an adapter folder into the library and write its ``voice.json``; returns the new record.

        ``typed_id`` puts the voice type into the folder name / id (``anna_male``) - used for freshly trained voices.
        ``info`` is the dictionary from :func:`core.voice_info.build_voice_info` (read from the folder's own
        ``voice.json`` when omitted).  The source folder is left untouched.
        """
        src = Path(adapter_dir)
        if not adapter_complete(src):
            raise VoiceLibraryError(tr("err.voice_invalid"), details=str(src))
        data = voice_info.normalize_info(info if info is not None else (voice_info.read_voice_json(src) or {}),
                                         fallback_id=src.name)
        if name:
            data["name"] = voice_info.clean_line(name)
        vid = self.unique_id(data["name"] or src.name, data.get("voice_type", "") if typed_id else None)
        dst = self.root / vid
        tmp = Path(tempfile.mkdtemp(prefix=f".{vid}.", dir=self.root))
        try:
            for fname in ALLOWED_FILES:
                if (src / fname).is_file() and fname != voice_info.VOICE_FILENAME:
                    shutil.copy2(src / fname, tmp / fname)
            data["id"] = vid
            voice_info.write_voice_json(tmp, data)
            tmp.rename(dst)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        log.info("voice added: %s", vid)
        return VoiceRecord(vid, dst, data)

    def import_folder(self, folder: Path) -> VoiceRecord:
        """Import an adapter folder (as produced by training, or downloaded)."""
        return self.add_from_adapter(Path(folder))

    def import_zip(self, zip_path: Path, overrides: Optional[Dict[str, Any]] = None) -> VoiceRecord:
        """Import a voice from a ``.zip`` safely.

        Only files with :data:`ALLOWED_FILES` names are extracted (a single wrapping folder inside the archive is
        accepted); directory traversal, absolute paths, symlink-like names and oversized members are rejected.
        ``overrides`` (e.g. licence data declared by a repository index) replace fields of the archive's ``voice.json``.
        """
        stage = Path(tempfile.mkdtemp(prefix=".import.", dir=self.root))
        try:
            self._safe_extract(Path(zip_path), stage)
            info = voice_info.read_voice_json(stage) or {}
            if overrides:
                info = {**info, **{k: v for k, v in overrides.items() if v not in (None, "")}}
                if overrides.get("license") and not overrides.get("license_url"):
                    info["license_url"] = ""          # do not keep the archive's URL for a different licence
            return self.add_from_adapter(stage, info if info else None, name=str(info.get("name", "")) if info else "")
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _safe_extract(zip_path: Path, dest: Path) -> None:
        """Extract the allowed members of ``zip_path`` flat into ``dest`` (see :meth:`import_zip`)."""
        try:
            zf = zipfile.ZipFile(zip_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise VoiceLibraryError(tr("err.voice_zip_unsafe"), details=str(exc)) from exc
        total = 0
        with zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                raw = member.filename.replace("\\", "/")
                pure = PurePosixPath(raw)
                if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", raw):
                    raise VoiceLibraryError(tr("err.voice_zip_unsafe"), details=raw)
                if len(pure.parts) > 2 or pure.name not in ALLOWED_FILES:
                    continue                                     # unknown file or deeper nesting: ignored
                if member.file_size > MAX_FILE_BYTES:
                    raise VoiceLibraryError(tr("err.voice_zip_unsafe"), details=f"{raw}: too large")
                total += member.file_size
                if total > MAX_TOTAL_BYTES:
                    raise VoiceLibraryError(tr("err.voice_zip_unsafe"), details="archive too large")
                with zf.open(member) as src, open(dest / pure.name, "wb") as out:
                    copied = 0
                    while chunk := src.read(1 << 20):
                        copied += len(chunk)
                        if copied > MAX_FILE_BYTES:
                            raise VoiceLibraryError(tr("err.voice_zip_unsafe"), details=f"{raw}: too large")
                        out.write(chunk)
        if not adapter_complete(dest):
            raise VoiceLibraryError(tr("err.voice_invalid"), details="adapter files missing in the archive")

    # ------------------------------------------------------------------ changing
    def update(self, voice_id: str, **fields: Any) -> VoiceRecord:
        """Change editable details (``name, author, license, license_url, voice_type, description``) and save voice.json."""
        rec = self.get(voice_id)
        if rec is None:
            raise VoiceLibraryError(tr("err.voice_not_found"), details=voice_id)
        info = dict(rec.info)
        for key in ("name", "author", "license", "license_url", "voice_type", "description", "consent"):
            if key in fields and fields[key] is not None:
                info[key] = fields[key]
        if "license" in fields and "license_url" not in fields:
            info["license_url"] = ""                                   # re-derived for the new licence
        # Per-language names/descriptions (downloaded voices) are shown *instead of* name/description, so an edit
        # looked lost; the user's own text now replaces them.
        if fields.get("name") is not None and fields["name"] != rec.info.get("name"):
            info.pop("names", None)
        if fields.get("description") is not None and fields["description"] != rec.info.get("description"):
            info.pop("descriptions", None)
        info = voice_info.normalize_info(info, fallback_id=voice_id)
        new_id, path = voice_id, rec.path
        if fields.get("name") and slugify(info["name"]) != slugify(str(rec.info.get("name") or "")):
            cand = self.unique_id(info["name"])
            try:                                           # the folder follows the new name; if Windows holds a file
                rec.path.rename(self.root / cand)          # open (voice in use), keep the old folder - the name still changes
                new_id, path = cand, self.root / cand
            except OSError as exc:
                log.warning("cannot rename voice folder %s -> %s: %s", voice_id, cand, exc)
        info["id"] = new_id
        voice_info.write_voice_json(path, info)
        return VoiceRecord(new_id, path, info)

    def confirm_consent(self, voice_id: str, scope: str, name: Optional[str] = None) -> VoiceRecord:
        """The user's one-click confirmation (or change) of the detected usage scope: marks the consent confirmed and maps the
        scope onto the licence fields (private only -> personal-only licence, public -> CC-BY-NC, commercial -> CC-BY)."""
        from core import consent

        rec = self.get(voice_id)
        if rec is None:
            raise VoiceLibraryError(tr("err.voice_not_found"), details=voice_id)
        block = dict(consent.clean_consent(rec.info.get("consent")) or consent.build_consent(None, scope=scope, method="manual",
                                                                                           recorded=False, confirmed=True))
        block.update(scope=scope if scope in consent.SCOPES else consent.DEFAULT_SCOPE, confirmed=True)
        if name is not None:
            block["name"] = name
        if block["method"] == "spoken":
            block["method"] = "spoken_confirmed"        # the user has seen the detected scope and confirmed or changed it
        return self.update(voice_id, consent=block, license=consent.license_for_scope(block["scope"]),
                           author=block["name"] or rec.info.get("author", ""))

    def delete(self, voice_id: str) -> None:
        """Delete a voice folder.  Refuses anything that is not a direct child of the library root."""
        folder = self._folder_for(voice_id)
        if folder is None or not folder.is_dir():
            raise VoiceLibraryError(tr("err.voice_not_found"), details=voice_id)
        if folder.resolve().parent != self.root.resolve():
            raise VoiceLibraryError(tr("err.voice_not_found"), details=voice_id)
        shutil.rmtree(folder)
        log.info("voice deleted: %s", voice_id)
