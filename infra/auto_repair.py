"""Auto-repair (Settings -> "Auto-repair"): check every component and model, re-download what is missing, repair what is broken.

Built on the existing pieces instead of duplicating them:

* **environment** - :func:`infra.install_state.verify_install` (the ``--verify-install`` check); a source/venv install is
  rebuilt with :func:`infra.install_state.repair_install` (the ``--repair`` path), an installer build only reports;
* **runtime modules** (thin build) - :func:`infra.modules.modules` / :func:`infra.modules.install` (resumable, SHA-256);
* **ffmpeg** (Windows) - :func:`infra.assets.ensure_ffmpeg_tool` (smoke test, pinned build re-installed when broken);
* **models** - every file of each known model is checked by size + SHA-256 against ``infra/model_mirrors.json`` (the
  verified hashes of our backup mirror).  Missing models are fetched with :func:`infra.model_downloader.ensure_model`
  (all sources, fallbacks, resume).  A model with bad files: the bad files are deleted, the folder becomes
  ``<name>.partial`` and ``ensure_model`` completes it (only the missing files are downloaded), then it is re-checked.
  Copies of other programs (Hugging Face cache ...) are read-only and never touched.

Everything is injectable (tests run without network, GPU or real downloads).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from core.errors import CancelledByUser
from core.events import CancelToken
from core.i18n import tr

log = logging.getLogger("voxprint.auto_repair")

OK, REPAIRED, DOWNLOADED, FAILED, SKIPPED = "ok", "repaired", "downloaded", "failed", "skipped"
Progress = Callable[[float, str], None]


@dataclass
class Item:
    """One checked thing: ``kind`` (environment | module | ffmpeg | model), its name, the outcome and a short detail."""
    kind: str
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    """All checked items; :meth:`summary` is the localized final line."""
    items: List[Item] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.status != FAILED for i in self.items)

    def count(self, *statuses: str) -> int:
        return sum(1 for i in self.items if i.status in statuses)

    def summary(self) -> str:
        text = tr("autorepair.summary", checked=len(self.items), fixed=self.count(REPAIRED, DOWNLOADED),
                  failed=self.count(FAILED))
        bad = [f"{i.name} ({i.detail})" if i.detail else i.name for i in self.items if i.status == FAILED]
        return text + ("\n" + tr("autorepair.failed_list", names="; ".join(bad)) if bad else "")


# ------------------------------------------------------------------------------------------------ models
def model_repos() -> List[str]:
    """Models to check: the ones the program needs on this PC plus every known model that exists locally."""
    from infra import model_downloader as md
    from infra import model_mirrors

    known = list(model_mirrors.load())
    need: List[str] = []
    try:
        from workers.pipeline_runner import required_model_repos

        need = list(required_model_repos())
    except Exception as exc:  # noqa: BLE001 - GPU detection may fail; then only the local models are checked
        log.warning("required models unknown: %s", exc)
    present = [r for r in known if md.local_dir_for(r).exists() or md.partial_dir_has_data(r)]
    return list(dict.fromkeys(need + present))


def _patterns(repo: str) -> Tuple[str, ...]:
    """File patterns of a text model (e.g. Opus-MT keeps only the PyTorch weights); () = the whole repository."""
    try:
        from infra import text_models

        for m in text_models.REGISTRY:
            if m.repo == repo:
                return tuple(m.files)
    except Exception:  # noqa: BLE001
        pass
    return ()


def expected_files(repo: str) -> Optional[Tuple[str, Dict[str, Dict[str, Any]]]]:
    """``(revision, {file: {"size", "sha256"}})`` from the verified mirror manifest, or None (no hashes known)."""
    from infra import model_mirrors

    e = model_mirrors.load().get(repo)
    if e is None:
        return None
    try:
        return e.source_revision, e.downloadable(_patterns(repo) or None)
    except model_mirrors.MirrorError:
        return None


def bad_files(folder: Path, files: Dict[str, Dict[str, Any]], cancel: Optional[CancelToken] = None,
              on_file: Callable[[str], None] = lambda n: None) -> List[str]:
    """Files of ``folder`` that are missing or differ in size / SHA-256 (size first: cheap)."""
    from infra import model_release

    out = []
    for name, meta in files.items():
        if cancel is not None:
            cancel.check()
        on_file(name)
        p = folder / Path(*name.split("/"))
        if not model_release.file_ok(p, meta):
            out.append(name)
    return out


def _revision(folder: Path) -> str:
    try:
        return (folder / ".revision").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def check_and_fix_model(repo: str, progress: Progress, cancel: CancelToken, ensure: Callable[..., Path]) -> Item:
    """Check one model by hash and repair / download it; never raises except on cancel."""
    from infra import model_downloader as md

    short = repo.split("/")[-1]
    folder = md.local_dir_for(repo)
    spec = expected_files(repo)
    patterns = list(_patterns(repo)) or None
    kw: Dict[str, Any] = {"allow_patterns": patterns} if patterns else {}
    try:
        if not md.verify_local_model(folder):
            ext = md.external_model(repo)
            if ext is not None:
                return Item("model", short, SKIPPED, tr("autorepair.external", where=str(ext.location)))
            progress(0.0, tr("autorepair.downloading", name=short))
            ensure(repo, lambda s, f, m: progress(f, m), **kw)           # all sources, fallbacks, resume
            return Item("model", short, DOWNLOADED)
        if spec is None:
            return Item("model", short, OK, tr("autorepair.no_hashes"))
        revision, files = spec
        if _revision(folder) and _revision(folder) != revision:
            return Item("model", short, OK, tr("autorepair.other_revision"))
        n = max(1, len(files))
        seen = [0]

        def on_file(name: str) -> None:
            seen[0] += 1
            progress(seen[0] / n, tr("autorepair.checking", name=short, file=name))

        bad = bad_files(folder, files, cancel, on_file)
        if not bad:
            return Item("model", short, OK)
        log.warning("auto-repair: %s has %d bad file(s): %s", repo, len(bad), ", ".join(bad[:5]))
        for name in bad:
            (folder / Path(*name.split("/"))).unlink(missing_ok=True)
        partial = folder.with_name(folder.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)           # a stale interrupted download: the complete-but-damaged copy is better
        os.replace(folder, partial)                          # ensure_model resumes a .partial folder: only the bad files are fetched
        progress(0.0, tr("autorepair.repairing", name=short, n=len(bad)))
        ensure(repo, lambda s, f, m: progress(f, m), root=folder.parent, revision=revision, **kw)
        still = bad_files(folder, files, cancel)
        if still:
            return Item("model", short, FAILED, tr("autorepair.still_bad", n=len(still)))
        return Item("model", short, REPAIRED, tr("autorepair.fixed_files", n=len(bad)))
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - one broken model must not stop the others
        log.exception("auto-repair of %s failed", repo)
        msg = getattr(exc, "user_message", "") or type(exc).__name__
        return Item("model", short, FAILED, msg)


# ------------------------------------------------------------------------------------------------ components
def _default_env_repair(progress: Progress) -> Tuple[int, str]:
    from infra import install_state
    from infra.updater import run_subprocess

    return install_state.repair_install(run_subprocess, shutil.which, progress)


def check_environment(progress: Progress, verify: Optional[Callable[[], Any]] = None,
                      repair: Optional[Callable[[Progress], Tuple[int, str]]] = None) -> Item:
    """The ``--verify-install`` check; a source/venv install is repaired like ``--repair``."""
    from infra import install_state

    rep = (verify or (lambda: install_state.verify_install()))()
    if rep.ok:
        return Item("environment", tr("autorepair.environment"), OK)
    reasons = "; ".join(install_state.describe_reasons(rep))
    if getattr(sys, "frozen", False) and repair is None:
        # an installer build carries its libraries inside: missing ones come back with the modules step or a reinstall
        return Item("environment", tr("autorepair.environment"), FAILED, reasons)
    rc, text = (repair or _default_env_repair)(progress)
    return Item("environment", tr("autorepair.environment"), REPAIRED if rc == 0 else FAILED, "" if rc == 0 else text)


def check_modules(progress: Progress, cancel: CancelToken, api: Any = None) -> List[Item]:
    """Thin build: every required runtime module installed with its recorded SHA-256; missing ones are re-installed."""
    from infra import modules as default_api

    api = api or default_api
    if not api.is_thin():
        return []
    try:
        mods = api.modules(api.load_manifest(offline_ok=True))
    except Exception as exc:  # noqa: BLE001
        return [Item("module", tr("autorepair.modules"), FAILED, str(exc))]
    missing = [m for m in mods if m.required and not m.installed]
    items = [Item("module", m.title, OK) for m in mods if m.installed]
    if missing:
        try:
            api.install([m.id for m in missing], progress, cancelled=lambda: cancel.is_cancelled)
            items += [Item("module", m.title, DOWNLOADED) for m in missing]
        except Exception as exc:  # noqa: BLE001
            if cancel.is_cancelled:
                raise CancelledByUser() from exc
            items += [Item("module", m.title, FAILED, str(exc)) for m in missing]
    return items


def check_ffmpeg(progress: Progress, ensure: Optional[Callable[..., Any]] = None) -> List[Item]:
    """Windows: ffmpeg passes its smoke test, else the pinned build is (re)installed (the bundled one stays the fallback)."""
    if ensure is None:
        if sys.platform != "win32":
            return []
        from infra import assets

        ensure = assets.ensure_ffmpeg_tool
    path = ensure(progress)
    return [Item("ffmpeg", "ffmpeg", OK if path else SKIPPED, "" if path else tr("autorepair.ffmpeg_bundled"))]


# ------------------------------------------------------------------------------------------------ all
def run(progress: Progress = lambda f, m: None, cancel: Optional[CancelToken] = None, *,
        repos: Optional[Iterable[str]] = None, ensure: Optional[Callable[..., Path]] = None,
        env_verify: Optional[Callable[[], Any]] = None, env_repair: Optional[Callable[[Progress], Tuple[int, str]]] = None,
        modules_api: Any = None, ffmpeg_ensure: Optional[Callable[..., Any]] = None) -> Report:
    """Check and repair everything; ``progress(fraction 0..1, message)``.  Raises ``CancelledByUser`` when cancelled."""
    from infra import model_downloader as md

    cancel = cancel or CancelToken()
    ensure = ensure or md.ensure_model
    report = Report()
    repo_list = list(repos) if repos is not None else model_repos()
    steps = 3 + len(repo_list)
    done = [0]
    last = [0.0]

    def sub(f: float, m: str) -> None:
        # one item = one share of the bar; the share is passed twice for a repaired model (check, then download), so the
        # value is kept monotonic: the bar never goes back
        last[0] = max(last[0], min(1.0, (done[0] + max(0.0, min(1.0, f))) / steps))
        progress(last[0], m)

    def step() -> None:
        done[0] += 1
        cancel.check()

    sub(0.0, tr("autorepair.environment"))
    report.items.append(check_environment(sub, env_verify, env_repair))
    step()
    report.items += check_modules(sub, cancel, modules_api)
    step()
    report.items += check_ffmpeg(sub, ffmpeg_ensure)
    step()
    for repo in repo_list:
        report.items.append(check_and_fix_model(repo, sub, cancel, ensure))
        step()
    progress(1.0, report.summary())
    log.info("auto-repair: %s", "; ".join(f"{i.kind}:{i.name}={i.status}" for i in report.items))
    return report
