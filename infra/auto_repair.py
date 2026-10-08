"""Check & repair (Settings -> "Check & repair", formerly "Auto-repair"): check every component and model, re-download what is missing, repair what is broken.

Built on the existing pieces instead of duplicating them:

* **environment** - :func:`infra.install_state.verify_install` (the ``--verify-install`` check); a source/venv install is
  rebuilt with :func:`infra.install_state.repair_install` (the ``--repair`` path), an installer build only reports;
* **runtime modules** (thin build) - :func:`infra.modules.modules` / :func:`infra.modules.install` (resumable, SHA-256);
* **ffmpeg** (Windows) - :func:`infra.assets.ensure_ffmpeg_tool` (smoke test, pinned build re-installed when broken);
* **single-file models** - OpenVoice V2, DeepFilterNet, DNSMOS and Gemma (llama.cpp): size + SHA-256 of the pinned files,
  a damaged file is deleted and fetched again, a missing one downloaded (Gemma only in a Full / Quick setup);
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
    """Models to check: the ones the program needs on this PC, the translators of the standard download (tc-big ...), the
    complete set of a Full / Quick setup, plus every known model that exists locally."""
    from infra import model_downloader as md
    from infra import model_mirrors, setup_mode, text_models

    extras = [text_models.get(k).repo for k in text_models.COMPONENT_EXTRAS]
    known = list(dict.fromkeys(list(model_mirrors.load()) + text_models.repos()))
    need: List[str] = []
    try:
        from workers import pipeline_runner

        need = list(pipeline_runner.all_model_repos() if setup_mode.everything() else pipeline_runner.required_model_repos())
        if setup_mode.everything():
            extras += text_models.repos()
    except Exception as exc:  # noqa: BLE001 - GPU detection may fail; then only the local models are checked
        log.warning("required models unknown: %s", exc)
    present = [r for r in known if md.local_dir_for(r).exists() or md.partial_dir_has_data(r)]
    from infra import vc_model

    return [r for r in dict.fromkeys(need + extras + present) if r != vc_model.REPO]   # OpenVoice: check_tools


def _text_model(repo: str):
    """The text-model registry entry of ``repo`` (translators, SAGE), or None."""
    try:
        from infra import text_models

        return next((m for m in text_models.REGISTRY if m.repo == repo and m.integrated), None)
    except Exception:  # noqa: BLE001
        return None


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
    tm = _text_model(repo)
    if tm is not None and tm.revision:
        kw["revision"] = tm.revision                      # the translator's pinned commit (not "latest")
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
        kw["revision"] = revision
        ensure(repo, lambda s, f, m: progress(f, m), root=folder.parent, **kw)
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


# ------------------------------------------------------------------------------------------------ single-file tools
@dataclass
class Tool:
    """A model that is not an ``Owner--Name`` snapshot: OpenVoice V2, DeepFilterNet, DNSMOS, Gemma with llama.cpp.

    ``files()`` gives ``{path: {"size", "sha256"}}`` of the pinned files, ``extra_ok()`` checks what has no hash (the unpacked
    llama.cpp program), ``ensure(progress)`` downloads what is missing and ``wanted`` says whether a tool that is not
    installed at all is downloaded (the big optional Gemma only in a Full / Quick setup)."""
    name: str
    files: Callable[[], Dict[Path, Dict[str, Any]]]
    ensure: Callable[[Progress], Any]
    wanted: bool = True
    extra_ok: Callable[[], bool] = lambda: True


def default_tools() -> List[Tool]:
    """The single-file models of this platform."""
    from infra import denoise_tool, llm_tool, quality_models, setup_mode, vc_model

    def vc_files() -> Dict[Path, Dict[str, Any]]:
        e = vc_model.entry()
        return {vc_model.model_dir().joinpath(*n.split("/")): e.files[n] for n in vc_model.FILES}

    tools = [Tool(vc_model.LABEL, vc_files, lambda p: vc_model.ensure(p)),
             Tool(quality_models.DNSMOS_LABEL, lambda: {quality_models.dnsmos_path(): dict(quality_models.DNSMOS_META)},
                  lambda p: quality_models.ensure_dnsmos(p))]
    a = denoise_tool.asset()
    if a is not None:
        tools.append(Tool(denoise_tool.LABEL, lambda: {denoise_tool.tool_path(): {"size": a["size"], "sha256": a["sha256"]}},
                          lambda p: denoise_tool.ensure(p)))
    if llm_tool.platform_key() is not None:
        tools.append(Tool(llm_tool.LABEL,
                          lambda: {llm_tool.model_path(): {"size": llm_tool.MODEL["size"], "sha256": llm_tool.MODEL["sha256"]}},
                          lambda p: llm_tool.ensure(p), wanted=setup_mode.everything(),
                          extra_ok=lambda: llm_tool.server_exe() is not None))
    return tools


def _ours(path: Path) -> bool:
    """True for a file inside Voxprint's own models folder (a verified copy of another program is never changed)."""
    from infra import paths

    try:
        Path(path).resolve().relative_to(paths.models_dir().resolve())
        return True
    except (ValueError, OSError):
        return False


def check_tool(tool: Tool, progress: Progress, cancel: CancelToken) -> Item:
    """Check one single-file model by size + SHA-256 and download / repair it; never raises except on cancel."""
    from infra import model_release

    try:
        files = tool.files()
        missing = [p for p in files if not Path(p).is_file()]
        if len(missing) == len(files) and not tool.wanted:
            return Item("model", tool.name, SKIPPED, tr("autorepair.optional_missing"))
        bad: List[Path] = []
        for i, (path, meta) in enumerate(files.items()):
            cancel.check()
            if path in missing:
                continue
            progress(i / max(1, len(files)), tr("autorepair.checking", name=tool.name, file=Path(path).name))
            if not model_release.file_ok(Path(path), meta):
                bad.append(Path(path))
        extra = tool.extra_ok()
        if not missing and not bad and extra:
            return Item("model", tool.name, OK)
        foreign = [p for p in bad if not _ours(p)]
        if foreign:
            return Item("model", tool.name, FAILED, tr("autorepair.external", where=str(foreign[0].parent)))
        for p in bad:
            p.unlink(missing_ok=True)
        if bad:
            log.warning("auto-repair: %s has %d bad file(s): %s", tool.name, len(bad), ", ".join(p.name for p in bad))
            progress(0.0, tr("autorepair.repairing", name=tool.name, n=len(bad)))
        else:
            progress(0.0, tr("autorepair.downloading", name=tool.name))
        tool.ensure(lambda f, m="": progress(f, m))
        still = [p for p, meta in tool.files().items() if not model_release.file_ok(Path(p), meta)]
        if still or not tool.extra_ok():
            return Item("model", tool.name, FAILED, tr("autorepair.still_bad", n=max(1, len(still))))
        if bad:
            return Item("model", tool.name, REPAIRED, tr("autorepair.fixed_files", n=len(bad)))
        return Item("model", tool.name, DOWNLOADED)
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - one broken tool must not stop the others
        log.exception("auto-repair of %s failed", tool.name)
        return Item("model", tool.name, FAILED, getattr(exc, "user_message", "") or str(exc) or type(exc).__name__)


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
    missing = [m for m in mods if m.required and not m.installed and not m.update]     # repair, never an update
    items = [Item("module", m.title, OK) for m in mods if m.installed or m.update]
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
        modules_api: Any = None, ffmpeg_ensure: Optional[Callable[..., Any]] = None,
        tools: Optional[Iterable[Tool]] = None) -> Report:
    """Check and repair everything; ``progress(fraction 0..1, message)``.  Raises ``CancelledByUser`` when cancelled."""
    from infra import model_downloader as md

    cancel = cancel or CancelToken()
    ensure = ensure or md.ensure_model
    report = Report()
    repo_list = list(repos) if repos is not None else model_repos()
    if tools is None:
        try:
            tool_list = default_tools() if repos is None else []
        except Exception as exc:  # noqa: BLE001
            log.warning("single-file models unknown: %s", exc)
            tool_list = []
    else:
        tool_list = list(tools)
    steps = 3 + len(repo_list) + len(tool_list)
    done = [0]
    last = [0.0]

    def sub(f: float, m: str) -> None:
        # one item = one share of the bar; the share is passed twice for a repaired model (check, then download), so the
        # value is kept monotonic: the bar never goes back
        last[0] = max(last[0], min(1.0, (done[0] + max(0.0, min(1.0, f))) / steps))
        progress(last[0], tr("autorepair.step", n=min(done[0] + 1, steps), total=steps, text=m) if m else m)

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
    for tool in tool_list:
        report.items.append(check_tool(tool, sub, cancel))
        step()
    progress(1.0, report.summary())
    log.info("auto-repair: %s", "; ".join(f"{i.kind}:{i.name}={i.status}" for i in report.items))
    return report
