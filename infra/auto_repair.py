"""Check & repair (Settings -> "Check & repair", formerly "Auto-repair"): check every component, repair what is broken.

Built on the existing pieces instead of duplicating them:

* **environment** - :func:`infra.install_state.verify_install` (the ``--verify-install`` check); a source/venv install is
  rebuilt with :func:`infra.install_state.repair_install` (the ``--repair`` path), an installer build only reports;
* **runtime modules** (thin build) - :func:`infra.modules.modules` / :func:`infra.modules.install` (resumable, SHA-256);
* **ffmpeg** (Windows) - :func:`infra.assets.ensure_ffmpeg_tool` (smoke test, pinned build re-installed when broken);
* **models** - the shared store is checked and repaired by VOXPrint MDL itself
  (``voxprint-mdl verify --repair`` through :mod:`infra.mdl_models`); this app never deletes or re-downloads a model.

Everything is injectable (tests run without network, GPU or real downloads).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

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
        """True when no checked item failed."""
        return all(i.status != FAILED for i in self.items)

    def count(self, *statuses: str) -> int:
        """How many items have one of ``statuses``."""
        return sum(1 for i in self.items if i.status in statuses)

    def summary(self) -> str:
        """The translated summary, plus the names of the items that failed."""
        text = tr("autorepair.summary", checked=len(self.items), fixed=self.count(REPAIRED, DOWNLOADED),
                  failed=self.count(FAILED))
        bad = [f"{i.name} ({i.detail})" if i.detail else i.name for i in self.items if i.status == FAILED]
        return text + ("\n" + tr("autorepair.failed_list", names="; ".join(bad)) if bad else "")


# ------------------------------------------------------------------------------------------------ models
def check_models(progress: Progress, cancel: CancelToken,
                 verify: Optional[Callable[[Callable[[float, str], None]], bool]] = None) -> Item:
    """The shared model store checked and repaired by VOXPrint MDL (``voxprint-mdl verify --repair``).

    Never raises except on cancel; a failure names Manage models."""
    from infra import mdl_gate, mdl_models

    name = tr("autorepair.models")
    if verify is None and mdl_gate.skipped():     # the unit tests: MDL never runs here (a fake verify is honoured)
        return Item("model", name, SKIPPED)
    try:
        ok_ = (verify or mdl_models.verify_repair)(progress)
        if ok_:
            return Item("model", name, OK)
        return Item("model", name, FAILED, tr("autorepair.models_hint"))
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as one failed item
        log.exception("model store check failed")
        return Item("model", name, FAILED, getattr(exc, "user_message", "") or type(exc).__name__)


# ------------------------------------------------------------------------------------------------ components
def _default_env_repair(progress: Progress) -> Tuple[int, str]:
    from infra import install_state
    from infra.updater import run_subprocess

    return install_state.repair_install(run_subprocess, shutil_which(), progress)


def shutil_which() -> Callable[[str], Optional[str]]:
    """Deferred ``shutil.which`` (kept out of the module's hot import path)."""
    import shutil

    return shutil.which


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
        env_verify: Optional[Callable[[], Any]] = None, env_repair: Optional[Callable[[Progress], Tuple[int, str]]] = None,
        modules_api: Any = None, ffmpeg_ensure: Optional[Callable[..., Any]] = None,
        models_verify: Optional[Callable[[Callable[[float, str], None]], bool]] = None) -> Report:
    """Check and repair everything; ``progress(fraction 0..1, message)``.  Raises ``CancelledByUser`` when cancelled."""
    cancel = cancel or CancelToken()
    report = Report()
    steps = 4
    done = [0]
    last = [0.0]

    def sub(f: float, m: str) -> None:
        # one item = one share of the bar; the value is kept monotonic: the bar never goes back
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
    report.items.append(check_models(sub, cancel, models_verify))
    step()
    progress(1.0, report.summary())
    log.info("auto-repair: %s", "; ".join(f"{i.kind}:{i.name}={i.status}" for i in report.items))
    return report
