"""Updating components: check PyPI + HF Hub, install into ``.staging/``, check compatibility, swap or roll back.

The log goes to ``logs/updater.log``.

The default channel is ``verified`` (the "verified by Voxprint" manifest, ``infra/verified_manifest.py``): exactly the
tested package versions and model revisions are installed; newer PyPI releases are only recorded as "not verified
yet".  The ``latest`` channel (``VOXPRINT_CHANNEL=latest`` or ``state/updater_state.json``: ``{"channel": "latest"}``)
takes the newest stable release.  :meth:`Updater.restore_verified` returns everything to the verified set (called
automatically after a failed update / on request from the CLI); :meth:`Updater.rollback` restores the previous
``packages`` directory.

Scheme (works from source and from the packaged exe):

* Python packages are installed with ``pip install --no-deps --target .staging/<ts>/site`` (pip comes from a system
  Python; in an exe without any Python the package update is skipped, model updates still work).
* The current ``packages/`` directory (earlier updates) is copied into the staging area first and the new versions are
  installed on top.  Then a separate Python process with that directory on ``sys.path`` runs a smoke test (imports +
  a tokenizer check of the aligner).  Only on success is ``packages/`` swapped (the old one is kept as
  ``packages.bak-<ts>``); otherwise the staging area is deleted.
* New versions are picked up on the next start: ``main.py`` calls :func:`activate_overlay` before importing the heavy
  libraries.
* Models are downloaded into ``.staging/models/<name>``, verified and only then replace the local copy; on failure the
  old one stays.
* Components in the USER's environment are never touched silently: they are offered (see :meth:`Updater.apply`).
"""
from __future__ import annotations

from core.i18n import tr
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from core.errors import UpdateError
from core.events import ProgressCallback, Stage, noop_progress
from infra import model_downloader as md
from infra import paths
from infra.verified_manifest import Manifest, load_manifest
from infra.version_manager import (CHANNEL_LATEST, CHANNEL_VERIFIED, TRACKED_MODELS, TRACKED_PACKAGES,
                                   Offer, VersionReport, check_versions, http_fetch_json, installed_version,
                                   make_offers)

log = logging.getLogger("voxprint.updater")
#: The UI re-checks for updates automatically at most once per this many days.
AUTOCHECK_DAYS = 7

# Compatibility smoke test, run in a SEPARATE interpreter with the staged site directory first on sys.path.
# The Russian sentence is test data: it exercises the aligner's tokenizer on non-English text.
SMOKE_CODE = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
import transformers
assert int(transformers.__version__.split('.')[0]) < 5, 'transformers>=5 is not supported by qwen-asr/qwen-tts'
import peft, accelerate
from peft import LoraConfig
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
words, text = Qwen3ForceAlignProcessor().encode_timestamp('Привет, мир! Тест-кейс', 'Russian')
assert words == ['Привет', 'мир', 'Тесткейс'], words
assert text.startswith('<|audio_start|><|audio_pad|><|audio_end|>')
import qwen_asr
import qwen_tts
from importlib import metadata
out = {}
for n in sys.argv[2:]:
    try: out[n] = metadata.version(n)
    except Exception: out[n] = None
print('SMOKE_OK ' + json.dumps(out))
"""

#: ``run(argv) -> (return_code, output)`` - injectable so tests never start real processes.
Runner = Callable[[List[str]], Tuple[int, str]]


def setup_log() -> None:
    """Attach the ``logs/updater.log`` file handler (re-created if the data directory changed)."""
    target = str(paths.logs_dir() / "updater.log")
    for h in list(log.handlers):
        if getattr(h, "_vox", False):
            if getattr(h, "baseFilename", "") == os.path.abspath(target):
                return
            log.removeHandler(h)
            h.close()
    h = logging.handlers.RotatingFileHandler(target, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    h._vox = True  # type: ignore[attr-defined]
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


def activate_overlay() -> Optional[Path]:
    """Put the directory of updated packages first on ``sys.path``.  Call at the very start of ``main.py``."""
    p = paths.packages_dir()
    if p.is_dir() and any(p.iterdir()):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        return p
    return None


def _creationflags() -> int:
    """``CREATE_NO_WINDOW`` on Windows so child processes do not flash a console window."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def run_subprocess(args: List[str], timeout: int = 1800) -> Tuple[int, str]:
    """Run a command, capture stdout+stderr; returns ``(return_code, output)`` and never raises (timeout/OSError -> rc 1)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           creationflags=_creationflags(), encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def find_python() -> Optional[str]:
    """Python used for pip / the smoke test: the current interpreter, or a system one when running as a packaged exe."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    try:   # the packaged exe needs a Python of the same version (binary wheels); pip --target writes only into our folder
        from infra import env_probe

        envs = [e for e in (env_probe.probe_python(x, src) for x, src in env_probe.candidate_pythons()) if e]
        exe = env_probe.pick_pip_python(envs)
        if exe:
            return exe
    except Exception:  # noqa: BLE001 - the choice is optional, the old fallback below still applies
        pass
    for name in ("python", "python3", "py"):
        w = shutil.which(name)
        if w:
            return w
    return None


@dataclass
class UpdateResult:
    """What an update run did: versions before/after, updated models, rollback flag, messages and restart need."""
    before: Dict[str, Optional[str]] = field(default_factory=dict)
    after: Dict[str, Optional[str]] = field(default_factory=dict)
    models_updated: List[str] = field(default_factory=list)
    rolled_back: bool = False
    unverified_newer: List[str] = field(default_factory=list)   # newer on PyPI than the verified set (informational)
    needs_restart: bool = False
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Localized multi-line summary of the run for the UI/log."""
        lines: List[str] = []
        for k in sorted(set(self.before) | set(self.after)):
            b, a = self.before.get(k), self.after.get(k)
            if b != a:
                lines.append(tr("upd.change", name=k, before=b or tr("upd.none"), after=a or tr("upd.none")))
        for m in self.models_updated:
            lines.append(tr("upd.model_updated", name=m))
        if self.rolled_back:
            lines.append(tr("upd.rolled_back"))
        if not lines:
            lines.append(tr("upd.up_to_date"))
        lines.extend(self.messages)
        if self.needs_restart:
            lines.append(tr("upd.restart"))
        return "\n".join(lines)


class Updater:
    """Checks for and applies updates of packages and models (see the module docstring for the scheme)."""
    def __init__(self, fetch_json=http_fetch_json, pip_runner: Optional[Runner] = None,
                 smoke_runner: Optional[Runner] = None, python_exe: Optional[str] = None,
                 now: Callable[[], float] = time.time,
                 installed_fn: Callable[[str], Optional[str]] = installed_version,
                 snapshot_download: Optional[Callable[..., str]] = None,
                 get_remote_sha: Optional[Callable[[str], Optional[str]]] = None,
                 packages: Optional[Dict[str, str]] = None, models: Optional[tuple] = None,
                 manifest: Optional[Manifest] = None, channel: Optional[str] = None,
                 probe_env: Optional[Callable[[], Any]] = None, external_env: Optional[bool] = None) -> None:
        """All collaborators are injectable (network, pip, smoke test, clock ...) so the updater is testable offline."""
        setup_log()
        if external_env is None:
            from infra.env_probe import is_own_environment

            external_env = not is_own_environment()
        self.external_env = external_env     # True: Voxprint runs inside the user's own environment (outdated components only on request)
        self.probe_env = probe_env
        self._manifest = manifest
        self._channel = channel
        self.fetch_json = fetch_json
        self.run = pip_runner or run_subprocess
        self.smoke = smoke_runner or run_subprocess
        self.python = python_exe if python_exe is not None else find_python()
        self.now = now
        self.installed_fn = installed_fn
        self.snapshot_download = snapshot_download
        self.get_remote_sha = get_remote_sha
        self.packages = packages if packages is not None else TRACKED_PACKAGES
        self.models = models if models is not None else TRACKED_MODELS

    # ---- persistent state (state/updater_state.json)
    @property
    def state_file(self) -> Path:
        """``state/updater_state.json``: last check time, channel, history, declined offers."""
        return paths.state_dir() / "updater_state.json"

    def _load_state(self) -> dict:
        """Read the state file; ``{}`` if missing or broken."""
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self, st: dict) -> None:
        """Write the state file."""
        self.state_file.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    def should_autocheck(self, interval_days: float = AUTOCHECK_DAYS) -> bool:
        """True if the last successful check is older than ``interval_days``."""
        last = self._load_state().get("last_check", 0)
        return (self.now() - float(last)) >= interval_days * 86400

    # ---- channel and manifest
    @property
    def channel(self) -> str:
        """Active channel: constructor argument, ``VOXPRINT_CHANNEL``, or the state file; anything but ``latest`` means ``verified``."""
        c = self._channel or os.environ.get("VOXPRINT_CHANNEL") or self._load_state().get("channel")
        return CHANNEL_LATEST if c == CHANNEL_LATEST else CHANNEL_VERIFIED

    def manifest(self) -> Manifest:
        """The verified manifest (loaded lazily, with the remote/cached override)."""
        if self._manifest is None:
            self._manifest = load_manifest(self.fetch_json)
        return self._manifest

    # ---- checking
    def check(self, channel: Optional[str] = None) -> VersionReport:
        """Compare installed packages/models with the target versions of the channel; remembers the time of a successful check."""
        rep = check_versions(md.local_revisions(self.models), self.fetch_json, self.installed_fn,
                             self.packages, self.models, manifest=self.manifest(), channel=channel or self.channel,
                             external_env=self.external_env)
        st = self._load_state()
        if rep.network_ok:
            st["last_check"] = self.now()
        self._save_state(st)
        log.info("check[%s, manifest %s]: network_ok=%s to_change=%s outdated_models=%s unverified_newer=%s",
                 rep.channel, rep.manifest_date, rep.network_ok,
                 [(p.name, p.installed, p.target) for p in rep.outdated_packages],
                 [m.repo_id for m in rep.outdated_models], rep.unverified_newer)
        return rep

    # ---- applying
    def apply(self, report: VersionReport, progress: ProgressCallback = noop_progress,
              accepted: Optional[Iterable[str]] = None) -> UpdateResult:
        """Apply a :class:`VersionReport`.

        ``accepted`` holds the names of outdated components IN THE USER'S ENVIRONMENT that the user agreed to update.
        Agreed -> pip updates them there.  Declined and still compatible -> used as they are.  Declined and incompatible ->
        Voxprint's own copy goes into its ``packages`` directory (the user's environment is not changed).
        """
        res = UpdateResult()
        todo = list(report.outdated_packages)           # Voxprint's own environment: updated without asking
        yes = set(accepted or ())
        for p in report.offered_packages:
            if p.name in yes:
                continue
            if p.decision.compatible:
                res.messages.append(tr("upg.kept", name=p.name, old=p.installed or "-"))
                log.info("offer for %s declined; the installed %s is still compatible - reusing it", p.name, p.installed)
            else:
                todo.append(p)
                res.messages.append(tr("upg.own_copy", name=p.name, old=p.installed or "-", new=p.target or "-"))
                log.info("offer for %s declined; %s is not compatible - using Voxprint's own copy %s",
                         p.name, p.installed, p.target)
        for p in todo:
            res.before[p.name] = p.installed
        for p in report.offered_packages:
            if p.name in yes and p.target:
                progress(Stage.UPDATES, 0.05, tr("upg.progress", name=p.name, old=p.installed or "-", new=p.target,
                                                 env=sys.prefix))
                self._upgrade_in_place(p, res)
        if todo:
            progress(Stage.UPDATES, 0.1, tr("upd.installing"))
            self._apply_packages(todo, res)
        for i, m in enumerate(report.outdated_models):
            progress(Stage.UPDATES, 0.5 + 0.4 * i / max(1, len(report.outdated_models)),
                     tr("upd.updating_model", name=m.repo_id.split("/")[-1]))
            if self._apply_model(m.repo_id, progress, m.target_sha):
                res.models_updated.append(m.repo_id)
        st = self._load_state()
        st.setdefault("history", []).append({
            "time": self.now(), "before": res.before, "after": res.after,
            "models": res.models_updated, "rolled_back": res.rolled_back})
        st["history"] = st["history"][-20:]
        self._save_state(st)
        progress(Stage.UPDATES, 1.0, tr("upd.processed"))
        log.info("apply result: %s", res.summary().replace("\n", " | "))
        return res

    def _upgrade_in_place(self, p, res: UpdateResult) -> None:
        """Only after the user's consent: pip upgrades the package in the environment Voxprint runs from.

        The result is verified by importing; on failure the previous version is restored.
        """
        if not self.python:
            res.messages.append(tr("upd.no_python"))
            return
        name, old, new = p.name, p.installed, p.target
        base = [self.python, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check"]
        log.info("user accepted: pip upgrade %s %s -> %s in %s", name, old, new, sys.prefix)
        res.before[name] = old
        rc, out = self.run([*base, "--upgrade", f"{name}=={new}"])
        log.info("pip rc=%s\n%s", rc, out[-2000:])
        ok = rc == 0
        if ok:
            ok, _ = self._smoke(paths.staging_dir(), [name])
        if not ok:
            res.rolled_back = True
            if old:
                rc2, out2 = self.run([*base, f"{name}=={old}"])   # restore what the user had
                log.info("rollback pip rc=%s\n%s", rc2, out2[-1000:])
            res.messages.append(tr("upg.rolled_back", name=name, old=old or "-"))
            return
        res.after[name] = new
        res.needs_restart = True

    def _apply_packages(self, todo, res: UpdateResult) -> None:
        """Install ``todo`` into a staging copy of ``packages/``, smoke-test it and swap it in (or discard it)."""
        if not self.python:
            res.messages.append(tr("upd.no_python"))
            log.warning("no python for pip; skipping package update")
            return
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.now()))
        stage_root = paths.staging_dir() / f"pkgs-{ts}"
        site = stage_root / "site"
        try:
            current = paths.packages_dir()
            if current.is_dir():
                shutil.copytree(current, site)
            else:
                site.mkdir(parents=True)
            specs = [f"{p.name}=={p.target}" for p in todo]
            cmd = [self.python, "-m", "pip", "install", "--upgrade", "--no-deps",
                   "--disable-pip-version-check", "--target", str(site), *specs]
            log.info("pip: %s", " ".join(cmd))
            rc, out = self.run(cmd)
            log.info("pip rc=%s\n%s", rc, out[-4000:])
            if rc != 0:
                res.rolled_back = True
                res.messages.append(tr("upd.download_failed"))
                return
            # compatibility check in a separate process with the new directory on sys.path
            ok, versions = self._smoke(site, [p.name for p in todo])
            if not ok:
                res.rolled_back = True
                log.error("smoke test failed for staging")
                return
            # swap the directory (the old one is kept as packages.bak-<ts>)
            backup = None
            if current.exists():
                backup = current.with_name(f"packages.bak-{ts}")
                os.replace(current, backup)
            try:
                os.replace(site, current)
            except OSError:
                if backup is not None:
                    os.replace(backup, current)
                raise
            self._prune_backups()
            res.after.update(versions or {p.name: p.target for p in todo})
            res.needs_restart = True
            log.info("swap done: %s", res.after)
        except OSError as exc:
            log.exception("package update failed")
            res.rolled_back = True
            res.messages.append(tr("upd.install_failed", error=exc))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    def _smoke(self, site: Path, names: List[str]) -> Tuple[bool, Dict[str, Optional[str]]]:
        """Run the smoke test in a separate interpreter; returns ``(ok, {package: version})``."""
        assert self.python
        rc, out = self.smoke([self.python, "-c", SMOKE_CODE, str(site), *names])
        log.info("smoke rc=%s %s", rc, out[-2000:])
        if rc == 0 and "SMOKE_OK" in out:
            try:
                data = json.loads(out.split("SMOKE_OK", 1)[1].strip().splitlines()[0])
            except (ValueError, IndexError):
                data = {}
            return True, data
        return False, {}

    @staticmethod
    def _prune_backups(keep: int = 2) -> None:
        """Keep only the newest ``keep`` ``packages.bak-*`` directories."""
        baks = sorted(paths.app_home().glob("packages.bak-*"))
        for b in baks[:-keep]:
            shutil.rmtree(b, ignore_errors=True)

    def rollback(self) -> bool:
        """Manual rollback to the previous ``packages`` directory; False if there is no backup."""
        baks = sorted(paths.app_home().glob("packages.bak-*"))
        if not baks:
            return False
        cur = paths.packages_dir()
        if cur.exists():
            shutil.rmtree(cur)
        os.replace(baks[-1], cur)
        log.info("rolled back to %s", baks[-1].name)
        return True

    def _apply_model(self, repo_id: str, progress: ProgressCallback, revision: Optional[str] = None) -> bool:
        """Download a model into staging, verify it and swap it in; on any failure the old copy is kept.  True on success."""
        stage_models = paths.staging_dir() / "models"
        stage_models.mkdir(parents=True, exist_ok=True)
        new_dir = md.local_dir_for(repo_id, stage_models)
        shutil.rmtree(new_dir, ignore_errors=True)
        try:
            md.ensure_model(repo_id, progress, root=stage_models, snapshot_download=self.snapshot_download,
                            get_remote_sha=self.get_remote_sha, stage=Stage.UPDATES, revision=revision)
            if not md.verify_local_model(new_dir):
                raise UpdateError(tr("upd.model_check_failed"))
        except Exception as exc:  # noqa: BLE001 - any error = we stay on the old model
            log.error("model update failed for %s: %s", repo_id, exc)
            shutil.rmtree(new_dir, ignore_errors=True)
            return False
        target = md.local_dir_for(repo_id)
        bak = target.with_name(target.name + ".bak")
        shutil.rmtree(bak, ignore_errors=True)
        try:
            if target.exists():
                os.replace(target, bak)
            os.replace(new_dir, target)
        except OSError as exc:
            log.error("model swap failed: %s", exc)
            if bak.exists() and not target.exists():
                os.replace(bak, target)
            return False
        shutil.rmtree(bak, ignore_errors=True)
        log.info("model %s updated", repo_id)
        return True

    def restore_verified(self, progress: ProgressCallback = noop_progress) -> UpdateResult:
        """Return packages and models to the "verified by Voxprint" set (downgrading is allowed)."""
        rep = self.check(channel=CHANNEL_VERIFIED)
        if not rep.network_ok:
            return UpdateResult(messages=[tr("upd.no_net_restore")])
        res = self.apply(rep, progress)
        res.unverified_newer = rep.unverified_newer
        return res

    def _report_environment(self, progress: ProgressCallback) -> None:
        """Read-only: report what is already installed and what will happen to it (reuse / upgrade / install)."""
        if self.probe_env is None and os.environ.get("VOXPRINT_NO_ENV_PROBE"):
            return
        try:
            from infra import env_probe

            env = (self.probe_env or (lambda: env_probe.probe_environment(
                manifest_pins=dict(self.manifest().packages), installed_fn=self.installed_fn,
                scan_other_pythons=False)))()
            self.last_env = env
            for line in env_probe.user_messages(env):
                progress(Stage.UPDATES, 0.0, line)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not get in the way of the update
            log.warning("environment probe failed: %s", exc)

    # ---- everything at once, for the UI
    def check_and_apply(self, progress: ProgressCallback = noop_progress,
                        ask: Optional[Callable[[List[Offer]], Iterable[str]]] = None) -> Tuple[VersionReport, UpdateResult]:
        """Check and apply in one go (what the UI's "Check for updates" does).

        ``ask(offers)`` returns the names the user allowed to update (a dialog in the UI); without ``ask`` refusal is assumed.
        An offer the user already declined for the same target version is not shown again.
        """
        progress(Stage.UPDATES, 0.0, tr("upd.checking"))
        self._report_environment(progress)
        rep = self.check()
        if not rep.network_ok:
            r = UpdateResult(messages=[tr("upd.no_net_check")])
            progress(Stage.UPDATES, 1.0, r.messages[0])
            return rep, r
        if not rep.has_updates:
            progress(Stage.UPDATES, 1.0, tr("upd.verified_installed"))
            return rep, UpdateResult(unverified_newer=rep.unverified_newer)
        accepted: List[str] = []
        offers = make_offers(rep, sys.prefix)
        if offers:
            st = self._load_state()
            declined = dict(st.get("declined_offers") or {})
            fresh = [o for o in offers if declined.get(o.name) != o.target]
            if fresh and ask is not None:
                accepted = [n for n in ask(fresh) if n in {o.name for o in fresh}]
                for o in fresh:
                    if o.name in accepted:
                        declined.pop(o.name, None)
                    else:
                        declined[o.name] = o.target
                st["declined_offers"] = declined
                self._save_state(st)
        res = self.apply(rep, progress, accepted)
        res.unverified_newer = rep.unverified_newer
        return rep, res
