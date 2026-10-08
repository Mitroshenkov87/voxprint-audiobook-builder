"""The "Components" window of the thin installer: downloads the runtime modules (PyTorch, audio libraries ...) and the small
text-model "extras" (SAGE clean-up) in ONE pass and then, on the first start, hands over to the download of all heavy models.

What the user sees is one live line naming what is being downloaded right now ("Downloading: PyTorch (torch-...whl) - 42% -
35 MB/s", "Verifying: ... (SHA-256)", or an error) and one overall progress bar, instead of a list of installed / not installed
rows.  Every module is enabled as soon as it is unpacked (:func:`infra.modules.install` activates the runtime folder; the heavy
libraries are imported lazily, no restart).

Flow on the first start (``autostart=True``, see ``main._offer_components``):

1. step 1 "program components": the missing modules, then :data:`infra.text_models.COMPONENT_EXTRAS` and the DNSMOS file of
   the voice check (a failed extra is not fatal: the model download of step 2 lists it again);
2. step 2 "models": ``models_start(hook)`` starts the main window's first-run model download (TTS, alignment, speech recognition
   ...: :func:`workers.pipeline_runner.prefetch_models`, which honours the chosen models folder, existing copies and a backup)
   and the window mirrors its progress.  The download keeps running in the main window if this window is closed.

The window never touches the network itself: the work runs in :class:`ModulesWorker` (and the main window's prefetch worker);
the functions that talk to the network (``manifest_fn`` / ``install_fn`` / ``extras_fn``) are injectable for the tests."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from core.i18n import tr
from infra import modules as mods

log = logging.getLogger("voxprint.ui.modules")

Progress = Callable[[float, str], None]


def default_extras(progress: Progress) -> List[str]:
    """Step 1 extras: the text models of :data:`infra.text_models.COMPONENT_EXTRAS` (SAGE), then the voice-check MOS model
    (DNSMOS, 1.2 MB, :mod:`infra.quality_models`); progress ``(fraction, message)``."""
    from infra import quality_models, text_models

    text_bytes = sum(m.size_mb for m in text_models.missing_component_extras()) * 1024 ** 2
    mos_bytes = quality_models.missing_bytes()
    share = text_bytes / float(text_bytes + mos_bytes) if text_bytes + mos_bytes else 1.0
    done = text_models.ensure_component_extras(lambda _stage, f, m="": progress(share * float(f), m))
    if mos_bytes:
        quality_models.ensure_dnsmos(lambda f, m="": progress(share + (1.0 - share) * float(f), m))
        done.append("dnsmos")
    return done


def default_extras_bytes() -> int:
    """Download size of the extras that are still missing (0 = nothing to do)."""
    from infra import quality_models, text_models

    return sum(m.size_mb for m in text_models.missing_component_extras()) * 1024 ** 2 + quality_models.missing_bytes()


class ModulesWorker(QThread):
    """Loads the module list (``kind="list"``) or installs the missing required modules and then the extras (``kind="install"``)."""

    progress = Signal(float, str)
    listed = Signal(object)          # list[Module] or an error string
    done = Signal(int)
    failed = Signal(str)
    cancelled = Signal()
    extras_failed = Signal(str)      # an extra (SAGE) could not be fetched; not fatal, step 2 tries it again

    def __init__(self, kind: str, manifest_fn: Callable[[], Dict[str, Any]], install_fn: Callable[..., int],
                 ids: Optional[List[str]] = None, parent=None, extras_fn: Optional[Callable[[Progress], Any]] = None,
                 share: float = 1.0) -> None:
        """``share`` is the part of the overall bar that belongs to the modules (the rest goes to the extras)."""
        super().__init__(parent)
        self.kind, self.manifest_fn, self.install_fn, self.ids = kind, manifest_fn, install_fn, ids
        self.extras_fn, self.share = extras_fn, max(0.0, min(1.0, share))
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401
        try:
            if self.kind == "list":
                self.listed.emit(mods.modules(self.manifest_fn()))
                return
            n = 0
            if self.ids:
                n = self.install_fn(self.ids, lambda f, m="": self.progress.emit(self.share * float(f), m), lambda: self._cancel)
            if self.extras_fn is not None and not self._cancel:
                # a model download cannot be interrupted half-way: Cancel takes effect when the (small) extra is done
                base, rest = self.share, 1.0 - self.share
                try:
                    self.extras_fn(lambda f, m="": self.progress.emit(base + rest * float(f), m))
                except Exception as exc:  # noqa: BLE001 - network / disk errors of any kind
                    log.warning("components: extra text model failed: %s", exc)
                    self.extras_failed.emit(getattr(exc, "user_message", "") or str(exc))
            if self._cancel:
                raise mods.Cancelled("cancelled")
            self.done.emit(int(n))
        except mods.Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - network / disk errors of any kind
            log.warning("components: %s failed: %s", self.kind, exc)
            if self.kind == "list":
                self.listed.emit(str(exc))
            else:
                self.failed.emit(str(exc))


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{max(1, n // 1024 ** 2)} MB"


class ModulesDialog(QDialog):
    """One live status line + one overall progress bar; Download / Retry, Cancel, Close."""

    ready = Signal()                 # every required module is installed
    all_done = Signal()              # step 2 (models) finished as well

    def __init__(self, style: str = "", parent: Optional[QWidget] = None,
                 manifest_fn: Callable[[], Dict[str, Any]] = mods.load_manifest,
                 install_fn: Callable[..., int] = mods.install,
                 extras_fn: Optional[Callable[[Progress], Any]] = None,
                 extras_bytes_fn: Optional[Callable[[], int]] = None,
                 models_start: Optional[Callable[[Callable[[Any], None]], Any]] = None,
                 autostart: bool = False) -> None:
        """``extras_fn`` / ``extras_bytes_fn``: step 1 extras (:func:`default_extras`); ``models_start(hook)``: starts the step 2
        model download and returns its worker (signals ``progress(int, str)``, ``done(list)``, ``failed(str, str)``) or None;
        ``hook(worker)`` is called BEFORE the worker starts, so no signal is missed.  ``autostart``: first start, no clicks."""
        super().__init__(parent)
        self.setObjectName("root")
        self.setMinimumWidth(560)
        if style:
            self.setStyleSheet(style)
        self.manifest_fn, self.install_fn = manifest_fn, install_fn
        self.extras_fn = extras_fn
        self.extras_bytes_fn = extras_bytes_fn or ((lambda: 0) if extras_fn is None else default_extras_bytes)
        self.models_start, self.autostart = models_start, autostart
        self.modules: List[mods.Module] = []
        self.worker: Optional[ModulesWorker] = None
        self.models_worker: Optional[Any] = None
        self._note = ""                  # the last failure / cancel message; it survives the list refresh that follows
        self._note_error = False
        self._list_failed = False
        self._auto_tried = False         # autostart runs once; afterwards the user presses Download / Retry
        self._models_state = ""          # "" | "running" | "done" | "failed" | "later"
        self._step = 1
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        self.lbl_hint = QLabel()
        self.lbl_hint.setObjectName("cardnote")
        self.lbl_hint.setWordWrap(True)
        self.lbl_status = QLabel()       # THE live line: what is being downloaded / verified right now, or the error
        self.lbl_status.setObjectName("liveline")
        self.lbl_status.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        self.lbl_overall = QLabel()      # "Step 1 of 2: program components - 42% overall"
        self.lbl_overall.setObjectName("hint")
        self.lbl_overall.setVisible(False)
        self.btn_download = QPushButton()
        self.btn_download.setObjectName("primary")
        self.btn_cancel = QPushButton()
        self.btn_close = QPushButton()
        for w in (self.lbl_title, self.lbl_hint, self.lbl_status, self.bar, self.lbl_overall):
            lay.addWidget(w)
        brow = QHBoxLayout()
        brow.addStretch(1)
        for b in (self.btn_download, self.btn_cancel, self.btn_close):
            brow.addWidget(b)
        lay.addLayout(brow)
        self.btn_download.clicked.connect(self.on_download_clicked)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_close.clicked.connect(self.accept)
        self.retranslate()
        self._render()

    # ---------------------------------------------------------------- texts
    def retranslate(self) -> None:
        self.setWindowTitle(tr("modules.title"))
        self.lbl_title.setText(tr("modules.title"))
        hint = tr("modules.hint")
        if self.models_start is not None:
            hint += " " + tr("modules.hint_models")
        folder = mods.setup_folder()
        if folder is not None:                       # the installer's "portable setup folder": its files are used before any download
            hint += "\n" + tr("modules.setup_folder", folder=str(folder))
        self.lbl_hint.setText(hint)
        self.btn_cancel.setText(tr("ui.cancel"))
        self.btn_close.setText(tr("modules.btn_close"))
        self._render()

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @property
    def models_running(self) -> bool:
        return self._models_state == "running"

    def missing(self) -> List[mods.Module]:
        return [m for m in self.modules if m.required and not m.installed]

    def extras_missing(self) -> bool:
        if self.extras_fn is None:
            return False
        try:
            return self.extras_bytes_fn() > 0
        except Exception:  # noqa: BLE001 - an unreadable models folder must not break the window
            return False

    def _set_line(self, text: str, error: bool = False) -> None:
        """The live line; an error is shown on the same line in the error colour."""
        self.lbl_status.setText(text)
        self.lbl_status.setProperty("state", "error" if error else "")
        st = self.lbl_status.style()
        st.unpolish(self.lbl_status)
        st.polish(self.lbl_status)

    def _set_overall(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        self.bar.setValue(int(fraction * 1000))
        pct = int(fraction * 100)
        if self.models_start is None:
            self.lbl_overall.setText(tr("modules.overall", pct=pct))
        else:
            what = tr("modules.step_components") if self._step == 1 else tr("modules.step_models")
            self.lbl_overall.setText(tr("modules.overall_step", step=self._step, steps=2, what=what, pct=pct))

    def _render(self) -> None:
        working = self.busy or self.models_running
        self.bar.setVisible(working)
        self.lbl_overall.setVisible(working)
        retry = self._list_failed or self._models_state == "failed"
        self.btn_download.setText(tr("modules.btn_retry") if retry else tr("modules.btn_download"))
        can = bool(self.missing()) or self.extras_missing() or self._list_failed or self._models_state == "failed"
        self.btn_download.setEnabled(not working and can)
        self.btn_cancel.setVisible(self.busy)                # the model download of step 2 has no cancel: it can be closed
        self.btn_close.setEnabled(not self.busy)

    # ---------------------------------------------------------------- list
    def refresh(self) -> None:
        """Read the manifest in the background and show the modules."""
        if self.busy:
            return
        if not self._note:
            self._set_line(tr("modules.checking"))
        w = ModulesWorker("list", self.manifest_fn, self.install_fn, parent=self)
        w.listed.connect(self._on_listed)
        self.worker = w
        w.start()

    def _summary(self) -> str:
        """Idle line: what a press on Download would fetch."""
        miss = self.missing()
        names = [m.title for m in miss]
        size = sum(m.size for m in miss)
        if self.extras_missing():
            names.append(tr("modules.extra_sage"))
            try:
                size += self.extras_bytes_fn()
            except Exception:  # noqa: BLE001
                pass
        return tr("modules.to_download", names=", ".join(names), size=_gb(size)) if names else tr("modules.all_ready")

    def _on_listed(self, res) -> None:
        if self.worker is not None:
            self.worker.wait(2000)
        self.worker = None
        if isinstance(res, str):
            self.modules = []
            self._list_failed = True
            self._set_line(tr("modules.offline", error=res), error=True)
            self._render()
            return
        self._list_failed = False
        self.modules = list(res)
        pending = bool(self.missing()) or self.extras_missing()
        if pending and self.autostart and not self._auto_tried:
            self._auto_tried = True
            self.start_install()                    # first start: no click needed
            return
        if self._note:
            self._set_line(self._note, error=self._note_error)
        elif not self.missing():
            self._set_line(tr("modules.all_ready"))
        else:
            self._set_line(self._summary())
        self._render()
        if self.modules and not self.missing():
            self.ready.emit()
            self._start_models()

    # ---------------------------------------------------------------- step 1: install
    def on_download_clicked(self) -> None:
        if self.missing() or self.extras_missing():
            self.start_install()
        elif self._models_state == "failed":
            self._models_state = ""
            self._start_models()
        elif self._list_failed:
            self._note = ""
            self.refresh()

    def start_install(self) -> bool:
        if self.busy or not (self.missing() or self.extras_missing()):
            return False
        self._note = ""
        self._step = 1
        ids = [m.id for m in self.missing()]
        mod_bytes = sum(m.size for m in self.missing())
        extra_bytes = 0
        if self.extras_fn is not None:
            try:
                extra_bytes = max(0, int(self.extras_bytes_fn()))
            except Exception:  # noqa: BLE001
                extra_bytes = 0
        share = mod_bytes / float(mod_bytes + extra_bytes) if mod_bytes + extra_bytes > 0 else (1.0 if ids else 0.0)
        w = ModulesWorker("install", self.manifest_fn, self.install_fn, ids, parent=self, extras_fn=self.extras_fn, share=share)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        w.extras_failed.connect(self._on_extras_failed)
        self.worker = w
        self._set_overall(0.0)
        self._set_line(tr("modules.downloading"))
        w.start()
        self._render()
        return True

    def cancel(self) -> None:
        if self.busy and self.worker is not None:
            self.worker.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._set_overall(fraction)
        if message:
            self._set_line(message)

    def _finish(self) -> None:
        if self.worker is not None:
            self.worker.wait(2000)
        self.refresh_after()

    def refresh_after(self) -> None:
        self.worker = None
        self.refresh()

    def _on_done(self, _n: int) -> None:
        self._finish()
        self._set_line(tr("modules.all_ready"))

    def _on_failed(self, message: str) -> None:
        self._note, self._note_error = tr("modules.failed", error=message), True
        self._finish()
        self._set_line(self._note, error=True)

    def _on_cancelled(self) -> None:
        self._note, self._note_error = tr("modules.cancelled"), False
        self._finish()
        self._set_line(self._note)

    def _on_extras_failed(self, message: str) -> None:
        log.info("components: the extra will be retried with the models: %s", message)

    # ---------------------------------------------------------------- step 2: models
    def _start_models(self) -> None:
        """Right after the components: the download of ALL heavy models, without asking (first start only)."""
        if self.models_start is None or self._models_state in ("running", "done"):
            return
        self._step = 2
        self._models_state = "running"
        self._set_overall(0.0)
        self._set_line(tr("modules.models_starting"))
        try:
            w = self.models_start(self._wire_models)
        except Exception as exc:  # noqa: BLE001
            log.warning("components: could not start the model download: %s", exc)
            w = None
        if w is None:
            self._models_state = "later"
            self._set_line(tr("modules.models_later"))
        self.models_worker = w
        self._render()

    def _wire_models(self, w: Any) -> None:
        w.progress.connect(self._on_models_progress)
        w.done.connect(self._on_models_done)
        w.failed.connect(self._on_models_failed)

    def _on_models_progress(self, pct: int, message: str) -> None:
        if self._models_state != "running":
            return
        self._set_overall(pct / 100.0)
        if message:
            self._set_line(message)

    def _on_models_done(self, _downloaded: list) -> None:
        self._models_state = "done"
        self._set_overall(1.0)
        self._set_line(tr("modules.everything_ready"))
        self._render()
        self.all_done.emit()

    def _on_models_failed(self, message: str, _url: str = "") -> None:
        self._models_state = "failed"
        self._set_line(tr("modules.models_failed", error=message), error=True)
        self._render()

    def shutdown(self) -> None:
        if self.busy and self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
