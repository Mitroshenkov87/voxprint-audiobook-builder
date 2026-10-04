"""The "Components" window of the thin installer: lists the runtime modules (PyTorch, audio libraries ...), downloads the
missing ones with a progress bar, can be cancelled and resumed.  Every module is enabled as soon as it is unpacked
(:func:`infra.modules.install` activates the runtime folder; the heavy libraries are imported lazily, no restart).

The window never touches the network itself: the work runs in :class:`ModulesWorker`; the functions that talk to the network
(``manifest_fn`` / ``install_fn``) are injectable for the tests."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from core.i18n import tr
from infra import modules as mods

log = logging.getLogger("voxprint.ui.modules")


class ModulesWorker(QThread):
    """Loads the module list (``kind="list"``) or installs the missing required modules (``kind="install"``)."""

    progress = Signal(float, str)
    listed = Signal(object)          # list[Module] or an error string
    done = Signal(int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, kind: str, manifest_fn: Callable[[], Dict[str, Any]], install_fn: Callable[..., int],
                 ids: Optional[List[str]] = None, parent=None) -> None:
        super().__init__(parent)
        self.kind, self.manifest_fn, self.install_fn, self.ids = kind, manifest_fn, install_fn, ids
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401
        try:
            if self.kind == "list":
                self.listed.emit(mods.modules(self.manifest_fn()))
                return
            n = self.install_fn(self.ids, lambda f, m="": self.progress.emit(float(f), m), lambda: self._cancel)
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
    """Rows of modules + one Download button for all that are missing."""

    ready = Signal()                 # every required module is installed

    def __init__(self, style: str = "", parent: Optional[QWidget] = None,
                 manifest_fn: Callable[[], Dict[str, Any]] = mods.load_manifest,
                 install_fn: Callable[..., int] = mods.install) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self.setMinimumWidth(520)
        if style:
            self.setStyleSheet(style)
        self.manifest_fn, self.install_fn = manifest_fn, install_fn
        self.modules: List[mods.Module] = []
        self.worker: Optional[ModulesWorker] = None
        self._note = ""                  # the last failure / cancel message; it survives the list refresh that follows
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("title")
        self.lbl_hint = QLabel()
        self.lbl_hint.setObjectName("cardnote")
        self.lbl_hint.setWordWrap(True)
        self.rows = QVBoxLayout()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.btn_download = QPushButton()
        self.btn_download.setObjectName("primary")
        self.btn_cancel = QPushButton()
        self.btn_close = QPushButton()
        for w in (self.lbl_title, self.lbl_hint):
            lay.addWidget(w)
        lay.addLayout(self.rows)
        for w in (self.bar, self.lbl_status):
            lay.addWidget(w)
        brow = QHBoxLayout()
        brow.addStretch(1)
        for b in (self.btn_download, self.btn_cancel, self.btn_close):
            brow.addWidget(b)
        lay.addLayout(brow)
        self.btn_download.clicked.connect(self.start_install)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_close.clicked.connect(self.accept)
        self.retranslate()
        self._render()

    # ---------------------------------------------------------------- texts
    def retranslate(self) -> None:
        self.setWindowTitle(tr("modules.title"))
        self.lbl_title.setText(tr("modules.title"))
        self.lbl_hint.setText(tr("modules.hint"))
        self.btn_download.setText(tr("modules.btn_download"))
        self.btn_cancel.setText(tr("ui.cancel"))
        self.btn_close.setText(tr("modules.btn_close"))

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def missing(self) -> List[mods.Module]:
        return [m for m in self.modules if m.required and not m.installed]

    def _render(self) -> None:
        while self.rows.count():
            it = self.rows.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for m in self.modules:
            state = tr("modules.installed") if m.installed else tr("modules.missing")
            lab = QLabel(f"{m.title}  -  {_gb(m.size)}  -  {state}")
            lab.setObjectName("modulerow")
            self.rows.addWidget(lab)
        self.btn_download.setEnabled(not self.busy and bool(self.missing()))
        self.btn_cancel.setVisible(self.busy)
        self.btn_close.setEnabled(not self.busy)

    # ---------------------------------------------------------------- list
    def refresh(self) -> None:
        """Read the manifest in the background and show the modules."""
        if self.busy:
            return
        if not self._note:
            self.lbl_status.setText(tr("modules.checking"))
        w = ModulesWorker("list", self.manifest_fn, self.install_fn, parent=self)
        w.listed.connect(self._on_listed)
        self.worker = w
        w.start()

    def _on_listed(self, res) -> None:
        if isinstance(res, str):
            self.modules = []
            self.lbl_status.setText(tr("modules.offline", error=res))
        else:
            self.modules = list(res)
            self.lbl_status.setText(tr("modules.all_ready") if not self.missing() else self._note)
        self._render()
        if self.modules and not self.missing():
            self.ready.emit()

    # ---------------------------------------------------------------- install
    def start_install(self) -> bool:
        if self.busy or not self.missing():
            return False
        self._note = ""
        w = ModulesWorker("install", self.manifest_fn, self.install_fn, [m.id for m in self.missing()], parent=self)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        self.worker = w
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.lbl_status.setText(tr("modules.downloading"))
        w.start()
        self._render()
        return True

    def cancel(self) -> None:
        if self.busy and self.worker is not None:
            self.worker.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self.bar.setValue(int(fraction * 1000))
        self.lbl_status.setText(f"{tr('modules.downloading')} {int(fraction * 100)} %  {message}")

    def _finish(self) -> None:
        self.bar.setVisible(False)
        if self.worker is not None:
            self.worker.wait(2000)
        self.refresh_after()

    def refresh_after(self) -> None:
        self.worker = None
        self.refresh()

    def _on_done(self, _n: int) -> None:
        self._finish()
        self.lbl_status.setText(tr("modules.all_ready"))

    def _on_failed(self, message: str) -> None:
        self._note = tr("modules.failed", error=message)
        self._finish()
        self.lbl_status.setText(self._note)

    def _on_cancelled(self) -> None:
        self._note = tr("modules.cancelled")
        self._finish()
        self.lbl_status.setText(self._note)

    def shutdown(self) -> None:
        if self.busy and self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
