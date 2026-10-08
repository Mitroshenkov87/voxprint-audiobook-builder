"""Standard dialog buttons (OK, Cancel, Yes, No ...) in the app's UI language, not the system's.

Qt labels the standard buttons of ``QMessageBox`` / ``QDialogButtonBox`` itself, which showed a Russian "OK" in the
privacy notice of an English UI on a Russian Windows.  Two layers fix that for every locale:

* :class:`ButtonTranslator` (installed once on the application, :func:`install`) answers Qt's own lookup of the button
  texts with our catalog (``qt.btn.*``), for every dialog - also the ones Qt builds internally;
* :func:`localize` sets the texts directly on a message box / button box; :func:`information` and :func:`question` are
  the ``QMessageBox`` shortcuts the app uses, with the labels set explicitly.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox, QWidget

from core.i18n import tr

SB = QMessageBox.StandardButton
#: standard button -> catalog key (the same values are used by QDialogButtonBox.StandardButton)
KEYS = {
    SB.Ok: "qt.btn.ok", SB.Cancel: "qt.btn.cancel", SB.Yes: "qt.btn.yes", SB.No: "qt.btn.no", SB.Close: "qt.btn.close",
    SB.Apply: "qt.btn.apply", SB.Save: "qt.btn.save", SB.Open: "qt.btn.open", SB.Discard: "qt.btn.discard",
    SB.Retry: "qt.btn.retry", SB.Ignore: "qt.btn.ignore", SB.Abort: "qt.btn.abort", SB.Help: "qt.btn.help",
    SB.Reset: "qt.btn.reset", SB.RestoreDefaults: "qt.btn.restore_defaults", SB.YesToAll: "qt.btn.yes_all",
    SB.NoToAll: "qt.btn.no_all", SB.SaveAll: "qt.btn.save_all",
}
#: Qt's English source texts of the standard buttons (QPlatformTheme / QDialogButtonBox / QMessageBox contexts)
SOURCES = {
    "OK": "qt.btn.ok", "&OK": "qt.btn.ok", "Cancel": "qt.btn.cancel", "&Cancel": "qt.btn.cancel", "&Yes": "qt.btn.yes",
    "Yes": "qt.btn.yes", "&No": "qt.btn.no", "No": "qt.btn.no", "Close": "qt.btn.close", "&Close": "qt.btn.close",
    "Apply": "qt.btn.apply", "Save": "qt.btn.save", "&Save": "qt.btn.save", "Open": "qt.btn.open", "Discard": "qt.btn.discard",
    "Retry": "qt.btn.retry", "Ignore": "qt.btn.ignore", "Abort": "qt.btn.abort", "Help": "qt.btn.help",
    "Reset": "qt.btn.reset", "Restore Defaults": "qt.btn.restore_defaults", "Yes to &All": "qt.btn.yes_all",
    "N&o to All": "qt.btn.no_all", "Save All": "qt.btn.save_all",
}
CONTEXTS = ("QPlatformTheme", "QDialogButtonBox", "QMessageBox", "QGnomeTheme")


class ButtonTranslator(QTranslator):
    """Translates only the standard button texts, always into the CURRENT UI language (a language switch needs no reload)."""

    def translate(self, context: str, source: str, disambiguation: Optional[str] = None,
                  n: int = -1) -> Optional[str]:  # noqa: D401
        key = SOURCES.get(source) if context in CONTEXTS else None
        # None = "not mine": Qt keeps its own text (an empty string would REPLACE every other text of Qt with nothing)
        return tr(key) if key else None

    def isEmpty(self) -> bool:  # noqa: N802 - Qt name
        return False


_installed: Optional[ButtonTranslator] = None


def install(app: Optional[QCoreApplication] = None) -> Optional[ButtonTranslator]:
    """Install the translator on the application once (idempotent); returns it, or None without an application."""
    global _installed
    app = app or QCoreApplication.instance()
    if app is None:
        return None
    if _installed is None:
        _installed = ButtonTranslator(app)
        app.installTranslator(_installed)
    return _installed


def localize(box) -> None:
    """Put the UI-language texts on every standard button of a ``QMessageBox`` or ``QDialogButtonBox``."""
    for sb, key in KEYS.items():
        btn = box.button(sb) if isinstance(box, QMessageBox) else box.button(QDialogButtonBox.StandardButton(sb.value))
        if btn is not None:
            btn.setText(tr(key))


def information(parent: Optional[QWidget], title: str, text: str) -> None:
    """``QMessageBox.information`` with an OK button in the UI language."""
    box = QMessageBox(QMessageBox.Icon.Information, title, text, SB.Ok, parent)
    localize(box)
    box.exec()


def question(parent: Optional[QWidget], title: str, text: str) -> bool:
    """``QMessageBox.question`` (Yes / No, in the UI language); True for Yes."""
    box = QMessageBox(QMessageBox.Icon.Question, title, text, SB.Yes | SB.No, parent)
    box.setDefaultButton(SB.No)
    localize(box)
    return box.exec() == SB.Yes
