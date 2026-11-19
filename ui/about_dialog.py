""""About / Help" dialog: what Voxprint does, the steps, requirements, privacy, authors and the component list.

All texts come from the locale files; the component list is read from ``credits.json`` through :mod:`core.appinfo`.
"""
from __future__ import annotations

import html
import os
from typing import Callable, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core import appinfo
from core.i18n import get_language, tr
from ui.glass import GlassDialog


def _open_url_default(url: QUrl) -> bool:
    """Open a URL in the system's default browser/application."""
    return QDesktopServices.openUrl(url)


class AboutDialog(GlassDialog):
    """Modal About dialog with a scrollable rich-text body and buttons for the licence notices and the repository."""
    def __init__(self, parent: Optional[QWidget] = None, repo_url: Optional[str] = None,
                 open_url: Callable[[QUrl], bool] = _open_url_default) -> None:
        """``open_url`` is injectable so tests never launch a real browser."""
        super().__init__(parent)
        self._open_url = open_url
        self.repo_link = appinfo.public_repo_url(repo_url)   # None while the address is still the OWNER placeholder
        self.setWindowTitle(tr("about.title"))
        self.setObjectName("root")
        self.setMinimumSize(640, 600)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)

        title = QLabel(f"{html.escape(appinfo.APP_DISPLAY_NAME)}  <span style='font-size:14px;color:#9aa0aa'>"
                       f"{html.escape(tr('about.version', version=appinfo.version_label()))}</span>")
        title.setObjectName("title")
        title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(title)

        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(True)
        self.body.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        self.body.setHtml(self._build_html())
        lay.addWidget(self.body, 1)

        row = QHBoxLayout()
        self.btn_notices = QPushButton(tr("about.btn_notices"))
        self.btn_repo = QPushButton(tr("about.btn_repo"))
        self.btn_repo.setVisible(self.repo_link is not None)
        if self.repo_link:
            self.btn_repo.setToolTip(self.repo_link)
        self.btn_close = QPushButton(tr("about.btn_close"))
        row.addWidget(self.btn_notices)
        row.addWidget(self.btn_repo)
        row.addStretch(1)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.btn_notices.clicked.connect(self.open_notices)
        self.btn_repo.clicked.connect(self.open_repo)
        self.btn_close.clicked.connect(self.accept)

    # ------------------------------------------------------------------ content
    def _build_html(self) -> str:
        """Build the HTML body (what/concept/steps/requirements/privacy/authors/components); all values are escaped."""
        e = html.escape
        lang = get_language()
        steps = "".join(f"<li>{e(tr(f'about.step{i}'))}</li>" for i in range(1, 6))
        parts = [
            f"<p>{e(tr('about.what'))}</p>",
            f"<h3>{e(tr('about.concept_title'))}</h3><p>{e(tr('about.concept'))}</p>",
            f"<h3>{e(tr('about.steps_title'))}</h3><ul>{steps}</ul>",
            f"<p>{e(tr('about.requirements'))}</p>",
            f"<p>{e(tr('about.privacy'))}</p>",
            f"<p>{e(tr('about.author', author=appinfo.APP_AUTHOR))}</p>",
            f"<h3>{e(tr('about.components'))}</h3><p>{e(tr('about.components_intro'))}</p><ul>",
        ]
        for c in appinfo.components():
            kind = tr(f"about.kind_{c['kind']}")
            tag = f", {e(tr('about.optional'))}" if c.get("ship") == "optional" else ""
            parts.append(
                f"<li><b><a href='{e(c['url'])}'>{e(c['name'])}</a></b> - "
                f"{e(appinfo.localized(c.get('purpose'), lang))} "
                f"<i>({e(c['license'])}; {e(kind)}{tag})</i></li>")
        parts.append("</ul>")
        parts.append(f"<p>{e(tr('about.lgpl'))}</p>")
        return "<html><body style='font-size:13px'>" + "".join(parts) + "</body></html>"

    # ------------------------------------------------------------------ actions
    def open_repo(self) -> bool:
        """Open the repository in the default browser (only if a real address is configured)."""
        if not self.repo_link:
            return False
        return bool(self._open_url(QUrl(self.repo_link)))

    def open_notices(self) -> bool:
        """Open the bundled THIRD_PARTY_NOTICES file; show a message (outside the offscreen platform) if it is missing."""
        path = appinfo.notices_path()
        if path.exists():
            return bool(self._open_url(QUrl.fromLocalFile(str(path))))
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            from ui import std_buttons

            std_buttons.information(self, tr("about.title"), tr("about.notices_missing"))
        return False
