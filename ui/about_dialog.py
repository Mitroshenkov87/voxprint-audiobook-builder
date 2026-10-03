"""Окно «О программе / Справка»: что делает Voxprint, шаги, требования, приватность, авторы и список компонентов."""
from __future__ import annotations

import html
import os
from typing import Callable, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser, QVBoxLayout,
                               QWidget)

from core import appinfo
from core.i18n import get_language, tr


def _open_url_default(url: QUrl) -> bool:
    return QDesktopServices.openUrl(url)


class AboutDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, repo_url: Optional[str] = None,
                 open_url: Callable[[QUrl], bool] = _open_url_default) -> None:
        super().__init__(parent)
        self._open_url = open_url
        self.repo_link = appinfo.public_repo_url(repo_url)   # None, пока адрес - заглушка (OWNER)
        self.setWindowTitle(tr("about.title"))
        self.setObjectName("root")
        self.setMinimumSize(640, 600)
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        title = QLabel(f"{appinfo.APP_NAME}  <span style='font-size:14px;color:#9aa0aa'>"
                       f"{html.escape(tr('about.version', version=appinfo.APP_VERSION))}</span>")
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

    # ------------------------------------------------------------------ содержимое
    def _build_html(self) -> str:
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

    # ------------------------------------------------------------------ действия
    def open_repo(self) -> bool:
        """Открывает репозиторий в браузере по умолчанию (если адрес задан)."""
        if not self.repo_link:
            return False
        return bool(self._open_url(QUrl(self.repo_link)))

    def open_notices(self) -> bool:
        path = appinfo.notices_path()
        if path.exists():
            return bool(self._open_url(QUrl.fromLocalFile(str(path))))
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(self, tr("about.title"), tr("about.notices_missing"))
        return False
