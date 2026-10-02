"""Главное окно Voxprint (PySide6): выбрать аудио, выбрать текст, одна кнопка - и всё остальное автоматически."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from core.events import Stage
from infra import paths, platform_win
from workers.pipeline_runner import KIND_DATASET, KIND_LORA, TaskRequest, plan_for, run_task
from workers.process_worker import PrefetchWorker, ProcessWorker, UpdateWorker

log = logging.getLogger("voxprint.ui")

APP_TITLE = "Voxprint"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".mp4"}
TEXT_EXT = {".txt"}
ALL_STAGES: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.TRAIN, Stage.SAVE]

# Тёмная тема. В режиме Acrylic фон полупрозрачный (сквозь него виден размытый рабочий стол).
STYLE_TEMPLATE = """
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; font-size: 14px; color: #f2f2f5; }}
QWidget#root {{ background: {root_bg}; }}
QLabel#title {{ font-size: 26px; font-weight: 600; }}
QLabel#subtitle {{ color: rgba(242,242,245,170); }}
QFrame#card {{ background: rgba(255,255,255,{card_a}); border: 1px solid rgba(255,255,255,40); border-radius: 12px; }}
QLabel#fileLabel {{ color: rgba(242,242,245,190); }}
QPushButton {{ background: rgba(255,255,255,30); border: 1px solid rgba(255,255,255,50); border-radius: 8px;
              padding: 8px 16px; }}
QPushButton:hover {{ background: rgba(255,255,255,55); }}
QPushButton:pressed {{ background: rgba(255,255,255,20); }}
QPushButton:disabled {{ color: rgba(242,242,245,90); background: rgba(255,255,255,12); border-color: rgba(255,255,255,20); }}
QPushButton#primary {{ background: rgba(96,165,250,200); border: 1px solid rgba(147,197,253,220); font-size: 17px;
                      font-weight: 600; padding: 14px 20px; color: #0b1220; }}
QPushButton#primary:hover {{ background: rgba(125,185,255,230); }}
QPushButton#primary:disabled {{ background: rgba(96,165,250,60); color: rgba(242,242,245,110); border-color: rgba(147,197,253,60); }}
QProgressBar {{ background: rgba(255,255,255,25); border: 1px solid rgba(255,255,255,40); border-radius: 8px;
               height: 16px; text-align: center; }}
QProgressBar::chunk {{ background: rgba(96,165,250,220); border-radius: 7px; }}
QLabel#chip {{ color: rgba(242,242,245,110); padding: 3px 8px; border-radius: 10px; font-size: 12px; }}
QLabel#chip[state="active"] {{ color: #0b1220; background: rgba(147,197,253,230); font-weight: 600; }}
QLabel#chip[state="done"] {{ color: rgba(242,242,245,200); background: rgba(255,255,255,35); }}
QLabel#status {{ color: rgba(242,242,245,200); }}
QLabel#ready {{ font-size: 22px; font-weight: 600; color: #86efac; }}
QLabel#footer {{ color: rgba(242,242,245,120); font-size: 12px; }}
"""

PRIVACY_FOOTER = "Используйте только свой собственный голос. Записи и результаты хранятся только на этом компьютере."
PRIVACY_TITLE = "Конфиденциальность"
PRIVACY_TEXT = (
    "Используйте только свой собственный голос (или голос человека, который дал на это явное согласие).\n\n"
    "Записи, текст и готовый голос хранятся только на вашем компьютере; программа ничего не отправляет "
    "в интернет, кроме скачивания моделей и проверки обновлений.")


def privacy_marker() -> Path:
    return paths.state_dir() / "privacy_ack"


def privacy_acknowledged() -> bool:
    return privacy_marker().exists()


def acknowledge_privacy() -> None:
    try:
        privacy_marker().write_text("1", encoding="utf-8")
    except OSError:
        log.warning("cannot store privacy ack")



def build_style(glass: bool) -> str:
    if glass:
        return STYLE_TEMPLATE.format(root_bg="rgba(20,20,26,110)", card_a=18)
    return STYLE_TEMPLATE.format(root_bg="#17171c", card_a=10)


def open_folder(path: Path) -> None:
    """Открывает папку в Проводнике (Windows) или файловом менеджере."""
    p = str(path)
    if sys.platform == "win32":
        try:
            os.startfile(p)  # type: ignore[attr-defined]  # noqa: S606
            return
        except OSError:
            pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(p))


class MainWindow(QWidget):
    def __init__(self, runner: Callable[..., Any] = run_task, updater_factory: Optional[Callable[[], Any]] = None,
                 autocheck: bool = True, auto_open_folder: bool = True,
                 prefetch_fn: Optional[Callable[..., Any]] = None, prefetch: bool = False) -> None:
        super().__init__()
        self.runner = runner
        self.updater_factory = updater_factory
        self.auto_open_folder = auto_open_folder
        self.audio: Optional[Path] = None
        self.text: Optional[Path] = None
        self.result_dir: Optional[Path] = None
        self.worker: Optional[ProcessWorker] = None
        self.update_worker: Optional[UpdateWorker] = None
        self.prefetch_worker: Optional[PrefetchWorker] = None
        self.prefetch_fn = prefetch_fn
        self._last_request: Optional[TaskRequest] = None
        self._chips: dict = {}
        self.backdrop = "plain"
        self.last_error_text = ""

        self.setWindowTitle(APP_TITLE)
        self.setObjectName("root")
        self.setMinimumSize(800, 620)
        self.resize(820, 700)
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build()
        self.setStyleSheet(build_style(False))
        self._refresh_buttons()
        self._os_check = platform_win.check_os()
        if prefetch:
            QTimer.singleShot(300, self.start_prefetch)
        if autocheck:
            QTimer.singleShot(1500, self._startup_checks)

    # ------------------------------------------------------------------ интерфейс
    def _card(self) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        return f

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        sub = QLabel("Ваш голос из записи — в один клик. Выберите запись и текст, который вы читали.")
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(sub)

        card = self._card()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        self.btn_audio = QPushButton("Выбрать аудио")
        self.lbl_audio = QLabel("Файл не выбран (можно перетащить в окно)")
        self.btn_text = QPushButton("Выбрать текст")
        self.lbl_text = QLabel("Файл не выбран (txt, UTF-8)")
        for b, l in ((self.btn_audio, self.lbl_audio), (self.btn_text, self.lbl_text)):
            row = QHBoxLayout()
            b.setMinimumWidth(170)
            l.setObjectName("fileLabel")
            l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row.addWidget(b)
            row.addWidget(l, 1)
            cl.addLayout(row)
        root.addWidget(card)

        self.btn_lora = QPushButton("Создать голос (LoRA)")
        self.btn_lora.setObjectName("primary")
        self.btn_lora.setToolTip("Всё сразу: разметка записи, нарезка и обучение голоса")
        root.addWidget(self.btn_lora)

        row2 = QHBoxLayout()
        self.btn_dataset = QPushButton("Создать датасет")
        self.btn_dataset.setToolTip("Только подготовить датасет (без обучения)")
        self.btn_update = QPushButton("Проверить обновления")
        row2.addWidget(self.btn_dataset)
        row2.addWidget(self.btn_update)
        root.addLayout(row2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        for st in ALL_STAGES:
            c = QLabel(st.label)
            c.setObjectName("chip")
            c.setProperty("state", "idle")
            self._chips[st] = c
            chips.addWidget(c)
        chips.addStretch(1)
        root.addLayout(chips)

        self.lbl_status = QLabel("Выберите аудио и текст — дальше всё сделаю сам.")
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.lbl_ready = QLabel("Готово")
        self.lbl_ready.setObjectName("ready")
        self.lbl_ready.hide()
        self.btn_open = QPushButton("Открыть папку с результатом")
        self.btn_open.hide()
        self.btn_cancel = QPushButton("Отменить")
        self.btn_cancel.hide()
        root.addWidget(self.lbl_ready)
        root.addWidget(self.btn_open)
        root.addWidget(self.btn_cancel)
        root.addStretch(1)
        self.lbl_privacy = QLabel(PRIVACY_FOOTER)
        self.lbl_privacy.setObjectName("footer")
        self.lbl_privacy.setWordWrap(True)
        root.addWidget(self.lbl_privacy)

        self.btn_audio.clicked.connect(self.choose_audio)
        self.btn_text.clicked.connect(self.choose_text)
        self.btn_dataset.clicked.connect(lambda: self.start(KIND_DATASET))
        self.btn_lora.clicked.connect(lambda: self.start(KIND_LORA))
        self.btn_update.clicked.connect(self.check_updates)
        self.btn_open.clicked.connect(self.open_result)
        self.btn_cancel.clicked.connect(self.cancel)

    # ------------------------------------------------------------------ окно/эффекты
    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        if sys.platform == "win32" and self.backdrop == "plain":
            self.backdrop = platform_win.apply_backdrop(int(self.winId()))
            self.setStyleSheet(build_style(self.backdrop == "acrylic"))
            if self.backdrop != "acrylic":
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    # ------------------------------------------------------------------ выбор файлов
    def set_audio(self, path: Path) -> None:
        self.audio = Path(path)
        self.lbl_audio.setText(self.audio.name)
        self.lbl_audio.setToolTip(str(self.audio))
        self._refresh_buttons()

    def set_text(self, path: Path) -> None:
        self.text = Path(path)
        self.lbl_text.setText(self.text.name)
        self.lbl_text.setToolTip(str(self.text))
        self._refresh_buttons()

    def choose_audio(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXT))
        f, _ = QFileDialog.getOpenFileName(self, "Выберите запись голоса", "", f"Аудио ({exts});;Все файлы (*.*)")
        if f:
            self.set_audio(Path(f))

    def choose_text(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Выберите текстовый файл", "", "Текст (*.txt);;Все файлы (*.*)")
        if f:
            self.set_text(Path(f))

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in TEXT_EXT:
                self.set_text(p)
            elif p.suffix.lower() in AUDIO_EXT:
                self.set_audio(p)

    # ------------------------------------------------------------------ состояние
    @property
    def busy(self) -> bool:
        return bool((self.worker and self.worker.isRunning()) or
                    (self.prefetch_worker and self.prefetch_worker.isRunning()))

    def _refresh_buttons(self) -> None:
        busy = self.busy
        ready = bool(self.audio and self.text) and not busy
        self.btn_audio.setEnabled(not busy)
        self.btn_text.setEnabled(not busy)
        self.btn_lora.setEnabled(ready)
        self.btn_dataset.setEnabled(ready)
        self.btn_update.setEnabled(not busy and not (self.update_worker and self.update_worker.isRunning()))
        self.btn_cancel.setVisible(bool(self.worker and self.worker.isRunning()))

    def _set_chip(self, stage_label: str) -> None:
        active_idx = next((i for i, s in enumerate(ALL_STAGES) if s.label == stage_label), -1)
        for i, s in enumerate(ALL_STAGES):
            chip = self._chips[s]
            state = "active" if i == active_idx else ("done" if 0 <= active_idx and i < active_idx else "idle")
            chip.setProperty("state", state)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def _reset_chips(self) -> None:
        self._set_chip("")

    # ------------------------------------------------------------------ запуск
    def start(self, kind: str, force_cpu: bool = False) -> None:
        if self.busy or not (self.audio and self.text):
            return
        req = TaskRequest(kind=kind, audio=self.audio, text=self.text, force_cpu=force_cpu)
        self._last_request = req
        self.lbl_ready.hide()
        self.btn_open.hide()
        self.progress.setValue(0)
        self._reset_chips()
        self.lbl_status.setText("Начинаю работу…")
        self.worker = ProcessWorker(req, self.runner, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self._refresh_buttons)
        self.worker.start()
        self._refresh_buttons()

    def cancel(self) -> None:
        if self.worker:
            self.lbl_status.setText("Останавливаю…")
            self.worker.cancel()

    def on_progress(self, pct: int, stage_label: str, message: str) -> None:
        self.progress.setValue(pct)
        self._set_chip(stage_label)
        self.lbl_status.setText(message)

    def on_done(self, result: Any) -> None:
        self.progress.setValue(100)
        self._set_chip(Stage.SAVE.label)
        self.result_dir = Path(result.open_dir)
        extra = ""
        if getattr(result, "adapter_path", None):
            extra = "\nГолосовой адаптер сохранён в папке output."
        warn = ("\n" + "\n".join(result.warnings[-3:])) if getattr(result, "warnings", None) else ""
        self.lbl_status.setText(f"Фрагментов в датасете: {result.n_segments}.{extra}{warn}"
                                + (f"\n{result.update_summary}" if getattr(result, "update_summary", "") else ""))
        self.lbl_ready.show()
        self.btn_open.show()
        self._refresh_buttons()
        if self.auto_open_folder:
            open_folder(self.result_dir)

    def on_cancelled(self) -> None:
        self.lbl_status.setText("Отменено.")
        self.progress.setValue(0)
        self._reset_chips()

    def on_failed(self, kind: str, message: str, details: str, url: str) -> None:
        self.last_error_text = message
        self.progress.setValue(0)
        self._reset_chips()
        self.lbl_status.setText(message)
        log.error("task failed: kind=%s msg=%s details=%s", kind, message, details)
        self.show_error(kind, message, details, url)

    # ------------------------------------------------------------------ дружелюбные ошибки
    def show_error(self, kind: str, message: str, details: str = "", url: str = "") -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(APP_TITLE)
        titles = {"mismatch": "Запись и текст не совпадают", "oom": "Не хватило видеопамяти",
                  "download": "Не удалось скачать модель", "text_read": "Проблема с текстовым файлом",
                  "audio_read": "Проблема с аудиофайлом"}
        box.setText(f"<b>{titles.get(kind, 'Не получилось')}</b>")
        body = message
        if kind == "download" and url:
            body += f'<br><br>Страница модели: <a href="{url}">{url}</a>'
        elif details and kind == "other":
            body += f"<br><small>{details}</small>"
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setInformativeText(body.replace("\n", "<br>"))
        retry_cpu = None
        if kind == "oom":
            retry_cpu = box.addButton("Продолжить на процессоре (медленно)", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        else:
            box.addButton("Понятно", QMessageBox.ButtonRole.AcceptRole)
        self._error_box = box
        box.setModal(True)
        box.finished.connect(lambda _=0: None)
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # в тестах не блокируем
            self._retry_cpu_button = retry_cpu
            return
        box.exec()
        if retry_cpu is not None and box.clickedButton() is retry_cpu and self._last_request:
            self.start(self._last_request.kind, force_cpu=True)

    # ------------------------------------------------------------------ обновления
    def check_updates(self) -> None:
        if self.update_worker and self.update_worker.isRunning():
            return
        self.lbl_status.setText("Проверяю обновления…")
        self.progress.setValue(0)
        self.update_worker = UpdateWorker(self.updater_factory, parent=self)
        self.update_worker.progress.connect(lambda p, s, m: (self.progress.setValue(p), self.lbl_status.setText(m)))
        self.update_worker.done.connect(self._on_update_done)
        self.update_worker.failed.connect(lambda k, m, d, u: self.lbl_status.setText(m))
        self.update_worker.finished.connect(self._refresh_buttons)
        self.update_worker.start()
        self._refresh_buttons()

    def _on_update_done(self, summary: str, changed: bool) -> None:
        self.progress.setValue(100 if changed else 0)
        self.lbl_status.setText(summary or "Установлены проверенные версии.")

    def _startup_checks(self) -> None:
        """Памятка о конфиденциальности (один раз), предупреждение об ОС (мягкое) и тихая еженедельная проверка."""
        self.maybe_show_privacy_notice()
        if not self._os_check.ok and self._os_check.message and \
                os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.information(self, APP_TITLE, self._os_check.message)
        if self.busy:
            return
        w = UpdateWorker(self.updater_factory, silent=True, only_if_due=True, parent=self)
        w.done.connect(lambda s, changed: changed and self.lbl_status.setText(s))
        self.update_worker = w
        w.start()

    def maybe_show_privacy_notice(self) -> bool:
        """Один раз при первом запуске. В тестах (offscreen) окно не показывается и отметка не ставится."""
        if privacy_acknowledged() or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return False
        QMessageBox.information(self, PRIVACY_TITLE, PRIVACY_TEXT)
        acknowledge_privacy()
        return True

    # ------------------------------------------------------------------ первый запуск
    def start_prefetch(self) -> None:
        """Автоматическая докачка моделей при первом запуске (без вопросов пользователю)."""
        if self.busy:
            return
        w = PrefetchWorker(self.prefetch_fn, parent=self)
        self.prefetch_worker = w
        self.lbl_status.setText("Первый запуск: подготавливаю программу (скачиваю модели, это делается один раз)…")
        w.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl_status.setText(m)))
        w.done.connect(self._on_prefetch_done)
        w.failed.connect(self._on_prefetch_failed)
        w.finished.connect(self._refresh_buttons)
        w.start()
        self._refresh_buttons()

    def _on_prefetch_done(self, downloaded: list) -> None:
        self.progress.setValue(0)
        self.lbl_status.setText("Всё готово к работе. Выберите аудио и текст." if downloaded
                                else "Выберите аудио и текст — дальше всё сделаю сам.")

    def _on_prefetch_failed(self, message: str, url: str) -> None:
        self.progress.setValue(0)
        self.lbl_status.setText(message + " Скачивание повторится автоматически при запуске работы.")

    def open_result(self) -> None:
        if self.result_dir:
            open_folder(self.result_dir)

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(e)
