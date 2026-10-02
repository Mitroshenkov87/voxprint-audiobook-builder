"""Автоматическая загрузка моделей с Hugging Face в локальный каталог приложения.

Загрузка идёт в `<имя>.partial`, после проверки каталог переименовывается (частично скачанная
модель никогда не считается готовой). Ревизия (sha коммита) хранится в файле `.revision`.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.errors import ModelDownloadError
from core.events import ProgressCallback, Stage, noop_progress
from infra import paths

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
log = logging.getLogger("voxprint.models")

ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
#: Примерный размер, ГБ (для проверки свободного места; оценка, не точное значение).
APPROX_SIZE_GB = {
    ALIGNER_REPO: 2.0,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": 4.5,
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": 2.5,
}


def hf_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def local_dir_for(repo_id: str, root: Optional[Path] = None) -> Path:
    return (root or paths.models_dir()) / repo_id.replace("/", "--")


def verify_local_model(path: Path) -> bool:
    """Модель считается готовой, если есть config.json и хотя бы один файл весов."""
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_weights


def local_revision(repo_id: str, root: Optional[Path] = None) -> Optional[str]:
    f = local_dir_for(repo_id, root) / ".revision"
    try:
        return f.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def local_revisions(repos, root: Optional[Path] = None) -> Dict[str, str]:
    out = {}
    for r in repos:
        d = local_dir_for(r, root)
        rev = local_revision(r, root)
        if rev and verify_local_model(d):
            out[r] = rev
    return out


class _ByteProgress:
    """Суммирует прогресс по всем файловым tqdm-барам huggingface_hub."""

    def __init__(self, cb: Callable[[float], None]) -> None:
        self.cb = cb
        self.lock = threading.Lock()
        self.total = 0
        self.done = 0

    def make_tqdm_class(self):
        from huggingface_hub.utils import tqdm as hf_tqdm

        tracker = self

        class _Tqdm(hf_tqdm):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **kw: Any) -> None:
                kw["disable"] = True if kw.get("unit") != "B" else kw.get("disable", False)
                super().__init__(*a, **kw)
                self._is_bytes = kw.get("unit") == "B"
                if self._is_bytes and self.total:
                    with tracker.lock:
                        tracker.total += int(self.total)

            def update(self, n: float = 1) -> Optional[bool]:
                r = super().update(n)
                if getattr(self, "_is_bytes", False):
                    with tracker.lock:
                        tracker.done += int(n)
                        if tracker.total:
                            tracker.cb(min(1.0, tracker.done / tracker.total))
                return r

        return _Tqdm


def ensure_model(
    repo_id: str,
    progress: ProgressCallback = noop_progress,
    root: Optional[Path] = None,
    snapshot_download: Optional[Callable[..., str]] = None,
    get_remote_sha: Optional[Callable[[str], Optional[str]]] = None,
    stage: Stage = Stage.MODEL,
    revision: Optional[str] = None,
) -> Path:
    """Возвращает путь к локальной модели; скачивает при первом запуске (автоматически).

    revision - коммит HF; по умолчанию берётся проверенная ревизия из манифеста «проверено Voxprint».
    Если закреплённая ревизия недоступна, один раз пробуем актуальную (с записью в журнал)."""
    target = local_dir_for(repo_id, root)
    if verify_local_model(target):
        return target

    short = repo_id.split("/")[-1]
    progress(stage, 0.0, f"Первый запуск: скачиваю модель {short} (это делается один раз)…")
    need_gb = APPROX_SIZE_GB.get(repo_id, 3.0)
    try:
        free_gb = shutil.disk_usage(target.parent).free / 1024 ** 3
    except OSError:
        free_gb = None
    if free_gb is not None and free_gb < need_gb * 1.2:
        raise ModelDownloadError(
            f"Недостаточно места на диске для модели {short}: нужно около {need_gb:.0f} ГБ, свободно {free_gb:.1f} ГБ.",
            url=hf_url(repo_id))

    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as _sd
        except ImportError as exc:
            raise ModelDownloadError("Не найден компонент загрузки моделей. Переустановите программу.",
                                     url=hf_url(repo_id), details=str(exc)) from exc
        snapshot_download = _sd
    if revision is None:
        try:
            from infra.verified_manifest import load_bundled

            revision = load_bundled().models.get(repo_id)
        except Exception:  # noqa: BLE001 - манифест необязателен
            revision = None
    sha: Optional[str] = revision
    try:
        if revision:
            pass  # ревизия закреплена - запрос HEAD не нужен
        elif get_remote_sha is not None:
            sha = get_remote_sha(repo_id)
        else:
            from huggingface_hub import HfApi

            sha = HfApi().model_info(repo_id).sha
    except Exception as exc:  # noqa: BLE001 - сеть может быть недоступна, ошибка всплывёт ниже
        log.warning("remote sha unavailable: %s", exc)

    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
    tracker = _ByteProgress(lambda f: progress(stage, f, f"Скачиваю модель {short}: {int(f * 100)}%"))
    def _download(rev: Optional[str]) -> None:
        kwargs: Dict[str, Any] = {"repo_id": repo_id, "local_dir": str(partial)}
        if rev:
            kwargs["revision"] = rev
        try:
            snapshot_download(tqdm_class=tracker.make_tqdm_class(), **kwargs)
        except TypeError:  # старые/новые версии hub без tqdm_class
            snapshot_download(**kwargs)

    try:
        try:
            _download(revision)
            if revision:
                sha = revision
        except Exception as exc:  # noqa: BLE001
            if not revision:
                raise
            log.warning("pinned revision %s of %s unavailable (%s) - trying latest", revision[:8], repo_id, exc)
            shutil.rmtree(partial, ignore_errors=True)
            _download(None)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(partial, ignore_errors=True)
        raise ModelDownloadError(
            f"Не удалось скачать модель {short}. Проверьте подключение к интернету и повторите. "
            f"Модель можно скачать вручную: {hf_url(repo_id)}",
            url=hf_url(repo_id), details=f"{type(exc).__name__}: {exc}") from exc
    if not verify_local_model(partial):
        shutil.rmtree(partial, ignore_errors=True)
        raise ModelDownloadError(f"Модель {short} скачалась не полностью. Повторите попытку.",
                                 url=hf_url(repo_id))
    if sha:
        (partial / ".revision").write_text(sha, encoding="utf-8")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    progress(stage, 1.0, f"Модель {short} готова")
    return target


def ensure_aligner_model(progress: ProgressCallback = noop_progress, **kw: Any) -> Path:
    return ensure_model(ALIGNER_REPO, progress, **kw)


def ensure_tts_models(base_repo: str, progress: ProgressCallback = noop_progress, **kw: Any) -> Dict[str, Path]:
    """База TTS для обучения. Аудио-токенайзер (speech_tokenizer/) лежит внутри репозитория Base-модели
    (проверено по списку файлов HF), отдельный репозиторий токенайзера не нужен."""
    return {"base": ensure_model(base_repo, progress, stage=Stage.MODEL, **kw)}
