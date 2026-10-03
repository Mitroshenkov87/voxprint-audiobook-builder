"""Автоматическая загрузка моделей с Hugging Face в локальный каталог приложения.

Загрузка идёт в `<имя>.partial`, после проверки каталог переименовывается (частично скачанная
модель никогда не считается готовой). Ревизия (sha коммита) хранится в файле `.revision`.
"""
from __future__ import annotations

from core.i18n import tr
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.errors import ModelDownloadError
from core.events import ProgressCallback, Stage, noop_progress
from infra import modelscope_mirror, paths

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


def pinned_revision(repo_id: str) -> Optional[str]:
    """Commit pinned for this model in the bundled «verified by Voxprint» manifest (None if not pinned)."""
    try:
        from infra.verified_manifest import load_bundled

        return load_bundled().models.get(repo_id)
    except Exception:  # noqa: BLE001 - манифест необязателен
        return None


ENV_NO_MIRROR = "VOXPRINT_NO_MIRROR"


def mirror_enabled() -> bool:
    """ModelScope-зеркало включено (отключить: VOXPRINT_NO_MIRROR=1)."""
    return os.environ.get(ENV_NO_MIRROR, "").strip().lower() not in ("1", "true", "yes", "on")


def _default_mirror_download(repo_id, dest, progress, expected_sizes):
    return modelscope_mirror.download_repo(repo_id, dest, progress, expected_sizes)


def external_model(repo_id: str, revision: Optional[str] = None):
    """Копия модели, скачанная другой программой (только чтение), либо None.

    См. core/model_locator.py: проверяются полнота файлов и закреплённая ревизия; ничего не копируется
    и не изменяется. Эта папка НИКОГДА не обновляется и не удаляется Voxprint."""
    from core import model_locator

    try:
        return model_locator.find_model(repo_id, revision if revision else pinned_revision(repo_id))
    except Exception as exc:  # noqa: BLE001 - поиск чужих копий не должен ломать обычную загрузку
        log.warning("external model search failed for %s: %s", repo_id, exc)
        return None


STATE_MISSING, STATE_PARTIAL, STATE_READY = "missing", "partial", "ready"


def partial_dir_has_data(repo_id: str, root: Optional[Path] = None) -> bool:
    p = local_dir_for(repo_id, root)
    p = p.with_name(p.name + ".partial")
    try:
        return p.is_dir() and any(p.iterdir())
    except OSError:
        return False


def model_state(repo_id: str) -> str:
    """ready: usable now (Voxprint's own folder or another app's copy); partial: an interrupted download that will
    be resumed (ours, or an unfinished one in another app's cache - which we never touch); missing: nothing yet."""
    if verify_local_model(local_dir_for(repo_id)) or external_model(repo_id) is not None:
        return STATE_READY
    if partial_dir_has_data(repo_id):
        return STATE_PARTIAL
    try:
        from core import model_locator

        if model_locator.has_unfinished_download(repo_id):
            return STATE_PARTIAL
    except Exception:  # noqa: BLE001
        pass
    return STATE_MISSING


def model_states(repos) -> Dict[str, str]:
    return {r: model_state(r) for r in repos}


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
    reuse_external: bool = True,
    mirror_download: Optional[Callable[..., Any]] = None,
    hf_probe: Optional[Callable[[str], bool]] = None,
) -> Path:
    """Возвращает путь к локальной модели; скачивает при первом запуске (автоматически).

    Перед скачиванием ищем полную копию, оставленную другой программой (кэш Hugging Face, Pinokio/Alexandria,
    ModelScope...): она используется на месте, только для чтения (core/model_locator.py). Только для основного
    каталога моделей (root=None): обновление через промежуточный каталог всегда скачивает свою копию.

    revision - коммит HF; по умолчанию берётся проверенная ревизия из манифеста «проверено Voxprint».
    Если закреплённая ревизия недоступна, один раз пробуем актуальную (с записью в журнал)."""
    target = local_dir_for(repo_id, root)
    if verify_local_model(target):
        return target

    short = repo_id.split("/")[-1]
    if reuse_external and root is None:
        found = external_model(repo_id, revision)
        if found is not None:
            progress(stage, 1.0, tr("progress.model_reused", short=short, where=str(found.location)))
            return found.path
    if partial_dir_has_data(repo_id, root):
        progress(stage, 0.0, tr("progress.model_resume", short=short))
    else:
        progress(stage, 0.0, tr("progress.first_download", short=short))
    need_gb = APPROX_SIZE_GB.get(repo_id, 3.0)
    try:
        free_gb = shutil.disk_usage(target.parent).free / 1024 ** 3
    except OSError:
        free_gb = None
    if free_gb is not None and free_gb < need_gb * 1.2:
        raise ModelDownloadError(
            tr("err.disk_model", short=short, need=f"{need_gb:.0f}", free=f"{free_gb:.1f}"),
            url=hf_url(repo_id))

    if revision is None:
        revision = pinned_revision(repo_id)
    sha: Optional[str] = revision
    if not revision:
        try:
            if get_remote_sha is not None:
                sha = get_remote_sha(repo_id)
            else:
                from huggingface_hub import HfApi

                sha = HfApi().model_info(repo_id).sha
        except Exception as exc:  # noqa: BLE001 - сеть может быть недоступна, ошибка всплывёт ниже
            log.warning("remote sha unavailable: %s", exc)

    partial = target.with_name(target.name + ".partial")   # остаётся между запусками: докачка с места обрыва
    tracker = _ByteProgress(lambda f: progress(stage, f, tr("progress.downloading", short=short, pct=int(f * 100))))
    state = {"sha": sha}

    def _download(rev: Optional[str]) -> None:
        sd = snapshot_download
        if sd is None:
            try:
                from huggingface_hub import snapshot_download as sd  # type: ignore[no-redef]
            except ImportError as exc:
                raise ModelDownloadError(tr("err.hub_missing"), url=hf_url(repo_id), details=str(exc)) from exc
        kwargs: Dict[str, Any] = {"repo_id": repo_id, "local_dir": str(partial)}
        if rev:
            kwargs["revision"] = rev
        try:
            sd(tqdm_class=tracker.make_tqdm_class(), **kwargs)
        except TypeError:  # старые/новые версии hub без tqdm_class
            sd(**kwargs)

    def _from_hf() -> None:
        try:
            _download(revision)
        except Exception as exc:  # noqa: BLE001
            if not revision:
                raise
            log.warning("pinned revision %s of %s unavailable (%s) - trying latest", revision[:8], repo_id, exc)
            if not isinstance(exc, OSError):    # сетевой сбой: файлы пригодны для докачки; ошибка ревизии: убираем
                shutil.rmtree(partial, ignore_errors=True)
            state["sha"] = None
            _download(None)
            try:
                state["sha"] = (get_remote_sha(repo_id) if get_remote_sha is not None else None)
            except Exception:  # noqa: BLE001
                state["sha"] = None

    def _from_modelscope() -> None:
        from core import model_locator

        known = model_locator.KNOWN_SIZES.get(repo_id)
        expected = known[1] if known and revision and known[0] == revision else None
        progress(stage, 0.0, tr("progress.mirror_modelscope", short=short))
        mirror_download(repo_id, partial,
                        lambda f: progress(stage, f, tr("progress.downloading", short=short, pct=int(f * 100))),
                        expected)
        # ModelScope has no commit sha: the revision is confirmed only when the sizes matched the pinned commit
        state["sha"] = revision if expected else None

    mirror_ok = mirror_enabled() and root is None   # обновление через staging всегда идёт напрямую с HF
    if mirror_ok:
        mirror_download = mirror_download or _default_mirror_download
        fast = (hf_probe or modelscope_mirror.hf_is_fast)(repo_id)
        order = ["hf", "ms"] if fast else ["ms", "hf"]
        if not fast:
            log.warning("Hugging Face is slow or unreachable - trying ModelScope first for %s", repo_id)
    else:
        order = ["hf"]
    errors: List[str] = []
    ok_source = ""
    for src in order:
        try:
            if src == "hf":
                _from_hf()
            else:
                if errors:
                    log.warning("download of %s from Hugging Face failed - trying ModelScope", repo_id)
                _from_modelscope()
            ok_source = src
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {type(exc).__name__}: {exc}")
            log.warning("download of %s from %s failed: %s", repo_id, src, exc)
    if not ok_source:
        # .partial остаётся на диске: следующая попытка продолжит с места обрыва
        raise ModelDownloadError(
            tr("err.download_failed", short=short, url=hf_url(repo_id)),
            url=hf_url(repo_id), details=" | ".join(errors))
    sha = state["sha"]
    if not verify_local_model(partial):
        shutil.rmtree(partial, ignore_errors=True)
        raise ModelDownloadError(tr("err.model_partial", short=short),
                                 url=hf_url(repo_id))
    if sha:
        (partial / ".revision").write_text(sha, encoding="utf-8")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    progress(stage, 1.0, tr("progress.model_done", short=short))
    return target


def ensure_aligner_model(progress: ProgressCallback = noop_progress, **kw: Any) -> Path:
    return ensure_model(ALIGNER_REPO, progress, **kw)


def ensure_tts_models(base_repo: str, progress: ProgressCallback = noop_progress, **kw: Any) -> Dict[str, Path]:
    """База TTS для обучения. Аудио-токенайзер (speech_tokenizer/) лежит внутри репозитория Base-модели
    (проверено по списку файлов HF), отдельный репозиторий токенайзера не нужен."""
    return {"base": ensure_model(base_repo, progress, stage=Stage.MODEL, **kw)}
