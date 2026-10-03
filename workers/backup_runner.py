"""Qt-free glue for the backup / restore buttons of the Settings dialog: which models belong to a backup, and the run functions.

The UI worker calls :func:`run_backup_job` / :func:`run_restore_job`; tests inject fakes or call :mod:`infra.backup` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from core.events import CancelToken
from infra import backup, existing_models, model_downloader as md, text_models, vram_optimizer


def backup_repos() -> List[str]:
    """Model repositories the app uses (looked up in Voxprint's folder and in other programs' caches): aligner, both TTS
    bases and the integrated text clean-up models.  Whatever is not present anywhere is simply not part of the backup."""
    repos = [md.ALIGNER_REPO, md.ASR_REPO, vram_optimizer.MODEL_1_7B, vram_optimizer.MODEL_0_6B]
    repos += [m.repo for m in text_models.REGISTRY if m.integrated and m.repo]
    return repos


def collect(include_voices: bool, **kw) -> List[backup.Item]:
    """The items of a backup (see :func:`infra.backup.collect_items`)."""
    return backup.collect_items(include_voices=include_voices, repos=backup_repos(), **kw)


def run_backup_job(target: Path, include_voices: bool, progress: Callable[[float, str], None], cancel: CancelToken,
                   items: Optional[List[backup.Item]] = None, **kw) -> backup.Report:
    """Collect the items and copy them to ``target``."""
    items = items if items is not None else collect(include_voices)
    return backup.run_backup(items, target, progress, cancel, **kw)


def run_restore_job(source: Path, include_voices: bool, progress: Callable[[float, str], None], cancel: CancelToken,
                    **kw) -> backup.Report:
    """Restore the backup in ``source`` into the app folders."""
    return backup.run_restore(source, include_voices, progress, cancel, **kw)


def run_import_job(progress: Callable[[float, str], None], cancel: CancelToken, **kw) -> backup.Report:
    """Import the models of the "existing models folder" that Voxprint does not have yet."""
    return existing_models.import_available(backup_repos(), progress, cancel, **kw)
