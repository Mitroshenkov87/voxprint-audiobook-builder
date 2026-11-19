"""Qt-free glue for the backup / restore buttons of the Settings dialog: the items a backup holds and the run functions.

Backups cover the voice library and the downloaded tools only.  The models live in the shared store, which belongs to
VOXPrint MDL - they are never part of an app backup (and never restored into an app folder).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from core.events import CancelToken
from infra import backup


def collect(include_voices: bool, **kw) -> List[backup.Item]:
    """The folders a backup contains: voices (optional) and tools."""
    return backup.collect_items(include_voices=include_voices, **kw)


def run_backup_job(target: Path, include_voices: bool, progress: Callable[[float, str], None], cancel: CancelToken,
                   items: Optional[List[backup.Item]] = None, **kw) -> backup.Report:
    """Copy the voices / tools into a backup folder (resumable, hash-verified)."""
    items = items if items is not None else collect(include_voices)
    return backup.run_backup(items, target, progress, cancel, **kw)


def run_restore_job(source: Path, include_voices: bool, progress: Callable[[float, str], None], cancel: CancelToken,
                    **kw) -> backup.Report:
    """Restore voices / tools from the backup in ``source`` into the normal folders."""
    return backup.run_restore(source, include_voices, progress, cancel, kinds=(backup.KIND_TOOLS, backup.KIND_VOICES), **kw)
