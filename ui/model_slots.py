"""Speech-model choices for the shell, read from the app manifest.

The window does not name a model. Each choice value is ``Model.id`` and each
label is the catalog key ``model.{id}``.

    from ui.model_slots import choices
    choices()
"""
from __future__ import annotations

from voxprint_common.mdl import load_manifest
from voxprint_common.ui.settings import Choice

from infra.mdl_gate import manifest_path


def choices() -> tuple[Choice, ...]:
    """Return one choice per available model id, in manifest order.

    A repeated id is kept once. A model marked unavailable is left out.
    A missing manifest returns no choices.
    """
    path = manifest_path()
    if not path.is_file():
        return ()
    seen: set[str] = set()
    found: list[Choice] = []
    for model in load_manifest(path).models:
        if model.unavailable or model.id in seen:
            continue
        seen.add(model.id)
        found.append(Choice(model.id, f"model.{model.id}"))
    return tuple(found)
