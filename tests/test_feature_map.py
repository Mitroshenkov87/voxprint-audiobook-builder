"""The migration map: one row per old-UI feature and per CLI feature, nothing dropped."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs" / "ui" / "FEATURE-MAP.md"
HEADER = "| Feature | Where it was | Where it lives now | Status |"
ALLOWED = {"mapped", "CLI-only", "dropped"}
CLI = (
    "narrate", "train", "voices list", "voices export", "voices catalog", "voices download",
    "diag", "status", "capabilities", "backup", "restore", "models list", "models download",
    "revoice", "speakers", "check", "repair", "prepare", "translate", "settings list",
    "settings get", "settings set", "bench", "--json", "--yes", "--dry-run", "--version",
)


def _rows(text: str) -> list[list[str]]:
    found = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Piece"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells == ["Feature", "Where it was", "Where it lives now", "Status"]:
            continue
        if cells == ["Piece", "Designed use"]:
            continue
        if len(cells) == 4:
            found.append(cells)
    return found


def test_every_row_is_mapped_or_cli_only_and_names_a_shell_place():
    text = MAP.read_text(encoding="utf-8")
    assert HEADER in text
    assert "f4e95ccfa25780abaf50412cb095a7843725102a" in text
    assert "Sections (a tab row)" in text and "Steps (a wizard)" in text
    rows = _rows(text)
    assert len(rows) > 80
    statuses = {row[3] for row in rows}
    assert statuses <= ALLOWED
    assert "dropped" not in statuses
    places = "\n".join(row[2] for row in rows)
    assert "Section: Narrate" in places
    assert "Section: Voices" in places
    assert "Section: Roles" in places
    assert "Section: Re-voice" in places
    assert "Step 1: The recording" in places
    assert "Step 2: How to train" in places
    assert "Step 3: Consent and save" in places
    for name in CLI:
        assert any(name in row[0] for row in rows), name
