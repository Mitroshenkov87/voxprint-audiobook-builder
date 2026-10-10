"""BUILD.json's codename must not repeat an earlier build in this repository."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import codenames

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, *, name: str, changelog: str, notes: dict[str, str] | None = None) -> None:
    (root / "BUILD.json").write_text(
        json.dumps({"offset": 1, "codename": name, "offset_note": "codename is Alpha"}),
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    for filename, body in (notes or {}).items():
        (docs / filename).write_text(body, encoding="utf-8")


def test_current_repository_codename_is_allowed_and_build_json_is_not_rewritten():
    raw = (ROOT / "BUILD.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert codenames.problems(ROOT) == []
    assert hashlib.sha256((ROOT / "BUILD.json").read_bytes()).hexdigest() == digest
    assert codenames.main() == 0


def test_newest_name_may_stay_and_an_older_name_may_not(tmp_path):
    changelog = (
        '## [0.1.0] - 2020-01-01 - build 1 "Alpha"\n'
        '## [0.2.0] - 2020-02-01 - build 2 "Beta"\n'
    )
    notes = {
        "RELEASE-NOTES-1-ALPHA.md": "codename is Alpha.\n",
        "RELEASE-NOTES-2-BETA.md": "codename is Beta.\n",
    }
    _write(tmp_path, name="Beta", changelog=changelog, notes=notes)
    assert codenames.problems(tmp_path) == []
    _write(tmp_path, name="Gamma", changelog=changelog, notes=notes)
    assert codenames.problems(tmp_path) == []
    _write(tmp_path, name="Alpha", changelog=changelog, notes=notes)
    found = codenames.problems(tmp_path)
    assert found and "Alpha" in found[0]
    with pytest.raises(codenames.CodenameError):
        codenames.assert_unique(tmp_path)


def test_two_builds_cannot_share_a_name_and_one_build_cannot_have_two_spellings(tmp_path):
    _write(
        tmp_path, name="Gamma",
        changelog=(
            '## [0.1.0] - 2020-01-01 - build 1 "Alpha"\n'
            '## [0.2.0] - 2020-02-01 - build 2 "Alpha"\n'
        ),
    )
    shared = codenames.problems(tmp_path)
    assert any("alpha" in line and "1" in line and "2" in line for line in shared)
    _write(
        tmp_path, name="Gamma",
        changelog='## [0.1.0] - 2020-01-01 - build 1 "Alpha"\n',
        notes={"RELEASE-NOTES-3-GAMMA.md": "codename is Delta.\n"},
    )
    spellings = codenames.problems(tmp_path)
    assert any("build 3" in line for line in spellings)
