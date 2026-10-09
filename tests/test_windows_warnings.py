"""The install pages tell Windows users to check the hash, and to keep the file in Edge before SmartScreen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    ROOT / "README.md",
    ROOT / "docs" / "RELEASE-NOTES-TEMPLATE.md",
    ROOT / "docs" / "DOWNLOADS.md",
    ROOT / "docs" / "FAQ.md",
)


def test_windows_warnings_check_the_hash_and_put_edge_before_smartscreen():
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        assert "Get-FileHash" in text or "SHA-256" in text
        assert "Keep anyway" in text
        assert "SignPath Foundation" in text
        assert "Windows protected your PC" in text
        assert text.index("Edge") < text.index("SmartScreen")
        assert "Delete" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    template = (ROOT / "docs" / "RELEASE-NOTES-TEMPLATE.md").read_text(encoding="utf-8")
    assert "### Windows security warnings" in readme
    assert "## Windows security warnings" in template
    assert "github.com/Mitroshenkov87/voxprint-audiobook-builder/releases" in readme
