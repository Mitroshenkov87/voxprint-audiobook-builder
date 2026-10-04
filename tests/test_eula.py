"""End User Agreements (docs/legal): the installer text is in sync with the Markdown, and the key clauses are present."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGAL = ROOT / "docs" / "legal"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_eula_txt", ROOT / "tools" / "gen_eula_txt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_installer_text_is_in_sync_with_the_markdown():
    gen = _gen()
    expected = gen.to_text((LEGAL / "EULA-audiobook-builder.md").read_text(encoding="utf-8"))
    assert (LEGAL / "EULA-audiobook-builder.txt").read_bytes().decode("utf-8") == expected, "run: python tools/gen_eula_txt.py"
    assert "**" not in expected and "`" not in expected
    assert expected.isascii()                                   # the installer reads a file without BOM as ANSI


def test_audiobook_builder_agreement_covers_the_required_points():
    t = (LEGAL / "EULA-audiobook-builder.md").read_text(encoding="utf-8").lower()
    for needle in ("apache license 2.0", "personal", "business", "solely responsible", "explicit consent of the voice owner", "third-party",
                   "without warranty", "beta", "no telemetry", "not legal advice", "not a lawyer", "offline"):
        assert needle in t, needle


def test_template_and_movie_dubber_draft():
    tpl = (LEGAL / "EULA-TEMPLATE.md").read_text(encoding="utf-8")
    assert "{{PRODUCT NAME}}" in tpl and "not a lawyer" in tpl
    d = (LEGAL / "EULA-movie-dubber-DRAFT.md").read_text(encoding="utf-8")
    assert "DRAFT" in d and "OpenSubtitles" in d and "singing" in d and "solely responsible for having all rights" in d and "not a lawyer" in d


def test_installer_shows_the_agreement():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8")
    assert "LicenseFile=..\\docs\\legal\\EULA-audiobook-builder.txt" in iss
