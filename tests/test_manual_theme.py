"""Manual PDF themes (docs/manual/build_manual.py): every text/background pair is readable in both palettes."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("markdown")
pytest.importorskip("weasyprint")
pytest.importorskip("PIL")

_SPEC = importlib.util.spec_from_file_location("build_manual", Path(__file__).resolve().parents[1] / "docs" / "manual" / "build_manual.py")
bm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bm)


def test_contrast_of_both_themes_is_sufficient():
    assert bm.check_contrast() == []
    assert bm.contrast("#000000", "#ffffff") == pytest.approx(21.0)


def test_dark_is_default_and_print_is_light():
    assert bm.SUFFIX == {"dark": "", "print": "-print"}
    assert bm._lum(bm.THEMES["dark"]["page_bg"]) < 0.02 and bm._lum(bm.THEMES["print"]["page_bg"]) > 0.9


def test_page_css_is_full_bleed_and_logo_is_vector():
    css = bm.page_css("dark")
    assert "background: #0d0f14" in css and "@page :first" in css and "image/svg+xml" in css
    assert bm.logo_svg("print").startswith("<svg") and "http" not in bm.logo_svg("print").replace("http://www.w3.org/2000/svg", "")
