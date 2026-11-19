"""Paint the suite action icons for toolbars, menus and buttons.

The drawings live in ``assets/icons/`` (see ``assets/icons/LICENSE.txt``). Each
SVG uses ``stroke="currentColor"``. This module swaps that colour and renders
at 1x, 2x and 3x so the glyph stays sharp on a HiDPI screen, on a dark toolbar
and on a light primary button.

Example::

    from ui.suite_icons import apply_button
    apply_button(save_button, "save", surface="primary")
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from infra.paths import resource_dir

#: Names of the shipped drawings. The string is the file name without ``.svg``.
NAMES = ("save", "save-as", "save-copy", "export", "download")
#: Ink on a dark toolbar. Same value as ``ui.main_window.TEXT``.
DARK = "#f2f2f5"
#: Ink on a light surface (a yellow primary button, or a light theme). Same as ``TEXT_ON_PRIMARY``.
LIGHT = "#1c1604"
#: Hover ink. Same as ``ACCENT_HOVER``.
HOVER = "#7db9ff"
#: Disabled ink. Same as ``TEXT_DISABLED``.
DISABLED = "#8e8e9e"
#: Selected menu row. White on the menu selection colour.
SELECTED = "#ffffff"
_LOGICAL = (16, 22)
_DPR = (1.0, 2.0, 3.0)
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_cache: dict[tuple[str, str], QIcon] = {}


def icons_dir() -> Path:
    """Directory that holds the five SVG files (the unpacked build, or the source tree)."""
    return resource_dir() / "assets" / "icons"


def svg_text(name: str) -> str:
    """Raw SVG for ``name``. Raises ``FileNotFoundError`` when the drawing is missing."""
    if name not in NAMES:
        raise FileNotFoundError(name)
    return (icons_dir() / f"{name}.svg").read_text(encoding="utf-8")


def render_pixmap(name: str, color: str, *, logical: int = 22, dpr: float = 1.0) -> QPixmap:
    """One sharp pixmap. ``logical`` is the size in CSS pixels; ``dpr`` is the device pixel ratio."""
    if _COLOR.fullmatch(color) is None:
        raise ValueError(color)
    svg = svg_text(name).replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    side = max(1, int(round(logical * dpr)))
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def _surface_color(surface: str) -> str:
    if surface in ("light", "primary"):
        return LIGHT
    return DARK


def make_icon(name: str, *, surface: str = "dark") -> QIcon:
    """Icon for ``surface``: ``dark`` (toolbar), ``light`` or ``primary`` (ink on a light fill).

    Normal, hover, disabled and selected pixmaps are included at 16 px and 22 px, each at
    device pixel ratios 1, 2 and 3.
    """
    key = (name, surface)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    normal = _surface_color(surface)
    icon = QIcon()
    colors = (
        (QIcon.Mode.Normal, normal),
        (QIcon.Mode.Active, HOVER),
        (QIcon.Mode.Disabled, DISABLED),
        (QIcon.Mode.Selected, SELECTED),
    )
    for logical in _LOGICAL:
        for dpr in _DPR:
            for mode, color in colors:
                icon.addPixmap(
                    render_pixmap(name, color, logical=logical, dpr=dpr),
                    mode,
                    QIcon.State.Off,
                )
    _cache[key] = icon
    return icon


def apply_button(button, name: str, *, surface: str = "dark") -> None:
    """Put ``name`` on a button. The button's text stays; the icon is 22 px."""
    button.setIcon(make_icon(name, surface=surface))
    button.setIconSize(QSize(22, 22))
