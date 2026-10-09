"""The frozen exe carries its own build number, ahead of credits.json."""
from __future__ import annotations

import sys
from pathlib import Path

from core import appinfo, build_stamp
from tools import embed_build_stamp as emb

ROOT = Path(__file__).resolve().parents[1]


def test_trailer_and_module_win_over_credits(tmp_path, monkeypatch):
    exe = tmp_path / "Voxprint.exe"
    exe.write_bytes(b"MZ" + b"\x00" * 32)
    build_stamp.write_trailer(exe, 669, "StampName")
    monkeypatch.setattr(appinfo, "APP_BUILD", 668)
    monkeypatch.setattr(appinfo, "APP_CODENAME", "OldName")
    monkeypatch.setattr(appinfo, "APP_VERSION", "0.1.4")
    monkeypatch.setattr(appinfo, "APP_CHANNEL", "beta")
    monkeypatch.setattr(sys, "executable", str(exe))
    appinfo.apply_embedded_stamp()
    assert appinfo.APP_BUILD == 669 and appinfo.APP_CODENAME == "StampName"
    assert 'build 669 "StampName"' in appinfo.version_label()
    import cli as user_cli

    payload = user_cli.version_payload()
    assert payload["build"] == 669 and payload["codename"] == "StampName"
    again = exe.read_bytes()
    build_stamp.write_trailer(exe, 670, "Kolot")
    assert again != exe.read_bytes()
    assert exe.read_bytes().count(build_stamp.MARKER) == 1
    assert build_stamp.read_trailer(exe) == (670, "Kolot")

    plain = tmp_path / "python"
    plain.write_bytes(b"#!/usr/bin/python")
    monkeypatch.setattr(sys, "executable", str(plain))
    monkeypatch.setattr(appinfo, "APP_BUILD", 668)
    monkeypatch.setattr(appinfo, "APP_CODENAME", "OldName")
    monkeypatch.setattr(build_stamp, "BUILD", 669)
    monkeypatch.setattr(build_stamp, "CODENAME", "Kolot")
    appinfo.apply_embedded_stamp()
    assert appinfo.APP_BUILD == 669 and appinfo.APP_CODENAME == "Kolot"


def test_embed_tool_stamps_the_module_and_the_exe_then_restores(tmp_path, monkeypatch):
    src = tmp_path / "build_stamp.py"
    src.write_text((ROOT / "core" / "build_stamp.py").read_text(encoding="utf-8"), encoding="utf-8")
    credits = tmp_path / "credits.json"
    credits.write_text('{"app": {"version": "0.1.4"}}', encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_BUILD", "669")
    monkeypatch.setenv("VOXPRINT_CODENAME", "Kolot")
    assert emb.main(["--write-module", str(src), "--credits", str(credits)]) == 0
    text = src.read_text(encoding="utf-8")
    assert "BUILD = 669" in text and 'CODENAME = "Kolot"' in text
    exe = tmp_path / "Voxprint.exe"
    exe.write_bytes(b"MZ-bootloader")
    assert emb.main(["--exe", str(exe), "--credits", str(credits)]) == 0
    assert build_stamp.read_trailer(exe) == (669, "Kolot")
    original = (ROOT / "core" / "build_stamp.py").read_text(encoding="utf-8")
    assert emb.main(["--restore-module", str(src)]) == 0
    assert src.read_text(encoding="utf-8") == original
    assert (ROOT / "core" / "build_stamp.py").read_text(encoding="utf-8") == original
    monkeypatch.delenv("VOXPRINT_BUILD")
    monkeypatch.delenv("VOXPRINT_CODENAME")
    before = exe.read_bytes()
    assert emb.main(["--exe", str(exe), "--credits", str(credits)]) == 0
    assert exe.read_bytes() == before
