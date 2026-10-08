"""Full / Quick setup (installer page "Setup type", infra/setup_mode.py) and the small optional models in the standard download.
Fakes only: nothing is downloaded."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from core import i18n  # noqa: E402
from infra import modules as mods  # noqa: E402
from infra import setup_mode  # noqa: E402
from tests.test_components_flow import FakeModelsWorker  # noqa: E402
from tests.test_studio import app, wait_for  # noqa: E402,F401
from workers import pipeline_runner as pr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_setup_mode_file(tmp_path):
    assert setup_mode.mode(tmp_path) == "" and not setup_mode.everything(tmp_path) and setup_mode.auto_download(tmp_path)
    (tmp_path / "install_mode.txt").write_text("\ufeffQuick\r\n", encoding="utf-8")     # what Inno's UTF-8 writer may leave
    assert setup_mode.mode(tmp_path) == "quick" and setup_mode.everything(tmp_path) and not setup_mode.auto_download(tmp_path)
    setup_mode.set_mode("full", tmp_path)
    assert setup_mode.mode(tmp_path) == "full" and setup_mode.everything(tmp_path) and setup_mode.auto_download(tmp_path)
    setup_mode.set_mode("bogus", tmp_path)
    assert setup_mode.mode(tmp_path) == ""


def test_installer_offers_exactly_full_and_quick_with_the_pinned_total():
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    sizes = setup_mode.full_sizes()
    models_mb = int(re.search(r'#define FullModelsMB "(\d+)"', iss).group(1))
    runtime_mb = int(re.search(r'#define FullRuntimeMB "(\d+)"', iss).group(1))
    # the wizard's "about N GB" is the honest sum of the pins (update the defines when a pinned model changes)
    assert abs(models_mb - sizes["models"] // 2 ** 20) <= 50 and abs(runtime_mb - sizes["runtime"] // 2 ** 20) <= 50
    assert 20 * 2 ** 30 < sizes["total"] < 40 * 2 ** 30
    for lang in ("english", "russian", "german"):
        assert f"{lang}.ModeFull=" in iss and f"{lang}.ModeQuick=" in iss and f"{lang}.ModeNoSpace=" in iss
    assert "ModePage.Add(" in iss and iss.count("ModePage.Add(") == 2                    # exactly two choices
    assert "{param:Mode|full}" in iss and "install_mode.txt" in iss and "SpaceProblem(" in iss


def test_standard_download_includes_the_small_optional_models_but_not_gemma(monkeypatch):
    calls = []
    fake_ensure = lambda repo, cb: calls.append(repo)      # noqa: E731
    monkeypatch.setattr(pr.md, "ensure_model", fake_ensure)
    monkeypatch.setattr(pr, "required_model_repos", lambda: ["tts", "aligner", "asr"])
    monkeypatch.setattr(pr, "models_missing", lambda repos: list(repos))
    monkeypatch.setattr(pr, "_restore_backup_source", lambda p: None)
    monkeypatch.setattr(pr, "_prefetch_dnsmos", lambda p: calls.append("dnsmos"))
    monkeypatch.setattr(pr, "_prefetch_text_extras", lambda p: calls.append("translators"))
    monkeypatch.setattr(pr.sys, "platform", "linux")
    from infra import denoise_tool, llm_tool, model_mirrors, vc_model

    monkeypatch.setattr(vc_model, "ready", lambda: False)
    monkeypatch.setattr(vc_model, "ensure", lambda progress: calls.append("openvoice"))
    monkeypatch.setattr(denoise_tool, "ready", lambda: None)
    monkeypatch.setattr(denoise_tool, "ensure", lambda progress: calls.append("deepfilternet"))
    monkeypatch.setattr(llm_tool, "platform_key", lambda: "win-x64")
    monkeypatch.setattr(llm_tool, "server_exe", lambda: None)
    monkeypatch.setattr(llm_tool, "ensure", lambda progress: calls.append("gemma"))
    monkeypatch.setattr(model_mirrors, "load", lambda: {"tts": 1, "tts-small": 1, vc_model.REPO: 1})

    assert pr.prefetch_models(everything=False) == ["tts", "aligner", "asr"]
    assert calls == ["tts", "aligner", "asr", "dnsmos", "translators", "openvoice", "deepfilternet"]   # no Gemma
    calls.clear()
    # Full / Quick: every pinned model (the OpenVoice folder comes through vc_model) and the big optional Gemma too
    assert pr.prefetch_models(everything=True) == ["tts", "aligner", "asr", "tts-small"]
    assert calls[-4:] == ["translators", "openvoice", "deepfilternet", "gemma"] and vc_model.REPO not in calls


def _dialog(monkeypatch, autostart):
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    m = mods.Module("torch", "PyTorch", True, 3 << 30, 4 << 30, ["a"], True)
    monkeypatch.setattr(mods, "modules", lambda man: [m])
    workers = []

    def models_start(hook):
        w = FakeModelsWorker()
        hook(w)
        workers.append(w)
        return w

    d = ModulesDialog(manifest_fn=lambda: {"modules": []}, install_fn=lambda *a: 0, models_start=models_start,
                      autostart=autostart, offer_size=28 << 30)
    d.refresh()
    return d, workers


def test_quick_setup_offers_everything_and_waits_for_the_click(app, monkeypatch):   # noqa: F811
    d, workers = _dialog(monkeypatch, autostart=False)
    assert wait_for(lambda: "Quick setup" in d.lbl_status.text(), 10), d.lbl_status.text()
    assert "28.0 GB" in d.lbl_status.text() and d.btn_download.isEnabled() and not workers   # offered, nothing started
    d.btn_download.click()
    assert wait_for(lambda: workers, 10)                                                    # the click starts the models
    workers[0].done.emit([])
    assert d.lbl_status.text().startswith("Everything is ready")
    d.shutdown()


def test_full_setup_starts_without_a_click(app, monkeypatch):   # noqa: F811
    d, workers = _dialog(monkeypatch, autostart=True)
    assert wait_for(lambda: workers, 10) and "Quick setup" not in d.lbl_status.text()
    d.shutdown()


def test_main_quick_setup_opens_the_components_window_as_an_offer(app, tmp_path, monkeypatch):   # noqa: F811
    import main
    from core.voice_library import VoiceLibrary
    from tests.test_studio import make_studio
    from ui import modules_dialog

    monkeypatch.setattr(modules_dialog.ModulesDialog, "refresh", lambda self: None)
    monkeypatch.setattr(modules_dialog.ModulesDialog, "show", lambda self: None)
    s = make_studio(VoiceLibrary(tmp_path / "voices"))
    try:
        main._offer_components(s, app, True, autostart=False)
        dlg = s._components
        assert not dlg.autostart and dlg.models_start is not None
        assert dlg.offer_size == setup_mode.full_sizes()["total"]
    finally:
        s.shutdown()
