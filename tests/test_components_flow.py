"""First-start download flow of the thin build (THURSDAY roadmap items 1-3):

1. the Components window shows ONE live line naming what is downloaded right now (percentage, speed), that a file is being
   verified, or the error - plus an overall progress bar;
2. SAGE is downloaded in the same Components pass (no separate click in the Narrate window);
3. right after the components, ALL heavy models are downloaded automatically (no "Maximum quality" button)."""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402

from core import i18n  # noqa: E402
from core.events import Stage  # noqa: E402
from infra import modules as mods  # noqa: E402
from infra import text_models  # noqa: E402
from tests.test_online_installer import dist, served  # noqa: E402,F401
from tests.test_studio import app, lib, make_studio, wait_for  # noqa: E402,F401
from tools import online_fetch as of  # noqa: E402


# ------------------------------------------------------------------------------------------------ the downloader (structured events)
class EventStatus:
    """A status object like the app's: it has ``event`` and so gets the structured values."""

    def __init__(self) -> None:
        self.events, self.texts = [], []

    def write(self, state, fraction, text, force=False):
        self.texts.append(text)

    def event(self, fraction, phase, comp_id, name, done, total, speed):
        self.events.append((phase, comp_id, name, done, total))


class TextStatus:
    """A status object like the installer's status file: plain English text only."""

    def __init__(self) -> None:
        self.texts = []

    def write(self, state, fraction, text, force=False):
        self.texts.append((text, force))


def test_fetch_reports_download_then_verify_then_unpack(tmp_path, served):          # noqa: F811
    srv, mp = served
    man = json.loads(mp.read_text(encoding="utf-8"))
    st = EventStatus()
    of.run(str(mp), tmp_path / "app", tmp_path / "cache", st)
    for c in man["components"]:
        phases = [e[0] for e in st.events if e[1] == c["id"]]
        assert phases and phases[0] == "download" and "verify" in phases and phases[-1] == "unpack"
        assert phases.index("verify") > max(i for i, p in enumerate(phases) if p == "download")    # verify AFTER the download
        assert {e[2] for e in st.events if e[1] == c["id"]} == {c["file"]}
        last_dl = [e for e in st.events if e[1] == c["id"] and e[0] == "download"][-1]
        assert last_dl[3] == last_dl[4] == c["size"]


def test_status_file_text_names_the_file_and_says_verifying(tmp_path, served):      # noqa: F811
    srv, mp = served
    st = TextStatus()
    of.run(str(mp), tmp_path / "app", tmp_path / "cache", st)
    texts = [t for t, _ in st.texts]
    assert any(re.match(r"Downloading \S*payload-01\S*: \d+% - \d+ of \d+ MB - ", t) for t in texts)
    verify = [(t, f) for t, f in st.texts if t.startswith("Verifying ")]
    assert verify and all(f for _, f in verify)            # forced: never swallowed by the 0.2 s throttle of the status file
    assert all(t.isascii() for t in texts)                 # the status file is written as ASCII


def test_rate_is_smoothed_and_restarts_on_a_new_file():
    now = [0.0]
    r = of._Rate(lambda: now[0])
    assert r.update(0) == 0.0
    now[0] = 1.0
    assert r.update(10 << 20) == pytest.approx(10 << 20)
    now[0] = 2.0
    s = r.update(10 << 20)                                  # a stalled second pulls the average down, not to zero
    assert 0 < s < (10 << 20)
    assert of._mb_s(35 * (1 << 20)) == "35.0 MB/s" and of._mb_s(0) == "..."


# ------------------------------------------------------------------------------------------------ the translated live line
def test_live_line_in_three_languages():
    try:
        i18n.set_language("en")
        assert mods.live_line("download", "PyTorch (torch.whl)", 42, 100, 35 * 1024 ** 2) == \
            "Downloading: PyTorch (torch.whl) — 42% · 35 MB/s"
        assert mods.live_line("verify", "x.whl").startswith("Verifying: x.whl")
        assert mods.live_line("unpack", "x.whl", 1, 2) == "Unpacking: x.whl — 50%"
        assert "\u2026" in mods.live_line("download", "x", 0, 0, 0)          # speed not known yet
        i18n.set_language("ru")
        assert mods.live_line("download", "x", 1, 2, 2 ** 20).startswith("Скачивается: x — 50%")
        i18n.set_language("de")
        assert mods.live_line("verify", "x").startswith("Wird geprüft: x")
    finally:
        i18n.set_language("en")


def test_status_event_names_the_module_and_throttles(monkeypatch):
    i18n.set_language("en")
    got = []
    st = mods._Status(lambda f, m: got.append((f, m)), lambda: False, {"torch-cpu": "PyTorch"})
    st.event(0.1, "download", "torch-cpu", "torch-2.11.0-cp311-win_amd64.whl", 10, 100, 0.0)
    st.event(0.2, "download", "torch-cpu", "torch-2.11.0-cp311-win_amd64.whl", 20, 100, 0.0)    # within MIN_GAP: dropped
    st.event(0.5, "download", "torch-cpu", "torch-2.11.0-cp311-win_amd64.whl", 100, 100, 0.0)   # the last chunk always shows
    st.event(0.5, "verify", "torch-cpu", "torch-2.11.0-cp311-win_amd64.whl", 0, 0, 0.0)         # a phase change always shows
    assert len(got) == 3
    assert got[0][1].startswith("Downloading: PyTorch (torch-2.11.0-cp311-win_amd64.whl) — 10%")
    assert got[2][1].startswith("Verifying: PyTorch (torch-2.11.0")
    cancelled = mods._Status(lambda f, m: None, lambda: True)
    with pytest.raises(of.FetchError):
        cancelled.event(0.1, "download", "a", "a", 1, 2, 0.0)


# ------------------------------------------------------------------------------------------------ SAGE with the components
def test_component_extras_are_sage_and_only_missing_ones_are_fetched(monkeypatch):
    assert "sage-ru" in text_models.COMPONENT_EXTRAS
    ready = set()
    monkeypatch.setattr(text_models, "state",
                        lambda m: text_models.STATE_READY if m.key in ready else text_models.STATE_NEEDS_DOWNLOAD)
    seen, fetched = [], []

    def fake_ensure(model, progress):
        fetched.append(model.key)
        progress(Stage.MODEL, 0.5, "half")
        progress(Stage.MODEL, 1.0, "done")
        ready.add(model.key)

    keys = list(text_models.COMPONENT_EXTRAS)
    assert text_models.ensure_component_extras(lambda s, f, m="": seen.append(f), ensure_fn=fake_ensure) == keys
    assert fetched == keys and seen[-1] == pytest.approx(1.0) and seen == sorted(seen)
    assert text_models.missing_component_extras() == []
    assert text_models.ensure_component_extras(ensure_fn=fake_ensure) == [] and fetched == keys


# ------------------------------------------------------------------------------------------------ the window
class FakeModelsWorker(QObject):
    """Stands in for the main window's PrefetchWorker (same signals)."""

    progress = Signal(int, str)
    done = Signal(list)
    failed = Signal(str, str)


def _module(installed=False):
    return mods.Module("torch", "PyTorch", True, 3 * 1024 ** 3, 4 * 1024 ** 3, ["a"], installed)


def test_first_start_runs_components_sage_and_models_without_a_click(app, monkeypatch):     # noqa: F811
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    m = _module()
    monkeypatch.setattr(mods, "modules", lambda man: [m])
    order, lines = [], []

    def install_fn(ids, progress, cancelled):
        order.append(("modules", ids))
        progress(0.5, "Downloading: PyTorch (torch.whl) — 50% · 35 MB/s")
        m.installed = True
        return 1

    sage = {"missing": True}

    def extras_fn(progress):
        order.append(("sage",))
        progress(0.5, "Downloading: sage-fredt5-distilled-95m — 50%")
        sage["missing"] = False

    workers = []

    def models_start(hook):
        order.append(("models",))
        w = FakeModelsWorker()
        hook(w)                          # connected BEFORE the worker would start
        workers.append(w)
        return w

    d = ModulesDialog(manifest_fn=lambda: {"modules": []}, install_fn=install_fn, extras_fn=extras_fn,
                      extras_bytes_fn=lambda: (365 << 20) if sage["missing"] else 0, models_start=models_start, autostart=True)
    d.refresh()
    assert wait_for(lambda: workers, 10)                                   # no button was pressed
    assert order == [("modules", ["torch"]), ("sage",), ("models",)]
    assert "models" in d.lbl_overall.text() and "Step 2 of 2" in d.lbl_overall.text()
    w = workers[0]
    w.progress.emit(40, "Downloading: Qwen3-TTS-12Hz-1.7B-Base — 40% · 1.8 GB of 4.5 GB · 35 MB/s · from Hugging Face")
    assert d.lbl_status.text().startswith("Downloading: Qwen3-TTS") and d.bar.value() == 400 and not d.bar.isHidden()
    assert d.btn_cancel.isHidden() and d.btn_close.isEnabled()        # the model download keeps running if closed
    w.progress.emit(99, "Verifying: Qwen3-TTS-12Hz-1.7B-Base (SHA-256 checksum)…")
    assert d.lbl_status.text().startswith("Verifying:")
    w.done.emit(["Qwen/Qwen3-TTS-12Hz-1.7B-Base"])
    assert d.lbl_status.text().startswith("Everything is ready") and d.bar.isHidden()
    assert not d.btn_download.isEnabled()
    d.shutdown()


def test_a_failed_sage_does_not_stop_the_models_and_a_model_error_can_be_retried(app, monkeypatch):   # noqa: F811
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    m = _module(installed=True)
    monkeypatch.setattr(mods, "modules", lambda man: [m])

    def extras_fn(progress):
        raise RuntimeError("hf down")

    workers = []

    def models_start(hook):
        w = FakeModelsWorker()
        hook(w)
        workers.append(w)
        return w

    d = ModulesDialog(manifest_fn=lambda: {"modules": []}, install_fn=lambda *a: 0, extras_fn=extras_fn,
                      extras_bytes_fn=lambda: 365 << 20, models_start=models_start, autostart=True)
    d.refresh()
    assert wait_for(lambda: workers, 10)                       # SAGE failed, the models still start (they list SAGE again)
    workers[0].failed.emit("No internet connection.", "")
    assert "No internet connection." in d.lbl_status.text() and d.lbl_status.property("state") == "error"
    assert d.btn_download.isEnabled() and d.btn_download.text() == "Retry"
    d.on_download_clicked()                                    # SAGE is still "missing" here, so Retry runs step 1 again ...
    assert wait_for(lambda: len(workers) == 2, 10)             # ... and then the models again
    workers[1].done.emit([])
    assert d.lbl_status.text().startswith("Everything is ready") and d.lbl_status.property("state") == ""
    d.shutdown()


def test_components_errors_are_shown_on_the_live_line(app, monkeypatch):     # noqa: F811
    from ui.modules_dialog import ModulesDialog

    i18n.set_language("en")
    d = ModulesDialog(manifest_fn=lambda: (_ for _ in ()).throw(mods.ModulesError("dns failure")))
    d.refresh()
    assert wait_for(lambda: "dns failure" in d.lbl_status.text() and not d.busy, 10)
    assert d.lbl_status.property("state") == "error" and d.btn_download.text() == "Retry" and d.btn_download.isEnabled()
    d.shutdown()


def test_main_hands_the_components_window_to_the_first_run_download(app, lib, monkeypatch):    # noqa: F811
    import main
    from ui import modules_dialog

    monkeypatch.setattr(modules_dialog.ModulesDialog, "refresh", lambda self: None)
    monkeypatch.setattr(modules_dialog.ModulesDialog, "show", lambda self: None)
    s = make_studio(lib)
    try:
        calls = []
        s.trainer.prefetch_fn = lambda progress: (calls.append(1), [])[1]
        main._offer_components(s, app, True)
        dlg = s._components
        assert dlg.autostart and dlg.extras_fn is modules_dialog.default_extras and dlg.models_start is not None
        hooked = []
        w = dlg.models_start(lambda worker: hooked.append(worker.isRunning()))
        assert w is not None and hooked == [False]               # wired before start, so no signal is lost
        assert wait_for(lambda: calls and not s.trainer.busy, 10)
    finally:
        s.shutdown()


def test_settings_has_no_maximum_quality_gate(app, lib):          # noqa: F811
    s = make_studio(lib)
    try:
        d = s.settings_dialog()
        assert not hasattr(d, "btn_max") and not hasattr(d, "start_max_quality")
    finally:
        s.shutdown()


def test_prefetch_progress_is_weighted_by_model_size(tmp_path, monkeypatch):
    from infra import model_downloader as md
    from workers import pipeline_runner as pr

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "h"))
    monkeypatch.setattr(md, "APPROX_SIZE_GB", {"S/small": 0.5, "B/big": 4.5})
    seen = []

    def ensure(repo, progress):
        progress(Stage.MODEL, 1.0, repo)

    assert pr.prefetch_models(lambda s, f, m: seen.append((m, f)), ["S/small", "B/big"], ensure=ensure) == ["S/small", "B/big"]
    assert dict(seen)["S/small"] == pytest.approx(0.1) and dict(seen)["B/big"] == pytest.approx(1.0)


def test_window_texts_in_every_language(app):          # noqa: F811
    from ui.modules_dialog import ModulesDialog

    d = ModulesDialog(manifest_fn=lambda: {"modules": []}, models_start=lambda hook: None)
    seen = set()
    try:
        for lang in i18n.LANGS:
            i18n.set_language(lang)
            d.retranslate()
            d._set_overall(0.5)
            assert "modules." not in d.lbl_hint.text() + d.lbl_overall.text() + d.btn_download.text()
            assert "50" in d.lbl_overall.text()
            seen.add(d.lbl_hint.text())
    finally:
        i18n.set_language("en")
        d.shutdown()
    assert len(seen) == len(i18n.LANGS)
