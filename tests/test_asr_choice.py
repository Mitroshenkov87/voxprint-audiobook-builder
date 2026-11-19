"""Speech recognition model choice (infra/asr_choice.py): Qwen3-ASR-1.7B from ~8 GB of VRAM, else 0.6B; Settings override;
download-all only; the MDL manifest row pins 1.7B."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra import asr_choice as ac
from infra import model_downloader as md
from infra.vram_optimizer import GpuInfo

GPU8 = GpuInfo(True, "RTX 4060 Laptop", 7.996, 7.5)
GPU6 = GpuInfo(True, "RTX 3060 Laptop", 6.0, 5.5)
NOGPU = GpuInfo(False)


def test_preference_is_saved_and_unknown_values_mean_auto(tmp_path):
    from infra import paths

    assert ac.preference() == ac.AUTO
    ac.set_preference(ac.USE_LARGE)
    assert ac.preference() == ac.USE_LARGE
    (paths.state_dir() / ac.SETTINGS_FILE).write_text(json.dumps({"model": "huge"}), encoding="utf-8")
    assert ac.preference() == ac.AUTO
    with pytest.raises(ValueError):
        ac.set_preference("huge")


def test_auto_choice_by_vram_and_overrides():
    assert ac.auto_repo(GPU8) == ac.LARGE                 # an "8 GB" card reports slightly under 8 GiB
    assert ac.auto_repo(GPU6) == ac.SMALL and ac.auto_repo(NOGPU) == ac.SMALL
    assert ac.preferred_repo(ac.AUTO, GPU8) == ac.LARGE
    assert ac.preferred_repo(ac.AUTO, GPU8, force_cpu=True) == ac.SMALL      # "CPU only": the light model ...
    assert ac.preferred_repo(ac.USE_LARGE, NOGPU, force_cpu=True) == ac.LARGE   # ... unless 1.7B was chosen explicitly
    assert ac.preferred_repo(ac.USE_SMALL, GPU8) == ac.SMALL
    assert ac.download_repos(ac.AUTO, GPU8) == [ac.LARGE]
    assert ac.download_repos(ac.AUTO, GPU6) == [ac.SMALL]
    assert ac.download_repos(ac.BOTH, GPU6) == [ac.SMALL, ac.LARGE]
    assert ac.download_repos(ac.BOTH, GPU8) == [ac.LARGE, ac.SMALL]


def test_ready_prefers_the_choice_and_falls_back_to_the_installed_variant_without_downloading():
    installed = {ac.SMALL: Path("/m/small")}
    calls = []

    def ready(repo):
        calls.append(repo)
        return installed.get(repo)

    assert ac.ready(ready, ac.AUTO, GPU8) == (ac.SMALL, Path("/m/small"))   # 1.7B not there yet: the 0.6B keeps working
    assert calls == [ac.LARGE, ac.SMALL]
    installed[ac.LARGE] = Path("/m/large")
    assert ac.ready(ready, ac.AUTO, GPU8) == (ac.LARGE, Path("/m/large"))
    assert ac.ready(lambda r: None, ac.AUTO, GPU8) is None


def test_download_all_requests_the_variant_for_this_pc(monkeypatch):
    from infra import vram_optimizer
    from workers import pipeline_runner as pr

    monkeypatch.setattr(vram_optimizer, "detect_gpu", lambda: GpuInfo(True, "x", 16.0, 15.0))
    repos = pr.required_model_repos()
    assert ac.LARGE in repos and ac.SMALL not in repos
    monkeypatch.setattr(vram_optimizer, "detect_gpu", lambda: GPU6)
    repos = pr.required_model_repos()
    assert ac.SMALL in repos and ac.LARGE not in repos
    ac.set_preference(ac.BOTH)
    assert {ac.SMALL, ac.LARGE} <= set(pr.required_model_repos())


def test_voice_check_uses_the_installed_variant(monkeypatch):
    from infra import vram_optimizer
    from workers import pipeline_runner as pr

    monkeypatch.setattr(vram_optimizer, "detect_gpu", lambda: GPU8)
    monkeypatch.setattr(md, "ready_model_path", lambda repo: Path("/m/small") if repo == ac.SMALL else None)
    made = []
    import core.asr as asr_mod
    monkeypatch.setattr(asr_mod, "make_default_asr", lambda path, device="auto": made.append((path, device)) or "ASR")
    req = pr.TaskRequest(kind=pr.KIND_LORA)
    assert pr._asr_for_check(req, None) == "ASR" and made == [(str(Path("/m/small")), "auto")]
    monkeypatch.setattr(md, "ready_model_path", lambda repo: None)
    assert pr._asr_for_check(req, None) is None                 # never a surprise download


def test_bundled_manifest_pins_qwen3_asr_1_7b_consistently():
    """The app's MDL manifest row ``qwen3-asr-1.7b`` pins the same revision the ASR choice expects."""
    from voxprint_common.mdl.catalog import load_manifest

    from infra.mdl_gate import manifest_path

    row = next(m for m in load_manifest(manifest_path()).models if m.id == "qwen3-asr-1.7b")
    weights = [f for f in row.files if f.path.endswith(".safetensors")]
    assert len(weights) == 2 and all(len(f.sha256) == 64 for f in weights)
    assert sum(f.size for f in row.files) == pytest.approx(4.70e9, rel=0.01)
    assert md.APPROX_SIZE_GB[ac.LARGE] == 4.7


def test_settings_combo_saves_the_choice_and_starts_the_download_all_step(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from core.i18n import tr
    from core.voice_library import VoiceLibrary
    from tests.test_studio import make_studio

    QApplication.instance() or QApplication([])
    s = make_studio(VoiceLibrary(tmp_path / "voices"))
    try:
        d = s.settings_dialog()
        started = []
        d._win.start_prefetch = lambda *a, **k: started.append(1)
        d.asr_resolve = lambda choice: ac.LARGE if choice in (ac.AUTO, ac.USE_LARGE, ac.BOTH) else ac.SMALL
        d.asr_missing = lambda: []
        d.refresh_asr()
        assert d.cmb_asr.currentData() == ac.AUTO and d.cmb_asr.count() == 4
        assert d.lbl_asr_note.text() == tr("asrmodel.note_ready", model="Qwen3-ASR-1.7B")
        d.cmb_asr.setCurrentIndex(d.cmb_asr.findData(ac.USE_SMALL))
        assert ac.preference() == ac.USE_SMALL and started == []     # installed: nothing to download
        d.asr_missing = lambda: [ac.LARGE]
        d.cmb_asr.setCurrentIndex(d.cmb_asr.findData(ac.USE_LARGE))
        assert ac.preference() == ac.USE_LARGE and started == [1]
        assert d.lbl_asr_note.text() == tr("asrmodel.note_missing", model="Qwen3-ASR-1.7B")
    finally:
        s.shutdown()
