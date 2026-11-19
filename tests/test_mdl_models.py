"""infra/mdl_models.py: the app's one door to the shared VOXPrint MDL store (offline, fake runner, tmp store)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from voxprint_common.mdl.catalog import load_manifest
from voxprint_common.paths import models_dir

from core.errors import ModelDownloadError
from infra import mdl_models
from infra.mdl_gate import APP_ID, manifest_path
from tests.mdl_harness import build_store, digest12, fake_runner, row_maps

#: repo id -> manifest row id, derived from the manifest paths (every Org--Name row of voxprint-app-models.json).
EXPECTED_REPO_MAP = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "qwen3-tts",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "qwen3-tts",
    "Qwen/Qwen3-ForcedAligner-0.6B": "qwen3-forcedaligner",
    "Qwen/Qwen3-ASR-1.7B": "qwen3-asr-1.7b",
    "Qwen/Qwen3-ASR-0.6B": "qwen3-asr-0.6b",
    "ai-forever/sage-fredt5-distilled-95m": "sage-fredt5",
    "myshell-ai/OpenVoiceV2": "openvoice-v2",
    "Helsinki-NLP/opus-mt-en-ru": "opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-ru-en": "opus-mt-ru-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw": "opus-mt-tc-bible-big-deu-eng-fra-por-spa-gmw",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa": "opus-mt-tc-bible-big-gmw-deu-eng-fra-por-spa",
    "Helsinki-NLP/opus-mt-tc-big-en-zle": "opus-mt-tc-big-en-zle",
    "Helsinki-NLP/opus-mt-tc-big-zle-en": "opus-mt-tc-big-zle-en",
    "Helsinki-NLP/opus-mt-tc-big-zle-de": "opus-mt-tc-big-zle-de",
    "Helsinki-NLP/opus-mt-tc-big-de-zle": "opus-mt-tc-big-de-zle",
}


def _all_rows():
    return load_manifest(manifest_path()).models


def test_every_org_name_row_is_derived_from_the_manifest():
    """Every row whose files live under one ``Org--Name`` folder maps back to that repo; the expected table covers them all."""
    covered = set()
    for row in _all_rows():
        tops = {f.path.split("/", 1)[0] for f in row.files}
        if len(tops) != 1:
            continue
        top = tops.pop()
        if "--" not in top:
            continue                      # app folders (llm/, dnsmos/, ace-step-1.5/, deepfilternet/, voices/)
        repo = top.replace("--", "/")
        assert mdl_models.model_id_for_repo(repo) == EXPECTED_REPO_MAP[repo]
        covered.add(repo)
    assert covered == set(EXPECTED_REPO_MAP)


def test_model_id_for_repo_rejects_unknown_repos():
    with pytest.raises(ModelDownloadError):
        mdl_models.model_id_for_repo("Owner/NotInTheManifest")


def test_opt_in_creates_the_app_views_folder():
    assert not (models_dir() / "views" / APP_ID).exists()
    mdl_models.opt_in()
    assert (models_dir() / "views" / APP_ID).is_dir()


def test_view_dir_is_views_app_id_digest12():
    files, _sizes = row_maps("qwen3-forcedaligner")
    expected = models_dir() / "views" / APP_ID / f"qwen3-forcedaligner-{digest12(files)}"
    assert mdl_models.view_dir("qwen3-forcedaligner") == expected


def test_files_of_picks_the_complete_view_across_tiers():
    """qwen3-tts exists as 8gb and 16gb rows: the row whose view is complete wins, whatever the card."""
    files_8g, _ = row_maps("qwen3-tts")          # first row = 8gb
    view_8g = build_store("qwen3-tts")
    assert mdl_models.files_of("qwen3-tts") == files_8g
    assert mdl_models.is_ready("qwen3-tts") and mdl_models.state("qwen3-tts") == "installed"
    # a stub (empty folder) counts as missing, not ready:
    shutil.rmtree(view_8g)
    view_8g.mkdir(parents=True)
    assert not mdl_models.is_ready("qwen3-tts") and mdl_models.state("qwen3-tts") == "missing"


def test_ensure_runs_mdl_with_app_manifest_and_model_and_returns_the_view(monkeypatch):
    calls = fake_runner(monkeypatch, mdl_models, ["qwen3-forcedaligner"])
    progress = []
    view = mdl_models.ensure("qwen3-forcedaligner", progress=lambda f, t: progress.append((f, t)))
    assert view.is_dir() and view.parent == models_dir() / "views" / APP_ID
    assert len(calls) == 1
    argv = calls[0]
    assert "--run-mdl" in argv and "ensure" in argv
    assert argv[argv.index("--app-manifest") + 1] == str(manifest_path())
    assert argv[argv.index("--model") + 1] == "qwen3-forcedaligner"
    assert argv[-1] == "--json"
    assert mdl_models.ensure("qwen3-forcedaligner") == view       # ready now: the runner is not called again
    assert len(calls) == 1


def test_ensure_missing_raises_model_download_error(monkeypatch):
    calls = fake_runner(monkeypatch, mdl_models, [], code=2)      # MDL could not fetch the model
    with pytest.raises(ModelDownloadError) as err:
        mdl_models.ensure("qwen3-forcedaligner")
    assert "Manage models" in str(err.value)
    assert len(calls) == 1


def test_ensure_repairs_once_when_the_view_stays_missing(monkeypatch):
    calls = fake_runner(monkeypatch, mdl_models, [], code=0)      # exit 0 but no view: repair-store runs once, then the error
    repairs = []
    monkeypatch.setattr(mdl_models._repair_store, "repair_store",
                        lambda *a, **k: repairs.append((a, k)) or type("R", (), {"ready": False})())
    with pytest.raises(ModelDownloadError):
        mdl_models.ensure("qwen3-forcedaligner")
    assert len(calls) == 1 and len(repairs) == 1


def test_skip_suite_never_spawns_mdl(monkeypatch):
    """VOXPRINT_SKIP_SUITE=1 (conftest) refuses to start the real MDL process."""
    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "1")
    import subprocess

    def boom(*a, **k):
        raise AssertionError("subprocess started despite VOXPRINT_SKIP_SUITE")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(ModelDownloadError):
        mdl_models.ensure("qwen3-forcedaligner")


def test_default_runner_feeds_json_progress(monkeypatch):
    """The default runner parses ``--json`` progress lines into ``progress(fraction, text)``."""
    monkeypatch.setenv("VOXPRINT_SKIP_SUITE", "0")
    seen = []

    class Proc:
        def __init__(self, *a, **k):
            pass

        stdout = [json.dumps({"type": "progress", "stage": "download", "fraction": 0.5, "message": "half way"}) + "\n",
                  "not json\n",
                  json.dumps({"type": "result", "ok": True, "code": 0}) + "\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(mdl_models.subprocess, "Popen", Proc)
    assert mdl_models._subprocess_runner(["voxprint-mdl"], lambda f, t: seen.append((f, t))) == 0
    assert seen == [(0.5, "half way")]


def test_path_for_repo_returns_the_org_name_folder_inside_the_view(monkeypatch):
    fake_runner(monkeypatch, mdl_models, ["qwen3-forcedaligner"])
    path = mdl_models.path_for_repo("Qwen/Qwen3-ForcedAligner-0.6B")
    assert path.is_dir() and path.name == "Qwen--Qwen3-ForcedAligner-0.6B"
    assert path.parent == mdl_models.view_dir("qwen3-forcedaligner")


def test_derived_dir_uses_the_sha12_of_the_main_file():
    files, sizes = row_maps("gemma-4-12b")
    main_sha = max(files, key=lambda p: sizes[p])
    expected = models_dir() / "derived" / APP_ID / "gemma-4-12b" / files[main_sha][:12]
    assert mdl_models.derived_dir("gemma-4-12b") == expected


def test_denoise_tool_copies_and_chmods_into_the_derived_dir(monkeypatch):
    """DeepFilterNet: the executable is copied from the read-only view into the derived dir and chmod +x there."""
    build_store("deepfilternet")
    from infra import denoise_tool

    dest = denoise_tool.ensure(key="linux-x64")
    name = "deep-filter-0.5.6-x86_64-unknown-linux-musl"
    assert dest == mdl_models.derived_dir("deepfilternet") / name
    assert dest.is_file() and os.access(dest, os.X_OK)
    view_entry = mdl_models.view_dir("deepfilternet") / "deepfilternet" / name
    assert view_entry.is_file()
    assert os.stat(dest).st_ino != os.stat(view_entry).st_ino      # a copy, not a link into the store
    assert denoise_tool.ready(key="linux-x64") == dest           # a second call reuses the copy


def test_no_private_model_downloads_anywhere():
    """No module in infra/ core/ workers/ ui/ fetches models itself (grep over the source)."""
    root = Path(__file__).resolve().parents[1]
    bad = []
    for folder in ("infra", "core", "workers", "ui"):
        for f in (root / folder).glob("*.py"):
            text = f.read_text(encoding="utf-8")
            for token in ("snapshot_download(", "hf_hub_download(", "urlretrieve(", "huggingface.co/"):
                if token in text:
                    bad.append(f"{f.relative_to(root)}: {token}")
    assert not bad, bad
