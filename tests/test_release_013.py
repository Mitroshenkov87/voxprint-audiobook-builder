"""Build 667 fixes from the clean install of build 666 (fake downloads only, no network, no models)."""
import hashlib
import io
import json
import logging
import re
import threading
import urllib.error
from pathlib import Path

import pytest

from core import i18n
from core.errors import CancelledByUser, ModelDownloadError
from infra import download_watch as dw
from infra import model_downloader as md
from infra import parallel_download as pd
from tests.test_download_watch import fast_watch  # noqa: F401  (fixture)
from tests.test_model_mirrors import FILES, REPO, SHA_A, manifest  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def _iss() -> str:
    return (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")


@pytest.fixture
def held(tmp_path):
    """An abandoned download thread of this process that still 'holds' a folder until released."""
    events = []

    def hold(folder: Path):
        ev = threading.Event()
        th = threading.Thread(target=ev.wait, daemon=True)
        th.start()
        dw._abandon(folder, th)
        events.append(ev)
        return ev

    yield hold
    for ev in events:
        ev.set()


def _partial_of(repo):
    t = md.local_dir_for(repo)
    return t.with_name(t.name + ".partial")


def _hub_leftovers(part: Path):
    d = part / ".cache" / "huggingface" / "download"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors.lock").write_bytes(b"")
    (d / "abc.etag.incomplete").write_bytes(b"x" * 10)


# ------------------------------------------------------------------------------------------------ 1. WinError 5 on rename
def test_hub_requests_always_have_connect_and_read_timeouts(monkeypatch):
    from infra import netroute as nr

    assert nr.hub_timeout(None) == (nr.HUB_CONNECT_TIMEOUT, nr.HUB_READ_TIMEOUT)
    assert nr.hub_timeout(30) == (nr.HUB_CONNECT_TIMEOUT, 30.0) and nr.hub_timeout((1, 2)) == (1, 2)
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import configure_http_backend, constants
    from huggingface_hub.utils import get_session
    from requests.adapters import HTTPAdapter

    import requests

    seen = []

    def send(self, req, stream=False, timeout=None, *a, **k):
        seen.append(timeout)
        r = requests.Response()
        r.status_code, r.url = 200, req.url
        return r

    monkeypatch.setattr(HTTPAdapter, "send", send)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    old = (constants.HF_HUB_DOWNLOAD_TIMEOUT, constants.HF_HUB_ETAG_TIMEOUT)
    try:
        nr.ensure_hub_session()
        get_session().get("https://huggingface.co/api/whoami")               # no timeout given: the adapter adds one
        assert seen == [(nr.HUB_CONNECT_TIMEOUT, nr.HUB_READ_TIMEOUT)]
        assert constants.HF_HUB_DOWNLOAD_TIMEOUT == int(nr.HUB_READ_TIMEOUT)
    finally:
        constants.HF_HUB_DOWNLOAD_TIMEOUT, constants.HF_HUB_ETAG_TIMEOUT = old
        configure_http_backend()
        nr.reset_memory()


def test_a_retry_while_the_abandoned_thread_lives_continues_in_a_fresh_folder(manifest, fast_watch, monkeypatch, held):  # noqa: F811
    monkeypatch.setattr(md, "FRESH_PARTIAL_AFTER", 0.05)
    part = _partial_of(REPO)
    part.mkdir(parents=True)
    (part / "config.json").write_bytes(FILES["config.json"])                 # finished before the stall
    (part / "model.safetensors.incomplete").write_bytes(b"W" * 10)            # the stuck thread's file
    _hub_leftovers(part)
    release = held(part)
    seen = {}

    def hub(repo_id, local_dir, **kw):
        seen["dir"] = Path(local_dir).name
        seen["had_config"] = (Path(local_dir) / "config.json").is_file()      # the finished file was taken over
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists():
                t.write_bytes(b)

    got = md.ensure_model(REPO, snapshot_download=hub, revision=SHA_A, mirror_manifest=manifest)
    assert md.verify_local_model(got) and not (got / ".cache").exists()
    assert seen == {"dir": part.name + "-1", "had_config": True}
    assert part.is_dir()                                                      # still held: left alone
    release.set()
    dw.wait_released(part, 2.0)
    md.ensure_model(REPO, snapshot_download=hub, revision=SHA_A, mirror_manifest=manifest)
    assert not part.exists()                                                  # cleaned once nobody holds it


def test_finalize_moves_finished_files_when_our_own_thread_holds_the_folder(tmp_path, monkeypatch, held):
    part, target = tmp_path / "M.partial", tmp_path / "M"
    for n, b in FILES.items():
        (part / n).parent.mkdir(parents=True, exist_ok=True)
        (part / n).write_bytes(b)
    _hub_leftovers(part)
    (part / "big.bin.incomplete.parts").mkdir()
    renames = []
    monkeypatch.setattr(Path, "rename", lambda self, t: renames.append(t) or (_ for _ in ()).throw(PermissionError(5, "x")))
    monkeypatch.setattr(md, "_rmtree", lambda p: None)                       # Windows: the held .cache cannot go
    held(part)
    md.finalize_download(part, target)
    assert renames == []                                                      # waiting would not help: no rename loop
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == set(FILES)


def test_fallback_copy_skips_the_hub_cache_and_unfinished_pieces(tmp_path, monkeypatch):
    part, target = tmp_path / "M.partial", tmp_path / "M"
    for n, b in FILES.items():
        (part / n).parent.mkdir(parents=True, exist_ok=True)
        (part / n).write_bytes(b)
    monkeypatch.setattr(md, "_sleep", lambda s: None)
    monkeypatch.setattr(md, "FINALIZE_ATTEMPTS", 2)
    monkeypatch.setattr(Path, "rename", lambda self, t: (_ for _ in ()).throw(PermissionError(5, "denied")))
    monkeypatch.setattr(md, "_move_finished", lambda *a: (_ for _ in ()).throw(PermissionError(5, "move denied")))
    real_rmtree = md._rmtree
    monkeypatch.setattr(md, "_rmtree", lambda p: None if p.name == ".cache" else real_rmtree(p))   # the .cache stays held
    _hub_leftovers(part)
    (part / "x.bin.incomplete").write_bytes(b"1")
    md.finalize_download(part, target)
    names = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    assert names == set(FILES)


def test_the_error_does_not_blame_an_antivirus_when_the_holder_is_our_own_thread(manifest, fast_watch, monkeypatch, held):  # noqa: F811
    monkeypatch.setattr(md, "_move_finished", lambda *a: (_ for _ in ()).throw(PermissionError(5, "move denied")))
    monkeypatch.setattr(md.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "copy denied")))
    part = _partial_of(REPO)
    for n, b in FILES.items():
        (part / n).parent.mkdir(parents=True, exist_ok=True)
        (part / n).write_bytes(b)
    held(part)
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model(REPO, snapshot_download=lambda **k: None, revision=SHA_A, mirror_manifest=manifest)
    assert ei.value.user_message == i18n.tr("err.model_finalize_busy", short="Tiny-1.7B")
    for lang in ("en", "de", "ru", "uk", "lv"):
        text = i18n.tr_lang("err.model_finalize_busy", lang, short="X").lower()
        assert "antivir" not in text and "virenscanner" not in text and "антивір" not in text and "антивир" not in text


def test_fresh_folders_of_an_earlier_run_are_resumed(manifest, fast_watch):  # noqa: F811
    target = md.local_dir_for(REPO)
    extra = target.with_name(target.name + ".partial-1")
    extra.mkdir(parents=True)
    (extra / "config.json").write_bytes(FILES["config.json"])
    seen = []

    def hub(repo_id, local_dir, **kw):
        seen.append((Path(local_dir).name, (Path(local_dir) / "config.json").is_file()))
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    md.ensure_model(REPO, snapshot_download=hub, revision=SHA_A, mirror_manifest=manifest)
    assert seen == [(target.name + ".partial", True)] and not extra.exists()


def test_check_and_repair_covers_translators_and_the_single_file_models(monkeypatch):
    from infra import auto_repair, model_mirrors, setup_mode, text_models

    monkeypatch.setattr("workers.pipeline_runner.required_model_repos", lambda: ["Qwen/Qwen3-ForcedAligner-0.6B"])
    repos = auto_repair.model_repos()
    for key in text_models.COMPONENT_EXTRAS:                                   # tc-big & co. are checked by hash
        assert text_models.get(key).repo in repos and auto_repair.expected_files(text_models.get(key).repo) is not None
    assert "myshell-ai/OpenVoiceV2" not in repos                              # it has its own folder: a tool
    names = {t.name for t in auto_repair.default_tools()}
    assert {"OpenVoice V2", "DNSMOS P.835"} <= names
    setup_mode.set_mode("full")
    assert set(model_mirrors.load()) - {"myshell-ai/OpenVoiceV2"} <= set(auto_repair.model_repos())


def test_check_tool_repairs_a_damaged_file_and_skips_a_missing_optional_one(tmp_path, monkeypatch):
    from core.events import CancelToken
    from infra import auto_repair, paths

    monkeypatch.setattr(paths, "models_dir", lambda: tmp_path)
    good = b"model bytes"
    f = tmp_path / "llm" / "m.gguf"
    meta = {"size": len(good), "sha256": hashlib.sha256(good).hexdigest()}
    calls = []

    def ensure(progress):
        calls.append(1)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(good)

    tool = auto_repair.Tool("Gemma", lambda: {f: meta}, ensure, wanted=False)
    assert auto_repair.check_tool(tool, lambda *a: None, CancelToken()).status == auto_repair.SKIPPED and not calls
    f.parent.mkdir(parents=True)
    f.write_bytes(b"model bytez")                                              # same size, wrong hash
    item = auto_repair.check_tool(tool, lambda *a: None, CancelToken())
    assert item.status == auto_repair.REPAIRED and calls == [1] and f.read_bytes() == good
    assert auto_repair.check_tool(tool, lambda *a: None, CancelToken()).status == auto_repair.OK
    rep = auto_repair.run(repos=[], tools=[tool], env_verify=lambda: type("R", (), {"ok": True})(),
                          modules_api=type("M", (), {"is_thin": staticmethod(lambda: False)}), ffmpeg_ensure=lambda p: None)
    assert any(i.name == "Gemma" and i.status == auto_repair.OK for i in rep.items)


def test_check_and_repair_fetches_a_translator_at_its_pinned_revision(monkeypatch):
    from core.events import CancelToken
    from infra import auto_repair, text_models

    m = text_models.get("opus-big-en-ru")
    seen = {}

    def ensure(repo, progress, **kw):
        seen.update(kw)
        raise ModelDownloadError("offline")

    auto_repair.check_and_fix_model(m.repo, lambda *a: None, CancelToken(), ensure)
    assert seen["revision"] == m.revision and seen["allow_patterns"] == list(m.files)


# ------------------------------------------------------------------------------------------------ 2. progress counted once
def test_progress_bytes_counts_every_file_once(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "a.bin.incomplete").write_bytes(b"x" * 50)                  # stale partial copy of a finished file
    (tmp_path / "b.bin.incomplete.parts").mkdir()
    (tmp_path / "b.bin.incomplete.parts" / "0000").write_bytes(b"x" * 30)   # parts already joined into b.bin.incomplete
    (tmp_path / "b.bin.incomplete").write_bytes(b"x" * 60)
    (tmp_path / "b.bin.incomplete.assembling").write_bytes(b"x" * 60)
    (tmp_path / "c.bin.incomplete.parts").mkdir()
    (tmp_path / "c.bin.incomplete.parts" / "0000").write_bytes(b"x" * 40)   # still downloading
    _hub_leftovers(tmp_path)                                                 # lock (0) + hub .incomplete (10)
    assert dw.progress_bytes(tmp_path) == 100 + 60 + 40 + 10
    assert dw.dir_bytes(tmp_path) > dw.progress_bytes(tmp_path)              # the watchdog still sees every byte


def test_the_progress_log_never_shows_more_than_the_total(tmp_path, caplog):
    meter = dw.Meter("Hugging Face")
    meter.set_total(100)

    def writes_duplicates():
        (tmp_path / "w.bin").write_bytes(b"x" * 90)
        (tmp_path / "w.bin.incomplete.parts").mkdir()
        (tmp_path / "w.bin.incomplete.parts" / "0").write_bytes(b"x" * 90)
        import time
        time.sleep(0.3)

    with caplog.at_level(logging.INFO, logger="voxprint.models"):
        dw.run_watched(writes_duplicates, tmp_path, meter, threading.Event(), poll=0.05, log_every=0.05)
    assert meter.done <= 100
    for line in re.findall(r"download via .*", caplog.text):
        assert "180" not in line


def test_the_hub_fallback_fetches_only_unfinished_files_without_our_pieces(fast_watch, monkeypatch):  # noqa: F811
    hub = pytest.importorskip("huggingface_hub")
    from infra import netroute

    repo = "Org/Fallback"
    sizes = {"config.json": len(FILES["config.json"]), "model.safetensors": len(FILES["model.safetensors"])}
    part = _partial_of(repo)
    seen = {}

    def parallel(files, dest, url_for, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)
        Path(dest, "config.json").write_bytes(FILES["config.json"])
        Path(dest, "model.safetensors.incomplete.parts").mkdir(parents=True)
        Path(dest, "model.safetensors.incomplete.parts", "0000").write_bytes(b"W" * 10)
        raise pd.ParallelError("TLS handshake timed out")

    def snapshot(repo_id, local_dir, **kw):
        seen["patterns"] = kw.get("allow_patterns")
        seen["parts_left"] = Path(local_dir, "model.safetensors.incomplete.parts").exists()
        Path(local_dir, "model.safetensors").write_bytes(FILES["model.safetensors"])

    monkeypatch.setattr(pd, "download_listed", parallel)
    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    monkeypatch.setattr(md, "_prepare_network", lambda: None)
    monkeypatch.setattr(netroute, "ensure_hub_session", lambda: None)
    got = md.ensure_model(repo, get_remote_sha=lambda r: SHA_A, get_remote_sizes=lambda r, rev, p: dict(sizes))
    assert md.verify_local_model(got) and seen == {"patterns": ["model.safetensors"], "parts_left": False}
    assert not part.exists()


# ------------------------------------------------------------------------------------------------ 3. one failed file is retried
class _Resp(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_a_failed_small_file_is_retried_before_the_model_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "_sleep", lambda s: None)
    monkeypatch.setenv(pd.ENV_CONNECTIONS, "1")                              # one request per attempt
    calls = {"a.json": 0, "b.json": 0}
    data = {"a.json": b"A" * 7, "b.json": b"B" * 5}

    def opener(req, timeout):
        name = req.full_url.rsplit("/", 1)[-1]
        calls[name] += 1
        if name == "b.json" and calls[name] <= 2:
            raise urllib.error.URLError("_ssl.c:989: The handshake operation timed out")
        return _Resp(data[name])

    fractions = []
    pd.download_listed({"a.json": 7, "b.json": 5}, tmp_path, lambda n: "https://x/" + n, opener=opener,
                       progress=fractions.append, timeout=1)
    assert (tmp_path / "b.json").read_bytes() == data["b.json"] and calls["b.json"] == 3
    assert max(fractions) <= 1.0 and fractions[-1] == 1.0


def test_a_file_that_always_fails_or_is_missing_ends_the_parallel_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "_sleep", lambda s: None)
    monkeypatch.setenv(pd.ENV_CONNECTIONS, "1")
    n = [0]

    def always(req, timeout):
        n[0] += 1
        raise urllib.error.URLError("timed out")

    with pytest.raises(pd.ParallelError):
        pd.download_listed({"a.json": 3}, tmp_path, lambda x: "https://x/" + x, opener=always, timeout=1)
    assert n[0] == pd.FILE_RETRIES + 1
    n[0] = 0

    def missing(req, timeout):
        n[0] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    with pytest.raises(pd.ParallelError):
        pd.download_listed({"a.json": 3}, tmp_path, lambda x: "https://x/" + x, opener=missing, timeout=1)
    assert n[0] == 1                                                          # a 404 is not asked again


def test_the_speed_log_counts_only_the_bytes_of_this_attempt(manifest, fast_watch, caplog):  # noqa: F811
    part = _partial_of(REPO)
    part.mkdir(parents=True)
    (part / "old-attempt.bin").write_bytes(b"o" * 3_000_000)                 # fetched by an earlier attempt

    def hub(repo_id, local_dir, **kw):
        for n, b in FILES.items():
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)

    with caplog.at_level(logging.INFO, logger="voxprint.models"):
        md.ensure_model(REPO, snapshot_download=hub, revision=SHA_A, mirror_manifest=manifest)
    line = re.search(r"from Hugging Face: (\d+) MB in .* since the download started", caplog.text)
    assert line and line.group(1) == "0"


# ------------------------------------------------------------------------------------------------ 4. translators one by one
def test_one_failed_translator_does_not_skip_the_others(monkeypatch):
    from infra import text_models

    todo = [text_models.get(k) for k in ("opus-big-en-ru", "opus-big-ru-en", "opus-big-ru-de")]
    monkeypatch.setattr(text_models, "missing_component_extras", lambda: list(todo))
    tried = []

    def ensure(m, progress):
        tried.append(m.key)
        if m.key == "opus-big-ru-en":
            raise ModelDownloadError("WinError 5")

    with pytest.raises(ModelDownloadError) as ei:
        text_models.ensure_component_extras(ensure_fn=ensure)
    assert tried == ["opus-big-en-ru", "opus-big-ru-en", "opus-big-ru-de"]
    assert ei.value.failed == ["opus-big-ru-en"] and ei.value.fetched == ["opus-big-en-ru", "opus-big-ru-de"]
    tried.clear()

    def cancel(m, progress):
        tried.append(m.key)
        raise CancelledByUser()

    with pytest.raises(CancelledByUser):
        text_models.ensure_component_extras(ensure_fn=cancel)
    assert tried == ["opus-big-en-ru"]


def test_the_complete_set_fetches_translators_only_with_their_file_patterns(monkeypatch):
    from infra import text_models
    from workers import pipeline_runner as pr

    monkeypatch.setattr(pr, "required_model_repos", lambda: ["Qwen/Qwen3-ForcedAligner-0.6B",
                                                              "ai-forever/sage-fredt5-distilled-95m"])
    repos = pr.all_model_repos()
    assert "ai-forever/sage-fredt5-distilled-95m" in repos
    assert not [r for r in repos if r.startswith("Helsinki-NLP/")]           # never as a whole repository
    assert {m.key for m in text_models.missing_other_integrated()} == {"opus-ru-en", "opus-en-ru"}


# ------------------------------------------------------------------------------------------------ 5. / 6. / 8. installer
def test_silent_uninstall_keeps_the_data_without_a_question():
    iss = _iss()
    assert "MsgBox(FmtMessage(CustomMessage('UninstallDataQuestion')" not in iss.replace("SuppressibleMsgBox", "")
    m = re.search(r"SuppressibleMsgBox\(FmtMessage\(CustomMessage\('UninstallDataQuestion'\).*?\)\s*=\s*IDYES", iss, re.S)
    assert m and "MB_DEFBUTTON2, IDNO)" in m.group(0)                        # /SUPPRESSMSGBOXES answers No: models stay


def test_the_wizard_language_becomes_the_app_language_once():
    iss = _iss()
    assert "function AppLanguageCode(): String;" in iss
    for wizard, code in (("russian", "ru"), ("german", "de")):
        assert f"ActiveLanguage = '{wizard}' then\n    Result := '{code}'" in iss
    assert "if not FileExists(StateDir + '\\language') then" in iss
    assert "SaveStringsToUTF8FileWithoutBOM(StateDir + '\\language', Lines, False);" in iss


def test_state_files_are_written_without_bom_and_read_with_or_without(monkeypatch):
    from infra import paths, setup_mode

    iss = _iss()
    assert "SaveStringsToUTF8FileWithoutBOM(StateDir + '\\install_mode.txt', Lines, False);" in iss
    assert "SaveStringsToUTF8File(StateDir + '\\install_mode.txt'" not in iss
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    (paths.state_dir() / "install_mode.txt").write_bytes(b"\xef\xbb\xbfquick\r\n")
    assert setup_mode.mode() == "quick"
    (paths.state_dir() / "language").write_bytes(b"\xef\xbb\xbfde\r\n")       # an older installer's BOM
    assert i18n.saved_language() == "de"
    (paths.state_dir() / "language").write_bytes(b"en\r\n")
    assert i18n.saved_language() == "en"


# ------------------------------------------------------------------------------------------------ 7. standard buttons
def test_standard_buttons_follow_the_ui_language():
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

    from ui import std_buttons

    app = QApplication.instance() or QApplication([])
    std_buttons.install(app)
    try:
        for lang in ("en", "de", "ru", "uk", "lv"):
            i18n.set_language(lang)
            box = QMessageBox(QMessageBox.Icon.Information, "t", "x", QMessageBox.StandardButton.Ok)
            std_buttons.localize(box)
            assert box.button(QMessageBox.StandardButton.Ok).text() == i18n.tr("qt.btn.ok")
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            std_buttons.localize(bb)
            assert bb.button(QDialogButtonBox.StandardButton.Cancel).text() == i18n.tr("qt.btn.cancel")
            assert QCoreApplication.translate("QPlatformTheme", "Cancel") == i18n.tr("qt.btn.cancel")
            assert all(i18n.tr(k) and i18n.tr(k) != k for k in std_buttons.KEYS.values())
        i18n.set_language("en")
        assert QCoreApplication.translate("QPlatformTheme", "OK") == "OK"
        assert QCoreApplication.translate("SomethingElse", "OK") == "OK"         # only the button texts are touched
    finally:
        i18n.set_language("ru")


def test_the_privacy_notice_and_other_info_boxes_use_the_localized_buttons():
    src = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert 'std_buttons.information(self, tr("ui.privacy_title"), tr("ui.privacy_text"))' in src
    for f in ("ui/main_window.py", "ui/settings_dialog.py", "ui/voices_window.py", "ui/about_dialog.py"):
        text = (ROOT / f).read_text(encoding="utf-8")
        assert "QMessageBox.information(" not in text and "QMessageBox.question(" not in text, f
    assert "std_buttons.install(app)" in (ROOT / "main.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ 9. pinned hashes
def test_every_integrated_text_model_has_pinned_sizes_and_hashes():
    from infra import model_mirrors, text_models

    entries = model_mirrors.load()
    for m in text_models.REGISTRY:
        if not (m.integrated and m.repo):
            continue
        e = entries.get(m.repo)
        assert e is not None and e.source_revision == m.revision, m.repo
        files = e.downloadable(list(m.files) or None)
        if m.files:
            assert set(files) == set(m.files) or set(files) >= {f for f in m.files if "*" not in f}, m.repo
        for name, digest in m.sha256:
            assert files[name]["sha256"] == digest, (m.repo, name)


def test_pin_tool_takes_lfs_hashes_and_hashes_small_files_without_the_weights():
    import tools.pin_hf_hashes as pin

    cfg = b'{"a": 1}'
    tree = [{"type": "file", "path": "config.json", "size": len(cfg), "oid": pin.git_blob_id(cfg)},
            {"type": "file", "path": "model.safetensors", "size": 999, "oid": "x",
             "lfs": {"oid": "f" * 64, "size": 999}}]
    fetched = []

    def fetch(url):
        fetched.append(url)
        return json.dumps(tree).encode() if "/api/" in url else cfg

    out = pin.pin("Org/M", "r" * 40, ["config.json", "model.safetensors"], fetch)
    assert out == {"config.json": {"sha256": hashlib.sha256(cfg).hexdigest(), "size": len(cfg)},
                   "model.safetensors": {"sha256": "f" * 64, "size": 999}}
    assert not [u for u in fetched if u.endswith("model.safetensors")]      # the weights are never downloaded
    with pytest.raises(SystemExit):
        pin.pin("Org/M", "r" * 40, ["config.json"], lambda u: json.dumps(tree).encode() if "/api/" in u else b"tampered")


# --------------------------------------------------------------------------- added scope: train CLI, title block, work dir

def _train(tmp_path, *extra):
    import cli as user_cli

    audio = tmp_path / "clips"
    audio.mkdir(exist_ok=True)
    seen = []

    def fake_task(req, progress, cancel=None, **_kw):
        seen.append(req)
        return type("R", (), {"voice_id": "v", "adapter_path": None, "root_dir": req.out_root, "warnings": []})()
    code = user_cli.main(["train", str(audio), *extra], run_task_fn=fake_task)
    return code, seen


def test_train_cli_license_consent_language(tmp_path):
    code, seen = _train(tmp_path, "--name", "Boaz", "--license", "CC0-1.0", "--consent", "commercial",
                        "--speaker", "Mark Chulsky", "--language", "Russian")
    req = seen[0]
    assert code == 0 and req.license == "CC0-1.0" and req.consent_mode == "manual" and req.consent_scope == "commercial"
    assert req.consent_name == req.speaker == "Mark Chulsky" and req.language == "ru" and req.asr_language == "Russian"
    code, seen = _train(tmp_path, "--consent", "auto")
    assert code == 0 and seen[0].consent_mode == "auto" and seen[0].license == "" and seen[0].language == ""
    code, seen = _train(tmp_path)
    assert code == 0 and seen[0].consent_mode == "none"


@pytest.mark.parametrize("extra", [["--license", "CC0-1.0"], ["--license", "CC-BY-NC-4.0", "--consent", "private_only"],
                                   ["--license", "WTFPL", "--consent", "commercial"], ["--language", "klingon"]])
def test_train_cli_rejects_inconsistent_options(tmp_path, extra):
    code, seen = _train(tmp_path, *extra)
    assert code != 0 and not seen


def test_voice_json_takes_cli_license_and_language(tmp_path):
    from workers import pipeline_runner as pr

    req = pr.TaskRequest(kind=pr.KIND_LORA, audio=tmp_path / "a.wav", voice_display_name="Boaz", license="CC0-1.0",
                         language="ru", consent_mode="manual", consent_scope="commercial", consent_name="Reader")
    block = {"scope": "commercial", "name": "Reader", "method": "manual", "confirmed": True}
    info = pr._write_voice_json(req, tmp_path, "en", 60.0, block)
    assert info["license"] == "CC0-1.0" and info["language"] == "ru" and info["commercial_use"]


def test_train_work_dir_is_per_voice(tmp_path):
    out = tmp_path / "out"
    _, a = _train(tmp_path, "--name", "Boaz", "--out", str(out))
    _, b = _train(tmp_path, "--name", "Tirzah", "--out", str(out))
    assert a[0].out_root == out / "Boaz_Voxprint" and b[0].out_root == out / "Tirzah_Voxprint"


def test_title_block_before_first_heading_opens_chapter_one():
    from core.book_parsers import parse_txt

    book = parse_txt("Бытие\n\nГлава 1\n\nВ начале сотворил Бог небо и землю.\n\nГлава 2\n\nИ совершил Бог к седьмому дню.")
    assert [c.title for c in book.chapters] == ["Глава 1", "Глава 2"]
    assert book.chapters[0].text.startswith("Бытие\n\nВ начале") and book.title == "Бытие"
    intro = parse_txt("Это длинное вступление к книге, а не заголовок.\n\nГлава 1\n\nТекст.")
    assert len(intro.chapters) == 2                        # real prose before the first heading stays its own part
    assert len(parse_txt("Бытие\n\nВ начале сотворил Бог небо и землю.").chapters) == 1
