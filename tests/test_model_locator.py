"""Reuse of models downloaded by other apps (core/model_locator.py) and ModelScope fallback - fake directory trees."""
import io
import json
import logging
import os
import struct
import urllib.request
from pathlib import Path

import pytest

from core import i18n, model_locator as ml
from core.errors import ModelDownloadError
from infra import model_downloader as md
from infra import modelscope_mirror as mm

REPO = "Org/Tiny-1.7B"
SHA_A = "a" * 40
SHA_B = "b" * 40
SIZES = {"config.json": 20, "tokenizer.json": 11, "model.safetensors": 0}   # weights size is filled in below


def write_st(path: Path, payload: int = 64) -> int:
    """A valid safetensors file with one tensor; returns the file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"w": {"dtype": "F32", "shape": [payload // 4], "data_offsets": [0, payload]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload)
    return path.stat().st_size


def make_model(d: Path, shard: bool = False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "x"}), encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    if shard:
        s1 = write_st(d / "model-00001-of-00002.safetensors")
        s2 = write_st(d / "model-00002-of-00002.safetensors")
        (d / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 128},
            "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    else:
        write_st(d / "model.safetensors")
    return d


def make_hf_cache(hub: Path, repo: str, sha: str, symlinks: bool = True, main: bool = True) -> Path:
    """models--Org--Name/{blobs,refs/main,snapshots/<sha>}; files are symlinks into blobs/ (or real files)."""
    rd = hub / ("models--" + repo.replace("/", "--"))
    snap = rd / "snapshots" / sha
    tmp = hub / "_src"
    make_model(tmp)
    (rd / "blobs").mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        if symlinks:
            blob = rd / "blobs" / (sha[:6] + f.name.replace(".", ""))
            blob.write_bytes(f.read_bytes())
            os.symlink(os.path.relpath(blob, snap), snap / f.name)
        else:
            (snap / f.name).write_bytes(f.read_bytes())
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    if main:
        (rd / "refs").mkdir(exist_ok=True)
        (rd / "refs" / "main").write_text(sha)
    return snap


def snapshot_of_tree(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        out[str(p.relative_to(root))] = (st.st_size, int(st.st_mtime), os.readlink(p) if p.is_symlink() else None)
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated 'user computer': empty HOME, no HF/ModelScope variables, locator switched on."""
    home = tmp_path / "home"
    home.mkdir()
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME", "XDG_CACHE_HOME",
              "MODELSCOPE_CACHE", "PINOKIO_HOME", "VOXPRINT_MODEL_DIRS", "HF_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("VOXPRINT_NO_EXTERNAL_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    return home


# ------------------------------------------------------------------------------------- Hugging Face cache
def test_hf_cache_with_symlinks_found(env):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    assert (snap / "config.json").is_symlink()
    f = ml.find_model(REPO)
    assert f and f.path == snap and f.kind == "hf" and f.revision == SHA_A and f.location == hub
    assert ml.find_model(REPO, SHA_A).match == "pinned"


def test_hf_cache_without_symlinks_windows_style(env):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A, symlinks=False)
    assert not (snap / "config.json").is_symlink()
    assert ml.find_model(REPO, SHA_A).path == snap


def test_hf_cache_root_precedence_and_env(env, monkeypatch, tmp_path):
    default = env / ".cache" / "huggingface" / "hub"
    hf_home = tmp_path / "hfhome"
    hub_cache = tmp_path / "customhub"
    make_hf_cache(default, REPO, SHA_A)
    make_hf_cache(hf_home / "hub", REPO, SHA_B)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert ml.find_model(REPO).revision == SHA_B                  # HF_HOME/hub before ~/.cache
    make_hf_cache(hub_cache, REPO, SHA_A)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    assert ml.find_model(REPO).location == hub_cache              # HF_HUB_CACHE before HF_HOME
    extra = tmp_path / "extra"
    make_hf_cache(extra, REPO, SHA_B)
    monkeypatch.setenv("VOXPRINT_MODEL_DIRS", os.pathsep.join([str(tmp_path / "nothing"), str(extra)]))
    f = ml.find_model(REPO)
    assert f.kind == "env" and f.location == extra                # our own override is searched first
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    assert ml.find_model(REPO) is None and ml.locate_models([REPO]) == {REPO: None}


def test_pinned_revision_mismatch_falls_back_with_log(env, caplog):
    hub = env / ".cache" / "huggingface" / "hub"
    make_hf_cache(hub, REPO, SHA_A)
    with caplog.at_level(logging.INFO, logger="voxprint.models"):
        assert ml.find_model(REPO, SHA_B) is None
    assert "differs from the verified" in caplog.text and SHA_A[:8] in caplog.text and SHA_B[:8] in caplog.text


def test_several_snapshots_pinned_one_wins_then_refs_main(env):
    hub = env / ".cache" / "huggingface" / "hub"
    make_hf_cache(hub, REPO, SHA_A, main=True)
    snap_b = make_hf_cache(hub, REPO, SHA_B, main=False)
    assert ml.find_model(REPO, SHA_B).path == snap_b
    assert ml.find_model(REPO).revision == SHA_A                  # no pin: refs/main


def test_incomplete_shard_and_missing_pieces_are_rejected(env):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    assert ml.find_model(REPO)
    # truncated weights (download cut off): header promises more than the file holds
    blob = (snap / "model.safetensors").resolve()
    blob.write_bytes(blob.read_bytes()[:-10])
    assert ml.find_model(REPO) is None
    assert "truncated" in ml.check_model_dir(snap, REPO)


def test_sharded_checkpoint_complete_and_incomplete(env, tmp_path):
    d = make_model(tmp_path / "plain" / "Tiny-1___7B", shard=True)
    assert ml.check_model_dir(d, REPO) is None
    (d / "model-00002-of-00002.safetensors").unlink()
    assert "shard" in ml.check_model_dir(d, REPO)
    write_st(d / "model-00002-of-00002.safetensors")
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    idx["metadata"]["total_size"] = 10 ** 9
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    assert "too small" in ml.check_model_dir(d, REPO)
    # shards without an index: all "-of-N" parts must exist
    (d / "model.safetensors.index.json").unlink()
    (d / "model-00002-of-00002.safetensors").unlink()
    assert ml.check_model_dir(d, REPO) == "shards missing"


def test_missing_or_invalid_config_and_broken_symlink(env):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    (snap / "config.json").resolve().write_text("not json")
    assert ml.find_model(REPO) is None
    (snap / "config.json").resolve().unlink()                      # dangling symlink
    assert "config.json" in ml.check_model_dir(snap, REPO)
    snap2 = make_hf_cache(hub, "Org/Other", SHA_A)
    (snap2 / "config.json").unlink()
    assert ml.find_model("Org/Other") is None


def test_unfinished_download_leftovers_are_rejected(env, tmp_path):
    d = make_model(tmp_path / "Tiny-1.7B")
    assert ml.check_model_dir(d, REPO) is None
    (d / "model.safetensors.incomplete").write_bytes(b"x")
    assert "unfinished" in ml.check_model_dir(d, REPO)
    part = make_model(tmp_path / "x" / "Tiny-1.7B.partial")
    assert ml.check_model_dir(part, REPO) == "unfinished download folder"


# ------------------------------------------------------------------------------------- Alexandria / Pinokio
def test_alexandria_pinokio_layout(env, monkeypatch):
    """Alexandria's recommended install: Pinokio sets HF_HOME=./cache/HF_HOME inside the app folder."""
    pin = env / "pinokio"
    hub = pin / "api" / "alexandria-audiobook.git" / "cache" / "HF_HOME" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    f = ml.find_model(REPO, SHA_A)
    assert f and f.path == snap and f.kind == "pinokio" and f.location == hub
    # shared Pinokio cache and a custom Pinokio home (PINOKIO_HOME)
    other = env / "elsewhere"
    shared = make_hf_cache(other / "cache" / "HF_HOME" / "hub", "Org/Shared", SHA_A)
    assert ml.find_model("Org/Shared") is None
    monkeypatch.setenv("PINOKIO_HOME", str(other))
    assert ml.find_model("Org/Shared").path == shared


def test_alexandria_default_hf_cache_models(env):
    """Without Pinokio Alexandria uses the plain HF cache (try_to_load_from_cache + from_pretrained(model_id))."""
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    assert ml.find_model(REPO).path == snap


# ------------------------------------------------------------------------------------- plain folders / ModelScope
def test_plain_folder_needs_matching_sizes_when_pinned(env, tmp_path, monkeypatch):
    models = env / "models"
    d = make_model(models / "Tiny-1.7B")
    sizes = {"config.json": (d / "config.json").stat().st_size, "model.safetensors": (d / "model.safetensors").stat().st_size,
             "tokenizer.json": (d / "tokenizer.json").stat().st_size}
    monkeypatch.setitem(ml.KNOWN_SIZES, REPO, (SHA_A, dict(sizes)))
    f = ml.find_model(REPO, SHA_A)
    assert f and f.match == "sizes" and f.path == d and f.revision is None
    assert ml.find_model(REPO, SHA_B) is None                      # sizes known only for the pinned commit SHA_A
    monkeypatch.setitem(ml.KNOWN_SIZES, REPO, (SHA_A, {**sizes, "model.safetensors": sizes["model.safetensors"] + 1}))
    assert ml.find_model(REPO, SHA_A) is None                      # same names, different size = other revision
    assert ml.find_model(REPO).match == "unpinned"                 # no pin: any complete copy


def test_plain_folder_revision_file_counts_as_sha(env):
    d = make_model(env / "models" / "Org--Tiny-1.7B")
    (d / ".revision").write_text(SHA_A)
    assert ml.find_model(REPO, SHA_A).match == "pinned"
    assert ml.find_model(REPO, SHA_B) is None


def test_modelscope_cache_layout_dots_become_triple_underscore(env, tmp_path, monkeypatch):
    ms = env / ".cache" / "modelscope" / "hub" / "models" / "Org" / "Tiny-1___7B"
    make_model(ms)
    f = ml.find_model(REPO)
    assert f and f.path == ms and f.kind == "modelscope"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "msroot"))
    d2 = make_model(tmp_path / "msroot" / "hub" / "models" / "Org" / "Tiny-1___7B")
    assert ml.find_model(REPO).path == d2                          # MODELSCOPE_CACHE first


def test_real_qwen_layout_sparse_files(env, monkeypatch):
    """The three Voxprint models with their real file names and sizes (sparse files, no disk used)."""
    repo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    sha, sizes = ml.KNOWN_SIZES[repo]
    d = env / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
    for rel, size in sizes.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".safetensors"):
            write_st(p)
            with open(p, "r+b") as fh:
                fh.truncate(size)
        elif rel.endswith("config.json"):
            p.write_text("{}" + " " * (size - 2))
        else:
            with open(p, "wb") as fh:
                fh.truncate(size)
    assert ml.check_model_dir(d, repo) is None
    assert ml.find_model(repo, sha).match == "sizes"
    (d / "speech_tokenizer" / "model.safetensors").unlink()        # the audio tokenizer is inside the Base repo
    assert ml.check_model_dir(d, repo) == "speech_tokenizer/model.safetensors missing"
    assert ml.find_model(repo, sha) is None


# ------------------------------------------------------------------------------------- read-only guarantee
def test_search_never_modifies_other_apps_files(env):
    hub = env / ".cache" / "huggingface" / "hub"
    make_hf_cache(hub, REPO, SHA_A)
    make_hf_cache(hub, "Org/Broken", SHA_A)
    make_model(env / "models" / "Tiny-1.7B")
    make_model(env / ".cache" / "modelscope" / "hub" / "models" / "Org" / "Tiny-1___7B")
    before = [snapshot_of_tree(r) for r in (hub, env / "models", env / ".cache" / "modelscope")]
    ml.find_model(REPO, SHA_B)
    ml.find_model(REPO)
    ml.find_model("Org/Broken", SHA_B)
    ml.locate_models([REPO, "Org/Broken", "Org/None"])
    after = [snapshot_of_tree(r) for r in (hub, env / "models", env / ".cache" / "modelscope")]
    assert before == after


def test_read_only_tree_is_usable(env):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    for p in [hub, *hub.rglob("*")]:
        if not p.is_symlink():
            os.chmod(p, 0o555 if p.is_dir() else 0o444)
    try:
        assert ml.find_model(REPO, SHA_A).path == snap
    finally:
        for p in [hub, *hub.rglob("*")]:
            if not p.is_symlink():
                os.chmod(p, 0o755)


# ------------------------------------------------------------------------------------- wiring into the downloader
def _fail_download(*a, **k):
    raise AssertionError("must not download: a complete copy exists in another app")


def test_ensure_model_reuses_external_copy_without_download(env, tmp_path):
    hub = env / ".cache" / "huggingface" / "hub"
    snap = make_hf_cache(hub, REPO, SHA_A)
    msgs = []
    before = snapshot_of_tree(hub)
    got = md.ensure_model(REPO, lambda s, f, m: msgs.append((f, m)), snapshot_download=_fail_download,
                          revision=SHA_A)
    assert got == snap
    assert msgs and msgs[-1][0] == 1.0 and "Tiny-1.7B" in msgs[-1][1] and str(hub) in msgs[-1][1]
    assert msgs[-1][1] == i18n.tr("progress.model_reused", short="Tiny-1.7B", where=str(hub))
    assert snapshot_of_tree(hub) == before
    assert not md.local_dir_for(REPO).exists()                    # nothing copied into the Voxprint folder


def test_own_copy_wins_over_external(env):
    make_hf_cache(env / ".cache" / "huggingface" / "hub", REPO, SHA_A)
    own = make_model(md.local_dir_for(REPO))
    assert md.ensure_model(REPO, snapshot_download=_fail_download, revision=SHA_A) == own


def test_pin_mismatch_downloads_into_own_folder(env, caplog):
    make_hf_cache(env / ".cache" / "huggingface" / "hub", REPO, SHA_A)
    calls = []

    def fake(repo_id, local_dir, revision=None, **kw):
        calls.append(revision)
        make_model(Path(local_dir))

    with caplog.at_level(logging.WARNING, logger="voxprint.models"):
        got = md.ensure_model(REPO, snapshot_download=fake, revision=SHA_B)
    assert calls == [SHA_B] and got == md.local_dir_for(REPO)
    assert "not reusing" in caplog.text


def test_staging_updates_never_use_external_copies(env, tmp_path):
    make_hf_cache(env / ".cache" / "huggingface" / "hub", REPO, SHA_A)
    stage = tmp_path / "stage"
    called = []

    def fake(repo_id, local_dir, **kw):
        called.append(local_dir)
        make_model(Path(local_dir))

    md.ensure_model(REPO, root=stage, snapshot_download=fake, revision=SHA_A)
    assert called and md.verify_local_model(md.local_dir_for(REPO, stage))


def test_models_missing_and_prefetch_know_external_copies(env, monkeypatch):
    from workers import pipeline_runner as pr

    make_hf_cache(env / ".cache" / "huggingface" / "hub", REPO, SHA_A)
    monkeypatch.setattr(md, "pinned_revision", lambda r: SHA_A if r == REPO else None)
    assert pr.models_missing([REPO, "Org/Absent"]) == ["Org/Absent"]
    msgs, ensured = [], []

    def fake_ensure(repo, cb):
        ensured.append(repo)
        cb(None, 1.0, "m:" + repo)

    downloaded = pr.prefetch_models(lambda s, f, m: msgs.append(m), [REPO, "Org/Absent"], ensure=fake_ensure)
    assert downloaded == ["Org/Absent"] and ensured == [REPO, "Org/Absent"]


def test_updater_ignores_external_models(env):
    from infra.version_manager import check_versions

    make_hf_cache(env / ".cache" / "huggingface" / "hub", REPO, SHA_A)
    assert md.local_revisions([REPO]) == {}                        # only Voxprint's own folder has a revision
    rep = check_versions(md.local_revisions([REPO]), lambda url: {"sha": SHA_B, "releases": {}}, lambda n: None,
                         packages={}, models=(REPO,))
    assert not rep.outdated_models


# ------------------------------------------------------------------------------------- locale message
def test_reuse_message_in_all_locales():
    texts = set()
    for lang in i18n.LANGS:
        i18n.set_language(lang)
        for key in ("progress.model_reused", "progress.mirror_modelscope"):
            t = i18n.tr(key, short="Qwen3-TTS-12Hz-1.7B-Base", where="C:\\x")
            assert t != key and "{" not in t and "Qwen3-TTS-12Hz-1.7B-Base" in t
        texts.add(i18n.tr("progress.model_reused", short="S", where="W"))
    assert len(texts) == len(i18n.LANGS)


# ------------------------------------------------------------------------------------- ModelScope mirror
class FakeResp:
    def __init__(self, data: bytes, status: int = 200):
        self._b = io.BytesIO(data)
        self.status = status

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeModelScope:
    def __init__(self, files, broken_after=None):
        self.files = files
        self.broken_after = broken_after     # {path: bytes served before the connection drops}
        self.requests = []

    def __call__(self, req, timeout):
        url = req.full_url
        self.requests.append((url, req.headers.get("Range")))
        if "/repo/files" in url:
            data = {"Code": 200, "Data": {"Files": [{"Path": p, "Size": len(b), "Type": "blob"}
                                                   for p, b in self.files.items()]}}
            return FakeResp(json.dumps(data).encode())
        from urllib.parse import parse_qs, urlparse
        rel = parse_qs(urlparse(url).query)["FilePath"][0]
        body = self.files[rel]
        rng = req.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = body[start:]
        if self.broken_after and rel in self.broken_after and not rng:
            cut = self.broken_after.pop(rel)
            return _Dropping(body[:cut])
        return FakeResp(body, 206 if rng else 200)


class _Dropping(FakeResp):
    def read(self, n=-1):
        d = super().read(n)
        if not d:
            raise OSError("connection reset")
        return d


def _tiny_files():
    sio = io.BytesIO()
    p = Path("/tmp/_x.st")
    write_st(p)
    st = p.read_bytes()
    p.unlink()
    return {"config.json": b'{"a": 1}', "model.safetensors": st, "sub/tok.json": b"{}"}


def test_mirror_download_resumes_and_verifies(tmp_path):
    files = _tiny_files()
    srv = FakeModelScope(files, broken_after={"model.safetensors": 30})
    dest = tmp_path / "x.partial"
    with pytest.raises(mm.MirrorError):
        mm.download_repo(REPO, dest, opener=srv)
    assert (dest / "model.safetensors.incomplete").stat().st_size == 30       # kept for resuming
    assert not (dest / "model.safetensors").exists()
    seen = []
    sizes = mm.download_repo(REPO, dest, progress=seen.append, opener=srv)
    assert srv.requests[-1][1] == "bytes=30-" or any(r[1] == "bytes=30-" for r in srv.requests)
    assert (dest / "model.safetensors").read_bytes() == files["model.safetensors"]
    assert not list(dest.rglob("*.incomplete")) and seen[-1] == 1.0 and sizes["sub/tok.json"] == 2
    assert md.verify_local_model(dest)


def test_mirror_rejects_other_revision_by_size_and_unsafe_paths(tmp_path):
    files = _tiny_files()
    with pytest.raises(mm.MirrorError, match="differs from the verified"):
        mm.download_repo(REPO, tmp_path / "a", expected_sizes={"config.json": 999}, opener=FakeModelScope(files))
    with pytest.raises(mm.MirrorError, match="unsafe"):
        mm.download_repo(REPO, tmp_path / "b", opener=FakeModelScope({"../evil.txt": b"x", "config.json": b"{}"}))
    assert not (tmp_path / "evil.txt").exists()


def test_hf_is_fast_probe(monkeypatch):
    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()

    def slow(req, timeout):
        clock.t += 5.5
        return FakeResp(b"{}")

    def down(req, timeout):
        raise OSError("unreachable")

    assert mm.hf_is_fast(REPO, lambda r, t: FakeResp(b"{}"), 6.0, clock) is True
    assert mm.hf_is_fast(REPO, slow, 6.0, clock) is False
    assert mm.hf_is_fast(REPO, down, 6.0, clock) is False
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")        # the user's own mirror is respected
    assert mm.hf_is_fast(REPO, down, 6.0, clock) is True


def _mirror_stub(files):
    def dl(repo_id, dest, progress, expected):
        for rel, body in files.items():
            p = Path(dest) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
        progress(1.0)
    return dl


def test_ensure_model_uses_modelscope_first_when_hf_is_slow(env, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    files = _tiny_files()
    msgs = []
    got = md.ensure_model(REPO, lambda s, f, m: msgs.append(m), snapshot_download=_fail_download, revision=SHA_A,
                          hf_probe=lambda r: False, mirror_download=_mirror_stub(files))
    assert md.verify_local_model(got) and got == md.local_dir_for(REPO)
    assert i18n.tr("progress.mirror_modelscope", short="Tiny-1.7B") in msgs
    assert not (got / ".revision").exists()          # sizes of the pinned commit unknown for this fake repo


def test_modelscope_copy_gets_revision_only_if_sizes_match_pinned_commit(env, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    files = _tiny_files()
    monkeypatch.setitem(ml.KNOWN_SIZES, REPO, (SHA_A, {k: len(v) for k, v in files.items()}))
    got = md.ensure_model(REPO, snapshot_download=_fail_download, revision=SHA_A, hf_probe=lambda r: False,
                          mirror_download=_mirror_stub(files))
    assert (got / ".revision").read_text() == SHA_A


def test_hf_failure_falls_back_to_modelscope_and_total_failure_keeps_partial(env, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    files = _tiny_files()

    def broken_hf(**kw):
        raise OSError("network down")

    got = md.ensure_model(REPO, snapshot_download=broken_hf, revision=SHA_A, hf_probe=lambda r: True,
                          mirror_download=_mirror_stub(files))
    assert md.verify_local_model(got)

    def broken_ms(*a, **k):
        raise mm.MirrorError("down")

    shutil_target = md.local_dir_for("Org/Other")
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model("Org/Other", snapshot_download=broken_hf, revision=SHA_A, hf_probe=lambda r: True,
                        mirror_download=broken_ms)
    assert "network down" in (ei.value.details or "") and "down" in (ei.value.details or "")
    assert not shutil_target.exists()


def test_mirror_disabled_by_env_uses_only_hugging_face(env, monkeypatch):
    def broken_hf(**kw):
        raise OSError("network down")

    with pytest.raises(ModelDownloadError):
        md.ensure_model(REPO, snapshot_download=broken_hf, revision=SHA_A, hf_probe=lambda r: False,
                        mirror_download=_fail_download)         # VOXPRINT_NO_MIRROR=1 (conftest) => never called
