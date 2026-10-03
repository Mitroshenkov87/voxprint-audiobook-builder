"""Online voice repository: index parsing, graceful failures, download with SHA-256 check."""
import hashlib
import io
import json
import zipfile

import pytest

from core.errors import CancelledByUser, VoiceRepositoryError
from core.events import CancelToken
from core.voice_library import VoiceLibrary
from infra import voice_repository as repo


def make_zip_bytes(name="Anna"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("adapter_model.safetensors", b"w" * 50)
        z.writestr("adapter_config.json", "{}")
        z.writestr("ref_sample.wav", b"RIFF")
    return buf.getvalue()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def opener_for(mapping):
    def opener(req, timeout=0):
        url = req.full_url
        if url not in mapping:
            raise OSError("404")
        return FakeResponse(mapping[url])
    return opener


def index_doc(data):
    sha = hashlib.sha256(data).hexdigest()
    return {"schema": 1, "voices": [
        {"id": "anna-en", "name": "Anna", "language": "english", "author": "A", "license": "CC-BY-4.0",
         "url": "https://x.example/anna.zip", "sha256": sha, "size_bytes": len(data), "voice_type": "female",
         "description": "calm"},
        {"id": "bad-nohash", "name": "Bad", "url": "https://x.example/b.zip"},
        {"id": "bad-http", "name": "Bad", "url": "http://x.example/b.zip", "sha256": sha},
        {"id": "anna-en", "name": "dup", "url": "https://x.example/d.zip", "sha256": sha},
        "garbage"]}


def test_default_url_points_to_the_project_index_and_a_placeholder_is_not_configured(monkeypatch):
    monkeypatch.delenv(repo.ENV_INDEX_URL, raising=False)
    assert repo.index_url() == repo.DEFAULT_INDEX_URL and repo.is_configured() and "OWNER" not in repo.DEFAULT_INDEX_URL
    monkeypatch.setattr(repo, "DEFAULT_INDEX_URL", "https://raw.githubusercontent.com/OWNER/voxprint-voices/main/index.json")
    assert not repo.is_configured()
    res = repo.fetch_index(opener=lambda *a, **k: pytest.fail("no network for an unconfigured repository"))
    assert res.error == "not_configured" and res.voices == []


def test_url_is_configurable_by_state_file_and_environment(monkeypatch):
    monkeypatch.delenv(repo.ENV_INDEX_URL, raising=False)
    repo.set_index_url("https://example.org/voices/index.json")
    assert repo.index_url() == "https://example.org/voices/index.json" and repo.is_configured()
    monkeypatch.setenv(repo.ENV_INDEX_URL, "https://env.example/i.json")
    assert repo.index_url() == "https://env.example/i.json"
    monkeypatch.delenv(repo.ENV_INDEX_URL)
    repo.set_index_url("")
    assert repo.index_url() == repo.DEFAULT_INDEX_URL
    assert not repo.is_configured("http://insecure.example/i.json") and not repo.is_configured("")


def test_fetch_and_parse_index_skips_bad_entries():
    data = make_zip_bytes()
    url = "https://example.org/index.json"
    res = repo.fetch_index(url, opener=opener_for({url: json.dumps(index_doc(data)).encode()}))
    assert res.error == "" and [v.id for v in res.voices] == ["anna-en"]
    v = res.voices[0]
    assert v.commercial_use is True and v.voice_type == "female" and v.license_url.endswith("by/4.0/")


def test_empty_unreachable_and_invalid_index_are_handled():
    url = "https://example.org/index.json"
    empty = repo.fetch_index(url, opener=opener_for({url: b'{"schema": 1, "voices": []}'}))
    assert empty.error == "" and empty.voices == []
    assert repo.fetch_index(url, opener=opener_for({})).error == "unreachable"
    assert repo.fetch_index(url, opener=opener_for({url: b"<html>not json"})).error == "invalid"
    assert repo.fetch_index(url, opener=opener_for({url: b'{"nope": 1}'})).error == "invalid"
    big = b" " * (repo.MAX_INDEX_BYTES + 10)
    assert repo.fetch_index(url, opener=opener_for({url: big})).error == "invalid"


def test_download_verifies_hash_and_imports_with_declared_license(tmp_path):
    data = make_zip_bytes()
    entry = repo.parse_index(index_doc(data))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    seen = []
    rec = repo.download_voice(entry, lib, opener=opener_for({entry.url: data}), progress=lambda f, n: seen.append(f))
    assert rec.name == "Anna" and rec.license == "CC-BY-4.0" and rec.commercial_use and rec.info["author"] == "A"
    assert seen and seen[-1] == 1.0 and lib.get(rec.id) is not None


def test_download_with_wrong_hash_imports_nothing(tmp_path):
    data = make_zip_bytes()
    entry = repo.parse_index(index_doc(data))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    with pytest.raises(VoiceRepositoryError):
        repo.download_voice(entry, lib, opener=opener_for({entry.url: data + b"tampered"}))
    assert lib.is_empty() and not list(lib.root.iterdir())


def test_download_refuses_http_oversize_and_network_errors(tmp_path):
    data = make_zip_bytes()
    entry = repo.parse_index(index_doc(data))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    http = repo.RepoVoice(id="x", name="x", url="http://insecure/x.zip", sha256="0" * 64)
    with pytest.raises(VoiceRepositoryError):
        repo.download_voice(http, lib, opener=opener_for({}))
    with pytest.raises(VoiceRepositoryError):
        repo.download_voice(entry, lib, opener=opener_for({}))                  # 404 / offline
    entry.size_bytes = 5                                                       # the server sends more than announced
    with pytest.raises(VoiceRepositoryError):
        repo.download_voice(entry, lib, opener=opener_for({entry.url: b"z" * (3 << 20)}))
    assert lib.is_empty()


def test_download_can_be_cancelled(tmp_path):
    data = make_zip_bytes()
    entry = repo.parse_index(index_doc(data))[0]
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledByUser):
        repo.download_voice(entry, VoiceLibrary(tmp_path / "lib"), opener=opener_for({entry.url: data}), cancel=tok)
