"""Bundled open voices Levi and Miriam (build 703, voices-v2): pinned entries, first-run download, read-only in the library,
default multi-voice cast (fake downloads only).

Gideon and Tirzah were bundled up to build 702 (voices-v1): retired and hidden now, but an install that has them keeps them.
Asher and Noa are optional catalog voices on voices-v1; Natan, Shimon and Rivka on voices-v2.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from core import voice_info
from core.errors import VoiceLibraryError
from core.voice_library import VoiceLibrary
from infra import bundled_voices as bv
from infra import voice_repository as repo

ROOT = Path(__file__).resolve().parents[1]


def _zip(vid: str) -> bytes:
    spec = json.loads((ROOT / "tools" / "voice_specs" / f"{vid}.json").read_text(encoding="utf-8"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{vid}/adapter_model.safetensors", b"w" * 50)
        z.writestr(f"{vid}/adapter_config.json", "{}")
        z.writestr(f"{vid}/ref_sample.wav", b"RIFF")
        z.writestr(f"{vid}/voice.json", json.dumps(spec))
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def served(monkeypatch):
    """Serve a fake zip for the bundled voice and pin its hash (the real one is a ~53 MB release asset)."""
    blobs = {v["url"]: _zip(v["id"]) for v in bv.VOICES}
    monkeypatch.setattr(bv, "VOICES", tuple({**v, "sha256": hashlib.sha256(blobs[v["url"]]).hexdigest(),
                                             "size_bytes": len(blobs[v["url"]])} for v in bv.VOICES))
    calls = []

    def opener(req, timeout=0):
        calls.append(req.full_url)
        return _Resp(blobs[req.full_url])

    def download(entry, library, **kw):
        return repo.download_voice(entry, library, opener=opener, **kw)
    return download, calls


def test_pinned_entries_match_index_and_specs():
    index = {e["id"]: e for e in json.loads((ROOT / "voices" / "index.json").read_text(encoding="utf-8"))["voices"]}
    for v in bv.VOICES:
        e = index[v["id"]]
        assert (e["url"], e["sha256"], e["size_bytes"], e.get("bundled")) == (v["url"], v["sha256"], v["size_bytes"], True)
        assert e["license"] == "CC0-1.0" and e["url"].endswith(f"/voices-v2/{v['id']}.zip")
        spec = json.loads((ROOT / "tools" / "voice_specs" / f"{v['id']}.json").read_text(encoding="utf-8"))
        assert spec["license"] == "CC0-1.0" and spec["bundled"] is True
        assert spec["consent"]["method"] == "manual" and spec["consent"]["scope"] == "commercial"
        assert len(spec["consent"]["name"]) <= 80 and spec["speaker"] in spec["consent"]["name"]
        assert spec["project_url"].startswith(("https://librivox.org/", "https://archive.org/"))
        assert v["sha256"] in (ROOT / "voices" / "BUNDLED.md").read_text(encoding="utf-8")
    assert {e.id for e in bv.entries()} == {"levi", "miriam"} and all(e.bundled for e in bv.entries())
    for old in ("gideon", "tirzah"):        # no longer downloaded; still in the index (hidden) for installs that have them
        assert index[old].get("bundled") is not True and index[old]["hidden"] is True
    assert not any(p.suffix in (".safetensors", ".zip") for p in (ROOT / "voices").iterdir())    # no weights in git


def test_ensure_imports_read_only_with_licence(tmp_path, served):
    download, calls = served
    lib = VoiceLibrary(tmp_path / "voices")
    seen = []
    assert bv.ensure(lambda f, n: seen.append(f), library=lib, download=download) == ["levi", "miriam"]
    assert seen and max(seen) <= 1.0
    recs = {r.info["repo_id"]: r for r in lib.list_voices()}
    miriam = recs["miriam"]
    levi = recs["levi"]
    assert miriam.bundled and miriam.license == "CC0-1.0" and miriam.commercial_use
    assert levi.bundled and levi.license == "CC0-1.0" and levi.commercial_use
    assert miriam.info["consent"]["method"] == "manual" and miriam.info["consent"]["scope"] == "commercial"
    assert "Maya S" in miriam.info["consent"]["name"]
    assert levi.info["gender"] == "male" and miriam.info["gender"] == "female"
    for rec in (miriam, levi):
        with pytest.raises(VoiceLibraryError):
            lib.update(rec.id, name="Mine")
        with pytest.raises(VoiceLibraryError):
            lib.delete(rec.id)
    assert lib.update(miriam.id, adapter_scale=0.7).info["adapter_scale"] == 0.7      # the strength stays adjustable
    assert bv.missing(lib) == []
    n = len(calls)
    assert bv.ensure(library=lib, download=download) == [] and len(calls) == n       # nothing downloaded twice


def test_ensure_is_best_effort_per_voice(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")

    def download(entry, library, **kw):
        raise OSError("offline")
    assert bv.ensure(library=lib, download=download) == []
    assert [e.id for e in bv.missing(lib)] == ["levi", "miriam"]


def test_bundled_flag_is_only_true_or_absent():
    assert voice_info.normalize_info({"bundled": True})["bundled"] is True
    assert "bundled" not in voice_info.normalize_info({"bundled": "yes"})
    assert "bundled" not in voice_info.normalize_info({})


def test_first_run_download_includes_bundled_voices(monkeypatch):
    from infra import denoise_tool, vc_model
    from workers import pipeline_runner as pr

    monkeypatch.setattr(vc_model, "ready", lambda: True)
    monkeypatch.setattr(denoise_tool, "ready", lambda: "x")
    got = []
    monkeypatch.setattr(bv, "ensure", lambda progress=None, **kw: got.append(progress) or [])
    pr._prefetch_small_optional(lambda *a: None)
    assert len(got) == 1
    monkeypatch.setattr(bv, "missing", lambda library=None: ["miriam"])
    assert pr.everything_missing()


def test_full_size_counts_bundled_voices():
    from infra import setup_mode

    assert bv.total_bytes() == 53227088 + 53215075
    assert setup_mode.full_sizes()["models"] > bv.total_bytes()


def test_package_scrubs_local_paths(tmp_path):
    from tools import make_voice_package as mvp

    meta = tmp_path / "training_meta.json"
    meta.write_text(json.dumps({"ref_sample_audio": "C:\\Users\\someone\\ref.wav", "epochs": 2}), encoding="utf-8")
    mvp.scrub_training_meta(meta)
    assert json.loads(meta.read_text(encoding="utf-8")) == {"ref_sample_audio": "ref_sample.wav", "epochs": 2}
    assert mvp.index_entry({"id": "x", "name": "X", "license": "CC0-1.0", "bundled": True},
                           _write(tmp_path / "x.zip"), "https://e/x.zip")["bundled"] is True


def _write(p: Path) -> Path:
    p.write_bytes(b"zip")
    return p


def test_card_is_read_only_and_shows_licence(tmp_path, served):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ui.voices_window import VoiceCard

    _ = QApplication.instance() or QApplication([])
    download, _calls = served
    lib = VoiceLibrary(tmp_path / "voices")
    bv.ensure(library=lib, download=download)
    card = VoiceCard(lib.list_voices()[0])
    assert not card.btn_edit.isEnabled() and not card.btn_delete.isEnabled() and card.btn_narrate.isEnabled()
    assert card.lbl_note is not None and "CC0-1.0" in card.lbl_note.text()


def test_remote_card_states_licence_of_bundled_voice(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from infra import voice_catalog as catalog
    from ui.voices_window import RemoteCard

    _ = QApplication.instance() or QApplication([])
    items = catalog.build(VoiceLibrary(tmp_path / "voices"), bv.entries() + repo.parse_index({"voices": [
        {"id": "other", "url": "https://e/o.zip", "sha256": "a" * 64, "license": "CC-BY-4.0"}]}))
    cards = {i.key: RemoteCard(i) for i in items}
    assert "CC0-1.0" in cards["repo:miriam"].lbl_note.text()
    assert getattr(cards["repo:other"], "lbl_note", None) is None


def test_asher_and_noa_are_catalog_only():
    raw = json.loads((ROOT / "voices" / "index.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in raw["voices"]}
    assert "boaz" not in by_id and "PENDING" not in json.dumps(raw)
    assert {v["id"] for v in bv.VOICES} == {"levi", "miriam"}
    shown = {e.id for e in repo.parse_index(raw)}
    assert {"asher", "noa", "tirzah", "gideon", "levi", "miriam", "natan", "shimon", "rivka"} <= shown
    host = "https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/"
    pins = {
        "asher": ("795778a10b85495e5850d885484c313e3f7799f76ea4d81d41fe50162165a583", 53066747),
        "noa": ("755aeb47cc90a7bfb03d9349ed0fa8effbcc270fda22e5e9dc7b9539a462beed", 53197567),
    }
    for vid, reader, url in (
            ("asher", "Vladimir Anyanov", "https://archive.org/details/teachingsofchrist_1204_librivox"),
            ("noa", "Hanna Ponomarenko", "https://archive.org/details/izbrannye_2212_librivox")):
        entry = by_id[vid]
        sha, size = pins[vid]
        assert entry["sha256"] == sha and entry["url"] == host + vid + ".zip"
        assert entry.get("bundled") is not True and entry["license"] == "CC0-1.0"
        assert entry["size_bytes"] == size and entry["project_url"] == url and reader in entry["speaker"]
        spec = json.loads((ROOT / "tools" / "voice_specs" / f"{vid}.json").read_text(encoding="utf-8"))
        assert spec.get("bundled") is not True and spec["license"] == "CC0-1.0"
        assert spec["consent"]["scope"] == "commercial" and spec["consent"]["method"] == "manual"
        assert spec["consent"]["confirmed"] is True and reader in spec["consent"]["name"]
        assert len(spec["consent"]["name"]) <= 80 and spec["project_url"] == url
        parsed = repo.parse_index({"schema": 1, "voices": [entry]})
        assert len(parsed) == 1 and parsed[0].id == vid and parsed[0].bundled is False
    gideon = by_id["gideon"]
    assert gideon.get("bundled") is not True and gideon["size_bytes"] == 53031093
    assert gideon["sha256"] == "6ca0e6824080ee86c21a4b34472bd013bd182ef15bb393c6e494fd554e1c2266"
    assert gideon["url"] == host + "gideon.zip" and gideon["speaker"] == "Kazbek"


def test_locally_trained_bundled_voice_is_not_downloaded_again(tmp_path):
    from tests.test_voice_library import make_adapter

    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_from_adapter(make_adapter(tmp_path / "g", name="Gideon"))      # trained on the PC: id "gideon", no repo_id
    lib.add_from_adapter(make_adapter(tmp_path / "l", name="Levi"))          # the same for a locally trained Levi
    assert [e.id for e in bv.missing(lib)] == ["miriam"]


class _Rec:
    def __init__(self, vid, name, gender, repo_id=""):
        self.id, self.name = vid, name
        self.info = {"gender": gender, **({"repo_id": repo_id} if repo_id else {})}


def test_default_cast_prefers_levi_natan_shimon_miriam():
    """Build 703: narrator Levi, men Natan and Shimon, woman Miriam, whatever order the library lists them in."""
    from core import speakers as spk

    recs = [_Rec("gideon", "Gideon", "male", "gideon"), _Rec("noa", "Noa", "female", "noa"),
            _Rec("x1", "Shimon", "male", "shimon"), _Rec("levi", "Levi", "male", "levi"),
            _Rec("natan-2", "Natan", "male", "natan"), _Rec("miriam", "Miriam", "female", "miriam")]
    pref = bv.preferred_ids(recs)
    assert pref == {"narrator": "levi", "male": "natan-2", "male2": "x1", "female": "miriam"}
    assert spk.default_role_picks(recs, pref["narrator"], pref) == {"male": "natan-2", "male2": "x1", "female": "miriam"}


def test_default_cast_falls_back_when_voices_are_missing():
    """An old install without the new voices: the generic first-of-gender rule, never a missing id."""
    from core import speakers as spk

    recs = [_Rec("gideon", "Gideon", "male", "gideon"), _Rec("asher", "Asher", "male"), _Rec("noa", "Noa", "female")]
    pref = bv.preferred_ids(recs)
    assert pref == {}
    assert spk.default_role_picks(recs, "gideon", pref) == {"male": "asher", "male2": "", "female": "noa"}
    # only Natan installed: he is used, the second male falls back to the next male voice, the narrator is never a role
    recs.append(_Rec("natan", "Natan", "male", "natan"))
    pref = bv.preferred_ids(recs)
    assert spk.default_role_picks(recs, "gideon", pref) == {"male": "natan", "male2": "asher", "female": "noa"}
    # the preferred male voice is the narrator: it is not used for a role
    assert spk.default_role_picks(recs, "natan", pref)["male"] == "gideon"
    # a preferred id of the wrong gender is ignored
    assert spk.default_role_picks(recs, "gideon", {"female": "natan"})["female"] == "noa"
