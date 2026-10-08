"""Online voices in the app: localized names, merging the index with the local library, download on first use,
resumable downloads, the offline cache, the test-only licence and the voice package builder."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from core import consent as c
from core import i18n, voice_info
from core.voice_library import VoiceLibrary
from infra import voice_catalog as cat
from infra import voice_repository as repo
from tests.test_voice_library import make_adapter
from tests.test_voice_repository import FakeResponse, opener_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_voice_package as mvp  # noqa: E402

SPEC = json.loads((ROOT / "tools" / "voice_specs" / "example-open-voice.json").read_text(encoding="utf-8"))     # the open placeholder entry
#: A restricted variant: a voice shared for tests only (owner is the publisher, no consent clip).
TEST_ONLY_SPEC = {**SPEC, "license": voice_info.LICENSE_TEST_ONLY, "author": "Owner",
                  "consent": {"scope": "private_only", "name": "Owner", "date": "2026-10-03", "method": "owner", "owner_confirmed": True, "confirmed": True}}
URL = "https://host.example/open-voice.zip"


def package(tmp_path, spec=None):
    """Build a package from a fake adapter (default: the test-only variant); returns (zip bytes, index entry)."""
    ad = make_adapter(tmp_path / "ad", name="Open Voice", with_voice_json=True)
    entry = mvp.build(ad, spec or TEST_ONLY_SPEC, tmp_path / "dist", URL, typed_names=False)     # plain names: typed ones are covered in test_voice_type_names
    return (tmp_path / "dist" / "open-voice.zip").read_bytes(), entry


def index_for(entry):
    return {"schema": 1, "voices": [entry]}


# ----------------------------------------------------------------------------- package -> index -> library
def test_package_builder_writes_final_voice_json_zip_and_entry(tmp_path):
    data, entry = package(tmp_path)
    info = json.loads((tmp_path / "dist" / "open-voice" / "voice.json").read_text(encoding="utf-8"))
    assert info["names"] == {"ru": "Открытый голос", "en": "Open Voice", "de": "Offene Stimme"}
    assert info["license"] == voice_info.LICENSE_TEST_ONLY and info["commercial_use"] is False
    assert info["consent"]["method"] == "owner" and info["consent"]["owner_confirmed"] and info["consent"]["scope"] == c.PRIVATE
    assert info["consent"]["confirmed"] and not info["consent"]["recorded_statement"] and info["author"] == "Owner"
    assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["size_bytes"] == len(data) and entry["url"] == URL
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "open-voice/voice.json" in names and "open-voice/adapter_model.safetensors" in names
    assert not (tmp_path / "dist" / "open-voice" / "consent_statement.wav").exists()
    parsed = repo.parse_index(index_for(entry))
    assert parsed[0].names["ru"] == "Открытый голос" and parsed[0].license == voice_info.LICENSE_TEST_ONLY
    assert parsed[0].descriptions["de"].startswith("Eine vollständig")


def test_download_imports_the_package_with_names_licence_and_owner_consent(tmp_path):
    data, entry = package(tmp_path)
    ent = repo.parse_index(index_for(entry))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    rec = repo.download_voice(ent, lib, opener=opener_for({URL: data}))
    assert rec.info["repo_id"] == "open-voice" and rec.test_only and rec.scope == c.PRIVATE and not rec.commercial_use
    assert rec.info["consent"]["method"] == "owner"
    assert (rec.path / "adapter_model.safetensors").is_file() and not (rec.path / c.CLIP_NAME).exists()


DETAILS = {"speaker": "Анна", "prepared_by": "Studio Nord", "organization": "Voxprint e.V.",
           "project_url": "https://example.org/voice", "gender": "female", "age_group": "adult", "language": "ru-RU"}


def test_package_carries_schema3_details_into_entry_model_card_and_library(tmp_path):
    data, entry = package(tmp_path, {**SPEC, **DETAILS, "license": "CC-BY-4.0"})
    for k, v in DETAILS.items():
        assert entry[k] == v, k
    assert entry["voice_type"] == "female"                               # derived, for older app versions
    card = (tmp_path / "dist" / "open-voice" / "README.md").read_text(encoding="utf-8")
    head = card.split("---")[1]
    assert 'license: "cc-by-4.0"' in head and '- "ru"' in head and "pipeline_tag: text-to-speech" in head
    for text in ("| Speaker | Анна |", "| Prepared by | Studio Nord |", "| Organization | Voxprint e.V. |",
                 "| Project | https://example.org/voice |", "| Gender | female |", "| Age group | adult |",
                 "| Language | Russian (`ru-RU`) |"):
        assert text in card, text
    assert "open-voice/README.md" in zipfile.ZipFile(io.BytesIO(data)).namelist()
    ent = repo.parse_index(index_for(entry))[0]
    assert (ent.speaker, ent.gender, ent.age_group, ent.project_url) == ("Анна", "female", "adult", "https://example.org/voice")
    rec = repo.download_voice(ent, VoiceLibrary(tmp_path / "lib"), opener=opener_for({URL: data}))
    assert all(rec.info[k] == v for k, v in DETAILS.items())            # README.md is not imported, the fields are
    assert not (rec.path / "README.md").exists()


def test_model_card_for_a_custom_licence_and_hostile_text(tmp_path):
    info = voice_info.normalize_info({"id": "v", "name": "A | <b>B</b>", "license": voice_info.LICENSE_TEST_ONLY,
                                      "speaker": "x|y"})
    card = mvp.model_card(info)
    assert 'license: "other"' in card and 'license_name: "custom-test-use-only"' in card
    assert "# A \\| &lt;b>B&lt;/b>" in card and "| Speaker | x\\|y |" in card
    assert "language:" not in card                                       # no language known: no metadata line
    bad = voice_info.normalize_info({"project_url": "javascript:alert(1)"})
    assert "javascript" not in mvp.model_card(bad)


def test_old_spec_voice_type_still_names_the_package(tmp_path):
    ad = make_adapter(tmp_path / "ad", name="Open Voice", with_voice_json=True)
    entry = mvp.build(ad, {**SPEC, "voice_type": "child"}, tmp_path / "dist", URL)
    assert entry["voice_type"] == "child" and entry["age_group"] == "child"
    assert (tmp_path / "dist" / "open-voice_child.zip").is_file()


def test_names_and_descriptions_follow_the_ui_language_with_fallback(tmp_path):
    data, entry = package(tmp_path)
    lib = VoiceLibrary(tmp_path / "lib")
    rec = repo.download_voice(repo.parse_index(index_for(entry))[0], lib, opener=opener_for({URL: data}))
    ent = repo.parse_index(index_for(entry))[0]
    for lang, name in (("ru", "Открытый голос"), ("en", "Open Voice"), ("de", "Offene Stimme")):
        i18n.set_language(lang)
        assert rec.name == name and ent.display_name == name
    i18n.set_language("ru")
    assert rec.description.startswith("Полностью открытый") and ent.display_description.startswith("Полностью открытый")
    i18n.set_language("en")
    assert rec.description.startswith("A fully open")
    assert rec.info["name"] == "Open Voice"                      # the stored default stays untouched
    plain = lib.add_from_adapter(make_adapter(tmp_path / "p", name="Anna"))
    assert plain.name == "Anna" and plain.description == plain.info["description"]    # no table: the default
    rec.info["names"] = {"fr": "Exemple"}                     # no entry for the UI language: the default name
    assert rec.name == "Open Voice"
    i18n.set_language("ru")


def test_names_are_normalised_and_dirty_entries_dropped():
    out = voice_info.normalize_info({"id": "x", "name": "N", "names": {"RU": " Имя ", "english": "no", "de": "", "e": "x", 5: "z"},
                                     "descriptions": {"en": "d" * 500}, "repo_id": " a "})
    assert out["names"] == {"ru": "Имя"} and len(out["descriptions"]["en"]) <= voice_info.MAX_DESCRIPTION_CHARS
    assert out["repo_id"] == "a" and "names" not in voice_info.normalize_info({"id": "x", "name": "N", "names": "str"})
    assert voice_info.LICENSE_TEST_ONLY in voice_info.LICENSES and not voice_info.license_allows_commercial(voice_info.LICENSE_TEST_ONLY)
    assert c.clean_consent({"method": "owner", "owner_confirmed": 1})["owner_confirmed"] is True
    assert "owner_confirmed" not in c.clean_consent({"method": "manual"})


# ----------------------------------------------------------------------------- catalog
def test_catalog_lists_local_voices_then_remote_ones_and_hides_installed_downloads(tmp_path):
    data, entry = package(tmp_path)
    entries = repo.parse_index(index_for(entry) | {"voices": [entry, {**entry, "id": "bob", "name": "Bob", "names": {},
                                                                      "license": "CC-BY-4.0"}]})
    lib = VoiceLibrary(tmp_path / "lib")
    lib.add_from_adapter(make_adapter(tmp_path / "own", name="Anna"))
    items = cat.build(lib, entries)
    assert [(i.name, i.installed) for i in items] == [("Anna", True), ("Открытый голос", False), ("Bob", False)]
    assert items[1].key == "repo:open-voice" and items[1].scope == c.PRIVATE and items[2].scope == c.COMMERCIAL
    assert items[1].remote and items[1].size_bytes == entry["size_bytes"]
    rec = repo.download_voice(entries[0], lib, opener=opener_for({URL: data}))
    items = cat.build(lib, entries)
    assert [i.installed for i in items] == [True, True, False]       # matched by repo_id: shown once, as local
    assert cat.build(lib, [])[0].key == "anna" and rec.id in {i.key for i in items}
    assert cat.scope_for_license("CC-BY-NC-4.0") == c.PUBLIC_NC and cat.scope_for_license("custom/personal-only") == c.PRIVATE
    assert cat.size_text(0) == "" and cat.size_text(58_000_000) == "58 MB" and cat.size_text(2_500_000_000) == "2.5 GB"


def test_ensure_local_downloads_a_remote_key_and_passes_local_ones_through(tmp_path):
    data, entry = package(tmp_path)
    entries = repo.parse_index(index_for(entry))
    lib = VoiceLibrary(tmp_path / "lib")
    local = lib.add_from_adapter(make_adapter(tmp_path / "own", name="Anna"))
    assert cat.ensure_local(local.id, lib, entries, download=lambda *a, **k: pytest.fail("no download")).id == local.id
    assert cat.ensure_local("repo:nope", lib, entries) is None
    got = cat.ensure_local("repo:open-voice", lib, entries, opener=opener_for({URL: data}))
    assert got.id and lib.get(got.id) is not None


# ----------------------------------------------------------------------------- resume / cache
class RangeOpener:
    """Serves ``data``; honours Range with 206 (or ignores it with 200 when ``ignore_range``)."""

    def __init__(self, data, ignore_range=False):
        self.data, self.ignore, self.ranges = data, ignore_range, []

    def __call__(self, req, timeout=0):
        rng = req.headers.get("Range") or req.get_header("Range")
        self.ranges.append(rng)
        if rng and not self.ignore:
            start = int(rng.split("=")[1].rstrip("-"))
            r = FakeResponse(self.data[start:])
            r.status = 206
        else:
            r = FakeResponse(self.data)
            r.status = 200
        return r


@pytest.mark.parametrize("ignore", [False, True])
def test_interrupted_download_resumes_and_is_verified(tmp_path, ignore):
    data, entry = package(tmp_path)
    ent = repo.parse_index(index_for(entry))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    part_dir = repo.paths.state_dir() / "voice_downloads"
    part_dir.mkdir(parents=True)
    (part_dir / f"{ent.sha256}.part").write_bytes(data[:1000])          # what a previous, cut-off try left behind
    op = RangeOpener(data, ignore_range=ignore)
    rec = repo.download_voice(ent, lib, opener=op)
    assert op.ranges == ["bytes=1000-"] and rec.info["repo_id"] == "open-voice"
    assert not list(part_dir.iterdir())                                   # the partial file is gone after the import


def test_corrupt_partial_is_discarded_after_a_hash_mismatch(tmp_path):
    data, entry = package(tmp_path)
    ent = repo.parse_index(index_for(entry))[0]
    lib = VoiceLibrary(tmp_path / "lib")
    part_dir = repo.paths.state_dir() / "voice_downloads"
    part_dir.mkdir(parents=True)
    (part_dir / f"{ent.sha256}.part").write_bytes(b"x" * 1000)
    with pytest.raises(Exception):
        repo.download_voice(ent, lib, opener=RangeOpener(data))
    assert not list(part_dir.iterdir()) and lib.is_empty()
    assert repo.download_voice(ent, lib, opener=RangeOpener(data)).id           # the next try starts clean


def test_offline_index_falls_back_to_the_last_good_one(tmp_path):
    data, entry = package(tmp_path)
    url = "https://example.org/index.json"
    good = repo.fetch_index(url, opener=opener_for({url: json.dumps(index_for(entry)).encode()}))
    assert good.voices and not good.offline
    off = repo.fetch_index(url, opener=opener_for({}))
    assert off.offline and [v.id for v in off.voices] == ["open-voice"] and off.error == "unreachable"
    assert [v.id for v in repo.load_cache()] == ["open-voice"]
    bad = repo.fetch_index(url, opener=opener_for({url: b"<html>"}))
    assert bad.offline and bad.voices                                     # a broken index must not wipe the list
    assert repo.fetch_index(url, opener=opener_for({url: json.dumps(index_for(entry)).encode()})).voices[0].id == "open-voice"
