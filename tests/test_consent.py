"""Voice-owner consent: statement parsing (ru/en/de), voice.json block, licence mapping, pipeline modes, confirmation, badges."""
import json
import time

import pytest

from core import audio_utils as au
from core import consent as c
from core import voice_info
from core.asr import FakeASR
from core.voice_library import VoiceLibrary
from tests.synth import TrueRateAligner, make_text, synth_reading

NAMES = {"ru": ("Иван Петров", "3 октября 2026"), "en": ("Ivan Petrov", "October 3, 2026"), "de": ("Ivan Petrov", "03.10.2026")}


@pytest.mark.parametrize("lang", ["ru", "en", "de"])
@pytest.mark.parametrize("scope", list(c.SCOPES))
def test_every_template_is_read_back_exactly(lang, scope):
    name, date = NAMES[lang]
    p = c.parse_statement(c.example_statement(lang, scope, name, date))
    assert p.scope == scope and p.confident
    assert p.name == name and p.date == "2026-10-03"


def test_asr_style_variants_and_unclear_statements_default_to_the_strictest_level():
    assert c.parse_statement("я иван петров сегодня разрешаю использовать мой голос в коммерческих целях можно продавать").scope == c.COMMERCIAL
    assert c.parse_statement("озвучки можно публиковать в открытом доступе но нельзя продавать, я разрешаю").scope == c.PUBLIC_NC
    for text in ("", "просто какой-то текст о погоде", "I allow it"):
        p = c.parse_statement(text)
        assert p.scope == c.PRIVATE and not p.confident
    mixed = c.parse_statement("Я разрешаю использовать в коммерческих целях, но только в личных целях и нельзя публиковать")
    assert mixed.scope == c.PRIVATE and not mixed.confident                     # contradictory: strictest, ask the user
    assert not c.parse_statement("commercial purposes").confident               # a scope word without any permission phrase


def test_scope_is_not_confused_by_negations_of_commercial():
    for t in ("для некоммерческих целей, разрешаю", "for non-commercial purposes I give permission", "nichtkommerzielle Zwecke, ich erlaube"):
        assert c.parse_statement(t).scope == c.PUBLIC_NC, t


def test_voice_json_block_roundtrip_sanitising_and_licence_mapping():
    block = c.build_consent(c.parse_statement(c.example_statement("en", c.PUBLIC_NC, "Ann", "03.10.2026")), scope=c.PUBLIC_NC,
                            method="spoken", recorded=True, confirmed=False, clip=c.CLIP_NAME)
    info = voice_info.build_voice_info("V", "russian", 100, 5, "m", license=c.license_for_scope(c.PUBLIC_NC), consent=block)
    assert info["consent"]["scope"] == c.PUBLIC_NC and info["consent"]["recorded_statement"] and info["consent"]["name"] == "Ann"
    assert info["license"] == "CC-BY-NC-4.0" and info["commercial_use"] is False
    assert voice_info.normalize_info(json.loads(json.dumps(info)))["consent"] == info["consent"]
    evil = voice_info.normalize_info({"name": "x", "consent": {"scope": "everything", "method": "hack", "name": "A\n\nB", "confirmed": "yes"}})
    assert evil["consent"]["scope"] == c.PRIVATE and evil["consent"]["method"] == "none" and "\n" not in evil["consent"]["name"]
    assert "consent" not in voice_info.normalize_info({"name": "x"})
    assert [c.license_for_scope(s) for s in c.SCOPES] == ["CC-BY-4.0", "CC-BY-NC-4.0", "custom/personal-only"]


def test_scope_of_falls_back_to_the_licence():
    assert c.scope_of({"license": "CC-BY-4.0", "commercial_use": True}) == c.COMMERCIAL
    assert c.scope_of({"license": "CC-BY-NC-4.0", "commercial_use": False}) == c.PUBLIC_NC
    assert c.scope_of({"license": "custom/personal-only", "commercial_use": False}) == c.PRIVATE
    assert c.scope_of({"license": "CC-BY-4.0", "commercial_use": True, "consent": {"scope": "private_only"}}) == c.PRIVATE


class _NoUpd:
    def should_autocheck(self):
        return False


def _recording(tmp_path, statement_audio=True):
    text = make_text(40, 1)
    body, _ = synth_reading(text, seed=1)
    statement, _ = synth_reading(make_text(8, 2), seed=2)         # the spoken statement is audio at the end
    import numpy as np
    au.write_wav(tmp_path / "in.wav", np.concatenate([body, statement]) if statement_audio else body, 16000)
    (tmp_path / "in.txt").write_text(text, encoding="utf-8")


def _run(tmp_path, monkeypatch, **req_kw):
    import core.lora_trainer as lt
    from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task

    def fake_train(d, o, *a, **k):
        o.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav"):
            (o / f).write_bytes(b"x")
        (o / "training_meta.json").write_text('{"epochs": 1, "ref_sample_text": "t"}', encoding="utf-8")
        return o
    monkeypatch.setattr(lt, "train_lora_from_dataset", fake_train)
    _recording(tmp_path, req_kw.get("consent_mode") == "auto")
    lib = VoiceLibrary(tmp_path / "lib")
    res = run_task(TaskRequest(kind=KIND_LORA, audio=tmp_path / "in.wav", text=tmp_path / "in.txt", out_root=tmp_path / "out",
                               **req_kw), updater=_NoUpd(), aligner_factory=TrueRateAligner, voice_library=lib,
                   asr_factory=lambda: asr_holder["asr"])
    return res, lib


asr_holder = {}


def test_auto_mode_reads_the_statement_saves_the_clip_and_maps_the_licence(tmp_path, monkeypatch):
    asr_holder["asr"] = FakeASR([c.example_statement("ru", c.COMMERCIAL, "Иван Петров", "3 октября 2026")])
    res, lib = _run(tmp_path, monkeypatch, consent_mode="auto")
    rec = lib.get(res.voice_id)
    cb = rec.info["consent"]
    assert cb["scope"] == c.COMMERCIAL and cb["name"] == "Иван Петров" and cb["date"] == "2026-10-03"
    assert cb["method"] == "spoken" and cb["recorded_statement"] and not cb["confirmed"] and cb["clip"] == c.CLIP_NAME
    assert rec.info["license"] == "CC-BY-4.0" and rec.info["commercial_use"] and rec.info["author"] == "Иван Петров"
    assert (rec.path / c.CLIP_NAME).is_file() and rec.scope == c.COMMERCIAL
    assert res.consent == cb


def test_unclear_or_missing_statement_is_private_with_a_warning(tmp_path, monkeypatch):
    asr_holder["asr"] = FakeASR(["a few random words about the weather"])
    res, lib = _run(tmp_path, monkeypatch, consent_mode="auto")
    rec = lib.get(res.voice_id)
    assert rec.scope == c.PRIVATE and rec.info["license"] == "custom/personal-only"
    assert any("заявлени" in w.lower() for w in res.warnings) and not rec.info["consent"]["confirmed"]


def test_asr_failure_never_fails_the_training_and_manual_none_modes(tmp_path, monkeypatch):
    class Broken(FakeASR):
        def load(self):
            raise RuntimeError("no model")
    asr_holder["asr"] = Broken([""])
    res, lib = _run(tmp_path, monkeypatch, consent_mode="auto")
    assert lib.get(res.voice_id).scope == c.PRIVATE and res.consent["recorded_statement"] is False
    (tmp_path / "m").mkdir()
    res, lib = _run(tmp_path / "m", monkeypatch, consent_mode="manual", consent_scope=c.PUBLIC_NC, consent_name="Ann")
    rec = lib.get(res.voice_id)
    assert rec.scope == c.PUBLIC_NC and rec.info["consent"]["method"] == "manual" and rec.info["consent"]["confirmed"]
    assert rec.info["license"] == "CC-BY-NC-4.0" and rec.info["author"] == "Ann"
    (tmp_path / "n").mkdir()
    res, lib = _run(tmp_path / "n", monkeypatch, consent_mode="none")
    assert lib.get(res.voice_id).info["consent"]["method"] == "none" and lib.get(res.voice_id).scope == c.PRIVATE


def test_confirm_consent_changes_scope_and_licence(tmp_path, monkeypatch):
    asr_holder["asr"] = FakeASR([c.example_statement("en", c.COMMERCIAL, "Ann", "03.10.2026")])
    res, lib = _run(tmp_path, monkeypatch, consent_mode="auto")
    rec = lib.confirm_consent(res.voice_id, c.PUBLIC_NC)
    assert rec.scope == c.PUBLIC_NC and rec.info["license"] == "CC-BY-NC-4.0" and rec.info["commercial_use"] is False
    assert rec.info["consent"]["confirmed"] and rec.info["consent"]["method"] == "spoken_confirmed"
    assert lib.get(res.voice_id).info["consent"]["scope"] == c.PUBLIC_NC           # persisted in voice.json


pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_train_window_consent_controls_and_confirmation(app, tmp_path):
    from types import SimpleNamespace
    from ui.main_window import MainWindow
    seen = []
    lib = VoiceLibrary(tmp_path / "lib")
    w = MainWindow(runner=lambda req, p, cc: seen.append(req), autocheck=False, auto_open_folder=False)
    w.library = lib
    assert w.consent_mode == "auto" and w.manual_consent.isHidden() and not w.chk_consent_clip.isHidden()
    w.cmb_consent.setCurrentIndex(1)
    assert not w.manual_consent.isHidden() and w.chk_consent_clip.isHidden()
    w.edt_consent_name.setText("Ann"); w.cmb_consent_scope.setCurrentIndex(1)
    assert w._consent_kwargs() == dict(consent_mode="manual", consent_scope="public_noncommercial", consent_name="Ann",
                                       consent_save_clip=True)
    # a finished task with an unconfirmed detection shows the result panel; one click confirms a changed scope
    d = tmp_path / "ad"; d.mkdir()
    for f in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav"):
        (d / f).write_bytes(b"x")
    rec = lib.add_from_adapter(d, voice_info.build_voice_info("V", "russian", 1, 1, "m", license="custom/personal-only",
                                                              consent=c.build_consent(None, scope=c.PRIVATE, method="spoken", recorded=True, confirmed=False)))
    result = SimpleNamespace(kind="lora", open_dir=tmp_path, adapter_path=d, n_segments=3, warnings=[], update_summary="",
                             merged_path=None, speaker="", voice_id=rec.id, consent=rec.info["consent"])
    w.on_done(result)
    assert not w.consent_result.isHidden() and "Иван" not in w.lbl_consent_result.text()
    w.cmb_consent_confirm.setCurrentIndex(c.SCOPES.index(c.COMMERCIAL))
    w.confirm_consent()
    assert w.consent_result.isHidden() and lib.get(rec.id).scope == c.COMMERCIAL and lib.get(rec.id).info["consent"]["confirmed"]


def test_voice_card_and_narrate_show_the_scope(app, tmp_path):
    from ui.narrate_window import NarrateWindow
    from ui.voices_window import VoiceCard
    lib = VoiceLibrary(tmp_path / "lib")
    d = tmp_path / "ad"; d.mkdir()
    for f in ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav"):
        (d / f).write_bytes(b"x")
    for name, sc in (("Priv", c.PRIVATE), ("Pub", c.PUBLIC_NC), ("Com", c.COMMERCIAL)):
        lib.add_from_adapter(d, voice_info.build_voice_info(name, "russian", 1, 1, "m", license=c.license_for_scope(sc),
                                                            consent=c.build_consent(None, scope=sc, method="manual", recorded=False, confirmed=True)))
    recs = {r.name: r for r in lib.list_voices()}
    assert [recs[n].scope for n in ("Priv", "Pub", "Com")] == [c.PRIVATE, c.PUBLIC_NC, c.COMMERCIAL]
    card = VoiceCard(recs["Priv"])
    assert card.scope_badge.text() and card.scope_badge.property("commercial") == "false"
    assert VoiceCard(recs["Com"]).scope_badge.property("commercial") == "true"
    n = NarrateWindow(lib)
    n.refresh_voices()
    for i in range(n.cmb_voice.count()):
        if n.cmb_voice.itemData(i) == recs["Pub"].id:
            n.cmb_voice.setCurrentIndex(i)
    assert n.badge_box.count() == 2 and "бесплатно" in n.lbl_voice_info.text()


def test_consent_texts_in_every_language_and_templates_cover_all_scopes():
    from core import i18n
    from core.i18n import tr
    for lang in ("en", "ru", "de"):
        i18n.set_language(lang, persist=False)
        for k in ("consent.title", "consent.hint_auto", "consent.scope_private_only", "consent.detected", "warn.consent_unclear",
                  "narr.done_private_reminder"):
            assert tr(k) != k
        assert set(c.TEMPLATES[lang]) == set(c.SCOPES)
