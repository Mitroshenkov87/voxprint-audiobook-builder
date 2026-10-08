"""Structured pauses (punctuation / structure aware, trimmed pieces) and the adaptive reading speed (fakes only)."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from core import chunker
from core import pace as pc
from core import pauses as pz
from core.audio_utils import time_stretch, trim_silence
from core.book_parsers import Book, Chapter

GENESIS = ("1 В начале сотворил Бог небо и землю.\n2 Земля же была безвидна и пуста, и тьма над бездною, "
           "и Дух Божий носился над водою.\n3 И сказал Бог: да будет свет. И стал свет.")


def test_default_lengths_are_ordered_and_clamped():
    d = pz.PauseLengths()
    assert (d.comma, d.mid, d.sentence, d.paragraph, d.chapter) == (250, 400, 600, 1000, 2000)
    assert d.ms(pz.COMMA) < d.ms(pz.MID) < d.ms(pz.SENTENCE) < d.ms(pz.ELLIPSIS) < d.ms(pz.PARAGRAPH) < d.ms(pz.CHAPTER)
    assert d.ms(pz.DASH) == d.mid and d.ms(pz.TITLE) == d.ms(pz.SCENE) == d.chapter
    assert pz.PauseLengths(comma=-5, chapter=99999).to_dict()["comma"] == 0
    assert pz.PauseLengths.from_dict({"sentence": "x", "mid": 700}).to_dict() == {**pz.DEFAULT_LENGTHS_MS, "mid": 700}
    pz.save_lengths(pz.PauseLengths(sentence=800))
    assert pz.load_lengths().sentence == 800
    assert pz.PauseProfile(2, lengths=pz.PauseLengths(sentence=800)).ms(pz.SENTENCE) == 800


def test_structured_chunks_follow_punctuation_and_structure():
    text = ("Он долго шёл по пустой дороге через поле, но никто не встретился ему на пути. "
            "Дома его ждали трое: мать, отец и сестра, которых он не видел много лет.\n\n* * *\n\nКонец.")
    pieces = chunker.chunk_text_structured(text)
    kinds = [k for _, k in pieces]
    assert pieces[0] == ("Он долго шёл по пустой дороге через поле,", pz.MID)      # comma + "но" = strong break
    assert kinds[1] == pz.SENTENCE and pz.SCENE in kinds and kinds[-1] == pz.PARAGRAPH
    assert " ".join(t for t, _ in pieces).replace("  ", " ").count("сестра") == 1
    # plain commas stay inside a chunk, short sides are never cut off
    assert len(chunker.chunk_text_structured("Да, конечно: это так.")) == 1
    assert chunker.chunk_text_structured("He waited for a very long time, and nobody came to the old door.")[0][1] == pz.MID


def test_numbered_verse_lines_get_a_paragraph_pause():
    pieces = chunker.chunk_text_structured(GENESIS)
    ends = [t for t, k in pieces if k == pz.PARAGRAPH]
    assert len(ends) == 3 and ends[0].startswith("1 В начале")
    assert chunker.blocks("a line\nanother line") == ["a line\nanother line"]       # plain line breaks are not verses


def test_chunk_book_uses_lengths_and_tempo():
    book = Book("Бытие", chapters=[Chapter("Глава 1", GENESIS)])
    lengths = pz.PauseLengths(paragraph=1234, chapter=2345)
    chunks = chunker.chunk_book(book, speak_titles=True, lengths=lengths, pace=pc.Pace(1.0, pc.AUTO))
    assert chunks[0].pause_ms == 2345 and chunks[0].pause_kind == pz.TITLE
    assert any(c.pause_ms == 1234 for c in chunks)
    assert all(c.tempo < 1.0 for c in chunks)                              # scripture detected: read slower
    legacy = chunker.chunk_book(book)
    assert all(c.tempo == 1.0 for c in legacy) and legacy[0].pause_kind == ""


def test_explicit_mode_gives_colon_and_conjunction_a_mid_pause():
    kinds = dict(chunker.chunk_text_pauses("He opened the heavy door slowly, but the room was empty: nobody was there."))
    assert kinds["He opened the heavy door slowly,"] == pz.MID
    assert kinds["but the room was empty:"] == pz.MID


def test_segment_tempo_heuristics():
    long_desc = ("Огромные, величественные, безмолвные горы возвышались над бесконечной, туманной, просыпающейся долиной, "
                 "освещённой первыми холодными лучами восходящего солнца и окутанной прозрачной дымкой утреннего тумана.")
    assert pc.segment_tempo("— Ну, пошли!", pc.FICTION) == 1.0
    assert pc.segment_tempo("Он ушёл.", pc.FICTION) == 1.0
    assert pc.segment_tempo(long_desc, pc.FICTION) < 0.95
    assert pc.segment_tempo("Он ушёл.", pc.SCRIPTURE) < 1.0
    assert pc.segment_tempo("Он ушёл.", pc.FICTION, speed=1.2) == 1.2
    assert pc.MIN_TEMPO <= pc.segment_tempo(long_desc, pc.SCRIPTURE, 0.7) <= 1.0
    assert pc.Pace(9, "nonsense").speed == pc.MAX_SPEED and pc.Pace(9, "nonsense").style == pc.AUTO
    dialogue = Book("x", chapters=[Chapter("1", "\n\n".join(["— Привет, — сказал он.", "— Здравствуй.", "Они помолчали."] * 5))])
    assert pc.book_style(dialogue) == pc.DIALOGUE and pc.book_style(dialogue, pc.FICTION) == pc.FICTION
    assert pc.book_style(Book("x", chapters=[Chapter("1", "Обычный рассказ о жизни.\n\nЕщё абзац.")])) == pc.FICTION
    pc.save(pc.Pace(0.9, pc.SCRIPTURE))
    assert pc.load() == pc.Pace(0.9, pc.SCRIPTURE)


def test_trim_and_stretch_keep_pitch():
    sr = 24000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    padded = np.concatenate([np.zeros(sr // 2, np.float32), tone, np.zeros(sr // 4, np.float32)])
    trimmed = trim_silence(padded, sr)
    assert abs(len(trimmed) - len(tone)) <= int(0.07 * sr)
    assert trim_silence(np.zeros(1000, np.float32), sr).size == 1000             # all quiet: unchanged
    for rate in (0.85, 1.15):
        y = time_stretch(tone, sr, rate)
        assert abs(len(y) - len(tone) / rate) <= 2
        spec = np.abs(np.fft.rfft(y[2000:-2000]))
        assert abs(np.argmax(spec) * sr / len(y[2000:-2000]) - 200) < 5          # same pitch
    assert time_stretch(tone, sr, 1.0) is tone


def test_assembled_chapter_has_the_configured_silence(tmp_path):
    import soundfile as sf

    from core import narration as nr

    sr = 24000
    voice = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr)).astype(np.float32)
    piece = np.concatenate([np.zeros(sr // 3, np.float32), voice, np.zeros(sr // 3, np.float32)])   # model's own silence
    book = Book("b", chapters=[Chapter("c", "First sentence here. Second sentence here.")])
    lengths = pz.PauseLengths(sentence=600, paragraph=600, chapter=1000)
    chunks = chunker.chunk_book(book, lengths=lengths)

    class Cache:
        def key(self, tag, text):
            return text

        def load(self, key):
            return piece, sr
    out = nr.assemble_chapter(book, 0, 0, chunks, "tag", Cache(), {c.index: c.text for c in chunks}, tmp_path,
                              lengths=lengths, shape=True)
    data, _ = sf.read(str(out.wav))
    quiet = np.abs(data) < 0.01
    gaps, run = [], 0
    for q in quiet:
        if q:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    long_gaps = [g / sr for g in gaps if g > sr * 0.2]
    assert long_gaps and abs(long_gaps[0] - 0.66) < 0.08                       # 0.6 s + the 30 ms margins, not 0.6 + 0.67


def test_cli_flags_override_settings(monkeypatch):
    import cli

    pz.save_lengths(pz.PauseLengths(comma=300))
    pc.save(pc.Pace(0.9, pc.FICTION))
    ns = argparse.Namespace(pause_comma=None, pause_mid=None, pause_sentence=0.8, pause_paragraph=None, pause_chapter=None,
                            speed=None, style=pc.SCRIPTURE)
    lengths, pace = cli.narration_shaping(ns)
    assert lengths.comma == 300 and lengths.sentence == 800 and pace == pc.Pace(0.9, pc.SCRIPTURE)
    ap = cli.build_parser() if hasattr(cli, "build_parser") else None
    if ap is not None:
        a = ap.parse_args(["narrate", "b.txt", "--voice", "v", "--out", "o", "--pause-mid", "0.5", "--speed", "1.1"])
        assert a.pause_mid == 0.5 and a.speed == 1.1
    with pytest.raises(cli.CliError):
        cli.narration_shaping(argparse.Namespace(**{**vars(ns), "speed": 3.0}))


def test_settings_dialog_saves_pauses_and_speed(tmp_path):
    pytest.importorskip("PySide6")
    from core import i18n
    from core.voice_library import VoiceLibrary
    from tests.test_studio import make_studio
    from PySide6.QtWidgets import QApplication
    from ui.settings_dialog import SettingsDialog

    _ = QApplication.instance() or QApplication([])
    i18n.set_language("ru")
    d = SettingsDialog(make_studio(VoiceLibrary(tmp_path / "voices")))
    assert d.lbl_pause["mid"].text() == i18n.tr("narrset.mid") and d.spn_pause["comma"].value() == pytest.approx(0.25)
    d.spn_pause["sentence"].setValue(0.9)
    d.sld_speed.setValue(85)
    d.cmb_style.setCurrentIndex(d.cmb_style.findData(pc.SCRIPTURE))
    assert pz.load_lengths().sentence == 900 and pc.load() == pc.Pace(0.85, pc.SCRIPTURE)
    assert d.lbl_speed_value.text() == "85 %"
    d.reset_narration()
    assert pz.load_lengths() == pz.PauseLengths() and pc.load() == pc.Pace()
    i18n.set_language("en")
