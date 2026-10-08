"""Re-voice phase 1: audio -> recognised text (fake recogniser) -> editable text -> the narrator."""
from pathlib import Path

import numpy as np
import soundfile as sf

from core import i18n, revoice
from core.asr import FakeASR
from core.book_parsers import load_book
from tests.test_studio import app, lib, make_studio, wait_for  # noqa: F401 - fixtures


def speech_like(path: Path, seconds=(3.0, 2.0)) -> Path:
    """Tone bursts separated by silence (enough for the pause splitter and the voiced check)."""
    sr = 16000
    parts = []
    for s in seconds:
        t = np.arange(int(sr * s)) / sr
        parts += [0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(int(sr * 0.6))]
    sf.write(str(path), np.concatenate(parts).astype(np.float32), sr)
    return path


def test_files_become_chapters_and_the_text_is_a_book(tmp_path):
    files = [speech_like(tmp_path / "02 - Second part.wav"), speech_like(tmp_path / "01_Intro.wav")]
    asr = FakeASR(lambda x, sr: "Hello there, this is a test.")
    chapters = revoice.transcribe_files(files, asr)
    assert [t for t, _ in chapters] == ["Second part", "Intro"] and all("Hello there" in b for _, b in chapters)
    book = load_book(revoice.save_text(revoice.to_text(chapters), tmp_path / "out", "My talk"))
    assert book.title == "My talk" and [c.title for c in book.chapters] == ["Second part", "Intro"]


def test_window_transcribes_and_hands_the_edited_text_to_the_narrator(app, lib, tmp_path):
    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    files = [str(speech_like(tmp_path / "b.wav")), str(speech_like(tmp_path / "a.wav"))]
    rv = RevoiceWindow(asr_factory=lambda: FakeASR(lambda x, sr: "Recognised words of a sentence."),
                       pick_files=lambda: files, out_dir=tmp_path / "rv")
    s = make_studio(lib, revoice=rv)
    assert "revoice" in s.pages and s.card_revoice.lbl_title.text() == "Re-voice"
    rv.add_files()
    assert [f.name for f in rv.files] == ["a.wav", "b.wav"] and not rv.btn_narrate.isEnabled()
    assert rv.transcribe()
    assert wait_for(lambda: "Recognised words" in rv.ed_text.toPlainText(), 10)
    rv.ed_text.setPlainText(rv.ed_text.toPlainText().replace("Recognised", "Edited"))
    path = rv.narrate()
    assert path.parent == tmp_path / "rv" and "Edited words" in path.read_text(encoding="utf-8")
    assert s.current_page == "narrate" and s.narrate_window.book is not None
    assert [c.title for c in s.narrate_window.book.chapters] == ["a", "b"]


def test_missing_recogniser_is_reported(app, tmp_path):
    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    rv = RevoiceWindow(asr_factory=lambda: None, pick_files=lambda: [str(speech_like(tmp_path / "a.wav"))], out_dir=tmp_path)
    rv.add_files()
    rv.transcribe()
    assert wait_for(lambda: "No speech recognition model" in rv.lbl_state.text())


def test_dictaphone_recording_is_stored_as_opus(tmp_path):
    from core import revoice

    src = tmp_path / "recording.flac"
    src.write_bytes(b"flac")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"opus")
        return type("R", (), {"returncode": 0})()

    out = revoice.to_opus(src, "ffmpeg", run=run)
    assert out == tmp_path / "recording.opus" and out.is_file() and not src.exists()
    assert "libopus" in calls[0] and "-ac" in calls[0]
    keep = tmp_path / "b.flac"
    keep.write_bytes(b"x")
    assert revoice.to_opus(keep, None) == keep and revoice.to_opus(out, "ffmpeg", run=run) == out
    failed = revoice.to_opus(keep, "ffmpeg", run=lambda cmd, **kw: type("R", (), {"returncode": 1})())
    assert failed == keep and keep.exists() and not (tmp_path / "b.opus").exists()
