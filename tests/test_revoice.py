"""Re-voice: audio -> text (fake recogniser) -> the narrator, and direct conversion (fake converter, never a real model)."""
from pathlib import Path

import numpy as np
import soundfile as sf

from core import i18n, revoice, voice_convert
from core.asr import FakeASR
from core.book_parsers import load_book
from tests.test_studio import add_voice, app, lib, make_studio, wait_for  # noqa: F401 - fixtures


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
    from PySide6.QtWidgets import QApplication

    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    voice = add_voice(lib, tmp_path, name="Anna")
    files = [str(speech_like(tmp_path / "b.wav")), str(speech_like(tmp_path / "a.wav"))]
    rv = RevoiceWindow(asr_factory=lambda: FakeASR(lambda x, sr: "Recognised words of a sentence."),
                       pick_files=lambda: files, out_dir=tmp_path / "rv", library=lib,
                       vc_status=lambda: "needs_download")
    s = make_studio(lib, revoice=rv)
    assert "revoice" in s.pages and s.card_revoice.lbl_title.text() == "Re-voice"
    rv.show()
    QApplication.processEvents()
    left, right = rv.btn_to_text.mapTo(rv, rv.btn_to_text.rect().topLeft()), rv.btn_direct.mapTo(rv, rv.btn_direct.rect().topLeft())
    assert rv.btn_to_text.text() == "Convert to text" and rv.btn_direct.text() == "Re-voice recording"
    assert rv.lbl_to_text.text() and rv.lbl_direct.text() and abs(left.y() - right.y()) <= 4 and left.x() < right.x()
    assert not rv.btn_advanced.isChecked() and not rv.advanced_box.isVisible()
    assert rv.selected_voice_id() == voice.id and not rv.btn_direct.isEnabled() and rv.btn_vc_download.isVisible()
    rv.add_files()
    assert [f.name for f in rv.files] == ["a.wav", "b.wav"] and not rv.btn_narrate.isEnabled()
    assert rv.transcribe()
    assert wait_for(lambda: "Recognised words" in rv.ed_text.toPlainText(), 10)
    rv.ed_text.setPlainText(rv.ed_text.toPlainText().replace("Recognised", "Edited"))
    path = rv.narrate()
    assert path is not None and path.parent == tmp_path / "rv" and "Edited words" in path.read_text(encoding="utf-8")
    assert s.current_page == "narrate" and s.narrate_window.book is not None
    assert s.narrate_window.selected_voice_id() == voice.id
    assert [c.title for c in s.narrate_window.book.chapters] == ["a", "b"]
    rv.hide()
    s.shutdown()


def test_direct_revoice_uses_the_fake_converter_and_the_voice_reference(app, lib, tmp_path, monkeypatch):
    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    monkeypatch.setattr("core.audio_utils.ensure_ffmpeg", lambda: None)
    voice = add_voice(lib, tmp_path, name="Anna")
    speech_like(voice.preview_path, seconds=(0.8,))
    src = speech_like(tmp_path / "talk.wav", seconds=(1.2,))
    made = []

    def factory():
        made.append(voice_convert.FakeVoiceConverter())
        return made[-1]

    from tests.test_studio import FakePreviewer
    rv = RevoiceWindow(asr_factory=lambda: None, pick_files=lambda: [str(src)], out_dir=tmp_path / "rv", library=lib,
                       vc_status=lambda: "ready", vc_factory=factory, previewer=FakePreviewer())
    rv.add_files()
    assert rv.btn_direct.isEnabled() and not rv.btn_vc_download.isVisible()
    assert rv.convert_direct()
    assert wait_for(lambda: rv.output is not None and rv.output.is_file() and not rv.busy, 10), rv.lbl_state.text()
    y, sr = sf.read(str(rv.output), dtype="float32")
    x, xsr = sf.read(str(src), dtype="float32")
    assert sr == xsr and len(y) == len(x) and made and made[0].unloaded and made[0].calls[0][2] > 0
    assert float(np.max(np.abs(y))) < float(np.max(np.abs(x)))
    rv.play_selected()
    assert rv.previewer.played[-1] == rv.output
    rv.shutdown()


def test_download_model_is_explicit_and_hides_the_download_button(app, lib, tmp_path):
    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    voice = add_voice(lib, tmp_path, name="Anna")
    speech_like(voice.preview_path, seconds=(0.5,))
    state = {"ready": False}
    calls = []

    def ensure(progress=None, **kw):
        calls.append(1)
        if progress:
            progress(1.0, "done")
        state["ready"] = True

    rv = RevoiceWindow(asr_factory=lambda: None, pick_files=lambda: [str(speech_like(tmp_path / "talk.wav"))],
                       out_dir=tmp_path, library=lib, vc_status=lambda: "ready" if state["ready"] else "needs_download",
                       vc_ensure=ensure, vc_factory=lambda: (_ for _ in ()).throw(AssertionError("no conversion")))
    rv.show()
    app.processEvents()
    rv.add_files()
    # the small model is standard: the direct button is NOT grey while it is missing (a click fetches it first)
    assert rv.btn_direct.isEnabled() and rv.btn_vc_download.isVisible() and "131" in rv.btn_vc_download.text()
    assert calls == [] and rv.download_model()
    assert wait_for(lambda: state["ready"] and not rv.busy and rv.btn_direct.isEnabled(), 10), rv.lbl_state.text()
    assert calls == [1] and not rv.btn_vc_download.isVisible()
    rv.shutdown()


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


def test_direct_click_without_the_model_downloads_it_then_converts(app, lib, tmp_path, monkeypatch):
    from ui.revoice_window import RevoiceWindow

    i18n.set_language("en")
    monkeypatch.setattr("core.audio_utils.ensure_ffmpeg", lambda: None)
    voice = add_voice(lib, tmp_path, name="Anna")
    speech_like(voice.preview_path, seconds=(0.8,))
    src = speech_like(tmp_path / "talk.wav", seconds=(1.2,))
    state, made = {"ready": False}, []

    def ensure(progress=None, **kw):
        state["ready"] = True

    def factory():
        made.append(voice_convert.FakeVoiceConverter())
        return made[-1]

    rv = RevoiceWindow(asr_factory=lambda: None, pick_files=lambda: [str(src)], out_dir=tmp_path / "rv", library=lib,
                       vc_status=lambda: "ready" if state["ready"] else "needs_download", vc_ensure=ensure,
                       vc_factory=factory)
    rv.add_files()
    assert rv.btn_direct.isEnabled()
    rv.btn_direct.click()
    assert wait_for(lambda: rv.output is not None and rv.output.is_file() and not rv.busy, 10), rv.lbl_state.text()
    assert state["ready"] and made
    rv.shutdown()
