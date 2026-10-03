"""No-transcript mode: plausibility score, pause splitting, level normalisation, the merged dataset from many files, the Train window flow."""
import json
import os
import time

import numpy as np
import pytest

from core import audio_utils as au
from core.asr import AsrResult, FakeASR, plausibility, split_at_pauses
from core.asr_dataset import AsrConfig, build_from_audio, expand_inputs, normalize_level
from core.dataset_builder import read_metadata_jsonl
from core.errors import AlignmentError, AudioReadError
from core.events import CancelToken
from core.i18n import tr
from tests.synth import make_text, synth_reading


def _clip(tmp, name, n_sent, seed):
    audio, _ = synth_reading(make_text(n_sent, seed), seed=seed)
    au.write_wav(tmp / name, audio, 16000)
    return len(audio) / 16000


VOCAB = ("дом река лес окно ветер город дорога книга голос свет тихий старый новый далёкий большой маленький тёплый "
         "быстро медленно вчера сегодня говорит читает слышит видит идёт ждёт помнит любит знает пишет").split()


def _speech_like(_unused, x16, sr):
    """Fake recogniser: ~2.5 distinct words per second of audio; the words depend on the audio, so only a copy repeats."""
    rnd = np.random.default_rng(int(np.abs(x16[:2000]).sum() * 1e6) % (2 ** 32))
    return " ".join(rnd.choice(VOCAB, size=max(1, int(len(x16) / sr * 2.5)), replace=True))


def test_plausibility_accepts_speech_and_rejects_typical_asr_failures():
    assert plausibility("Это обычная фраза, которую произнёс диктор.", 3.0, "Russian") > 0.8
    assert plausibility("", 3.0) == 0.0
    assert plausibility("да", 6.0) < 0.5                                           # far too little text for the clip
    assert plausibility("а " * 80, 5.0) < 0.5                                       # looping hallucination / too dense
    assert plausibility("слово слово слово слово слово слово", 2.5) < 0.5           # repetition run
    assert plausibility("Hello there, this is a sentence.", 3.0, "Russian") < 0.5   # wrong script for the language
    assert plausibility("###### $$$$ @@@@ ####", 3.0) < 0.5


def test_split_at_pauses_cuts_long_audio_into_pieces_at_pauses():
    audio, _ = synth_reading(make_text(60, 3), seed=3)
    parts = split_at_pauses(audio, 16000, max_s=14.0, min_s=3.0)
    assert len(parts) >= 3 and parts[0][0] == 0 and parts[-1][1] == len(audio)
    assert all(b > a for a, b in parts) and all((b - a) / 16000 <= 14.01 for a, b in parts)
    assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))      # gap-free
    assert split_at_pauses(audio[:16000 * 5], 16000) == [(0, 16000 * 5)]


def test_normalize_level_targets_rms_without_clipping():
    quiet = (np.random.default_rng(0).normal(0, 0.01, 24000)).astype(np.float32)
    loud = (np.random.default_rng(1).normal(0, 0.3, 24000)).astype(np.float32)
    for x in (quiet, loud):
        y = normalize_level(x)
        assert np.max(np.abs(y)) <= 0.951
    assert abs(20 * np.log10(np.sqrt(np.mean(normalize_level(quiet) ** 2))) - (20 * np.log10(0.01) + 12)) < 0.5   # +12 dB cap
    assert np.sqrt(np.mean(normalize_level(loud) ** 2)) < np.sqrt(np.mean(loud ** 2)) or np.max(np.abs(loud)) < 0.95


def test_many_files_become_one_dataset_with_totals_and_gates(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    for i in range(6):
        _clip(d, f"c{i}.wav", 2, seed=10 + i)
    (d / "dup.wav").write_bytes((d / "c0.wav").read_bytes())              # the same clip twice
    au.write_wav(d / "tiny.wav", np.zeros(8000, dtype=np.float32) + 0.01, 16000)   # 0.5 s
    _clip(d, "long.wav", 60, seed=5)                                      # a recording that must be sliced
    texts = iter(["а а а а а а а а а а а а"])                             # first call: unplausible loop
    def asr(x, sr):
        try:
            return next(texts)
        except StopIteration:
            return _speech_like(None, x, sr)
    fake = FakeASR(asr)
    stages = []
    res = build_from_audio([d], tmp_path / "ds", fake, AsrConfig(language="Russian"), lambda s, f, m: stages.append(s))
    rep = res.asr_report
    assert rep.files == 9 and rep.files_failed == 0
    assert rep.dropped.get("too_short") == 1 and rep.dropped.get("duplicate") == 1 and rep.dropped.get("low_confidence") == 1
    assert rep.kept == res.n_segments >= 8                                # long file gave several pieces
    assert 0.5 < rep.share_kept < 1.0 and rep.seconds_total > rep.seconds_kept
    rows = read_metadata_jsonl(tmp_path / "ds" / "metadata.jsonl")
    assert len(rows) == rep.kept and (tmp_path / "ds" / "ref.wav").exists() and (tmp_path / "ds" / "ref_text.txt").exists()
    report = json.loads((tmp_path / "ds" / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "no_transcript" and report["asr"]["kept"] == rep.kept
    assert all(s["source"] and 0 <= s["confidence"] <= 1 for s in report["segment_times"])
    assert any("распознан" in w for w in res.warnings)                    # the "not checked by a human" note is always there
    x, sr = au.load_audio(tmp_path / "ds" / "segment_001.wav", 24000)
    assert sr == 24000 and np.max(np.abs(x)) <= 0.96


def test_errors_for_empty_input_and_nothing_usable(tmp_path):
    with pytest.raises(AudioReadError):
        build_from_audio([tmp_path], tmp_path / "ds", FakeASR(["x"]))
    _clip(tmp_path, "a.wav", 2, seed=1)
    with pytest.raises(AlignmentError):
        build_from_audio([tmp_path / "a.wav"], tmp_path / "ds", FakeASR([""]))   # recogniser heard nothing


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    _clip(tmp_path, "ok.wav", 3, seed=2)
    res = build_from_audio([tmp_path], tmp_path / "ds", FakeASR(lambda x, sr: _speech_like(None, x, sr)))
    assert res.asr_report.files_failed == 1 and res.n_segments >= 1


def test_cancel_stops_between_files(tmp_path):
    from core.errors import CancelledByUser
    for i in range(3):
        _clip(tmp_path, f"c{i}.wav", 2, seed=i)
    tok = CancelToken()
    asr = FakeASR(lambda x, sr: (tok.cancel(), "слово слово слово слово")[1])
    with pytest.raises(CancelledByUser):
        build_from_audio([tmp_path], tmp_path / "ds", asr, cancel=tok)


def test_expand_inputs_folders_recursive_sorted_unique(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ("b.wav", "a.mp3", "sub/c.flac", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    got = expand_inputs([tmp_path, tmp_path / "a.mp3"])
    assert [p.name for p in got] == ["a.mp3", "b.wav", "c.flac"]


def test_run_task_uses_the_asr_instead_of_the_aligner(tmp_path):
    from workers.pipeline_runner import KIND_DATASET, TaskRequest, run_task
    for i in range(4):
        _clip(tmp_path, f"c{i}.wav", 3, seed=20 + i)
    def boom():
        raise AssertionError("the aligner must not be used")
    class NoUpdates:
        def should_autocheck(self): return False
    res = run_task(TaskRequest(kind=KIND_DATASET, no_transcript=True, audio_files=[tmp_path], out_root=tmp_path / "out"),
                   updater=NoUpdates(), aligner_factory=boom,
                   asr_factory=lambda: FakeASR(lambda x, sr: _speech_like(None, x, sr)))
    assert res.n_segments >= 4 and res.asr_report.kept == res.n_segments


# ----------------------------------------------------------------------------- Train window
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_train_window_no_transcript_flow(app, tmp_path):
    from ui.main_window import MainWindow
    seen = []
    w = MainWindow(runner=lambda req, p, c: seen.append(req), autocheck=False, auto_open_folder=False)
    assert not w.no_transcript and w.asr_box.isHidden() and w.btn_text.isVisibleTo(w)
    w.chk_no_text.setChecked(True)
    assert w.no_transcript and not w.asr_box.isHidden() and not w.btn_text.isVisibleTo(w)
    assert "ошибаться" in w.lbl_asr_warning.text()
    for f in ("a.wav", "b.mp3"):
        (tmp_path / f).write_bytes(b"x")
    (tmp_path / "t.txt").write_bytes(b"x")
    w.set_audio_files([tmp_path])
    assert "2" in w.lbl_audio.text()
    assert not w.btn_lora.isEnabled()                        # files alone are not enough: explicit opt-in is required
    w.chk_asr_ok.setChecked(True)
    assert w.btn_lora.isEnabled() and w.btn_dataset.isEnabled()
    w.chk_no_text.setChecked(False)                          # leaving the mode forgets the opt-in
    w.chk_no_text.setChecked(True)
    assert not w.chk_asr_ok.isChecked() and not w.btn_lora.isEnabled()
    w.chk_asr_ok.setChecked(True)
    w.start("dataset")
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        QApplication.processEvents(); time.sleep(0.01)
    assert seen and seen[0].no_transcript and seen[0].audio_files == [tmp_path] and seen[0].text is None
    w.worker.wait(3000)


def test_asr_texts_exist_in_every_language():
    from core import i18n
    for lang in ("en", "ru", "de"):
        i18n.set_language(lang, persist=False)
        for k in ("asr.checkbox", "asr.warning", "asr.confirm", "asr.report", "err.asr_nothing_kept", "warn.asr_unverified"):
            assert tr(k) != k
    i18n.set_language("en", persist=False)
    assert "mistakes" in tr("asr.warning") and "{kept}" not in tr("warn.asr_unverified", kept=3, files=2)
