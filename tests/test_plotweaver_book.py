"""One-command narration of a Plotweaver book: Markdown chapters, explicit yo, stress, speaker marks.

No GPU and no model. A fake engine records the text each voice would have spoken.
"""
import json
from pathlib import Path

import cli as user_cli
from core import languages
from core import narration as nr
from core import voice_info
from core.book_parsers import load_book
from core.voice_library import VoiceLibrary
from tests.test_narration import FakeEngine, FakeFfmpeg

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "plotweaver"
BOOK = FIXTURE / "book.md"
MARKS = FIXTURE / "marks.txt"


def _json_lines(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _add_voice(lib: VoiceLibrary, folder: Path, name: str, voice_id: str, voice_type: str) -> None:
    src = folder / f"_src_{voice_id}"
    src.mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "ref_sample.wav").write_bytes(b"RIFFxxxx")
    (src / "training_meta.json").write_text(json.dumps({"ref_sample_text": "hello"}), encoding="utf-8")
    voice_info.write_voice_json(src, voice_info.build_voice_info(
        name, "ru", 60, 3, "Qwen/Base", license="CC-BY-4.0", voice_type=voice_type))
    rec = lib.add_from_adapter(src, name=name)
    if rec.id != voice_id and not (lib.root / voice_id).exists():
        rec.path.rename(lib.root / voice_id)
        info = voice_info.read_voice_json(lib.root / voice_id) or {}
        info["id"] = voice_id
        voice_info.write_voice_json(lib.root / voice_id, info)


def _cast(tmp_path: Path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "lib")
    _add_voice(lib, tmp_path, "Narrator", "narrator", "female")
    _add_voice(lib, tmp_path, "Tom", "tom", "male")
    _add_voice(lib, tmp_path, "Ann", "ann", "female")
    return lib


def _narrate(engines: dict):
    """The real ``narrate_book`` path, with the language the CLI would pick from the book."""

    def run(job, progress, cancel, pause):
        narr = FakeEngine()
        narr.tag = "narr"
        engines["narr"] = narr
        extra = {}
        for vid in job.extra_voices:
            eng = FakeEngine()
            eng.tag = vid
            engines[vid] = eng
            extra[vid] = (lambda e=eng: e, vid)
        lang = languages.narration_language(job.book, job.voice.language)
        return nr.narrate_book(
            job.book, lambda: narr, "narr", job.out_dir,
            language=lang, narrator=job.voice.name, options=job.options,
            progress=progress, cancel=cancel, pause=pause,
            ffmpeg="ffmpeg", run=FakeFfmpeg(), extra_engines=extra or None,
        )

    return run


def test_plotweaver_markdown_narrates_chapters_voices_yo_and_stress(tmp_path, capsys):
    book = load_book(BOOK)
    assert [c.title for c in book.chapters] == ["Глава первая", "Глава вторая"]
    assert book.chapters[0].text.count("\n\n") == 1
    assert "ёще" in book.chapters[0].text and "све\u0301т" in book.chapters[0].text
    assert "ещё\u0301" in book.chapters[0].text
    assert "еще" in book.chapters[1].text

    engines = {}
    code = user_cli.main(
        ["narrate", str(BOOK), "--voice", "narrator",
         "--male-voice", "tom", "--female-voice", "ann",
         "--character", "Ivan=tom", "--character", "Anna=ann",
         "--speaker-marks", str(MARKS),
         "--out", str(tmp_path / "audiobooks"), "--format", "wav", "--json"],
        run_narration_fn=_narrate(engines), library=_cast(tmp_path),
    )
    err = capsys.readouterr()
    assert code == 0, err.err
    data = _json_lines(err.out)[-1]
    assert data["ok"] and data["warnings"] == []
    names = [Path(p).name for p in data["outputs"]]
    assert "01 - Глава первая.wav" in names
    assert "02 - Глава вторая.wav" in names

    narr = " ".join(engines["narr"].calls)
    ivan = " ".join(engines["tom"].calls)
    anna = " ".join(engines["ann"].calls)
    assert "ёще" in narr and "ещё" in narr and "еще" not in narr
    assert "све\u0301т" in narr
    # ru-normalizr capitalizes after a period ("Мед. Училище"); the builtin pass does not.
    assert "мед. училище" in narr.casefold()
    assert "ещё\u0301" in ivan and "ёще" not in ivan
    assert "Привет" in anna and "Привет" not in narr
    assert "ещё\u0301" not in narr
