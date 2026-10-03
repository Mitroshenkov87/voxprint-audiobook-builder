from pathlib import Path
from types import SimpleNamespace

from infra import paths
from workers import selftest_narrate


class Lib:
    def __init__(self, voices):
        self.voices = voices

    def list_voices(self):
        return self.voices


def voice(i="v1", lang="Russian"):
    return SimpleNamespace(id=i, name=i, language=lang, path=Path("."), info={})


def test_ok_writes_log_and_returns_0(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    seen = {}

    def narrate(job, progress, cancel, pause):
        seen["text"] = job.book.chapters[0].text
        seen["voice"] = job.voice.id
        return SimpleNamespace(files=[f], seconds=3.2)

    assert selftest_narrate.run(library=Lib([voice("a"), voice("b")]), narrate=narrate, out_dir=tmp_path / "o") == 0
    assert seen["voice"] == "a" and "проверка" in seen["text"].lower()
    assert (tmp_path / "selftest_narrate.txt").read_text(encoding="utf-8").startswith("OK")


def test_named_voice_and_language(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    got = {}

    def narrate(job, *a):
        got["v"], got["t"] = job.voice.id, job.book.chapters[0].text
        return SimpleNamespace(files=[f], seconds=1)

    assert selftest_narrate.run("b", library=Lib([voice("a"), voice("b", "German")]), narrate=narrate, out_dir=tmp_path) == 0
    assert got["v"] == "b" and got["t"].startswith("Dies ist")


def test_no_voice_is_2_and_failures_are_1(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    assert selftest_narrate.run(library=Lib([]), out_dir=tmp_path) == 2
    assert "NO_VOICE" in (tmp_path / "selftest_narrate.txt").read_text(encoding="utf-8")
    assert selftest_narrate.run("zzz", library=Lib([voice()]), out_dir=tmp_path) == 2

    def boom(*a):
        raise RuntimeError("no gpu")
    assert selftest_narrate.run(library=Lib([voice()]), narrate=boom, out_dir=tmp_path) == 1
    assert "FAIL: RuntimeError" in (tmp_path / "selftest_narrate.txt").read_text(encoding="utf-8")
    assert selftest_narrate.run(library=Lib([voice()]), narrate=lambda *a: SimpleNamespace(files=[], seconds=0), out_dir=tmp_path) == 1
