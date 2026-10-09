"""``voxprint prepare``, ``translate`` and ``settings``: fake models only, no download, no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import cli as user_cli
from core.llm_text import LLMPlan
from tests.test_llm_text import FakeModel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "yo"


def _json_lines(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


def test_prepare_restores_yo_and_writes_a_report(tmp_path, capsys):
    out = tmp_path / "out" / "prepared.txt"
    code = user_cli.main(["prepare", str(FIXTURES / "dialog-ai-torah-01.txt"), "--out", str(out),
                          "--steps", "yo", "--no-typos", "--json"])
    assert code == 0, capsys.readouterr().err
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["ok"] and data["command"] == "prepare"
    assert out.read_text(encoding="utf-8") == (FIXTURES / "dialog-ai-torah-01.yo-reference.txt").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "out" / "prepared.prep_report.json").read_text(encoding="utf-8"))
    assert report["rules"] == ["yo"] and report["rule_counts"] == {"yo": 22} and report["language"] == "ru"
    assert data["report"]["rule_counts"] == {"yo": 22}


def test_prepare_default_steps_typo_model_and_llm(tmp_path, capsys):
    book = tmp_path / "book.txt"
    book.write_text("Глава 1\n\nОн все понял и пришел в 1999 г.\n", encoding="utf-8")
    out = tmp_path / "p.txt"
    report = tmp_path / "r.json"
    code = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(book), "--out", str(out), "--report", str(report)]),
        typo_state_fn=lambda lang: (type("M", (), {"key": "sage-ru"})(), True),
        cleanup_factory=lambda lang: None)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "всё понял" in text and "пришёл" in text and "1999" not in text
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert "yo" in rep["rules"] and "numbers" in rep["rules"] and rep["typo_model"] == "sage-ru"
    capsys.readouterr()

    missing = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(book), "--out", str(out), "--typos", "--json"]),
        typo_state_fn=lambda lang: (type("M", (), {"key": "sage-ru"})(), False))
    assert missing == 4
    capsys.readouterr()

    llm = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(book), "--out", str(out), "--llm", "--no-typos", "--json"]),
        plan_fn=lambda: LLMPlan(lambda: FakeModel(lambda p: p.rsplit("\n", 1)[-1]), "fake-llm"))
    assert llm == 0
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["report"]["llm"] == "fake-llm"
    no_llm = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(book), "--out", str(out), "--llm", "--no-typos"]),
        plan_fn=lambda: None)
    assert no_llm == 4
    assert user_cli.main(["prepare", str(tmp_path / "missing.txt"), "--out", str(out)]) == 3
    assert user_cli.main(["prepare", str(book), "--out", str(out), "--steps", "nope"]) == 2


def test_translate_uses_the_offline_engine_and_checks_the_models(tmp_path, capsys):
    book = tmp_path / "book.txt"
    book.write_text("Chapter 1\n\nThe dog sat on the mat. It was a good dog and the day was warm.\n", encoding="utf-8")

    class Engine:
        tag = "fake-opus"

        def translate(self, batch):
            return [f"[ru] {t}" for t in batch]

        def close(self):
            pass

    out = tmp_path / "book.ru.txt"
    args = user_cli.build_parser().parse_args(["translate", str(book), "--to", "ru", "--out", str(out), "--json"])
    code = user_cli.cmd_translate(args, factory=lambda s, t: Engine())
    assert code == 0, capsys.readouterr().err
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["source"] == "en" and data["target"] == "ru" and data["model"] == "Opus-MT"
    assert "[ru] The dog sat on the mat." in out.read_text(encoding="utf-8")

    missing = user_cli.cmd_translate(args, missing_fn=lambda s, t: ["opus-big-en-ru"])
    assert missing == 4
    assert "opus-big-en-ru" in _json_lines(capsys.readouterr().out)[-1]["hint"]
    same = user_cli.build_parser().parse_args(["translate", str(book), "--from", "en", "--to", "en", "--out", str(out)])
    assert user_cli.cmd_translate(same, factory=lambda s, t: Engine()) == 2
    capsys.readouterr()


def test_settings_list_get_set(capsys):
    assert user_cli.main(["settings", "list", "--json"]) == 0
    values = _json_lines(capsys.readouterr().out)[-1]["settings"]
    assert values["narration.ordinals"] is True and values["narration.pause.sentence"] == 0.6
    assert set(values) == set(user_cli.SETTINGS_HELP)
    assert user_cli.main(["settings", "set", "narration.ordinals", "off"]) == 0
    assert user_cli.main(["settings", "set", "narration.pause.sentence", "0.8"]) == 0
    assert user_cli.main(["settings", "set", "narration.speed", "1.2"]) == 0
    assert user_cli.main(["settings", "set", "narration.style", "scripture"]) == 0
    assert user_cli.main(["settings", "set", "narration.pauses", "on"]) == 0
    capsys.readouterr()
    assert user_cli.main(["settings", "get", "narration.pause.sentence", "--json"]) == 0
    assert _json_lines(capsys.readouterr().out)[-1]["value"] == 0.8
    from core import ordinals, pace, pauses

    assert ordinals.load_enabled() is False and pauses.load_enabled() is True
    assert pace.load().speed == 1.2 and pace.load().style == "scripture"
    assert user_cli.main(["settings", "set", "narration.speed", "5"]) == 2
    assert user_cli.main(["settings", "set", "narration.ordinals", "maybe"]) == 2
    assert user_cli.main(["settings", "set", "models.folder", "/x"]) == 2
    assert user_cli.main(["settings", "get", "no.such"]) == 2
    assert user_cli.is_user_cli(["Voxprint.exe", "settings", "list"])
    assert user_cli.is_user_cli(["Voxprint.exe", "prepare", "b.txt", "--out", "p.txt"])
    assert user_cli.is_user_cli(["Voxprint.exe", "translate", "b.txt", "--to", "ru", "--out", "p.txt"])
    capsys.readouterr()


class _CorruptingCleaner:
    """A typo model that would break the yo-only output: doubles the yo of "стерёг" and inserts a comma."""
    tag = "corrupting@1"

    def __init__(self):
        self.calls = 0

    def correct(self, texts):
        self.calls += 1
        return [t.replace("стерёг", "стёрёг").replace("бодает отвечает", "бодает, отвечает") for t in texts]

    def close(self):
        pass


def test_prepare_steps_yo_skips_the_typo_model_unless_asked(tmp_path, capsys):
    """``--steps yo`` changes nothing but the letter yo even when the typo model is downloaded (701 bug)."""
    src = FIXTURES / "dialog-ai-torah-01.txt"
    ref = (FIXTURES / "dialog-ai-torah-01.yo-reference.txt").read_text(encoding="utf-8")
    ready = lambda lang: (type("M", (), {"key": "sage-ru"})(), True)
    cleaners = []

    def factory(lang):
        cleaners.append(_CorruptingCleaner())
        return cleaners[-1]

    out = tmp_path / "yo.txt"
    code = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(src), "--out", str(out), "--steps", "yo", "--json"]),
        typo_state_fn=ready, cleanup_factory=factory)
    assert code == 0
    assert out.read_text(encoding="utf-8") == ref
    data = _json_lines(capsys.readouterr().out)[-1]
    assert data["report"]["typo_model"] == "" and data["report"]["rule_counts"] == {"yo": 22} and not cleaners

    out2 = tmp_path / "yo-typos.txt"
    code = user_cli.cmd_prepare(
        user_cli.build_parser().parse_args(["prepare", str(src), "--out", str(out2), "--steps", "yo", "--typos", "--json"]),
        typo_state_fn=ready, cleanup_factory=factory)
    assert code == 0 and cleaners and cleaners[0].calls > 0
    text = out2.read_text(encoding="utf-8")
    assert "стёрёг" not in text and "стерёг" in text                    # the validator refuses a second yo
    assert _json_lines(capsys.readouterr().out)[-1]["report"]["typo_model"] == "sage-ru"
