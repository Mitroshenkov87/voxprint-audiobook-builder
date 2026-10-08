"""The optional AI text model passes (core/llm_text.py) with a fake model: guard, fallback, cache, glossary, UI gating."""
from types import SimpleNamespace

import pytest

from core import llm_text
from core import translate as tl
from core.book_parsers import Book, Chapter
from infra import llm_tool
from tests.test_studio import app, lib  # noqa: F401 - fixtures

RU = ["Анна пришла домой в 1812 году. Анна устала.", "Анна села у окна, Анна молчала.", "Был тихий вечер, и Анна спала."]


class FakeModel:
    """Answers with ``fn(prompt)``; counts calls and closes."""

    def __init__(self, fn):
        self.fn, self.calls, self.closed = fn, 0, False

    def complete(self, prompt, max_tokens=2048, temperature=0.2):
        self.calls += 1
        return self.fn(prompt)

    def close(self):
        self.closed = True


def _text_of(prompt):
    return prompt.rsplit("TEXT:\n", 1)[1].strip()


def en_answer(prompt):
    if "nominative" in prompt:                                         # the names glossary request
        return "Анна = Anna\nfoo = bar"
    paras = _text_of(prompt).split("\n\n")
    return "\n\n".join("Anna came home and the evening was quiet, she was tired %d." % i for i, _ in enumerate(paras))


def test_literary_paragraphs_glossary_guard_and_cache(tmp_path):
    model = FakeModel(en_answer)
    plan = llm_text.LLMPlan(lambda: model, "fake@1")
    cache = tl.TranslationCache(tmp_path / "c.json")
    gl = tmp_path / "names_en.txt"
    out = llm_text.translate_paragraphs(RU, "ru", "en", plan, cache, gl)
    assert sorted(out) == [0, 1, 2] and model.closed
    assert gl.read_text(encoding="utf-8") == "Анна = Anna\n"           # only real names from the list are kept
    # resumed: everything cached, the model is not even started
    started = []
    out2 = llm_text.translate_paragraphs(RU, "ru", "en", llm_text.LLMPlan(lambda: started.append(1), "fake@1"),
                                         tl.TranslationCache(tmp_path / "c.json"), gl)
    assert out2 == out and not started


@pytest.mark.parametrize("answer", [
    lambda p: "Only one paragraph.",                                            # wrong number of paragraphs
    lambda p: "\n\n".join("I'm sorry, I cannot translate this text for you." for _ in _text_of(p).split("\n\n")),
    lambda p: _text_of(p),                                                       # left in Russian
])
def test_guard_rejects_and_the_caller_falls_back(tmp_path, answer):
    plan = llm_text.LLMPlan(lambda: FakeModel(answer), "fake@1")
    assert llm_text.translate_paragraphs(RU, "ru", "en", plan, tl.TranslationCache(None), None) == {}


def test_book_translation_uses_opus_for_titles_and_rejected_paragraphs(tmp_path):
    def flaky(prompt):
        if "nominative" in prompt:
            return ""
        return "\n\n".join("Anna came home and the evening was quiet, she was tired." if "1812" in p else "Извините, я не могу."
                           for p in _text_of(prompt).split("\n\n"))

    class Opus:
        tag = "opus"

        def translate(self, s):
            return ["[opus] " + x for x in s]

        def close(self):
            pass

    book = Book("Книга", "", "ru", [Chapter("Глава", RU[0] + "\n\n" + RU[1])])
    plan = tl.TranslatePlan("en", "ru", lambda s, d: Opus(), llm=llm_text.LLMPlan(lambda: FakeModel(flaky), "fake@1"))
    out, _ = tl.ensure_translation(book, plan, tmp_path)
    text = out.chapters[0].text.split("\n\n")
    assert text[0].startswith("Anna came home") and text[1].startswith("[opus] ")
    assert out.chapters[0].title == "[opus] Глава"
    assert "literary translation" in (tmp_path / "translation_en.txt").read_text(encoding="utf-8")


def test_model_crash_falls_back_and_prepare_keeps_the_original(tmp_path):
    def crash():
        raise RuntimeError("llama-server exited")

    book = Book("T", "", "ru", [Chapter("c", "\n\n".join(RU))])
    out = llm_text.prepare_book(book, llm_text.LLMPlan(crash, "fake@1"), "ru", tmp_path)
    assert out.chapters[0].text == "\n\n".join(RU)
    good = FakeModel(lambda p: "" if "GLOSSARY" not in p else "\n\n".join(
        x.replace("1812", "тысяча восемьсот двенадцатом") for x in _text_of(p).split("\n\n")))
    out = llm_text.prepare_book(book, llm_text.LLMPlan(lambda: good, "fake@2"), "ru", tmp_path / "j2")
    assert "тысяча восемьсот двенадцатом" in out.chapters[0].text and good.closed
    assert (tmp_path / "j2" / ".debug" / "llm_prepared.txt").is_file()


def test_prompts_are_bundled_with_their_placeholders():
    t = llm_text.load_prompt("literary_translate")
    assert all(k in t for k in ("{source_language}", "{target_language}", "{glossary}", "{context}", "{text}"))
    assert "{language}" in llm_text.load_prompt("prepare_narration") and "{names}" in llm_text.load_prompt("names_glossary")


def test_status_gate_and_server_failure(tmp_path):
    assert llm_tool.status(tmp_path, vram_gb=6.0) == "low_vram"
    assert llm_tool.status(tmp_path, vram_gb=16.0) == "needs_download"
    assert llm_tool.make_plan(tmp_path) is None
    exe = tmp_path / "llama-server"
    srv = llm_tool.LlamaServer(exe, tmp_path / "m.gguf", log_dir=tmp_path, run=lambda cmd, **kw: SimpleNamespace(stdout="", stderr=""),
                               popen=lambda cmd, **kw: SimpleNamespace(cmd=cmd, poll=lambda: 1, returncode=1))
    with pytest.raises(RuntimeError, match="exited"):
        srv.start()


def test_narrate_window_options_are_greyed_until_downloaded(app, lib, tmp_path):
    from tests.test_translate import RU_TXT, add_voice, book_file, make_studio

    state = {"st": "needs_download"}
    n = make_studio(lib).narrate_window
    n.llm_status, n.llm_plan = (lambda: state["st"]), (lambda: llm_text.LLMPlan(lambda: None, "fake"))
    add_voice(lib, tmp_path)
    n.refresh_voices()
    n.load_book_file(book_file(tmp_path, RU_TXT))
    n._refresh_buttons()
    assert not n.chk_llm_prepare.isEnabled() and not n.chk_literary.isEnabled() and not n.btn_llm_download.isHidden()
    state["st"] = "ready"
    n._refresh_buttons()
    assert n.chk_llm_prepare.isEnabled() and n.btn_llm_download.isHidden()
    assert n.options().llm_prepare is None                     # off by default
    n.chk_llm_prepare.setChecked(True)
    assert n.options().llm_prepare.tag == "fake"


def test_the_nvidia_vulkan_device_is_chosen_on_hybrid_laptops(tmp_path):
    listing = ("Available devices:\n  Vulkan0: AMD Radeon(TM) 780M Graphics (16384 MiB, 15000 MiB free)\n"
               "  Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 7900 MiB free)\n")
    run = lambda cmd, **kw: SimpleNamespace(stdout=listing, stderr="")  # noqa: E731
    assert llm_tool.pick_device(tmp_path / "x", run) == "Vulkan1"
    assert llm_tool.pick_device(tmp_path / "x", lambda cmd, **kw: SimpleNamespace(stdout="Vulkan0: Intel Arc", stderr="")) is None
    seen = []
    srv = llm_tool.LlamaServer(tmp_path / "x", tmp_path / "m.gguf", log_dir=tmp_path, run=run,
                               popen=lambda cmd, **kw: seen.append(cmd) or SimpleNamespace(poll=lambda: 1, returncode=1))
    with pytest.raises(RuntimeError):
        srv.start()
    assert seen[0][-2:] == ["--device", "Vulkan1"]
