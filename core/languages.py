"""Language names and codes in one place (Qt-free, no torch).

Two forms are used across the program:

* **BCP-47 codes** (``ru``, ``en``, ``de``, ``pt-BR`` ...) - stored in ``voice.json`` and book metadata;
* **Qwen language names** - what ``Qwen3TTSModel.generate_voice_clone(language=...)`` accepts.  The exact list comes from
  the ``qwen_tts`` source (0.1.1): ``get_supported_languages()`` returns ``"auto"`` plus the keys of
  ``talker_config.codec_language_id`` of the base model, and the HF configs of both Qwen3-TTS-12Hz Base models (0.6B and
  1.7B) list exactly the ten languages of :data:`QWEN_TTS_LANGUAGES`.  The check there is case-insensitive; we pass the
  canonical capitalised names anyway.

:func:`narration_language` picks the token for a narration run: the language of the TEXT that is spoken (translation
target, else the book's language), the voice's language only as the last fallback, ``"Auto"`` if nothing is known.
"""
from __future__ import annotations

from typing import Optional

#: Languages Qwen3-TTS Base models can be told to speak (see the module docstring), plus :data:`AUTO`.
QWEN_TTS_LANGUAGES = ("Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian",
                      "Spanish")
#: "Let the model decide" - the model then guesses from the text itself.
AUTO = "Auto"

#: BCP-47 primary subtag -> English name.  Covers the Qwen3-TTS languages, the Qwen3-ASR languages (which name the language of
#: a recording when a voice is trained) and a few neighbours a book can be in.
NAMES = {
    "zh": "Chinese", "en": "English", "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "es": "Spanish", "yue": "Cantonese", "ar": "Arabic", "id": "Indonesian",
    "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian", "uk": "Ukrainian", "be": "Belarusian", "kk": "Kazakh",
    "bg": "Bulgarian", "sr": "Serbian", "no": "Norwegian", "he": "Hebrew",
}
#: Other spellings that mean the same language: ISO 639-2/3 codes and native names (lower case).
_ALIASES = {
    "zho": "zh", "chi": "zh", "cmn": "zh", "mandarin": "zh", "中文": "zh", "汉语": "zh", "普通话": "zh",
    "eng": "en", "fra": "fr", "fre": "fr", "français": "fr", "francais": "fr", "deu": "de", "ger": "de", "deutsch": "de",
    "ita": "it", "italiano": "it", "jpn": "ja", "日本語": "ja", "kor": "ko", "한국어": "ko", "por": "pt", "português": "pt",
    "portugues": "pt", "rus": "ru", "русский": "ru", "spa": "es", "español": "es", "espanol": "es", "ukr": "uk",
    "українська": "uk", "bel": "be", "беларуская": "be", "nld": "nl", "dut": "nl", "nederlands": "nl", "pol": "pl",
    "polski": "pl", "ces": "cs", "cze": "cs", "čeština": "cs", "tur": "tr", "türkçe": "tr", "ara": "ar", "tagalog": "fil",
    "farsi": "fa", "ell": "el", "gre": "el", "nb": "no", "nn": "no", "iw": "he",
}
_BY_NAME = {name.lower(): code for code, name in NAMES.items()}


def language_code(value: Optional[str]) -> str:
    """BCP-47 code for a language name or code in any common form; ``""`` if empty or not recognisable.

    ``"Russian"``, ``"russian"``, ``"русский"``, ``"rus"``, ``"ru"`` -> ``"ru"``; ``"ru_RU"`` / ``"ru-ru"`` -> ``"ru-RU"``
    (a region given in the input is kept, in canonical case); ``"auto"`` -> ``""``.
    """
    s = (value or "").strip().replace("_", "-")
    if not s:
        return ""
    low = s.lower()
    if low in _BY_NAME:
        return _BY_NAME[low]
    if low in _ALIASES:
        return _ALIASES[low]
    parts = low.split("-")
    base = _ALIASES.get(parts[0], parts[0])
    if base not in NAMES:
        # Not one we know by name: accept anything shaped like a language tag (2-3 letters + subtags), drop the rest
        if not (2 <= len(base) <= 3 and base.isascii() and base.isalpha()) or base == "und":
            return ""
    rest = []
    for p in parts[1:]:
        if not (p.isascii() and p.isalnum() and 1 <= len(p) <= 8):
            break
        rest.append(p.upper() if len(p) == 2 and p.isalpha() else (p.title() if len(p) == 4 and p.isalpha() else p))
    return "-".join([base] + rest)


def language_name(value: Optional[str]) -> str:
    """English name of a language (``"ru"`` / ``"ru-RU"`` / ``"Russian"`` -> ``"Russian"``); the code itself if unnamed."""
    code = language_code(value)
    if not code:
        return ""
    return NAMES.get(code.split("-")[0], code)


def qwen_language(value: Optional[str]) -> str:
    """The exact Qwen3-TTS language name for any form of a language (code, BCP-47 tag, English or native name).

    ``"auto"`` -> :data:`AUTO`; a language the model does not know (Ukrainian, Polish ...) or garbage -> ``""``.
    """
    if (value or "").strip().lower() == "auto":
        return AUTO
    name = language_name(value)
    return name if name in QWEN_TTS_LANGUAGES else ""


def book_text_language(book) -> str:
    """Qwen language name of a book's TEXT, :data:`AUTO` for a known language Qwen3-TTS lacks, ``""`` = unknown.

    The book's metadata tag is used unless it is missing or contradicted by the text itself (EPUB tools often stamp
    ``en`` on any book): a sample of the text is checked with :func:`core.translate.detect_language` (ru/en/de/uk), and for
    scripts it does not cover, :func:`core.text_utils.detect_language` (Chinese/Japanese/Korean/Russian by script; its
    "English" answer only means "some Latin script", so it is not trusted).  Detection runs once per book, never per chunk.
    """
    from core import text_utils, translate

    meta = qwen_language(getattr(book, "language", "") or "")
    sample = " ".join(c.text[:3000] for c in list(getattr(book, "chapters", []) or [])[:6])[:12000]
    code = translate.detect_language(sample) if sample.strip() else ""
    if code == "uk":
        return AUTO                                 # Ukrainian text: Qwen3-TTS has no such language, so let the model decide
    detected = qwen_language(code)
    if detected:
        return detected                             # the text itself wins over a (possibly wrong) tag
    if meta:
        return meta
    if sample.strip():
        guess = text_utils.detect_language(sample)
        if guess != "English":
            return qwen_language(guess)
    return ""


def narration_language(book, voice_language: str = "", translate_target: str = "") -> str:
    """Language token for ``generate_voice_clone`` in a narration run: always one of :data:`QWEN_TTS_LANGUAGES` or
    :data:`AUTO`.

    Order: the translation target (the translated text is what gets spoken) -> the book's text language
    (:func:`book_text_language`) -> the voice's language -> ``"Auto"``.  A Russian voice reading an English book therefore
    gets ``"English"``.
    """
    if translate_target:
        target = qwen_language(translate_target)
        return target or AUTO                       # translated into a language the model cannot be told: let it guess
    return book_text_language(book) or qwen_language(voice_language) or AUTO
