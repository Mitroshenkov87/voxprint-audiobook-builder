# Recording scripts for voice training

Ready-made texts to read aloud when you record a voice for Voxprint (the Train window takes the TXT file as the "text"; see the
[manual](../manual/), chapter *Train your voice*). Version 4, written natively in each language - they are **not** translations of each other:

| Language | Text | PDF dark (screen) | PDF light (print) |
|---|---|---|---|
| Russian | [TXT](Voxprint-RecordingScript-v4-ru.txt) | [PDF](Voxprint-RecordingScript-v4-ru.pdf) | [PDF](Voxprint-RecordingScript-v4-ru-print.pdf) |
| English | [TXT](Voxprint-RecordingScript-v4-en.txt) | [PDF](Voxprint-RecordingScript-v4-en.pdf) | [PDF](Voxprint-RecordingScript-v4-en-print.pdf) |
| Deutsch | [TXT](Voxprint-RecordingScript-v4-de.txt) | [PDF](Voxprint-RecordingScript-v4-de.pdf) | [PDF](Voxprint-RecordingScript-v4-de-print.pdf) |

* **A recording aid, not a product text:** 17 blocks - greeting, the story of the idea, numbers and dates, smooth sentences for the sounds of the language,
  neutral / calm / joy / sadness / irritation / surprise / whisper and loud reading, three short computer stories, an optional "bad day" story with
  mild everyday colour words, an optional block of technical terms, and the voice owner's consent at the very end. Reading time: about 10-13 minutes
  of text plus about 2-3 minutes of optional blocks; the PDFs contain a reading guide (tempo, pauses, emotions, breaks).
* **No profanity.** The "rough" words (e.g. *idiot, jerk, damn* / *дурак, балбес, чёрт возьми* / *Idiot, Dummkopf, verflixt*) are mild, occasional colour inside
  ordinary sentences and are aimed at machines and circumstances, never at people or groups. A test (`tests/test_script_match.py`) checks the scripts for
  profanity stems.
* **Consent text per language.** Block 17 holds the three consent sentences (commercial / public non-commercial / private only) of the app's
  consent feature (`core/consent.py`, `TEMPLATES`) - only in the language of the file. Read **one** of them at the very end of the recording, saying your own
  name and the date; read it only if the voice is yours or you have the voice owner's permission. It is a record of what was said, not legal advice. The
  parser (`core/script_match.py`) stops at the consent block, so it is never used as training text.
* **Format:** one line = one sentence; `[text in square brackets]` and `=== headers ===` are hints and are not read aloud.

The PDFs are built from the TXT files with `python build_script_pdf.py` (needs `markdown`, `weasyprint`, `pillow`, poppler-utils); the layout is tuned
until `../manual/check_pagination.py` passes.
