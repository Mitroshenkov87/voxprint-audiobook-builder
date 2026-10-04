# Voxprint AI Audiobook Builder - End User Agreement

> **Not legal advice.** The author is not a lawyer. This is a plain-language agreement written in good faith and not checked by a lawyer; it may not fit every country.

Version 0.1 (beta) · 2026-10-04 · Copyright 2026 Aleksandr Mitroshenkov ("the author")

By installing or using Voxprint AI Audiobook Builder ("the program") you agree to this text. If you do not agree, do not install or use the program.

## 1. What you get (licence)
* The **source code** of the program is licensed under the **Apache License 2.0** (see `LICENSE` and `NOTICE`). You may use, modify and redistribute it, also commercially, under that licence.
* You may use the program for **personal** purposes and for **business** purposes (for example a publisher or a company narrating its own books), free of charge.
* The Apache licence covers the code only. **Models, voices and third-party components have their own licences** (section 5).

## 2. Your content, your responsibility
* You decide what you put in: **voice recordings, texts, books** (TXT, FB2, EPUB) and any other files or sources you choose. You are **solely responsible** for having the right to use them (copyright, licences, terms of the services you take them from, personality and privacy rights of other people). This includes the right to **translate** a book and to **narrate** it.
* You are **solely responsible for how you use and publish the output** (audiobooks, translations, trained voices, files), including labelling synthetic speech where the law or a platform requires it.
* The author does not check your content or your output and does not take responsibility for them. Machine translation and text clean-up can be wrong.

## 3. Voices and consent
* Clone or imitate a voice **only with the explicit consent of the voice owner**, and respect the licence and usage scope recorded for that voice (`voice.json`, the consent record, the badge in *My voices*). A voice with the scope "private only" must not be published or sold.
* No impersonation, fraud, deception or misleading content. No use against the law or against the rights of others.
* The consent record the program keeps is a note of what the speaker said, not a legal document.

## 4. Beta / experimental software
The program is **beta software** (v0.1.0). It can be wrong, slow, incomplete or change without notice; only one machine and one speaker were tested (`docs/TESTING.md`). Back up your recordings and voices.

## 5. Third-party models and components
The program downloads and uses models and libraries made by others: Qwen3-TTS, Qwen3-ASR and Qwen3-ForcedAligner (Apache-2.0), the optional SAGE spelling model (MIT), Opus-MT translation models by Helsinki-NLP (CC-BY-4.0 / Apache-2.0), ffmpeg, PySide6/Qt (LGPL-3.0) and others. Each keeps its own licence, listed in `THIRD_PARTY_NOTICES.md` and `docs/LICENSES.md`; by using the program you accept those licences too. The AAC/M4B export is patent-encumbered and the author gives no patent licence for it (`docs/AAC-M4B.md`).

## 6. No warranty, limited liability
The program is provided **"as is", without warranty of any kind**. To the extent the law allows, the author is **not liable** for any damage, loss of data, loss of profit or claim of a third party that arises from using the program, the output, or your content. Rights that the law of your country gives you and that cannot be excluded stay untouched.

## 7. Privacy
The program works **offline** on your computer: recordings, texts, voices and audiobooks stay there. There is **no telemetry and no account**. It connects to the internet only to download models (Hugging Face, ModelScope, the project's backup mirrors), to check for updates (PyPI, Hugging Face API), to fetch the voice index when you open *Download voices from repository*, and to install packages when you accept an offer. Details: `docs/PRIVACY.md`.

## 8. Changes and contact
The author may update this text; the version that ships with a release applies to that release. Questions, problems and legal notices: open an issue at https://github.com/Mitroshenkov87/voxprint-audiobook-builder/issues.

*This text is not legal advice, and the author is not a lawyer.*
