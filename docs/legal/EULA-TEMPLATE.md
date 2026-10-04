# End User Agreement - TEMPLATE for Voxprint programs

> **Not legal advice.** The author of Voxprint is not a lawyer. This text is a plain-language agreement written in good faith; it has not been checked by a lawyer and may not fit every country. If a lot depends on it (a company, a commercial release), have it reviewed locally.

How to use this template: copy it to `EULA-<program>.md`, replace everything in `{{double braces}}`, delete the optional sections you do not need, keep the numbering and the order (so that all Voxprint programs read alike). The installer shows the plain-text version (`tools/gen_eula_txt.py`).

---

# {{PRODUCT NAME}} - End User Agreement

Version {{0.1}} · {{date}} · Copyright {{year}} Aleksandr Mitroshenkov ("the author")

By installing or using {{PRODUCT NAME}} ("the program") you agree to this text. If you do not agree, do not install or use the program.

## 1. What you get (licence)
* The **source code** of the program is licensed under the **Apache License 2.0** (see `LICENSE` and `NOTICE`). You may use, modify and redistribute it, also commercially, under that licence.
* You may use the program for **personal** purposes and for **business** purposes (for example inside a company), free of charge.
* The Apache licence covers the code only. **Models, voices and third-party components have their own licences** (section 5).

## 2. Your content, your responsibility
* You decide what you put in: {{INPUT KINDS - e.g. voice recordings, texts, books}}. You are **solely responsible** for having the right to use it (copyright, neighbouring rights, licences, terms of the services you take it from, personality and privacy rights of other people).
* You are **solely responsible for how you use and publish the output** (the audio, text, files the program creates), including labelling synthetic speech where the law or a platform requires it.
* The author does not check your content or your output and does not take responsibility for them.

## 3. Voices and consent
* Clone or imitate a voice **only with the explicit consent of the voice owner**, and respect the licence and usage scope recorded for that voice.
* No impersonation, fraud, deception or misleading content. No use against the law or against the rights of others.

{{PRODUCT-SPECIFIC SECTION - e.g. "3a. Subtitles and media"; delete if not needed}}

## 4. Beta / experimental software
The program is **beta software**. It can be wrong, slow, incomplete or change without notice. Back up what matters to you.

## 5. Third-party models and components
The program downloads and uses models and libraries made by others (for example Qwen3-TTS and Qwen3-ASR, Opus-MT translation models, ffmpeg, PySide6/Qt). Each keeps its own licence (Apache-2.0, MIT, CC-BY, LGPL/GPL and others), listed in `THIRD_PARTY_NOTICES.md` and `docs/LICENSES.md`. By using the program you accept those licences too. Some formats (for example AAC/M4B) are patent-encumbered; the author gives no patent licence for them (`docs/AAC-M4B.md`).

## 6. No warranty, limited liability
The program is provided **"as is", without warranty of any kind**. To the extent the law allows, the author is **not liable** for any damage, loss of data, loss of profit or claim of a third party that arises from using the program, the output, or your content. Rights that the law of your country gives you and that cannot be excluded stay untouched.

## 7. Privacy
The program works **offline** on your computer: your recordings, texts and results stay there. There is **no telemetry and no account**. It connects to the internet only to download models, to check for updates, to fetch the voice index when you open it, and to install packages when you accept an offer ({{ADD PRODUCT-SPECIFIC NETWORK USE}}). Details: `docs/PRIVACY.md`.

## 8. Changes and contact
The author may update this text; the version that ships with a release applies to that release. Questions, problems and legal notices: open an issue at https://github.com/Mitroshenkov87/voxprint-audiobook-builder/issues.

*This text is not legal advice, and the author is not a lawyer.*
