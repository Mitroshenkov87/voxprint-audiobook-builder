# Voxprint AI Movie Dubber - End User Agreement (DRAFT)

> **DRAFT for a program that does not exist yet.** Nothing here is in force. Items in [square brackets] are open decisions.
> **Not legal advice.** The author is not a lawyer; have this text reviewed before the program is released, especially for commercial use.

Version 0.0-draft · 2026-10-04 · Copyright 2026 Aleksandr Mitroshenkov ("the author") · based on [`EULA-TEMPLATE.md`](EULA-TEMPLATE.md)

By installing or using Voxprint AI Movie Dubber ("the program") you agree to this text. If you do not agree, do not install or use the program.

## 1. What you get (licence)
* The **source code** is licensed under the **Apache License 2.0** (see `LICENSE` and `NOTICE`).
* You may use the program for **personal** purposes (for example dubbing a film you own for your family) and for **business** purposes (for example a company dubbing **its own** videos, courses or series), free of charge.
* The Apache licence covers the code only. Models, voices and third-party components have their own licences (section 5).

## 2. Your content, your responsibility
* You decide which **films, series, videos, audio tracks, scripts and subtitles** you put in. You are **solely responsible for having all rights** to them and to dub, translate and publish them: copyright of the video, the script and the original voices, neighbouring rights, licences, contracts, and the terms of the platform or service you took them from.
* The program is **not a tool to get around copy protection (DRM)** and must not be used for that. Do not dub or distribute material you have no right to.
* You are **solely responsible for how you use and publish the output** (dubbed audio, new tracks, videos), including labelling AI-dubbed or synthetic speech where the law, a platform or a contract requires it.
* The author does not check your media or your output and does not take responsibility for them.

## 3. Voices and consent
* Use or imitate a voice **only with the explicit consent of the voice owner** (an actor, a speaker, an employee, yourself). Respect the licence and usage scope recorded for the voice. For professional actors, check your contract: many require a separate permission for synthetic voices.
* No impersonation, fraud, deception, deep-fake or misleading content.

## 3a. Subtitles and other third-party sources
* The program may help you **search for or load subtitles from third-party services** (for example OpenSubtitles) and read subtitle files you provide. Such services are **not part of Voxprint**; the program does not ship their data.
* You use them **at your own responsibility and under their terms** (account, API key, quotas, licences of the subtitle files, permitted use). Subtitles are usually protected by copyright of their authors; check that you may use them for your purpose. [Open decision: the program uses only your own account/API key; Voxprint offers no shared key.]
* Subtitles and automatic translation can be wrong or out of sync.

## 3b. Songs and singing
The program is designed to **leave songs and singing in the original** (music, background and sung parts are kept; only speech is replaced). It does not clone singing voices. [Open decision: automatic detection is not perfect; the user checks the result.]

## 4. Beta / experimental software
The program is expected to be **beta software**: dubbing quality, lip-sync and timing can be poor, and results can change between versions. Keep your originals.

## 5. Third-party models and components
Speech, translation and separation models and libraries made by others (for example Qwen3-TTS/ASR, Opus-MT, ffmpeg) are used; each keeps its own licence, listed in `THIRD_PARTY_NOTICES.md`. By using the program you accept those licences. [To fill in when the model set is final; some video/audio codecs are patent-encumbered, the author gives no patent licence.]

## 6. No warranty, limited liability
The program is provided **"as is", without warranty of any kind**. To the extent the law allows, the author is **not liable** for any damage, loss of data, loss of profit, or claim of a rights holder or other third party arising from the program, your media, or the output. Rights that cannot be excluded by law stay untouched.

## 7. Privacy
The program works **offline** on your computer: your media, transcripts and results stay there. **No telemetry, no account.** It connects to the internet only to download models, check for updates and - only when you start it - to search or download subtitles from the service you selected (your account and terms apply). [Update `docs/PRIVACY.md` when the program exists.]

## 8. Changes and contact
The author may update this text; the version shipped with a release applies to that release. Questions and notices: open an issue at https://github.com/Mitroshenkov87/voxprint-audiobook-builder/issues [repository of the dubber to be named].

*This text is not legal advice, and the author is not a lawyer.*
