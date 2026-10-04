# My voices and licences

*My voices* lists the library as cards: name, language, length of speech, epochs, voice type, author, a **licence badge** (green: commercial use allowed, amber: personal use only), **Preview** (plays the reference sample),
*Narrate with this voice*, *Details…* (edit name, author, licence, description) and *Delete* (asks first). **Import voice…** accepts an adapter folder or a `.zip` (archives are checked: no path tricks, only the
adapter files, size limits). A voice without a declared licence is treated as `custom/personal-only`.

| Licence | Commercial use |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | allowed (attribution / share-alike per the licence) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | not allowed |
| custom/personal-only (default) | not allowed |
| custom/test-use-only (voices meant for testing) | not allowed; test use only, no public release of the audio |

`commercial_use` in `voice.json` is always derived from the licence, never trusted from a file you import. The badge is information, not legal advice.

**Download voices from repository.** The button reads a static `index.json` (schema 1) from a configurable URL: environment variable `VOXPRINT_VOICES_INDEX`, or the first line of
`%LOCALAPPDATA%\Voxprint\state\voice_index_url.txt`. The built-in default is `voices/index.json` of this repository (`raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/voices/index.json`); the voice zips are release assets (tag `voices-v1`). A placeholder URL makes the dialog say "not set up yet". An empty list and no network
are handled with friendly messages. Downloads are HTTPS-only, size-capped, **resumable** (a `.part` file named by the SHA-256 and an HTTP `Range` request; a server that ignores `Range` restarts the download), the SHA-256 from the index must match, and the licence in the index is the one shown. The last good index is cached (`state\voice_index_cache.json`), so the list still shows offline. Index format:
```json
{"schema": 1, "voices": [{"id": "anna-ru", "name": "Anna", "language": "russian", "author": "...", "license": "CC-BY-4.0",
  "description": "...", "voice_type": "female", "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "url": "https://example.org/anna-ru.zip", "sha256": "<64 hex digits>", "size_bytes": 61000000,
  "names": {"ru": "Анна", "en": "Anna", "de": "Anna"}, "descriptions": {"ru": "...", "en": "...", "de": "..."}}]}
```
`names` / `descriptions` are optional: the UI shows the entry for the current UI language and falls back to `name` / `description`. The same two maps may be in `voice.json`
(a downloaded voice keeps them, so a voice can show a different name per UI language; the stored `name` is never rewritten).

**Online voices appear by themselves.** *My voices* and the voice list of *Narrate a book* merge the local library with the index (refreshed when the window opens; *Refresh list* in My voices): an online voice
that is not installed yet is a card with the licence and scope badges, the size and a **Download** button; in *Narrate* it is listed as "<name> (to download, N MB)" and is downloaded and verified automatically when you press
Start. A downloaded voice remembers its index id (`repo_id`) and is then shown once, as a local voice. A voice added to the index later shows up after the next refresh - no program update needed.

**Voice packages.** A voice package is a zip with the LoRA adapter and a `voice.json` (name, localized `names` / `descriptions`, licence, consent block). Packages are **not** part of this repository or the installer;
they are hosted separately (a model host or a release asset) and listed in the voices index. Voices meant for testing carry the licence **`custom/test-use-only`** - *Test use only - no public release of generated audio, no commercial use*;
their consent block has `method: "owner"` (the publisher owns the voice).

**The open universal voice ("Open universal voice" / "Открытый универсальный голос" / "Offene Universalstimme").** A fully open English voice for anyone who has no recording of their own: trained with Voxprint's own pipeline
(LoRA, learning rate 1e-6, 10 epochs, 66 clips = 7.5 min) **only** from the public-domain **LJ Speech 1.1** dataset (reader Linda Johnson, LibriVox recordings; compiled by Keith Ito, <https://keithito.com/LJ-Speech-Dataset/>;
the dataset is dedicated to the public domain, the Hugging Face copy `keithito/lj_speech` is tagged `unlicense`) on top of `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, which is **Apache-2.0** (commercial use of the model, its outputs and adapters is allowed).
The voice package is licensed **CC0-1.0** (free for any use, also commercial; attribution to the LJ Speech dataset is appreciated). It is not stored in this repository: the spec is `tools/voice_specs/open-universal.json`, the zip is a release asset (`voices-v1`) listed in `voices/index.json`.
Measured on the finished voice: a 15.8 s sample was recognised back with WER 0.04 and a pitch shift of +1.0 semitone from the reference clip.

The programme shows a "test use only" reminder on its card, in Narrate and when an audiobook made with it is ready. To publish a voice package yourself:
```
python tools/make_voice_package.py --adapter <trained voice folder> --spec tools/voice_specs/example-open-voice.json --out dist_voices --url https://huggingface.co/<user>/voxprint-voices/resolve/main/open-voice.zip
```
It writes the folder, `open-voice.zip` and `open-voice.index-entry.json` (one object for the `voices` list, with SHA-256 and size). Upload the zip to a model host or a release asset and add the entry to `index.json`.
