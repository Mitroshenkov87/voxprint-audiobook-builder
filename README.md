# Voxprint AI Audiobook Builder

*Train a voice, narrate books.* (Short name and technical identifier: **Voxprint** - the package, the executable and the data folder keep that name.)

**Your voice from a recording in one click - and then whole books in that voice.** Give Voxprint a 5-15 minute recording of a voice (yours, or of a person who has given permission) and
the text you read: it aligns the text to the audio, cuts a training dataset and trains your voice as a LoRA adapter for
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (optionally merged into a standalone model for any app that runs Qwen3-TTS).
The voice goes into your **voice library**; then pick a book (TXT, FB2, EPUB) and a voice and get a finished audiobook with chapters
(one Opus file, per-chapter MP3, ...). A native Windows 11 application - no command line, no browser, no Gradio, no WSL.

<p align="center"><img src="docs/screenshots/studio_home.png" alt="Voxprint Studio" width="560"></p>

> **Requires Windows 11 (24H2 / build 26100 or newer) and an NVIDIA GPU (16 GB VRAM recommended).**
> Without an NVIDIA GPU the dataset is still created, but voice training falls back to the CPU (very slow, with a warning).
> The code itself also runs on Linux (that is where the test-suite runs); a Linux/macOS GUI release is not a goal yet.

> **Status: early (0.1.0).** The whole pipeline was exercised on a real RTX 4090 / Windows Server 2025 machine (see
> [Tested on Windows](#tested-on-windows)), but there has been no public release yet and voice *quality* has not been
> judged by ear. Expect rough edges.

## Contents
[Concept](#concept-and-philosophy) · [Features](#features) · [Screenshots](#screenshots) · [Installation](#installation) ·
[Using Voxprint](#using-voxprint) · [Narrate a book](#narrate-a-book) · [My voices](#my-voices-and-licences) · [AAC / M4B notice](#aac--m4b-patents-please-read) ·
[How it works](#how-it-works) · [Output format](#output-format) ·
[Hardware requirements](#hardware-requirements) · [Model reuse](#reusing-what-is-already-on-your-computer) · [Backup and restore](#backup-restore-and-existing-models) ·
[Tested on Windows](#tested-on-windows) · [Privacy](#privacy) · [Roadmap](#roadmap) · [Contributing](#contributing) ·
[Licences](#licences-and-third-party-components)

## Concept and philosophy
Voxprint grew out of a search for a truly **one-button voice-cloning tool**. The tools we found needed a command line,
manual setup of components or hand-prepared files, or covered only a single step. Voxprint covers the **whole path** - from a
voice recording and a text file to ready output files you can load into a neural network. Your effort is minimal:
**read a text aloud and attach two files.**

* **One button, no expertise.** Hyper-parameters, model sizes, chunking and cleaning are chosen automatically for your GPU.
* **Economical.** Models and tools that other apps (Hugging Face cache, Pinokio/Alexandria, ModelScope) already downloaded are reused
  in place, read-only - nothing is downloaded twice, nothing of other apps is touched.
* **Honest and verifiable.** Package versions and model revisions are pinned to a "verified by Voxprint" set; failures produce stable,
  localized messages instead of tracebacks; every claim in this README is backed by a test or a documented real run.
* **Private by design.** Everything runs on your computer (see [Privacy](#privacy)).
* **Windows first**, other platforms later. Built jointly by a human and an AI - an honest attempt.

## Features
* **Studio** - the first window: three big cards (*Narrate a book*, *Train your voice*, *My voices*) and a gear button for Settings. Every other window has a *← Studio* button.
* **Narrate a book** - TXT / FB2 (also `.fb2.zip`) / EPUB parsing with chapters (pure standard library), sentence-sized chunks, synthesis
  chunk by chunk with your voice (Qwen3-TTS + LoRA adapter), progress and time left, **pause / cancel / resume** (finished chunks are kept on disk).
* **Audiobook formats** - default: **one `.opus` file with chapter markers**; **MP3 per chapter** (most compatible, ID3 tags + `.m3u8`); opt-in **M4B (AAC)** for Apple Books
  (patent notice below); more under *Other formats* (M4B with Opus, Opus/FLAC/WAV per chapter, one MP3 with chapter marks). Quality is three buttons (*Compact / Standard / High*); exact bitrates sit under *Advanced*.
* **Book preparation** - fully automatic text clean-up before synthesis (no editor, nothing to review): rule-based steps for Russian and English (layout, footnotes/page numbers, quotes and dashes, links, chapter headings, **numbers in words** with Russian case/gender agreement, abbreviations) and an optional on-demand AI typo/comma fixer for Russian that is checked by a strict validator ([details](#prepare-the-text)).
* **Backup and restore** of models and voices to any folder or drive (resumable, skips identical files, free-space check, SHA-256 verified), an *existing models folder* that is imported before any download, and an optional installer page for it ([details](#backup-restore-and-existing-models)).
* **Voice library** (`%LOCALAPPDATA%\Voxprint\voices\`) - every trained voice is registered automatically; preview, licence badge, delete, **import** from a folder or a zip,
  and **download voices from a repository** (index URL is configurable; SHA-256 checked).
* **Voice licences** - each voice carries a licence (CC0, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA or custom/personal-only) and the UI shows whether *commercial use* is allowed.
* Forced alignment of your text onto the recording with `Qwen/Qwen3-ForcedAligner-0.6B` (+ optional CTC backup aligner);
  long recordings are cut at pauses and aligned in chunks.
* Automatic **dataset in the Alexandria `train_lora.py` format**: 3-12 s clips cut at pauses (never mid-word), a clean reference
  clip, `metadata.jsonl`, a report. Bad fragments (clipping, silence, noise) are dropped automatically.
* Russian text normalization (numbers, abbreviations are spelled out) before alignment; UTF-8 / cp1251 text input; WAV, FLAC, MP3, M4A, OGG ... audio input.
* **LoRA voice training** with the Alexandria recipe (r=32, alpha=128 on the talker), parameters chosen from your VRAM
  (1.7B or 0.6B base, 8-bit Adam, CPU fallback, automatic retry plan after out-of-memory).
* Optional **universal model** (~4 GB, `custom_voice` format) merged from the adapter - works in any Qwen3-TTS app.
* **`voice.json`** next to the adapter (schema 2): id, name, language, creation date, speech duration, epochs, base model, author, licence (+ URL), voice type, description and the derived `commercial_use`.
* **Settings** dialog (gear): language (English, Deutsch, Русский), update check, model/data folders, repair, About.
* Scrollable window that stays usable on small screens (e.g. 1366x768 at 150% scaling).
* Safe self-maintenance: verified-version updates with staging + smoke test + rollback, a repair command, a completion manifest.
* Acrylic (glass) look on Windows 11 with a dark, high-contrast (WCAG AA) theme and a plain fallback.

## Screenshots
All screenshots are English-UI renders of the current theme (offscreen Qt; see `docs/screenshots/`; Russian and German renders are in `docs/screenshots/ru/` and `docs/screenshots/de/`).

| Studio (no voices yet) | Studio | My voices |
|---|---|---|
| <img src="docs/screenshots/studio_empty.png" width="300"> | <img src="docs/screenshots/studio_home.png" width="300"> | <img src="docs/screenshots/voices.png" width="300"> |

| Narrate a book (sections collapsed) | Everything expanded, AAC notice | Narration running |
|---|---|---|
| <img src="docs/screenshots/narrate.png" width="300"> | <img src="docs/screenshots/narrate_aac.png" width="300"> | <img src="docs/screenshots/narrate_running.png" width="300"> |

Same windows in Russian and German: [`ru/studio_home`](docs/screenshots/ru/studio_home.png), [`ru/voices`](docs/screenshots/ru/voices.png), [`ru/settings`](docs/screenshots/ru/settings.png), [`de/settings`](docs/screenshots/de/settings.png), [`ru/narrate`](docs/screenshots/ru/narrate.png), [`ru/narrate_aac`](docs/screenshots/ru/narrate_aac.png),
[`de/studio_home`](docs/screenshots/de/studio_home.png), [`de/voices`](docs/screenshots/de/voices.png), [`de/narrate`](docs/screenshots/de/narrate.png), [`de/narrate_aac`](docs/screenshots/de/narrate_aac.png).

Training window ("Train your voice"):

| Choose files | Running | Done |
|---|---|---|
| <img src="docs/screenshots/main_files_chosen.png" width="300"> | <img src="docs/screenshots/main_running.png" width="300"> | <img src="docs/screenshots/main_done.png" width="300"> |

| Idle | Settings (gear) | About |
|---|---|---|
| <img src="docs/screenshots/main_idle.png" width="300"> | <img src="docs/screenshots/settings.png" width="300"> | <img src="docs/screenshots/about.png" width="300"> |

Acrylic preview - the translucent window with its strong dark tint, composited over a deliberately bright desktop (the worst case for contrast):

<p align="center"><img src="docs/screenshots/main_acrylic_preview.png" alt="Acrylic preview" width="520"></p>

Real desktop screenshots from the Windows test run (Windows Server 2025, taken *before* the darker, more opaque theme - the current look is the one above):

| 1920x1080 at 125% | 1366x768 at 125% |
|---|---|
| <img src="docs/screenshots/windows_desktop_1920x1080_125.png" width="420"> | <img src="docs/screenshots/windows_desktop_1366x768_125.png" width="420"> |

## Installation
There is no public release yet. Planned installer channels are listed in the [Roadmap](#roadmap). Today you have two options.

### Option 1: the installer (built locally)
`build.bat` produces `installer\Output\Voxprint-Setup.exe` (Inno Setup 6, per-machine install, Windows 11 x64; about 1.8 GB because PyTorch is inside).
The installer does not contain the models: on first start Voxprint downloads about 7 GB once (internet needed).
The wizard is available in **English, Russian and German** and has an optional page **"Existing models"** (right after the install folder): *Do you already have downloaded models from a previous install?* - choose the
folder or leave the field empty to skip. The installer **copies nothing**; it only writes the chosen path to `%LOCALAPPDATA%\Voxprint\state\existing_models_dir.txt` (UTF-8), and on the first start the program imports the
models from there instead of downloading them (see [Backup, restore and existing models](#backup-restore-and-existing-models)). Silent install: `Voxprint-Setup.exe /VERYSILENT /ModelsDir="D:\old\models"`.
Caveat: the installer runs elevated (per-machine); if the administrator account differs from the account that uses the program, `%LOCALAPPDATA%` is the administrator's - then set the folder in *Settings* instead.

### Option 2: run from source (Windows)
```bat
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install uv
uv pip install torch torchaudio --torch-backend=auto       :: fallback: pip install -r requirements-torch.txt
uv pip install -r requirements.txt -r requirements-verified.txt
uv pip install --no-deps -r requirements-nodeps.txt        :: qwen-asr/qwen-tts: their transformers pins conflict
python -m bitsandbytes                                     :: optional check of the 8-bit optimizer (plain AdamW otherwise - fine)
python main.py
```
**Never run `uv run` without `--no-sync`** - it re-syncs the environment and replaces CUDA torch with the CPU build. flash-attn is not needed on Windows.

### Build the installer yourself
```bat
build.bat              :: venv + torch (uv, CUDA auto-detect) + dependencies + notices + tests + dist\Voxprint.exe (--onefile) + installer if Inno Setup 6 exists
build.bat onedir       :: dist\Voxprint\ folder (preferred: faster start, and Qt/PySide6 stay replaceable - see Licences)
:: quick rebuild:  set VOX_SKIP_TESTS=1 & set VOX_SKIP_INSTALLER=1 & build.bat onedir
:: verify a frozen build contains every library:  dist\Voxprint\Voxprint.exe --selftest-imports   (result in <app home>\logs\selftest_imports.txt)
```
Command-line maintenance flags of `main.py`: `--prefetch` (download models now), `--selftest`, `--selftest-imports`, `--verify-install`, `--repair`.
App data lives in `%LOCALAPPDATA%\Voxprint` (`models\`, `logs\`, `state\`, ...; override with `VOXPRINT_HOME`).
Interface language override: `VOXPRINT_LANG=en|de|ru`.

## Using Voxprint
Start the program: the **Studio** opens. *Narrate a book* is the main card; if the library has no voices yet it says so and points to *Train your voice* / *My voices*.
Use **← Studio** in any window to come back; the gear (top right) opens Settings.

### Train your voice
1. Press **Choose audio** and **Choose text** (or drag the files into the window). The text is a UTF-8 `.txt` of what you read.
2. Optional, under the main button: pick a **voice type** (male / female / child / other) and type a short **description** - both are saved in `voice.json`.
3. Press **Create voice (LoRA)** - everything else is automatic. When it finishes, the folder opens:
   `...\<recording name>_Voxprint\dataset` (the dataset) and `...\output\<voice name>` (the small adapter, tens of MB).
   **Create dataset** builds only the dataset.
4. Optional: **Build universal model (~4 GB)** merges the trained adapter into a standalone model in `...\output\<voice name>\merged_model`
   (Qwen3-TTS `custom_voice` format; the voice is `speaker=<voice name>`). The small adapter works only in a few apps (e.g. Alexandria);
   the merged model works in any app that runs Qwen3-TTS. Voxprint checks the free disk space and asks for confirmation first; the big model is never built automatically.
5. When training finishes the voice is **registered in the library** automatically (status line "voice registered"; if that fails you get a warning, the adapter folder is still there).
6. The **gear** button opens **Settings**: language, *Check for updates* (also checked weekly in the background), open the models / data & log folders,
   *Repair the installation*, and *About* (help, authors, open-source components with their licences, third-party notices).

Use voices responsibly: only with the voice owner's permission, in line with the voice's licence (see *Privacy* below).

### Training presets

The Train window offers **Fast / Balanced (default) / Maximum / Manual**, each with a short meaning and an **estimated training time for your graphics card and recording**.
Balanced is the automatic plan. Fast uses about half the passes (epochs) and a half-size adapter (LoRA rank 16); Maximum about twice the passes and rank 64; Manual opens the *Advanced* panel
(epochs, rank, alpha, learning rate, gradient accumulation). The learning rate is never raised by a preset: in our first real GPU test, 15 epochs at 3e-6 and 8e-6 gave a lower training loss
but a voice that babbled and never stopped, so the loss alone is not a quality signal.
The estimate is calibrated on a measured run (RTX 4090: 24 s per epoch for 70 clips, 12 s model load; other cards are scaled by a table, unknown ones by VRAM, a CPU is roughly 40x slower and is not measured);
read it as "about".

### No transcript? (audio only)

Tick **I have no transcript** in the Train window and the program recognises the speech itself (Qwen3-ASR, offline; the model, ~1.9 GB, is downloaded on first use).
You can choose **many audio files at once**, add a **folder** (searched recursively) or drag and drop them; each file is recognised separately. Long recordings are cut at pauses
into pieces of at most 14 s, short clips are used as they are, and everything is merged into **one dataset** (volume-normalised to about -20 dBFS).
Skipped automatically: clips under 1.5 s or over 20 s, empty or **implausible recognition** (looping text, absurd speaking rate, wrong script - a score computed from the clip
and the text, because the ASR API gives no per-token confidence), duplicates (identical audio, or the same words at the same length) and clips that fail the usual signal-quality filter.
When it is done the window reports the totals: files, kept clips and how many minutes of the audio were kept. Automatic recognition can contain errors, so a clear localized warning is shown and
**an explicit "I understand" tick is required** before the buttons work; the dataset's `report.json` records every clip's source file and plausibility.

### Narrate a book
1. **Choose book…**: `.txt` (UTF-8 / cp1251; headings like "Chapter 1", "Глава 2", "Part I" become chapters), `.fb2` / `.fb2.zip`, `.epub`. The book title, author and cover are read when present.
2. Pick the **voice** (a licence badge is shown; for a personal-use-only voice you are reminded not to publish or sell the narration). No voices yet? The window offers *Train your voice* / *My voices*.
3. **Prepare the text** (automatic, see [below](#prepare-the-text)): all rule-based steps are on by default; leave them as they are or untick what you do not want.
4. Pick the **format** (see below) and a **quality** button (*Compact / Standard / High*), then **Start narration**. Rarely needed things are collapsed: *Other formats*, and *Advanced* (exact bitrates per format, the output folder - default `Documents\Voxprint\Audiobooks` - whether chapter titles are read aloud, and a read-only sample of how the first changed paragraph looks after preparation).
5. Progress with the time left; **Pause / Resume** and **Cancel** are always available. Every synthesized fragment is stored (`<output>\<book>\.cache`), so after a cancel, a crash or closing the program
   **Start continues where it stopped**; the cache is deleted after a successful run. Texts are split into chunks at sentence/clause boundaries (reusing the clause splitter of the aligner), chapters get a pause between them.

| Format | What you get | Where it plays |
|---|---|---|
| **Opus, one file** (default) | `Author - Title.opus` with chapter markers (32 kbit/s mono), open and royalty-free, small | VLC, Audiobookshelf, Voice (Android), Smart AudioBook Player and other free players; chapter support varies by player |
| **MP3, one file per chapter** | folder `... - MP3\NN - Chapter.mp3` + `.m3u8` playlist, ID3 tags (title, album, artist, track n/N, cover); 96 kbit/s mono | practically everything (old players, car radios, phones) |
| **M4B (AAC)** - opt-in, [see the notice](#aac--m4b-patents-please-read) | `Author - Title.m4b` with chapters, 64 kbit/s mono | Apple Books (iPhone/Mac) and most audiobook apps |
| *Other formats* | M4B with Opus audio, Opus per chapter, one MP3 with chapter marks (ID3 CHAP frames), FLAC / WAV per chapter | - |

**Quality presets** (mono speech; `core/audiobook_export.py: QUALITY_PRESETS`):

| Preset | Opus | MP3 | AAC | About per hour of audio (Opus / MP3 / AAC) |
|---|---|---|---|---|
| Compact | 24 kbit/s | 64 kbit/s | 48 kbit/s | 11 / 29 / 22 MB |
| **Standard** (default) | 32 kbit/s | 96 kbit/s | 64 kbit/s | 14 / 43 / 29 MB |
| High | 48 kbit/s | 128 kbit/s | 96 kbit/s | 22 / 58 / 43 MB |

Changing a bitrate by hand under *Advanced* switches the quality to "Custom" (no button highlighted).

### Prepare the text
Pipeline: **rules** (deterministic, instant) -> optional **AI clean-up** (small model, on demand) -> **synthesis**. Nothing is shown for review; the original book is never modified, and chapter titles in the output
files / metadata / cover come from the original, unprepared book. For debugging, the prepared text is written next to the job cache: `<output>\<book>\.debug\prepared_text.txt` and `prep_report.json`
(counts per step, skipped steps, model revision). Unlike `.cache` they are **kept** after a successful run, so you can see exactly what was read; delete the `.debug` folder if you do not need it.

| Step (checkbox) | What it does |
|---|---|
| Tidy the layout | joins hard-wrapped lines and hyphenated line ends, removes soft hyphens, zero-width and odd space characters, ligatures, double spaces |
| Remove footnote marks and page numbers | `[1]`, superscripts, lone page numbers, running headers repeated on many pages |
| Normalize quotes and dashes | typographic quotes/apostrophes, dialogue dashes, `--`, ellipses |
| Shorten links and e-mail addresses | a URL / address is read as "ссылка" / "link" instead of being spelled out |
| Fix chapter headings | `ГЛАВА XII` -> "Глава двенадцатая", `CHAPTER IV` -> "Chapter four"; ALL-CAPS headings become normal text |
| Say numbers in words | integers, decimals, ordinals (`5-му` -> "пятому"), years ("в 1999 г." -> "в тысяча девятьсот девяносто девятом году"), dates, percents, money, units; Russian case/gender/number agreement is derived from the surrounding words |
| Expand abbreviations | `т. д.`, `и т. п.`, `им.`, `г.`, `Dr.`, `etc.` ... (context-aware: "г." after a year vs. "г. Москва") |
| AI: fix typos and add missing commas | Russian only; see below |

Languages: the rule steps cover **Russian and English** completely (own number-to-words code in `core/num_words.py`; `num2words` was not used because it is LGPL-2.1 and does not decline numbers by the following word). For other languages
(German is the first one with a UI translation) only the language-neutral steps (layout, quotes, noise, links) run; digits and abbreviations are left for the TTS engine's own normalizer. The language is taken from the book's metadata, else
detected from the text. Very long sentences are not split by the preparation step: the chunker already splits them at clause boundaries.

**AI clean-up (Russian).** Model: `ai-forever/sage-fredt5-distilled-95m` (SAGE, MIT licence, about 365 MB, pinned revision `ed51b4a`...). It is **not shipped**: the row shows "Needs a one-time download (365 MB)" with a *Download* button; afterwards the
box is ticked by default for Russian books and greyed out ("Russian books only") for others. The model is only a *proposer*; a rule-based **validator** (`core/text_cleanup.py`) decides what is applied: it accepts only close spelling fixes of
a word (edit distance <= 2-3, the source word rare in the book itself, no names, no digits, same letter case), `е` -> `ё` in known cases and inserted commas / semicolons / colons / dashes (at most one per four words); everything else - deletions, rewording, changed
punctuation, case changes - is rejected, and a paragraph where the model wants to change more than 5 % of the words is left untouched. Results are cached per paragraph (`.cache\cleanup.json`), so a resumed job does not run the model again.
*Not verified on real hardware*: the `SageEngine` (transformers `AutoModelForSeq2SeqLM`, greedy generation) and the real download have never run - tests use a fake engine and a fake `snapshot_download`.

**Announced, not available yet** (greyed out under *More preparation (coming later)*, entries exist in the registry `infra/text_models.py` as placeholders): punctuation model for fluent reading pauses (RUPunct), Russian stress marks and `ё`, translation before narrating
(Opus-MT / MADLAD), multi-voice speaker markup, and English / German neural spell checkers.

### My voices and licences
*My voices* lists the library as cards: name, language, length of speech, epochs, voice type, author, a **licence badge** (green: commercial use allowed, amber: personal use only), **Preview** (plays the reference sample),
*Narrate with this voice*, *Details…* (edit name, author, licence, description) and *Delete* (asks first). **Import voice…** accepts an adapter folder or a `.zip` (archives are checked: no path tricks, only the
adapter files, size limits). A voice without a declared licence is treated as `custom/personal-only`.

| Licence | Commercial use |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | allowed (attribution / share-alike per the licence) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | not allowed |
| custom/personal-only (default) | not allowed |

`commercial_use` in `voice.json` is always derived from the licence, never trusted from a file you import. The badge is information, not legal advice.

**Download voices from repository.** The button reads a static `index.json` (schema 1) from a configurable URL: environment variable `VOXPRINT_VOICES_INDEX`, or the first line of
`%LOCALAPPDATA%\Voxprint\state\voice_index_url.txt`. The built-in default is a placeholder (`.../OWNER/voxprint-voices/...`): until a real repository exists the dialog says "not set up yet". An empty list and no network
are handled with friendly messages. Downloads are HTTPS-only, size-capped, the SHA-256 from the index must match, and the licence in the index is the one shown. Index format:
```json
{"schema": 1, "voices": [{"id": "anna-ru", "name": "Anna", "language": "russian", "author": "...", "license": "CC-BY-4.0",
  "description": "...", "voice_type": "female", "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "url": "https://example.org/anna-ru.zip", "sha256": "<64 hex digits>", "size_bytes": 61000000}]}
```

### AAC / M4B: patents (please read)
> The **M4B (AAC)** export exists only as a convenience for **Apple Books compatibility**. The **AAC codec is patent-encumbered** and not fully open. **This project does not provide a patent licence** for it.
> **You are solely responsible for any legal compliance** (patent licensing, royalties, distribution rules that apply to you) when you choose this format. The default formats (Opus, MP3 per chapter, FLAC, WAV) are not affected.

The same text (localized into English, Russian and German) is shown in the program when M4B is selected, and in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Hide or disable AAC.** One switch, three ways (highest priority first):
* environment variable `VOXPRINT_ENABLE_AAC=0` (`1` forces it on);
* `%LOCALAPPDATA%\Voxprint\state\features.json` containing `{"aac_m4b": false}`;
* the default `AAC_DEFAULT` in `infra/features.py` (a distributor can set it to `False`).

When disabled the M4B entry disappears from the window and the narrator refuses to produce M4B/AAC even if asked programmatically. The encoders themselves (`aac`, `libopus`, `libmp3lame`) come from the ffmpeg build in use; the export checks for them
before synthesis starts and stops with a clear message if one is missing (whether the pinned ffmpeg build contains all three has **not** been verified yet).

## How it works
```
 audio + text
     │  read + normalize (numbers/abbreviations spelled out, UTF-8/cp1251)
     ▼
 forced alignment ── Qwen3-ForcedAligner (long audio: cut at pauses, chunks ≤ 150 s, text split by clauses)
     │  word timestamps
     ▼
 slicing ── 3-12 s clips cut at pauses, never mid-word, + ~1 s trailing silence
     ▼
 quality filter ── drops clipped / silent / noisy clips (adaptive if too many are rejected)
     ▼
 dataset (Alexandria format) ──► LoRA training on the Qwen3-TTS talker ──► adapter + voice.json
                                                                              │ optional
                                                                              ▼
                                                                   merged "universal" model (~4 GB)
```
Narration: `book → chapters → chunks → (cache hit or Qwen3-TTS + adapter) → WAV with pauses per chapter → ffmpeg → .opus / .mp3 / .m4b ...`
(`core/book_parsers.py`, `core/chunker.py`, `core/narration.py`, `core/tts_engine.py`, `core/audiobook_export.py`).
Heavy work runs in background threads (`workers/`); the UI only reacts to progress signals. Before anything else a quiet weekly update
check runs (it never changes components in *your* environment without asking). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and data flow.

Hyper-parameters are chosen automatically (Alexandria `lora.md`): LoRA r=32, alpha=128 on q/k/v/o_proj of the **talker**, batch 1 with
accumulation 4-8, lr 1e-6 (< 90 fragments) or 2e-6, epochs ~ 320 / fragments, eager attention, bf16, gradient checkpointing; a warning is
shown if the final loss is < 3.5 (the threshold is not confirmed for Russian).

## Output format
Alexandria `train_lora.py` contract:
```
dataset\  metadata.jsonl  {"audio":"segment_001.wav","text":"...","ref_audio":"ref.wav"}  (UTF-8, relative paths)
          segment_NNN.wav (24 kHz mono, 3-12 s of speech + ~1 s trailing silence, cuts at pauses, never mid-word)
          ref.wav (24 kHz, 5-10 s, the cleanest fragment)   ref_text.txt (exact transcript of ref.wav)
          report.json (incl. text_raw - the original text before normalization, quality drops, training language)
output\<voice name>\  adapter_model.safetensors, adapter_config.json, ref_sample.wav, training_meta.json, voice.json
          checkpoints\epoch_NN\ (adapter copy after every epoch; may be deleted)
          merged_model\ (only after "Build universal model": model.safetensors bf16 ~4 GB, config.json with
          tts_model_type=custom_voice + talker_config.spk_id, speech_tokenizer\, ref_sample.wav, ref_text.txt,
          speaker_embedding.safetensors, voxprint_voice.json, USAGE.txt)
```
`voice.json` (schema 2) example:
```json
{"schema": 2, "id": "my-voice", "name": "My voice", "language": "russian", "created": "2026-10-03T12:00:00Z",
 "duration": 412.7, "epochs": 15, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "author": "Aleksandr",
 "license": "custom/personal-only", "license_url": "", "voice_type": "female",
 "description": "Warm narrator voice, calm pace", "commercial_use": false}
```
`voice_type` (`male|female|child|other`) and `description` (whitespace collapsed, ≤ 500 characters) are optional (empty strings when unset). `license` is an SPDX-like id (see the table above; default
`custom/personal-only`), `license_url` is filled for the known licences, and `commercial_use` is **derived** from the licence. Schema-1 files (`voice_name`, `speech_seconds`) are migrated when read.
Library layout: `%LOCALAPPDATA%\Voxprint\voices\<id>\` = `adapter_model.safetensors`, `adapter_config.json`, `ref_sample.wav`, `training_meta.json`, `voice.json`.

Audiobook output: `<output>\<Book title>\` with the chosen files (`Author - Title.opus` / `.m4b` / `.mp3`, or folders `... - MP3`, `... - Opus`, `... - FLAC`, `... - WAV` with `NN - Chapter` files and a `.m3u8`);
the resumable chunk cache lives in `.cache\` and temporary files in `.work\` inside that folder and are removed after success.

Using the merged model: `Qwen3TTSModel.from_pretrained(folder).generate_custom_voice(text, language="Russian", speaker="<voice name>")`.

## Hardware requirements
| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 24H2 (build 26100) x64 | Windows 11 26H2 |
| GPU | NVIDIA with ~6 GB VRAM (0.6B model, 8-bit Adam) | NVIDIA with 16 GB VRAM (1.7B model); a 4090 peaked at 6.1 GB whole-GPU usage for a 170 s recording |
| Without NVIDIA | dataset only; training on the CPU is possible but very slow | - |
| Disk | ~12 GB (models ~7 GB, program 3.6 GB installed) + ~4.2 GB for the optional universal model | SSD |
| Network | needed once for the model download (Hugging Face, or the ModelScope mirror) | - |
| Driver | NVIDIA driver with CUDA 11.8+ (PyTorch flavor cu118-cu130 is picked from `nvidia-smi`) | current driver |

The VRAM tiers used by the planner: >= 14 GB -> 1.7B; >= 10 GB -> 1.7B + 8-bit Adam; >= 6 GB -> 0.6B + 8-bit Adam; otherwise CPU.

## Reusing what is already on your computer
Before downloading a Qwen model Voxprint looks for a complete copy left by another app and uses it **in place - read-only, nothing
is copied, moved, changed or deleted** (`core/model_locator.py`). Where it looks (confidence: high = verified in code/docs, medium = derived, not checked on a real install):

| Source | Location | Confidence |
|---|---|---|
| Standard Hugging Face cache (also what **Alexandria** uses) | `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` (Windows: `%USERPROFILE%\.cache\huggingface\hub`); layout `models--Qwen--<Name>/{blobs,refs/main,snapshots/<sha>}` | high |
| Alexandria installed via **Pinokio** (Pinokio sets `HF_HOME=./cache/HF_HOME`) | `<pinokio>\api\alexandria-audiobook.git\cache\HF_HOME\hub` (folder name inferred, so `api\*\cache\...` is scanned), or `<pinokio>\cache\HF_HOME\hub` | medium |
| Alexandria in Docker | volume inside Docker - not reachable from the host (use `VOXPRINT_MODEL_DIRS` with a bind mount) | high |
| ModelScope cache | `%MODELSCOPE_CACHE%` or `~/.cache/modelscope`, then `hub/models/<Owner>/<Name>` (dots in the name become `___`) | medium |
| Manual downloads (`--local_dir`) | `~/models`, `./models`, the working folder: `<Name>`, `<Owner>/<Name>`, `<Owner>--<Name>`; a previous Voxprint install | medium |
| Your own list | `VOXPRINT_MODEL_DIRS` (paths separated by `;` on Windows) - searched first. `VOXPRINT_NO_EXTERNAL_MODELS=1` switches reuse off | - |

A copy is accepted only if it is **complete** (valid `config.json`, every file of the repository incl. `speech_tokenizer/`, consistent and non-truncated
`*.safetensors`, all shards, resolvable symlinks, no `*.incomplete`) and matches the **verified revision** from `infra/verified_manifest.json`
(HF snapshot folder name = commit sha; folders without a sha are accepted only when the weight-file sizes equal the verified commit).
On a mismatch the verified revision is downloaded. The updater never touches reused copies.

**Download mirror.** If Hugging Face is slow or unreachable (6 s probe; skipped when you set your own `HF_ENDPOINT`) or fails, Voxprint downloads from
**ModelScope** (modelscope.cn, org `Qwen`; same repository ids, byte-identical file sizes - verified 2026-10-03) and confirms the revision by file sizes.
Downloads resume after an interruption. `VOXPRINT_NO_MIRROR=1` disables the mirror.

**Components.** The same idea applies to Python packages and tools (`infra/env_probe.py`, read-only): installed and current / proven compatible -> reused;
missing -> the newest verified version is installed into **Voxprint's own environment** (its venv / `packages/` overlay), which is auto-updated.
An *outdated component in your own environment* (system Python, Pinokio, conda) is **never touched silently**: a one-click prompt offers to upgrade it
and says exactly what changes. *Upgrade* -> pip upgrades it there (smoke-tested, rolled back on failure); *Not now* and still compatible -> used as is;
*Not now* and too old -> Voxprint uses its own copy and your environment stays untouched. torch: any build is reusable, but its CUDA flavor must fit the driver.
ffmpeg: a system one is used only if `ffmpeg -version` works, otherwise a pinned LGPL build (sha256, staging, smoke test, atomic swap, rollback).
**Unsloth is deliberately not used**: as of 2026-10-03 it has no Qwen3-TTS fine-tuning support and its dependency pins conflict with the verified set;
Voxprint trains with its own LoRA loop.

## Backup, restore and existing models
**Settings (gear) -> Models and voices.**
* **Back up models and voices...** - pick any folder or drive (external disk, NAS share). Voxprint copies to `<folder>\Voxprint-backup\`: every complete model it uses (its own models folder plus complete copies from the Hugging Face cache of other
  programs: TTS bases, forced aligner, clean-up models), the pinned ffmpeg build if it was downloaded, and - with *Include my voices* ticked (default) - the voice library. Before anything is written you see the size and the free space of the target;
  if it will not fit, you get a clear message ("needs X, Y free") and **nothing** is copied. Progress and Cancel are shown in the dialog.
* **Resumable, skips identical files.** Each file is written as `name.part` and renamed when complete; a cancelled / crashed run loses at most that file and the next run continues. A file is skipped when the size and modification time match
  (or, if the time differs, the SHA-256 matches). `voxprint-backup.json` (the manifest) lists every file with size, SHA-256 and time; an interrupted run never removes items the manifest already vouched for. The backup is plain files: browse it, zip it, copy it by hand.
* **Restore from backup...** - pick the folder that contains `Voxprint-backup` (or that folder itself). Every file is verified against the manifest hash (the copy is read back); models are staged in `<name>.restoring` and swapped in only when complete, so a half-restored model is
  never used. A model that differs from the backup is replaced after the new copy is verified; a **voice that exists with different content is never overwritten** (it is reported). Files already identical are skipped.
* **Existing models folder** - name a folder with models from a previous install (an old `...\Voxprint\models` folder, a Voxprint backup, a Hugging Face cache). Before downloading anything, Voxprint **imports** a complete copy from there
  (`infra/existing_models.py`, ahead of the read-only reuse above): a **hard link** if the folder is on the same drive (instant, no extra space), otherwise a **copy**; verified by SHA-256 (against the backup manifest when there is one, else by reading the copy back);
  the same completeness and revision rules as for any foreign copy apply, and a failed or corrupt import falls back to the normal download. The source folder is never modified. Storage: `state\existing_models_dir.txt` (override: `VOXPRINT_EXISTING_MODELS`);
  `VOXPRINT_NO_EXTERNAL_MODELS` does not switch off this explicitly chosen folder. Choosing a folder in Settings offers to import right away (nothing is downloaded); the installer page sets the same file and the first start imports.
Limits: resume works per file (a half-copied multi-GB file starts over); not verified on real removable media / NTFS hard links / a NAS (tests use temporary folders and fake disk usage).

## Tested on Windows
Real run on **Windows Server 2025 + NVIDIA RTX 4090** (driver 610.88 / CUDA 13.3, Python 3.11.9, torch 2.14.1+cu130), 2026-10-03.
The 4090 has 22.5 GiB; the app was tested against a 16 GiB cap to emulate a 16 GB laptop GPU.

| Area | Result |
|---|---|
| Test-suite on Windows | **236 passed, 1 skipped** (2 Windows-only failures found and fixed) |
| Install (Python, Inno Setup, venv, torch cu130, dependencies) | passed, about 2:50 with `uv`; silent installer run 88 s |
| Model download (Hugging Face) | aligner 34 s, TTS-1.7B 58 s; reuse from another app's HF cache: 0 files copied |
| Pipeline A - SAPI voice, English, 170.6 s | 20 segments, 154.9 s kept, 0 dropped; LoRA r=32, 15 epochs: **112 s** (~7.5 s/epoch); peak VRAM **5.18 GiB** (torch) / **6.13 GiB** (whole GPU); adapter **58 MB**; merged export **13 s, 4.21 GiB** |
| Pipeline B - LibriSpeech, 293 s, unpunctuated text | 30 segments (5.4-10.8 s), 721 of 721 words used; cut-edge error mean 0.33 s (start) / 0.24 s (end); 11 epochs in 130 s |
| ASR round trip (Qwen3-ASR-0.6B) on generated speech | 2 of 2 sentences transcribed exactly (WER 0) |
| Adapter loading | `PeftModel.from_pretrained` + voice-clone generation works; merged model generation works |
| Build | PyInstaller onedir 3.59 GiB; installer 1.83 GiB; full `build.bat` about 16 min |
| Installer | silent install, installed exe `--selftest-imports` and `--selftest` OK, uninstall in 4 s |
| `--verify-install` / `--repair` | passed after two fixes (idempotent) |
| Input formats | mp3 passed; m4a passed after a fix; Russian text handling (UTF-8/cp1251, normalizer) passed |
| GUI as a normal window | verified at 1920x1080 at 100/125/150% and 1366x768 at 125% |

**Not verified (honest caveats):** voice *similarity/quality by ear* was not judged (the loss stays above 3.5 with few epochs; the ASR check proves
intelligibility only); real **Russian** audio through alignment and training was not run (only English audio; Russian is verified for text handling);
a long training run started from the GUI and the frozen exe's full pipeline were not exercised; loading the adapter in Alexandria with its own pinned
peft was not tested; the ModelScope fallback and the pinned LGPL ffmpeg download were not re-tested after a certificate fix; the 8-bit optimizer was not used in training.
Cosmetic: the `sox` package prints a "SoX could not be found" notice on import; the exe has no version info yet.

The Linux test-suite (`QT_QPA_PLATFORM=offscreen python -m pytest`) covers the pipeline with synthetic audio and a tiny randomly initialised model.

## Privacy
**Use voices responsibly.** A voice is personal data and a personality right of its owner: use it only with the owner's explicit permission and within the licence in `voice.json` (for example CC BY-NC means personal use only). No impersonation, fraud or misleading content; label synthetic speech where required; you are responsible for lawful use. Never commit recordings of other people without their permission. Recordings, text and the finished voice stay on your computer.
Voxprint sends nothing to the internet except: downloading models (Hugging Face, or the ModelScope mirror), checking PyPI / the Hugging Face API for
updates, the voice repository (only when you open *Download voices from repository*, and only the index and the voices you pick), and (only if you accept an offer) installing packages. There is no telemetry and no account. A notice is shown on first start and a reminder
sits at the bottom of the window. Logs (`%LOCALAPPDATA%\Voxprint\logs`) stay local too.

## Roadmap
Ideas, not promises:
* **Narrator improvements** - a queue of several books, more book formats (e.g. PDF, DOCX).
* **Preparation, next steps** - punctuation restoration for Russian (RUPunct), stress marks and `ё`, English / German neural spelling, and tuning of the abbreviation / unit tables on real books (the rule steps themselves are done).
* **Translation** - translate a text and speak it in your own voice in another language (a registry entry and a greyed-out placeholder exist, not implemented).
* **Multi-voice markup** - mark up a text with several speakers / voices and render them in one pass (placeholder only).
* A real **voices repository** (the index URL in the program is a placeholder today).
* **Android** - later: convert the merged model for phones (GGUF / LiteRT; today the merged folder is a plain Hugging Face directory a converter can start from).
* **Installer channels** - *online* (small installer, downloads components), *offline* (everything included), and *beta* (pre-releases).
* A public repository URL and a published licence for the Voxprint source code.
* More languages in the UI (see "Adding a language" below) and Linux/macOS builds.

## Contributing
Contributions are welcome - see [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, tests, code style, localization, how to propose changes) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (module map, data flow). Quick start for developers:
```bash
python -m venv .venv && . .venv/bin/activate        # Linux: enough to run the tests
pip install -r requirements.txt -r requirements-verified.txt -r requirements-dev.txt
pip install --no-deps -r requirements-nodeps.txt
QT_QPA_PLATFORM=offscreen python -m pytest          # ~350 tests, no GPU, no network
python -m core.cli audio.wav text.txt --out dataset --fake-aligner     # dry run of the pipeline without a neural network
```
CLI for stage-by-stage checks on a GPU machine: `python -m core.cli audio.wav text.txt --out dataset [--language Russian] [--device cuda] [--train --output-dir output]`.
Regenerate the notices after editing `credits.json`: `python tools/gen_notices.py` (a test checks they are in sync).

### Localization
Supported UI languages: **English (default), German, Russian**. Catalogs are flat JSON files `locales/{en,de,ru}.json` (`"ui.start": "...{name}..."`);
`core/i18n.py` provides `tr(key, **params)`. Language order: `VOXPRINT_LANG`, saved choice (`state\language`), Windows user locale, English. A system language that
is not supported (e.g. Ukrainian) falls back to English. Tests check that all catalogs have identical keys and placeholders and that no user-facing literals are left in the code.

#### Adding a language
Ukrainian and Belarusian were dropped on purpose (fewer languages to keep in sync); their last complete catalogs are in git history
(`git show cdea827:locales/uk.json`, `...be.json`; component texts in `git show cdea827:credits.json`). To add a language `xx`:
1. `core/i18n.py`: append `"xx"` to `LANGS` and its own-language name to `LANG_NAMES` (the language switcher and the system-locale mapping `normalize_code` are driven by these two).
2. `locales/xx.json`: copy `en.json` and translate all values; keep every key and every `{placeholder}` (a test enforces both).
3. `credits.json`: add an `"xx"` text to every `purpose` (and `note`) entry - the credits test requires all `LANGS`.
4. Run `python -m pytest` - the parity tests (`tests/test_i18n.py`, `tests/test_credits.py`) and the per-language UI tests iterate over `i18n.LANGS`; update the
   places that count languages (the switcher test in `tests/test_i18n.py` and the `len(seen) == 3` checks in `tests/test_env_install.py` / `tests/test_model_locator.py`).
5. Mention it in this README.

### Repository link
The GitHub URL lives in one place: `"repo_url"` in `credits.json` (read as `core.appinfo.REPO_URL`). While it still contains the placeholder `OWNER`
the link is hidden in **About**; replace it with the real URL and the button appears.

### Versions: "verified by Voxprint"
`infra/verified_manifest.json` pins the package versions and model revisions (HF commit shas) Voxprint was tested with. The updater installs exactly those
(rolling back if needed: `restore_verified()`) and merely logs newer PyPI releases as "not verified yet". Channel "latest": `VOXPRINT_CHANNEL=latest`
(or `{"channel":"latest"}` in `state\updater_state.json`). `requirements-verified.txt` / `requirements-nodeps.txt` are generated from the manifest
(`python -m infra.verified_manifest [--nodeps]`; a test keeps them in sync).

## Licences and third-party components
Voxprint stands on open-source software; the single source of truth is `credits.json`. From it come the **About** list, `THIRD_PARTY_NOTICES.md`
(shipped by the installer together with the `licenses\` folder of full licence texts) and the build-time appendix with the licences of all installed
packages (`tools\gen_notices.py --with-installed`, called by `build.bat`). Voxprint's *own* source-code licence has not been chosen yet and will be added
before the first public release. Important points:
* **Qt / PySide6 - LGPL-3.0.** Used unmodified and dynamically linked. To let users replace the Qt/PySide6 libraries, prefer `build.bat onedir`
  (the libraries are ordinary files); `--onefile` makes this harder. The LGPL/GPL texts and the source pointers (<https://code.qt.io>, <https://pyside.org>) are included.
* **FFmpeg is a GPL-3.0 build.** The ffmpeg binary inside the `imageio-ffmpeg` wheel was built with `--enable-gpl --enable-version3` (verified in the Windows 7.1 binary of
  imageio-ffmpeg 0.6.0). It is a separate executable started as a subprocess, but you are distributing a GPL binary: keep `licenses\gpl-3.0.txt` and the source link,
  or ship an LGPL ffmpeg build / require ffmpeg on `PATH`.
* **AAC (M4B) is patent-encumbered.** Voxprint provides no patent licence for it; the option is a convenience for Apple Books, is off by default, carries a disclaimer and can be disabled completely (see
  [AAC / M4B](#aac--m4b-patents-please-read)). The same notice is in `THIRD_PARTY_NOTICES.md`.
* **soynlp (GPLv3) is excluded.** `qwen-asr` imports it lazily, only for Korean. It is not in `requirements.txt` and the build excludes it (`--exclude-module soynlp`),
  so **Korean alignment is unavailable**.
* **CC-BY-NC-4.0 model.** The optional backup aligner model `MahmoudAshraf/mms-300m-1130-forced-aligner` (used only if you install the optional `ctc-forced-aligner`) is
  non-commercial. The default Qwen models and libraries are Apache-2.0 / MIT / BSD-style.
* PyInstaller is GPL-2.0-or-later **with a bootloader exception** (apps may use any licence); Inno Setup has its own permissive licence.

## What was verified against primary sources
* **qwen-asr 0.0.6** (PyPI sources): `Qwen3ForcedAligner.from_pretrained(path, dtype, device_map)`, `.align(audio=(np, sr), text, language="Russian")` ->
  `ForcedAlignItem(text, start_time, end_time)` in seconds; the unit is a space-separated word and punctuation is dropped, so the text is normalized first.
  The package cuts audio at 180 s and the model handles ~5 min: recordings are cut at pauses into chunks <= 150 s (`align_long`).
* **Alexandria** (`train_lora.py`, `tts.py`, `lora.md`): dataset and adapter formats, the teacher-forcing input (ported to `core/teacher_forcing.py`), peft on `hf_model.talker`,
  `PeftModel.from_pretrained(talker, adapter_path)`. Verified on CPU with real qwen-tts classes and a tiny model, including reload (peft 0.18.1 and 0.21.2).
* **Merged model** (`core/model_export.py`, mirrors the official `finetuning/sft_12hz.py`): on the tiny model the merged talker output equals the adapter model output (atol 1e-4),
  the folder has no `lora_`/`speaker_encoder` keys, `codec_embedding[spk_id]` holds the speaker embedding, and the weights reload.
* **Qwen3-TTS-12Hz-1.7B-Base on HF**: the audio tokenizer is inside the repo (`speech_tokenizer/`).
* **DWM** (Microsoft Learn): `DWMWA_USE_IMMERSIVE_DARK_MODE=20`, `DWMWA_WINDOW_CORNER_PREFERENCE=33`, `DWMWA_SYSTEMBACKDROP_TYPE=38`; backdrop types AUTO 0, NONE 1,
  MAINWINDOW 2, **TRANSIENTWINDOW 3 (Acrylic)**, TABBEDWINDOW 4. Only the native DWM through `ctypes` is used (no GPL frameless-window libraries).
* Licences in `credits.json` were read from PyPI metadata, GitHub and Hugging Face cards on 2026-10-03.
* **Model reuse / mirror / environment probe** (tests with fake directory trees and fake machines): HF cache layouts incl. symlinks, refs/snapshots, truncated and sharded weights,
  Pinokio/Alexandria-style tree, ModelScope layout, pinned-revision mismatch, read-only guarantee, resumable ModelScope download (fake server), reuse/upgrade/install decisions,
  torch flavor mapping, completion manifest + reason codes, pinned-asset installer (sha256, staging, smoke test, rollback).

## Still to verify (TODO-needs-GPU-test)
* **The narrator on real hardware.** `core/tts_engine.py` (loading Qwen3-TTS with the LoRA adapter and `generate_voice_clone`) is written against the qwen-tts API but has **never run on a GPU**; everything around it (parsers, chunker, cache/resume,
  assembly, ffmpeg command lines, UI) is tested with a fake engine and a fake ffmpeg. Also untested: the real encoders in the pinned ffmpeg build (`libopus`, `libmp3lame`, `aac`, MP4 with Opus, Opus chapter markers, ID3 CHAP frames),
  preview playback (QtMultimedia) on Windows, how chapters show up in the listed players, and the Studio/Acrylic look on a real desktop.
* **Backup / restore and the installer page on real hardware.** `infra/backup.py` / `infra/existing_models.py` are tested with temporary folders and fake disk usage only (cancel/resume, space check, hash mismatch, hard link vs copy). Untested: real USB/NAS targets, NTFS hard links, exFAT timestamps, multi-GB files. `installer/Voxprint.iss` (trilingual texts, the *Existing models* page, `SaveStringsToUTF8File`) was linted by tests but **never compiled with Inno Setup**.
* **Text preparation on real books.** The rule steps are unit-tested on synthetic Russian/English samples only (not on real FB2/EPUB corpora); the AI clean-up (`SageEngine`, model download, speed on CPU/GPU, how many of its proposals the validator accepts) has never run for real.
* Reuse against a **real** Alexandria/Pinokio install on Windows; a real ModelScope download after the certificate fix; the pinned LGPL ffmpeg download and swap on Windows (file locking).
* Alignment quality/speed and the `align_long` and quality-filter thresholds on real (non-synthetic, non-English) recordings.
* Training thresholds (loss 3.5, lr 1e-6...2e-6) for **Russian**; reading the adapter in Alexandria with its pinned `peft==0.18.1`; the optional `ctc-forced-aligner`.
* The frozen exe's full pipeline, a long training run from the GUI, and the real-world behaviour of the Acrylic backdrop with the new darker theme.
