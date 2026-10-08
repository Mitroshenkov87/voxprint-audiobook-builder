# User guide

Start the program: the **Studio** opens. *Narrate a book* is the main card; if the library has no voices yet it says so and points to *Train your voice* / *My voices*.
Use **← Studio** in any window to come back; the gear (top right) opens Settings.

## Train your voice
1. Press **Choose audio** and **Choose text** (or drag the files into the window). The text is a UTF-8 `.txt` of what you read.
2. Optional, under the main button: pick the **gender** (male / female) and **age group** (child / young / adult / elderly) and type a short **description**; *More about the voice* adds speaker, prepared by, organization and a project link. Everything is saved in `voice.json` and can be changed later in the voice's Properties.
3. Press **Create voice (LoRA)** - everything else is automatic. When it finishes, the folder opens:
   `...\<recording name>_Voxprint\dataset` (the dataset) and `...\output\<voice name>` (the small adapter, tens of MB).
   **Create dataset** builds only the dataset.
4. Optional: **Build universal model (~4 GB)** merges the trained adapter into a standalone model in `...\output\<voice name>\merged_model`
   (Qwen3-TTS `custom_voice` format; the voice is `speaker=<voice name>`). The small adapter works only in a few apps (e.g. Alexandria);
   the merged model works in any app that runs Qwen3-TTS. Voxprint checks the free disk space and asks for confirmation first; the big model is never built automatically.
5. When training finishes the voice is **registered in the library** automatically (status line "voice registered"; if that fails you get a warning, the adapter folder is still there).
6. The **gear** button opens **Settings**: language, *Check for updates* (also checked weekly in the background), open the models / data & log folders,
   *Repair the installation*, *Auto-repair* (checks every component and model file by checksum, downloads what is missing and replaces only damaged files; also `Voxprint --auto-repair`), *Preload models into memory at startup* (off by default; offered only when the PC has enough RAM for the installed models - the note next to it says how much is needed; the models are freed again when memory runs low), and *About* (help, authors, open-source components with their licences, third-party notices).

Use voices responsibly: only with the voice owner's permission, in line with the voice's licence (see [Privacy](PRIVACY.md)).

## Training presets

The Train window offers **Fast / Balanced (default) / Maximum / Manual**, each with a short meaning and an **estimated training time for your graphics card and recording**.
Balanced is the automatic plan. Fast uses about half the passes (epochs) and a half-size adapter (LoRA rank 16); Maximum about twice the passes and rank 64; Manual opens the *Advanced* panel
(epochs, rank, alpha, learning rate, gradient accumulation). The learning rate is never raised by a preset: in our first real GPU test, 15 epochs at 3e-6 and 8e-6 gave a lower training loss
but a voice that babbled and never stopped, so the loss alone is not a quality signal.
The estimate is calibrated on a measured run (RTX 4090: 24 s per epoch for 70 clips, 12 s model load; other cards are scaled by a table, unknown ones by VRAM, a CPU is roughly 40x slower and is not measured);
read it as "about".

## No transcript? (audio only)

Tick **I have no transcript** in the Train window and the program recognises the speech itself (Qwen3-ASR, offline; the model is part of the first-run / installer download-all: Qwen3-ASR-1.7B, ~4.7 GB, on NVIDIA graphics cards with 8 GB of video memory or more, else Qwen3-ASR-0.6B, ~1.9 GB; **Settings -> Speech recognition model** can force one of them or keep both).
You can choose **many audio files at once**, add a **folder** (searched recursively) or drag and drop them; each file is recognised separately. Long recordings are cut at pauses
into pieces of at most 14 s, short clips are used as they are, and everything is merged into **one dataset** (volume-normalised to about -20 dBFS).
Skipped automatically: clips under 1.5 s or over 20 s, empty or **implausible recognition** (looping text, absurd speaking rate, wrong script - a score computed from the clip
and the text, because the ASR API gives no per-token confidence), duplicates (identical audio, or the same words at the same length) and clips that fail the usual signal-quality filter.
When it is done the window reports the totals: files, kept clips and how many minutes of the audio were kept. Automatic recognition can contain errors, so a clear localized warning is shown and
**an explicit "I understand" tick is required** before the buttons work; the dataset's `report.json` records every clip's source file and plausibility.

## Quick preview and automatic voice check

Before a long run, **Quick preview…** in the Train window trains a short adapter on a small subset (at most 12 clips, at most 4 epochs, capped at about 3 minutes of estimated time; the estimate is shown next to the button),
then synthesizes a ~10 s sample and plays it in the app. Each sample shows **automatic metrics**: pitch distance to your recording in semitones and the word error rate of a Qwen3-ASR re-reading of the sample, with a good / warning / bad verdict
(thresholds: pitch 3 / 6 semitones, WER 0.25 / 0.5; a sample that runs to the length cap is flagged as babbling).
**Compare 2 variants** makes two previews: A with the current settings, B with rank and alpha doubled and 1.5x the epochs (same learning rate). Listen, press **Use these settings** under the better one: the Manual preset is filled in and the full run uses it.
**Voice strength**: every variant is synthesized at three adapter strengths (0.35, 0.50 and 1.00 = exactly as trained); the selector in the row switches what *Play* and the metrics show, the best-checking one is pre-selected. *Use these settings* also stores the chosen strength with the trained voice. Without a choice a new voice gets 0.50 (or the value picked by the automatic checkpoint selection); voices trained before this setting keep 1.00. The strength can be changed later in the voice's **Properties** (0.10-1.00); cached chapters are then synthesized again.
**Cancel** stops after the current step; the trainer's own VRAM handling applies (a preview never raises the learning rate). A preview adapter is temporary and is not added to *My voices*; the full run re-runs the alignment check.
The same check can run after a full training (tick *Check the voice after training*, on by default): a poor result adds a warning with a suggestion (e.g. fewer epochs, lower rank). Metrics are a hint, not a verdict: the loss is not a quality signal, always listen.

## Voice owner's consent and usage scope

End the recording with **one spoken consent sentence** (the recording script, block 18, offers three templates in Russian, English and German: your name, the date, and what is allowed).
After training the program recognises the statement (Qwen3-ASR, rule-based keyword reading in ru/en/de), shows the detected **name, date and scope** and asks for one click to confirm or change it.
If the statement is unclear or missing, the strictest level is used. You can also choose the scope by hand, or train without a statement (private only).

| Scope | Meaning | Licence field it maps to |
| --- | --- | --- |
| Commercial | the audio may be sold | `CC-BY-4.0` |
| Public, non-commercial | may be published free of charge, not sold | `CC-BY-NC-4.0` |
| Private only | may narrate, but results stay on the user's computer | `custom/personal-only` |

`voice.json` gets a `consent` block (`scope`, `name`, `date`, `recorded_statement`, `method` = `spoken` / `spoken_confirmed` / `manual` / `none`, `confirmed`, the recognised `statement`, and the clip name); the spoken clip
(the last 45 s of the recording) is stored next to the adapter as `consent_statement.wav` unless you untick it. *My voices* and *Narrate* show a scope badge; Narrate repeats the limit under the voice and when the audiobook is ready
(a private voice: "keep it on your computer, do not publish or sell"). This is a record of what the speaker said, not legal advice, and it does not replace the law of your country.
When a statement ends the recording, up to 60 s of unscripted speech at the end is tolerated by the alignment check (15 s otherwise).

## Narrate a book
1. **Choose book…**: `.txt` (UTF-8 / cp1251; headings like "Chapter 1", "Глава 2", "Part I" become chapters), `.fb2` / `.fb2.zip`, `.epub`. The book title, author and cover are read when present.
2. Pick the **voice** (a licence badge is shown; for a personal-use-only voice you are reminded not to publish or sell the narration). No voices yet? The window offers *Train your voice* / *My voices*.
3. **Prepare the text** (automatic, see [below](#prepare-the-text)): all rule-based steps are on by default; leave them as they are or untick what you do not want.
4. Pick the **format** (see below) and a **quality** button (*Compact / Standard / High*), then **Start narration**. Rarely needed things are collapsed: *Other formats*, and *Advanced* (exact bitrates per format, the **working folder** - default `Documents\Voxprint\Audiobooks`, remembered; each book gets its own sub-folder there for everything of the job - whether chapter titles are read aloud, and a read-only sample of how the first changed paragraph looks after preparation).
5. Progress with the time left; **Pause / Resume** and **Cancel** are always available. A **mini player** appears as soon as the first fragments are ready: *Play / Pause*, a seek slider and the clock work on the finished part while the rest is still being made, and the player follows new fragments (it waits if it catches up with the synthesis). It only checks a few file names every 1.5 s and plays through Qt Multimedia, so it does not slow the narration down; when the job ends the whole result becomes the playlist. Every synthesized fragment is stored (`<output>\<book>\.cache`), so after a cancel, a crash or closing the program
   **Start continues where it stopped**; the cache is deleted after a successful run. Texts are split into chunks at sentence/clause boundaries (reusing the clause splitter of the aligner), chapters get a pause between them.

| Format | What you get | Where it plays |
|---|---|---|
| **Opus, one file** (default) | `Author - Title.opus` with chapter markers (32 kbit/s mono), open and royalty-free, small | VLC, Audiobookshelf, Voice (Android), Smart AudioBook Player and other free players; chapter support varies by player |
| **MP3, one file per chapter** | folder `... - MP3\NN - Chapter.mp3` + `.m3u8` playlist, ID3 tags (title, album, artist, track n/N, cover); 96 kbit/s mono | practically everything (old players, car radios, phones) |
| **M4B (AAC)** - opt-in, [see the notice](AAC-M4B.md) | `Author - Title.m4b` with chapters, 64 kbit/s mono | Apple Books (iPhone/Mac) and most audiobook apps |
| *Other formats* | M4B with Opus audio, Opus per chapter, one MP3 with chapter marks (ID3 CHAP frames), FLAC / WAV per chapter | - |

**Quality presets** (mono speech; `core/audiobook_export.py: QUALITY_PRESETS`):

| Preset | Opus | MP3 | AAC | About per hour of audio (Opus / MP3 / AAC) |
|---|---|---|---|---|
| Compact | 24 kbit/s | 64 kbit/s | 48 kbit/s | 11 / 29 / 22 MB |
| **Standard** (default) | 32 kbit/s | 96 kbit/s | 64 kbit/s | 14 / 43 / 29 MB |
| High | 48 kbit/s | 128 kbit/s | 96 kbit/s | 22 / 58 / 43 MB |

Changing a bitrate by hand under *Advanced* switches the quality to "Custom" (no button highlighted).

## Prepare the text
Pipeline: **rules** (deterministic, instant) -> optional **AI clean-up** (small model, on demand) -> **synthesis**. Nothing is shown for review; the original book is never modified, and chapter titles in the output
files / metadata / cover come from the original, unprepared book. For debugging, the prepared text is written next to the job cache: `<output>\<book>\.debug\prepared_text.txt` and `prep_report.json`
(counts per step, skipped steps, model revision). Unlike `.cache` they are **kept** after a successful run, so you can see exactly what was read; delete the `.debug` folder if you do not need it.

**AI disclosure.** The check box under the voice (off by default) makes the audiobook begin with a short spoken note in the book's language, in the chosen voice - e.g. *"This book was narrated with the help of artificial intelligence, voice Anna, October 2026"* (month and year only). It helps with labelling duties for AI-generated content such as the EU AI Act.

**Working folder and clean-up.** On start Voxprint asks whether the book file may be **copied** or **moved** into the book's sub-folder (or left where it is) - nothing is copied without asking. When the audiobook is ready a dialog lets you tick what to keep: the audiobook, the original book file, the prepared / translated text; temporary parts (`.cache`, `.work`) are always removed. *Keep everything* deletes nothing.

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

**Announced, not available yet** (greyed out under *More preparation (coming later)*, entries exist in the registry `infra/text_models.py` as placeholders): punctuation model for fluent reading pauses (RUPunct), Russian stress marks and `ё`, multi-voice speaker markup, and English / German neural spell checkers.

Translation before narrating (optional): see [TRANSLATION.md](TRANSLATION.md).
