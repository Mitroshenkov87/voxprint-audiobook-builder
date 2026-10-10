# What Voxprint does

**Voxprint AI Audiobook Builder** ({{studio.tagline}}) is a Windows 11 application that does two things, entirely on your own computer:

1. **It teaches a computer voice to sound like a particular person.** You give it a recording of a voice (yours, or of a person who has given you permission) – ideally 5 to 15 minutes – and, if you have it, the text that was read. Voxprint aligns text and sound, cuts the recording into clean pieces and trains a small voice add-on (an *adapter*, technically LoRA) for the speech model **Qwen3-TTS**.
2. **It reads whole books aloud with that voice.** You pick a book (TXT, FB2 or EPUB) and a voice from your library, and get a finished audiobook with chapters – one Opus file, one MP3 per chapter, and so on.

You need no command line, no browser and no account (a command line exists for scripts and agents – chapter 11). Recordings, texts, voices and results stay on the computer. The program only goes online to download models, to check for updates and, if you ask for it, to download voices from the voice repository.

![The Studio – the first screen of Voxprint (build 667).](@studio_home)

> **Beta software.** Version 0.1.3 (build 667 “Menuchah”) has run end to end on one machine (Windows 11 with an NVIDIA RTX 4090) with mainly one speaker. Expect rough edges; keep a copy of your recordings and voices (see {{backup.title}} in chapter 8).

## What you need

| | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 11 24H2 (build 26100), 64-bit | Windows 11 26H2 |
| Graphics card | NVIDIA with about 6 GB video memory (VRAM) | NVIDIA with 16 GB VRAM or more |
| Without an NVIDIA card | The dataset can still be created; training runs on the processor and takes many hours (the program warns you) | – |
| Disk space | about 30 GB for the *Full* setup (models about 26 GB + components about 3 GB) plus about 4.2 GB if you build the optional universal model | an SSD |
| Internet | during setup (the libraries) and once for the models (about 28 GB together with *Full*; 1.7B speech recognition on GPUs with 8 GB of video memory or more) | – |

Supported systems: current and previous year OS releases (Windows 11 24H2 and newer; Linux distributions released from 2025). Older systems are not a goal.

## How it works in five steps

1. Read a known text aloud and record it (5–15 minutes).
2. Choose the recording and the text file – or only the recording, if you have no text.
3. Voxprint aligns the text to the audio with a neural forced aligner and cuts it into clean fragments – the *dataset*.
4. It trains a small voice adapter on your graphics card.
5. The voice appears in **{{studio.voices_title}}**. Now **{{studio.narrate_title}}** with it.

> **Use voices responsibly.** A voice is personal data and a personality right of its owner. Use a voice only with the explicit permission of the person it belongs to and within the licence of that voice. See chapter 12.

# Installing Voxprint

## The installer

Run **Voxprint-Setup-online.exe** (Inno Setup, per-machine installation, about 34 MB; build 667 has 35,235,971 bytes and the SHA-256 `c145a9fc84c736d655fbbe9bfd5c1cc94794b1be7fb8e9a5eec293e6f1578e42`). It is an *online* installer: the libraries (PyTorch and the rest) are downloaded from PyTorch/PyPI during setup. The installer is not code-signed, so Windows SmartScreen warns – choose *More info → Run anyway* only after the SHA-256 matches the one on the release page. The wizard is available in English, Russian and German; the language you pick there becomes the program's interface language.

The page **Setup type** offers two choices. **Full** (default) downloads everything on the first start without another click – the components and all models, about 28 GB; the wizard shows the total and checks the free space first. **Quick** installs only the program; the first start opens the **{{modules.title}}** window, which offers the same complete download and starts it when you press Download. Nothing is left out in either mode.

After the install-folder page there is an optional page **Existing models (optional)**: if you already have downloaded models from an earlier installation or from a backup, choose that folder; otherwise leave the field empty. The installer copies nothing – it only remembers the folder, and the first start of the program imports the models from there instead of downloading them.

Silent installation for administrators: `Voxprint-Setup-online.exe /VERYSILENT /Mode=full /ModelsDir="D:\old\models"` (`/Mode=quick` for the Quick type).

> If the administrator account that runs the installer differs from the account that uses Voxprint, the data folder belongs to the administrator. In that case set the existing-models folder later in {{ui.settings_title}} (chapter 8).

## First start

On the first start Voxprint prepares itself. With the online installer the **{{modules.title}}** window opens and works without a click: one line shows what is being downloaded right now (*{{modules.now_download}}*), then the checksum check, and a bar shows the overall progress. Step 1 brings the program components together with the SAGE text clean-up model and the translation models, step 2 right after it ALL models (voice, alignment, speech recognition) into your models folder. Otherwise the status line says *{{ui.prefetch_start}}* and downloads the models it needs (up to about 15 GB, once). If Hugging Face is slow or unreachable, Voxprint switches to the ModelScope mirror automatically. The download can be interrupted and continues where it stopped. Models already present on the computer (for example from the Hugging Face cache or Alexandria/Pinokio) are reused without downloading them again, read-only. The bundled voices *Levi* and *Miriam* (chapter 7) are part of this download.

When it is done, the status shows *{{ui.prefetch_done}}* A privacy notice is shown on the first start; a short reminder (*{{ui.footer_privacy}}*) stays at the bottom of the windows.

## Where things are stored

| What | Where |
|---|---|
| Program data, models, logs, voice library | `%LOCALAPPDATA%\Voxprint\` (`models\`, `logs\`, `voices\`, `state\`) |
| Projects: audiobooks, voice trainings, Re-voice | the projects folder `%LOCALAPPDATA%\Voxprint\Projects` with `Audiobooks\<book>`, `Voices\<voice name>_Voxprint` and `Re-voice`; a *Voxprint Projects* shortcut is in Documents. Change it under **{{projects.title}}** in {{ui.settings_title}}. |
| Training results | `Projects\Voices\<voice name>_Voxprint`: the `dataset` and the adapter in `output\<voice name>` |

You can open the model folder and the data/log folder with buttons in {{ui.settings_title}}.

# Quick start

**I just want to hear a book (no voice of my own yet).** Open **{{studio.voices_title}}**, press **{{voices.repo_button}}** (or simply wait: online voices are listed by themselves), and download the *Open universal voice* (English, licence CC0 – free for any use). Go back to the Studio, open **{{studio.narrate_title}}**, choose a TXT/FB2/EPUB file, pick the voice and press **{{narr.start}}**.

**I want my own voice.** Record yourself reading the recording script (section 6.1), open **{{studio.train_title}}**, choose the recording and the script text, leave the training quality at *{{preset.balanced}}* and press **{{ui.btn_lora}}**. After training, confirm the consent card. The voice is now in **{{studio.voices_title}}** and in **{{studio.narrate_title}}**.

# The Studio (home screen)

The Studio is the first window. It has no settings of its own – it is a menu of four big cards and a gear.

![The Studio.](@studio_home)

| Element | What it does and when to use it |
|---|---|
| Title and tagline | "Voxprint AI Audiobook Builder – {{studio.tagline}}". Only informational. |
| **{{studio.narrate_title}}** card | {{studio.narrate_desc}} Opens the narration window. This is the main card. If the library is empty, the card says *{{studio.narrate_no_voice}}* While a book is being narrated it shows *{{studio.narrate_running}}* |
| **{{studio.train_title}}** card | {{studio.train_desc}} Opens the training window. While training runs it shows *{{studio.train_running}}* |
| **{{studio.voices_title}}** card | {{studio.voices_desc}} At the bottom of the card a counter shows how many voices you have (for example "1 voice(s) in your library" – the bundled voice). |
| **{{studio.revoice_title}}** card | {{studio.revoice_desc}} |
| Gear button (top right) | Opens **{{ui.settings_title}}** (chapter 8). |
| Note at the bottom | *{{ui.footer_privacy}}* |

Every other window has a **{{nav.back}}** button at the top left that brings you back here. Heavy work (training, narration) keeps running in the background if you go back.

# Narrate a book

Open this window from the **{{studio.narrate_title}}** card. {{narr.intro}}

![The Narrate window, upper half: book, voice, text preparation, translation and the optional AI text model. This build 667 screenshot shows Boaz, which is retired; from build 703 Levi and Miriam are the bundled voices.](@narrate_top)

## Step 1 – the book

| Control | What it does |
|---|---|
| **{{narr.choose_book}}** | Opens a file dialog. Supported: `.txt` (UTF-8 or cp1251; headings such as "Chapter 1" or "Глава 2" become chapters), `.fb2` and `.fb2.zip`, `.epub`. Title, author and cover are read when present. |
| File name and info line | After choosing, Voxprint shows the title and "3 chapters · about 7 min of audio" – the number of chapters and the expected length. Before a choice it says *{{narr.no_book}}* |

If the file cannot be opened you get a message such as *{{err.book_unsupported}}* or *{{err.book_empty}}* (see the troubleshooting chapter).

## Step 2 – the voice

The drop-down under **{{narr.voice}}** lists the voices of your library. Online voices that are not installed yet appear as "<name> (to download, <size>)"; they are downloaded and verified automatically when you press Start. Below the list a coloured **licence badge** shows what you may do with the result (see chapter 7 and the glossary in chapter 9):

* green – commercial use allowed (for example "CC0-1.0 · commercial use OK" and "Commercial OK" for the bundled voices);
* amber – restricted ("Personal use only · …", "Private only", "Public, non-commercial").

Under the badge a reminder repeats the limit, for example *{{narr.voice_personal_note}}* If you have no voices at all, the window says *{{narr.no_voice}}* and offers the buttons **{{studio.train_title}}** and **{{studio.voices_title}}**.

## Step 3 – prepare the text

*{{narr.prep_hint}}* Nothing is shown for review and your book file is never changed.

| Check box | What it does |
|---|---|
| {{prep.one}} | {{prep.one_d}} A **{{prep.model_download}}** button appears when the Russian typo model is not on this computer yet (about 365 MB). |

The rules cover Russian and English fully. For other languages (for example German books) only the language-neutral steps – layout, footnotes, quotes, links – are applied.

### Optional – translate the book

The card **{{narr.translate_title}}** has a check box (*{{narr.translate_check}}*) and a list of target languages. Voxprint detects the language of the book itself and translates it sentence by sentence **offline on your computer** with the open Opus-MT models (English, Russian and German, including Russian ↔ German directly). The models are downloaded once, with a SHA-256 check, when you press **{{prep.model_download}}**. A graphics card is used if there is one, otherwise the processor. Chapters, chapter titles, paragraphs, verse lines, scene breaks and therefore the pauses stay as they are; then the usual text preparation runs for the new language and the book is narrated in it – choose a voice of the target language for the best result.

* The translated text is saved next to the audiobook as `translation_xx.txt` (in the folder `Title (xx)`). Open it, read it and edit it; start the same narration again and **your edited text** is narrated. Delete the file to translate anew. Translated sentences are cached, so a stopped job never translates the same sentence twice.
* *{{narr.translate_note}}*
* Machine translation is not a human translation: expect wrong names, idioms, tenses and gender; short titles suffer most. Poems lose their rhyme.

### Optional – the AI text model

The card **{{llm.title}}** belongs to the optional Gemma 4 12B text model (about 7 GB with the llama.cpp runtime; it is fetched by the *Full* setup or with `voxprint models download llm`). When it is installed the card says *{{llm.ready}}* It offers two check boxes: *{{llm.literary}}* and *{{llm.prepare}}*. *{{llm.note}}* **{{llm.prompts}}** opens the folder with the instructions for the model; edited files are used instead of the built-in ones.

> Known issue in build 667: the two check boxes of this card may not react to a click even though the model is downloaded. The rule-based text preparation (step 3) is not affected. This will be fixed in a coming build.

## Step 4 – output format and quality

![The Narrate window, lower half: text preparation, output format, quality, the measured-pauses option and the Start button.](@narrate_bottom)

**{{narr.format}}** – choose one:

| Format | What you get | Where it plays |
|---|---|---|
| {{fmt.opus_single}} | one file `Author - Title.opus` with chapter markers | {{fmt.opus_single_where}} |
| {{fmt.mp3_chapters}} | a folder with `NN - Chapter.mp3`, a playlist and a cover | {{fmt.mp3_chapters_where}} |
| {{fmt.m4b}} | one `.m4b` file with chapters | {{fmt.m4b_where}} |

Choosing M4B shows a yellow patent notice: *{{narr.aac_note}}* In short, AAC is patent-encumbered, Voxprint grants no patent licence for it, and you alone are responsible for legal compliance when you choose it. If you do not need Apple Books, do not use it.

**▸ {{narr.other_formats}}** (collapsed) holds further formats: {{fmt.m4b_opus}}, {{fmt.opus_chapters}}, {{fmt.mp3_single}}, {{fmt.flac_chapters}}, {{fmt.wav_chapters}}.

**{{narr.quality}}** – three buttons that set the bitrate (the amount of data per second of sound – more data, better sound, bigger files):

| Button | Meaning | Opus / MP3 / AAC | About per hour of audio |
|---|---|---|---|
| {{narr.preset_compact}} | {{narr.preset_compact_d}} | 24 / 64 / 48 kbit/s | 11 / 29 / 22 MB |
| {{narr.preset_standard}} | {{narr.preset_standard_d}} | 32 / 96 / 64 kbit/s | 14 / 43 / 29 MB |
| {{narr.preset_high}} | {{narr.preset_high_d}} | 48 / 128 / 96 kbit/s | 22 / 58 / 43 MB |

**{{narr.pauses_enable}}** – off by default. Ticking it cuts the text at **every** comma as well and inserts measured silence there; the pauses are the clearest, but the voice may swallow short words. Only then is the slider **{{narr.pauses}}** active: five positions from *{{narr.pauses_1}}* through *{{narr.pauses_3}}* (the default) to *{{narr.pauses_5}}*; at *{{narr.pauses_3}}* a sentence is followed by 430 ms and a paragraph by 1050 ms, and the slider scales every kind of pause. Changing it for a book that was already narrated re-joins the stored fragments instead of speaking them again, as long as the text pieces stay the same.

**{{narr.check_chunks}}** – off by default. After synthesis every fragment is re-read by speech recognition; fragments that skip or repeat words are generated again.

### Pauses and reading speed

Since version 0.1.3 every narration follows the punctuation and the structure of the text, even with the box above unticked. Voxprint cuts the text per sentence and at strong transitions inside a sentence (dash, colon, semicolon, a comma before "and" / "but"), trims every spoken piece of its own silence and inserts measured silence: comma 0.25 s, strong break 0.40 s, sentence 0.60 s, paragraph or verse line 1.00 s, chapter title, chapter end or scene break 2.00 s. Plain commas stay inside a piece and no piece is shorter than 20 characters, so the voice does not swallow words.

The **reading speed adapts to the text**: long, comma-rich and descriptive sentences and scripture-like text are read a little slower, dialogue and short lines at the voice's own speed. The five pause lengths, a global **{{narrset.speed}}** (70–130 %) and the **{{narrset.style}}** (*{{narrset.style_auto}}*, *{{narrset.style_scripture}}*, *{{narrset.style_fiction}}*, *{{narrset.style_dialogue}}*) are set in {{ui.settings_title}} → **{{narrset.title}}** (chapter 8) and apply to the next narration. The speed is applied after synthesis by time-stretching with the pitch kept, so changing pauses or speed never synthesizes finished fragments again – the next run only joins them anew.

**▸ {{narr.advanced}}** (collapsed) contains: the exact bitrates per format (changing one switches the quality to "custom": no button is highlighted); **{{narr.choose_folder}}** – the folder where the audiobook is placed; the check box **{{narr.speak_titles}}**; and a read-only *{{narr.sample}}* showing how the first changed paragraph looks after preparation.

## Starting, pausing, stopping

| Control | What it does |
|---|---|
| **{{narr.start}}** | Starts the narration. The first start loads the voice model (*{{narr.loading_model}}*). |
| **{{narr.pause}}** / **{{narr.resume}}** | Pause stops after the current fragment; the same button then reads Resume. |
| **{{ui.cancel}}** | Stops the job. Finished fragments stay saved. |
| Progress bar and status line | e.g. "Fragment 3 of 8 · about 7 min left" – how many pieces (chunks) of the book are done and the estimated time left. |
| **{{ui.open_folder}}** | Appears at the end (*{{ui.ready}}*): opens the folder with the finished audiobook. |

**Resumable.** Every finished fragment is stored on disk. If you pause, cancel, the program crashes or you close it, press **{{narr.start}}** again later: *{{narr.resuming_plain}}* The stored fragments are deleted after a successful run.

**Listen while it is being made.** As soon as the first fragments are ready the **mini player** appears (*{{player.title_live}}*): a Play/Pause button, a slider and the time. *{{player.live_status_plain}}* It follows new fragments and waits if it catches up with the narration. When the job ends, the whole result becomes its playlist.

After a successful run you see *{{narr.done}}* and, depending on the voice licence, a reminder, for example *{{narr.done_private_reminder}}*

## Speed

Narration uses **batched generation**: several chunks go through the voice model in one call instead of one at a time. The batch size is chosen from the free video memory (up to 12) and is halved automatically if memory runs out; if a batch fails for any other reason, Voxprint falls back to one chunk at a time, so batching can never make a book fail. Finished chunks are saved by a helper thread while the graphics card already makes the next batch.

Measured on an RTX 4090 (12 English chunks, 80 s of audio, open voice): the old mode needed a real-time factor (RTF) of **3.05** (243 s of work for 80 s of sound); with batch 12 it is **0.31** (25 s) – about **10 times faster**. RTF is "seconds of work per second of audio": 0.31 means one hour of audiobook takes about 19 minutes. Planning figures at RTF 0.31–0.5: 5 hours of audio ≈ 1.6–2.5 hours of work; 20 hours ≈ 6–10 hours. On cards with 12–16 GB the batch is smaller, so the speed is lower.

# Train your voice

## Recording the voice

The quality of the voice depends mostly on the recording. Plan **5 to 15 minutes** of clean speech (under 2 minutes triggers a warning; under 10 seconds is refused). The program contains no recorder – use any recorder or phone app and save WAV, FLAC, MP3, M4A or OGG.

**Before you record:** a quiet room, the microphone about a hand's width from the mouth, no sound "enhancements" (noise suppression, equaliser, reverb), even volume.

**The recording script.** The repository contains ready-made scripts in `docs/recording-scripts/`, written separately in Russian, English and German (TXT and PDF, dark and print). Each has 17 blocks: greeting, the story of the idea, numbers and dates, smooth sentences for the sounds of the language, neutral reading, calm, joy, sadness, irritation, surprise, whisper and loud speech, three short computer stories, an optional "bad day" story with mild colour words (no profanity), optional technical terms, and – at the very end – the consent statement in the language of the file. The PDFs add a reading guide (tempo, pauses, emotions, breaks). The script itself states three simple rules:

1. **Read only ordinary lines.** Everything in square brackets and every line with `===` is a hint – do not read it aloud.
2. **One line – one sentence.** Read it, breathe, stay silent for a second, then read the next line.
3. **Stumbled? Stay silent for a second and read the same sentence again from the beginning.** Do not stop the recording.

Further tips from the script: no need to act the emotions – a slightly different voice is enough, and if it does not work, read in your normal voice. Blocks marked OPTIONAL can be skipped. You choose this same script file as the "text" in the Train window.

> Voxprint is built to cope with re-read sentences: when a recording is matched with the script, the best reading of every line is kept. Still, the cleaner the recording, the better the voice.

**The consent statement (block 18).** Finish the recording with **one** of the consent sentences – the one that is true for you: *commercial* (A), *public non-commercial* (B) or *private only* (C), in Russian, English or German. Say your own name and today's date in place of the bracketed words. Read it only if the voice is yours or you have the owner's permission. Details in section 6.6.

## The Train window

Open it from the **{{studio.train_title}}** card. {{ui.subtitle}}

![The Train window, upper part: audio and text, training quality, the voice owner's consent and the main button.](@train_top)

| Control | What it does and when to use it |
|---|---|
| **{{ui.choose_audio}}** | Choose the voice recording (you can also drag the file into the window). Before a choice: *{{ui.audio_none}}* |
| **{{ui.choose_text}}** | Choose the text file you read (UTF-8 `.txt`; the recording script, for example). Before a choice: *{{ui.text_none}}* In no-transcript mode the same row becomes *{{asr.choose_script}}* (optional). |
| Check box *{{asr.checkbox}}* | Audio only, no text: see section 6.4. |
| **{{preset.label}}** | Drop-down with the training presets: see section 6.3. Below it the program shows a short description and an estimate of the training time for your graphics card. |
| ▶ **{{preset.advanced}}** | Shows the manual numbers (only editable in *{{preset.manual}}*). |
| **{{consent.title}}** | How the permission of the voice owner is recorded: see section 6.6. |
| **{{ui.btn_lora}}** | The main button: aligns the recording, cuts it, trains the voice and puts it in your library. |
| **{{preview.button}}** and the check box **{{preview.compare}}** | A quick listen-and-compare test before the long run: section 6.5. |
| ▸ **{{train.checks_options}}** – check box *{{check.checkbox}}* | After training, Voxprint re-reads a test sentence with the new voice and checks pitch and clarity automatically. Leave it on. |
| *{{ui.voice_gender_none}}* and *{{ui.voice_age_none}}*, ▸ **{{ui.voice_more}}** | Optional details of the voice. The gender (*{{ui.voice_type_male}}*, *{{ui.voice_type_female}}*, *{{ui.voice_type_child}}*, *{{ui.voice_type_other}}*) is saved in voice.json and added to the name of the trained voice's folder (`anna_male`, `anna_female`, `anna_unspecified`). The program suggests male or female from the pitch of your recording; change it if that is wrong. *{{ui.voice_more}}* holds further optional fields for voice.json. |
| Description field | *{{ui.voice_desc_placeholder}}* |
| **{{ui.btn_merge}}** | Merges the trained adapter into one standalone model that works in any app running Qwen3-TTS (section 6.7). |
| **{{ui.btn_dataset}}** | Only prepares the dataset (aligned and cut clips) without training. Useful if you want to train elsewhere. |
| Progress bar and stage chips | {{stage.model}}, {{stage.align}}, {{stage.slice}}, {{stage.train}}, {{stage.save}} – the stage that is running is highlighted. **{{ui.cancel}}** appears while a job runs. |
| Status line | Starts as *{{ui.status_idle}}* |
| Gear (top right) | Opens {{ui.settings_title}}. |

![The Train window, lower part: quick preview, name and description, gender and age, the universal model, the dataset button and the progress stages.](@train_bottom)

When the job ends, the status shows *{{ui.voice_registered}}* and a button **{{ui.open_folder}}** appears. The training folder is `Projects\Voices\<voice name>_Voxprint` in the projects folder: the dataset is in `dataset`, the adapter (tens of MB) in `output\<voice name>`.

The pictures show the default mode with a text: **{{ui.choose_audio}}** and **{{ui.choose_text}}** one under the other, each with the name of the chosen file. Ticking *{{asr.checkbox}}* switches to the audio-only mode (section 6.4): the text row becomes an optional script and a warning block appears.

## Training presets

**{{preset.label}}** offers four presets. For each one the program shows what it means and an **estimated time** – for example "about 2 min on NVIDIA GeForce RTX 4090 (5 passes, adapter size 32, about 71 clips)". Read the estimate as "about".

| Preset | Meaning |
|---|---|
| **{{preset.fast}}** | {{preset.desc_fast}} |
| **{{preset.balanced}}** | {{preset.desc_balanced}} |
| **{{preset.maximum}}** | {{preset.desc_maximum}} |
| **{{preset.manual}}** | {{preset.desc_manual}} |

Under **{{preset.advanced}}** you see the numbers behind a preset:

| Field | Plain meaning |
|---|---|
| {{preset.adv_epochs}} | How many times the training goes over your whole recording. More passes: closer to the voice, but too many can make it "babble". |
| {{preset.adv_rank}} | How big the voice add-on is. Larger – more detail, more memory. |
| {{preset.adv_alpha}} | How strongly the add-on influences the base model. |
| {{preset.adv_lr}} | How big each learning step is. The most sensitive number: do not raise it unless you know why. |
| {{preset.adv_accum}} | How many small steps are summed before the model is updated (saves memory). |

*{{preset.adv_hint}}* No preset ever raises the learning rate. In our tests a lower "training number" (loss) did not mean a better voice – always listen to the result.

## No transcript? (audio only)

Tick *{{asr.checkbox}}* and the program recognises the speech itself (Qwen3-ASR, offline; the recognition model, about 1.9 GB, is part of the first-start download). The text row stays as an **optional script** (*{{asr.choose_script}}*, or drop a `.txt` into the window): if you choose the text that was read aloud, every recognised piece is matched to it tolerantly (stumbles and re-read lines are handled), so the final text is the correct one, not the recognised one. In addition you have:

* **{{ui.choose_audio}}** – now you can pick **many files** at once; the line shows "N audio file(s) chosen";
* **{{asr.choose_folder}}** – a whole folder of clips, searched with its subfolders;
* drag and drop of files into the window.

Each file is recognised separately; long recordings are cut at pauses into pieces of up to 14 s; everything is merged into one dataset and made equally loud. Skipped automatically: clips under 1.5 s or over 20 s, empty or implausible recognition, duplicates and clips with poor sound. At the end the window reports, for example, "Recognised N file(s): kept X of Y clips = … min of … min of audio".

> *{{asr.warning}}*

Because of this, you must tick *{{asr.confirm}}* before the buttons work (the tick resets every time you switch the mode on).

## Quick preview and comparing two variants

Before a long run you can test settings cheaply. **{{preview.button}}** trains a very short adapter on a few clips (at most 12 clips and 4 passes, about 3 minutes at most) and then makes a sample of about 10 seconds that you can listen to in the player. The estimate is shown next to the button: *{{preview.estimate_plain}}*

![Quick preview of two variants, with the player and the automatic metrics.](@train_preview)

* With **{{preview.compare}}** ticked, two samples are made: A with the current settings and B with a larger adapter and 1.5 times the passes.
* Every variant gets a line of automatic metrics: the length of the sample, "recognition errors" (how many words an automatic listener got wrong when re-reading the sample – lower is better) and "pitch vs. your recording" in semitones (how far the voice is higher or lower than yours – close to 0 is best). A verdict follows: *{{check.verdict_good}}*, or a warning.
* **{{preview.play}}** plays a variant. **{{preview.use}}** copies its numbers into the *{{preset.manual}}* preset; then press **{{ui.btn_lora}}** for the full run.
* A preview voice is temporary and is **not** added to **{{studio.voices_title}}**. The metrics are a hint, not a verdict: always listen.

## The consent card

Every voice carries a record of what its owner allows. Under **{{consent.title}}** choose how it is set:

| Option | When to use it |
|---|---|
| {{consent.mode_auto}} | You finished the recording with one of the three consent sentences (block 18 of the script). After training the program recognises the statement and shows the result. |
| {{consent.mode_manual}} | You type the speaker's name (placeholder "{{consent.name_placeholder}}") and choose the scope yourself. Use it only for your own voice or with the owner's permission. |
| {{consent.mode_none}} | No statement. The voice is marked *private only*. |

The check box **{{consent.save_clip}}** stores the spoken statement (the last 45 seconds of the recording) next to the voice. 

After training in the automatic mode the **consent card** appears:

![The consent card after training: detected name, date and scope, and the Confirm button.](@train_consent)

It shows the *detected* name, date and scope, and the sentence that was recognised. Check it, change the scope in the drop-down if needed and press **{{consent.confirm}}**. If the statement is unclear or missing, the strictest level – *{{consent.badge_private_only}}* – is used. This is a record of what was said, **not legal advice**.

| Scope in the drop-down | Badge | Meaning | Licence stored |
|---|---|---|---|
| {{consent.scope_commercial}} | {{consent.badge_commercial}} | {{consent.tip_commercial}} | CC-BY-4.0 |
| {{consent.scope_public_noncommercial}} | {{consent.badge_public_noncommercial}} | {{consent.tip_public_noncommercial}} | CC-BY-NC-4.0 |
| {{consent.scope_private_only}} | {{consent.badge_private_only}} | {{consent.tip_private_only}} | custom/personal-only |

## The universal model (optional)

The small adapter (tens of MB) works only in a few apps (for example Alexandria). **{{ui.btn_merge}}** merges it into a standalone model folder (`merged_model`) that works in **any** app that runs Qwen3-TTS. Voxprint checks the free disk space and asks first (*{{ui.merge_confirm_title}}*); it is never built automatically. The button is disabled until a voice has been created. Details are in `USAGE.txt` inside the folder.

# My voices

Open this from the **{{studio.voices_title}}** card. {{voices.intro}}

![My voices (build 667): Tirzah is shown as a bundled voice (CC0-1.0, read-only); she was bundled up to build 702, Levi and Miriam are bundled from build 703. The picture also shows Boaz, which is retired, and the Open universal voice, which is not downloaded yet.](@voices)

Each voice is a card:

| Element | Meaning |
|---|---|
| Name and two badges | The **licence badge** (green "… · commercial use OK" or amber "Personal use only · …") and the **scope badge** ({{consent.badge_commercial}}, {{consent.badge_public_noncommercial}}, {{consent.badge_private_only}}). |
| Information line | Language · minutes of speech used for training · epochs (passes) · voice type · author · (for online voices) size. |
| Description | What the author wrote. A "test use only" voice also shows a warning. |
| **{{voices.preview}}** | Plays a short sample of the voice. **{{voices.stop}}** stops it. If the voice has no sample: *{{voices.no_preview}}* |
| **{{voices.narrate}}** | Opens the Narrate window with this voice already chosen. |
| **{{voices.edit}}** | Opens *{{voices.edit_title}}*: {{voices.field_name}}, {{voices.field_author}}, {{voices.field_license}}, {{voices.field_gender_age}}, {{voices.field_description}}. Press **{{voices.save}}**. |
| **{{voices.export}}** | Saves the voice for another computer: *{{voices.export_small_plain}}* (the trained voice; another Voxprint imports it in a second) or *{{voices.export_full_plain}}* (the universal model). |
| **{{voices.open_folder}}** | Opens the voice's folder. |
| **{{voices.delete}}** | Deletes the voice from this computer after a confirmation: *{{voices.delete_confirm_plain}}* Bundled voices are read-only: **{{voices.edit}}** and **{{voices.delete}}** are greyed out (*{{voices.bundled_tip}}*). |

The footer of the window repeats: *{{voices.rights_note}}*

When the library is empty the window says *{{voices.empty}}* and offers **{{voices.empty_train}}**.

## Online voices and the voice repository

The third card in the picture above, the *Open universal voice*, is such an online voice: *{{voices.remote_state}}* and a **{{voices.remote_download}}** button.

* Voices from the online index appear by themselves as cards with the state *{{voices.remote_state}}*, their licence and size and a **{{voices.remote_download}}** button. The download is checked by a SHA-256 checksum and can be resumed.
* **{{voices.remote_refresh}}** reloads the list. Offline, Voxprint shows the last known list (*{{voices.repo_offline}}*).
* **{{voices.repo_button}}** opens the *{{voices.repo_title}}* dialog with **{{voices.repo_refresh}}**, a field for the address of the index (`index.json`) and **{{voices.repo_download}}**.

## Importing a voice

**{{voices.import}}** offers two ways: **{{voices.import_folder}}** (an adapter folder) and **{{voices.import_zip}}** (a voice package). Archives are checked for unsafe paths and size limits. A voice that arrives without a declared licence is treated as *personal use only*.

## Licences and the voice that Voxprint offers

| Licence | Commercial use |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | allowed (keep attribution / share-alike if the licence asks for it) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | not allowed |
| custom/personal-only (default) | not allowed; results stay on your computer |
| custom/test-use-only | not allowed; for testing only, no public release |

* **Levi** (male) and **Miriam** (female) – the open **Russian** voices that come with Voxprint from build 703 (part of the standard model download). Levi is trained only from the public-domain LibriVox recording of Anton Chekhov's short stories read by Виталий, Miriam only from *Портреты русских поэтов* by Ilya Ehrenburg read by Maya S. Levi is the default narrator; for multi-voice narration Voxprint picks Natan and Shimon for men and Miriam for women when they are installed (Natan, Shimon and Rivka are one click away in the catalog). Licence **CC0-1.0**. Up to build 702 the bundled voices were **Tirzah** (female) and **Gideon** (male); an install that has them keeps them. Tirzah is trained only from the public-domain LibriVox recording *Степные сказки (Stepnyia skazki)* by Grigory Danilevsky, read by Anastasiia Solokha. Gideon is trained only from *Вехи (Vekhi)*, a 1909 essay collection, read by Kazbek, and replaces *Boaz*. Licence **CC0-1.0** – free for any use, also commercial; attribution is appreciated. The names are Voxprint's own and do not imply the readers' endorsement. In the library they are read-only (*{{voices.bundled_note_plain}}*). A Tirzah sample – Genesis 1:1–2:3 (Russian Synodal translation, public domain), also CC0 – is in the repository folder `samples/`. *Boaz* shipped with 0.1.3 and is retired; the old Boaz sample is not a Gideon recording.
* **Asher** (male) and **Noa** (female) – optional Russian catalog voices, not part of the standard download. They are on the `voices-v1` release and appear in the in-app catalog. Asher is trained from *Teachings of Christ* by Leo Tolstoy, read by Vladimir Anyanov; Noa from *Izbrannye* by Sholem Aleichem, read by Hanna Ponomarenko. Licence **CC0-1.0**.
* **Open universal voice** (German UI: *Offene Universalstimme*, Russian UI: *Открытый универсальный голос*) – an English voice for people who have no recording of their own. Trained only from the public-domain LJ Speech dataset (reader Linda Johnson, LibriVox); licence **CC0-1.0** – free for any use, also commercial; attribution to the LJ Speech dataset (Keith Ito) is appreciated.

The Open universal voice is downloaded separately from the voice repository.

> The pictures in this manual show the bundled voices. Voices you train yourself are *personal use only* by default (scope *{{consent.badge_private_only}}*): no public use of the output, no commercial projects.

The badge is information, not legal advice. The licence covers the voice model only.

# Settings

Open **{{ui.settings_title}}** with the gear button (top right of the Studio and the other windows).

![Settings (build 667; the projects path and the network address are blurred).](@settings)

| Control | What it does |
|---|---|
| **{{ui.language}}** | The interface language: English, Deutsch, Русский, Українська, Latviešu. It changes at once in all windows. |
| **{{ui.transparency}}** | How translucent the windows are: *{{ui.transparency_default}}*, *{{ui.transparency_more}}* or *{{ui.transparency_off}}*. |
| **{{ui.net_iface}}** | Which network adapter the downloads use. *{{ui.net_iface_auto}}*: if a download cannot connect (VPN, unusual adapter), the other interfaces are tried and the working one is remembered. |
| **{{asrmodel.label}}** | Which speech-recognition model is used. *{{asrmodel.auto}}* takes the 1.7B model on cards with about 8 GB of video memory or more, otherwise 0.6B. The line below the check box shows the model in use. |
| **{{preload.option}}** | After the start, loads the installed speech models into RAM in the background, so that narration starts at once. The available RAM is shown next to it. |
| **{{narrset.title}}** | {{narrset.comma}}, {{narrset.mid}}, {{narrset.sentence}}, {{narrset.paragraph}}, {{narrset.chapter}} – the five pause lengths in seconds (defaults 0.25 / 0.40 / 0.60 / 1.00 / 2.00 s); **{{narrset.speed}}** (70–130 %; 100 % is the voice's own speed); **{{narrset.style}}**. See section 5.4. |
| **{{narrset.defaults}}** | Restores the default pauses and speed. |
| **{{ui.btn_update}}** | Checks whether newer verified versions of components or models exist and installs them (with a safety check and automatic rollback). Voxprint also checks quietly once a week. Messages: *{{upd.up_to_date}}*, *{{upd.restart}}* |
| **{{ui.settings_models_folder}}** | Opens the folder with the downloaded models. |
| **{{ui.settings_data_folder}}** | Opens the data and log folder. |
| **{{diag.button}}** | Saves the log files, information about this computer (system, GPU, memory) and the settings as one zip file; the names of your files are cut out. Attach it when you report an error. |
| **{{modules.title}}** | Opens the Components window: what is installed, updates, and the complete download. |
| **{{projects.title}}** | Where new projects (audiobooks, voice trainings, Re-voice) are kept. **{{projects.open}}**, **{{projects.change}}** (prefer a local drive with plenty of space that is not synced to the cloud), **{{projects.default}}**. Existing projects stay where they are. |
| **{{backup.title}}** – **{{backup.include_models}}**, **{{backup.include_voices}}** | What goes into a backup (both on by default). |
| **{{backup.link_models}}** | When restoring, read the models from the backup drive instead of copying them. The drive must stay connected. |
| **{{backup.btn_backup}}** | Copies models (and voices) to a folder or drive you choose. Shows the size and free space first; resumable; identical files are skipped; every copy is checked. A progress bar and a **{{ui.cancel}}** button appear. |
| **{{backup.btn_restore}}**, **{{backup.btn_restore_folder}}** | Restores from a `Voxprint-backup` folder and checks every file by its checksum. Voices that already exist with different content are never overwritten; damaged or missing files are downloaded as usual. |
| **{{existing.title}}** | {{existing.hint}} **{{existing.choose}}** picks the folder, **{{existing.clear}}** forgets it. |
| **{{autorepair.title}}** – **{{autorepair.button}}** | {{autorepair.desc}} See section 8.2. |
| **{{ui.about}}** | Version, the idea, how it works, open-source components with their licences, a link to the GitHub repository and third-party notices. |

The bottom line shows the version, for example *Voxprint AI Audiobook Builder 0.1.3-beta · build 667 "Menuchah"*.

If the program notices a damaged installation, a banner offers the repair (*{{ui.repair_offer_plain}}*). If **{{autorepair.button}}** does not help, run the Voxprint installer again – voices and models are kept.

When the program finds an older component in an environment it does not own (for example your own Python), it asks first: *{{upg.title}}* with **{{upg.btn_upgrade}}** or **{{upg.btn_later}}**; it never changes your environment silently.

## Maximum quality out of the box

Out of the box, everything that can run automatically is **already selected** and marked with a star and the word *{{auto.recommended}}*: if you touch nothing, Voxprint works at maximum quality. The models these options need are part of the first-start download, so there is nothing extra to click. The single check boxes stay in their windows.

* Text preparation (Narrate a book): one switch, *{{prep.one}}*, on by default (layout, footnotes, quotes, links, headings, numbers, abbreviations, the Russian letter yo where a dictionary is sure, and Russian typos when that model is downloaded).
* Translation of the book before narrating (English, Russian, German; Opus-MT models, downloaded once). Off until you turn it on.
* Training window: *{{check.checkbox}}* and *{{preview.compare}}* (both need the speech-recognition model, about 1.9 GB, downloaded once).
* Always on, without a switch: alignment of the text onto the audio and its plausibility check, the audio quality filter, the recognition filters of the audio-only mode.
* Not included, because they do not exist yet: a punctuation model. Russian stress marks are not written, because the base speech model does not read them; the letter yo is part of Prepare text. Speaker marks are a separate option on the AI text model card.

The training preset stays at *{{preset.balanced}}* on purpose: in our tests longer training with a higher learning rate made voices babble, so more is not better there. Instead, after **{{preview.compare}}** the better of the two variants is marked **{{auto.recommended}}** (verdict first, then fewer recognition errors, then the smaller pitch shift; with no clear difference variant A, which is cheaper). The choice is still yours: always listen.

## Check and repair

**{{autorepair.button}}** checks the program, its components and every model file by file against its SHA-256 checksum and downloads only what is missing or damaged. Your voices and books are not touched. While it runs, a progress bar and a step line (for example *Step 8 of 21: Checking Qwen3-ASR-0.6B: model.safetensors*) show where it is, and the button becomes **{{autorepair.stop}}**; a stopped check keeps what it has finished.

![Check and repair while it runs: the progress bar and the current step.](@repair_running)

At the end a summary appears, for example *Check finished: 24 checked, 0 repaired or downloaded, 0 failed.* Whatever could not be repaired is named below it.

![Check and repair finished: the summary line.](@repair_done)

> Known issue in build 667: while the check runs, the right column of Settings is squeezed and the explanatory texts overlap (first picture). This is cosmetic and will be fixed in a coming build.

The same check runs without the window as `Voxprint.exe --auto-repair` (chapter 11).

# Glossary

<dl class="glossary">

<dt>Adapter (LoRA)</dt><dd>A small add-on file (tens of MB) that "teaches" the big speech model one particular voice. Technically LoRA – low-rank adaptation. The button {{ui.btn_lora}} creates it.</dd>

<dt>Universal model</dt><dd>The adapter merged into a full ~4 GB model folder. It works in any app that runs Qwen3-TTS.</dd>

<dt>Qwen3-TTS</dt><dd>The open speech-synthesis model that Voxprint uses to speak. The base model is Apache-2.0 licensed.</dd>

<dt>Dataset</dt><dd>The training material Voxprint cuts from your recording: short clips (3–12 s) with their texts, a reference clip and a report.</dd>

<dt>Alignment</dt><dd>Finding out at what second each word of the text is spoken in the recording. It allows clean cutting at pauses, never in the middle of a word.</dd>

<dt>Clip / fragment</dt><dd>A short piece of the recording in the dataset. In the narration window "Fragment 3 of 8" means a piece of the book.</dd>

<dt>Chunk</dt><dd>A piece of the book text of about a sentence or a clause that is sent to the voice model in one go. The UI calls it a "fragment". Finished chunks are saved, which makes narration resumable.</dd>

<dt>Batch</dt><dd>Several chunks that the voice model makes together in one call. Bigger batches are much faster but use more video memory (up to 12 at a time).</dd>

<dt>RTF (real-time factor)</dt><dd>Seconds of computing per second of produced audio. 0.31 means one hour of audiobook takes about 19 minutes. Lower is faster.</dd>

<dt>Epoch / pass</dt><dd>One complete pass of the training over your whole recording (UI: "{{preset.adv_epochs}}"). The voice list shows how many epochs a voice was trained for.</dd>

<dt>Preset</dt><dd>A ready-made set of training numbers: {{preset.fast}}, {{preset.balanced}}, {{preset.maximum}}, {{preset.manual}}.</dd>

<dt>Rank, alpha, learning rate, gradient accumulation</dt><dd>Technical training numbers shown under {{preset.advanced}}; see section 6.3.</dd>

<dt>Preview</dt><dd>In {{studio.voices_title}}: a short sample of a voice ({{voices.preview}}). In the Train window: the <i>quick preview</i> – a short test training with a ~10 s sample and automatic checks; nothing is added to your voices.</dd>

<dt>Recognition errors (WER)</dt><dd>The share of words that an automatic listener (Qwen3-ASR) gets wrong when it transcribes the generated sample. Lower is better.</dd>

<dt>Pitch, semitone</dt><dd>How high the voice is. A semitone is the smallest step on a piano. "Pitch vs. your recording: +1.2 semitones" means the voice is slightly higher than the original.</dd>

<dt>Consent</dt><dd>The voice owner's permission, spoken at the end of the recording or entered by hand, saved with the voice. A record, not legal advice.</dd>

<dt>Scope</dt><dd>What the consent allows: {{consent.badge_commercial}}, {{consent.badge_public_noncommercial}} or {{consent.badge_private_only}}.</dd>

<dt>Licence badge / scope badge</dt><dd>Coloured labels on a voice card and in the Narrate window. Green: commercial use allowed. Amber: restricted. The badge is information, not legal advice.</dd>

<dt>Voice type</dt><dd>An optional label of the voice: male, female, child or other. Saved with the voice.</dd>

<dt>Mini player</dt><dd>The small player in the Narrate window that plays the finished part of the book while the rest is still being made.</dd>

<dt>Bitrate (kbit/s)</dt><dd>How much data one second of sound takes. Higher means better quality and bigger files. Speech needs little: Opus 32 kbit/s is plenty.</dd>

<dt>Opus, MP3, AAC / M4B, FLAC, WAV</dt><dd>Audio formats. Opus: small, open, free. MP3: plays everywhere. AAC in an M4B container: for Apple Books (patent-encumbered). FLAC and WAV: lossless, large files for archiving and editing.</dd>

<dt>Chapter marks</dt><dd>Bookmarks inside a single audio file that let a player jump from chapter to chapter.</dd>

<dt>Video memory (VRAM)</dt><dd>The memory of the graphics card. It limits the size of the model and of the batch.</dd>

<dt>Resume</dt><dd>Continuing a stopped narration or download from where it stopped.</dd>

<dt>Checksum (SHA-256)</dt><dd>A fingerprint of a file that proves it was downloaded or copied without damage.</dd>

<dt>Mirror (ModelScope)</dt><dd>An alternative download server that Voxprint uses if Hugging Face is slow or unreachable.</dd>

<dt>Check and repair</dt><dd>Checks every component and model file by its checksum and downloads only what is missing or damaged ({{ui.settings_title}} → {{autorepair.button}}, or <code>--auto-repair</code>).</dd>

<dt>Measured pauses</dt><dd>Silence of a fixed length that Voxprint inserts between the spoken pieces – after a comma, a sentence, a paragraph, a chapter – independent of how the voice model phrases the text.</dd>

<dt>Bundled voices</dt><dd>Levi and Miriam (from build 703): the open Russian voices (CC0-1.0) that come with Voxprint and are read-only in the library. Tirzah and Gideon were bundled up to build 702, Boaz with 0.1.3; they are retired.</dd>

</dl>

# Troubleshooting and FAQ

**Which messages can I get, and what do I do?**

| Message | What to do |
|---|---|
| {{err.audio_short}} | Record longer – 5–15 minutes. |
| {{err.audio_unreadable}} | Convert the recording to WAV or MP3. |
| {{err.mismatch_length}} | Check that you chose the right audio and text and read the whole text. |
| {{err.mismatch_extra}} | Trim the extra speech from the start or end of the recording, or choose the correct text. |
| {{err.all_dropped}} | Check the microphone and level; record again in a quieter place. |
| {{err.oom}} | Close other programs that use the graphics card; Voxprint retries with a lighter load (*{{progress.oom_retry}}*). As a last resort **{{ui.retry_cpu}}**. |
| {{err.download_failed}} | Check your internet connection and start again; the download continues where it stopped. |
| {{err.narration_chunk}} | Press **{{narr.start}}** again – finished fragments are kept. |
| {{err.narration_no_ffmpeg}} | Settings → **{{autorepair.button}}**, or choose WAV. |
| {{err.book_unsupported}} | Convert the book to TXT, FB2 or EPUB. |
| {{err.book_unsafe}} | The file looks damaged or too large; try another copy. |
| {{warn.no_gpu}} | Training works without an NVIDIA card but is extremely slow. |
| {{warn.loss_low}} | The voice may babble. Use **{{preset.balanced}}**, record more text, and listen with the quick preview. |

**The voice sounds higher than mine.** A small upward pitch drift (about +2…+3 semitones was seen on a male voice) is a known effect of this kind of training. The automatic check warns above +4. Try the quick preview with both variants and take the one that is closer.

**The voice babbles or never stops.** Too many passes or too high a learning rate. Use *{{preset.balanced}}*, never raise the learning rate by hand.

**Can I stop and continue later?** Narration: yes, always (section 5.5). Downloads and backups: yes. Training: cancel is possible, but a full run starts over; use the quick preview first.

**Can I use my voice for a published audiobook?** Only if the consent scope allows it ({{consent.badge_commercial}} to sell; {{consent.badge_public_noncommercial}} to publish for free). Voices you train without a consent statement are *private only*. You can change the licence under **{{voices.edit}}** if you own the voice.

**Does Voxprint send my recordings anywhere?** No. It sends nothing to the internet except downloading models, checking for updates and, when you open the repository, fetching the voice index and the voices you choose. No telemetry, no account.

**Which languages are supported?** The interface: English, Deutsch, Русский, Українська, Latviešu. Text preparation covers Russian and English fully. The speech model supports further languages, but they are not tested.

**Where is the log?** {{ui.settings_data_folder}} in Settings. Send the file from there if an error repeats.

**Is the AAC / M4B option safe to use?** It is optional and off by default for good reason: AAC is patent-encumbered and you are responsible for compliance (section 5.4).

# Command line

Everything in this chapter is optional: the windows do all of it. The command line is meant for scripts, servers and AI agents and uses the same pipeline as the windows. The full reference is `docs/CLI.md`; the file to hand to an agent is `docs/AGENTS.md` (both in the repository).

## Where the program is and the rules

The installed program is `C:\Program Files\Voxprint\Voxprint.exe` (another folder if you chose one; `where.exe Voxprint` finds it). In a source checkout use `python main.py`. Below, `voxprint` stands for either.

* Nothing prompts; `--yes` is accepted by every command.
* `--json` prints progress as JSON lines and ends with one result object (`type: result`, `ok`, `exit_code`, `outputs`, `warnings`, `error`, `hint`).
* `voxprint --help` and `voxprint <command> --help` show copy-paste examples; `voxprint --version` prints, for example, `Voxprint 0.1.3-beta build 667 "Menuchah"`.
* Set `VOXPRINT_LANG=en` if you parse the messages of the pipeline.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | Unexpected failure – save `voxprint diag` |
| 2 | Missing or unknown arguments |
| 3 | Book, audio, transcript, voice or backup missing or unreadable |
| 4 | A model or component is not installed, or its download failed |
| 5 | Out of GPU memory – for training, retry with `--force-cpu` |
| 6 | Cancelled – run the same command again to resume |

## Narrate a book

```
voxprint narrate BOOK --voice ID_OR_NAME --out DIR [--format mp3,m4b,opus,...]
    [--pause-comma S] [--pause-mid S] [--pause-sentence S] [--pause-paragraph S]
    [--pause-chapter S] [--speed X] [--style auto|scripture|fiction|dialogue]
    [--pauses] [--ai-disclosure] [--json]
```

The book is a TXT, FB2, `.fb2.zip` or EPUB file; the result lands in `<DIR>/<book title>/`; the default format is one Opus file. The pause flags (seconds), `--speed` (0.7–1.3) and `--style` override the values from {{ui.settings_title}} → {{narrset.title}} for one run; `--pauses` is the opt-in cut at every comma. Running the same command again skips the fragments that are already cached, and changing pauses or speed never synthesizes them again.

```
voxprint narrate genesis.txt --voice levi --out ./audiobooks --format mp3 --json
```

## Train a voice

```
voxprint train AUDIO [--text SCRIPT] [--name NAME] [--type male|female|child|other]
    [--out DIR] [--consent none|auto|commercial|public_noncommercial|private_only]
    [--speaker NAME] [--license ID] [--language CODE] [--force-cpu] [--json]
```

With `--text` the recording is aligned to the script; without it (one file or a folder of clips) speech recognition builds the dataset. `--out` is the parent folder; every voice gets its own `<voice name>_Voxprint`. Without `--consent` the voice is stored as *private only*; `--consent auto` reads the spoken statement, the other values record a scope you confirmed yourself, and `--license` can never allow more than the consent. Only train a voice you have the right to use.

```
voxprint train recording.wav --text script.txt --name Anna --type female --consent auto
```

## Back up and restore

```
voxprint backup --out DIR [--no-models] [--no-voices] [--json]
voxprint restore --from DIR [--link] [--json]
```

`backup` copies the models and the voice library into `<DIR>/Voxprint-backup/` with the manifest `voxprint-backup.json` (version, build, size and SHA-256 of every file). Files already there with the same size and hash are skipped, so an interrupted copy continues; nothing is written when the drive is too small. `restore` copies the backup into the normal folders and checks every file; a damaged or missing file is named and later downloaded as usual. `--link` keeps the models on the backup drive (it must stay connected) and copies only the voices. Errors exit with code 3. These are the same jobs as the buttons in {{ui.settings_title}} → {{backup.title}}.

## Check the installation

```
voxprint status --json                 # version, GPU, voices, formats, installed models
voxprint models download required      # TTS, aligner and speech recognition, if missing
Voxprint.exe --verify-install          # quick install check with reason codes
Voxprint.exe --auto-repair             # Check & repair: every file by SHA-256
voxprint diag --out report.zip         # diagnostic report
```

`status` (also `capabilities`) prints one JSON object and downloads nothing – run it before `train` or `narrate`. `--auto-repair` is the same job as {{ui.settings_title}} → **{{autorepair.button}}** (section 8.2): it prints one line per checked item, downloads only the damaged or missing files and exits with 0 when everything is in order; the report is also written to `logs\auto_repair.txt`. `--verify-install` writes `logs\verify_install.txt`. `diag` is the same as **{{diag.button}}**.

# Licences and responsibility

* **Voxprint's own source code is licensed under the Apache License 2.0** (Copyright 2026 Aleksandr Mitroshenkov; see the `LICENSE` and `NOTICE` files).
* **The Apache licence of the code does not cover models and voices.** The models Voxprint downloads (Qwen3-TTS, Qwen3-ASR and the optional text models) keep their authors' licences. Every voice has its own licence, shown as a badge and stored in `voice.json`.
* Third-party components, with their licences, are listed under **{{ui.about}}** and in `THIRD_PARTY_NOTICES.md` and the `licenses` folder of the installation. Qt/PySide6 is used under the LGPL-3.0; the bundled ffmpeg is a separate program with its own licence; the AAC codec is patent-encumbered.
* **Use voices responsibly.** A voice is personal data and a personality right. Use a voice only with the explicit permission of its owner and within its licence; no impersonation, fraud or misleading content; label synthetic speech where this is required. You are responsible for the lawful use of the voices and audio you create.

> **This manual and the licence badges are information, not legal advice.** The consent card records what a speaker said; it does not replace the law of your country.

*Voxprint AI Audiobook Builder 0.1.3 (beta), build 667 “Menuchah”. Manual of 9 October 2026.*

*Voxprint AI Audiobook Builder is an independent project and is not affiliated with other products or services with similar names.*
