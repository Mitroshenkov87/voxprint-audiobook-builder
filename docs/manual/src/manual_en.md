# What Voxprint does

**Voxprint AI Audiobook Builder** ({{studio.tagline}}) is a Windows 11 application that does two things, entirely on your own computer:

1. **It teaches a computer voice to sound like a particular person.** You give it a recording of a voice (yours, or of a person who has given you permission) – ideally 5 to 15 minutes – and, if you have it, the text that was read. Voxprint aligns text and sound, cuts the recording into clean pieces and trains a small voice add-on (an *adapter*, technically LoRA) for the speech model **Qwen3-TTS**.
2. **It reads whole books aloud with that voice.** You pick a book (TXT, FB2 or EPUB) and a voice from your library, and get a finished audiobook with chapters – one Opus file, one MP3 per chapter, and so on.

There is no command line, no browser and no account. Recordings, texts, voices and results stay on the computer. The program only goes online to download models, to check for updates and, if you ask for it, to download voices from the voice repository.

![The Studio – the first screen of Voxprint (dark theme, English interface).](@studio_home)

> **Beta software.** Version 0.1.0 has been tested end to end on one machine (Windows with an NVIDIA RTX 4090) and one main speaker. Expect rough edges; keep a copy of your recordings and voices (see {{backup.title}} in chapter 8).

## What you need

| | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 11 24H2 (build 26100), 64-bit | Windows 11 26H2 |
| Graphics card | NVIDIA with about 6 GB video memory (VRAM) | NVIDIA with 16 GB VRAM or more |
| Without an NVIDIA card | The dataset can still be created; training runs on the processor and takes many hours (the program warns you) | – |
| Disk space | about 12 GB (models about 7 GB + program) plus about 4.2 GB if you build the optional universal model | an SSD |
| Internet | once, to download the models (about 7 GB) | – |

## How it works in five steps

1. Read a known text aloud and record it (5–15 minutes).
2. Choose the recording and the text file – or only the recording, if you have no text.
3. Voxprint aligns the text to the audio with a neural forced aligner and cuts it into clean fragments – the *dataset*.
4. It trains a small voice adapter on your graphics card.
5. The voice appears in **{{studio.voices_title}}**. Now **{{studio.narrate_title}}** with it.

> **Use voices responsibly.** A voice is personal data and a personality right of its owner. Use a voice only with the explicit permission of the person it belongs to and within the licence of that voice. See chapter 11.

# Installing Voxprint

## The installer

Run **Voxprint-Setup.exe** (Inno Setup, per-machine installation, about 1.8 GB because the PyTorch libraries are inside). The wizard is available in English, Russian and German. After the install-folder page there is an optional page **Existing models (optional)**: if you already have downloaded models from an earlier installation or from a backup, choose that folder; otherwise leave the field empty. The installer copies nothing – it only remembers the folder, and the first start of the program imports the models from there instead of downloading them.

Silent installation for administrators: `Voxprint-Setup.exe /VERYSILENT /ModelsDir="D:\old\models"`.

> If the administrator account that runs the installer differs from the account that uses Voxprint, the data folder belongs to the administrator. In that case set the existing-models folder later in {{ui.settings_title}} (chapter 8).

## First start

On the first start Voxprint prepares itself: the status line says *{{ui.prefetch_start}}* It downloads the models it needs (about 7 GB, once). If Hugging Face is slow or unreachable, Voxprint switches to the ModelScope mirror automatically. The download can be interrupted and continues where it stopped. Models already present on the computer (for example from the Hugging Face cache or Alexandria/Pinokio) are reused without downloading them again, read-only.

When it is done, the status shows *{{ui.prefetch_done}}* A privacy notice is shown on the first start; a short reminder (*{{ui.footer_privacy}}*) stays at the bottom of the windows.

## Where things are stored

| What | Where |
|---|---|
| Program data, models, logs, voice library | `%LOCALAPPDATA%\Voxprint\` (`models\`, `logs\`, `voices\`, `state\`) |
| Audiobooks | `Documents\Voxprint\Audiobooks` (can be changed under *{{narr.advanced}}* in the Narrate window) |
| Training results | next to your recording: `<recording name>_Voxprint\dataset` and `output\<voice name>` |

You can open the model folder and the data/log folder with buttons in {{ui.settings_title}}.

# Quick start

**I just want to hear a book (no voice of my own yet).** Open **{{studio.voices_title}}**, press **{{voices.repo_button}}** (or simply wait: online voices are listed by themselves), and download the *Open universal voice* (English, licence CC0 – free for any use). Go back to the Studio, open **{{studio.narrate_title}}**, choose a TXT/FB2/EPUB file, pick the voice and press **{{narr.start}}**.

**I want my own voice.** Record yourself reading the recording script (section 6.1), open **{{studio.train_title}}**, choose the recording and the script text, leave the training quality at *{{preset.balanced}}* and press **{{ui.btn_lora}}**. After training, confirm the consent card. The voice is now in **{{studio.voices_title}}** and in **{{studio.narrate_title}}**.

# The Studio (home screen)

The Studio is the first window. It has no settings of its own – it is a menu of three big cards and a gear.

![The Studio.](@studio_home)

| Element | What it does and when to use it |
|---|---|
| Title and tagline | "Voxprint AI Audiobook Builder – {{studio.tagline}}". Only informational. |
| **{{studio.narrate_title}}** card | {{studio.narrate_desc}} Opens the narration window. This is the main card. If the library is empty, the card says *{{studio.narrate_no_voice}}* While a book is being narrated it shows *{{studio.narrate_running}}* |
| **{{studio.train_title}}** card | {{studio.train_desc}} Opens the training window. While training runs it shows *{{studio.train_running}}* |
| **{{studio.voices_title}}** card | {{studio.voices_desc}} At the bottom of the card a counter shows how many voices you have (for example "3 voice(s) in your library"). |
| Gear button (top right) | Opens **{{ui.settings_title}}** (chapter 8). |
| Note at the bottom | *{{ui.footer_privacy}}* |

Every other window has a **{{nav.back}}** button at the top left that brings you back here. Heavy work (training, narration) keeps running in the background if you go back.

# Narrate a book

Open this window from the **{{studio.narrate_title}}** card. {{narr.intro}}

![The Narrate window, upper half: book, voice and text preparation (demonstration voice).](@narrate_top)

## Step 1 – the book

| Control | What it does |
|---|---|
| **{{narr.choose_book}}** | Opens a file dialog. Supported: `.txt` (UTF-8 or cp1251; headings such as "Chapter 1" or "Глава 2" become chapters), `.fb2` and `.fb2.zip`, `.epub`. Title, author and cover are read when present. |
| File name and info line | After choosing, Voxprint shows the title and "3 chapters · about 7 min of audio" – the number of chapters and the expected length. Before a choice it says *{{narr.no_book}}* |

If the file cannot be opened you get a message such as *{{err.book_unsupported}}* or *{{err.book_empty}}* (see the troubleshooting chapter).

## Step 2 – the voice

The drop-down under **{{narr.voice}}** lists the voices of your library. Online voices that are not installed yet appear as "<name> (to download, <size>)"; they are downloaded and verified automatically when you press Start. Below the list a coloured **licence badge** shows what you may do with the result (see chapter 7 and the glossary in chapter 9):

* green – commercial use allowed (for example "CC-BY-4.0 · commercial use OK" and "Commercial OK");
* amber – restricted ("Personal use only · …", "Private only", "Public, non-commercial").

Under the badge a reminder repeats the limit, for example *{{narr.voice_personal_note}}* If you have no voices at all, the window says *{{narr.no_voice}}* and offers the buttons **{{studio.train_title}}** and **{{studio.voices_title}}**.

## Step 3 – prepare the text

*{{narr.prep_hint}}* Nothing is shown for review and your book file is never changed. All rule-based steps are ticked by default; untick what you do not want.

| Check box | What it does |
|---|---|
| {{prep.layout}} | {{prep.layout_d}} |
| {{prep.noise}} | {{prep.noise_d}} |
| {{prep.quotes}} | {{prep.quotes_d}} |
| {{prep.links}} | {{prep.links_d}} |
| {{prep.headings}} | {{prep.headings_d}} |
| {{prep.numbers}} | {{prep.numbers_d}} |
| {{prep.abbrev}} | {{prep.abbrev_d}} |
| {{prep.spellfix}} | {{prep.spellfix_d}} It needs a one-time download (a button **{{prep.model_download}}** appears; the model is about 365 MB) and is available for Russian books only. |
| ▸ {{prep.more}} | Greyed-out ideas for later versions: pauses and punctuation by AI, Russian stress marks, translation before narrating, different voices for characters. They are marked "{{prep.later_tag}}" and do nothing yet. |

The rules cover Russian and English fully. For other languages (for example German books) only the language-neutral steps – layout, footnotes, quotes, links – are applied.

## Step 4 – output format and quality

![The Narrate window, lower half: output format, quality, progress and the mini player.](@narrate_bottom)

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

**The recording script.** The repository contains a ready-made script, `docs/voice-script-ru-v3.txt` (Russian; it also has English and German passages). It has 18 blocks: greeting, numbers and dates, sounds of the language, neutral reading, calm, joy, sadness, irritation, surprise, whisper and loud speech, three short stories, optional English, optional technical terms, an optional adults-only block, and – at the very end – the consent statement. The script itself states three simple rules:

1. **Read only ordinary lines.** Everything in square brackets and every line with `===` is a hint – do not read it aloud.
2. **One line – one sentence.** Read it, breathe, stay silent for a second, then read the next line.
3. **Stumbled? Stay silent for a second and read the same sentence again from the beginning.** Do not stop the recording.

Further tips from the script: no need to act the emotions – a slightly different voice is enough, and if it does not work, read in your normal voice. Blocks marked OPTIONAL can be skipped. You choose this same script file as the "text" in the Train window.

> Voxprint is built to cope with re-read sentences: when a recording is matched with the script, the best reading of every line is kept. Still, the cleaner the recording, the better the voice.

**The consent statement (block 18).** Finish the recording with **one** of the consent sentences – the one that is true for you: *commercial* (A), *public non-commercial* (B) or *private only* (C), in Russian, English or German. Say your own name and today's date in place of the bracketed words. Read it only if the voice is yours or you have the owner's permission. Details in section 6.6.

## The Train window

Open it from the **{{studio.train_title}}** card. {{ui.subtitle}}

![The Train window with the "no transcript" mode switched on (English interface).](@train_notranscript)

| Control | What it does and when to use it |
|---|---|
| **{{ui.choose_audio}}** | Choose the voice recording (you can also drag the file into the window). Before a choice: *{{ui.audio_none}}* |
| **{{ui.choose_text}}** | Choose the text file you read (UTF-8 `.txt`; the recording script, for example). Before a choice: *{{ui.text_none}}* Hidden in no-transcript mode. |
| Check box *{{asr.checkbox}}* | Audio only, no text: see section 6.4. |
| **{{preset.label}}** | Drop-down with the training presets: see section 6.3. Below it the program shows a short description and an estimate of the training time for your graphics card. |
| ▶ **{{preset.advanced}}** | Shows the manual numbers (only editable in *{{preset.manual}}*). |
| **{{consent.title}}** | How the permission of the voice owner is recorded: see section 6.6. |
| **{{ui.btn_lora}}** | The main button: aligns the recording, cuts it, trains the voice and puts it in your library. |
| **{{preview.button}}** and the check box **{{preview.compare}}** | A quick listen-and-compare test before the long run: section 6.5. |
| Check box *{{check.checkbox}}* | After training, Voxprint re-reads a test sentence with the new voice and checks pitch and clarity automatically. Leave it on. |
| Voice type drop-down | Optional: *{{ui.voice_type_male}}*, *{{ui.voice_type_female}}*, *{{ui.voice_type_child}}*, *{{ui.voice_type_other}}*, or *{{ui.voice_type_none}}*. Saved with the voice. |
| Description field | *{{ui.voice_desc_placeholder}}* |
| **{{ui.btn_merge}}** | Merges the trained adapter into one standalone model that works in any app running Qwen3-TTS (section 6.7). |
| **{{ui.btn_dataset}}** | Only prepares the dataset (aligned and cut clips) without training. Useful if you want to train elsewhere. |
| Progress bar and stage chips | {{stage.model}}, {{stage.align}}, {{stage.slice}}, {{stage.train}}, {{stage.save}} – the stage that is running is highlighted. **{{ui.cancel}}** appears while a job runs. |
| Status line | Starts as *{{ui.status_idle}}* |
| Gear (top right) | Opens {{ui.settings_title}}. |

When the job ends, the status shows *{{ui.voice_registered}}* and a button **{{ui.open_folder}}** appears. The adapter folder is `…\output\<voice name>` (tens of MB); the dataset is in `…\<recording name>_Voxprint\dataset`.

In the default mode (with a text) the two buttons **{{ui.choose_audio}}** and **{{ui.choose_text}}** are shown one under the other, each with the name of the chosen file, and the transcript block with the warning is hidden. *(There is no separate screenshot of this default state; the picture above shows the no-transcript mode, which has all the other controls.)*

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

Tick *{{asr.checkbox}}* and the program recognises the speech itself (Qwen3-ASR, offline; the recognition model, about 1.9 GB, is downloaded on first use). The **{{ui.choose_text}}** row disappears; instead you have:

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

![My voices: three voice cards (demonstration voices) with licence and scope badges.](@voices)

Each voice is a card:

| Element | Meaning |
|---|---|
| Name and two badges | The **licence badge** (green "… · commercial use OK" or amber "Personal use only · …") and the **scope badge** ({{consent.badge_commercial}}, {{consent.badge_public_noncommercial}}, {{consent.badge_private_only}}). |
| Information line | Language · minutes of speech used for training · epochs (passes) · voice type · author · (for online voices) size. |
| Description | What the author wrote. A "test use only" voice also shows a warning. |
| **{{voices.preview}}** | Plays a short sample of the voice. **{{voices.stop}}** stops it. If the voice has no sample: *{{voices.no_preview}}* |
| **{{voices.narrate}}** | Opens the Narrate window with this voice already chosen. |
| **{{voices.edit}}** | Opens *{{voices.edit_title}}*: {{voices.field_name}}, {{voices.field_author}}, {{voices.field_license}}, {{voices.field_type}}, {{voices.field_description}}. Press **{{voices.save}}**. |
| **{{voices.delete}}** | Deletes the voice from this computer after a confirmation: *{{voices.delete_confirm_plain}}* |

The footer of the window repeats: *{{voices.rights_note}}*

When the library is empty the window says *{{voices.empty}}* and offers **{{voices.empty_train}}**.

## Online voices and the voice repository

![Voices that are available online but not installed yet show a Download button (demonstration data).](@voices_online)

* Voices from the online index appear by themselves as cards with the state *{{voices.remote_state}}*, their licence and size and a **{{voices.remote_download}}** button. The download is checked by a SHA-256 checksum and can be resumed.
* **{{voices.remote_refresh}}** reloads the list. Offline, Voxprint shows the last known list (*{{voices.repo_offline}}*).
* **{{voices.repo_button}}** opens the *{{voices.repo_title}}* dialog with **{{voices.repo_refresh}}**, a field for the address of the index (`index.json`) and **{{voices.repo_download}}**.

## Importing a voice

**{{voices.import}}** offers two ways: **{{voices.import_folder}}** (an adapter folder) and **{{voices.import_zip}}** (a voice package). Archives are checked for unsafe paths and size limits. A voice that arrives without a declared licence is treated as *personal use only*.

## Licences and the two voices that Voxprint offers

| Licence | Commercial use |
|---|---|
| CC0-1.0, CC-BY-4.0, CC-BY-SA-4.0 | allowed (keep attribution / share-alike if the licence asks for it) |
| CC-BY-NC-4.0, CC-BY-NC-SA-4.0 | not allowed |
| custom/personal-only (default) | not allowed; results stay on your computer |
| custom/test-use-only | not allowed; for testing only, no public release |

* **Open universal voice** (German UI: *Offene Universalstimme*, Russian UI: *Открытый универсальный голос*) – an English voice for people who have no recording of their own. Trained only from the public-domain LJ Speech dataset (reader Linda Johnson, LibriVox); licence **CC0-1.0** – free for any use, also commercial; attribution to the LJ Speech dataset (Keith Ito) is appreciated.
* **Александр (Alexander)** – a male Russian voice, the first one trained with Voxprint. Licence: **personal use only** – no public use of the output, no commercial projects (scope *{{consent.badge_private_only}}*).

Both are downloaded separately (not inside the installer) from the voice repository.

> The pictures here and in chapter 5 use demonstration voices; in the screenshots the voice "Alexander" still carries an earlier test-licence label. In the released voice list its badge reads "Personal use only" as described here.

The badge is information, not legal advice. The licence covers the voice model only.

# Settings

Open **{{ui.settings_title}}** with the gear button (top right of the Studio and the other windows).

![Settings.](@settings)

| Control | What it does |
|---|---|
| **{{ui.language}}** | The interface language: English, Deutsch, Русский. It changes at once in all windows. |
| **{{ui.btn_update}}** | Checks whether newer verified versions of components or models exist and installs them (with a safety check and automatic rollback). Voxprint also checks quietly once a week. Messages: *{{upd.up_to_date}}*, *{{upd.restart}}* |
| **{{ui.settings_models_folder}}** | Opens the folder with the downloaded models. |
| **{{ui.settings_data_folder}}** | Opens the data and log folder – send the log to the developer if an error repeats. |
| **{{backup.title}}** – check box **{{backup.include_voices}}** | Whether your voice library is part of the backup (on by default). |
| **{{backup.btn_backup}}** | Copies models (and voices) to a folder or drive you choose. Shows the size and free space first; resumable; identical files are skipped; every copy is checked. A progress bar and a **{{ui.cancel}}** button appear. |
| **{{backup.btn_restore}}** | Restores from a `Voxprint-backup` folder. Voices that already exist with different content are never overwritten. |
| **{{existing.title}}** | {{existing.hint}} **{{existing.choose}}** picks the folder, **{{existing.clear}}** forgets it. |
| **{{ui.settings_repair}}** | {{ui.settings_repair_tip}}. Your voices and models are kept. |
| **{{ui.about}}** | Version, the idea, how it works, open-source components with their licences, a link to the GitHub repository and third-party notices. |
| **{{about.btn_close}}** | Closes the dialog. |

If the program notices a damaged installation, a banner offers the repair (*{{ui.repair_offer_plain}}*). In the installed version, if repair does not help, run the Voxprint installer again – voices and models are kept.

When the program finds an older component in an environment it does not own (for example your own Python), it asks first: *{{upg.title}}* with **{{upg.btn_upgrade}}** or **{{upg.btn_later}}**; it never changes your environment silently.

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

<dt>Repair</dt><dd>Re-checks Voxprint's own environment and fixes what is broken.</dd>

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
| {{err.narration_no_ffmpeg}} | Settings → **{{ui.settings_repair}}**, or choose WAV. |
| {{err.book_unsupported}} | Convert the book to TXT, FB2 or EPUB. |
| {{err.book_unsafe}} | The file looks damaged or too large; try another copy. |
| {{warn.no_gpu}} | Training works without an NVIDIA card but is extremely slow. |
| {{warn.loss_low}} | The voice may babble. Use **{{preset.balanced}}**, record more text, and listen with the quick preview. |

**The voice sounds higher than mine.** A small upward pitch drift (about +2…+3 semitones was seen on a male voice) is a known effect of this kind of training. The automatic check warns above +4. Try the quick preview with both variants and take the one that is closer.

**The voice babbles or never stops.** Too many passes or too high a learning rate. Use *{{preset.balanced}}*, never raise the learning rate by hand.

**Can I stop and continue later?** Narration: yes, always (section 5.5). Downloads and backups: yes. Training: cancel is possible, but a full run starts over; use the quick preview first.

**Can I use my voice for a published audiobook?** Only if the consent scope allows it ({{consent.badge_commercial}} to sell; {{consent.badge_public_noncommercial}} to publish for free). Voices you train without a consent statement are *private only*. You can change the licence under **{{voices.edit}}** if you own the voice.

**Does Voxprint send my recordings anywhere?** No. It sends nothing to the internet except downloading models, checking for updates and, when you open the repository, fetching the voice index and the voices you choose. No telemetry, no account.

**Which languages are supported?** The interface: English, Deutsch, Русский. Text preparation covers Russian and English fully. The speech model supports further languages, but they are not tested.

**Where is the log?** {{ui.settings_data_folder}} in Settings. Send the file from there if an error repeats.

**Is the AAC / M4B option safe to use?** It is optional and off by default for good reason: AAC is patent-encumbered and you are responsible for compliance (section 5.4).

# Licences and responsibility

* **Voxprint's own source code is licensed under the Apache License 2.0** (Copyright 2026 Aleksandr Mitroshenkov; see the `LICENSE` and `NOTICE` files).
* **The Apache licence of the code does not cover models and voices.** The models Voxprint downloads (Qwen3-TTS, Qwen3-ASR and the optional text models) keep their authors' licences. Every voice has its own licence, shown as a badge and stored in `voice.json`.
* Third-party components, with their licences, are listed under **{{ui.about}}** and in `THIRD_PARTY_NOTICES.md` and the `licenses` folder of the installation. Qt/PySide6 is used under the LGPL-3.0; the bundled ffmpeg is a separate program with its own licence; the AAC codec is patent-encumbered.
* **Use voices responsibly.** A voice is personal data and a personality right. Use a voice only with the explicit permission of its owner and within its licence; no impersonation, fraud or misleading content; label synthetic speech where this is required. You are responsible for the lawful use of the voices and audio you create.

> **This manual and the licence badges are information, not legal advice.** The consent card records what a speaker said; it does not replace the law of your country.

*Voxprint AI Audiobook Builder 0.1.0 (beta). Manual of 3 October 2026.*
