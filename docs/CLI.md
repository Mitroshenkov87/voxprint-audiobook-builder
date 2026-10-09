# Command-line interface

Headless entry points for servers, agents and scripts. Same runners as the Studio window; no GPU in the unit tests (runners are injectable).

The file you can hand to an agent is [AGENTS.md](AGENTS.md) (what the program is, where the executable lives, recipes, JSON schemas, exit codes). This page is the command reference.

```
python main.py narrate|prepare|translate|speakers|train|voices|settings|bench|check|status|models|revoice|diag|backup|restore ...
Voxprint.exe narrate|prepare|translate|speakers|train|voices|settings|bench|check|status|models|revoice|diag|backup|restore ...    # packaged Windows build
voxprint ...                                                         # Linux launcher, when installed
```

Every command is non-interactive. Nothing waits for a prompt. `--yes` / `-y` is accepted on every command and skips a confirmation if one is ever added; today no command asks.

Maintenance flags (`--selftest`, `--auto-repair`, `--modules-status`, `--install-modules`, …) are unchanged and are not these subcommands — see [BUILDING.md](BUILDING.md). The developer dataset dry-run stays at `python -m core.cli … --fake-aligner`.

## Global flags

| Flag | Meaning |
|---|---|
| `--json` | Progress as JSON lines (`type: progress`, with `stage` and `percent`), then one result object (`type: result`) |
| `--yes`, `-y` | Accepted everywhere. No command prompts |
| `--version`, `-V` | Print version, build number and codename, then exit |
| `--help`, `-h` | Help for the current command, with examples. `voxprint narrate --help` is the narrate page |

`--json` and `--yes` may sit before the command or after it. `--version` exits before the command runs.

A packaged `Voxprint.exe` is a windowed program. Stdout reaches a caller that redirects it (a pipe or a file). Every line is flushed. When nothing is attached, or the handle is invalid (Windows error 22, which PowerShell can leave on a windowed exe), the write is skipped and the process still exits with the command's code. It does not open a traceback window.

`status` and `capabilities` always print one JSON object, even without `--json`.

Pipeline errors that come from the UI catalogs follow `VOXPRINT_LANG` (default from the saved language). The CLI's own `ERROR:` / `Fix:` lines stay English. Set `VOXPRINT_LANG=en` when you parse those pipeline sentences.

## Exit codes

| Code | Name | When |
|---|---|---|
| 0 | ok | Success, including a module that was already installed |
| 1 | internal | Unexpected failure. Save `voxprint diag` |
| 2 | bad args | Missing or unknown arguments, unknown format, unknown module id |
| 3 | input | Book, audio, transcript or voice is missing or unreadable |
| 4 | missing | A model or component is not installed, or its download failed |
| 5 | gpu | Out of GPU memory |
| 6 | cancelled | The job stopped on the cancel token. Run the same `narrate` command again to resume cached chunks |

`ERROR:` goes to stderr and names a `Fix:` when there is one. With `--json`, stdout still ends with one result object (`ok: false`, `exit_code`, `error`, `hint`).

## Version

```
voxprint --version
voxprint --json --version
```

Prints the app version (for example `0.1.1-beta`), the build number and the codename. The codename is read from `BUILD.json` when that file is next to the program; otherwise the codename stamped into the app metadata is used. The build number is the CI stamp when one was written, otherwise the number computed from `BUILD.json` (`0` on a developer machine).

```
Voxprint 0.1.1-beta build 665 "Tikkun"
```

## Status

```
voxprint status --json
voxprint capabilities
```

`capabilities` is the same command. One JSON object: version, build, codename, GPU (`cuda_available`, `name`, VRAM), installed voices (id, name, licence, `consent_scope`), output formats and aliases, and module ids with `installed`. Use it before train or narrate. It does not download anything. With `VOXPRINT_NO_ENV_PROBE=1` the GPU probe does not import PyTorch.

## Narrate a book

```
voxprint narrate BOOK --voice ID_OR_NAME --out DIR
    [--format NAME] [--no-pauses|--pauses] [--pause-comma SEC] [--pause-mid SEC] [--pause-sentence SEC]
    [--pause-paragraph SEC] [--pause-chapter SEC] [--speed X] [--style auto|scripture|fiction|dialogue]
    [--ordinals|--no-ordinals] [--yo|--no-yo] [--ai-disclosure] [--work-dir DIR]
    [--speakers] [--male-voice ID] [--male2-voice ID] [--female-voice ID] [--female2-voice ID]
    [--character NAME=ID ...] [--speaker-marks FILE] [--json]
```

| Argument | Meaning |
|---|---|
| `BOOK` | TXT, FB2, `.fb2.zip` or EPUB |
| `--voice` | Voice library id or display name |
| `--out DIR` | Working folder; the job lands in `<DIR>/<book title>/` |
| `--format` | Repeatable / comma-separated. Default: `opus_single`. Aliases: `opus`, `mp3`, `m4b`, `flac`, `wav`, … |
| `--no-pauses` | Default: explicit pauses off (same as the Studio) |
| `--pauses` | Turn explicit pauses on (every comma is its own piece; may make the voice swallow short words) |
| `--pause-comma SEC` | Silence after a plain comma (default 0.25; used with `--pauses` and for forced cuts) |
| `--pause-mid SEC` | Strong break inside a sentence: dash, colon, semicolon, comma before a conjunction (и / но / а / and / but / und ...) (default 0.4) |
| `--pause-sentence SEC` | After . ! ? (default 0.6) |
| `--pause-paragraph SEC` | After a paragraph or a numbered / verse line (default 1.0) |
| `--pause-chapter SEC` | After a chapter title, at the end of a chapter and at a scene break (default 2.0) |
| `--speed X` | Global reading speed 0.7-1.3 (1 = the voice's own speed) |
| `--style NAME` | `auto` (detected), `scripture` (solemn, a little slower), `fiction`, `dialogue` |
| `--no-ordinals` / `--ordinals` | Read numbers after words like chapter / day / verse as ordinals by context ("день 1" -> "день первый", "21st", "3. Kapitel"); default: Settings (on). See [ORDINALS.md](ORDINALS.md) |
| `--no-yo` / `--yo` | For a Russian book, restore the letter yo where a dictionary is sure ("еще" -> "ещё"). "все" / "всё" is decided from the neighbouring words ("всё равно", "вот и всё", "всё было", "всё, что"; "все люди", "пришли все" stay). Other words the dictionary does not list, including "текст" and ambiguous pairs such as "берег", stay as written. Default: on. Stress marks are not inserted; the base speech model does not read them. |
| `--ai-disclosure` | Speak a short AI note at the start (opt-in) |
| `--work-dir DIR` | Remember this folder as the app working folder |
| `--speakers` | Ask the text model (Gemma) to mark each paragraph narrator, male or female, then narrate those voices. Same path as the Narrate window. Without any voice flag or `--character`, the default cast is used: Natan and Shimon for men, Miriam for women when they are installed, else the first library voices of each gender (never the narrator unless it is the only one). A block whose reply has the wrong number of marks is split and asked again, down to single paragraphs |
| `--male-voice ID` | Voice for paragraphs marked male. The narrator is `--voice` |
| `--female-voice ID` | Voice for paragraphs marked female |
| `--male2-voice ID` | Second male voice. Different male characters (by the name in the marks) alternate between `--male-voice` and this voice in order of first appearance: the first man gets `--male-voice`, the second this one, the third `--male-voice` again. A mark without a name uses `--male-voice`. Needs `--male-voice` |
| `--female2-voice ID` | Second female voice, the same way. Needs `--female-voice` |
| `--character NAME=ID` | Pin one character to a voice. Repeatable. The name is matched case-insensitively against the marks and wins over the role voices |
| `--speaker-marks FILE` | Narrate this marks file and do not run Gemma. `voxprint speakers` writes the file |

```
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json
voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speakers --out ./audiobooks --json
voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speaker-marks marks.txt --out ./audiobooks --json
voxprint narrate dialog.txt --voice levi --speakers --out ./out --json        # default cast: Natan, Shimon, Miriam
voxprint narrate dialog.txt --voice levi --male-voice natan --male2-voice shimon --female-voice rivka --speakers --out ./out --json
voxprint narrate dialog.txt --voice levi --male-voice natan --character Michael=shimon --female-voice miriam --speaker-marks marks.txt --out ./out
```

Without `--male2-voice`, `--female2-voice` and `--character`, every man is `--male-voice` and every woman `--female-voice`, as before. `--speakers` and `--speaker-marks` cannot be used together. At least one role voice must differ from the narrator (exit code 2 otherwise). `--male2-voice` without `--male-voice`, a `--character` value without `=`, and any voice flag without `--speakers` or `--speaker-marks` are exit code 2. Voices are loaded one at a time, in book order. When the marks do not match the prepared paragraph count, the narrator reads the whole book, the exit code stays 0, and the JSON `warnings` array contains `Speaker marks do not match the prepared text, so the narrator reads the whole book.` The job writes `<out>/<book>/.debug/speakers.txt` and lists that file in `outputs`.

Every narration cuts the text per sentence and at strong transitions, trims each spoken piece of its own silence and joins the pieces with the pause lengths above; long, comma-rich, descriptive or scripture-like sentences are time-stretched a little slower (pitch kept), dialogue stays at the voice's speed. Defaults come from Settings (*Narration: pauses and speed*); the flags override them for one run.

Running the same narrate command again skips chunks that are already cached (pause, speed and style changes never re-synthesize).

## Speaker marks

```
voxprint speakers BOOK --out FILE [--json]
```

Ask Gemma who speaks each paragraph and write an editable text file. This command does not synthesize audio. The file is one numbered line per paragraph:

```
1. NARRATOR
2. FEMALE: Ann
3. MALE: Tom
```

Blank lines are ignored. A trailing `mismatch` line (written when a previous narration could not apply the marks) is ignored. Edit the file, then pass it to `narrate --speaker-marks`. If Gemma is not installed the exit code is 4 and the fix is `voxprint models download llm --json`.

If the model's reply cannot be read as marks, or the text has dialogue (quotes or a leading em dash) but every paragraph is marked `NARRATOR`, the exit code stays 0 and the JSON `warnings` array says so. The raw reply is saved next to the marks file as `<name>.speakers-raw.txt` and listed in `outputs`. Narration saves the same reply as `<out>/<book>/.debug/speakers-raw.txt`.

```
voxprint speakers book.txt --out marks.txt --json
voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speaker-marks marks.txt --out ./audiobooks --format wav --json
```

## Prepare text

```
voxprint prepare BOOK --out FILE [--report FILE] [--language ru|en|de]
    [--steps NAME,...] [--no-rules] [--yo|--no-yo] [--typos|--no-typos] [--llm] [--work-dir DIR] [--json]
```

The Narrate window's *Prepare text* step without narration. It writes the prepared book as plain UTF-8 text (chapters separated by two blank lines) and a JSON report. Nothing is synthesized.

| Argument | Meaning |
|---|---|
| `--out FILE` | Prepared text |
| `--report FILE` | JSON report. Default: `<out stem>.prep_report.json` next to `--out` |
| `--language` | Book language hint. Default: detected |
| `--steps` | Only these rule steps (repeatable or comma-separated): `layout`, `noise`, `quotes`, `links`, `headings`, `numbers`, `abbrev`, `yo`. Default: all, as the Prepare text switch |
| `--no-rules` | No rule step |
| `--yo` / `--no-yo` | Add or remove the letter-yo step (Russian only) |
| `--typos` / `--no-typos` | Russian typo model. Default: used on a full preparation when downloaded, otherwise skipped with a warning. With `--steps` it runs only when `--typos` is given, so `--steps yo` changes nothing but the letter yo. `--typos` exits 4 when the model is missing |
| `--llm` (alias `--markup`) | Also run the text model's narration rewrite (Gemma, off by default, as in the window). Exit 4 when Gemma is not installed |
| `--work-dir DIR` | Keep the text model's cache. Default: a temporary folder |

The report has `language`, `rules` (steps that ran), `rule_counts` (changes per step, for example `{"yo": 22}`), `rules_skipped` (steps the language does not have), `neural` (typo model statistics), `typo_model` and `llm`. With `--json`, the result object carries the same report as `report`.

```
voxprint prepare book.txt --out book.prepared.txt --json
voxprint prepare book.txt --out yo-only.txt --steps yo
voxprint prepare book.fb2 --out prepared.txt --llm --json
```

## Translate

```
voxprint translate BOOK --to en|ru|de --out FILE [--from en|ru|de|uk] [--literary] [--work-dir DIR] [--json]
```

The offline translation the Narrate window uses (Opus-MT, through English when there is no direct model). `--literary` lets the text model (Gemma) translate whole paragraphs, with Opus-MT for titles and as the fallback. Writes plain UTF-8 text; nothing is synthesized. A missing translation model is exit 4, and the hint names the module (`voxprint models download opus-big-en-ru --json`). The same language on both sides, or an unsupported pair, is exit 2. `--work-dir` keeps the sentence cache, so a second run is quick. The JSON result adds `source`, `target`, `chapters` and `model`.

```
voxprint translate book.epub --to ru --out book.ru.txt --json
voxprint translate book.txt --from de --to en --out book.en.txt --literary
```

## Settings

```
voxprint settings list [--json]
voxprint settings get KEY [--json]
voxprint settings set KEY VALUE [--json]
```

| Key | Value |
|---|---|
| `language` | UI language: `en`, `de`, `ru`, `uk`, `lv` |
| `projects.folder` | Projects (working) folder, an absolute path. Created when missing |
| `models.folder` | Read-only. Set by the installer or by `VOXPRINT_MODELS_DIR` |
| `narration.speed` | 0.7-1.3 |
| `narration.style` | `auto`, `scripture`, `fiction`, `dialogue` |
| `narration.pauses` | `on` / `off` |
| `narration.ordinals` | `on` / `off` |
| `narration.ai_disclosure` | `on` / `off` |
| `narration.pause.comma`, `.mid`, `.sentence`, `.paragraph`, `.chapter` | Seconds, 0-6 |
| `theme` | Shared by the Voxprint programs (`state/suite.json`). This program has one look, `glass-dark`; another program's theme id is kept and read as `glass-dark` |
| `gpu` | Shared by the Voxprint programs: `auto`, `cpu` or `cuda:N`. Narration uses it; a GPU that is not there falls back to `cuda:0` |
| `gpu.vram_fraction` | 0.70-0.80 (default 0.75): the most of the card's total video memory narration plans for, counting what other programs hold. Environment: `VOXPRINT_VRAM_FRACTION` |
| `gpu.fast_decode` | `off` (default) or `graphs`: one chunk at a time with CUDA Graphs (needs faster-qwen3-tts, see [Bench](#bench)). Environment: `VOXPRINT_FAST_DECODE` |

These are the files Settings writes, so the window and `narrate` use them. An unknown key or a bad value is exit 2. `--json` adds `settings` (for `list`) or `key` and `value`.

`language`, `theme` and `gpu` (and the models folder chosen in the installer) are also written to `state/suite.json`, the settings file shared with Voxprint AI Movie Dubber, so both programs follow one choice.

## Bench

```
voxprint bench [--voice VOICE] [--modes batched,graphs] [--out DIR] [--install-graphs] [--json]
```

Narrates a fixed Russian text (12 phrases) with the same voice once per mode and prints, per mode, the realtime factor (`x_realtime` = seconds of audio per second of synthesis; `rtf` = the inverse), the batch size, load and warm-up time, and the peak video memory (`peak_vram_gb` held by PyTorch, `card_peak_used_gb` including other programs). `batched` is the normal path (several phrases per generate call, batch planned by the VRAM rule); `graphs` is one phrase at a time through faster-qwen3-tts with CUDA Graphs. A mode that cannot run is reported as skipped with the reason (warning); when no mode runs the exit code is 5. `--out` also saves `bench-<mode>.wav`. `--install-graphs` first downloads faster-qwen3-tts 0.3.2 (MIT, 43 KB, SHA-256 checked) into the packages folder; nothing else is installed. Default voice: Levi.

## Train a voice

```
voxprint train AUDIO [--text SCRIPT] [--name NAME] [--type male|female|child|other] [--out DIR]
               [--consent none|auto|commercial|public_noncommercial|private_only] [--speaker NAME]
               [--license ID] [--language CODE] [--force-cpu] [--json]
```

Omit `--text` to train from a folder of clips, or from one recording, in no-transcript mode (speech recognition builds the dataset). With `--text` and a single file, the aligner path is used.

`--out DIR` is the parent folder; the work folder is `<DIR>/<voice name>_Voxprint` (one per voice).

By default the voice is stored with consent method `none` and scope `private_only` (the narration stays on that computer). `--consent auto` reads the spoken statement; `--consent commercial|public_noncommercial|private_only` records a manually confirmed scope, `--speaker` the speaker's name, `--license` the voice licence (never more than the consent allows) and `--language` the recording's language. Details: [AGENTS.md](AGENTS.md#train-from-a-folder-of-clips). Only train a voice you have the right to use.

```
voxprint train recording.wav --text script.txt --name Anna --type female
voxprint train ./clips --name Anna --type female --out ./voices --json
voxprint train recording.wav --text script.txt --name Anna --force-cpu
voxprint train ./clips --name MyVoice --language ru --consent commercial --speaker "Reader Name" --license CC0-1.0
```

`--force-cpu` is the retry after exit code 5. A second train creates another voice; it is not a no-op.

## Voices

```
voxprint voices list [--json]
voxprint voices export VOICE --out ZIP [--json]
voxprint voices catalog [--json]
voxprint voices download VOICE [--json]
```

Without `--json`, `voices list` prints one tab-separated line per voice: id, name, type, language code, licence, gender, age group (`-` = not set).

`VOICE` is an id or display name. The zip is the small voice package (adapter + `voice.json` + clips), importable in *My voices*.

```
voxprint voices export my-voice --out my-voice.zip
```

`voices catalog` lists the online voice catalog (`voices/index.json` in this repository; the cached copy when offline): id, name, gender, language, licence, size, and `installed`. `voices download VOICE` downloads one catalog voice by id or name, checks its SHA-256, imports it, and exits 0 without downloading when it is already installed. Exit 4: the catalog cannot be fetched and there is no cached copy. Exit 3: no such voice in the catalog.

```
voxprint voices catalog
voxprint voices download eitan --json
```

## Models

```
voxprint models list [--json]
voxprint models download MODULE [--yes] [--json]
```

`download` fetches one module if it is missing and exits 0 without fetching when it is already installed. Ids (also printed by `models list` and `models download --help`):

| Id | What it fetches |
|---|---|
| `tts` | The TTS base model this PC should use (1.7B or 0.6B) |
| `tts-1.7b`, `tts-0.6b` | That TTS size |
| `aligner` | Qwen3 forced aligner |
| `asr` | The speech-recognition model this PC should use |
| `asr-0.6b`, `asr-1.7b` | That recogniser size |
| `required` | `tts`, `aligner` and `asr` |
| `sage-ru`, `opus-big-en-ru`, … | One integrated text model (see `models list`) |
| `translate` | Every integrated Opus-MT direction |
| `denoise` | DeepFilterNet3 noise clean-up (optional; alias `deepfilternet`) |
| `llm` | Gemma 4 12B text model (optional; alias `gemma`) |
| `dnsmos` | DNSMOS file used by the voice check (alias `mos`) |
| `openvoice` | OpenVoice V2 voice converter for *Re-voice recording* (~130 MB; aliases `vc`, `openvoice-v2`) |

```
voxprint models list --json
voxprint models download denoise --json
voxprint models download opus-big-en-ru
```

Thin-build Python wheels (PyTorch and the rest) stay on `Voxprint.exe --modules-status` and `--install-modules`. `models download` does not install those wheels.

## Re-voice

```
voxprint revoice AUDIO [AUDIO ...] [--out DIR] [--title NAME] [--language CODE] [--json]
```

Recognises the files (or every audio file in a folder) with the installed speech model and writes a TXT book (`#` lines are chapter titles). It does not narrate. `--language` takes an ISO code or a language name in any case (`ru`, `EN`, `de-DE`, `Russian`, `русский`); `auto` or no flag means automatic detection, and an unknown value is exit code 2. If the recogniser is missing, exit code is 4 and the fix is `voxprint models download asr`.

```
voxprint revoice recording.wav --out ./revoice --json
voxprint narrate ./revoice/Intro.txt --voice my-voice --out ./audiobooks --format mp3
```

## Back up and restore models and voices

```
voxprint backup --out DIR [--no-models] [--no-voices] [--json]
voxprint restore --from DIR [--link] [--json]
```

`backup` copies the models folder (Hugging Face snapshots and the smaller trees `llm/`, `deepfilternet/`, `dnsmos/`, including the llama.cpp runtime) and the voice library into `<DIR>/Voxprint-backup/`. A backup or restore error (no room, unreadable or missing backup) exits 3. `voxprint-backup.json` records the app version and build and, for every file, its path relative to the backup, its size and its SHA-256. A file already in the backup with the same size and SHA-256 is skipped, so an interrupted copy continues. The command stops before writing anything when the drive does not have room.

`restore` copies that backup back into the normal folders (`%LOCALAPPDATA%\Voxprint\models` and `voices` on Windows) and checks each file against the manifest and the pinned hashes. A damaged or missing file is named and left out; the usual download fetches it. `--link` does not copy the models: it remembers the backup's models folder (`state/external_models_dir.txt`, override `VOXPRINT_EXTERNAL_MODELS`) and reads them from there. The drive has to stay connected. Voices are still copied. The model locator uses that folder even when `VOXPRINT_NO_EXTERNAL_MODELS` is set.

`--json` prints the usual result object (`type: result`, `ok`, `exit_code`, `outputs`, …) with these extra fields: `target`, `copied_files`, `skipped_files`, `copied_bytes`, `problems`, `conflicts`, `external_models`.

## Diagnostic report

```
voxprint diag [--out ZIP] [--json]
```

Writes the log files, `system_info.json` and a settings snapshot (paths cut to names, secrets dropped) into one zip (default: `voxprint-diagnostics-<date>.zip` in the current folder). Same as Settings -> *Save diagnostic report...*.

## Check and repair

```
voxprint check [--json]
voxprint repair [--json]
```

The same check as Settings -> *Check & repair* (`infra.auto_repair.run`): program, components and models by hash, missing or damaged files fetched again, stale model lock files removed. `repair` is an alias; the JSON `command` field is always `check`. Exit code 0 when every item is ok, repaired, downloaded or skipped; exit code 1 when any item failed.

`--json` adds `items` (each `{kind, name, status, detail}`), `checked`, `fixed` and `failed`. `fixed` counts items whose status is `repaired` or `downloaded`. A failed item is also named in `warnings`.

`Voxprint.exe --auto-repair` stays the maintenance flag (it also writes `logs/auto_repair.txt`). `voxprint check` is the user command and prints to stdout.

## JSON result

With `--json`, each progress line and the final line is one JSON object. Full field list: [AGENTS.md](AGENTS.md#json).

Progress:

```json
{"type":"progress","stage":"synth","percent":40.0,"message":"chapter 2","done":4,"total":10}
```

Result:

```json
{"type":"result","ok":true,"command":"narrate","exit_code":0,"outputs":["book.mp3"],"warnings":[],"duration_s":12.5,"error":null,"hint":null}
```

`percent` is 0–100 for the whole narrate or train job, and for the current download or transcription. Lines are flushed as they are printed.
