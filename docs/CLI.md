# Command-line interface

Headless entry points for servers, agents and scripts. Same runners as the Studio window; no GPU in the unit tests (runners are injectable).

The file you can hand to an agent is [AGENTS.md](AGENTS.md) (what the program is, where the executable lives, recipes, JSON schemas, exit codes). This page is the command reference.

```
python main.py narrate|train|voices|status|models|revoice|diag|backup|restore ...
Voxprint.exe narrate|train|voices|status|models|revoice|diag|backup|restore ...    # packaged Windows build
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
    [--format NAME] [--no-pauses|--pauses] [--ai-disclosure] [--work-dir DIR] [--json]
```

| Argument | Meaning |
|---|---|
| `BOOK` | TXT, FB2, `.fb2.zip` or EPUB |
| `--voice` | Voice library id or display name |
| `--out DIR` | Working folder; the job lands in `<DIR>/<book title>/` |
| `--format` | Repeatable / comma-separated. Default: `opus_single`. Aliases: `opus`, `mp3`, `m4b`, `flac`, `wav`, … |
| `--no-pauses` | Default: explicit pauses off (same as the Studio) |
| `--pauses` | Turn explicit pauses on |
| `--ai-disclosure` | Speak a short AI note at the start (opt-in) |
| `--work-dir DIR` | Remember this folder as the app working folder |

```
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json
```

Running the same narrate command again skips chunks that are already cached.

## Train a voice

```
voxprint train AUDIO [--text SCRIPT] [--name NAME] [--type male|female|child|other] [--out DIR] [--force-cpu] [--json]
```

Omit `--text` to train from a folder of clips, or from one recording, in no-transcript mode (speech recognition builds the dataset). With `--text` and a single file, the aligner path is used.

A voice trained from this command is stored with consent method `none` and scope `private_only` (the narration stays on that computer). Confirm a wider scope in the app only when the speaker agreed. Only train a voice you have the right to use.

```
voxprint train recording.wav --text script.txt --name Anna --type female
voxprint train ./clips --name Anna --type female --out ./voices --json
voxprint train recording.wav --text script.txt --name Anna --force-cpu
```

`--force-cpu` is the retry after exit code 5. A second train creates another voice; it is not a no-op.

## Voices

```
voxprint voices list [--json]
voxprint voices export VOICE --out ZIP [--json]
```

Without `--json`, `voices list` prints one tab-separated line per voice: id, name, type, language code, licence, gender, age group (`-` = not set).

`VOICE` is an id or display name. The zip is the small voice package (adapter + `voice.json` + clips), importable in *My voices*.

```
voxprint voices export my-voice --out my-voice.zip
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

Recognises the files (or every audio file in a folder) with the installed speech model and writes a TXT book (`#` lines are chapter titles). It does not narrate. If the recogniser is missing, exit code is 4 and the fix is `voxprint models download asr`.

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
