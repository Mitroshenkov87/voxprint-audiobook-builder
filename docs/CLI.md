# Command-line interface

Headless entry points for servers, agents and scripts. Same runners as the Studio window; no GPU in the unit tests (runners are injectable).

```
python main.py narrate|train|voices|diag|backup|restore ...
Voxprint.exe narrate|train|voices|diag|backup|restore ...    # packaged Windows build
```

Maintenance flags (`--selftest`, `--auto-repair`, …) are unchanged - see [BUILDING.md](BUILDING.md). The developer dataset dry-run stays at `python -m core.cli … --fake-aligner`.

## Narrate a book

```
python main.py narrate BOOK --voice ID_OR_NAME --out DIR
    [--format NAME] [--no-pauses|--pauses] [--ai-disclosure] [--work-dir DIR]
```

| Argument | Meaning |
|---|---|
| `BOOK` | TXT, FB2, `.fb2.zip` or EPUB |
| `--voice` | Voice library id or display name (example: `[model_voice]`) |
| `--out DIR` | Working folder; the job lands in `<DIR>/<book title>/` |
| `--format` | Repeatable / comma-separated. Default: `opus_single`. Aliases: `opus`, `mp3`, `m4b`, `flac`, `wav`, … |
| `--no-pauses` | Default: explicit pauses off (same as the Studio) |
| `--pauses` | Turn explicit pauses on |
| `--ai-disclosure` | Speak a short AI note at the start (opt-in) |
| `--work-dir DIR` | Remember this folder as the app working folder |

Example:

```
python main.py narrate book.epub --voice [model_voice] --out ./audiobooks --format opus,mp3
```

## Train a voice

```
python main.py train AUDIO [--text SCRIPT] [--name NAME] [--type male|female|child|other] [--out DIR] [--force-cpu]
```

Omit `--text` to use no-transcript mode (ASR builds the dataset from the recording). With `--text`, the aligner path is used.

Example:

```
python main.py train recording.wav --text script.txt --name Anna --type female
```

## Voices

```
python main.py voices list
python main.py voices export VOICE --out ZIP
```

`voices list` prints one tab-separated line per voice: id, name, type, language code, licence, gender, age group (`-` = not set).

`VOICE` is an id or display name (example: `[model_voice]`). The zip is the small voice package (adapter + `voice.json` + clips), importable in *My voices*.

```
python main.py voices export [model_voice] --out model_voice.zip
```

## Back up and restore models and voices

```
python main.py backup --out DIR [--no-models] [--no-voices] [--json]
python main.py restore --from DIR [--link] [--json]
```

`backup` copies the models folder (Hugging Face snapshots and the smaller trees `llm/`, `deepfilternet/`, `dnsmos/`, including the llama.cpp runtime) and the voice library into `<DIR>/Voxprint-backup/`. `voxprint-backup.json` records the app version and build and, for every file, its path relative to the backup, its size and its SHA-256. A file already in the backup with the same size and SHA-256 is skipped, so an interrupted copy continues. The command stops before writing anything when the drive does not have room.

`restore` copies that backup back into the normal folders (`%LOCALAPPDATA%\Voxprint\models` and `voices` on Windows) and checks each file against the manifest and the pinned hashes. A damaged or missing file is named and left out; the usual download fetches it. `--link` does not copy the models: it remembers the backup's models folder (`state/external_models_dir.txt`, override `VOXPRINT_EXTERNAL_MODELS`) and reads them from there. The drive has to stay connected. Voices are still copied. The model locator uses that folder even when `VOXPRINT_NO_EXTERNAL_MODELS` is set.

`--json` prints one object: `ok`, `target`, `copied_files`, `skipped_files`, `copied_bytes`, `problems`, `conflicts`, `external_models`.

## Diagnostic report

```
python main.py diag [--out ZIP]
```

Writes the log files, `system_info.json` and a settings snapshot (paths cut to names, secrets dropped) into one zip
(default: `voxprint-diagnostics-<date>.zip` in the current folder). Same as Settings -> *Save diagnostic report...*.

