# Command-line interface

Headless entry points for servers, agents and scripts. Same runners as the Studio window; no GPU in the unit tests (runners are injectable).

```
python main.py narrate|train|voices ...
Voxprint.exe narrate|train|voices ...          # packaged Windows build
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

`VOICE` is an id or display name (example: `[model_voice]`). The zip is the small voice package (adapter + `voice.json` + clips), importable in *My voices*.

```
python main.py voices export [model_voice] --out model_voice.zip
```
