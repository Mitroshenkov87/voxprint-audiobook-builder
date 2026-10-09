# Voxprint for agents

Hand this file to an agent that should **run** Voxprint (OpenClaw, Cursor, Claude Code, or another). It is not the contributor guide. Agents that change the source use the repository root [AGENTS.md](../AGENTS.md).

Voxprint is an offline desktop program. It trains a voice (LoRA on Qwen3-TTS) from a short recording and narrates TXT, FB2 and EPUB books in that voice. Optional pieces (translation, noise clean-up, a text model) are separate downloads. Beta. Apache-2.0. Windows 11 with an NVIDIA GPU is the supported setup; Linux is experimental.

The command reference with the same flags is [CLI.md](CLI.md).

## Where the program is

Substitute this executable for `voxprint` in every recipe below.

**Windows (default install).** `C:\Program Files\Voxprint\Voxprint.exe`

The installer uses that folder unless the person picked another one. The Start menu entry is "Voxprint AI Audiobook Builder". There is no desktop shortcut. If the file is not there:

```
where.exe Voxprint
```

**Linux (experimental).** The package is temporarily unavailable; the intended layout is still:

| Piece | Path |
|---|---|
| Launcher on `PATH` | `~/.local/bin/voxprint` |
| Program tree | `~/.local/share/voxprint/app` |
| Data, models, voices, logs | `~/.local/share/voxprint` (`$XDG_DATA_HOME/voxprint`, or `$VOXPRINT_HOME`) |

**Source checkout.** From the repository root, with the project environment active:

```
python main.py status --json
```

Same subcommands as the executable. Help text says `voxprint`.

**Data folders** (override the root with `VOXPRINT_HOME`):

| | Windows | Linux |
|---|---|---|
| Data root | `%LOCALAPPDATA%\Voxprint` | `~/.local/share/voxprint` |
| Voices | `%LOCALAPPDATA%\Voxprint\voices` | `~/.local/share/voxprint/voices` |
| Models | `%LOCALAPPDATA%\Voxprint\models` | `~/.local/share/voxprint/models` |
| Projects | `%LOCALAPPDATA%\Voxprint\Projects` | `~/.local/share/voxprint/Projects` |
| Logs | `%LOCALAPPDATA%\Voxprint\logs\voxprint.log` | `~/.local/share/voxprint/logs/voxprint.log` |

The models folder may be a different path if it was chosen during setup. `status` does not print that path (reports cut paths to names). A missing model still shows `"installed": false`.

## Rules for every command

- Nothing prompts. Do not expect a TTY question. `--yes` / `-y` is accepted on every command.
- Add `--json` whenever you will parse the output.
- Read `voxprint --help`, then `voxprint <command> --help`. Each command's help has copy-paste examples. Do not scrape the whole manual first.
- Set `VOXPRINT_LANG=en` so pipeline sentences stay English. The CLI's own `ERROR:` and `Fix:` lines are English either way.
- stderr carries the human error. With `--json`, stdout is only JSON.
- Exit 0 is success. Any other code is in the table below. The `Fix:` line and the JSON `hint` name the next command.

`status` and `capabilities` print JSON even without `--json`.

These are **not** user commands (they belong to install and repair): `--selftest`, `--auto-repair`, `--verify-install`, `--modules-status`, `--install-modules`. Python wheels on a thin install use `--install-modules`. Speech and translation models use `models download`.

## Quick start

```
voxprint --version
voxprint status --json
voxprint voices list --json
```

If `status` shows the TTS, aligner or speech model as not installed:

```
voxprint models download required --json
```

That downloads the TTS size, the aligner and the recogniser this PC should use. It does not download the optional noise tool or the Gemma text model.

### List voices

```
voxprint voices list --json
```

Use `id` as `--voice` on later commands. `license` and `consent_scope` say what you may do with the audio.

### Train from a folder of clips

Clips of one speaker, no transcript (the installed recogniser builds the text):

```
voxprint train ./clips --name Anna --type female --out ./voices --json
```

One recording plus the text that was read:

```
voxprint train recording.wav --text script.txt --name Anna --type female --json
```

On exit code 5, retry with `--force-cpu` (slow). When it finishes, `voices list --json` shows the new id. A second train creates another voice.

`--out DIR` is the parent folder: each voice gets its own work folder `<DIR>/<voice name>_Voxprint`, so two voices trained with the same `--out` never overwrite each other's `report.json`.

Licence, consent and language (all optional):

```
voxprint train ./clips --name MyVoice --type male --language ru --consent commercial --speaker "Reader Name" --license CC0-1.0 --json
```

| Option | Meaning |
|---|---|
| `--consent none` | Default. Consent method `none`, scope `private_only`, licence `custom/personal-only` |
| `--consent auto` | Read the speaker's spoken consent statement at the end of the recording (falls back to `private_only`) |
| `--consent commercial \| public_noncommercial \| private_only` | Consent confirmed manually (method `manual`) with that scope. Use it only when the speaker agreed, or the recording is public domain |
| `--speaker NAME` | Name of the person whose voice it is (consent block and `speaker` field) |
| `--license ID` | `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`, `custom/personal-only`, `custom/test-use-only`. Default: from the consent scope. A licence never allows more than the consent: a commercial licence needs `--consent commercial`, an `-NC` licence at least `public_noncommercial` (exit code 3 otherwise) |
| `--language CODE` | Language of the recording (`ru`, `en`, `de` or a name such as `Russian`); default: detected |

### Narrate

```
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3 --json
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json
voxprint narrate genesis.txt --voice tirzah --out ./audiobooks --style scripture --pause-sentence 0.7 --speed 0.95 --json
voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speakers --out ./audiobooks --json
```

Pauses and speed: `--pause-comma`, `--pause-mid`, `--pause-sentence`, `--pause-paragraph`, `--pause-chapter` (seconds), `--speed` (0.7-1.3) and `--style auto|scripture|fiction|dialogue`; defaults come from the app's Settings. For a Russian book, `--yo` (the default) restores the letter yo where a dictionary is sure; `--no-yo` leaves the letter e as written. Details: [CLI.md](CLI.md#narrate-a-book).

Multi-voice: `--voice` is the narrator. `--speakers` asks Gemma to mark each paragraph; `--speaker-marks FILE` narrates a file you already edited (`voxprint speakers` writes it). `--male-voice` and `--female-voice` are the role voices. `--male2-voice` / `--female2-voice` add a second voice per role: different characters (by the name in the marks) alternate between the two in order of first appearance. `--character NAME=VOICE` (repeatable) pins one character to a voice. Without those three flags every man and every woman share one voice, as before. The two speaker flags cannot be combined. If the marks do not match the prepared text, the narrator reads the whole book and the JSON `warnings` array says so. `<out>/<book>/.debug/speakers.txt` is listed in `outputs`.

### Speaker marks only

```
voxprint speakers book.txt --out marks.txt --json
voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speaker-marks marks.txt --out ./audiobooks --json
```

Does not synthesize. Exit 4 when Gemma is missing (`voxprint models download llm --json`).

### Prepare text, translate, settings

```
voxprint prepare book.txt --out book.prepared.txt --json
voxprint prepare book.txt --out yo-only.txt --steps yo --json
voxprint translate book.epub --to ru --out book.ru.txt --json
voxprint settings list --json
voxprint settings set narration.ordinals off --json
```

`prepare` is the window's Prepare text without narration: rule steps, the Russian letter yo, the Russian typo model when it is downloaded (not with `--steps` unless `--typos` is given; `--typos` requires it, exit 4), and with `--llm` the text model's rewrite. It writes plain text and `<out stem>.prep_report.json` (`rule_counts` has the changes per step, for example `{"yo": 22}`). `translate` uses the offline Opus-MT models (`--literary`: Gemma, Opus-MT fallback); a missing model is exit 4 with the module to download. `settings` reads and writes the values Settings shows; `models.folder` is read-only. Details: [CLI.md](CLI.md#prepare-text).

### Check and repair

```
voxprint check --json
voxprint repair --json
```

Same job as Settings -> Check & repair. `repair` is an alias; JSON `command` is `check`. Exit 0 when nothing failed, 1 when an item failed. Extra fields: `items`, `checked`, `fixed`, `failed`.

| You pass | File you get |
|---|---|
| `opus` or omit `--format` | one Opus file (`opus_single`) |
| `mp3` | MP3 per chapter |
| `m4b` | one M4B (AAC) |
| `flac` | FLAC per chapter |
| `wav` | WAV per chapter |

Also accepted: `mp3_single`, `opus_chapters`, `m4b_opus`. The job folder is `<out>/<book title>/`. The same command again skips chunks already on disk.

### Re-voice

Turn recordings into text, then narrate that text. Re-voice does not call the TTS model.

```
voxprint models download asr --json
voxprint revoice recording.wav --out ./revoice --json
voxprint narrate ./revoice/Intro.txt --voice my-voice --out ./audiobooks --format mp3 --json
```

A folder of audio files becomes one book, one chapter per file. `--title` sets the file name. `--language en` (or `ru`, `de`, …) is optional.

### Back up and restore models and voices

Copy every heavy file (model snapshots, `llm/`, `deepfilternet/`, `dnsmos/`, the voice library) to a drive, and bring it back on another PC or after a reinstall. Both commands resume: a file already there with the same size and SHA-256 is skipped.

```
voxprint backup --out E:\ --json
voxprint backup --out /mnt/usb --no-voices --json
voxprint restore --from E:\ --json
voxprint restore --from E:\ --link --json
```

`backup` writes `<DIR>/Voxprint-backup/` with the manifest `voxprint-backup.json` and stops before copying when the drive is too small. `--no-models` / `--no-voices` leave a part out. `restore` copies into the normal folders and checks every file against the manifest and the pinned hashes; a damaged or missing file is named in `problems` (and in `warnings`) and left for the normal download, so exit code 0 can still come with problems. `--link` copies the voices but reads the models from the backup in place: the drive must stay connected. A backup error (no room, no manifest, unreadable folder) exits 3.

### Diagnostics zip

```
voxprint diag --out voxprint-diagnostics.zip --json
```

The zip holds logs, `system_info.json` and a settings snapshot. Paths inside it are reduced to file names. Secret-like keys are dropped. Use this after exit code 1.

## JSON

Stdout with `--json` is newline-delimited JSON. Every line is one object. Progress lines come first. The last line is the result. Lines are flushed as they are written, so you can read them before the process exits.

### Progress line

```json
{"type":"progress","stage":"synth","percent":40.0,"message":"chapter 2","done":4,"total":10,"eta_s":90.0}
```

| Field | Always | Meaning |
|---|---|---|
| `type` | yes | `"progress"` |
| `stage` | yes | Stable stage id (`synth`, `train`, `model`, `download`, `transcribe`, …) |
| `percent` | yes | Number from 0 to 100. Narrate and train: whole job. Download and re-voice: that step |
| `message` | yes | Short status text |
| `done`, `total` | narrate | Chunks finished and planned |
| `eta_s` | narrate, when known | Seconds left |

### Result line

```json
{"type":"result","ok":true,"command":"narrate","exit_code":0,"outputs":["C:/work/book/book.mp3"],"warnings":[],"duration_s":12.5,"error":null,"hint":null,"out_dir":"C:/work/book"}
```

| Field | Always | Meaning |
|---|---|---|
| `type` | yes | `"result"` |
| `ok` | yes | `true` only when `exit_code` is 0 |
| `command` | yes | `version`, `status`, `capabilities`, `narrate`, `prepare`, `translate`, `train`, `voices list`, `voices export`, `speakers`, `settings`, `check`, `diag`, `models list`, `models download`, `revoice`, `backup`, `restore` |
| `exit_code` | yes | Same number the process returns |
| `outputs` | yes | Paths written (may be empty) |
| `warnings` | yes | Strings. Empty array when there are none |
| `duration_s` | yes | Seconds, 3 decimal places |
| `error` | yes | `null` on success, otherwise the error sentence |
| `hint` | yes | `null` on success, otherwise the fix |
| `details` | failures that have them | Technical detail. Do not show it to a listener |

Extra fields by command:

| Command | Extra fields |
|---|---|
| `version` | `name`, `version`, `build` (integer), `codename` |
| `status`, `capabilities` | see below. Always JSON, one object, no progress lines |
| `narrate` | `out_dir`. `outputs` includes `.debug/speakers.txt` when speaker marks were applied. `warnings` holds the mismatch sentence when the narrator read the whole book |
| `speakers` | `paragraphs` (integer). The marks file is `outputs[0]` |
| `prepare` | `report`: `{input, output, language, rules, rule_counts, rules_skipped, neural, typo_model, llm}`. `outputs` is the text, then the report file |
| `translate` | `source`, `target`, `chapters` (integer), `model` |
| `settings` | `settings` (object, for `list`) or `key` and `value` |
| `check` | `items` (`{kind, name, status, detail}`), `checked`, `fixed`, `failed`. `command` is `check` for `voxprint repair` too |
| `train` | `voice_id`, `adapter`, `root` |
| `voices list` | `voices`: array of `{id, name, voice_type, language, license, consent_scope, commercial_use, gender, age_group}` |
| `diag` | `summary`: array of text lines |
| `models list` | `modules`: array of `{id, title, optional, installed, kind}` |
| `models download` | `module`, `installed` (bool), `downloaded` (`false` when it was already there) |
| `revoice` | `chapters` (integer). The text path is `outputs[0]` |
| `backup`, `restore` | `target`, `copied_files`, `skipped_files`, `copied_bytes`, `problems`, `conflicts`, `external_models` (`--link`: the models folder in use, else empty). The backup folder is `outputs[0]` |

### `status` object

`gpu`: `{cuda_available, name, vram_total_gb, vram_free_gb, torch, driver}`. `name` is `null` when no CUDA device was seen.

`formats`: canonical ids (`opus_single`, `mp3_chapters`, `m4b`, `flac_chapters`, …). `format_aliases` maps `mp3`, `opus`, `flac`, `m4b`, `wav` onto those ids.

`modules`: one row per concrete model (`tts-1.7b`, `tts-0.6b`, `aligner`, `asr-0.6b`, `asr-1.7b`, text models such as `sage-ru` and `opus-big-en-ru`, `denoise`, `llm`, `dnsmos`, `openvoice`). `optional: false` means a core speech model. `installed: true` means the files are on disk.

`runtime.thin`: `false` on a normal install (Python libraries are already there). On a thin install, `runtime.modules` is filled only from a cached manifest. Otherwise use `Voxprint.exe --modules-status`.

`version`, `build`, `codename`, `platform` (`win32` or `linux`), `voices` (same shape as `voices list`).

## Exit codes

| Code | Name | What you do |
|---|---|---|
| 0 | ok | Read `outputs`. A download that was already installed is also 0, with `downloaded: false` |
| 1 | internal | `voxprint diag --out voxprint-diagnostics.zip --json` and read `error` |
| 2 | bad args | `voxprint <command> --help`. Unknown module: `voxprint models list --json` |
| 3 | input | The path or voice id is wrong. `voxprint voices list --json` for voices. Books are TXT, FB2, `.fb2.zip`, EPUB |
| 4 | missing | `voxprint models download <id> --json`. The `hint` names the id when it can |
| 5 | gpu | Free video memory, or `voxprint train ... --force-cpu`. Check `voxprint status --json` |
| 6 | cancelled | Run the same `narrate` again. Finished chunks are kept |

## Long jobs

Narrate and train often run for hours. Do not use a timeout of a few minutes.

- Parse stdout line by line. Do not wait for the process to exit before reading.
- The first progress line can be several minutes late while a model loads from disk. That quiet gap is normal.
- After progress has started, `percent` should move. Narrate also sends `done` / `total`.
- If you kill the process, it will not exit 6 (that code is the in-process cancel token, and the CLI has no cancel flag). Start the same `narrate` command again; cached chunks are skipped.
- Train does not resume as a no-op. Starting it again trains another voice.
- `models download`, `backup` and `restore` are safe to repeat.

## Safety

Voxprint stays on this computer. There is no account and no telemetry. The network is used to download models you asked for (and the installer's own component downloads). Details: [PRIVACY.md](PRIVACY.md).

Only train and re-voice speech you have the right to use. Do not commit recordings or someone else's voice into git.

By default a voice trained with this CLI is marked consent method `none` and scope `private_only` (licence `custom/personal-only`): the narration stays on that computer. `voices list --json` shows `license` and `consent_scope` for every voice. Obey them. Pass a wider `--consent` scope (and `--license`) only when the speaker agreed or the recording is public domain. This file is not legal advice. Terms: [EULA](legal/EULA-audiobook-builder.md). Voices: [VOICES.md](VOICES.md).

## Troubleshooting

| What you see | What to run |
|---|---|
| Exit 4, or `installed: false` on a model you need | `voxprint models download required --json` or the id from `models list` |
| Exit 5 | Close other GPU programs, or add `--force-cpu` on `train` |
| Exit 3, voice not found | `voxprint voices list --json` and pass `id` |
| Exit 2 | `voxprint <command> --help` |
| Exit 1 | `voxprint diag --out report.zip --json` |
| `voxprint` is not on PATH (Windows) | `"C:\Program Files\Voxprint\Voxprint.exe"`, or `where.exe Voxprint` |
| Linux launcher missing | The Linux package is temporarily unavailable. Use a source checkout: `python main.py …` |
| Translation model missing | `voxprint models download translate --json` or one id such as `opus-big-en-ru` |
| Re-voice says the recogniser is missing | `voxprint models download asr --json` |
| Models must move to another PC or drive | `voxprint backup --out <drive> --json` there, `voxprint restore --from <drive> --json` here |
| Help text is huge | You asked for the top-level page. Use `voxprint narrate --help` (or `train`, `models`, `revoice`) |
