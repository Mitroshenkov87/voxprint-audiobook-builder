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

### Narrate

```
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3 --json
voxprint narrate book.epub --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json
```

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
| `command` | yes | `version`, `status`, `capabilities`, `narrate`, `train`, `voices list`, `voices export`, `diag`, `models list`, `models download`, `revoice` |
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
| `narrate` | `out_dir` |
| `train` | `voice_id`, `adapter`, `root` |
| `voices list` | `voices`: array of `{id, name, voice_type, language, license, consent_scope, commercial_use, gender, age_group}` |
| `diag` | `summary`: array of text lines |
| `models list` | `modules`: array of `{id, title, optional, installed, kind}` |
| `models download` | `module`, `installed` (bool), `downloaded` (`false` when it was already there) |
| `revoice` | `chapters` (integer). The text path is `outputs[0]` |

### `status` object

`gpu`: `{cuda_available, name, vram_total_gb, vram_free_gb, torch, driver}`. `name` is `null` when no CUDA device was seen.

`formats`: canonical ids (`opus_single`, `mp3_chapters`, `m4b`, `flac_chapters`, …). `format_aliases` maps `mp3`, `opus`, `flac`, `m4b`, `wav` onto those ids.

`modules`: one row per concrete model (`tts-1.7b`, `tts-0.6b`, `aligner`, `asr-0.6b`, `asr-1.7b`, text models such as `sage-ru` and `opus-big-en-ru`, `denoise`, `llm`, `dnsmos`). `optional: false` means a core speech model. `installed: true` means the files are on disk.

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
- `models download` is safe to repeat.

## Safety

Voxprint stays on this computer. There is no account and no telemetry. The network is used to download models you asked for (and the installer's own component downloads). Details: [PRIVACY.md](PRIVACY.md).

Only train and re-voice speech you have the right to use. Do not commit recordings or someone else's voice into git.

A voice trained with this CLI is marked consent method `none` and scope `private_only` (licence `custom/personal-only`): the narration stays on that computer. `voices list --json` shows `license` and `consent_scope` for every voice. Obey them. Wider scopes (`public_noncommercial`, `commercial`) are confirmed in the app by the speaker, not by a CLI flag. This file is not legal advice. Terms: [EULA](legal/EULA-audiobook-builder.md). Voices: [VOICES.md](VOICES.md).

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
| Help text is huge | You asked for the top-level page. Use `voxprint narrate --help` (or `train`, `models`, `revoice`) |
