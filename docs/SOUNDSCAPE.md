# Soundscape

Optional music and effects under a narration. Off by default. It runs only for a `.vxbook` whose manifest lists `extensions: ["sound/1"]` and whose archive contains a valid `sound.json`. A TXT, FB2, EPUB, Markdown book, or a `.vxbook` without that pair, is narrated exactly as before.

Turning it on downloads the model. Narration never downloads it. `--no-soundscape` skips the mix for one run and does not change the saved setting.

## Licence

The newest ACE-Step release whose code and weights are both permissive is **ACE-Step 1.5** (turbo stack). MIT is treated as permissive in the same way as Apache-2.0.

| Piece | Licence | Where |
|---|---|---|
| ACE-Step 1.5 code | MIT | [github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) ([LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step-1.5/main/LICENSE), Copyright (c) 2026 ACEStep) |
| ACE-Step 1.0 code | Apache-2.0 | [github.com/ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) ([LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step/main/LICENSE)). Older than 1.5, so it is not the pin |
| ACE-Step 1.5 weights | MIT (`cardData.license = mit`, tag `license:mit`) | [huggingface.co/ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5), API [huggingface.co/api/models/ACE-Step/Ace-Step1.5](https://huggingface.co/api/models/ACE-Step/Ace-Step1.5). Revision `19671f406d603126926c1b7e2adc169acbcade22` (last modified 2026-02-03). The card allows commercial use of generated music |
| Qwen3-Embedding-0.6B (inside that repo) | Apache-2.0 | [huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

Stable Audio Open 1.0 was checked and rejected: the Hugging Face card says `license: other` (Stability AI Community License, non-commercial). [huggingface.co/stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0).

## What is pinned

`infra/model_mirrors.json`, repo id `ACE-Step/Ace-Step1.5`, hashes only (no backup mirror), the same pattern as OpenVoice.

The pin is the **turbo** stack, not XL: `Qwen3-Embedding-0.6B`, `acestep-5Hz-lm-1.7B`, `acestep-v15-turbo`, and `vae`. 28 files, **10,092,102,593 bytes** (about 10.09 GB), including the model card. Files live in `<models folder>/ace-step-1.5`.

The model card says generation fits in under 4 GB of video memory when weights are offloaded. With nothing offloaded, the resident turbo stack is on the order of the weight size (about 10 GB). Generation is refused when a user VRAM cap leaves less than 4 GB free on a CUDA device. A machine with no measurable GPU is allowed to try.

The full installer, `models_for("all")`, and Check & repair do not fetch this repo. `infra.modules.not_a_runtime_module("soundscape")` is true, so the thin installer does not treat it as a wheel. The only download path is Settings → the soundscape checkbox, or `voxprint settings set narration.soundscape on`.

## Markup

Format version **1.1** (minor-safe: a 1.0 reader still opens the book; a major other than 1 is refused). Consumers act only when both of these are true:

* the manifest `extensions` array contains `sound/1`
* `sound.json` is present, hashed, and its `extension` is `sound/1` with version major 1

`sound.json` version is **1.0** of the extension. Unknown extension ids are ignored. A declaration without the file, or a file without the declaration, loads the book and leaves the soundscape off, with a warning.

Cues have `kind` `bed`, `accent`, `transition`, or `sfx`. An `sfx` cue is rendered on the one-shot path used for accents. If that render fails, the cue is skipped and a warning is logged; the chapter still mixes.

A bed has `layer` `music` or `ambience`, and an end paragraph in the same chapter. At most one music bed and one ambience bed overlap. At most one one-shot per paragraph per position, and two one-shots per paragraph. Caps: 200 cues in a chapter, 5000 in a book. A lower `conf` loses.

`position` is `before`, `after`, or `at_text`. `at_text` is a 3–60 character snippet of the paragraph's plain text. The cue starts at the beginning of the synthesis chunk that contains that snippet. If the snippet is not in any chunk, the cue starts at the paragraph. A transition with `before` starts in the pause in front of the paragraph. Paragraph numbers are 1-based and book-wide; headings and scene breaks are not counted.

`para_fp` is the first 12 hex digits of the SHA-256 of the raw `book.md` paragraph line (UTF-8, no newline), taken before text preparation. When the fingerprint or the chapter does not match, the cue is moved to the paragraph with that fingerprint, looking in the named chapter first and then in the rest of the book. If it is not found, the cue is dropped and a warning is logged. The same rule applies to a bed's `end_para_fp`.

`sound-cast.json` is the user's override, keyed by cue id: `disabled`, `gain_db` (−60 to +6) and `prompt`. Unknown keys are ignored. `enabled: false` turns the soundscape off for that book. `master_gain_db` shifts every cue. `asset` is reserved and ignored: sound/1 generates every cue from its prompt.

An inline `<!-- vx:sound cue="ID" -->` comment may sit in front of `vx:speaker`. It is optional and is not used for placement. When it names a different paragraph than `sound.json`, the JSON paragraph wins and a warning is logged.

### Levels

Gains are relative to the narration's speech loudness. Defaults, also written in the sample's `defaults` object:

| Kind | Level | Duck while speech is present |
|---|---|---|
| bed | −22 dB | 8 dB |
| accent | −18 dB | — |
| sfx, transition | −16 dB | — |

A bed is one generated segment of 45 seconds (inside the 30–60 s range), looped across the scene with a crossfade (default 2000 ms) and faded in and out (defaults 2000 ms and 3000 ms; the sample's first bed uses 3000 ms and 4000 ms).

After the mix, speech is scaled back to the root-mean-square it had in the narration, and the peak is limited to 0.99. That root-mean-square is the app's loudness target.

## Provisional

* `at_text` starts the cue at the synthesis chunk that contains the snippet. It is not a word-level aligner timestamp.
* `infra/soundscape_model.generate` builds `ACEStepPipeline(checkpoint_dir=...)` and calls it with `audio_duration` and `infer_step=8`. The `acestep` package is imported only there. That call has not been run against the weights in this tree.

The sample archive used by the tests is `tests/fixtures/sound-sample.vxbook`.
