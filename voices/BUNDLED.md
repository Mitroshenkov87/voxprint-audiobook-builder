# Bundled voices

Two open Russian voices come with Voxprint: **Levi** (male) and **Miriam** (female). They are part of the standard model download
(Full setup: upfront; Quick setup: with the first-launch download), are checked by SHA-256 and imported into the voices library
**read-only** (they cannot be edited or deleted; the licence is shown on the voice card). The weights are not in git: each voice is
one zip on the [`voices-v2`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/voices-v2) release. Pinned in
`infra/bundled_voices.py` and listed in `index.json`.

The names *Levi* and *Miriam* are Voxprint's own; they do not imply any endorsement by the readers.

**Default cast for multi-voice narration** (`infra/bundled_voices.py: DEFAULT_CAST`): narrator Levi, male roles Natan and
Shimon, female role Miriam. Natan and Shimon are catalog voices (one click in the Voices window, or `voxprint voices download
natan`). A role whose voice is not installed falls back to the first library voice of that gender.

**Earlier bundled voices.** Up to build 702 the bundled pair was *Gideon* (male) and *Tirzah* (female) from `voices-v1`; *Boaz*
shipped with 0.1.3. They are retired: no longer downloaded and hidden in the catalog. An install that already has them keeps
them (nothing is deleted) and can still use them as narrator or for a role.

| Voice | Gender | Licence | Reader | Source recording (public domain) |
|---|---|---|---|---|
| Levi | male | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Виталий (LibriVox volunteer) | Anton Chekhov, «Капитанский мундир», «Либерал», «Радость» in the [Multilingual Short Works Collection 037](https://librivox.org/multilingual-short-works-collection-037-poetry-prose-by-various/), [Internet Archive](https://archive.org/details/mlswc037_2502_librivox) |
| Miriam | female | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Maya S (LibriVox volunteer) | Ilya Ehrenburg, [Портреты русских поэтов (Portraits of Russian Poets)](https://librivox.org/portraits-of-russian-poets-by-ilya-ehrenburg/), [Internet Archive](https://archive.org/details/portraitsrussianpoets_1612_librivox) |

Source files with SHA-1 and the cut list: `tools/voice_specs/levi.json`, `tools/voice_specs/miriam.json`.

## Licence

LibriVox recordings are dedicated to the public domain by their volunteer readers ("anyone can use all our recordings however
they wish (even to sell them)", <https://librivox.org/pages/public-domain/>); the Internet Archive items carry the Public Domain
Mark 1.0. Each voice was trained only from its recording on Qwen3-TTS-12Hz-1.7B-Base and is released under
**CC0-1.0**: free for any use, also commercial. Attribution is appreciated, e.g. "voice trained from the public-domain LibriVox
recording of Chekhov's short stories read by Виталий" or "voice trained from the public-domain LibriVox recording 'Портреты
русских поэтов' read by Maya S". Consent block in `voice.json`: method `manual`, scope `commercial`, the reader named as above.
The base model's own licence (Apache-2.0) applies to the base model.

## Package

| File | Size (bytes) | SHA-256 |
|---|---|---|
| [`levi.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v2/levi.zip) | 53227088 | `45d649f2665a0a98a7a69e6694e279745e2d9469245470f6681464fd02744ea3` |
| [`miriam.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v2/miriam.zip) | 53215075 | `a43a0e7e834603086e3aac623760e28271990a69074db6d968d2b1d2f1e9ef0c` |

The zip holds `voice.json`, the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`), `speaker_centroid.safetensors`,
`ref_sample.wav`, `training_meta.json` and a model card. Rebuild: `python tools/make_voice_package.py --adapter <voice folder>
--spec tools/voice_specs/VOICE.json --out <dir> --plain-names --url https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v2/VOICE.zip`
(a rebuilt zip has a new hash: update `infra/bundled_voices.py`, `index.json` and this file). The same voices are mirrored on
Hugging Face: <https://huggingface.co/Mitroshenkov87/voxprint-voices-ru>.

Older samples (`samples/genesis-tirzah.mp3`, `samples/genesis-boaz.mp3`) stay in [`samples/`](../samples/README.md); they are not
Levi or Miriam recordings.
