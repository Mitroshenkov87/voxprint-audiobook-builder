# Bundled voices

Two open Russian voices come with Voxprint: **Tirzah** (female) and **Gideon** (male). They are part of the standard model download (Full setup: upfront;
Quick setup: with the first-launch download), are checked by SHA-256 and imported into the voices library **read-only** (they cannot
be edited or deleted; the licence is shown on the voice card). The weights are not in git: each voice is one zip on the
[`voices-v1`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/voices-v1) release. Pinned in
`infra/bundled_voices.py` and listed in `index.json`.

The names *Tirzah* and *Gideon* are Voxprint's own; they do not imply any endorsement by the readers.

*Boaz* (male) shipped with 0.1.3 and is retired. Gideon replaces it, used as recorded, with no retraining. The 0.1.3
sample that used Boaz stays in [`samples/`](../samples/README.md); that file is not a Gideon recording, and Boaz is not part of the download.

| Voice | Gender | Licence | Reader | Source recording (public domain) |
|---|---|---|---|---|
| Tirzah | female | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Anastasiia Solokha (LibriVox volunteer) | [Степные сказки (Stepnyia skazki)](https://librivox.org/stepnyia-skazki-by-grigory-danilevsky/) by Grigory Danilevsky, [Internet Archive](https://archive.org/details/stepnyyeskazki_2204_librivox) |
| Gideon | male | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Kazbek (LibriVox volunteer) | [Вехи (Vekhi)](https://librivox.org/vekhi-by-various/), a 1909 essay collection, [Internet Archive](https://archive.org/details/vekhi_2011_librivox) |

## Licence

LibriVox recordings are dedicated to the public domain by their volunteer readers ("anyone can use all our recordings however
they wish (even to sell them)", <https://librivox.org/pages/public-domain/>); the Internet Archive items carry the Public Domain
Mark 1.0. Each voice was trained only from its recording on Qwen3-TTS-12Hz-1.7B-Base and is released under
**CC0-1.0**: free for any use, also commercial. Attribution is appreciated, e.g. "voice trained from the public-domain
LibriVox recording 'Степные сказки (Stepnyia skazki)' read by Anastasiia Solokha" or "voice trained from the public-domain
LibriVox recording 'Вехи (Vekhi)' read by Kazbek". Consent block in `voice.json`: method
`manual`, scope `commercial`, the reader named as above. The base model's own licence (Apache-2.0) applies to the base model.

## Package

| File | Size (bytes) | SHA-256 |
|---|---|---|
| [`tirzah.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/tirzah.zip) | 53274433 | `aef7c849298ec6e3f2aac1114b993ca111b6297bfd9fef0bf1634527be7ab95f` |
| [`gideon.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/gideon.zip) | 53031093 | `6ca0e6824080ee86c21a4b34472bd013bd182ef15bb393c6e494fd554e1c2266` |

The zip holds `voice.json`, the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`), `speaker_centroid.safetensors`,
`ref_sample.wav`, `training_meta.json` and a model card. Rebuild: `python tools/make_voice_package.py --adapter <voice folder>
--spec tools/voice_specs/VOICE.json --out <dir> --plain-names --url https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/VOICE.zip`
(a rebuilt zip has a new hash: update `infra/bundled_voices.py`, `index.json` and this file).

Listen: [`samples/genesis-tirzah.mp3`](../samples/README.md) (Genesis 1:1-2:3 in Russian). There is no Gideon sample in the repository yet.
