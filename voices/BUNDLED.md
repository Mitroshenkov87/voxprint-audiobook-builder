# Bundled voices

Two open Russian voices come with Voxprint (since 0.1.3-beta build 667). They are part of the standard model download (Full
setup: upfront; Quick setup: with the first-launch download), are checked by SHA-256 and imported into the voices library
**read-only** (they cannot be edited or deleted; the licence is shown on the voice card). The weights are not in git: each voice
is one zip on the [`voices-v1`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/tag/voices-v1) release.
Pinned in `infra/bundled_voices.py` and listed in `index.json`.

The names *Boaz* and *Tirzah* are Voxprint's own; they do not imply any endorsement by the readers.

| Voice | Gender | Licence | Reader | Source recording (public domain) |
|---|---|---|---|---|
| Boaz | male | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Mark Chulsky (LibriVox volunteer) | [Аграфена (Agrafena)](https://librivox.org/agrafena-by-boris-zaytsev/) by Boris Zaytsev, [Internet Archive](https://archive.org/details/agrafena_2007_librivox) |
| Tirzah | female | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Anastasiia Solokha (LibriVox volunteer) | [Степные сказки (Stepnyia skazki)](https://librivox.org/stepnyia-skazki-by-grigory-danilevsky/) by Grigory Danilevsky, [Internet Archive](https://archive.org/details/stepnyyeskazki_2204_librivox) |

## Licence

LibriVox recordings are dedicated to the public domain by their volunteer readers ("anyone can use all our recordings however
they wish (even to sell them)", <https://librivox.org/pages/public-domain/>); the Internet Archive items carry the Public Domain
Mark 1.0. The voices were trained only from those recordings (about 28 minutes each) on Qwen3-TTS-12Hz-1.7B-Base and are
released under **CC0-1.0**: free for any use, also commercial. Attribution is appreciated, e.g. "voice trained from the
public-domain LibriVox recording 'Аграфена (Agrafena)' read by Mark Chulsky". Consent block in `voice.json`: method `manual`,
scope `commercial`, the reader named as above. The base model's own licence (Apache-2.0) applies to the base model.

## Packages

| File | Size (bytes) | SHA-256 |
|---|---|---|
| [`boaz.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/boaz.zip) | 53177559 | `138acc6aeeeeb31de5ee31a3adc8320cf60446f17c5178ba83a37e47741638cc` |
| [`tirzah.zip`](https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/tirzah.zip) | 53274433 | `aef7c849298ec6e3f2aac1114b993ca111b6297bfd9fef0bf1634527be7ab95f` |

Each zip holds `voice.json`, the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`), `speaker_centroid.safetensors`,
`ref_sample.wav`, `training_meta.json` and a model card. Rebuild: `python tools/make_voice_package.py --adapter <voice folder>
--spec tools/voice_specs/boaz.json --out <dir> --plain-names --url https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/VOICE.zip`
(a rebuilt zip has a new hash: update `infra/bundled_voices.py`, `index.json` and this file).

Listen: [`samples/`](../samples/README.md) (Genesis 1:1-2:3 in Russian, narrated with each voice).
