# Voices index

`index.json` is the default voices index of Voxprint (schema 1). Published packages are **release assets** of the tag `voices-v1`
(`open-universal.zip`, `tirzah.zip`, `gideon.zip`, `asher.zip`, `noa.zip`, `eitan.zip`); the index carries their URL, size and SHA-256. Rebuild a package and its entry with
`tools/make_voice_package.py`.

| id | licence | note |
|---|---|---|
| `open-universal` | `CC0-1.0` | open English female voice trained only from public-domain LJ Speech; free for any use, attribution appreciated |
| `tirzah` | `CC0-1.0` | open Russian female voice, bundled with Voxprint (see [BUNDLED.md](BUNDLED.md)) |
| `gideon` | `CC0-1.0` | open Russian male voice, bundled with Voxprint; replaces retired Boaz (see [BUNDLED.md](BUNDLED.md)) |
| `asher` | `CC0-1.0` | optional Russian male catalog voice; not part of the standard download |
| `noa` | `CC0-1.0` | optional Russian female catalog voice; not part of the standard download |
| `eitan` | `CC0-1.0` | optional Russian male catalog voice (a third male voice for multi-voice narration); not part of the standard download |

*Boaz* shipped with 0.1.3 and is no longer in this index. It is not installed.

## Asher and Noa (catalog only)

Asher (male, LibriVox reader Vladimir Anyanov, *Teachings of Christ* by Leo Tolstoy) and Noa (female, LibriVox reader
Hanna Ponomarenko, *Izbrannye* by Sholem Aleichem) are **not** part of the standard download. Specs:
`tools/voice_specs/asher.json`, `tools/voice_specs/noa.json`. Their packages are
`voices-v1/asher.zip` (53,066,747 bytes, `795778a10b85495e5850d885484c313e3f7799f76ea4d81d41fe50162165a583`) and
`voices-v1/noa.zip` (53,197,567 bytes, `755aeb47cc90a7bfb03d9349ed0fa8effbcc270fda22e5e9dc7b9539a462beed`).
Do not set `bundled`.

Eitan (male, LibriVox reader tovarisch, *Записки охотника (Zapiski Okhotnika)* by Ivan Turgenev, sections 02 and 04, 28.6 min;
Internet Archive `zapiskiohotnika_2409_librivox`, Public Domain Mark 1.0) is catalog-only too. Spec: `tools/voice_specs/eitan.json`
(source files and SHA-1). Package: `voices-v1/eitan.zip` (53,156,021 bytes,
`81627365a5f23824b13798486207eaa48950a36cb5869388949eb1a1d5f7d708`). Trained on the PC with
`voxprint train <folder> --name Eitan --type male --language ru --consent commercial --speaker tovarisch --license CC0-1.0`
(no-transcript mode: 146 of 147 clips kept, 1711 s).

To rebuild one from a trained folder `train-<id>/<Name>_Voxprint`:

```
python tools/make_voice_package.py --adapter <train-id>/<Name>_Voxprint --spec tools/voice_specs/asher.json --out <dir> --plain-names --url https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/voices-v1/asher.zip
```

(same for `noa.json`). A rebuilt zip has a new hash: update `url`, `sha256` and `size_bytes` in `index.json`.

Files the packager reads from that folder:

| File | Role |
|---|---|
| `adapter_model.safetensors` | required |
| `adapter_config.json` | required |
| `voice.json` | training facts (duration, epochs, base model); merged with the spec into the published `voice.json` |
| `ref_sample.wav` | reference clip; copied when present |
| `training_meta.json` | reference text; copied when present (`ref_sample_audio` is rewritten to the file name) |
| `speaker_centroid.safetensors` | copied when present |
| `preview.wav`, `preview.mp3` | copied when present |

The importer also accepts `consent_statement.wav` if it is inside the zip. The packager does not copy that file.
