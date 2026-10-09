# Voices index

`index.json` is the default voices index of Voxprint (schema 1). Published packages are **release assets** of the tag `voices-v1`
(`open-universal.zip`, `tirzah.zip`); the index carries their URL, size and SHA-256. Rebuild a package and its entry with
`tools/make_voice_package.py`.

| id | licence | note |
|---|---|---|
| `open-universal` | `CC0-1.0` | open English female voice trained only from public-domain LJ Speech; free for any use, attribution appreciated |
| `tirzah` | `CC0-1.0` | open Russian female voice, bundled with Voxprint (see [BUNDLED.md](BUNDLED.md)) |
| `asher` | `CC0-1.0` | optional Russian male catalog voice; hidden until the package URL and SHA-256 below are filled in |
| `noa` | `CC0-1.0` | optional Russian female catalog voice; hidden until the package URL and SHA-256 below are filled in |

*Boaz* shipped with 0.1.3 and is no longer in this index. It is not installed.

## Asher and Noa (catalog only)

Asher (male, LibriVox reader Vladimir Anyanov, *Teachings of Christ* by Leo Tolstoy) and Noa (female, LibriVox reader
Hanna Ponomarenko, *Izbrannye* by Sholem Aleichem) are **not** part of the standard download. Specs:
`tools/voice_specs/asher.json`, `tools/voice_specs/noa.json`. Their `index.json` entries use `https://PENDING_HOST/<id>.zip`,
`sha256` `PENDING_SHA256` and `size_bytes` 0. `parse_index` skips a hash that is not 64 hex digits, so the in-app catalog
does not offer them until those three fields are replaced with the real hosting URL, SHA-256 and size.

After the adapters are uploaded, from each trained folder `train-<id>/<Name>_Voxprint`:

```
python tools/make_voice_package.py --adapter <train-id>/<Name>_Voxprint --spec tools/voice_specs/asher.json --out <dir> --plain-names --url https://HOST/asher.zip
```

(same for `noa.json`). Paste `url`, `sha256` and `size_bytes` from the printed `<id>.index-entry.json` over the placeholders.
Do not set `bundled`.

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
