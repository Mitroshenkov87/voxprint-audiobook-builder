# Voices index

`index.json` is the default voices index of Voxprint (schema 1). The voice packages themselves are **release assets** of the tag `voices-v1`
(`alexander.zip`, `open-universal.zip`); the index carries their URL, size and SHA-256. Rebuild a package and its entry with `tools/make_voice_package.py`.

| id | licence | note |
|---|---|---|
| `alexander` | `custom/personal-only` | male Russian voice; personal use only, no public use of the output, no commercial projects |
| `open-universal` | `CC0-1.0` | open English female voice trained only from public-domain LJ Speech; free for any use, attribution appreciated |
