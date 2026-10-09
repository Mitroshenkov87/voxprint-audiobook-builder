# Russian yo dataset

One record per word form for a text-to-speech pipeline, and the compact table Voxprint loads before narration.

The speech model in this project is stock Qwen3-TTS. It does not read stress marks, so narration uses only the runtime table and writes the letter yo where that table is sure. Stress position is stored here for other pipelines. It is not inserted into the text.

## Files

| File | What it is |
|---|---|
| `yo_runtime.tsv.gz` | Lookup table, 1,323,641 bytes. Each line is `ye-form<TAB>yo-form`. This is what `core/yo.py` loads. |
| `yo_dataset.jsonl.gz` | Published records, 1,257,057 bytes, 139,996 lines. |
| `yo_safe.txt` | Unambiguous source list, MIT, from [eyo-kernel](https://github.com/e2yo/eyo-kernel) `dictionary/safe.txt`. |
| `yo_not_safe.txt` | Ambiguous source list from the same project (`dictionary/not_safe.txt`). Not applied at runtime. |
| `yo_additions.json` | Project words. `текст` stays `текст` (plain e, stress on the first vowel). |
| `yo_safe.LICENSE` | MIT notice for the two source lists. |

Rebuild (no network): `python tools/build_yo_dataset.py`

## Record fields

`word` (ye spelling), `yo_form`, `stress` (0-based index of the stressed vowel; yo is always stressed), `type` (`unambiguous` or `homograph`), `source` (`eyo-kernel` or `project`). Homographs also have `variants`, each `{form, hint}`. The source lists do not give context, so those hints are empty. A project addition may set `hint`.

Unambiguous rows are safe to apply. Homograph rows are not: `все` / `всё` and `берег` / `берёг` stay as written unless a later model can choose.

## Licence

The source lists are MIT, Copyright (c) 2026 Denis Seleznev, https://github.com/e2yo/eyo-kernel. eyo-kernel says the lists were first taken from https://github.com/rin-nas/php-yoficator, which publishes no licence file, and are maintained under MIT. The build script and `core/yo.py` are Apache-2.0.

Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder

Nothing in this folder is published to Hugging Face from the repository. The gzipped JSONL is the form to release later.
