# Russian yo dataset

One record per word form for a text-to-speech pipeline, and the compact table Voxprint loads before narration.

The speech model in this project is stock Qwen3-TTS. It does not read stress marks, so narration uses only the runtime table and writes the letter yo where that table is sure. Stress position is stored here for other pipelines. It is not inserted into the text.

## Files

| File | What it is |
|---|---|
| `yo_runtime.tsv.gz` | Lookup table, 1,323,843 bytes. Each line is `ye-form<TAB>yo-form`. This is what `core/yo.py` loads. |
| `yo_dataset.jsonl.gz` | Published records, 1,257,996 bytes, 139,996 lines. |
| `yo_safe.txt` | Unambiguous source list, MIT, from [eyo-kernel](https://github.com/e2yo/eyo-kernel) `dictionary/safe.txt`. |
| `yo_not_safe.txt` | Ambiguous source list from the same project (`dictionary/not_safe.txt`). Not applied at runtime. |
| `yo_additions.json` | Project words. `текст` stays `текст` (plain e, stress on the first vowel). Build 703 adds `твёрдо`, `твёрды` and the present tense of `признавать`, `распознавать`, `осознавать` (`признаёт`, `распознаёт`, `осознаёт` ...): eyo-kernel lists them as homographs, the project writes yo because that reading is the common one (the hint in each record says so). |
| `yo_safe.LICENSE` | MIT notice for the two source lists. |
| `yo_context.json` | Project rules (Apache-2.0), read by `core/yo.py`. `все` / `всё` by the neighbouring words, and `чем` / `нем` / `всем` written `чём` / `нём` / `всём` after `о`, `об`, `обо`, `в`, `во`, `на`, `при`. |

Rebuild (no network): `python tools/build_yo_dataset.py`

## Record fields

`word` (ye spelling), `yo_form`, `stress` (0-based index of the stressed vowel; yo is always stressed), `type` (`unambiguous` or `homograph`), `source` (`eyo-kernel` or `project`). Homographs also have `variants`, each `{form, hint}`. The source lists do not give context, so those hints are empty. A project addition may set `hint`.

Unambiguous rows are safe to apply. Homograph rows are not: `берег` / `берёг` and the other homographs stay as written. Two context rules are the exception, both in `yo_context.json`.

`все` / `всё`, the most frequent one: `всё` before a singular verb or a neuter adjective, at the end of a clause, before `, что`, in `всё равно`, `всё-таки`, `всё ещё` and before a comparative; `все` before a plural word or pronoun, after a plural subject or verb, and whenever no rule applies.

`чем` / `нём` / `всём` in the prepositional case, and only there: after `о`, `об`, `обо`, `в`, `во`, `на` or `при`. `обо` is the form of `об` used before `всём` and `нём`. `на` is included because with these three pronouns it is always prepositional (the accusative is `что` / `него` / `всё`, not `чем` / `нем` / `всем`). `всем` after any other word stays, including a dative plural (`по всем`, `ко всем`) and an instrumental (`со всем`, `перед всем`). `чем` as the instrumental or a comparative (`чем мы`, `больше, чем`) stays. One word such as `зачем`, `совсем` or `причем` is not this rule (`причем` is already `причём` in the safe list).

These ye spellings are homographs, so they are not added as safe forms. Dialogue 4 therefore restores 15 of its 19 yo; the four below stay as written:

| Ye spelling | Yo reading | Why it stays |
|---|---|---|
| `узнает`, and `узнаешь`, `узнаем`, `узнаете`, `узнается` | `узнаёт` (present of `узнавать`) | The same letters are the future of perfective `узнать` (`Он узнает завтра`). The reflexive `узнаёшься`, `узнаёмся`, `узнаётесь` are already safe. |
| `берет`, `берете` | `берёт`, `берёте` (present of `брать`) | `берет` is also the hat, `берете` its prepositional (`в берете`). The rest of the family is already safe: `берёшь`, `берём`, `берёмся`, `берётесь`, `берётся`, `берёшься`. |
| `звезды`, and `кинозвезды`, `протозвезды`, `суперзвезды` | `звёзды` (nominative plural) | The genitive singular is `звезды` (stress on the ending, no yo). The oblique plural is already safe: `звёзд`, `звёздам`, `звёздами`, `звёздах`. |

## Licence

The source lists are MIT, Copyright (c) 2026 Denis Seleznev, https://github.com/e2yo/eyo-kernel. eyo-kernel says the lists were first taken from https://github.com/rin-nas/php-yoficator, which publishes no licence file, and are maintained under MIT. The build script and `core/yo.py` are Apache-2.0.

Credit: Voxprint AI Audiobook Builder https://github.com/Mitroshenkov87/voxprint-audiobook-builder

Nothing in this folder is published to Hugging Face from the repository. The gzipped JSONL is the form to release later.
