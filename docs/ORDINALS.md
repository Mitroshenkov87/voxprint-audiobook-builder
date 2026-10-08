# Ordinal numbers in narration

Before a chunk goes to the voice model, Voxprint reads numbers as **ordinals** where the text calls for it:

| Language | Written | Spoken |
|---|---|---|
| Russian | `день 1`, `Глава 2`, `в главе 3`, `стих 16`, `псалом 22`, `Глава IV` | день первый, Глава вторая, в главе третьей, стих шестнадцатый, псалом двадцать второй, Глава четвёртая |
| Russian | `1-й день`, `3-го января`, `в 5-м томе`, `90-х годов`, `в 90-е` | первый день, третьего января, в пятом томе, девяностых годов, в девяностые |
| English | `the 1st day`, `22nd`, `Chapter IV` (heading) | the first day, twenty second, Chapter four |
| German | `das 3. Kapitel`, `am 3. Oktober`, `1. Kapitel` | das dritte Kapitel, am dritten Oktober, erstes Kapitel |

Russian ordinals take the **gender and case of the noun** before them (`глава` -> feminine, `главе` -> dative/prepositional, `дня` -> genitive). Written endings (`-й`, `-го`, `-м`, `-х` ...) give the case; for ambiguous ones (`-й`, `-м`) the next word or a preposition (`в`, `на`, `о` ...) decides. English reads "Chapter 4" as "chapter four" (the normal English reading), so only written ordinals and Roman heading numerals are changed there. German turns a number with a dot before a listed noun into an ordinal whose ending follows the article (`der/die/das` -> `-e`, `dem/den/am/im/zum` -> `-en`, none -> strong ending by gender).

Left as they are (the language normalizer then reads them as cardinals): ranges (`главы 1-3`), references (`глава 1:5`, `стих 3.16`), lists (`главы 1 и 2`), dates after `день` (`день 1 сентября`), counts with the number first (`в 3 дня`), numbers above 9999.

## Switching it off

On by default. **Settings -> Narration: pauses and speed -> "Read ordinal numbers by context"**, or for one run `voxprint narrate ... --no-ordinals` (`--ordinals` forces it on). The Russian normalizer (`ru-normalizr`) already reads some headings such as `Глава 2` as ordinals on its own; the switch controls the Voxprint step only. Changing it re-synthesizes only the chunks whose text changes.

## How it works / adding words or a language

* `core/ordinal_rules.py` - **data only**: per language the trigger nouns with their gender and the case of each form (`after`), whether such a number is read as an ordinal or a cardinal (`after_reading`), German nouns after `N.` (`before`), written endings (`suffix`). Add a word here and a test in `tests/test_ordinals.py`.
* `core/ordinals.py` - the engine (`apply_ordinals(text, language)`), German number words, the setting (`state/narration_ordinals.json`).
* `core/num_words.py` - Russian and English number words (`ru_ordinal(n, case, gender)`, `en_ordinal`).
* The step runs in `core/narration.text_steps` **before** the language normalizer (which reads the remaining digits); the per-chunk check (`core/chunk_check.py`) prepares the recognised text the same way, so "глава 2" from the recogniser matches "глава вторая".

A new language needs an entry in `RULES` and an engine function registered in `core/ordinals._ENGINES` (the Qwen3-TTS narration languages are Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish).
