# Translation before narrating

**Translation before narrating (optional).** The card *Translate the book* in the Narrate window translates a book sentence by sentence, offline, into English, Russian or German before it is read (`core/translate.py`):
the source language is detected, the structure (chapters, titles, paragraphs, verse lines, scene breaks, therefore the pauses) is kept, a GPU is used if there is one (CPU otherwise). The models are the open **Opus-MT** (Marian)
models of Helsinki-NLP: the newer, larger **tc-big** models `opus-mt-tc-big-en-zle` (en->ru), `-zle-en` (ru->en), `-zle-de` (ru->de)
and `-de-zle` (de->ru), about 480 MB each, CC-BY-4.0, so **ru<->de is translated directly**; en<->de uses `opus-mt-de-en` / `-en-de` (about 300 MB
each; there is no tc-big model for that pair). Every file has a pinned commit and a SHA-256 check after the download. They are part of the
first-run download of all models (about 2.5 GB together). An install that already has the older `opus-mt-ru-en` / `-en-ru` keeps using them until the
tc-big model of that direction is downloaded.
Every translated sentence is cached on disk, and the result is saved next to the audiobook as `translation_xx.txt` (folder `Title (xx)`) - you can read and edit it, and the next run of the same job narrates your edited text.
**Machine translation quality varies** (names, idioms, tenses, short titles and poems suffer), and **you are responsible for having the right to translate and narrate a text** (copyright and licences; this is not legal advice).
NLLB (CC-BY-NC) is deliberately not used; MADLAD-400 (Apache-2.0, large) is a possible later option.

The 2020 Opus-MT models also have backup mirrors on Hugging Face (the tc-big models come from Helsinki-NLP only), see [MODELS.md](MODELS.md#reusing-what-is-already-on-your-computer).
