# Translation before narrating

**Translation before narrating (optional).** The card *Translate the book* in the Narrate window translates a book sentence by sentence, offline, into English, Russian or German before it is read (`core/translate.py`):
the source language is detected, the structure (chapters, titles, paragraphs, verse lines, scene breaks, therefore the pauses) is kept, a GPU is used if there is one (CPU otherwise). The models are the open **Opus-MT** (Marian)
models `Helsinki-NLP/opus-mt-ru-en`, `-en-ru`, `-de-en`, `-en-de` (CC-BY-4.0 / Apache-2.0, about 300 MB each, a pinned commit and a SHA-256 check of the weights after the download); ru<->de goes through English.
Every translated sentence is cached on disk, and the result is saved next to the audiobook as `translation_xx.txt` (folder `Title (xx)`) - you can read and edit it, and the next run of the same job narrates your edited text.
**Machine translation quality varies** (names, idioms, tenses, short titles and poems suffer), and **you are responsible for having the right to translate and narrate a text** (copyright and licences; this is not legal advice).
NLLB (CC-BY-NC) is deliberately not used; MADLAD-400 (Apache-2.0, large) is a possible later option.

The Opus-MT models also have backup mirrors on Hugging Face, see [MODELS.md](MODELS.md#reusing-what-is-already-on-your-computer).
