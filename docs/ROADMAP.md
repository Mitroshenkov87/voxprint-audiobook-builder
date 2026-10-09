# Roadmap

Ideas, not promises:
* **Narrator improvements** - a queue of several books, more book formats (e.g. PDF, DOCX).
* **Preparation, next steps** - punctuation restoration for Russian (RUPunct), stress marks and `ё`, English / German neural spelling, and tuning of the abbreviation / unit tables on real books (the rule steps themselves are done).
* **Translation** - translate a text and speak it in your own voice in another language (a registry entry and a greyed-out placeholder exist, not implemented).
* **Multi-voice markup** - the Narrate window and the command line can mark each paragraph as narrator, male or female (the text model), the user can edit the marks, and narration speaks those voices one after another. Still open: keeping more than one voice model loaded at once.
* **Quantization, side branch only** - compute-bound model variants, not for the models this app uses now. Qwen3-TTS 1.7B is already the largest model in use; quantizing it would only shrink it.
* **README line, later** - when the public README is rewritten, the product line to use is: the only open-source full-cycle combine (voice training + book narration + re-voicing, local, no cloud, no subscription).
* A real **voices repository** (the index URL in the program is a placeholder today).
* **Android** - later: convert the merged model for phones (GGUF / LiteRT; today the merged folder is a plain Hugging Face directory a converter can start from).
* **Installer channels** - *online* (small installer, downloads components), *offline* (everything included), and *beta* (pre-releases).
* A public repository URL (the source-code licence is chosen: Apache-2.0).
* More languages in the UI (see "Adding a language" in [BUILDING.md](BUILDING.md#adding-a-language)) and a macOS build (Linux: experimental, see [Linux](LINUX.md)).
