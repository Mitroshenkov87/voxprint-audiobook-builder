# Translation before narrating

**Translation before narrating (optional).** The card *Translate the book* in the Narrate window translates a book sentence by sentence, offline, into English, Russian or German before it is read (`core/translate.py`):
the source language is detected, the structure (chapters, titles, paragraphs, verse lines, scene breaks, therefore the pauses) is kept, a GPU is used if there is one (CPU otherwise). The models are the open **Opus-MT** (Marian)
models of Helsinki-NLP: the **tc-big** models `opus-mt-tc-big-en-zle` (en->ru), `-zle-en` (ru->en), `-zle-de` (ru->de)
and `-de-zle` (de->ru), about 480 MB each, CC-BY-4.0, so **ru<->de is translated directly**. en<->de uses the **tc-bible-big** pair Movie Dubber
uses, Apache-2.0, about 900 MB each: `opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw` (en->de, sentence-initial `>>deu<<`) and
`opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa` (de->en, `>>eng<<`). The download is that Hugging Face safetensors snapshot, so the shared models
folder holds one copy of each. Every file has a pinned commit and a SHA-256 check after the download. The six models are part of the
first-run download (about 3.7 GB together). An install that already has the older `opus-mt-ru-en` / `-en-ru` keeps using them until the
tc-big model of that direction is downloaded. The 2020 `opus-mt-en-de` / `opus-mt-de-en` models are not used.
Every translated sentence is cached on disk, and the result is saved next to the audiobook as `translation_xx.txt` (folder `Title (xx)`) - you can read and edit it, and the next run of the same job narrates your edited text.
**Machine translation quality varies** (names, idioms, tenses, short titles and poems suffer), and **you are responsible for having the right to translate and narrate a text** (copyright and licences; this is not legal advice).
NLLB (CC-BY-NC) is deliberately not used; MADLAD-400 (Apache-2.0, large) is a possible later option.

The 2020 Russian Opus-MT models also have backup mirrors on Hugging Face (the tc-big and tc-bible-big models come from Helsinki-NLP only), see [MODELS.md](MODELS.md#reusing-what-is-already-on-your-computer).

## Literary translation and "Prepare text for narration" (optional AI model)

The card *AI text model* in the Narrate window offers two options, both **off by default** and greyed out until the model is
downloaded (a separate download of about 7.2 GB, never part of "download all"; it needs an NVIDIA GPU with about 10 GB of video
memory):

* **Literary translation** - whole paragraphs (1-4 per request, with the two previous paragraphs as context) are translated by
  **Gemma 4 12B Instruct** (Apache-2.0; GGUF Q4_K_M by Unsloth, pinned revision + SHA-256) instead of sentence by sentence.
* **Prepare text for narration** - the same model rewrites the text in its own language for listening: numbers, dates,
  abbreviations as words, no brackets or footnote marks, obvious typos fixed, nothing added or left out.

The model runs in the pre-built `llama-server` of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, Vulkan build, ~33 MB; on laptops with two GPUs the NVIDIA one is chosen from `--list-devices`, logged)
as a separate process before the narration and is closed - all of its video memory freed - before the voice model loads.
The GGUF and that one llama.cpp build live in the shared models folder (`<models>/llm/gemma-4-12b-it-Q4_K_M.gguf` and `<models>/llm/llama.cpp-<build>/`), the same folder `state/suite.json` gives every Voxprint program, so Plotweaver can reuse the file.
Every answer passes a simple guard (same number of paragraphs and verse lines, a sane length ratio, the right language, no
refusal); a paragraph that fails, or everything after the model fails to start or crashes, is translated by Opus-MT instead
(or keeps its original text when preparing). Names are kept consistent through a glossary (`.translation/names_xx.txt` in the
job folder, made by one request at the start; edit it and the next run uses your version). The model's log is
`logs/llama-server.log`.

The instructions for the model are plain text files (`prompts/literary_translate.txt`, `prompts/prepare_narration.txt`,
`prompts/names_glossary.txt`). *Edit prompts* copies them to `prompts/` in the Voxprint data folder and opens it; edited copies
are used instead of the built-in ones (delete a file to get ours back). Changing a prompt translates again; the result is cached
per paragraph, so a resumed job does not ask the model twice. Literary quality is not measured yet: compare a few chapters
yourself before relying on it.
