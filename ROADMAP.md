# Roadmap

Plans for Voxprint AI Audiobook Builder. Nothing here is a promise, and there are no dates. What the app does today is in the [user guide](docs/USER-GUIDE.md) and the [command reference](docs/CLI.md).

## Next version

* **Command line for other programs.** The main jobs already print JSON, and `--dry-run` checks the inputs without loading a model, using the graphics card, or downloading. The next pass keeps that contract complete, so another program can prepare a book, mark speakers, narrate, train a voice, and check the install without opening the window.
* **Icons.** Save, export, and download already use the shared stroke icons. The next pass adds the remaining action icons and a file icon for `.vxbook`.
* **Russian yo, stress, and speaker marks.** Yo is filled in where a dictionary is sure, and a stress mark the author wrote is kept. A few yo words still depend on the sentence around them, and the speech model does not use stress marks yet. Speaker marks (narrator, male, female) still miss some paragraphs. The next pass makes yo, stress, and those marks more accurate.
* **A smarter updater.** An update is checked, swapped in, and rolled back when the new build does not start. The next updater should fetch only the pieces that changed, check their hashes, and leave the rest of the install where it is.

## Later

* **Offline install bundle.** A folder with the program, the components, and the models, plus a manifest of hashes, that can be copied to a computer with no network. The online installer uses that folder when it finds it, and downloads only a file that is missing or does not match its hash.
* **Hebrew in the interface, including right-to-left layout.**
* **Voice enhancement.** After training, an optional listen to a few short phrases, compared with the recording, with another pass when the new voice does not match.
* **Linux polish.** The Linux build is experimental. A later pass is a laptop-tested install with the same checks as Windows.
* **Code signing** for the Windows installer, so a fresh download is less likely to be blocked.
* **More voices.** More open voices in the download catalog, each with its own licence. A voice is included only when the speaker agreed, or the recording is in the public domain.

## Models & features

Narration stays on Qwen3-TTS, with a LoRA adapter for each voice. Speaker marks, literary translation, and the optional text rewrite use Gemma 4 12B in GGUF form, through llama.cpp. Offline sentence translation uses Opus-MT. Speech recognition uses Qwen3-ASR. The optional soundscape uses ACE-Step, downloaded only when you turn it on, and only for a `.vxbook` that asks for sound. A plain book is never given music.

Later model work is accuracy and fit: better yo, stress, and speaker marks; a soundscape that stays out of the way of the voice; and the memory placement below. A queue of several books is also later. Multi-voice narration already follows speaker marks. Holding more than one voice model in video memory at the same time is still open.

## Smart memory placement

This is a plan.

Video memory, system memory, and disk are three tiers. Video memory stays for the work that needs the graphics card. The rules already in the app stay in force: a reserve of the larger of 2 GB and 8 percent of the card, batches taken from the memory that is free right now, an optional cap of 70 to 80 percent of the card (off unless you set it), and the thermal pause (after two and a half hours, a five-minute median at or above 83 °C inserts a two-and-a-half-minute pause before the next batch). The placement below has to live inside those rules.

* **Qwen3-TTS.** Load the speech model in stages, one component after another, instead of holding every piece on the card at once. When video memory is short, keep the codec and the tokenizer in system memory and run them on the CPU. For a long chunk, keep the key-value cache in system memory.
* **Gemma 4 12B GGUF.** This is the text model for speaker marks and the optional rewrite. Offload layers according to how much video memory is free. If the file in use is a mixture-of-experts build, offload experts that are not needed to system memory.
* **Speech recognition and translation.** Keep those encoders in system memory and run them on the CPU, so they are not sitting on the card next to the voice.
* **Soundscape.** Load the soundscape model only when a book asks for one, and unload it before narration starts, so the voice model has the card.
* **Prefetch.** While one chunk is being spoken, read the next chunk, or the next model, from disk into system memory.

The aim is to keep the models this app already uses on a 16 GB or 24 GB card, with ordinary system memory taking the pieces that do not need the card, and without dropping the quality of the voice or turning off the thermal pause.
