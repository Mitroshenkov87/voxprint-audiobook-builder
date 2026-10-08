# Re-voice

**Re-voice** (fourth card in the Studio) turns speech into an audiobook read by one of your voices:

1. **Audio** - press *Record* to record from the default microphone (Qt Multimedia; *Stop* adds the recording to the list,
   *Play* listens to the selected file), or *Add audio files* (MP3, WAV, M4B, M4A, FLAC, OGG/Opus, AAC). Several files become
   chapters, in name order; a leading track number (`01 - `) is dropped from the chapter title. Recordings are saved in
   `Documents\Voxprint\Re-voice`.
2. **Text** - *Recognise speech* runs the installed Qwen3-ASR model (0.6B or 1.7B, whichever the model download put there;
   nothing extra is downloaded) on pieces of up to 25 s cut at pauses. The text appears in an editor: fix names and mistakes,
   lines starting with `#` are chapter titles.
3. *Narrate this text* saves the text as `<Title>.txt` in the same folder and opens it in **Narrate a book**, where you pick the
   voice, format, translation and so on as for any book.

The recogniser is unloaded before the narrator starts. Only re-voice speech you have the right to use.
