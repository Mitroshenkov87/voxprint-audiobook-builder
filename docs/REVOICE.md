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

## Direct voice conversion (phase 2): not included

Converting the recording straight into another voice (keeping your timing and intonation) was considered and skipped for now:

| Candidate | Licence (code / weights) | Why not now |
|---|---|---|
| OpenVoice V2 tone-colour converter (MyShell) | MIT / MIT | The only clean licence. It would need its model code vendored (~1,500 lines) plus a second audio pipeline; it changes only the timbre (accent and prosody stay the speaker's), so the result is expected to sound less like the trained voice than the text route above. Candidate if direct conversion is wanted later. |
| Seed-VC | GPL-3.0 | Copyleft code, repository archived. |
| EZ-VC | MIT / CC-BY-NC | Non-commercial weights. |
| RVC | MIT / per-voice models | Needs a separately trained model per voice. |
