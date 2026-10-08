# Re-voice

**Re-voice** (a card in the Studio) takes one recording, or an audio file you already have, and a voice from your library.

1. **Audio** - *Record* uses the default microphone (Qt Multimedia; *Stop* adds the recording, *Play* listens). *Choose audio file* opens WAV, MP3, M4A, M4B, FLAC, Ogg/Opus or AAC. The recording is stored as Ogg Opus (48 kbit/s mono) in `<projects folder>\Re-voice`. Qt often cannot encode Opus, so the file is FLAC first and ffmpeg converts it; the models only see PCM in memory. *Play* after a direct conversion plays the converted file.
2. **Voice** - a voice you already trained. Both actions use it. The narrator is opened with that voice already selected.
3. Two actions, side by side:
   * **Convert to text** - the installed Qwen3-ASR model (0.6B or 1.7B, whichever is already there; this screen does not download it) recognises the speech. The text is editable. Lines starting with `#` are chapter titles. *Voice this text* saves a `.txt` and opens **Narrate a book**. Several files, under *Advanced*, become chapters in name order.
   * **Re-voice recording** - direct conversion of the selected file into the chosen voice. There is no text step. Timing and intonation stay with the speaker; the timbre comes from the voice's reference clip (`preview.wav`, or `ref_sample.wav`).

Only re-voice speech you have the right to use. A library voice still needs the owner's consent (`voice.json`).

## The direct model

| Candidate | Licence (code / weights) | Decision |
|---|---|---|
| **OpenVoice V2** tone-colour converter (MyShell, `myshell-ai/OpenVoiceV2`) | **MIT / MIT** | **Used.** Commercial use is allowed. Zero-shot: the target timbre is an embedding of the reference clip, not a speaker id. Conversion keeps the source spectrogram, so timing and intonation stay. |
| Seed-VC (`Plachtaa/seed-vc`) | GPL-3.0 (the model card was MIT, then changed to GPL-3.0) | Not used. Copyleft does not meet the MIT / Apache-2.0 / BSD bar, and the repository is archived. |
| MeanVC | Apache-2.0 | Not used. Extra checkpoints (WavLM, an ASR model, a vocoder) and it rebuilds speech from content tokens rather than keeping the source timeline. |
| EZ-VC | MIT / CC-BY-NC | Not used. Non-commercial weights. |
| RVC | MIT / per-voice models | Not used. Needs a model trained for each voice. |

OpenVoice V2 is pinned in `infra/model_mirrors.json` (hashes only, no backup mirror):

* repository `myshell-ai/OpenVoiceV2`, revision `f36e7edfe1684461a8343844af60babc2efbb727`
* `converter/config.json` - 838 bytes, SHA-256 `9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47`
* `converter/checkpoint.pth` - 131,320,490 bytes, SHA-256 `9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e`

The inference code vendored under `third_party/openvoice/` is the upstream MIT modules at commit `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` (`models`, `modules`, `attentions`, `commons`, `mel_processing`, `transforms`). The text-to-speech frontend, the watermark and the base-speaker checkpoints are not included. The watermark is not applied.

Since build 666 it is part of the standard first-run download (small optional models up to ~300 MB are, decided 2026-10-08), and of the Full / Quick setup's complete download. It is not in Check & repair or the portable "all models" folder. The direct button is never grey because of it: if the model is still missing, a click downloads it first and then converts; *Download model* only downloads. The files go to `<models folder>/openvoice-v2` (not `myshell-ai--OpenVoiceV2`, so a repair pass does not try to fetch them through the generic model downloader). The button shows the size from the manifest. CUDA is used when PyTorch sees an NVIDIA GPU; otherwise the same checkpoint runs on the CPU.

## A later model

`core.voice_convert.VoiceConverter` is the only contract: `convert(source, source_sr, reference, reference_sr)` returns mono float32 of the same duration, and `unload()` frees it. `make_converter` returns OpenVoice V2 (`core.vc_openvoice.OpenVoiceConverter`, key `openvoice-v2`). A newer converter is another class with those methods, returned from `make_converter`. Callers (`convert_file`, the Re-voice window) do not import a particular model.
