# How it works

```
 audio + text
     │  read + normalize (numbers/abbreviations spelled out, UTF-8/cp1251)
     ▼
 forced alignment ── Qwen3-ForcedAligner (long audio: cut at pauses, chunks ≤ 150 s, text split by clauses)
     │  word timestamps
     ▼
 slicing ── 3-12 s clips cut at pauses, never mid-word, + ~1 s trailing silence
     ▼
 quality filter ── drops clipped / silent / noisy clips (adaptive if too many are rejected)
     ▼
 dataset (Alexandria format) ──► LoRA training on the Qwen3-TTS talker ──► adapter + voice.json
                                                                              │ optional
                                                                              ▼
                                                                   merged "universal" model (~4 GB)
```
Narration: `book → chapters → chunks → (cache hit or Qwen3-TTS + adapter) → WAV with pauses per chapter → ffmpeg → .opus / .mp3 / .m4b ...`
(`core/book_parsers.py`, `core/chunker.py`, `core/narration.py`, `core/tts_engine.py`, `core/audiobook_export.py`).
Heavy work runs in background threads (`workers/`); the UI only reacts to progress signals. Before anything else a quiet weekly update
check runs (it never changes components in *your* environment without asking). See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for the module map and data flow.

Hyper-parameters are chosen automatically (Alexandria `lora.md`): LoRA r=32, alpha=128 on q/k/v/o_proj of the **talker**, batch 1 with
accumulation 4-8, lr 1e-6 (< 90 fragments) or 2e-6, epochs ~ 320 / fragments, eager attention, bf16, gradient checkpointing; a warning is
shown if the final loss is < 3.5 (the threshold is not confirmed for Russian).

## Output format
Alexandria `train_lora.py` contract:
```
dataset\  metadata.jsonl  {"audio":"segment_001.wav","text":"...","ref_audio":"ref.wav"}  (UTF-8, relative paths)
          segment_NNN.wav (24 kHz mono, 3-12 s of speech + ~1 s trailing silence, cuts at pauses, never mid-word)
          ref.wav (24 kHz, 5-10 s, the cleanest fragment)   ref_text.txt (exact transcript of ref.wav)
          report.json (incl. text_raw - the original text before normalization, quality drops, training language)
output\<voice name>\  adapter_model.safetensors, adapter_config.json, ref_sample.wav, training_meta.json, voice.json
          checkpoints\epoch_NN\ (adapter copy after every epoch; may be deleted)
          merged_model\ (only after "Build universal model": model.safetensors bf16 ~4 GB, config.json with
          tts_model_type=custom_voice + talker_config.spk_id, speech_tokenizer\, ref_sample.wav, ref_text.txt,
          speaker_embedding.safetensors, voxprint_voice.json, USAGE.txt)
```
`voice.json` (schema 2) example:
```json
{"schema": 2, "id": "my-voice", "name": "My voice", "language": "russian", "created": "2026-10-03T12:00:00Z",
 "duration": 412.7, "epochs": 15, "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "author": "Aleksandr",
 "license": "custom/personal-only", "license_url": "", "voice_type": "female",
 "description": "Warm narrator voice, calm pace", "commercial_use": false}
```
`voice_type` (`male|female|child|other`) and `description` (whitespace collapsed, ≤ 500 characters) are optional (empty strings when unset). `license` is an SPDX-like id (see the table above; default
`custom/personal-only`), `license_url` is filled for the known licences, and `commercial_use` is **derived** from the licence. Schema-1 files (`voice_name`, `speech_seconds`) are migrated when read.
Library layout: `%LOCALAPPDATA%\Voxprint\voices\<id>\` = `adapter_model.safetensors`, `adapter_config.json`, `ref_sample.wav`, `training_meta.json`, `voice.json`.

Audiobook output: `<output>\<Book title>\` with the chosen files (`Author - Title.opus` / `.m4b` / `.mp3`, or folders `... - MP3`, `... - Opus`, `... - FLAC`, `... - WAV` with `NN - Chapter` files and a `.m3u8`);
the resumable chunk cache lives in `.cache\` and temporary files in `.work\` inside that folder and are removed after success.

Using the merged model: `Qwen3TTSModel.from_pretrained(folder).generate_custom_voice(text, language="Russian", speaker="<voice name>")`.

## Narration speed

Narration used one chunk at a time with the "eager" attention code (RTF 3.05 on an RTX 4090: 3 s of GPU time per second of audio). It now defaults to:
* **SDPA attention** (PyTorch fused kernels; FlashAttention 2 is used instead if the `flash_attn` package is installed - it is not available for Windows from PyPI; "eager" remains the fallback if the model refuses to load);
* **batched generation**: several chunks go through the model in one call (`Qwen3AdapterEngine.synthesize_batch`), the batch size comes from the free VRAM (up to 12), chunks of similar length are batched together (sorted inside a small window, so the book order is kept for the live player),
  and an out-of-memory error halves the batch automatically; any other batch error falls back to chunk-by-chunk synthesis (so batching can never make a book fail);
* **a writer thread**: finished chunks are encoded to FLAC and stored by a helper thread while the GPU already generates the next batch.
The model still runs only on the GPU (bfloat16, the dtype it was trained and validated in).

Measured (RTX 4090, 12 English chunks of 20-170 characters, 80 s of audio, open voice, same seed; WER = the audio recognised back by Qwen3-ASR; "pitch" = mean f0 shift against the reference clip):

| Setup | RTF | time for 80 s of audio | VRAM peak | mean WER | pitch shift |
|---|---|---|---|---|---|
| before: eager, one chunk at a time | 3.05 | 243 s | 5.95 GB | 0.014 | +0.9 st |
| SDPA, one chunk at a time | 2.09 | 166 s | 5.95 GB | 0.035 | +0.7 st |
| SDPA, batch of 4 | 0.72 | 57 s | 7.6 GB | 0.019 | +0.4 st |
| SDPA, batch of 6 | 0.51 | 41 s | 8.7 GB | 0.014 | -0.1 st |
| SDPA, batch of 12 (default on 24 GB) | **0.31** | **25 s** | 11.8 GB | 0.004 | -0.3 st |

That is about **10x faster** on this GPU. Estimates for a book at RTF 0.31-0.5 (one narrated hour of audio = 60 min x RTF of GPU time): **5 hours of audio ≈ 1.6-2.5 h, 20 hours ≈ 6-10 h** (before: 15 h and 61 h). Longer chunks use more VRAM per item, so the real batch is smaller on 12-16 GB cards.
Caveats: the WER differences between rows are within the sampling noise of 12 chunks (generation is random); speaker similarity was checked only through the pitch shift (the project has no speaker-embedding model); a small pitch drift (about 1 semitone lower than the single-chunk run) is visible with large batches - use the single-chunk
path (`MAX_BATCH = 1` in `core/tts_engine.py`) if you prefer. `torch.compile` / CUDA graphs were not tried (no Triton on Windows).
