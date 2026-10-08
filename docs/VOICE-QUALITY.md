# Voice quality: mechanisms and what still needs a GPU run

Voxprint's voice-quality batch 1 (October 2026). The mechanisms below were built and tested on a CPU-only machine with fakes
and tiny randomly initialised models; the **defaults are research-based and must be validated on a real GPU** with real voices.
Every new behaviour can be switched off.

## 1. Voice strength (adapter scale at inference)

`core/adapter_strength.py`. The LoRA delta is multiplied by `adapter_scale` (peft `LoraLayer.set_scale`, applied before the merge).

| Where | Value |
| --- | --- |
| New voice, nothing chosen | `DEFAULT_SCALE` = 0.50 |
| Voice without the field (trained before) | 1.00 (unchanged sound) |
| Quick preview candidates | 0.35, 0.50, 1.00 |
| Storage | `voice.json` `adapter_scale`; editable in Properties |

Sources: Instavar's Qwen3-TTS LoRA write-up (best 0.3-0.35), QwenLM/Qwen3-TTS issue #343 (0.3-0.4). Voxprint adapters use
alpha / r = 4 (r 32, alpha 128), twice the ratio of those experiments, so the effective strengths are not directly comparable.

GPU validation: listen to 3-4 voices at 0.35 / 0.5 / 0.7 / 1.0 (preview selector or Properties); check stop-token reliability
(no babbling) and similarity; adjust `DEFAULT_SCALE` and `PREVIEW_SCALES`; measure `EXTRA_SCALE_SEC` (preview time estimate per
extra strength, guessed at 9 s).

## 3. Speaker similarity and MOS in the automatic check

* **Similarity (SIM)**: cosine of the Qwen3-TTS speaker-encoder embeddings (`Qwen3AdapterEngine.speaker_embedding`) of the
  sample and of the voice's reference clip. No extra download. The
  encoder also conditions the voice, so it "judges itself": fine for ranking candidates, not an independent verdict. An
  independent encoder (SpeechBrain ECAPA, Apache-2.0, ~80 MB, new dependency) was not added.
* **MOS**: DNSMOS P.835 OVRL (`core/mos.py`, reference windowing and polynomial mapping of `dnsmos_local.py`). 1.16 MB ONNX,
  CPU, one thread, at most 12 windows of 9.01 s per sample. Trained on human (noisy / enhanced) speech, not TTS. NISQA was not
  used (licence).
* **CER** next to the WER (letters and digits only).
* Both SIM and MOS can only raise a **warning** (`SIM_WARN` = 0.65, `MOS_WARN` = 2.6).

GPU validation: record SIM and MOS for 3-4 good and 2-3 known-bad voices (babbling, wrong pitch, noisy); set the thresholds
between the groups; check that MOS does not punish a naturally breathy voice.
