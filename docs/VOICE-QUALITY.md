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
  sample and of the voice's reference (its averaged speaker embedding when it has one, see 4). No extra download. The
  encoder also conditions the voice, so it "judges itself": fine for ranking candidates, not an independent verdict. An
  independent encoder (SpeechBrain ECAPA, Apache-2.0, ~80 MB, new dependency) was not added.
* **MOS**: DNSMOS P.835 OVRL (`core/mos.py`, reference windowing and polynomial mapping of `dnsmos_local.py`). 1.16 MB ONNX,
  CPU, one thread, at most 12 windows of 9.01 s per sample. Trained on human (noisy / enhanced) speech, not TTS. NISQA was not
  used (licence).
* **CER** next to the WER (letters and digits only).
* Both SIM and MOS can only raise a **warning** (`SIM_WARN` = 0.65, `MOS_WARN` = 2.6).

GPU validation: record SIM and MOS for 3-4 good and 2-3 known-bad voices (babbling, wrong pitch, noisy); set the thresholds
between the groups; check that MOS does not punish a naturally breathy voice.

## 4. Averaged speaker embedding (centroid)

`core/speaker_centroid.py`. Instead of the x-vector of the single reference clip, the speaker encoder embeds up to 64 clean
training clips (3-15 s, no clipping, RMS above -40 dBFS, highest estimated SNR first; at least 4, else the old path). Each
embedding is normalised, the directions are averaged, and the result gets the median length back (a plain mean would shrink).
Stored as `speaker_centroid.safetensors` next to the adapter and used by:

* training (the speaker conditioning of every teacher-forced sample),
* narration (replaces `ref_spk_embedding` of the voice-clone prompt; the ICL reference codes and text stay; cache key marker),
* the universal model (its `codec_embedding` speaker row),
* the voice check (similarity reference).

Older voices have no file and keep the single-clip path. `VOXPRINT_SPEAKER_CENTROID=0` switches it off everywhere;
`TrainPlan.speaker_centroid=False` for one training. Source: Baseten's Qwen3-TTS voice-cloning notes (single clips agree at
~0.7 cosine, a centroid of 30-64 clips at 0.85+).

GPU validation: train one voice twice (centroid on / off, same seed), compare similarity and steadiness across 3-4 chapters;
check that a centroid voice narrates with the ICL prompt without artefacts (the x-vector no longer matches the ICL clip exactly).

## 2. Automatic checkpoint and strength pick

`core/checkpoint_pick.py`, Train window check box *Pick the best checkpoint and voice strength automatically* (on by default;
`TaskRequest.auto_pick`).

1. Training keeps ~5 % of the clips out (`TrainPlan.holdout_fraction`, at least 2, at most 16, evenly spread; datasets under 20
   clips keep everything) and records a validation loss per epoch (`checkpoints/losses.json` `epoch_val_loss`); the held-out
   texts go to `checkpoints/holdout.json`.
2. After training, the last 3 epoch checkpoints x strengths 0.35 / 0.50 / 1.00 (only the user's strength when one was chosen in
   the preview) synthesize up to 4 held-out sentences (short to long; the preview sentence when there is no holdout) on one
   unmerged engine (`switch_adapter`, `set_adapter_scale`).
3. Score = 0.5 x (1 - CER) + 0.3 x similarity + 0.2 x (MOS - 1) / 4 (weights re-normalised over the available metrics),
   minus 0.5 x the share of samples that never stopped. Ties: strength nearer 0.50, then the later epoch.
4. The winner's adapter replaces the final one, `training_meta.json` `epochs` becomes the picked epoch, the strength goes to
   `voice.json`, every score to `checkpoints/pick.json`, and the other epoch folders are deleted.

A failed pick only adds a warning; the voice keeps the last epoch. The validation loss is recorded but not used for the choice
(the loss is a weak quality signal); it is there to compare with the pick on the GPU.

GPU validation: time per candidate (estimate 20-40 s on an RTX 4090) and total; whether the pick agrees with listening on 3-4
voices; whether the weights or the 4-phrase set need changing; whether the validation loss tracks the pick.
