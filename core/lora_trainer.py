"""Training of the voice LoRA adapter for Qwen3-TTS-12Hz-Base - the format and recipe of Alexandria (``train_lora.py``).

Sources of the contract (read from the code of ``Finrandojin/alexandria-audiobook``):

* ``train_lora.py`` - dataset, teacher forcing, LoRA on the talker, output format;
* ``app/tts.py`` - the consumer of the adapter: ``PeftModel.from_pretrained(model.model.talker, adapter_path)``,
  ``ref_sample.wav`` + ``ref_sample_text`` from ``training_meta.json``;
* ``lora.md`` - hyper-parameters (lr 1e-6..2e-6, samples x epochs ~250-400, warning when loss < ~3.5).

What this module does:

1. reads ``metadata.jsonl`` (``{"audio" | "audio_filepath", "text", "ref_audio"}``, paths relative to the dataset
   folder), ``ref.wav`` and ``ref_text.txt``;
2. encodes the audio with the speech tokenizer of the Base model itself (it ships inside the model repo) -> codes ``[T, 16]``;
3. applies peft to ``hf_model.talker`` (*not* to the whole model): the adapter keys are relative to the talker, as the
   consumer expects; ``target_modules=[q_proj,k_proj,v_proj,o_proj]``, r=32, alpha=128, dropout 0.05;
4. every sample is a teacher-forcing step (``core/teacher_forcing.py``), batch 1 with gradient accumulation,
   loss = CE(first codec group) + 0.3 x the sub-talker loss; ``attn_implementation="eager"``;
5. the result is a *folder*: ``adapter_model.safetensors``, ``adapter_config.json``, ``ref_sample.wav``,
   ``training_meta.json``; after each epoch a copy of the adapter goes to ``checkpoints/epoch_NN/``.

Memory use is planned by ``infra/vram_optimizer.plan_training``; on an out-of-memory error the plan is reduced and the
run repeated (:func:`train_lora_from_dataset`).

TODO-needs-GPU-test (originally unverified, later exercised on an RTX 4090): real 1.7B-Base weights, bf16 + eager +
gradient checkpointing on Windows, the 8-bit AdamW (bitsandbytes), real VRAM use, whether the loss threshold of 3.5 suits
Russian, and loading the adapter in Alexandria with a newer peft (Alexandria pins ``peft==0.18.1``).
"""
from __future__ import annotations

from core.i18n import tr
import gc
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core import audio_utils as au
from core import speaker_centroid
from core.dataset_builder import TRAIN_SR, read_metadata_jsonl
from core.errors import DatasetMakerError, OutOfMemoryError_, TrainingError
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.teacher_forcing import build_assistant_text, build_teacher_forcing_input
from infra import model_downloader as md
from infra.vram_optimizer import (LOSS_WARN_BELOW, TrainPlan, VramMonitor, detect_gpu, plan_training,
                                  reduce_after_oom)

log = logging.getLogger("voxprint.lora")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # as in train_lora.py (suffix match)
LORA_DROPOUT = 0.05
SUB_TALKER_WEIGHT = 0.3
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "ref_sample.wav", "training_meta.json")
HOLDOUT_MIN_ROWS = 20      # smaller datasets keep every clip for training (no holdout)
HOLDOUT_MAX = 16           # validation clips at most (cost of the per-epoch validation loss)
HOLDOUT_FILE = "holdout.json"    # in checkpoints/: the held-out clips (texts for the automatic checkpoint pick)


# --------------------------------------------------------------------------- data


def load_training_rows(dataset_dir: Path) -> Dict[str, Any]:
    """Read the dataset the same way ``train_lora.py`` does.

    Returns ``{"rows", "ref_audio", "ref_text"}`` with absolute paths.  The reference is the first row's ``ref_audio``,
    else ``ref.wav``, else the first clip; a missing ``ref_text.txt`` falls back to the first sample's text (as Alexandria does).
    """
    dataset_dir = Path(dataset_dir)
    meta = dataset_dir / "metadata.jsonl"
    if not meta.exists():
        raise TrainingError(tr("err.no_metadata"))
    entries = read_metadata_jsonl(meta)
    if not entries:
        raise TrainingError(tr("err.dataset_empty"))
    rows = []
    for e in entries:
        rel = e.get("audio_filepath") or e.get("audio", "")
        rows.append({"audio": str(dataset_dir / rel), "text": e["text"], "rel": rel})
    for r in rows:
        if not Path(r["audio"]).exists():
            raise TrainingError(tr("err.dataset_file_missing", file=r["rel"]))
    if entries[0].get("ref_audio"):
        ref = dataset_dir / entries[0]["ref_audio"]
    elif (dataset_dir / "ref.wav").exists():
        ref = dataset_dir / "ref.wav"
    else:
        ref = Path(rows[0]["audio"])
    if not ref.exists():
        raise TrainingError(tr("err.no_ref"))
    rt = dataset_dir / "ref_text.txt"
    ref_text = rt.read_text(encoding="utf-8").strip() if rt.exists() else ""
    if not ref_text:
        log.warning("ref_text.txt missing - using the first sample text (as train_lora.py does)")
        ref_text = rows[0]["text"]
    return {"rows": rows, "ref_audio": str(ref), "ref_text": ref_text}


def split_holdout(rows: List[Dict[str, Any]], fraction: float) -> tuple:
    """``(training rows, held-out rows)``: about ``fraction`` of the clips, evenly spread over the recording (deterministic),
    at least 2 and at most :data:`HOLDOUT_MAX`; nothing is held out for ``fraction <= 0`` or fewer than
    :data:`HOLDOUT_MIN_ROWS` clips."""
    n = len(rows)
    if fraction <= 0 or n < HOLDOUT_MIN_ROWS:
        return list(rows), []
    k = min(HOLDOUT_MAX, max(2, int(round(n * fraction))))
    picks = {min(n - 1, int((i + 0.5) * n / k)) for i in range(k)}
    return [r for i, r in enumerate(rows) if i not in picks], [r for i, r in enumerate(rows) if i in picks]


def build_lora_config(r: int = 32, alpha: int = 128, dropout: float = LORA_DROPOUT):
    """peft ``LoraConfig`` for the talker's attention projections (r=32, alpha=128 by default)."""
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, target_modules=list(LORA_TARGET_MODULES),
                      lora_dropout=dropout, bias="none")


def speaker_embedding_from_ref(model: Any, ref_wav_path: str, device: Any, dtype: Any):
    """Speaker embedding of ``ref.wav`` (24 kHz): mel spectrogram + the model's speaker encoder, as in ``train_lora.py``."""
    wav, _ = au.load_audio(ref_wav_path, TRAIN_SR)
    return speaker_embedding_of(model, wav, device, dtype)


def speaker_embedding_of(model: Any, wav: Any, device: Any, dtype: Any):
    """Speaker embedding ``[1, D]`` of 24 kHz samples ``wav`` with the model's speaker encoder."""
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram  # type: ignore

    with torch.no_grad():
        mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        emb = model.speaker_encoder(mel.to(device).to(dtype))
    return emb.detach()


def speaker_conditioning(model: Any, data: Dict[str, Any], device: Any, dtype: Any, use_centroid: bool = True):
    """``(spk [1, D], centroid)``: the averaged embedding over the clean training clips when possible (``centroid`` =
    ``(vector, clip paths)``), else the embedding of the reference clip (``centroid`` = None)."""
    import torch

    if use_centroid:
        try:
            res = speaker_centroid.compute(
                lambda x, sr: speaker_embedding_of(model, x, device, dtype)[0].float().cpu().numpy(),
                [r["audio"] for r in data["rows"]], TRAIN_SR)
        except Exception as exc:  # noqa: BLE001 - the single-clip path always works
            log.warning("speaker centroid failed (%s): using the reference clip", exc)
            res = None
        if res is not None:
            log.info("speaker embedding: centroid of %d clips", len(res[1]))
            return torch.from_numpy(res[0]).to(device=device, dtype=dtype).unsqueeze(0), res
    return speaker_embedding_from_ref(model, data["ref_audio"], device, dtype), None


def prepare_samples(rows: List[Dict[str, Any]], tokenize: Callable[[str], Any],
                    encode_audio: Callable[[Any, int], Any], spk_embedding: Any, device: Any,
                    max_audio_seconds: float, progress: ProgressCallback, cancel: CancelToken,
                    warnings: List[str]) -> List[Dict[str, Any]]:
    """Audio -> codec codes, text -> tokens (assistant template). Clips longer than ``max_audio_seconds`` are skipped with a warning."""
    import torch

    samples: List[Dict[str, Any]] = []
    skipped = 0
    for i, r in enumerate(rows):
        cancel.check()
        audio, sr = au.load_audio(r["audio"], TRAIN_SR)
        dur = len(audio) / sr
        if dur > max_audio_seconds:
            skipped += 1
            continue
        with torch.no_grad():
            codes = encode_audio(audio, sr)                       # [T, G]
        ids = tokenize(build_assistant_text(r["text"]))
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        samples.append({"codec_ids": codes.to(device), "spk_embedding": spk_embedding,
                        "text_ids": ids.to(device), "text": r["text"], "duration": dur})
        progress(Stage.TRAIN, 0.03 * (i + 1) / len(rows), tr("progress.prep_codes"))
    if skipped:
        warnings.append(tr("warn.skipped_long", n=skipped))
    if not samples:
        raise TrainingError(tr("err.no_clips"))
    return samples


# --------------------------------------------------------------------------- optimizer


def check_bitsandbytes(device: str) -> bool:
    """Check that the 8-bit AdamW really works (import + one step on a tiny tensor); False off CUDA."""
    if not device.startswith("cuda"):
        return False
    try:
        import bitsandbytes as bnb  # type: ignore
        import torch

        p = torch.nn.Parameter(torch.zeros(4096, device=device))
        opt = bnb.optim.AdamW8bit([p], lr=1e-3)
        p.grad = torch.ones_like(p)
        opt.step()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("bitsandbytes is not usable (%s) - falling back to torch AdamW", exc)
        return False


def make_optimizer(params: List[Any], lr: float, use_8bit: bool, device: str, warnings: List[str]):
    """AdamW (8-bit via bitsandbytes when requested and usable, otherwise plain torch AdamW with a warning)."""
    import torch

    if use_8bit:
        if check_bitsandbytes(device):
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.01)
        warnings.append(tr("warn.adam8_unavailable"))
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


# --------------------------------------------------------------------------- training


def compute_sample_loss(sample: Dict[str, Any], hf_model: Any, base_talker: Any, device: Any, language: str):
    """Loss of one sample as ``(total, talker_loss, sub_loss)``: the first-codec-group cross-entropy plus 0.3 x the sub-talker loss (as in ``train_lora.py``)."""
    import torch.nn.functional as F

    full_input, labels, all_codec_ids, prefill_len = build_teacher_forcing_input(
        sample, hf_model, base_talker, device, language=language)
    T = all_codec_ids.shape[0]
    out = base_talker.model(inputs_embeds=full_input, use_cache=False)
    hidden = out.last_hidden_state
    logits = base_talker.codec_head(hidden)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    talker_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                  ignore_index=-100)
    audio_hidden = hidden[0, prefill_len - 1:prefill_len + T - 1, :]
    _, sub_loss = base_talker.forward_sub_talker_finetune(all_codec_ids, audio_hidden)
    return talker_loss + SUB_TALKER_WEIGHT * sub_loss, talker_loss, sub_loss


def validation_loss(val_samples: List[Dict[str, Any]], hf_model: Any, peft_talker: Any, base_talker: Any, device: Any,
                    language: str) -> float:
    """Mean loss over the held-out samples (no gradients, dropout off); the model is put back into training mode."""
    import torch

    peft_talker.eval()
    try:
        with torch.no_grad():
            vals = [float(compute_sample_loss(s, hf_model, base_talker, device, language)[0]) for s in val_samples]
    finally:
        peft_talker.train()
    return sum(vals) / max(1, len(vals))


def loss_warnings(final_loss: float, first_loss: float) -> List[str]:
    """User-facing warnings about the final loss: suspiciously low (over-fitting) or barely decreased."""
    out: List[str] = []
    if final_loss < LOSS_WARN_BELOW:
        out.append(tr("warn.loss_low", loss=f"{final_loss:.2f}", threshold=LOSS_WARN_BELOW))
    if final_loss >= first_loss * 0.98:
        out.append(tr("warn.loss_flat"))
    return out


def save_adapter_folder(peft_talker: Any, out_dir: Path, ref_audio: Optional[str] = None,
                        ref_text: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
    """``peft.save_pretrained`` into ``out_dir`` (+ ``ref_sample.wav`` and ``training_meta.json`` when given)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_talker.save_pretrained(str(out_dir), safe_serialization=True)
    if ref_audio is not None:
        shutil.copy2(ref_audio, out_dir / "ref_sample.wav")
    if meta is not None:
        (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def train_on_model(hf_model: Any, tokenize: Callable[[str], Any], encode_audio: Callable[[Any, int], Any],
                   data: Dict[str, Any], output_dir: Path, plan: TrainPlan, base_repo: str,
                   progress: ProgressCallback, cancel: CancelToken,
                   warnings_out: Optional[List[str]] = None, seed: int = 1234) -> Path:
    """Training core for an already loaded ``Qwen3TTSForConditionalGeneration`` (no network; unit-tested on the CPU).

    Freezes the model, wraps the talker with LoRA, then runs ``plan.epochs`` epochs with gradient accumulation, saving a
    checkpoint folder per epoch and the final adapter, ``training_meta.json`` and ``checkpoints/losses.json``.
    CUDA out-of-memory is converted to ``OutOfMemoryError_`` so the caller can retry with a lighter plan.
    """
    import torch
    from peft import get_peft_model

    warnings = warnings_out if warnings_out is not None else []
    warnings.extend(plan.warnings)
    output_dir = Path(output_dir)
    device = torch.device(plan.device)
    dtype = torch.bfloat16 if plan.dtype == "bfloat16" else torch.float32
    hf_model.to(device)
    spk, centroid = speaker_conditioning(hf_model, data, device, dtype, plan.speaker_centroid)
    train_rows, hold_rows = split_holdout(data["rows"], plan.holdout_fraction)
    samples = prepare_samples(train_rows, tokenize, encode_audio, spk, device, plan.max_seconds_per_item,
                              progress, cancel, warnings)
    val_samples = prepare_samples(hold_rows, tokenize, encode_audio, spk, device, plan.max_seconds_per_item,
                                  noop_progress, cancel, []) if hold_rows else []

    for p in hf_model.parameters():
        p.requires_grad_(False)
    # peft goes on the talker (like train_lora.py): the adapter keys are relative to the talker
    peft_talker = get_peft_model(hf_model.talker, build_lora_config(plan.lora_r, plan.lora_alpha))
    hf_model.talker = peft_talker
    base_talker = peft_talker.base_model.model
    try:
        peft_talker.enable_input_require_grads()
        if plan.gradient_checkpointing:
            base_talker.model.gradient_checkpointing_enable()
    except Exception as exc:  # noqa: BLE001 - without checkpointing it just uses more memory
        log.warning("gradient checkpointing unavailable: %s", exc)
    params = [p for p in peft_talker.parameters() if p.requires_grad]
    if not params:
        raise TrainingError(tr("err.no_lora_layers"))
    for p in params:  # LoRA weights in fp32 for stability (peft usually does this itself)
        p.data = p.data.float()
    log.info("LoRA trainable params: %d (r=%d alpha=%d lr=%g epochs=%d accum=%d, %d samples)",
             sum(p.numel() for p in params), plan.lora_r, plan.lora_alpha, plan.lr, plan.epochs,
             plan.grad_accum, len(samples))
    opt = make_optimizer(params, plan.lr, plan.use_8bit_adam, plan.device, warnings)

    rnd = random.Random(seed)
    oom = getattr(torch.cuda, "OutOfMemoryError", MemoryError)
    peft_talker.train()
    started = time.time()
    epoch_losses: List[float] = []
    val_losses: List[float] = []
    best = float("inf")
    n = len(samples)
    total_steps = n * plan.epochs
    done = 0
    for epoch in range(1, plan.epochs + 1):
        order = samples[:]
        rnd.shuffle(order)
        opt.zero_grad(set_to_none=True)
        acc_loss, count = 0.0, 0
        for k, sample in enumerate(order, start=1):
            cancel.check()
            try:
                total, _, _ = compute_sample_loss(sample, hf_model, base_talker, device, plan.language)
                (total / plan.grad_accum).backward()
            except oom as exc:
                raise OutOfMemoryError_(details=str(exc)) from exc
            acc_loss += float(total.detach())
            count += 1
            done += 1
            if k % plan.grad_accum == 0 or k == n:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if k % 4 == 0 or k == n:
                snap = VramMonitor.snapshot()
                vram = tr("progress.vram_part", used=f"{snap[0]:.1f}", total=f"{snap[1]:.1f}") if snap else ""
                progress(Stage.TRAIN, 0.03 + 0.97 * done / total_steps,
                         tr("progress.training", epoch=epoch, epochs=plan.epochs, k=k, n=n, vram=vram))
        avg = acc_loss / max(1, count)
        epoch_losses.append(avg)
        log.info("epoch %d/%d avg_loss=%.4f", epoch, plan.epochs, avg)
        if val_samples:
            val_losses.append(validation_loss(val_samples, hf_model, peft_talker, base_talker, device, plan.language))
            log.info("epoch %d/%d val_loss=%.4f", epoch, plan.epochs, val_losses[-1])
        best = min(best, avg)
        save_adapter_folder(peft_talker, output_dir / "checkpoints" / f"epoch_{epoch:02d}")

    progress(Stage.SAVE, 0.0, tr("progress.saving_adapter"))
    final_loss = epoch_losses[-1]
    warnings.extend(loss_warnings(final_loss, epoch_losses[0]))
    meta = {
        "model_name": base_repo, "epochs": plan.epochs, "lr": plan.lr, "lora_r": plan.lora_r,
        "lora_alpha": plan.lora_alpha, "gradient_accumulation_steps": plan.grad_accum,
        "batch_size": plan.batch_size, "num_samples": n, "final_loss": final_loss, "best_loss": best,
        "training_time_seconds": round(time.time() - started, 1), "language": plan.language,
        "ref_sample_audio": data["ref_audio"], "ref_sample_text": data["ref_text"],
    }
    save_adapter_folder(peft_talker, output_dir, data["ref_audio"], data["ref_text"], meta)
    if centroid is not None:
        speaker_centroid.save(output_dir, centroid[0], centroid[1])
    else:
        (output_dir / speaker_centroid.FILENAME).unlink(missing_ok=True)      # a stale one from an earlier run
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    holdout_file = output_dir / "checkpoints" / HOLDOUT_FILE
    if hold_rows:
        holdout_file.write_text(json.dumps({"fraction": plan.holdout_fraction,
                                            "rows": [{"audio": r["rel"], "text": r["text"]} for r in hold_rows]},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        holdout_file.unlink(missing_ok=True)
    (output_dir / "checkpoints" / "losses.json").write_text(
        json.dumps({"epoch_avg_loss": epoch_losses, "epoch_val_loss": val_losses, "warnings": warnings,
                    "speaker_embedding": {"source": "centroid" if centroid else "ref",
                                          "clips": len(centroid[1]) if centroid else 1}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    for w in warnings:
        log.warning("training: %s", w)
    progress(Stage.SAVE, 1.0, tr("progress.adapter_saved"))
    return output_dir


def _free_memory() -> None:
    """Garbage-collect and empty the CUDA cache (best effort)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _train_once(dataset_dir: Path, output_dir: Path, plan: TrainPlan, progress: ProgressCallback,
                cancel: CancelToken, warnings_out: Optional[List[str]] = None) -> Path:
    """Load the models and train once with the given plan (needs a GPU and the downloaded models; TODO-needs-GPU-test)."""
    import torch

    data = load_training_rows(dataset_dir)
    models = md.ensure_tts_models(plan.base_model, lambda s, f, m: progress(Stage.TRAIN, 0.0, m))
    cancel.check()
    from qwen_tts import Qwen3TTSModel  # type: ignore

    dtype = torch.bfloat16 if plan.dtype == "bfloat16" else torch.float32
    on_gpu = plan.device.startswith("cuda")
    q = Qwen3TTSModel.from_pretrained(str(models["base"]), device_map=plan.device if on_gpu else None,
                                      dtype=dtype, attn_implementation=plan.attn_implementation)
    hf_model = q.model
    # Bound as defaults: a later ``del`` in ``finally`` must not make the nested callables look unbound to the linter.
    def tokenize(text, _q=q):
        return _q.processor(text=text, return_tensors="pt", padding=True)["input_ids"]

    def encode(audio, sr, _m=hf_model):
        return _m.speech_tokenizer.encode(audio, sr=sr).audio_codes[0]
    monitor = VramMonitor(interval=10.0).start()
    try:
        return train_on_model(hf_model, tokenize, encode, data, output_dir, plan, plan.base_model,
                              progress, cancel, warnings_out)
    finally:
        monitor.stop()
        del q, hf_model
        _free_memory()


def train_lora_from_dataset(dataset_dir, output_dir, progress: ProgressCallback = noop_progress,
                            cancel: Optional[CancelToken] = None, force_cpu: bool = False,
                            plan: Optional[TrainPlan] = None, language: Optional[str] = None,
                            warnings_out: Optional[List[str]] = None, holdout_fraction: float = 0.0,
                            _run: Callable[..., Path] = _train_once) -> Path:
    """Train an adapter from a dataset folder and return the adapter *folder*.

    Chooses the parameters automatically (``infra/vram_optimizer.plan_training``), and on out-of-memory reduces the load and
    tries again; when nothing is left to reduce it raises ``OutOfMemoryError_`` (the UI then offers the CPU).  ``language``
    defaults to ``training_language`` from the dataset's ``report.json``.  ``holdout_fraction`` > 0 keeps that share of the
    clips out of training (validation loss, automatic checkpoint pick).  ``_run`` is the injection point used by tests.
    """
    cancel = cancel or CancelToken()
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    n_items = len(read_metadata_jsonl(dataset_dir / "metadata.jsonl"))
    if language is None:
        rep = dataset_dir / "report.json"
        try:
            language = json.loads(rep.read_text(encoding="utf-8")).get("training_language")
        except (OSError, ValueError):
            language = None
    plan = plan or plan_training(detect_gpu(), n_items, force_cpu=force_cpu, language=language or "russian")
    if holdout_fraction > 0:
        from dataclasses import replace

        plan = replace(plan, holdout_fraction=holdout_fraction)
    progress(Stage.TRAIN, 0.0, tr("progress.plan", device=plan.device, epochs=plan.epochs))
    while True:
        try:
            return _run(dataset_dir, output_dir, plan, progress, cancel, warnings_out)
        except OutOfMemoryError_ as exc:
            _free_memory()
            nxt = reduce_after_oom(plan)
            if nxt is None:
                raise
            log.warning("OOM -> retry with reduced plan: %s", exc.details[:200])
            progress(Stage.TRAIN, 0.0, tr("progress.oom_retry"))
            plan = nxt
        except DatasetMakerError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                _free_memory()
                nxt = reduce_after_oom(plan)
                if nxt is None:
                    raise OutOfMemoryError_(details=str(exc)) from exc
                plan = nxt
                continue
            raise TrainingError(tr("err.train_failed"),
                                details=str(exc)) from exc
