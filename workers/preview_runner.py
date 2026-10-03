"""Quick preview before the full training (Qt-free): a short training on a small subset, a ~10 s sample, automatic checks.

``Compare 2 variants`` trains two slightly different settings (the chosen preset and a bigger adapter with more passes - the
learning rate is deliberately the same: raising it made voices babble in real tests); the user listens and picks one.  Time
is capped by shrinking the subset until the estimate fits ``MAX_SECONDS``; the VRAM cap is the trainer's own
(out-of-memory -> lighter plan).  Training, synthesis and recognition are injectable, so tests need no GPU.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from core import audio_utils as au
from core import train_presets, voice_check
from core.dataset_builder import read_metadata_jsonl, write_metadata_jsonl
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr
from infra.vram_optimizer import GpuInfo, TrainPlan

log = logging.getLogger("voxprint.preview")

MAX_CLIPS = 12
MIN_CLIPS = 4
MAX_EPOCHS = 4
MAX_SECONDS = 180.0           # time cap of one quick training (estimated)
#: The ~10 s sentence the sample says, per language (the engine needs the language of the voice).
SAMPLE_TEXT = {
    "russian": "Здравствуйте! Это короткий образец моего голоса: так программа будет читать вам книги вслух.",
    "english": "Hello! This is a short sample of my voice, the way the program will read books aloud to you.",
    "german": "Guten Tag! Das ist eine kurze Hörprobe meiner Stimme, so liest Ihnen das Programm Bücher vor.",
}


@dataclass
class Variant:
    key: str
    plan: TrainPlan
    label: str = ""


@dataclass
class PreviewItem:
    key: str
    label: str
    epochs: int
    lora_r: int
    lora_alpha: int
    lr: float
    grad_accum: int
    wav: Path
    seconds: float
    train_seconds: float
    check: dict = field(default_factory=dict)
    clips: int = 0


def make_variants(base: TrainPlan, compare: bool) -> List[Variant]:
    """Quick versions of the plan: few epochs; the second variant has a bigger adapter and 50 % more passes."""
    a = replace(base, epochs=max(2, min(MAX_EPOCHS, base.epochs)))
    out = [Variant("A", a)]
    if compare:
        out.append(Variant("B", replace(a, epochs=max(a.epochs + 1, int(round(a.epochs * 1.5))), lora_r=a.lora_r * 2,
                                        lora_alpha=a.lora_alpha * 2)))
    return out


def subset(dataset_dir: Path, out: Path, n_max: int = MAX_CLIPS) -> int:
    """Copy up to ``n_max`` evenly spaced clips (and ref.wav / ref_text.txt) into ``out``; returns the count."""
    rows = read_metadata_jsonl(Path(dataset_dir) / "metadata.jsonl")
    n = max(1, min(n_max, len(rows)))
    pick = [rows[round(i * (len(rows) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 else rows[:1]
    seen, chosen = set(), []
    for r in pick:
        if r["audio"] not in seen:
            seen.add(r["audio"]); chosen.append(r)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for r in chosen:
        shutil.copy2(Path(dataset_dir) / r["audio"], out / r["audio"])
    for f in ("ref.wav", "ref_text.txt", "report.json"):
        if (Path(dataset_dir) / f).is_file():
            shutil.copy2(Path(dataset_dir) / f, out / f)
    write_metadata_jsonl(out / "metadata.jsonl", chosen)
    return len(chosen)


def clips_for_time_cap(plan: TrainPlan, gpu: GpuInfo, n: int, cap: float = MAX_SECONDS) -> int:
    """The largest clip count <= ``n`` whose estimated training time fits ``cap`` (but at least ``MIN_CLIPS``)."""
    while n > MIN_CLIPS and train_presets.estimate_seconds(plan, n, gpu) > cap:
        n -= 1
    return n


def run_previews(dataset_dir, out_dir, base_plan: TrainPlan, *, compare: bool, language: str, gpu: GpuInfo,
                 progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None,
                 train_fn: Optional[Callable] = None, engine_factory: Optional[Callable] = None, asr=None,
                 force_cpu: bool = False) -> List[PreviewItem]:
    """Train the quick variants on a subset, synthesize the sample of each and check it.

    ``train_fn(dataset, out, progress, cancel, force_cpu, plan=..., language=...) -> adapter folder`` defaults to
    :func:`core.lora_trainer.train_lora_from_dataset`; ``engine_factory(adapter_dir, language)`` returns an object with
    ``synthesize(text) -> samples`` and ``sample_rate`` (and ``close()``), defaulting to the real Qwen3 engine.
    """
    import time

    from core.tts_engine import max_tokens_for, FRAMES_PER_SECOND

    cancel = cancel or CancelToken()
    if train_fn is None:
        from core.lora_trainer import train_lora_from_dataset as train_fn   # noqa: N813
    out_dir = Path(out_dir)
    text = SAMPLE_TEXT.get((language or "english").lower(), SAMPLE_TEXT["english"])
    variants = make_variants(base_plan, compare)
    items: List[PreviewItem] = []
    ref, ref_sr = au.load_audio(Path(dataset_dir) / "ref.wav", 24000)
    for vi, v in enumerate(variants):
        cancel.check()
        base_f = vi / len(variants)

        def sub(stage, f, m, base_f=base_f):
            progress(stage, base_f + f / len(variants), m)
        sdir = out_dir / f"subset_{v.key}"
        n = subset(dataset_dir, sdir, clips_for_time_cap(v.plan, gpu, MAX_CLIPS))
        progress(Stage.TRAIN, base_f, tr("preview.training", v=v.key, n=n, epochs=v.plan.epochs))
        t0 = time.time()
        adapter = Path(train_fn(sdir, out_dir / f"adapter_{v.key}", sub, cancel, force_cpu, plan=v.plan, language=language))
        t_train = time.time() - t0
        cancel.check()
        progress(Stage.TRAIN, base_f + 0.9 / len(variants), tr("preview.synthesizing", v=v.key))
        eng = engine_factory(adapter, language) if engine_factory else _default_engine(adapter, language)
        try:
            audio = np.asarray(eng.synthesize(text), dtype=np.float32).reshape(-1)
            sr = int(getattr(eng, "sample_rate", 24000))
        finally:
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                pass
        wav = out_dir / f"preview_{v.key}.wav"
        au.write_wav(wav, audio, sr)
        chk = voice_check.check_sample(audio, sr, text, ref, ref_sr, asr=asr, language=(language or "").capitalize() or None,
                                       max_seconds=max_tokens_for(text) / FRAMES_PER_SECOND)
        items.append(PreviewItem(v.key, v.label or v.key, v.plan.epochs, v.plan.lora_r, v.plan.lora_alpha, v.plan.lr,
                                 v.plan.grad_accum, wav, len(audio) / sr, t_train, chk.as_dict(), n))
    progress(Stage.TRAIN, 1.0, tr("preview.done"))
    return items


def _default_engine(adapter: Path, language: str):
    """The real engine on a temporary voice record made from the quick adapter."""
    import json

    from core import voice_info
    from core.tts_engine import make_engine_factory
    from core.voice_library import VoiceRecord

    meta = {}
    try:
        meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    info = voice_info.normalize_info({"name": "preview", "language": language, "base_model": meta.get("model_name", "")}, "preview")
    return make_engine_factory(VoiceRecord("preview", adapter, info), language)()
