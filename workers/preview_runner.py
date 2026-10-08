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
SYNTH_CHECK_SEC = 25.0       # measured on an RTX 4090: engine load + ~7 s sample + pitch/WER per variant (~31 s wall minus the trainer's own load)
ASR_LOAD_SEC = 10.0          # measured: 8 s for Qwen3-ASR-0.6B
#: Each extra adapter strength of a variant (core.adapter_strength.PREVIEW_SCALES): one more ~7 s sample with checks on the
#: unmerged adapter.  An estimate, not measured yet (NEEDS GPU VALIDATION).
EXTRA_SCALE_SEC = 9.0
MAX_SECONDS = 180.0           # time cap of one quick training (estimated)
#: The ~10 s phrase of the quick preview and of the automatic voice check, per language (the engine needs the language of the voice).
#: Technical, developer-blog style about Voxprint itself, gender-neutral (no first-person past tense in Russian / German).
SAMPLE_TEXT = {
    "russian": "Voxprint — офлайн-приложение, которое читает книги вашим голосом. Запись в десять минут режется на фрагменты, и на них обучается LoRA-адаптер поверх Qwen3-TTS.",
    "english": "Voxprint is an offline app that reads books in your voice. A ten-minute recording is cut into clips, and a LoRA adapter is trained on them on top of Qwen3-TTS.",
    "german": "Voxprint ist eine Offline-App, die Bücher mit Ihrer Stimme vorliest. Eine zehnminütige Aufnahme wird in Clips geschnitten, darauf wird ein LoRA-Adapter auf Qwen3-TTS trainiert.",
}
#: The ~15-25 s demo passage (final narration sample, demo text in the docs): SAMPLE_TEXT plus how narration and consent work.
DEMO_TEXT = {
    "russian": SAMPLE_TEXT["russian"] + " Далее пайплайн нарезает книгу на чанки, синтезирует их с кэшем и умеет продолжить после сбоя. Перед обучением фиксируется согласие владельца голоса, а готовый адаптер проверяется по WER и высоте тона.",
    "english": SAMPLE_TEXT["english"] + " Then the pipeline splits a book into chunks and synthesizes them with a cache. Consent is recorded before training, and the finished adapter is checked by WER and pitch.",
    "german": SAMPLE_TEXT["german"] + " Danach zerlegt die Pipeline ein Buch in Chunks, synthetisiert sie mit Cache und kann nach einem Absturz fortsetzen. Vor dem Training wird die Einwilligung des Stimmeninhabers festgehalten, und der fertige Adapter wird per WER und Tonhöhe geprüft.",
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
    #: Adapter strength of ``wav`` / ``check`` (``None``: the engine could not change it, i.e. full strength).
    scale: Optional[float] = None
    #: Every strength that was synthesized: ``[{"scale", "wav", "seconds", "check"}]`` (one entry per value).
    scales: List[dict] = field(default_factory=list)

    def use_scale(self, scale: float) -> None:
        """Make the sample at ``scale`` the shown one (the UI's strength selector)."""
        for e in self.scales:
            if abs(e["scale"] - scale) < 1e-6:
                self.scale, self.wav, self.seconds, self.check = e["scale"], e["wav"], e["seconds"], e["check"]
                return


def make_variants(base: TrainPlan, compare: bool) -> List[Variant]:
    """Quick versions of the plan: few epochs; the second variant has a bigger adapter and 50 % more passes."""
    a = replace(base, epochs=max(2, min(MAX_EPOCHS, base.epochs)))
    out = [Variant("A", a)]
    if compare:
        out.append(Variant("B", replace(a, epochs=max(a.epochs + 1, int(round(a.epochs * 1.5))), lora_r=a.lora_r * 2,
                                        lora_alpha=a.lora_alpha * 2)))
    return out


_VERDICT_RANK = {voice_check.GOOD: 0, voice_check.WARN: 1, voice_check.BAD: 2}


def recommend(items: List[PreviewItem]) -> Optional[str]:
    """Key of the variant to recommend, or ``None`` for an empty list.

    Order of the criteria: the automatic verdict (good before warn before bad), then the lower word error rate, then the
    smaller pitch shift.  A tie goes to the first variant (A: fewer passes, the smaller adapter, the cheaper full run), and a
    difference below the noise of a single ~10 s sample (WER within 0.03, pitch within 0.5 semitone) counts as a tie."""
    if not items:
        return None

    def wer_of(it: PreviewItem) -> float:
        w = (it.check or {}).get("wer")
        return 1.0 if w is None else float(w)

    def pitch_of(it: PreviewItem) -> float:
        st = (it.check or {}).get("semitones")
        return 99.0 if st is None else abs(float(st))

    best = items[0]
    for it in items[1:]:
        rb, ri = (_VERDICT_RANK.get((best.check or {}).get("verdict", voice_check.GOOD), 1),
                  _VERDICT_RANK.get((it.check or {}).get("verdict", voice_check.GOOD), 1))
        if ri != rb:
            if ri < rb:
                best = it
        elif best_wer_gap(wer_of(best), wer_of(it)) != 0:
            if wer_of(it) < wer_of(best):
                best = it
        elif pitch_of(best) - pitch_of(it) > 0.5:
            best = it
    return best.key


def best_wer_gap(a: float, b: float) -> float:
    """Difference of two word error rates, 0 when it is within the sample noise (0.03)."""
    return 0.0 if abs(a - b) <= 0.03 else a - b


def best_scale(entries: List[dict]) -> Optional[float]:
    """The strength to pre-select among ``entries`` (``{"scale", "check"}``): the :func:`recommend` criteria, ties going to
    the default strength (then to the weaker one - the research found lower values less "over-dry")."""
    from core import adapter_strength

    if not entries:
        return None
    order = sorted(entries, key=lambda e: (abs(e["scale"] - adapter_strength.DEFAULT_SCALE), e["scale"]))
    pseudo = [PreviewItem(str(i), "", 0, 0, 0, 0.0, 0, Path(), 0.0, 0.0, e["check"]) for i, e in enumerate(order)]
    return order[int(recommend(pseudo))]["scale"]


AUTO = "auto"     # ``mos=AUTO``: the DNSMOS scorer if it is downloaded (core.mos.default_mos), never a download


def resolve_mos(mos):
    """``AUTO`` -> the downloaded DNSMOS scorer or ``None``; anything else is returned as it is (``None`` = no MOS)."""
    if mos == AUTO:
        from core.mos import default_mos

        return default_mos()
    return mos


def reference_embedding(eng, adapter_dir: Path, ref: np.ndarray, ref_sr: int):
    """The speaker embedding samples are compared with: the encoder's embedding of the reference clip; ``None`` when the
    engine has no speaker encoder (test fakes) or it failed."""
    fn = getattr(eng, "speaker_embedding", None)
    if not callable(fn):
        return None
    try:
        return fn(ref, ref_sr)
    except Exception as exc:  # noqa: BLE001 - an extra metric must never break the check
        log.warning("reference speaker embedding failed: %s", exc)
        return None


def sample_similarity(eng, audio: np.ndarray, sr: int, ref_emb) -> Optional[float]:
    """Speaker similarity of ``audio`` to ``ref_emb`` with the engine's own speaker encoder (0 MB extra; the same encoder
    that conditions the voice, so it "judges itself" - good for ranking, not an independent verdict)."""
    if ref_emb is None:
        return None
    try:
        return voice_check.cosine(eng.speaker_embedding(audio, sr), ref_emb)
    except Exception as exc:  # noqa: BLE001
        log.warning("speaker similarity failed: %s", exc)
        return None


def mos_of(mos_model, audio: np.ndarray, sr: int) -> Optional[float]:
    """DNSMOS OVRL of ``audio`` or ``None``."""
    if mos_model is None:
        return None
    try:
        r = mos_model.score(audio, sr)
        return None if not r else float(r["ovrl"])
    except Exception as exc:  # noqa: BLE001
        log.warning("MOS prediction failed: %s", exc)
        return None


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
                 force_cpu: bool = False, mos=AUTO) -> List[PreviewItem]:
    """Train the quick variants on a subset, synthesize the sample of each and check it.

    ``train_fn(dataset, out, progress, cancel, force_cpu, plan=..., language=...) -> adapter folder`` defaults to
    :func:`core.lora_trainer.train_lora_from_dataset`; ``engine_factory(adapter_dir, language)`` returns an object with
    ``synthesize(text) -> samples`` and ``sample_rate`` (and ``close()``), defaulting to the real Qwen3 engine.  When the
    engine also has ``set_adapter_scale(scale)``, every variant is synthesized at each of
    :data:`core.adapter_strength.PREVIEW_SCALES` and the best-checking strength is pre-selected (the user can switch).
    An engine with ``speaker_embedding(audio, sr)`` adds the speaker similarity; ``mos`` (default: DNSMOS if downloaded) the
    predicted MOS.
    """
    import time

    from core import adapter_strength
    from core.tts_engine import max_tokens_for, FRAMES_PER_SECOND

    cancel = cancel or CancelToken()
    if train_fn is None:
        from core.lora_trainer import train_lora_from_dataset as train_fn   # noqa: N813
    out_dir = Path(out_dir)
    text = SAMPLE_TEXT.get((language or "english").lower(), SAMPLE_TEXT["english"])
    variants = make_variants(base_plan, compare)
    items: List[PreviewItem] = []
    ref, ref_sr = au.load_audio(Path(dataset_dir) / "ref.wav", 24000)
    mos = resolve_mos(mos)
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
        entries: List[dict] = []
        try:
            ref_emb = reference_embedding(eng, adapter, ref, ref_sr)
            scalable = callable(getattr(eng, "set_adapter_scale", None))
            for scale in (adapter_strength.PREVIEW_SCALES if scalable else (None,)):
                cancel.check()
                if scale is not None:
                    eng.set_adapter_scale(scale)
                audio = np.asarray(eng.synthesize(text), dtype=np.float32).reshape(-1)
                sr = int(getattr(eng, "sample_rate", 24000))
                wav = out_dir / (f"preview_{v.key}.wav" if scale is None else f"preview_{v.key}_s{int(round(scale * 100)):03d}.wav")
                au.write_wav(wav, audio, sr)
                chk = voice_check.check_sample(audio, sr, text, ref, ref_sr, asr=asr,
                                               language=(language or "").capitalize() or None,
                                               max_seconds=max_tokens_for(text) / FRAMES_PER_SECOND,
                                               sim=sample_similarity(eng, audio, sr, ref_emb), mos=mos_of(mos, audio, sr))
                entries.append({"scale": scale, "wav": wav, "seconds": len(audio) / sr, "check": chk.as_dict()})
        finally:
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                pass
        first = entries[0]
        item = PreviewItem(v.key, v.label or v.key, v.plan.epochs, v.plan.lora_r, v.plan.lora_alpha, v.plan.lr,
                           v.plan.grad_accum, first["wav"], first["seconds"], t_train, first["check"], n,
                           scales=entries if first["scale"] is not None else [])
        if item.scales:
            item.use_scale(best_scale(item.scales))
        items.append(item)
    progress(Stage.TRAIN, 1.0, tr("preview.done"))
    return items


def _default_engine(adapter: Path, language: str, merge: bool = False):
    """The real engine on a temporary voice record made from the adapter folder (its voice.json strength, if any).

    Unmerged by default: the strength can then be changed between the samples without reloading the model."""
    import json

    from core import voice_info
    from core.tts_engine import make_engine_factory
    from core.voice_library import VoiceRecord

    meta = {}
    try:
        meta = json.loads((adapter / "training_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    raw = {"name": "preview", "language": language, "base_model": meta.get("model_name", "")}
    try:     # after a full training voice.json is already written: check the voice at the strength it will speak with
        stored = json.loads((adapter / "voice.json").read_text(encoding="utf-8"))
        if isinstance(stored, dict) and stored.get("adapter_scale") is not None:
            raw["adapter_scale"] = stored["adapter_scale"]
    except (OSError, ValueError):
        pass
    info = voice_info.normalize_info(raw, "preview")
    return make_engine_factory(VoiceRecord("preview", adapter, info), language, merge=merge)()
