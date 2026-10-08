"""Application scenarios without Qt: "dataset", "voice (LoRA)", "universal model" and the first-run model prefetch.

One call runs the whole chain of stages.  Failures are :class:`DatasetMakerError` exceptions with a user-readable
message (the UI shows them).  Everything here is plain Python so it can be driven from the GUI worker, the CLI and tests.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.aligner import make_default_aligner
from core.dataset_builder import BuildConfig, DatasetBuilder
from core.errors import DatasetMakerError
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr
from infra import model_downloader as md, paths
from infra.updater import Updater

log = logging.getLogger("voxprint.runner")

KIND_DATASET = "dataset"
KIND_LORA = "lora"
KIND_PREVIEW = "preview"   # quick preview: short training on a subset + a sample (nothing is registered)
KIND_MERGE = "merge"   # universal (merged) model from an already trained adapter (button only)

#: Stages of each scenario, used to turn per-stage progress into one overall percentage.
PLAN_DATASET: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.SAVE]
PLAN_LORA: List[Stage] = [Stage.UPDATES, Stage.MODEL, Stage.ALIGN, Stage.SLICE, Stage.TRAIN, Stage.SAVE]
PLAN_MERGE: List[Stage] = [Stage.MODEL, Stage.SAVE]


def plan_for(kind: str) -> List[Stage]:
    """The ordered list of stages a task kind goes through (for the overall progress percentage)."""
    if kind == KIND_MERGE:
        return PLAN_MERGE
    return PLAN_LORA if kind in (KIND_LORA, KIND_PREVIEW) else PLAN_DATASET


def safe_name(value: str) -> str:
    """Make ``value`` safe as a folder name (letters, digits, ``-._`` and spaces are kept; falls back to ``voice``)."""
    return re.sub(r"[^\w\-. ]", "_", value).strip() or "voice"


@dataclass
class TaskRequest:
    """What the user asked for: the task kind, input files, output folder, CPU switch and optional voice metadata."""
    kind: str
    audio: Optional[Path] = None
    text: Optional[Path] = None
    out_root: Optional[Path] = None
    force_cpu: bool = False
    adapter_dir: Optional[Path] = None   # KIND_MERGE only: folder of the trained adapter
    voice_display_name: str = ""         # optional, typed by the user: voice name, library folder and result folder
    voice_type: str = ""                 # optional, older form (CLI --type): male / female / child / other
    voice_description: str = ""          # optional free text, goes to voice.json
    gender: str = ""                     # optional voice.json details (schema 3, core/voice_info.py); gender / age group
    age_group: str = ""                  # win over voice_type when set
    speaker: str = ""
    prepared_by: str = ""
    organization: str = ""
    project_url: str = ""
    no_transcript: bool = False          # audio only: the app recognises the speech itself (see core.asr_dataset)
    audio_files: List[Path] = field(default_factory=list)   # no_transcript: files and/or folders (many clips)
    asr_language: Optional[str] = None   # no_transcript: "Russian" / "English" ... or None = automatic
    consent_mode: str = "none"           # voice-owner consent: "auto" (read the spoken statement), "manual" or "none" (private only)
    consent_scope: str = "private_only"  # manual: commercial / public_noncommercial / private_only
    consent_name: str = ""               # manual: the speaker's name
    consent_save_clip: bool = True       # auto: keep the recorded statement next to the adapter
    compare: bool = False                # KIND_PREVIEW: two variants to compare
    quality_check: bool = False          # LoRA: synthesize a sample after training and judge it (core/voice_check.py)
    preset: str = "balanced"             # training preset: fast / balanced / maximum / manual (core.train_presets)
    manual: Optional[object] = None      # core.train_presets.Manual for the "manual" preset
    adapter_scale: Optional[float] = None  # LoRA strength chosen in the preview (None = automatic, core/adapter_strength.py)

    def voice_name(self) -> str:
        """Voice name = the recording's file name (or the adapter folder's name); also the result folder name in ``output/``."""
        if self.voice_display_name.strip():
            return safe_name(self.voice_display_name.strip())
        if self.audio:
            return safe_name(Path(self.audio).stem)
        if self.audio_files:
            first = Path(self.audio_files[0])
            return safe_name(first.name if first.is_dir() else (first.parent.name if len(self.audio_files) > 1 else first.stem))
        if self.adapter_dir:
            return safe_name(Path(self.adapter_dir).name)
        return "voice"

    def effective_voice_type(self) -> str:
        """``voice_type`` as voice.json will store it: derived from gender / age group, else the older ``voice_type``."""
        from core import voice_info

        if voice_info.normalize_gender(self.gender) or voice_info.normalize_age_group(self.age_group):
            return voice_info.derive_voice_type(self.gender, self.age_group)
        return voice_info.normalize_voice_type(self.voice_type)

    def voice_folder(self) -> str:
        """Name of the folder with the trained voice: the voice name plus its type (``anna_male``, ``anna_unspecified``)."""
        from core import voice_info

        if self.voice_display_name.strip():
            return self.voice_name()                       # the user's own name, no type suffix
        return safe_name(voice_info.with_type_suffix(self.voice_name(), self.effective_voice_type()))

    def resolved_root(self) -> Path:
        """Result folder: ``out_root`` if given, else ``<audio folder>/<voice>_Voxprint``."""
        if self.out_root:
            return Path(self.out_root)
        if self.audio is None and not self.audio_files:
            raise ValueError("audio is required")
        base = Path(self.audio) if self.audio else Path(self.audio_files[0])
        base = base.resolve() if base.is_dir() else base.resolve().parent
        return base / f"{self.voice_name()}_Voxprint"


@dataclass
class TaskResult:
    """What a finished task produced: folders, number of segments, warnings and (for LoRA/merge) the artifact paths."""
    kind: str
    root_dir: Path
    dataset_dir: Path
    adapter_path: Optional[Path] = None
    n_segments: int = 0
    warnings: List[str] = field(default_factory=list)
    update_summary: str = ""
    merged_path: Optional[Path] = None   # folder of the universal model (KIND_MERGE)
    speaker: str = ""                   # voice (speaker) name inside the model
    voice_id: str = ""                  # id of the voice registered in the voice library (LoRA only)
    previews: List[object] = field(default_factory=list)   # KIND_PREVIEW: workers.preview_runner.PreviewItem list
    quality: Optional[dict] = None       # LoRA with quality_check: the checks of the sample (voice_check.Check.as_dict())
    consent: Optional[dict] = None       # the voice.json "consent" block written for this voice (LoRA only)
    asr_report: Optional[object] = None  # no-transcript mode: core.asr_dataset.AsrReport (files, kept clips, seconds ...)

    @property
    def open_dir(self) -> Path:
        """Folder the UI opens when the task is done."""
        return self.root_dir


def safe_auto_update(progress: ProgressCallback, updater: Optional[Updater] = None) -> str:
    """Weekly automatic update check.  Any error is swallowed: updates must never get in the way of the work."""
    try:
        u = updater or Updater()
        if not u.should_autocheck():
            progress(Stage.UPDATES, 1.0, tr("upd.recent"))
            return ""
        _, res = u.check_and_apply(progress)
        return res.summary() if (res.after or res.models_updated or res.rolled_back) else ""
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-update failed: %s", exc)
        progress(Stage.UPDATES, 1.0, tr("upd.skipped"))
        return ""


# --------------------------------------------------------------------------- last adapter


def _last_adapter_file() -> Path:
    """State file remembering the most recently trained adapter."""
    return paths.state_dir() / "last_adapter.json"


def remember_adapter(adapter_dir: Path, voice_name: str = "") -> None:
    """Remember the last trained adapter - for the "universal model" button."""
    try:
        _last_adapter_file().write_text(json.dumps(
            {"adapter_dir": str(adapter_dir), "voice_name": voice_name, "time": int(time.time())},
            ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("cannot remember adapter: %s", exc)


def last_adapter() -> Optional[Path]:
    """Folder of the last trained adapter; None if there was none or the folder was deleted."""
    try:
        d = Path(json.loads(_last_adapter_file().read_text(encoding="utf-8"))["adapter_dir"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return d if (d / "adapter_model.safetensors").exists() else None


def _write_voice_json(req: TaskRequest, adapter_dir: Path, language: str, speech_seconds: float,
                      consent_block: Optional[dict] = None, picked_scale: Optional[float] = None) -> dict:
    """Write voice.json next to the adapter and return its content. A failure here must not fail a finished training run.

    Adapter strength: the one the user chose in the preview, else the automatic checkpoint pick's, else the default."""
    from core import adapter_strength, voice_info

    epochs, base_model = 0, ""
    try:   # epochs and base model are recorded by the trainer in training_meta.json
        meta = json.loads((Path(adapter_dir) / "training_meta.json").read_text(encoding="utf-8"))
        epochs, base_model = int(meta.get("epochs", 0)), str(meta.get("model_name", ""))
    except (OSError, ValueError, TypeError):
        log.warning("training_meta.json unreadable; voice.json will lack epochs/base model")
    extra = {}
    if consent_block:   # the scope maps onto the licence fields; the speaker's name becomes the author
        from core import consent as consent_mod

        extra = dict(license=consent_mod.license_for_scope(consent_block["scope"]), consent=consent_block,
                     author=consent_block.get("name", ""))
    info = voice_info.build_voice_info(req.voice_display_name.strip() or req.voice_name(), language, speech_seconds, epochs, base_model,
                                       req.voice_type, req.voice_description, gender=req.gender, age_group=req.age_group,
                                       speaker=req.speaker, prepared_by=req.prepared_by, organization=req.organization,
                                       project_url=req.project_url, **extra)
    scale = next((v for v in (req.adapter_scale, picked_scale) if v is not None), adapter_strength.DEFAULT_SCALE)
    info["adapter_scale"] = adapter_strength.clamp(scale, adapter_strength.DEFAULT_SCALE)
    try:
        voice_info.write_voice_json(adapter_dir, info)
    except OSError as exc:
        log.warning("cannot write voice.json: %s", exc)
    return info


def _register_voice(adapter_dir: Path, info: dict, library=None, typed_id: bool = True) -> str:
    """Add the freshly trained adapter to the voice library; returns the voice id ("" if that failed)."""
    from core.voice_library import VoiceLibrary

    try:
        return (library or VoiceLibrary()).add_from_adapter(adapter_dir, info, typed_id=typed_id).id
    except Exception as exc:  # noqa: BLE001 - the adapter itself is already saved; never fail the run for the library
        log.warning("cannot register the voice in the library: %s", exc)
        return ""


def _asr_for_check(req: TaskRequest, asr_factory, progress=None):
    """The recogniser for WER checks, or None (the check then simply has no WER).

    Uses a model already fetched by the first-run / ``--prefetch`` download-all (or present elsewhere on the PC).
    Never starts a surprise download here - if the model is missing, WER is skipped with a log line.
    """
    try:
        if asr_factory is not None:
            return asr_factory()
        from core.asr import make_default_asr

        path = md.ready_model_path(md.ASR_REPO)
        if path is None:
            log.warning("no ASR for the voice check: %s is not downloaded yet "
                        "(expected from the first-run / --prefetch download-all)", md.ASR_REPO)
            return None
        return make_default_asr(str(path), "cpu" if req.force_cpu else "auto")
    except Exception as exc:  # noqa: BLE001 - WER is a bonus, never a reason to fail
        log.warning("no ASR for the voice check: %s", exc)
        return None


def _run_preview(req, build, dataset_dir, root, progress, cancel, asr_factory, deps) -> list:
    """Quick preview(s): see :mod:`workers.preview_runner`."""
    from core import train_presets
    from infra.vram_optimizer import detect_gpu
    from workers import preview_runner

    gpu = detect_gpu()
    plan = train_presets.build_plan(req.preset, gpu, build.n_segments, force_cpu=req.force_cpu,
                                    language=build.training_language, manual=req.manual)
    asr = _asr_for_check(req, asr_factory, progress)
    if asr is not None:
        try:
            asr.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("ASR could not be loaded for the preview check: %s", exc)
            asr = None
    return preview_runner.run_previews(dataset_dir, root / "preview", plan, compare=req.compare, language=build.training_language,
                                       gpu=gpu, progress=progress, cancel=cancel, force_cpu=req.force_cpu, asr=asr,
                                       train_fn=deps.get("train_fn"), engine_factory=deps.get("engine_factory"))


def _quality_check(req, res, build, asr_factory, deps, cancel) -> None:
    """Automatic post-training check: synthesize a short sample with the new voice, judge it, suggest what to change.  Never raises."""
    import numpy as np

    from core import audio_utils as au, voice_check
    from core.tts_engine import FRAMES_PER_SECOND, max_tokens_for
    from workers import preview_runner

    try:
        adapter = Path(res.adapter_path)
        lang = build.training_language
        text = preview_runner.SAMPLE_TEXT.get(lang, preview_runner.SAMPLE_TEXT["english"])
        eng = (deps["engine_factory"](adapter, lang) if deps.get("engine_factory")
               else preview_runner._default_engine(adapter, lang, merge=True))
        try:
            audio = np.asarray(eng.synthesize(text), dtype=np.float32).reshape(-1)
            sr = int(getattr(eng, "sample_rate", 24000))
        finally:
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                pass
        asr = _asr_for_check(req, asr_factory)
        if asr is not None:
            asr.load()
        try:
            ref, ref_sr = au.load_audio(adapter / "ref_sample.wav", 24000)
            chk = voice_check.check_sample(audio, sr, text, ref, ref_sr, asr=asr, language=lang.capitalize(),
                                           max_seconds=max_tokens_for(text) / FRAMES_PER_SECOND)
        finally:
            if asr is not None:
                asr.unload()
        res.quality = chk.as_dict()
        if chk.verdict != voice_check.GOOD:
            res.warnings.append(tr("check.verdict_" + chk.verdict) + " " + " ".join(tr("check.sugg_" + c) for c in chk.issues
                                                                                 if c != "no_pitch"))
    except DatasetMakerError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("voice quality check failed: %s", exc)


def _consent_block(req: TaskRequest, res: TaskResult, progress: ProgressCallback, cancel: CancelToken,
                   asr_factory) -> Optional[dict]:
    """The ``consent`` block for voice.json.  "auto" reads the spoken statement at the end of the recording; whatever goes wrong
    (no ASR model, no statement found) ends in the most restrictive scope with a warning - never in a failed training run."""
    from core import consent

    if req.consent_mode == "manual":
        res.consent = consent.build_consent(None, scope=req.consent_scope, name=req.consent_name, method="manual",
                                            recorded=False, confirmed=True)
        return res.consent
    if req.consent_mode != "auto":
        res.consent = consent.build_consent(None, scope=consent.PRIVATE, method="none", recorded=False, confirmed=True)
        return res.consent
    parsed, clip = None, None
    try:
        from core.asr import make_default_asr
        from core.asr_dataset import expand_inputs
        from workers import consent_runner

        files = expand_inputs(req.audio_files) if req.audio_files else [req.audio]
        progress(Stage.TRAIN, 0.0, tr("progress.consent_reading"))
        if asr_factory is not None:
            asr = asr_factory()
        else:
            path = md.ready_model_path(md.ASR_REPO)
            if path is None:
                raise RuntimeError(f"{md.ASR_REPO} is not downloaded yet "
                                   "(expected from the first-run / --prefetch download-all)")
            asr = make_default_asr(str(path), "cpu" if req.force_cpu else "auto")
        parsed, clip = consent_runner.detect_from_recording(files[-1], asr, req.asr_language)
        cancel.check()
    except DatasetMakerError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("consent statement could not be read: %s", exc)
    saved = ""
    if parsed is not None and req.consent_save_clip and clip is not None and res.adapter_path:
        try:
            saved = consent_runner.save_clip(res.adapter_path, clip)
        except OSError as exc:
            log.warning("cannot save the consent clip: %s", exc)
    if parsed is None or not parsed.text:
        res.warnings.append(tr("warn.consent_not_found"))
        parsed = parsed or consent.Parsed()
        res.consent = consent.build_consent(parsed, scope=consent.PRIVATE, method="spoken", recorded=False, confirmed=False)
        return res.consent
    res.consent = consent.build_consent(parsed, scope=parsed.scope, method="spoken", recorded=True, confirmed=False, clip=saved)
    if not parsed.confident:
        res.warnings.append(tr("warn.consent_unclear"))
    return res.consent


def _run_merge(req: TaskRequest, progress: ProgressCallback, cancel: CancelToken) -> TaskResult:
    """Export the adapter merged into the base model as a universal model (``KIND_MERGE``)."""
    from core.errors import ExportError
    from core.model_export import MERGED_DIRNAME, export_merged_model, speaker_name

    if not req.adapter_dir:
        raise ExportError(tr("err.export_no_adapter"))
    adapter = Path(req.adapter_dir)
    out = adapter / MERGED_DIRNAME
    name = req.voice_name()
    export_merged_model(adapter, out, progress=progress, cancel=cancel, voice_name=name)
    return TaskResult(KIND_MERGE, out, adapter, adapter_path=adapter, merged_path=out, speaker=speaker_name(name))


def run_task(req: TaskRequest, progress: ProgressCallback = noop_progress, cancel: Optional[CancelToken] = None,
             updater: Optional[Updater] = None,
             aligner_factory: Optional[Callable[[], object]] = None, voice_library=None,
             asr_factory: Optional[Callable[[], object]] = None, synth_deps: Optional[dict] = None) -> TaskResult:
    """Run one scenario end to end and return its :class:`TaskResult`.

    Dataset: check updates -> load aligner -> align -> slice -> quality filter -> save.  LoRA additionally trains the
    adapter, remembers it, writes ``voice.json`` next to it and registers the voice in the voice library.  ``updater`` and ``aligner_factory`` are injectable for
    tests (so is ``voice_library``, the registry that receives the new voice); ``cancel`` is checked between stages and inside the long loops.
    """
    cancel = cancel or CancelToken()
    if req.kind == KIND_MERGE:
        return _run_merge(req, progress, cancel)
    root = req.resolved_root()
    dataset_dir, output_dir = root / "dataset", root / "output" / req.voice_folder()
    lora = req.kind == KIND_LORA
    preview = req.kind == KIND_PREVIEW

    progress(Stage.UPDATES, 0.0, tr("upd.checking"))
    summary = safe_auto_update(progress, updater)
    cancel.check()

    if req.no_transcript:   # audio only: ASR -> quality gates -> merged dataset (no aligner, no text file)
        from core.asr import make_default_asr
        from core.asr_dataset import AsrConfig, build_from_audio

        if asr_factory is not None:
            asr = asr_factory()
        else:
            # intentional feature: download with visible progress if the first-run prefetch was skipped
            asr = make_default_asr(str(md.ensure_model(md.ASR_REPO, progress)), "cpu" if req.force_cpu else "auto")
        script_text = None
        if req.text:   # an optional script next to the audio: matched tolerantly (stumbles / re-read lines), see core.script_match
            from core.text_utils import read_text_file
            script_text = read_text_file(req.text).text
        build = build_from_audio(req.audio_files or [req.audio], dataset_dir, asr, AsrConfig(language=req.asr_language),
                                 progress, cancel, script_text=script_text)
    else:
        if aligner_factory is not None:
            aligner = aligner_factory()
        else:
            path = md.ensure_aligner_model(progress)
            aligner = make_default_aligner(str(path), "cpu" if req.force_cpu else "auto")
        cfg = BuildConfig(max_edge_gap=60.0 if req.consent_mode == "auto" else BuildConfig.max_edge_gap)
        try:
            build = DatasetBuilder(aligner, cfg, save_stage=Stage.SLICE if (lora or preview) else Stage.SAVE).run(
                req.audio, req.text, dataset_dir, progress, cancel)
        finally:
            try:
                aligner.unload()
            except Exception:  # noqa: BLE001
                pass
    res = TaskResult(req.kind, root, dataset_dir, n_segments=build.n_segments, warnings=list(build.warnings),
                     update_summary=summary, asr_report=getattr(build, "asr_report", None))
    if preview:
        res.previews = _run_preview(req, build, dataset_dir, root, progress, cancel, asr_factory, synth_deps or {})
        return res
    if lora:
        from core.lora_trainer import train_lora_from_dataset

        kw = {}
        if req.preset != "balanced":   # Balanced is the automatic plan (what train_lora_from_dataset chooses itself)
            from core.train_presets import build_plan
            from infra.vram_optimizer import detect_gpu

            kw["plan"] = build_plan(req.preset, detect_gpu(), build.n_segments, force_cpu=req.force_cpu,
                                    language=build.training_language, manual=req.manual)
        res.adapter_path = train_lora_from_dataset(dataset_dir, output_dir, progress, cancel, req.force_cpu,
                                                   language=build.training_language, warnings_out=res.warnings, **kw)
        remember_adapter(res.adapter_path, req.voice_name())
        cblock = _consent_block(req, res, progress, cancel, asr_factory)
        info = _write_voice_json(req, res.adapter_path, build.language, build.total_seconds, cblock)
        res.voice_id = _register_voice(res.adapter_path, info, voice_library,
                                       typed_id=not req.voice_display_name.strip())   # a typed name is used as is
        if not res.voice_id:
            res.warnings.append(tr("warn.voice_not_registered"))
        if req.quality_check:
            _quality_check(req, res, build, asr_factory, synth_deps or {}, cancel)
    return res


# --------------------------------------------------------------------------- first run


def required_model_repos() -> List[str]:
    """Models needed for the full scenario on this computer (the TTS base depends on the available VRAM).

    Includes speech recognition (Qwen3-ASR) and the Russian text clean-up model (SAGE) so the first-run /
    ``--prefetch`` / installer "download everything needed" pass fetches them once with visible progress.
    Features that need them later (voice check, A/B, spoken consent, narrate prep) then use the local copy
    instead of surprise-downloading in the background.
    """
    from infra import text_models
    from infra.vram_optimizer import detect_gpu, plan_training

    plan = plan_training(detect_gpu(), 100)
    repos = [md.ALIGNER_REPO, plan.base_model, md.ASR_REPO]
    sage = text_models.get("sage-ru")
    if sage.integrated and sage.repo:
        repos.append(sage.repo)
    return repos


def models_missing(repos: Optional[List[str]] = None) -> List[str]:
    """Models that exist neither in Voxprint's folder nor in another program's copy (they must be downloaded)."""
    repos = repos if repos is not None else required_model_repos()
    return [r for r in repos
            if not md.verify_local_model(md.local_dir_for(r)) and md.external_model(r) is None]


def _restore_backup_source(progress: ProgressCallback) -> None:
    """First run after the user pointed the installer (or Settings) at a Voxprint backup: restore it into the live folders
    BEFORE anything is looked up or downloaded (:mod:`infra.existing_models`).  Best effort: a failed restore is logged and
    the normal path (per-model import, then download) follows."""
    from core.errors import CancelledByUser
    from infra import existing_models

    try:
        existing_models.adopt_backup_choice()
        if existing_models.restore_pending():
            existing_models.restore_backup(lambda f, m="": progress(Stage.MODEL, f, m))
    except CancelledByUser:
        raise
    except Exception as exc:  # noqa: BLE001 - never block the first-run model step
        log.warning("restoring the backup source failed: %s", exc)


def prefetch_models(progress: ProgressCallback = noop_progress, repos: Optional[List[str]] = None,
                    ensure=None) -> List[str]:
    """First run: download ALL required models automatically (TTS, aligner, speech recognition, SAGE), with no "quality"
    button in between.  A Voxprint backup chosen as models source is restored first, copies in the chosen models folder /
    other programs are used as they are, and only what is still missing is downloaded (multi-connection, hash-checked).
    Returns the list of repositories that were downloaded."""
    ensure = ensure or md.ensure_model
    repos = repos if repos is not None else required_model_repos()
    _restore_backup_source(progress)
    todo = models_missing(repos)
    for repo in repos:   # copies of other programs: instant, only the "found, using it" message is shown
        if repo not in todo and not md.verify_local_model(md.local_dir_for(repo)):
            ensure(repo, lambda s, f, m: progress(Stage.MODEL, 0.0, m))
    # overall progress weighted by the approximate size, so the 0.5 GB SAGE does not count as much as the 4.5 GB TTS model
    sizes = [md.APPROX_SIZE_GB.get(r, 2.0) for r in todo]
    total = sum(sizes) or 1.0
    base = 0.0
    for repo, size in zip(todo, sizes):
        ensure(repo, lambda s, f, m, b=base, sz=size: progress(Stage.MODEL, min(1.0, (b + sz * float(f)) / total), m))
        base += size
    if ensure is md.ensure_model and sys.platform == "win32":
        from infra import assets   # system ffmpeg, else the pinned LGPL build (best effort, never blocks)

        assets.ensure_ffmpeg_tool(lambda f, m: progress(Stage.MODEL, 0.0, m))
    return todo
