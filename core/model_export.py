"""The universal (merged) model: the base Qwen3-TTS plus the LoRA adapter, merged with PEFT ``merge_and_unload``.

Runs *only* when the user presses the button (~4 GB on disk).  The output format is that of the official
``finetuning/sft_12hz.py`` from QwenLM/Qwen3-TTS - a plain Hugging Face folder that loads with
``Qwen3TTSModel.from_pretrained``:

* ``model.safetensors`` - bf16 weights with the adapter merged in; the ``speaker_encoder`` is removed and the speaker
  embedding is written into ``talker.model.codec_embedding.weight[3000]``;
* ``config.json`` - ``tts_model_type="custom_voice"``, ``talker_config.spk_id={<name>: 3000}``,
  ``spk_is_dialect={<name>: false}``;
* ``speech_tokenizer/``, tokenizer / preprocessor / generation config - copied from the base model;
* ``ref_sample.wav``, ``ref_text.txt``, ``speaker_embedding.safetensors``, ``voxprint_voice.json``, ``USAGE.txt`` - informational.

The voice is used like this: ``model.generate_custom_voice(text=..., language="Russian", speaker="<name>")``.

During training the speaker embedding came from ``ref.wav`` (see ``teacher_forcing``), so writing it into
``codec_embedding[3000]`` reproduces the training conditions.

TODO-needs-GPU-test (originally): real 1.7B weights.  Verified on a tiny model built from the real qwen-tts classes:
the merged and the "adapter" model give equivalent talker output, and the folder loads back with ``from_pretrained``.
Later also exercised with the real 1.7B model on an RTX 4090.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from core.errors import ExportError
from core.events import CancelToken, ProgressCallback, Stage, noop_progress
from core.i18n import tr
from infra import model_downloader as md

log = logging.getLogger("voxprint.export")

SPK_ID = 3000
MERGED_DIRNAME = "merged_model"
_SKIP_ROOT = (".safetensors", ".bin", ".pt")
_SKIP_NAMES = {".revision", ".cache", ".gitattributes", "README.md"}


def speaker_name(voice_name: str) -> str:
    """Speaker name for ``spk_id``: lower-case, only word characters and ``-`` (Qwen3-TTS compares names in lower case)."""
    s = re.sub(r"[^\w\-]+", "_", voice_name.strip().lower(), flags=re.UNICODE).strip("_")
    return s or "voice"


def read_adapter_meta(adapter_dir: Path) -> Dict[str, Any]:
    """Read ``training_meta.json`` of an adapter folder; raises ``ExportError`` if the adapter files are missing."""
    f = Path(adapter_dir) / "training_meta.json"
    if not (Path(adapter_dir) / "adapter_model.safetensors").exists() or not f.exists():
        raise ExportError(tr("err.export_no_adapter"))
    return json.loads(f.read_text(encoding="utf-8"))


def required_free_gb(adapter_dir: Path) -> Tuple[float, str]:
    """``(free disk space needed in GB, base model repo)`` - the base model's approximate size plus 10 %."""
    repo = read_adapter_meta(adapter_dir).get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    return md.APPROX_SIZE_GB.get(repo, 4.5) * 1.1, repo


def check_disk_space(out_parent: Path, need_gb: float) -> None:
    """Raise ``ExportError`` with a friendly message if ``out_parent`` has less than ``need_gb`` GB free."""
    out_parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_parent).free / 1024 ** 3
    if free < need_gb:
        raise ExportError(tr("err.export_disk", need=f"{need_gb:.1f}", free=f"{free:.1f}"))


def adapter_scale_of(adapter_dir: Path) -> float:
    """Strength the voice speaks with (``adapter_scale`` in the folder's voice.json; 1.0 if there is none)."""
    from core import adapter_strength

    try:
        info = json.loads((Path(adapter_dir) / "voice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    return adapter_strength.clamp(info.get("adapter_scale") if isinstance(info, dict) else None,
                                  adapter_strength.LEGACY_SCALE)


def merge_adapter_into_model(hf_model: Any, adapter_dir: Path, spk_id: int = SPK_ID, adapter_scale: Optional[float] = None):
    """Merge the adapter into the talker at ``adapter_scale`` (default: the voice's own strength, see :func:`adapter_scale_of`),
    so the universal model sounds like the voice in Voxprint.

    Returns ``(state_dict without speaker_encoder, speaker_embedding [1, D])``; the embedding is already stored in the
    state dict at row ``spk_id`` of ``codec_embedding``.  Tensors sharing memory are cloned because safetensors rejects them.
    """
    import torch
    from peft import PeftModel

    from core.lora_trainer import speaker_embedding_from_ref

    adapter_dir = Path(adapter_dir)
    hf_model.eval()
    from core import adapter_strength

    peft_talker = PeftModel.from_pretrained(hf_model.talker, str(adapter_dir))
    scale = adapter_scale_of(adapter_dir) if adapter_scale is None else adapter_strength.clamp(adapter_scale, 1.0)
    adapter_strength.apply(peft_talker, scale)            # merge_and_unload bakes in the current scaling
    hf_model.talker = peft_talker.merge_and_unload()
    dtype = next(hf_model.talker.parameters()).dtype
    spk = speaker_embedding_from_ref(hf_model, str(adapter_dir / "ref_sample.wav"), torch.device("cpu"), dtype)
    state: Dict[str, Any] = {}
    seen: set = set()
    for k, v in hf_model.state_dict().items():
        if k.startswith("speaker_encoder"):
            continue
        t = v.detach().cpu().contiguous()
        ptr = t.data_ptr()
        if ptr in seen:          # safetensors refuses shared memory between different keys
            t = t.clone()
        seen.add(ptr)
        state[k] = t
    key = "talker.model.codec_embedding.weight"
    if spk_id >= state[key].shape[0]:
        raise ExportError(tr("err.export_failed"), details=f"codec_embedding rows {state[key].shape[0]} <= {spk_id}")
    state[key][spk_id] = spk[0].to(state[key].dtype)
    return state, spk


def _copy_base_files(base_dir: Path, dst: Path) -> None:
    """Copy the base model folder into ``dst``, skipping weights at the root (we write our own) and cache/metadata files."""
    base_dir = Path(base_dir)

    def ignore(d: str, names):
        root = Path(d) == base_dir
        return [n for n in names if n in _SKIP_NAMES or (root and n.endswith(_SKIP_ROOT))]

    shutil.copytree(base_dir, dst, ignore=ignore, dirs_exist_ok=True)


USAGE = """Voxprint - merged Qwen3-TTS model (custom_voice) / универсальная модель Qwen3-TTS
=====================================================================================
Speaker / голос: {speaker}

    import torch
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(r"<path to this folder>", dtype=torch.bfloat16)
    wavs, sr = model.generate_custom_voice(text="...", language="{language}", speaker="{speaker}")

Reference sample for voice-clone apps / образец для приложений с клонированием: ref_sample.wav + ref_text.txt
Use this voice only with its owner's permission and within its licence. / Используйте этот голос только с разрешения его владельца и в рамках лицензии.
"""


def export_merged_model(adapter_dir, out_dir, base_dir=None, progress: ProgressCallback = noop_progress,
                        cancel: Optional[CancelToken] = None, voice_name: Optional[str] = None,
                        load_model: Optional[Callable[[Path], Any]] = None, spk_id: int = SPK_ID,
                        skip_disk_check: bool = False) -> Path:
    """Build the ``merged_model`` folder from an adapter folder and return its path.

    Writes into ``<out_dir>.partial`` first and renames at the end, so a failure or cancel never leaves a half-written
    model behind.  ``load_model`` / ``base_dir`` / ``skip_disk_check`` are injection points for tests.
    """
    import torch
    from safetensors.torch import save_file

    cancel = cancel or CancelToken()
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    meta = read_adapter_meta(adapter_dir)
    need, repo = required_free_gb(adapter_dir)
    if not skip_disk_check:
        check_disk_space(out_dir.parent, need)
    if base_dir is None:
        progress(Stage.MODEL, 0.0, tr("progress.model_prepare"))
        base_dir = md.ensure_model(repo, lambda s, f, m: progress(Stage.MODEL, f, m))
    base_dir = Path(base_dir)
    cancel.check()

    progress(Stage.MODEL, 0.9, tr("progress.model_loading"))
    if load_model is None:
        def load_model(d: Path):  # noqa: F811
            from qwen_tts import Qwen3TTSModel  # type: ignore

            return Qwen3TTSModel.from_pretrained(str(d), dtype=torch.bfloat16, attn_implementation="eager").model
    hf_model = load_model(base_dir)
    progress(Stage.MODEL, 1.0, tr("progress.model_ready"))
    cancel.check()

    name = speaker_name(voice_name or adapter_dir.name)
    progress(Stage.SAVE, 0.1, tr("progress.merging"))
    scale = adapter_scale_of(adapter_dir)
    state, spk = merge_adapter_into_model(hf_model, adapter_dir, spk_id, scale)
    del hf_model
    cancel.check()

    partial = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        progress(Stage.SAVE, 0.4, tr("progress.export_writing"))
        _copy_base_files(base_dir, partial)
        cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        cfg["tts_model_type"] = "custom_voice"
        tc = cfg.setdefault("talker_config", {})
        tc["spk_id"] = {name: spk_id}
        tc["spk_is_dialect"] = {name: False}
        (partial / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        save_file(state, str(partial / "model.safetensors"), metadata={"format": "pt"})
        del state
        language = str(meta.get("language", "russian"))
        shutil.copy2(adapter_dir / "ref_sample.wav", partial / "ref_sample.wav")
        (partial / "ref_text.txt").write_text(str(meta.get("ref_sample_text", "")).strip() + "\n", encoding="utf-8")
        save_file({"speaker_embedding": spk.detach().float().cpu().contiguous()},
                  str(partial / "speaker_embedding.safetensors"), metadata={"format": "pt"})
        (partial / "voxprint_voice.json").write_text(json.dumps({
            "app": "Voxprint", "speaker": name, "spk_id": spk_id, "base_model": repo, "language": language,
            "ref_sample": "ref_sample.wav", "ref_text": meta.get("ref_sample_text", ""),
            "merged_dtype": "bfloat16", "tts_model_type": "custom_voice", "adapter_scale": scale},
            ensure_ascii=False, indent=2),
            encoding="utf-8")
        (partial / "USAGE.txt").write_text(USAGE.format(speaker=name, language=language.capitalize()), encoding="utf-8")
        cancel.check()
        shutil.rmtree(out_dir, ignore_errors=True)
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    progress(Stage.SAVE, 1.0, tr("progress.export_done"))
    return out_dir
