"""OpenVoice V2 tone-colour converter: the direct re-voice model.

Code and weights are MIT (commercial use allowed). The converter changes timbre and leaves the source spectrogram's
timing in place, which is what "keep my timing and intonation" asks for. Seed-VC was not used: its code and weights are
GPL-3.0. See :doc:`docs/REVOICE.md`.

A newer model replaces this class and is selected in :func:`core.voice_convert.make_converter`; nothing else imports it.
Torch and the vendored ``openvoice`` package are imported only when a conversion actually starts (tests never load them).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from core import asr as asr_mod
from core import audio_utils as au
from core.errors import DatasetMakerError
from core.events import CancelToken
from core.i18n import tr
from core.voice_convert import fit_length
from infra import vc_model

log = logging.getLogger("voxprint.revoice")

#: Pieces no longer than this go through the converter (a whole chapter in one spectrogram is a lot of VRAM).
PIECE_S = 20.0
#: Official conversion temperature (OpenVoice ``ToneColorConverter.convert``).
TAU = 0.3


class _H:
    """Attribute access for the checkpoint's JSON config (``hps.data.sampling_rate``)."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, _H(value) if isinstance(value, dict) else value)


def _import_openvoice() -> Any:
    """Import the vendored package. It lives in ``third_party`` and keeps the name ``openvoice``."""
    root = Path(__file__).resolve().parents[1] / "third_party"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from openvoice.models import SynthesizerTrn

    return SynthesizerTrn


def _spectrogram(y: Any, n_fft: int, hop: int, win: int) -> Any:
    """Magnitude spectrogram, same padding and window as OpenVoice ``spectrogram_torch`` (complex STFT)."""
    import torch

    window = torch.hann_window(win, device=y.device, dtype=y.dtype)
    pad = int((n_fft - hop) / 2)
    padded = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(padded, n_fft, hop_length=hop, win_length=win, window=window, center=False,
                      pad_mode="reflect", normalized=False, return_complex=True)
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


class OpenVoiceConverter:
    """Zero-shot conversion from a reference clip. ``key`` is stable so a later model can sit beside it."""

    key = "openvoice-v2"

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = models_dir
        self._model: Any = None
        self._hps: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        folder = vc_model.model_dir(self.models_dir)
        if not vc_model.ready(self.models_dir):
            raise DatasetMakerError(tr("revoice.vc_missing"))
        import torch

        synthesizer = _import_openvoice()
        config = json.loads((folder / "converter" / "config.json").read_text(encoding="utf-8"))
        hps = _H(config)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = synthesizer(0, hps.data.filter_length // 2 + 1, n_speakers=int(hps.data.n_speakers), **config["model"])
        model = model.to(self._device)
        model.eval()
        # The checkpoint is the SHA-256-pinned converter file (checked on download, size checked by vc_model.ready), not
        # an arbitrary path. The safe tensor-only loader is tried first; the full unpickler is the fallback for a
        # training pickle that holds more than tensors.
        path = folder / "converter" / "checkpoint.pth"
        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        except Exception:  # noqa: BLE001 - pickle.UnpicklingError and friends: not a pure tensor archive
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)  # nosec B614 - pinned SHA-256 file
        model.load_state_dict(checkpoint["model"], strict=False)
        self._model, self._hps = model, hps
        log.info("OpenVoice V2 converter on %s", self._device)

    def unload(self) -> None:
        self._model = None
        self._hps = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - unloading must not hide the conversion result
            log.debug("cuda cache not cleared", exc_info=True)

    def _embed(self, audio: np.ndarray, sr: int) -> Any:
        """Tone-colour embedding ``[1, gin, 1]`` of one clip (OpenVoice ``extract_se``)."""
        import torch

        hps = self._hps
        rate = int(hps.data.sampling_rate)
        clip = au.resample(np.clip(audio.astype(np.float32), -1.0, 1.0), sr, rate)
        if clip.size < int(hps.data.filter_length):
            clip = np.pad(clip, (0, int(hps.data.filter_length) - clip.size))
        y = torch.FloatTensor(clip).to(self._device).unsqueeze(0)
        spec = _spectrogram(y, int(hps.data.filter_length), int(hps.data.hop_length), int(hps.data.win_length))
        with torch.no_grad():
            return self._model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)

    def _convert_piece(self, audio: np.ndarray, src_se: Any, tgt_se: Any) -> np.ndarray:
        """One piece already at the model's sample rate. The vocoder's length is fitted back to the piece."""
        import torch

        hps = self._hps
        y = torch.FloatTensor(np.clip(audio, -1.0, 1.0)).to(self._device).unsqueeze(0)
        spec = _spectrogram(y, int(hps.data.filter_length), int(hps.data.hop_length), int(hps.data.win_length))
        lengths = torch.LongTensor([spec.size(-1)]).to(self._device)
        with torch.no_grad():
            out = self._model.voice_conversion(spec, lengths, sid_src=src_se, sid_tgt=tgt_se, tau=TAU)[0][0, 0]
        return fit_length(out.detach().float().cpu().numpy(), len(audio))

    def convert(self, source: np.ndarray, source_sr: int, reference: np.ndarray, reference_sr: int,
                cancel: Optional[CancelToken] = None, progress: Optional[Callable[[float], None]] = None
                ) -> Tuple[np.ndarray, int]:
        """Convert ``source`` towards ``reference``. Duration matches ``source``; timbre comes from the reference."""
        cancel = cancel or CancelToken()
        self._load()
        rate = int(self._hps.data.sampling_rate)
        src = au.resample(np.asarray(source, dtype=np.float32), int(source_sr), rate)
        cancel.check()
        tgt_se = self._embed(np.asarray(reference, dtype=np.float32), int(reference_sr))
        src_se = self._embed(src, rate)
        pieces = asr_mod.split_at_pauses(src, rate, max_s=PIECE_S, min_s=2.0)
        parts: list = []
        for i, (a, b) in enumerate(pieces):
            cancel.check()
            piece = src[a:b]
            if au.voiced_seconds(piece, rate) < 0.3:
                parts.append(np.zeros(len(piece), dtype=np.float32))
            else:
                parts.append(self._convert_piece(piece, src_se, tgt_se))
            if progress is not None:
                progress((i + 1) / len(pieces))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), rate
