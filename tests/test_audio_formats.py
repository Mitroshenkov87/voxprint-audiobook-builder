"""m4a/aac must load without ffprobe (imageio-ffmpeg ships only ffmpeg; pydub needs ffprobe for non-wav -> [WinError 2])."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

from core import audio_utils as au


def _ffmpeg():
    exe = au.ensure_ffmpeg()
    if not exe:
        pytest.skip("no ffmpeg available")
    return exe


@pytest.mark.parametrize("ext,codec", [("m4a", ["-c:a", "aac"]), ("mp3", []), ("wav", [])])
def test_load_audio_decodes_via_ffmpeg_without_ffprobe(tmp_path, monkeypatch, ext, codec):
    exe = _ffmpeg()
    sr = 22050
    t = np.arange(sr * 2) / sr
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    out = tmp_path / f"tone.{ext}"
    if ext == "wav":
        out = wav
    else:
        r = subprocess.run([exe, "-v", "error", "-y", "-i", str(wav), *codec, str(out)], capture_output=True)
        if r.returncode != 0:
            pytest.skip("ffmpeg build lacks the encoder")

    def no_pydub(path):          # pydub path is the one that needs ffprobe
        raise FileNotFoundError("[WinError 2] ffprobe")

    monkeypatch.setattr(au, "_read_with_pydub", no_pydub)
    x, got_sr = au.load_audio(out, 16000)
    assert got_sr == 16000 and abs(len(x) / 16000 - 2.0) < 0.15 and float(np.abs(x).max()) > 0.1
    multi = au.load_audio_multi(out, (16000, 24000))
    assert set(multi) == {16000, 24000}
