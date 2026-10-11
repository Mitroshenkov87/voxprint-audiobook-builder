"""The download progress line: stable total, counter that never goes backwards, steady remaining time, current source."""
import re
import threading
import time
from pathlib import Path

import pytest

from infra import download_watch as dw
from infra import model_downloader as md
from infra import modelscope_mirror as ms

REPO, SHA = "Org/Tiny-1.7B", "a" * 40
FILES = {"config.json": b'{"model_type": "x"}', "model.safetensors": b"W" * 600_000, "speech_tokenizer/config.json": b"{}"}
TOTAL = sum(len(b) for b in FILES.values())


def tr_stub(key, **kw):
    if key == "progress.detail":
        return f"{kw['pct']}% {kw['done']} of {kw['total']} {kw['speed']} from {kw['source']}"
    if key == "progress.eta":
        return f" ETA {kw['eta']}"
    return key + str(sorted(kw.items()))


def test_the_counter_never_goes_backwards_and_the_total_is_fixed():
    t = [0.0]
    m = dw.Meter("Hugging Face", clock=lambda: t[0])
    m.set_total(1000 * 1024)
    seen = []
    for done in (0, 200 * 1024, 300 * 1024, 120 * 1024, 400 * 1024, 150 * 1024):    # a retry / source switch re-reads a file
        t[0] += 1
        m.sample(done)
        seen.append((m.done, m.total))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert {tot for _, tot in seen} == {1000 * 1024}
    m.set_total(5)                                                  # a smaller guess never replaces the known size
    assert m.total == 1000 * 1024
    m.set_source("ModelScope")
    assert "ModelScope" in m.text(tr_stub, "M", 1) and m.done == 400 * 1024


def test_without_a_known_total_no_total_is_shown():
    m = dw.Meter("X")
    m.sample(5)
    m.sample(4096)
    assert m.text(tr_stub, "M", 0).startswith("progress.detail_unknown")


def test_the_remaining_time_comes_from_the_smoothed_speed_and_does_not_flicker():
    t = [0.0]
    m = dw.Meter("GitHub", clock=lambda: t[0])
    m.set_total(1000 * 1024 ** 2)
    done, etas = 0, []
    m.sample(0)
    for i in range(1, 61):
        t[0] += 1
        done += (10 if i % 7 else 40) * 1024 ** 2 if i % 7 else 1 * 1024 ** 2     # bursts and near-stalls around ~10 MB/s
        m.sample(done)
        e = m.eta()
        if i < 6:
            assert e is None                                           # too early to tell
        elif e is not None:
            etas.append(e)
    assert etas
    steps = [abs(b - a) for a, b in zip(etas, etas[1:])]
    assert max(steps) < 30                                             # no wild swings second to second
    line = m.text(tr_stub, "M", 1)
    assert "ETA" in line and "GitHub" in line


def test_duration_format_is_coarse():
    assert dw.fmt_duration(3) == "5 s" and dw.fmt_duration(47) == "45 s"
    assert dw.fmt_duration(190) == "3 min 10 s" and dw.fmt_duration(600) == "10 min"
    assert dw.fmt_duration(4500) == "1 h 15 min"


# ------------------------------------------------------------------------------------------------ a whole download
def _numbers(line):
    m = re.search(r"([\d.]+) (KB|MB|GB) .* ?из ([\d.]+) (KB|MB|GB)|([\d.]+) (KB|MB|GB) of ([\d.]+) (KB|MB|GB)", line)
    return m


@pytest.fixture
def quick(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "STALL_SECONDS", 0.5)
    monkeypatch.setattr(dw, "POLL", 0.05)
    monkeypatch.setattr(dw, "ABORT_GRACE", 1.0)
    monkeypatch.setenv("VOXPRINT_NO_EXTERNAL_MODELS", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.setenv("VOXPRINT_LANG", "en")
    monkeypatch.delenv("VOXPRINT_NO_MIRROR", raising=False)
    monkeypatch.chdir(tmp_path)


def _kb(v, unit):
    return float(v) * {"KB": 1, "MB": 1024, "GB": 1024 ** 2}[unit]


