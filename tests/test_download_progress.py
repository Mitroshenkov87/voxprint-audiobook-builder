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


def test_a_source_switch_keeps_total_counter_and_bar_steady(quick, monkeypatch):
    """Hugging Face delivers part of the model and stalls, ModelScope (which restarts the big file) finishes it: the line
    "N of TOTAL" keeps one total, N never decreases, the bar never goes back and the source name changes."""
    lines, fracs = [], []

    def progress(stage, f, msg):
        lines.append(msg)
        fracs.append(f)

    def hf(repo_id, local_dir, tqdm_class=None, **kw):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_bytes(FILES["config.json"])
        with open(d / "model.safetensors.incomplete", "wb") as f:
            for _ in range(3):
                f.write(b"W" * 100_000)
                f.flush()
                time.sleep(0.2)
        for _ in range(200):                                          # then no data any more
            time.sleep(0.02)
            if tqdm_class is not None:
                tqdm_class(total=1, unit="B", disable=True).update(0)

    def mirror(repo, dest, report, expected, on_total=None):
        order.append("ms")
        (Path(dest) / "model.safetensors.incomplete").unlink(missing_ok=True)      # the mirror downloads the file from scratch
        for n, b in FILES.items():
            t = Path(dest).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(b), 100_000):
                with open(t, "ab" if i else "wb") as f:
                    f.write(b[i:i + 100_000])
                time.sleep(0.12)
                report(0.5)                                           # a source's own fraction is not trusted when the total is known
        return {n: len(b) for n, b in FILES.items()}

    order = []
    sizes = lambda repo, rev, pats: {n: len(b) for n, b in FILES.items()}      # noqa: E731 - the "API" answer
    got = md.ensure_model(REPO, progress, snapshot_download=hf, revision=SHA, hf_probe=lambda r: True,
                          mirror_download=mirror, get_remote_sizes=sizes)
    assert md.verify_local_model(got) and order == ["ms"]
    detail = [l for l in lines if " of " in l and "%" in l]
    assert detail
    pairs = []
    for l in detail:
        m = re.search(r"([\d.]+) (KB|MB|GB) of ([\d.]+) (KB|MB|GB)", l)
        pairs.append((_kb(m.group(1), m.group(2)), _kb(m.group(3), m.group(4))))
    assert len({round(t) for _, t in pairs}) == 1                      # ONE total for the whole download
    dones = [d for d, _ in pairs]
    assert dones == sorted(dones)                                      # never backwards, also across the source switch
    assert fracs == sorted(fracs)
    assert any("Hugging Face" in l for l in detail) and any("ModelScope" in l for l in detail)


def test_the_total_is_known_before_the_first_byte(quick):
    seen = []
    mirrors = {}

    def hf(repo_id, local_dir, **kw):
        for n, b in FILES.items():                                    # slow: the watchdog tick shows the line meanwhile
            t = Path(local_dir).joinpath(*n.split("/"))
            t.parent.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(b), 150_000):
                with open(t, "ab" if i else "wb") as f:
                    f.write(b[i:i + 150_000])
                time.sleep(0.15)

    md.ensure_model(REPO, lambda s, f, m: seen.append(m), snapshot_download=hf, revision=SHA, hf_probe=lambda r: True,
                    get_remote_sizes=lambda *a: {n: len(b) for n, b in FILES.items()})
    assert any(" of " in m for m in seen) and not any("of 0" in m for m in seen)
