"""Start-up noise: the SoX shell probe and the optional-accelerator (Triton) log records are dropped."""
import io
import logging
import os

from infra import diagnostics


def test_sox_probe_is_answered_without_a_shell_when_sox_is_missing(monkeypatch):
    calls = []

    def fake_popen(cmd, mode="r", buffering=-1):
        calls.append(cmd)
        return io.StringIO("real\n")

    monkeypatch.setattr(os, "popen", fake_popen)
    assert diagnostics.skip_missing_sox_probe(which=lambda _n: None) is True
    stream = os.popen("sox -h")
    assert stream.readlines() == [] and calls == []
    stream.close()
    assert os.popen("echo hi").read() == "real\n" and calls == ["echo hi"]
    assert diagnostics.skip_missing_sox_probe(which=lambda _n: None) is True     # idempotent, no double wrapping
    assert os.popen("echo again").read() == "real\n"


def test_sox_probe_is_left_alone_when_sox_is_installed(monkeypatch):
    marker = object()
    monkeypatch.setattr(os, "popen", marker)
    assert diagnostics.skip_missing_sox_probe(which=lambda _n: "/usr/bin/sox") is False
    assert os.popen is marker


def test_triton_and_sox_log_records_are_dropped(caplog):
    diagnostics.quiet_known_log_noise()
    diagnostics.quiet_known_log_noise()                    # a second call adds no second filter
    flop = logging.getLogger("torch.utils.flop_counter")
    assert sum(isinstance(f, diagnostics._PatternFilter) for f in flop.filters) == 1
    with caplog.at_level(logging.WARNING):
        flop.warning("triton not found; flop counting will not work for triton kernels")
        logging.getLogger("sox").warning("SoX could not be found!")
        flop.warning("something else")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["something else"]
