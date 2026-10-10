"""Windows shared-memory spill guard: baseline after load, 256 MB of growth, diagnostic text."""

import sys

from core import sysmem_spill
from infra import diagnostics, platform_win

MB = 1024 * 1024


def test_growth_over_256_mb_trips_once_per_step():
    monitor = sysmem_spill.Monitor(sample=lambda: None)
    monitor.note(None)
    assert monitor.baseline is None and not monitor.tripped()
    monitor.note(1_000_000)
    assert monitor.baseline == 1_000_000
    monitor.note(1_000_000 + 200 * MB)
    assert not monitor.tripped()
    monitor.note(1_000_000 + 300 * MB)
    assert monitor.tripped()
    monitor.mark()
    assert not monitor.tripped()
    monitor.note(1_000_000 + 300 * MB + sysmem_spill.SPILL_BYTES + 1)
    assert monitor.tripped()


def test_start_after_load_is_inert_off_windows():
    if sys.platform == "win32":
        return
    monitor = sysmem_spill.start_after_load()
    try:
        assert monitor._thread is None
        assert not monitor.tripped()
    finally:
        monitor.stop()


def test_typeperf_csv_sums_shared_usage_columns():
    text = (
        '"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_9_a)\\Shared Usage",'
        '"\\\\PC\\GPU Process Memory(pid_9_b)\\Shared Usage"\n'
        '"10/10/2026 04:23:01.000","1048576.000000","2097152.000000"\n'
        "Exiting, please wait...\n"
    )
    assert platform_win.parse_typeperf_csv(text) == 1048576 + 2097152
    assert platform_win.parse_typeperf_csv("Error: no counter") is None
    assert platform_win.parse_typeperf_csv("") is None


def test_shared_usage_probe_is_unavailable_off_windows():
    if sys.platform == "win32":
        return
    assert platform_win.gpu_shared_usage_bytes(1) is None
    assert platform_win.process_exists(1) is False


def test_diagnostics_explain_the_sysmem_fallback_policy():
    info = diagnostics.gpu_info(import_torch=False)
    assert "Prefer No Sysmem Fallback" in info["sysmem_fallback"]
    assert "CUDA - Sysmem Fallback Policy" in info["sysmem_fallback"]
    text = "\n".join(diagnostics.summary_lines({"app_version": "0.2.4", "os": "Linux", "python": "3.12", "gpu": info}))
    assert "Prefer No Sysmem Fallback" in text
    assert sysmem_spill.DIAGNOSTIC_NOTE in text
