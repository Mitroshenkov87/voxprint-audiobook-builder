"""Build 667 log: every command-line start logged ``"cuda_available": false`` (PyTorch is skipped there on purpose) while
the narration then ran on the GPU.  An unchecked GPU is now reported as unknown, not as "no CUDA"."""
import sys

from infra import diagnostics


def test_a_start_without_pytorch_says_not_checked(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    g = diagnostics.gpu_info(import_torch=False)
    assert g["cuda_available"] is None and "not checked" in g["cuda_note"]
    line = "\n".join(diagnostics.summary_lines({"app_version": "0.1.3", "os": "x", "python": "3.11", "gpu": g}))
    assert "CUDA available: not checked" in line


def test_status_payload_stays_boolean(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    g = diagnostics.gpu_info(import_torch=False)
    assert bool(g.get("cuda_available")) is False
