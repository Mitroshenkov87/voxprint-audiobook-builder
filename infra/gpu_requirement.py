"""Refuse to start unless this PC has an NVIDIA GPU of compute capability 8.9 or newer.

RTX 40-series (Ada) is 8.9; RTX 50-series (Blackwell) is higher. There is no CPU-only mode.

Detection prefers ``torch.cuda.get_device_capability`` when CUDA is usable, and otherwise reads
``nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader``.

``VOXPRINT_ALLOW_NO_GPU=1`` skips the refusal. It is internal: CI and ``tests/conftest.py`` set it
because those jobs have no GPU. It is not a user option and must not be documented for users.

Example::

    from infra.gpu_requirement import enforce, startup_requires_gpu
    if startup_requires_gpu(sys.argv):
        code = enforce(gui=True)
        if code:
            raise SystemExit(code)

PySide6 is imported only when a dialog is shown. PyTorch is imported only when CUDA can be queried.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from core.i18n import tr

#: Ada Lovelace and newer. RTX 4090 is 8.9; RTX 3090 is 8.6 and is refused.
MIN_COMPUTE: Tuple[int, int] = (8, 9)
MIN_COMPUTE_TEXT = "8.9"
#: Internal. Honoured only so CI and the unit tests can start without an RTX 40-series GPU.
ENV_ALLOW = "VOXPRINT_ALLOW_NO_GPU"
EXIT_GPU_REQUIRED = 7
_TRUE = {"1", "true", "yes", "on"}
_SKIP_FLAGS = {"--json", "--yes", "-y", "--dry-run"}
_HELP_FLAGS = {"--help", "-h", "--version", "-V"}
_LIGHT_COMMANDS = {"help", "status", "capabilities", "diag", "diagnostics"}
_BOOKKEEPING = {
    "--register-runtime-user", "--unregister-runtime-user",
    "--register-models-user", "--unregister-models-user",
    "--sync-suite-settings",
}
Runner = Callable[[Sequence[str]], Tuple[int, str]]


@dataclass(frozen=True)
class GpuDevice:
    """One NVIDIA GPU the probe could name."""

    name: str
    major: int
    minor: int
    driver: str = ""

    @property
    def compute_text(self) -> str:
        """``major.minor`` text for this device, for example ``8.9``."""
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class GpuReport:
    """Hardware result. ``ok`` ignores the internal override; ``overridden`` reports that switch."""

    ok: bool
    devices: Tuple[GpuDevice, ...]
    detected: str
    best: Optional[GpuDevice]
    source: str
    overridden: bool

    def to_dict(self) -> dict:
        """The requirement block printed by ``status``."""
        return {
            "ok": self.ok,
            "min_compute": MIN_COMPUTE_TEXT,
            "detected": self.detected,
            "compute_cap": self.best.compute_text if self.best else None,
            "source": self.source,
            "override": self.overridden,
        }


def allow_without_gpu(environ: Optional[dict] = None) -> bool:
    """True when the internal CI/test switch is set. ``1`` is the value the workflows use."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV_ALLOW, "")).strip().lower() in _TRUE


def meets(major: int, minor: int) -> bool:
    """True when ``major.minor`` is at least 8.9."""
    return (int(major), int(minor)) >= MIN_COMPUTE


def parse_nvidia_smi_query(text: str) -> Tuple[GpuDevice, ...]:
    """Parse ``name, compute_cap, driver_version`` CSV lines. A name may itself contain commas."""
    devices = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("name"):
            continue
        parts = [part.strip().strip('"') for part in line.split(",")]
        if len(parts) < 3:
            continue
        driver, cap = parts[-1], parts[-2]
        name = ",".join(parts[:-2]).strip().strip('"')
        if "." not in cap:
            continue
        major_s, minor_s = cap.split(".", 1)
        if not major_s.isdigit() or not minor_s.isdigit():
            continue
        devices.append(GpuDevice(name or "NVIDIA GPU", int(major_s), int(minor_s), driver))
    return tuple(devices)


def _default_run(args: Sequence[str]) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            list(args), capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return completed.returncode, completed.stdout or ""


def _torch_devices(torch_module) -> Optional[Tuple[GpuDevice, ...]]:
    """Devices from ``torch.cuda``, or None when CUDA cannot answer (fall back to nvidia-smi)."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
        count = int(cuda.device_count())
    except Exception:  # noqa: BLE001 - a broken CUDA build is not a report
        return None
    if count <= 0:
        return None
    devices = []
    for index in range(count):
        major, minor = cuda.get_device_capability(index)
        name = str(cuda.get_device_name(index) or "").strip() or "NVIDIA GPU"
        devices.append(GpuDevice(name, int(major), int(minor)))
    return tuple(devices)


def _load_torch():
    """Import PyTorch only to ask CUDA. Missing or CPU-only builds return None (nvidia-smi is next)."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional at this gate
        return None
    return torch


def _smi_devices(run: Runner, which: Callable[[str], Optional[str]]) -> Optional[Tuple[GpuDevice, ...]]:
    """Parsed GPUs, or None when ``nvidia-smi`` is not installed."""
    try:
        exe = which("nvidia-smi")
    except Exception:  # noqa: BLE001
        return None
    if not exe:
        return None
    try:
        code, text = run([exe, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    except Exception:  # noqa: BLE001
        return tuple()
    if code != 0:
        return tuple()
    return parse_nvidia_smi_query(text)


def _best(devices: Tuple[GpuDevice, ...]) -> Optional[GpuDevice]:
    best: Optional[GpuDevice] = None
    for device in devices:
        if best is None or (device.major, device.minor) > (best.major, best.minor):
            best = device
    return best


def check_gpu(*, use_torch: bool = True, torch_module=None, run: Optional[Runner] = None,
              which: Optional[Callable[[str], Optional[str]]] = None,
              environ: Optional[dict] = None) -> GpuReport:
    """The GPU this PC actually has. Does not exit. The override is recorded, not applied."""
    runner = run or _default_run
    finder = which or shutil.which
    source = "none"
    devices: Optional[Tuple[GpuDevice, ...]] = None
    module = torch_module
    if module is None and use_torch:
        module = _load_torch()
    if module is not None and use_torch:
        devices = _torch_devices(module)
        if devices is not None:
            source = "torch"
    if devices is None:
        devices = _smi_devices(runner, finder)
        if devices is not None:
            source = "nvidia-smi"
        else:
            devices = tuple()
            source = "none"
    best = _best(devices)
    ok = best is not None and meets(best.major, best.minor)
    detected = best.name if best is not None else "none"
    return GpuReport(ok, devices, detected, best, source, allow_without_gpu(environ))


def detected_label(report: GpuReport) -> str:
    """GPU name for the sentence, or the localized word for no card."""
    if report.best is None:
        return tr("gpu.none")
    return report.best.name


def requirement_message(detected: str) -> str:
    """The sentence the dialog, the installer and stderr share."""
    return tr("gpu.required", device=detected)


def show_dialog(message: str) -> None:
    """Modal error. PySide6 is imported here so a CLI refusal does not load Qt."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    _app = QApplication.instance() or QApplication([])
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Voxprint")
    box.setText(message)
    box.exec()


def enforce(*, gui: bool, dialog: Optional[Callable[[str], None]] = None, **detect_kwargs) -> int:
    """``0`` when the PC may run (or the internal override is set). Otherwise ``EXIT_GPU_REQUIRED``."""
    report = check_gpu(**detect_kwargs)
    if report.ok or report.overridden:
        return 0
    message = requirement_message(detected_label(report))
    if gui:
        (dialog or show_dialog)(message)
    else:
        print(message, file=sys.stderr)
    return EXIT_GPU_REQUIRED


def startup_requires_gpu(argv: Sequence[str], environ: Optional[dict] = None) -> bool:
    """False for ``--version``, help, ``status`` / ``diagnostics``, ``--dry-run``, and installer bookkeeping.

    ``argv`` is a full ``sys.argv`` (program name first). Other commands, including the GUI, are gated.
    ``--dry-run`` only validates arguments and files, so it runs on a machine with no RTX GPU.
    """
    if allow_without_gpu(environ):
        return False
    args = list(argv[1:] if argv else [])
    if any(arg == "--dry-run" for arg in args):
        return False
    if any(arg in _BOOKKEEPING for arg in args):
        return False
    if any(arg in _HELP_FLAGS for arg in args):
        return False
    tokens = [arg for arg in args if arg not in _SKIP_FLAGS]
    if tokens and tokens[0] in _LIGHT_COMMANDS:
        return False
    return True


def startup_is_gui(argv: Sequence[str]) -> bool:
    """True when nothing on the command line selects a subcommand or a maintenance flag."""
    args = [arg for arg in (argv[1:] if argv else []) if arg not in _SKIP_FLAGS]
    if not args:
        return True
    head = args[0]
    return not head.startswith("-") and head not in _LIGHT_COMMANDS and head not in {
        "narrate", "train", "voices", "diag", "status", "capabilities", "models", "revoice", "backup",
        "restore", "speakers", "check", "repair", "prepare", "translate", "settings", "bench", "diagnostics", "help",
    }
