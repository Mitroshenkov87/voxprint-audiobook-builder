"""Install and repair the shared suite resources during setup and on later launches.

``--install-suite`` (the installer runs this, including ``/VERYSILENT``) installs the Python
runtime modules, the torch/CUDA wheels, ffmpeg and the required base models into
:mod:`infra.suite_shared`, with progress. A later launch and ``--selftest`` re-check the same
tree and repair what is missing or empty. ``VOXPRINT_SKIP_SUITE=1`` (tests) only reports.

The NVIDIA driver is the one part this cannot install. A driver older than the cu130 wheels
need is reported with a download link and the install stops.

    python main.py --install-suite --status C:\\status.txt

The status file is four lines (state, permille, message, heartbeat), the same shape the
installer already reads for the shell download.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from core.i18n import tr
from infra import assets, modules, suite_shared
from infra.gpu_requirement import NVIDIA_DRIVER_URL, check_gpu, driver_too_old
from workers.pipeline_runner import prefetch_models, required_model_repos

APP_RESOURCES = ("runtime", "models", "tools")
DISK_MARGIN = 2 * 1024 ** 3
SKIP_ENV = "VOXPRINT_SKIP_SUITE"


def skipped(environ: Optional[dict] = None) -> bool:
    """True when suite downloads must not run (the unit tests set this)."""
    env = os.environ if environ is None else environ
    return str(env.get(SKIP_ENV, "")).strip().lower() not in ("", "0", "false", "no")


def should_repair() -> bool:
    """A frozen install repairs on launch. A source checkout only reports, unless asked."""
    if skipped():
        return False
    if str(os.environ.get("VOXPRINT_SUITE_REPAIR", "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    return bool(getattr(sys, "frozen", False))


def free_bytes(folder: Path) -> int:
    """Free bytes on the volume that holds ``folder`` (the folder is created if needed)."""
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder if folder.exists() else folder.anchor
    return int(shutil.disk_usage(probe).free)


def inspect_shared() -> dict:
    """Local check. Does not download. ``missing`` names resources that are absent or empty."""
    missing: List[str] = []
    for name in APP_RESOURCES:
        if not suite_shared.populated(suite_shared.root() / name):
            missing.append(name)
    return {
        "ok": not missing,
        "missing": missing,
        "root": str(suite_shared.root()),
        "skipped": skipped(),
    }


def driver_message(*, use_torch: bool = False) -> str:
    """A sentence plus the NVIDIA download link when the driver is too old for the CUDA wheels.

    Empty when there is no driver reading, the driver is new enough, or the internal
    no-GPU override is set.
    """
    report = check_gpu(use_torch=use_torch)
    return driver_too_old(report) or ""


def _default_modules(progress: Callable[[float, str], None]) -> None:
    if not modules.is_thin():
        progress(1.0, tr("suite.bundled"))
        return
    modules.install(progress=lambda fraction, message="": progress(float(fraction), message or "Runtime"))


def _default_ffmpeg(progress: Callable[[float, str], None]) -> None:
    assets.ensure_ffmpeg_tool(lambda fraction, message="": progress(float(fraction), message or "ffmpeg"),
                              root=suite_shared.resource_dir("tools"))


def _default_models(progress: Callable[[float, str], None]) -> None:
    prefetch_models(lambda _stage, fraction, message="": progress(float(fraction), message or "Models"),
                    repos=required_model_repos(), everything=False)


def install_suite(progress: Callable[[float, str], None] = lambda _f, _m: None, *,
                  modules_fn: Optional[Callable[[Callable[[float, str], None]], None]] = None,
                  ffmpeg_fn: Optional[Callable[[Callable[[float, str], None]], None]] = None,
                  models_fn: Optional[Callable[[Callable[[float, str], None]], None]] = None,
                  driver_fn: Optional[Callable[[], str]] = None,
                  free_fn: Optional[Callable[[Path], int]] = None,
                  need_bytes: int = 0,
                  force: bool = False) -> dict:
    """Install whatever of the shared runtime, ffmpeg and required models is still missing.

    Returns ``{ok, exit_code, message, missing}``. ``exit_code`` 7 is a driver that is too old.
    """
    if skipped() and not force:
        progress(1.0, tr("suite.skip"))
        report = inspect_shared()
        report.update(ok=True, exit_code=0, message="skipped")
        return report
    say_driver = driver_fn or driver_message
    problem = say_driver()
    if problem:
        progress(0.0, problem)
        return {"ok": False, "exit_code": 7, "message": problem, "missing": [],
                "url": NVIDIA_DRIVER_URL}
    target = suite_shared.root()
    free = (free_fn or free_bytes)(target)
    required = max(0, int(need_bytes)) + DISK_MARGIN
    if need_bytes and free < required:
        message = tr("suite.disk", need=str(required // (1024 ** 3)), free=str(free // (1024 ** 3)))
        progress(0.0, message)
        return {"ok": False, "exit_code": 1, "message": message, "missing": list(APP_RESOURCES)}
    steps = (
        (0.0, 0.5, tr("suite.runtime"), modules_fn or _default_modules),
        (0.5, 0.6, tr("suite.ffmpeg"), ffmpeg_fn or _default_ffmpeg),
        (0.6, 1.0, tr("suite.models"), models_fn or _default_models),
    )
    for start, end, label, fn in steps:
        span = end - start

        def report(fraction: float, message: str = "", s=start, w=span, title=label) -> None:
            progress(s + w * max(0.0, min(1.0, float(fraction))), message or title)

        progress(start, label)
        fn(report)
    suite_shared.register(suite_shared.APP_ID, APP_RESOURCES)
    progress(1.0, tr("suite.done"))
    report = inspect_shared()
    report.update(ok=True, exit_code=0, message="installed")
    return report


def launch_check(progress: Callable[[float, str], None] = lambda _f, _m: None, *, repair: bool = False) -> dict:
    """Re-check the shared tree. Repair only when ``repair`` is true (a frozen app, or an explicit request)."""
    report = inspect_shared()
    if repair and report["missing"] and not skipped():
        fixed = install_suite(progress, force=True)
        fixed["repaired"] = True
        return fixed
    report.update(exit_code=0 if report["ok"] or skipped() else 1, message="ok" if report["ok"] else "missing",
                  repaired=False)
    return report


def write_status(path: Path, state: str, permille: int, message: str, beat: int) -> None:
    """Four lines the installer polls: state, permille, message, heartbeat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{state}\n{int(permille)}\n{message}\n{int(beat)}\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def cli(argv: Sequence[str]) -> int:
    """``--install-suite [--status FILE]`` or ``--unregister-suite``. Prints a line on stderr when it fails."""
    args = list(argv)
    if "--unregister-suite" in args:
        unused = suite_shared.unregister(suite_shared.APP_ID)
        suite_shared.delete_resources(unused)
        return 0
    status: Optional[Path] = None
    if "--status" in args:
        index = args.index("--status")
        if index + 1 < len(args):
            status = Path(args[index + 1])
    beat = {"n": 0}

    def progress(fraction: float, message: str) -> None:
        beat["n"] += 1
        if status is not None:
            write_status(status, "running", int(max(0.0, min(1.0, fraction)) * 1000), message, beat["n"])

    if status is not None:
        progress(0.0, tr("suite.checking"))
    try:
        result = install_suite(progress, force="--force" in args)
    except Exception as exc:  # noqa: BLE001 - the installer shows this string
        result = {"ok": False, "exit_code": 1, "message": str(exc)}
    code = int(result.get("exit_code", 1 if not result.get("ok") else 0))
    if status is not None:
        state = "done" if code == 0 else "error"
        write_status(status, state, 1000 if code == 0 else 0, str(result.get("message") or ""), beat["n"] + 1)
    if code != 0:
        print(str(result.get("message") or "suite install failed"), file=sys.stderr, flush=True)
    return code
