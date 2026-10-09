"""User-facing command line for headless servers and agents.

Wired from ``main.py`` so both ``python main.py narrate ...`` and a packaged
``Voxprint.exe narrate ...`` work.  Subcommands call the same Qt-free runners the
GUI uses (:func:`workers.narration_runner.run_narration`,
:func:`workers.pipeline_runner.run_task`, :class:`core.voice_library.VoiceLibrary`).

Every command is non-interactive.  ``--json`` prints progress as JSON lines and a
final result object.  ``--yes`` is accepted everywhere (nothing prompts).
Exit codes: 0 ok, 1 internal, 2 bad args, 3 input file, 4 missing model,
5 GPU/OOM, 6 cancelled.  See ``docs/AGENTS.md``.

Examples::

    python main.py narrate book.epub --voice my-voice --out ./audiobooks --format mp3 --json
    python main.py train ./clips --name Anna --type female --json
    python main.py status --json
    python main.py --version

The developer dataset CLI remains ``python -m core.cli`` (see ``core/cli.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Set

from core import audiobook_export as ex
from core import pauses as pz
from core import ordinals
from core import pace as pc
from core import revoice
from core import speakers as spk
from core.translate import detect_book_language
from core import workspace as ws
from core.appinfo import APP_BUILD, APP_CHANNEL, APP_CODENAME, APP_VERSION
from core.asr import make_default_asr
from core.book_parsers import load_book
from core.errors import BackupError, CancelledByUser, DatasetMakerError, OutOfMemoryError_
from core.events import CancelToken, Stage, overall_percent
from core.narration import NarrationOptions, NarrationProgress, PauseToken
from core.voice_info import VOICE_TYPES, normalize_voice_type
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import auto_repair
from infra import backup as backup_mod
from infra import denoise_tool, diagnostics, keep_awake, llm_tool, projects, quality_models, text_models, vc_model
from infra import model_downloader as md
from infra import modules as runtime_modules
from infra import paths as app_paths
from infra.asr_choice import LARGE as ASR_LARGE
from infra.asr_choice import SMALL as ASR_SMALL
from infra.asr_choice import preferred_repo
from infra.asr_choice import ready as asr_ready
from infra.model_downloader import ALIGNER_REPO
from infra.model_release import ReleaseError
from infra.vram_optimizer import MODEL_0_6B, MODEL_1_7B, detect_gpu, plan_training
from workers.backup_runner import collect as collect_backup
from workers.narration_runner import NarrationJob, format_eta, run_narration
from workers.pipeline_runner import KIND_LORA, TaskRequest, plan_for, run_task

try:
    from tools.build_number import build_number
except ImportError:  # a frozen build may omit tools/
    build_number = None  # type: ignore[assignment]

log = logging.getLogger("voxprint.cli")

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_BAD_ARGS = 2
EXIT_INPUT = 3
EXIT_MISSING = 4
EXIT_GPU = 5
EXIT_CANCELLED = 6

_INPUT_KINDS = {"audio_read", "text_read", "book", "voice", "mismatch"}
_MISSING_KINDS = {"download", "update"}

#: Short aliases accepted by ``--format`` (comma-separated or repeated).
FORMAT_ALIASES = {
    "opus": ex.FORMAT_OPUS_SINGLE,
    "opus_single": ex.FORMAT_OPUS_SINGLE,
    "mp3": ex.FORMAT_MP3_CHAPTERS,
    "mp3_chapters": ex.FORMAT_MP3_CHAPTERS,
    "mp3_single": ex.FORMAT_MP3_SINGLE,
    "m4b": ex.FORMAT_M4B,
    "m4b_opus": ex.FORMAT_M4B_OPUS,
    "opus_chapters": ex.FORMAT_OPUS_CHAPTERS,
    "flac": ex.FORMAT_FLAC_CHAPTERS,
    "flac_chapters": ex.FORMAT_FLAC_CHAPTERS,
    "wav": ex.FORMAT_WAV_CHAPTERS,
    "wav_chapters": ex.FORMAT_WAV_CHAPTERS,
}

USER_COMMANDS = frozenset({
    "narrate", "train", "voices", "diag", "status", "capabilities", "models", "revoice", "backup", "restore",
    "speakers", "check", "repair",
})

CORE_MODULE_IDS = (
    "tts", "tts-1.7b", "tts-0.6b",
    "aligner",
    "asr", "asr-0.6b", "asr-1.7b",
    "denoise", "llm", "dnsmos", "openvoice",
    "translate", "required",
)
ALIASES = {
    "deepfilternet": "denoise",
    "deepfilter": "denoise",
    "gemma": "llm",
    "llm-gemma4-12b": "llm",
    "translation": "translate",
    "mos": "dnsmos",
    "vc": "openvoice",
    "openvoice-v2": "openvoice",
}
_SKIP_FLAGS = {"--json", "--yes", "-y"}
_ENTRY_FLAGS = {"--version", "-V", "--help", "-h"}


class CliError(Exception):
    """A CLI failure with a stable exit code, a message and a fix the caller can run."""

    def __init__(self, code: int, message: str, *, hint: str = "", details: str = "") -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.hint = hint
        self.details = details


def hint_for(code: int, command: str) -> str:
    """One-line fix for a stable exit code."""
    if code == EXIT_GPU and command == "train":
        return "Retry on the CPU: add --force-cpu. Check the GPU with: voxprint status --json"
    if code == EXIT_GPU:
        return "Free video memory and retry. Check the GPU with: voxprint status --json"
    if code == EXIT_MISSING:
        return "Install it with: voxprint models download <module>   (ids: voxprint models list --json)"
    if code == EXIT_CANCELLED:
        return "Run the same command again. Narration resumes from finished chunks."
    if code == EXIT_INPUT:
        return "Check the path and the file type, then run the command again. See: voxprint --help"
    if code == EXIT_BAD_ARGS:
        return "See: voxprint --help"
    if code == EXIT_INTERNAL:
        return "Save a diagnostic report: voxprint diag --out voxprint-diagnostics.zip"
    return ""


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to a stable process exit code."""
    if isinstance(exc, CancelledByUser) or getattr(exc, "kind", "") == "cancelled":
        return EXIT_CANCELLED
    text = str(exc).lower()
    if isinstance(exc, OutOfMemoryError_) or getattr(exc, "kind", "") == "oom" or "out of memory" in text:
        return EXIT_GPU
    kind = getattr(exc, "kind", "")
    if kind in _MISSING_KINDS:
        return EXIT_MISSING
    if kind in _INPUT_KINDS:
        return EXIT_INPUT
    return EXIT_INTERNAL


def _coerce(exc: BaseException, command: str) -> CliError:
    if isinstance(exc, CliError):
        return exc
    if isinstance(exc, DatasetMakerError):
        code = exit_code_for(exc)
        return CliError(code, exc.user_message, hint=hint_for(code, command), details=exc.details or "")
    if isinstance(exc, OSError):
        return CliError(EXIT_INPUT, str(exc), hint="Check that the folder is writable and the path is valid.")
    code = exit_code_for(exc)
    return CliError(code, f"{type(exc).__name__}: {exc}", hint=hint_for(code, command))


def _read_build_json() -> dict:
    """``BUILD.json`` next to the program (codename, offset).  Empty when the file is absent."""
    candidates = [Path(__file__).resolve().parent / "BUILD.json", app_paths.resource_dir() / "BUILD.json"]
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def version_payload() -> dict:
    """Version, build number and codename.  Codename comes from ``BUILD.json`` when that file is present."""
    meta = _read_build_json()
    codename = str(meta.get("codename") or APP_CODENAME or "")
    build = int(APP_BUILD or 0)
    if not build and build_number is not None:
        try:
            build = int(build_number())
        except (OSError, ValueError, TypeError, KeyError):
            build = 0
    version = f"{APP_VERSION}-{APP_CHANNEL}" if APP_CHANNEL else str(APP_VERSION)
    return {"name": "Voxprint", "version": version, "build": int(build), "codename": codename}


def version_line(info: Optional[dict] = None) -> str:
    """``Voxprint 0.1.1-beta build 665 "Tikkun"``."""
    info = info or version_payload()
    line = f'{info["name"]} {info["version"]} build {info["build"]}'
    if info.get("codename"):
        line += f' "{info["codename"]}"'
    return line


def _write_stream(stream, text: str) -> None:
    """Write one line and flush it.

    A packaged ``Voxprint.exe`` is a windowed program: stdout exists for a caller that redirects it, and
    ``write`` raises when nothing is attached. The exit code must still be the command's code.
    """
    try:
        stream.write(text if str(text).endswith("\n") else str(text) + "\n")
        stream.flush()
    except (OSError, ValueError, AttributeError):
        pass


def _emit_progress(stage: str, percent: float, message: str, **extra) -> None:
    payload = {
        "type": "progress",
        "stage": stage,
        "percent": round(max(0.0, min(100.0, float(percent))), 1),
        "message": message,
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    _write_stream(sys.stdout, json.dumps(payload, ensure_ascii=False))


def _emit_result(*, ok: bool, command: str, exit_code: int, outputs, warnings, duration_s: float,
                 error, hint, details: Optional[str] = None, extra: Optional[dict] = None) -> None:
    payload = {
        "type": "result",
        "ok": ok,
        "command": command,
        "exit_code": exit_code,
        "outputs": [str(p) for p in outputs],
        "warnings": list(warnings or []),
        "duration_s": round(float(duration_s), 3),
        "error": error,
        "hint": hint,
    }
    if details:
        payload["details"] = details
    if extra:
        payload.update(extra)
    _write_stream(sys.stdout, json.dumps(payload, ensure_ascii=False))


def _error(exc: CliError, *, json_mode: bool, command: str, started: float) -> int:
    _write_stream(sys.stderr, f"ERROR: {exc.message}")
    if exc.details:
        _write_stream(sys.stderr, f"  details: {exc.details}")
    if exc.hint:
        _write_stream(sys.stderr, f"Fix: {exc.hint}")
    if json_mode:
        _emit_result(
            ok=False, command=command, exit_code=exc.code, outputs=[], warnings=[],
            duration_s=time.perf_counter() - started, error=exc.message, hint=exc.hint or None,
            details=exc.details or None,
        )
    return exc.code


def _ok(json_mode: bool, command: str, started: float, *, outputs, warnings=None, human=(), extra=None) -> int:
    warnings = list(warnings or [])
    if json_mode:
        _emit_result(
            ok=True, command=command, exit_code=EXIT_OK, outputs=outputs, warnings=warnings,
            duration_s=time.perf_counter() - started, error=None, hint=None, extra=extra,
        )
    else:
        for line in human:
            _write_stream(sys.stdout, line)
    return EXIT_OK


def _run(command: str, args, fn: Callable[[bool, float], int]) -> int:
    json_mode = bool(getattr(args, "json_output", False))
    started = time.perf_counter()
    try:
        return fn(json_mode, started)
    except CliError as exc:
        return _error(exc, json_mode=json_mode, command=command, started=started)
    except Exception as exc:
        if not isinstance(exc, DatasetMakerError):
            log.exception("%s failed", command)
        return _error(_coerce(exc, command), json_mode=json_mode, command=command, started=started)


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """``--json`` / ``--yes`` / ``--version`` on one parser.

    Subparsers use ``default=SUPPRESS`` so a flag set on a parent parser is not reset to false.
    """
    default = argparse.SUPPRESS if suppress else False
    parser.add_argument(
        "--json", dest="json_output", action="store_true", default=default,
        help="Print progress as JSON lines and a final JSON result (paths, duration, warnings)",
    )
    parser.add_argument(
        "--yes", "-y", dest="assume_yes", action="store_true", default=default,
        help="Accept confirmations. Every command is already non-interactive; this flag is always safe",
    )
    parser.add_argument(
        "--version", "-V", dest="show_version", action="store_true", default=default,
        help='Print version, build number and codename from BUILD.json (example: 0.1.1-beta build 665 "Tikkun")',
    )


def _text_ids() -> set[str]:
    return {m.key for m in text_models.REGISTRY if m.integrated}


def _known_module_ids() -> set[str]:
    return set(CORE_MODULE_IDS) | _text_ids()


def _module_help_epilog() -> str:
    ids = ", ".join(sorted(_known_module_ids()))
    aliases = ", ".join(f"{src} -> {dst}" for src, dst in sorted(ALIASES.items()))
    return (
        f"Module ids:\n  {ids}\n"
        f"Aliases:\n  {aliases}\n"
        "\nExamples:\n"
        "  voxprint models list --json\n"
        "  voxprint models download denoise --json\n"
        "  voxprint models download asr\n"
        "  voxprint models download opus-big-en-ru --yes\n"
        "  voxprint models download llm\n"
        "  voxprint models download translate\n"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree.  Each subcommand owns its examples (``voxprint <command> --help``)."""
    parent = argparse.ArgumentParser(add_help=False)
    _add_global_flags(parent, suppress=False)
    child = argparse.ArgumentParser(add_help=False)
    _add_global_flags(child, suppress=True)
    ap = argparse.ArgumentParser(
        prog="voxprint",
        parents=[parent],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Voxprint AI Audiobook Builder - headless CLI for scripts and agents. "
            "Every command is non-interactive. Guide: docs/AGENTS.md."
        ),
        epilog=(
            "Examples:\n"
            "  voxprint --version\n"
            "  voxprint status --json\n"
            "  voxprint voices list\n"
            "  voxprint narrate --help\n"
            "  voxprint train --help\n"
            "  voxprint models list --json\n"
            "  voxprint diag --out report.zip\n"
            "  voxprint backup --out E:\\ --json\n"
            "  voxprint restore --from E:\\ --json\n"
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    n = sub.add_parser(
        "narrate", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Narrate a book with a trained voice",
        description="Narrate a TXT, FB2, FB2.ZIP or EPUB book with a voice from the library.",
        epilog=(
            "Examples:\n"
            "  voxprint narrate book.epub --voice my-voice --out ./audiobooks\n"
            "  voxprint narrate book.fb2 --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json\n"
            "  voxprint narrate book.txt --voice my-voice --out ./audiobooks --pauses --ai-disclosure\n"
            "  voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speakers --out ./audiobooks --json\n"
            "  voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speaker-marks marks.txt --out ./audiobooks\n"
        ),
    )
    n.add_argument("book", help="Path to a TXT, FB2, FB2.ZIP or EPUB file")
    n.add_argument("--voice", required=True, metavar="ID_OR_NAME",
                   help="Voice library id or display name")
    n.add_argument("--out", required=True, type=Path, metavar="DIR",
                   help="Working / output folder (job lands in <DIR>/<book title>/)")
    n.add_argument("--format", action="append", default=[], dest="formats", metavar="NAME",
                   help="Output format (repeatable or comma-separated). "
                        f"Default: {ex.DEFAULT_FORMATS[0]}. Aliases: {', '.join(sorted(FORMAT_ALIASES))}")
    pauses = n.add_mutually_exclusive_group()
    pauses.add_argument("--pauses", dest="pauses", action="store_true",
                        help="Enable explicit pauses between phrases")
    pauses.add_argument("--no-pauses", dest="pauses", action="store_false",
                        help="Disable explicit pauses (default)")
    n.set_defaults(pauses=False)
    for kind, default in pz.DEFAULT_LENGTHS_MS.items():
        n.add_argument(f"--pause-{kind}", type=float, default=None, metavar="SEC", dest=f"pause_{kind}",
                       help=f"Silence after a {PAUSE_HELP[kind]} in seconds (default: Settings, else {default / 1000:g})")
    n.add_argument("--speed", type=float, default=None, metavar="X",
                   help=f"Global reading speed {pc.MIN_SPEED:g}-{pc.MAX_SPEED:g} (1 = the voice's own; default: Settings)")
    n.add_argument("--style", default=None, choices=pc.STYLES,
                   help="Reading style: auto (detected) | scripture (solemn, slower) | fiction | dialogue (default: Settings)")
    ords = n.add_mutually_exclusive_group()
    ords.add_argument("--ordinals", dest="ordinals", action="store_true", default=None,
                      help="Read ordinal numbers by context (Russian 'глава 2' -> 'глава вторая', '21st', German '3. Kapitel'); "
                           "default: Settings (on)")
    ords.add_argument("--no-ordinals", dest="ordinals", action="store_false",
                      help="Read every number as written (cardinal)")
    n.set_defaults(ordinals=None)
    n.add_argument("--ai-disclosure", action="store_true",
                   help="Speak a short AI disclosure at the start (opt-in)")
    n.add_argument("--speakers", action="store_true",
                   help="Mark each paragraph narrator, male or female with the text model (Gemma), then narrate those voices")
    n.add_argument("--male-voice", default="", metavar="ID_OR_NAME",
                   help="Voice for paragraphs marked male (narrator is --voice)")
    n.add_argument("--female-voice", default="", metavar="ID_OR_NAME",
                   help="Voice for paragraphs marked female (narrator is --voice)")
    n.add_argument("--speaker-marks", type=Path, default=None, metavar="FILE",
                   help="Narrate from this marks file instead of running Gemma (voxprint speakers writes it)")
    n.add_argument("--work-dir", type=Path, default=None, metavar="DIR",
                   help="Remember this folder as the app working folder")
    n.set_defaults(_handler="narrate")

    t = sub.add_parser(
        "train", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Train a voice (LoRA) from a recording or a folder of clips",
        description="Train a LoRA voice. Omit --text for a folder of clips or for no-transcript mode.",
        epilog=(
            "Examples:\n"
            "  voxprint train recording.wav --text script.txt --name Anna --type female\n"
            "  voxprint train ./clips --name Anna --type female --out ./voices --json\n"
            "  voxprint train recording.wav --text script.txt --name Anna --force-cpu --yes\n"
            "  voxprint train ./clips --name Boaz --language ru --consent commercial --speaker \"Reader Name\" --license CC0-1.0\n"
        ),
    )
    t.add_argument("audio", type=Path, help="Recording file, or a folder of clips when --text is omitted")
    t.add_argument("--text", type=Path, default=None, metavar="SCRIPT",
                   help="Transcript of the recording; omit to use no-transcript (ASR) mode")
    t.add_argument("--name", default="", help="Display name for the new voice")
    t.add_argument("--type", dest="voice_type", default="", choices=("",) + VOICE_TYPES,
                   help="Voice type stored in voice.json: male | female | child | other")
    t.add_argument("--out", type=Path, default=None, metavar="DIR",
                   help="Parent folder of the work folder <voice name>_Voxprint (default: next to the recording); "
                        "two voices never share a work folder")
    t.add_argument("--license", dest="voice_license", default="", metavar="ID",
                   help="Licence stored in voice.json, e.g. CC0-1.0, CC-BY-4.0, CC-BY-NC-4.0 (default: from --consent)")
    t.add_argument("--consent", default="none", choices=TRAIN_CONSENT,
                   help="Voice owner's consent: none (private only, default) | auto (read the spoken statement at the end) | "
                        "commercial | public_noncommercial | private_only (confirmed manually)")
    t.add_argument("--speaker", default="", metavar="NAME", help="Name of the person whose voice it is (consent and voice.json)")
    t.add_argument("--language", default="", metavar="LANG",
                   help="Language of the recording, e.g. ru, en, de (default: detected)")
    t.add_argument("--force-cpu", action="store_true", help="Train on CPU even when a GPU is present")
    t.set_defaults(_handler="train")

    v = sub.add_parser(
        "voices", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List or export voices from the library",
        description="List installed voices or export one as a zip.",
        epilog=(
            "Examples:\n"
            "  voxprint voices list\n"
            "  voxprint voices list --json\n"
            "  voxprint voices export my-voice --out my-voice.zip\n"
        ),
    )
    vsub = v.add_subparsers(dest="voices_command", required=True)
    vsub.add_parser(
        "list", parents=[child], help="List installed voices",
        description="List installed voices. Without --json the columns are tab-separated.",
        epilog="Examples:\n  voxprint voices list\n  voxprint voices list --json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).set_defaults(_handler="voices_list")
    ve = vsub.add_parser(
        "export", parents=[child], help="Export a voice as a small zip (adapter + voice.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  voxprint voices export my-voice --out my-voice.zip --json\n",
    )
    ve.add_argument("voice", metavar="VOICE", help="Voice id or display name")
    ve.add_argument("--out", required=True, type=Path, metavar="ZIP", help="Destination .zip path")
    ve.set_defaults(_handler="voices_export")

    d = sub.add_parser(
        "diag", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Save a diagnostic report (logs + system information) as a zip",
        description="Zip logs, system_info.json and a settings snapshot (paths cut to names, secrets dropped).",
        epilog=(
            "Examples:\n"
            "  voxprint diag\n"
            "  voxprint diag --out report.zip --json\n"
        ),
    )
    d.add_argument("--out", type=Path, default=None, metavar="ZIP",
                   help="Destination .zip (default: voxprint-diagnostics-<date>.zip in the current folder)")
    d.set_defaults(_handler="diag")

    st = sub.add_parser(
        "status", aliases=["capabilities"], parents=[child],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List models, GPU, voices and formats as JSON (alias: capabilities)",
        description="One JSON object an agent can use to plan. Printed even without --json.",
        epilog=(
            "Examples:\n"
            "  voxprint status --json\n"
            "  voxprint capabilities --json\n"
        ),
    )
    st.set_defaults(_handler="status")

    b = sub.add_parser(
        "backup", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Copy models and voices to a folder (resumable, with a manifest)",
        description="Copy the models and the voice library into <DIR>/Voxprint-backup/ with a SHA-256 manifest. "
                    "Files already there with the same size and hash are skipped, so a second run continues.",
        epilog=(
            "Examples:\n"
            "  voxprint backup --out E:\\\n"
            "  voxprint backup --out /mnt/usb --no-voices --json\n"
        ),
    )
    b.add_argument("--out", required=True, type=Path, metavar="DIR", help="Folder the backup is written into")
    b.add_argument("--no-models", action="store_true", help="Leave models out")
    b.add_argument("--no-voices", action="store_true", help="Leave the voice library out")
    b.set_defaults(_handler="backup")

    r = sub.add_parser(
        "restore", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Restore a backup into the normal folders, or use its models in place",
        description="Copy a backup back and check every file against the manifest and the pinned hashes. "
                    "A damaged or missing file is reported and left for the normal download.",
        epilog=(
            "Examples:\n"
            "  voxprint restore --from E:\\\n"
            "  voxprint restore --from E:\\ --link --json\n"
        ),
    )
    r.add_argument("--from", dest="src", required=True, type=Path, metavar="DIR",
                   help="Folder that contains the backup (or the Voxprint-backup folder itself)")
    r.add_argument("--link", action="store_true",
                   help="Use models from this folder (don't copy). The drive must stay connected.")
    r.set_defaults(_handler="restore")

    m = sub.add_parser(
        "models", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List or download speech, translation and optional tool modules",
        description="Optional and required model modules. Already-installed modules are left in place.",
        epilog=_module_help_epilog(),
    )
    msub = m.add_subparsers(dest="models_command", required=True)
    msub.add_parser(
        "list", parents=[child], help="List module ids and whether each one is installed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  voxprint models list --json\n",
    ).set_defaults(_handler="models_list")
    md_p = msub.add_parser(
        "download", parents=[child], help="Download one module if it is not installed yet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download a module by id. Safe to repeat: an installed module is not fetched again.",
        epilog=_module_help_epilog(),
    )
    md_p.add_argument("module", help="Module id (voxprint models list --json)")
    md_p.set_defaults(_handler="models_download")

    rv = sub.add_parser(
        "revoice", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Transcribe speech to a text book you can narrate",
        description=(
            "Recognise one or more audio files (or a folder) with the installed speech model "
            "and write a TXT book. Narrate that file with the narrate command."
        ),
        epilog=(
            "Examples:\n"
            "  voxprint revoice recording.wav --out ./revoice --json\n"
            "  voxprint revoice ./chapters --title \"My talk\" --out ./revoice --language en\n"
            "  voxprint narrate ./revoice/My-talk.txt --voice my-voice --out ./audiobooks --format mp3\n"
        ),
    )
    rv.add_argument("audio", nargs="+", help="Audio file(s) or a folder of clips")
    rv.add_argument("--out", type=Path, default=None, metavar="DIR",
                    help="Folder for the text book (default: the Re-voice projects folder)")
    rv.add_argument("--title", default="", help="File name for the text book (default: the first clip's title)")
    rv.add_argument("--language", default=None, help="Recognition language hint (default: automatic)")
    rv.set_defaults(_handler="revoice")

    sp = sub.add_parser(
        "speakers", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Mark each paragraph narrator, male or female (no narration)",
        description=(
            "Ask the text model (Gemma) who speaks each paragraph and write an editable marks file. "
            "Narrate that file with narrate --speaker-marks. This command does not synthesize audio."
        ),
        epilog=(
            "Examples:\n"
            "  voxprint speakers book.txt --out marks.txt --json\n"
            "  voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann "
            "--speaker-marks marks.txt --out ./audiobooks --json\n"
        ),
    )
    sp.add_argument("book", help="Path to a TXT, FB2, FB2.ZIP or EPUB file")
    sp.add_argument("--out", required=True, type=Path, metavar="FILE", help="Marks file to write")
    sp.set_defaults(_handler="speakers")

    ck = sub.add_parser(
        "check", aliases=["repair"], parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Check models and files and fix what is damaged (alias: repair)",
        description=(
            "The same check as Settings -> Check & repair: program, components and models by hash, "
            "missing or damaged files fetched again, stale model lock files removed."
        ),
        epilog=(
            "Examples:\n"
            "  voxprint check --json\n"
            "  voxprint repair --json\n"
        ),
    )
    ck.set_defaults(_handler="check")
    return ap


def parse_formats(raw: Sequence[str]) -> Set[str]:
    """Turn ``--format`` values (aliases, commas, repeats) into format keys; empty -> defaults."""
    out: Set[str] = set()
    for item in raw:
        for part in str(item).split(","):
            key = part.strip().lower()
            if not key:
                continue
            mapped = FORMAT_ALIASES.get(key, key)
            if mapped not in ex.ALL_FORMATS:
                raise CliError(
                    EXIT_BAD_ARGS,
                    f"unknown format {part!r}",
                    hint=(
                        f"Choose from: {', '.join(ex.ALL_FORMATS)} "
                        f"(aliases: {', '.join(sorted(FORMAT_ALIASES))}). "
                        "Example: voxprint narrate book.epub --voice my-voice --out ./audiobooks "
                        "--format mp3,m4b,flac,opus"
                    ),
                )
            out.add(mapped)
    return out or set(ex.DEFAULT_FORMATS)


def resolve_voice(library: VoiceLibrary, id_or_name: str) -> VoiceRecord:
    """Resolve a voice by folder id or display name (case-insensitive)."""
    needle = (id_or_name or "").strip()
    if not needle:
        raise CliError(EXIT_INPUT, "voice id or name is required", hint="voxprint voices list --json")
    rec = library.get(needle)
    if rec is not None:
        return rec
    lower = needle.lower()
    matches = [v for v in library.list_voices() if v.id.lower() == lower or v.name.lower() == lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(m.id for m in matches)
        raise CliError(
            EXIT_INPUT, f"voice {needle!r} matches several entries: {ids}",
            hint="Pass the voice id. List them with: voxprint voices list --json",
        )
    raise CliError(
        EXIT_INPUT, f"voice not found: {needle}",
        hint="List voices with: voxprint voices list --json",
    )


def _voice_payload(voice: VoiceRecord) -> dict:
    return {
        "id": voice.id,
        "name": voice.name,
        "voice_type": voice.info.get("voice_type") or "",
        "language": voice.language or "",
        "license": voice.license,
        "consent_scope": voice.scope,
        "commercial_use": bool(voice.commercial_use),
        "gender": voice.info.get("gender") or "",
        "age_group": voice.info.get("age_group") or "",
    }


def _narration_progress(json_mode: bool):
    def cb(p: NarrationProgress) -> None:
        if json_mode:
            extra = {"done": p.done, "total": p.total}
            if p.eta is not None and p.eta >= 0:
                extra["eta_s"] = round(float(p.eta), 1)
            _emit_progress(p.phase, p.fraction * 100.0, p.message, **extra)
            return
        eta = ""
        if p.eta is not None and p.eta >= 0:
            eta = f"  ETA {format_eta(p.eta)}"
        _write_stream(sys.stdout, f"[{p.phase}] {p.done}/{p.total}{eta}  {p.message}")

    return cb


def _train_progress(json_mode: bool):
    plan = plan_for(KIND_LORA)

    def cb(stage, frac, msg) -> None:
        if isinstance(stage, Stage):
            name, label = stage.value, stage.label
            percent = float(overall_percent(plan, stage, float(frac)))
        else:
            name = label = str(stage)
            percent = float(frac) * 100.0
        if json_mode:
            _emit_progress(name, percent, str(msg))
        else:
            _write_stream(sys.stdout, f"[{label}] {float(frac) * 100:5.1f}%  {msg}")

    return cb


def cmd_narrate(args: argparse.Namespace, *,
                run_fn: Callable[..., object] = run_narration,
                library: Optional[VoiceLibrary] = None,
                plan_fn: Optional[Callable[[], object]] = None) -> int:
    """``narrate BOOK --voice ... --out DIR``: load the book, resolve the voice, call ``run_narration``."""

    def body(json_mode: bool, started: float) -> int:
        lib = library if library is not None else VoiceLibrary()
        voice = resolve_voice(lib, args.voice)
        book_path = Path(args.book)
        if not book_path.is_file():
            raise CliError(
                EXIT_INPUT, f"book not found: {book_path}",
                hint="Pass a TXT, FB2, FB2.ZIP or EPUB file that exists. "
                     "Example: voxprint narrate book.epub --voice my-voice --out ./audiobooks",
            )
        book = load_book(book_path)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.work_dir is not None:
            ws.save_folder(Path(args.work_dir))
        formats = parse_formats(args.formats)
        lengths, pace = narration_shaping(args)
        options = NarrationOptions(
            formats=formats,
            pauses=pz.PauseProfile(lengths=lengths) if args.pauses else None,
            pause_lengths=lengths, pace=pace,
            ai_disclosure=bool(args.ai_disclosure),
            ordinals=ordinals.load_enabled() if getattr(args, "ordinals", None) is None else bool(args.ordinals),
        )
        cast, extra_voices = _speaker_job(args, voice, lib, plan_fn)
        if cast is not None:
            options.speakers = cast
        job = NarrationJob(book=book, voice=voice, out_dir=out_dir, options=options, extra_voices=extra_voices)
        with keep_awake.keep_awake():
            result = run_fn(job, _narration_progress(json_mode), CancelToken(), PauseToken())
        files = list(getattr(result, "files", None) or [])
        done = Path(getattr(result, "out_dir", out_dir))
        marks = done / ".debug" / "speakers.txt"
        if marks.is_file():
            files.append(marks)
        warnings = []
        if getattr(result, "speaker_warning", "") == "mismatch":
            warnings.append(SPEAKER_MISMATCH)
        human = [f"Done: {done}"] + [f"  {f}" for f in files] + [f"  ! {w}" for w in warnings]
        return _ok(
            json_mode, "narrate", started, outputs=files, warnings=warnings,
            human=human, extra={"out_dir": str(done)},
        )

    return _run("narrate", args, body)


SPEAKER_MISMATCH = "Speaker marks do not match the prepared text, so the narrator reads the whole book."


def _speaker_job(args, narrator, library: VoiceLibrary, plan_fn: Optional[Callable[[], object]]):
    """Speaker cast and extra voices for ``narrate``, or ``(None, {})`` when multi-voice is off.

    The cast is the same object the Narrate window builds (:class:`core.speakers.SpeakerCast`). Narration applies it.
    """
    want = bool(getattr(args, "speakers", False)) or getattr(args, "speaker_marks", None) is not None
    male_name = (getattr(args, "male_voice", "") or "").strip()
    female_name = (getattr(args, "female_voice", "") or "").strip()
    if not want and not male_name and not female_name:
        return None, {}
    if not want:
        raise CliError(
            EXIT_BAD_ARGS, "--male-voice and --female-voice need --speakers or --speaker-marks",
            hint="Example: voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann "
                 "--speakers --out ./audiobooks",
        )
    if args.speakers and args.speaker_marks is not None:
        raise CliError(
            EXIT_BAD_ARGS, "--speakers and --speaker-marks cannot be used together",
            hint="Use --speakers to ask Gemma, or --speaker-marks FILE to narrate marks you already edited.",
        )
    male = resolve_voice(library, male_name) if male_name else None
    female = resolve_voice(library, female_name) if female_name else None
    lines = None
    tagger = None
    if args.speaker_marks is not None:
        path = Path(args.speaker_marks)
        if not path.is_file():
            raise CliError(
                EXIT_INPUT, f"speaker marks not found: {path}",
                hint="Write them with: voxprint speakers book.txt --out marks.txt",
            )
        try:
            lines = spk.load_marks(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CliError(
                EXIT_INPUT, f"speaker marks cannot be read: {exc}",
                hint="Each line is 'N. NARRATOR' or 'N. MALE: Name' or 'N. FEMALE: Name'.",
            ) from exc
    else:
        plan = (plan_fn or llm_tool.make_plan)()
        if plan is None:
            raise CliError(
                EXIT_MISSING, "text model is not installed",
                hint="voxprint models download llm --json",
            )
        tagger = plan
    cast = spk.SpeakerCast(
        lines=lines, male_id=male.id if male is not None else "", female_id=female.id if female is not None else "",
        tagger=tagger, narrator_id=narrator.id,
    )
    if not cast.uses_several(narrator.id):
        raise CliError(
            EXIT_BAD_ARGS, "pick a male or female voice that is not the narrator",
            hint="Example: voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann "
                 "--speakers --out ./audiobooks",
        )
    extra = {}
    for rec in (male, female):
        if rec is not None and rec.id != narrator.id:
            extra[rec.id] = rec
    return cast, extra


PAUSE_HELP = {"comma": "comma", "mid": "strong mid-sentence break (dash, colon, semicolon, comma + conjunction)",
              "sentence": "sentence end (. ! ?)", "paragraph": "paragraph or numbered / verse line",
              "chapter": "chapter title, chapter end or scene break"}


def narration_shaping(args: argparse.Namespace):
    """Pause lengths and reading pace: the saved Settings, overridden by ``--pause-* / --speed / --style``."""
    saved = pz.load_lengths().to_dict()
    for kind in pz.DEFAULT_LENGTHS_MS:
        sec = getattr(args, f"pause_{kind}", None)
        if sec is not None:
            if not 0 <= sec <= pz.MAX_PAUSE_MS / 1000:
                raise CliError(EXIT_INPUT, f"--pause-{kind} must be 0-{pz.MAX_PAUSE_MS / 1000:g} seconds")
            saved[kind] = round(sec * 1000)
    pace = pc.load()
    if getattr(args, "speed", None) is not None:
        if not pc.MIN_SPEED <= args.speed <= pc.MAX_SPEED:
            raise CliError(EXIT_INPUT, f"--speed must be {pc.MIN_SPEED:g}-{pc.MAX_SPEED:g}")
        pace.speed = args.speed
    if getattr(args, "style", None):
        pace.style = args.style
    return pz.PauseLengths.from_dict(saved), pc.Pace(pace.speed, pace.style)


TRAIN_CONSENT = ("none", "auto", "commercial", "public_noncommercial", "private_only")


def _train_meta(args: argparse.Namespace) -> dict:
    """TaskRequest fields from ``--consent / --speaker / --license / --language`` (checked: a licence never allows more than the
    consent, e.g. CC0-1.0 needs ``--consent commercial``)."""
    from core import consent, languages, voice_info

    mode = args.consent if args.consent in ("none", "auto") else "manual"
    scope = args.consent if mode == "manual" else consent.PRIVATE
    meta = {"consent_mode": mode, "consent_scope": scope, "consent_name": args.speaker or "", "speaker": args.speaker or ""}
    lic = (args.voice_license or "").strip()
    if lic:
        if lic not in voice_info.LICENSES:
            raise CliError(EXIT_INPUT, f"unknown licence: {lic}", hint="Use one of: " + ", ".join(voice_info.LICENSES) + ".")
        if mode != "auto" and (voice_info.license_allows_commercial(lic) and scope != consent.COMMERCIAL
                               or "-NC" in lic and scope == consent.PRIVATE):
            raise CliError(EXIT_INPUT, f"licence {lic} allows more than --consent {args.consent}",
                           hint="Pass --consent commercial for a licence that allows commercial use "
                                "(public_noncommercial for -NC licences).")
        meta["license"] = lic
    lang = (args.language or "").strip()
    if lang:
        code = languages.language_code(lang)
        if code not in languages.NAMES:
            raise CliError(EXIT_INPUT, f"unknown language: {lang}", hint="Use a code or name, e.g. ru, en, de, Russian.")
        meta["language"] = code
        meta["asr_language"] = languages.language_name(code)
    return meta


def cmd_train(args: argparse.Namespace, *,
              run_fn: Callable[..., object] = run_task) -> int:
    """``train AUDIO [--text SCRIPT]``: build a LoRA voice via ``run_task``."""

    def body(json_mode: bool, started: float) -> int:
        audio = Path(args.audio)
        if not audio.exists():
            raise CliError(
                EXIT_INPUT, f"audio not found: {audio}",
                hint="Pass a recording file or a folder of clips. "
                     "Example: voxprint train ./clips --name Anna --type female --json",
            )
        text = Path(args.text) if args.text else None
        if text is not None and not text.is_file():
            raise CliError(
                EXIT_INPUT, f"text not found: {text}",
                hint="Point --text at the transcript file, or omit --text for no-transcript mode.",
            )
        vtype = normalize_voice_type(args.voice_type)
        meta = _train_meta(args)
        if audio.is_dir() or text is None:
            req = TaskRequest(
                kind=KIND_LORA,
                no_transcript=True,
                audio_files=[audio],
                text=text,
                out_root=args.out,
                voice_display_name=args.name or "",
                voice_type=vtype,
                force_cpu=bool(args.force_cpu),
                **meta,
            )
        else:
            req = TaskRequest(
                kind=KIND_LORA,
                audio=audio,
                text=text,
                out_root=args.out,
                voice_display_name=args.name or "",
                voice_type=vtype,
                force_cpu=bool(args.force_cpu),
                **meta,
            )
        if args.out is not None:      # one work folder per voice: a second voice must not overwrite the first one's report.json
            req.out_root = Path(args.out) / f"{req.voice_name()}_Voxprint"
        with keep_awake.keep_awake():
            result = run_fn(req, _train_progress(json_mode), CancelToken())
        voice_id = getattr(result, "voice_id", "") or ""
        adapter = getattr(result, "adapter_path", None)
        root = getattr(result, "root_dir", "")
        warnings = list(getattr(result, "warnings", None) or [])
        outputs = [p for p in (adapter, root) if p]
        return _ok(
            json_mode, "train", started, outputs=outputs, warnings=warnings,
            human=[f"Done: voice_id={voice_id or '(none)'}  adapter={adapter}  root={root}"]
            + [f"  ! {w}" for w in warnings],
            extra={
                "voice_id": voice_id,
                "adapter": str(adapter) if adapter else None,
                "root": str(root) if root else None,
            },
        )

    return _run("train", args, body)


def cmd_voices_list(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None) -> int:
    """``voices list``: id, name, type, language, licence, gender, age group (tab-separated; \"-\" = not set)."""

    def body(json_mode: bool, started: float) -> int:
        lib = library if library is not None else VoiceLibrary()
        voices = lib.list_voices()
        rows = [_voice_payload(v) for v in voices]
        if json_mode:
            return _ok(json_mode, "voices list", started, outputs=[], extra={"voices": rows})
        if not voices:
            _write_stream(sys.stdout, "No voices installed.")
            return EXIT_OK
        for v in voices:
            vtype = v.info.get("voice_type") or "-"
            gender, age = v.info.get("gender") or "-", v.info.get("age_group") or "-"
            _write_stream(sys.stdout, f"{v.id}\t{v.name}\t{vtype}\t{v.language or '-'}\t{v.license}\t{gender}\t{age}")
        return EXIT_OK

    return _run("voices list", args, body)


def cmd_voices_export(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None) -> int:
    """``voices export VOICE --out ZIP``: pack the voice for transfer."""

    def body(json_mode: bool, started: float) -> int:
        lib = library if library is not None else VoiceLibrary()
        voice = resolve_voice(lib, args.voice)
        dest = Path(args.out)
        if dest.suffix.lower() != ".zip":
            dest = dest.with_suffix(dest.suffix + ".zip") if dest.suffix else dest.with_suffix(".zip")
        path = lib.export_zip(voice.id, dest)
        return _ok(json_mode, "voices export", started, outputs=[path], human=[f"Exported: {path}"])

    return _run("voices export", args, body)


def cmd_diag(args) -> int:
    """``diag``: zip the logs, system information and a settings snapshot (infra/diagnostics.py)."""

    def body(json_mode: bool, started: float) -> int:
        out = args.out or Path.cwd() / time.strftime("voxprint-diagnostics-%Y%m%d-%H%M%S.zip")
        info = diagnostics.system_info(import_torch=True, full=True)
        lines = diagnostics.summary_lines(info)
        path = diagnostics.write_report(out, info=lambda: info)
        return _ok(
            json_mode, "diag", started, outputs=[path],
            human=list(lines) + [f"Diagnostic report: {path}"],
            extra={"summary": list(lines)},
        )

    return _run("diag", args, body)


def _report_payload(report) -> dict:
    """JSON-friendly summary of a backup or restore report."""
    return {
        "target": str(report.target),
        "copied_files": report.copied_files,
        "skipped_files": report.skipped_files,
        "copied_bytes": report.copied_bytes,
        "problems": list(getattr(report, "problems", ()) or ()),
        "conflicts": list(report.conflicts),
        "external_models": getattr(report, "external", "") or "",
    }


def _report_lines(payload: dict) -> list[str]:
    if payload["external_models"]:
        lines = [f"Models stay in {payload['external_models']} (not copied)."]
        if payload["problems"]:
            lines.append("Damaged or missing: " + ", ".join(payload["problems"]))
        lines.append("The drive must stay connected. If it is missing, models are downloaded into the usual folder.")
        return lines
    lines = [f"Copied {payload['copied_files']} files, skipped {payload['skipped_files']}."]
    if payload["problems"]:
        lines.append("Damaged or missing (downloaded as usual): " + ", ".join(payload["problems"]))
    if payload["conflicts"]:
        lines.append("Not overwritten: " + ", ".join(payload["conflicts"]))
    return lines


def _backup_error(exc: BackupError) -> CliError:
    hint = "Free space on the drive or choose another folder." if exc.code == "space" else \
        "Check that the folder exists and is readable or writable."
    return CliError(EXIT_INPUT, exc.user_message, hint=hint, details=exc.details or "")


def cmd_backup(args) -> int:
    """``backup --out DIR [--no-models|--no-voices]``."""

    def body(json_mode: bool, started: float) -> int:
        items = collect_backup(include_voices=not args.no_voices, include_models=not args.no_models)
        if not items:
            raise CliError(EXIT_INPUT, "nothing to back up", hint="Download models or create a voice first.")
        try:
            report = backup_mod.run_backup(items, args.out, cancel=CancelToken())
        except BackupError as exc:
            raise _backup_error(exc) from exc
        payload = _report_payload(report)
        return _ok(json_mode, "backup", started, outputs=[report.target], human=_report_lines(payload), extra=payload)

    return _run("backup", args, body)


def cmd_restore(args) -> int:
    """``restore --from DIR [--link]``. Damaged files are reported; the app downloads those later."""

    def body(json_mode: bool, started: float) -> int:
        try:
            if args.link:
                report = backup_mod.run_link(args.src, cancel=CancelToken())
            else:
                report = backup_mod.run_restore(args.src, strict=False, cancel=CancelToken())
        except BackupError as exc:
            raise _backup_error(exc) from exc
        payload = _report_payload(report)
        warnings = [f"damaged or missing: {p}" for p in payload["problems"]]
        return _ok(json_mode, "restore", started, outputs=[report.target], warnings=warnings,
                   human=_report_lines(payload), extra=payload)

    return _run("restore", args, body)


def cmd_version(json_mode: bool) -> int:
    """Print version, build number and codename, then exit."""
    info = version_payload()
    if json_mode:
        _emit_result(
            ok=True, command="version", exit_code=EXIT_OK, outputs=[], warnings=[],
            duration_s=0.0, error=None, hint=None, extra=info,
        )
    else:
        _write_stream(sys.stdout, version_line(info))
    return EXIT_OK


def _repo_installed(repo: str) -> bool:
    try:
        return bool(md.verify_local_model(md.local_dir_for(repo)))
    except OSError:
        return False


def _preferred_tts_repo() -> str:
    """TTS repository this PC should use.  ``detect_gpu`` imports torch only when called."""
    return plan_training(detect_gpu(), 100).base_model


def _preferred_asr_repo() -> str:
    """Speech-recognition repository this PC should use.  ``preferred_repo`` may import torch."""
    return preferred_repo()


def _repo_for(key: str) -> str:
    table = {
        "tts": _preferred_tts_repo,
        "tts-1.7b": lambda: MODEL_1_7B,
        "tts-0.6b": lambda: MODEL_0_6B,
        "aligner": lambda: ALIGNER_REPO,
        "asr": _preferred_asr_repo,
        "asr-0.6b": lambda: ASR_SMALL,
        "asr-1.7b": lambda: ASR_LARGE,
    }
    fn = table.get(key)
    if fn is None:
        raise CliError(EXIT_BAD_ARGS, f"unknown module {key!r}", hint=hint_for(EXIT_BAD_ARGS, "models"))
    return fn()


def _translate_ids() -> list[str]:
    return [m.key for m in text_models.REGISTRY if m.integrated and m.kind == text_models.KIND_TRANSLATE]


def canonical_module(name: str) -> str:
    """Normalise a module id or alias.  Raises :class:`CliError` when it is unknown."""
    raw = (name or "").strip().lower()
    key = ALIASES.get(raw, raw)
    if key not in _known_module_ids():
        shown = ", ".join(sorted(_known_module_ids()))
        raise CliError(
            EXIT_BAD_ARGS, f"unknown module {name!r}",
            hint=f"Known modules: {shown}. Example: voxprint models download denoise",
        )
    return key


def module_installed(module_id: str) -> bool:
    """True when ``module_id`` is already on disk (no download)."""
    key = canonical_module(module_id)
    if key == "translate":
        return all(module_installed(part) for part in _translate_ids())
    if key == "required":
        return all(module_installed(part) for part in ("tts", "aligner", "asr"))
    if key == "denoise":
        return denoise_tool.ready() is not None
    if key == "llm":
        return bool(llm_tool.model_ready() and llm_tool.server_exe())
    if key == "dnsmos":
        return bool(quality_models.dnsmos_ready())
    if key == "openvoice":
        return bool(vc_model.ready())
    if key in _text_ids():
        return text_models.state(text_models.get(key)) == text_models.STATE_READY
    return _repo_installed(_repo_for(key))


def module_catalog() -> list[dict]:
    """Installed-state rows for status and ``models list``.  Does not download and does not probe the GPU."""
    rows = [
        {"id": "tts-1.7b", "title": "Qwen3-TTS 1.7B", "optional": False, "kind": "model",
         "installed": _repo_installed(MODEL_1_7B)},
        {"id": "tts-0.6b", "title": "Qwen3-TTS 0.6B", "optional": False, "kind": "model",
         "installed": _repo_installed(MODEL_0_6B)},
        {"id": "aligner", "title": "Qwen3 forced aligner", "optional": False, "kind": "model",
         "installed": _repo_installed(ALIGNER_REPO)},
        {"id": "asr-0.6b", "title": "Qwen3-ASR 0.6B", "optional": False, "kind": "model",
         "installed": _repo_installed(ASR_SMALL)},
        {"id": "asr-1.7b", "title": "Qwen3-ASR 1.7B", "optional": False, "kind": "model",
         "installed": _repo_installed(ASR_LARGE)},
    ]
    for model in text_models.REGISTRY:
        if not model.integrated:
            continue
        rows.append({
            "id": model.key,
            "title": model.name,
            "optional": True,
            "kind": "text",
            "installed": text_models.state(model) == text_models.STATE_READY,
        })
    rows.append({
        "id": "denoise", "title": denoise_tool.LABEL, "optional": True, "kind": "tool",
        "installed": denoise_tool.ready() is not None,
    })
    rows.append({
        "id": "llm", "title": llm_tool.LABEL, "optional": True, "kind": "tool",
        "installed": bool(llm_tool.model_ready() and llm_tool.server_exe()),
    })
    rows.append({
        "id": "dnsmos", "title": quality_models.DNSMOS_LABEL, "optional": True, "kind": "tool",
        "installed": bool(quality_models.dnsmos_ready()),
    })
    rows.append({
        "id": "openvoice", "title": vc_model.LABEL, "optional": True, "kind": "model",
        "installed": bool(vc_model.ready()),
    })
    return rows


def _paths_for(key: str) -> list[str]:
    if key == "translate":
        paths: list[str] = []
        for part in _translate_ids():
            paths.extend(_paths_for(part))
        return paths
    if key == "required":
        paths = []
        for part in ("tts", "aligner", "asr"):
            paths.extend(_paths_for(part))
        return paths
    if key == "denoise":
        path = denoise_tool.tool_path()
        return [str(path)] if path else []
    if key == "llm":
        return [str(llm_tool.model_path())]
    if key == "dnsmos":
        return [str(quality_models.dnsmos_path())]
    if key == "openvoice":
        return [str(vc_model.model_dir())]
    if key in _text_ids():
        return [str(text_models.get(key).local_dir)]
    return [str(md.local_dir_for(_repo_for(key)))]


def _stage_progress(progress: Callable[[float, str], None]):
    def cb(stage, frac, msg="") -> None:
        progress(float(frac or 0), str(msg or ""))

    return cb


def _fetch_one(key: str, progress: Callable[[float, str], None]) -> Path:
    stage_cb = _stage_progress(progress)
    try:
        if key == "denoise":
            return Path(denoise_tool.ensure(progress))
        if key == "llm":
            return Path(llm_tool.ensure(progress))
        if key == "dnsmos":
            return Path(quality_models.ensure_dnsmos(progress))
        if key == "openvoice":
            return Path(vc_model.ensure(progress))
        if key in _text_ids():
            return Path(text_models.ensure(text_models.get(key), stage_cb))
        return Path(md.ensure_model(_repo_for(key), stage_cb))
    except ReleaseError as exc:
        raise CliError(EXIT_MISSING, str(exc), hint=hint_for(EXIT_MISSING, "models")) from exc


def _fetch_group(key: str, progress: Callable[[float, str], None], check: Callable[[str], bool]) -> dict:
    parts = list(_translate_ids()) if key == "translate" else ["tts", "aligner", "asr"]
    todo = [part for part in parts if not check(part)]
    if not todo:
        paths: list[str] = []
        for part in parts:
            paths.extend(_paths_for(part))
        return {"module": key, "installed": True, "downloaded": False, "paths": paths}
    paths = []
    count = len(todo)
    for index, part in enumerate(todo):
        def sub(frac: float, msg: str, index: int = index) -> None:
            progress((index + float(frac)) / count, msg)

        paths.append(str(_fetch_one(part, sub)))
    return {"module": key, "installed": True, "downloaded": True, "paths": paths}


def fetch_module(module_id: str, progress: Callable[[float, str], None], *,
                 installed_fn: Optional[Callable[[str], bool]] = None) -> dict:
    """Download ``module_id`` when it is missing.  A second call is a no-op success.

    ``progress(fraction, message)`` receives a fraction from 0 to 1.  This function does not
    prompt.  Tests pass fakes; the default path is the real downloader.
    """
    key = canonical_module(module_id)
    check = installed_fn or module_installed
    if key in ("translate", "required"):
        return _fetch_group(key, progress, check)
    if check(key):
        return {"module": key, "installed": True, "downloaded": False, "paths": _paths_for(key)}
    path = _fetch_one(key, progress)
    return {"module": key, "installed": True, "downloaded": True, "paths": [str(path)]}


def _runtime_status() -> dict:
    """Thin-build Python wheels, from the cached manifest only (no network)."""
    if not runtime_modules.is_thin():
        return {"thin": False, "modules": []}
    cache = app_paths.state_dir() / runtime_modules.MANIFEST_CACHE
    if not cache.is_file():
        return {
            "thin": True,
            "modules": [],
            "note": "No cached runtime manifest. Voxprint.exe --modules-status lists Python components.",
        }
    try:
        manifest = json.loads(cache.read_text(encoding="utf-8"))
        rows = [{
            "id": m.id, "title": m.title, "installed": bool(m.installed), "required": bool(m.required),
        } for m in runtime_modules.modules(manifest)]
    except (OSError, ValueError, TypeError, KeyError, AttributeError, runtime_modules.ModulesError):
        return {"thin": True, "modules": [], "note": "Runtime manifest could not be read."}
    return {"thin": True, "modules": rows}


def collect_status(library: Optional[VoiceLibrary] = None) -> dict:
    """Facts an agent needs before it plans a job.  Does not download models."""
    lib = library if library is not None else VoiceLibrary()
    info = version_payload()
    probe = os.environ.get("VOXPRINT_NO_ENV_PROBE") == "1"
    gpu_raw = diagnostics.gpu_info(import_torch=not probe)
    return {
        "version": info["version"],
        "build": info["build"],
        "codename": info["codename"],
        "platform": sys.platform,
        "gpu": {
            "cuda_available": bool(gpu_raw.get("cuda_available")),
            "name": gpu_raw.get("gpu") or None,
            "vram_total_gb": gpu_raw.get("vram_total_gb"),
            "vram_free_gb": gpu_raw.get("vram_free_gb"),
            "torch": gpu_raw.get("torch"),
            "driver": gpu_raw.get("driver"),
        },
        "voices": [_voice_payload(v) for v in lib.list_voices()],
        "formats": list(ex.ALL_FORMATS),
        "format_aliases": dict(FORMAT_ALIASES),
        "modules": module_catalog(),
        "runtime": _runtime_status(),
    }


def cmd_status(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None) -> int:
    """``status`` / ``capabilities``: one JSON object (also without ``--json``)."""

    def body(_json_mode: bool, started: float) -> int:
        data = collect_status(library)
        command = getattr(args, "command", None) or "status"
        return _ok(True, command, started, outputs=[], extra=data)

    return _run(getattr(args, "command", None) or "status", args, body)


def cmd_models_list(args: argparse.Namespace) -> int:
    """``models list``: module id, installed, optional, kind, title."""

    def body(json_mode: bool, started: float) -> int:
        rows = module_catalog()
        if json_mode:
            return _ok(json_mode, "models list", started, outputs=[], extra={"modules": rows})
        for row in rows:
            installed = "yes" if row["installed"] else "no"
            optional = "yes" if row["optional"] else "no"
            _write_stream(sys.stdout, f"{row['id']}\t{installed}\t{optional}\t{row['kind']}\t{row['title']}")
        return EXIT_OK

    return _run("models list", args, body)


def cmd_models_download(args: argparse.Namespace, *,
                        download_fn: Optional[Callable[..., object]] = None,
                        installed_fn: Optional[Callable[[str], bool]] = None) -> int:
    """``models download MODULE``.  Does not prompt.  Repeating a finished download is success."""

    def body(json_mode: bool, started: float) -> int:
        key = canonical_module(args.module)

        def progress(frac: float, msg: str) -> None:
            if json_mode:
                _emit_progress("download", float(frac) * 100.0, msg)
            else:
                _write_stream(sys.stdout, f"[download] {float(frac) * 100:5.1f}%  {msg}")

        if download_fn is not None:
            if installed_fn is not None and installed_fn(key):
                return _ok(
                    json_mode, "models download", started, outputs=[],
                    human=[f"Already installed: {key}"],
                    extra={"module": key, "installed": True, "downloaded": False},
                )
            path = download_fn(key, progress)
            outputs = [str(path)] if path else []
            human = [f"Downloaded: {key}" + (f" -> {path}" if path else "")]
            return _ok(
                json_mode, "models download", started, outputs=outputs, human=human,
                extra={"module": key, "installed": True, "downloaded": True},
            )
        info = fetch_module(key, progress, installed_fn=installed_fn)
        paths = [str(p) for p in info.get("paths") or [] if p]
        verb = "Already installed" if not info["downloaded"] else "Downloaded"
        human = [f"{verb}: {info['module']}"] + [f"  {p}" for p in paths]
        return _ok(
            json_mode, "models download", started, outputs=paths, human=human,
            extra={"module": info["module"], "installed": True, "downloaded": bool(info["downloaded"])},
        )

    return _run("models download", args, body)


def _expand_audio(items: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for raw in items:
        path = Path(raw)
        if not path.exists():
            missing.append(str(path))
            continue
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in revoice.AUDIO_EXTENSIONS)
            if not found:
                names = ", ".join(revoice.AUDIO_EXTENSIONS)
                raise CliError(
                    EXIT_INPUT, f"no audio files in {path}",
                    hint=f"Put audio files in the folder ({names}). Example: voxprint revoice ./clips --out ./revoice",
                )
            files.extend(found)
        else:
            files.append(path)
    if missing:
        raise CliError(
            EXIT_INPUT, "audio not found: " + ", ".join(missing),
            hint="Example: voxprint revoice recording.wav --out ./revoice",
        )
    return files


def _default_transcribe(files, language, progress, cancel):
    """Installed Qwen3-ASR.  The model itself is loaded only when this runs."""
    found = asr_ready()
    if found is None:
        raise CliError(
            EXIT_MISSING, "speech recognition model is not installed",
            hint="voxprint models download asr",
        )
    asr = make_default_asr(str(found[1]), "auto")

    def prog(frac, title):
        progress(float(frac), str(title))

    return revoice.transcribe_files(files, asr, language, prog, cancel)


def cmd_revoice(args: argparse.Namespace, *,
                transcribe_fn: Optional[Callable[..., object]] = None) -> int:
    """``revoice AUDIO... --out DIR``: write a TXT book.  Does not narrate it."""

    def body(json_mode: bool, started: float) -> int:
        files = _expand_audio(args.audio)
        out_dir = Path(args.out) if args.out else projects.sub(projects.REVOICE)

        def progress(frac: float, title: str) -> None:
            if json_mode:
                _emit_progress("transcribe", float(frac) * 100.0, title)
            else:
                _write_stream(sys.stdout, f"[transcribe] {float(frac) * 100:5.1f}%  {title}")

        fn = transcribe_fn or _default_transcribe
        chapters = list(fn(files, args.language, progress, CancelToken()))
        if not any(str(text).strip() for _title, text in chapters):
            raise CliError(
                EXIT_INPUT, "recogniser returned no text",
                hint=("Check the audio and --language. "
                      "Example: voxprint revoice recording.wav --out ./revoice --language en"),
            )
        name = args.title or revoice.chapter_title(files[0])
        path = revoice.save_text(revoice.to_text(chapters), out_dir, name)
        return _ok(
            json_mode, "revoice", started, outputs=[path],
            human=[f"Text: {path}", "Narrate it with: voxprint narrate <file> --voice <id> --out <dir>"],
            extra={"chapters": len(chapters)},
        )

    return _run("revoice", args, body)


def _flag_present(argv: Sequence[str], *names: str) -> bool:
    return any(a in names for a in argv)


def cmd_speakers(args: argparse.Namespace, *,
                 plan_fn: Optional[Callable[[], object]] = None) -> int:
    """``speakers BOOK --out FILE``: Gemma marks each paragraph. Does not narrate."""

    def body(json_mode: bool, started: float) -> int:
        book_path = Path(args.book)
        if not book_path.is_file():
            raise CliError(
                EXIT_INPUT, f"book not found: {book_path}",
                hint="Example: voxprint speakers book.txt --out marks.txt --json",
            )
        book = load_book(book_path)
        paras = [text for _ci, text in spk.paragraphs(book)]
        if not paras:
            raise CliError(EXIT_INPUT, "book has no paragraphs", hint="The file needs at least one paragraph of text.")
        plan = (plan_fn or llm_tool.make_plan)()
        if plan is None:
            raise CliError(EXIT_MISSING, "text model is not installed", hint="voxprint models download llm --json")
        if json_mode:
            _emit_progress("prepare", 0.0, "Marking speakers")

        def progress(frac: float) -> None:
            if json_mode:
                _emit_progress("prepare", float(frac) * 100.0, "Marking speakers")

        lines = spk.tag_paragraphs(paras, detect_book_language(book) or "en", plan, progress)
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(spk.dump_marks(lines), encoding="utf-8")
        return _ok(
            json_mode, "speakers", started, outputs=[dest],
            human=[f"Marks: {dest}", f"  {len(lines)} paragraphs"],
            extra={"paragraphs": len(lines)},
        )

    return _run("speakers", args, body)


def cmd_check(args: argparse.Namespace, *,
              repair_fn: Optional[Callable[..., object]] = None) -> int:
    """``check`` / ``repair``: Settings -> Check & repair (:func:`infra.auto_repair.run`)."""

    def body(json_mode: bool, started: float) -> int:
        def progress(frac: float, message: str) -> None:
            if json_mode:
                _emit_progress("check", float(frac) * 100.0, message)
            elif message:
                _write_stream(sys.stdout, f"[{int(float(frac) * 100):3d}%] {message}")

        report = (repair_fn or auto_repair.run)(progress)
        items = [{"kind": i.kind, "name": i.name, "status": i.status, "detail": i.detail} for i in report.items]
        failed = [i for i in items if i["status"] == auto_repair.FAILED]
        fixed = [i for i in items if i["status"] in (auto_repair.REPAIRED, auto_repair.DOWNLOADED)]
        warnings = [
            f"{i['kind']}: {i['name']}" + (f" ({i['detail']})" if i["detail"] else "") for i in failed
        ]
        code = EXIT_OK if report.ok else EXIT_INTERNAL
        extra = {"items": items, "checked": len(items), "fixed": len(fixed), "failed": len(failed)}
        human = [report.summary()] + [
            f"{i['status'].upper():10s} {i['kind']}: {i['name']}" + (f" - {i['detail']}" if i["detail"] else "")
            for i in items
        ]
        if json_mode:
            _emit_result(
                ok=report.ok, command="check", exit_code=code, outputs=[], warnings=warnings,
                duration_s=time.perf_counter() - started, error=None if report.ok else report.summary(),
                hint=None, extra=extra,
            )
        else:
            for line in human:
                _write_stream(sys.stdout, line)
        return code

    return _run("check", args, body)


def main(argv: Optional[Sequence[str]] = None, *,
         run_narration_fn: Callable[..., object] = run_narration,
         run_task_fn: Callable[..., object] = run_task,
         library: Optional[VoiceLibrary] = None,
         transcribe_fn: Optional[Callable[..., object]] = None,
         download_fn: Optional[Callable[..., object]] = None,
         installed_fn: Optional[Callable[[str], bool]] = None,
         plan_fn: Optional[Callable[[], object]] = None,
         repair_fn: Optional[Callable[..., object]] = None) -> int:
    """Parse ``argv`` and run the matching subcommand.  Returns the process exit code."""
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    if _flag_present(raw, "--version", "-V") and not _flag_present(raw, "--help", "-h"):
        return cmd_version(_flag_present(raw, "--json"))
    ap = build_parser()
    try:
        args = ap.parse_args(raw)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            code = 0
        if not isinstance(code, int):
            code = EXIT_BAD_ARGS
        if _flag_present(raw, "--json") and code != 0:
            _emit_result(
                ok=False, command="", exit_code=code, outputs=[], warnings=[], duration_s=0.0,
                error="invalid arguments", hint="See voxprint --help or voxprint <command> --help",
            )
        return int(code)
    handler = getattr(args, "_handler", "")
    if handler == "narrate":
        return cmd_narrate(args, run_fn=run_narration_fn, library=library, plan_fn=plan_fn)
    if handler == "train":
        return cmd_train(args, run_fn=run_task_fn)
    if handler == "voices_list":
        return cmd_voices_list(args, library=library)
    if handler == "voices_export":
        return cmd_voices_export(args, library=library)
    if handler == "diag":
        return cmd_diag(args)
    if handler == "status":
        return cmd_status(args, library=library)
    if handler == "models_list":
        return cmd_models_list(args)
    if handler == "models_download":
        return cmd_models_download(args, download_fn=download_fn, installed_fn=installed_fn)
    if handler == "revoice":
        return cmd_revoice(args, transcribe_fn=transcribe_fn)
    if handler == "speakers":
        return cmd_speakers(args, plan_fn=plan_fn)
    if handler == "check":
        return cmd_check(args, repair_fn=repair_fn)
    if handler == "backup":
        return cmd_backup(args)
    if handler == "restore":
        return cmd_restore(args)
    ap.error(f"unknown command: {handler}")
    return EXIT_BAD_ARGS


def is_user_cli(argv: Sequence[str]) -> bool:
    """True when ``argv`` (full ``sys.argv``-style, with program name) starts a user subcommand.

    Global flags (``--json``, ``--yes``, ``--version``, ``--help``) may come first.
    Maintenance flags such as ``--selftest`` stay with ``main.py``.
    """
    rest = list(argv[1:] if argv else [])
    if not rest:
        return False
    tokens = [a for a in rest if a not in _SKIP_FLAGS]
    if not tokens:
        return True
    head = tokens[0]
    if head in _ENTRY_FLAGS:
        return True
    if head.startswith("-"):
        return False
    return head in USER_COMMANDS


if __name__ == "__main__":
    raise SystemExit(main())
