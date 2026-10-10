"""User-facing command line for headless servers and agents.

Wired from ``main.py`` so both ``python main.py narrate ...`` and a packaged
``Voxprint.exe narrate ...`` work.  Subcommands call the same Qt-free runners the
GUI uses (:func:`workers.narration_runner.run_narration`,
:func:`workers.pipeline_runner.run_task`, :class:`core.voice_library.VoiceLibrary`).

Every command is non-interactive.  ``--json`` prints progress as JSON lines and a
final result object.  ``--yes`` is accepted everywhere (nothing prompts).
``--dry-run`` checks arguments, input files (including ``.vxbook``) and the output
path, prints that result, and does not load a model, use the GPU, or download.
Exit codes: 0 ok, 1 internal, 2 bad args, 3 input file, 4 missing model,
5 GPU/OOM, 6 cancelled, 7 GPU requirement (no RTX 40-series or newer).  See ``docs/CLI.md``.

Examples::

    python main.py narrate book.epub --voice my-voice --out ./audiobooks --format mp3 --json
    python main.py train ./clips --name Anna --type female --json
    python main.py status --json
    python main.py --version

The developer dataset CLI remains ``python -m core.cli`` (see ``core/cli.py``).
"""
from __future__ import annotations

import sys

from infra.gpu_requirement import EXIT_GPU_REQUIRED, enforce, startup_requires_gpu

# Refuse before the heavy imports below when this file is the process entry.
# ``main.py`` checks first and then imports this module, so that path is unchanged.
if __name__ == "__main__" and startup_requires_gpu(sys.argv):
    _gpu_block = enforce(gui=False)
    if _gpu_block:
        raise SystemExit(_gpu_block)

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Set, cast

from core import audiobook_export as ex
from core import pauses as pz
from core import ordinals
from core import pace as pc
from core import revoice
from core import speakers as spk
from core.translate import TranslateError, detect_book_language, route
from core import workspace as ws
from core import appinfo
from core.build_stamp import read_trailer
from infra.stdio_guard import guard_stdio, install_cli_excepthook
from core.asr import make_default_asr
from core.book_parsers import load_book
from core.dry_run import DryRunFailure, check_output, existing_input, load_book_input, schema_error
from core.book_prep import PrepPlan
from core.text_prep import STEP_KEYS, STEP_YO, PrepOptions, resolve_language
from core.errors import BackupError, CancelledByUser, DatasetMakerError, OutOfMemoryError_
from core.events import CancelToken, Stage, overall_percent
from core.narration import NarrationOptions, NarrationProgress, PauseToken
from core import soundscape as soundscape_mod
from core import vram_policy
from core.voice_info import VOICE_TYPES, normalize_voice_type
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import auto_repair
from infra import backup as backup_mod
from infra import denoise_tool, diagnostics, keep_awake, llm_tool, projects, quality_models, soundscape_model, text_models, vc_model
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
    "speakers", "check", "repair", "prepare", "translate", "settings", "bench",
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
_SKIP_FLAGS = {"--json", "--yes", "-y", "--dry-run"}
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
    if code == EXIT_GPU_REQUIRED:
        return "An NVIDIA GeForce RTX 40-series or newer GPU is required. Check: voxprint status --json"
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
    """Version, build number and codename.

    A stamp on the running executable (an exe-only patch) wins over ``credits.json`` and over ``BUILD.json``.
    With no stamp, the codename comes from ``BUILD.json`` when that file is present.
    """
    appinfo.apply_embedded_stamp()
    meta = _read_build_json()
    trailer = read_trailer(sys.executable)
    if trailer is not None:
        build, name = trailer
        codename = name or str(meta.get("codename") or appinfo.APP_CODENAME or "")
    else:
        codename = str(meta.get("codename") or appinfo.APP_CODENAME or "")
        build = int(appinfo.APP_BUILD or 0)
        if not build and build_number is not None:
            try:
                build = int(build_number())
            except (OSError, ValueError, TypeError, KeyError):
                build = 0
    version = appinfo.release_version()
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

    A packaged ``Voxprint.exe`` is a windowed program. Stdout may be missing, or it may be a live object whose
    ``write`` fails with ``OSError`` errno 22 (invalid handle) when PowerShell redirects a windowed exe. Either way
    the exit code must still be the command's code, and the error must not escape into a traceback window.
    """
    try:
        stream.write(text if str(text).endswith("\n") else str(text) + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001 - a dead console, including errno 22, must not escape
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
    """``--json`` / ``--yes`` / ``--dry-run`` / ``--version`` on one parser.

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
        "--dry-run", dest="dry_run", action="store_true", default=default,
        help="Check arguments, input files and the output path, then print the result and exit. "
             "Does not load models, use the GPU, or download anything. Works without an RTX GPU",
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
            "  voxprint prepare book.txt --out book.prepared.txt --json\n"
            "  voxprint translate book.txt --to ru --out book.ru.txt --json\n"
            "  voxprint settings list --json\n"
            "  voxprint status --dry-run --json\n"
            "  voxprint diag --out report.zip\n"
            "  voxprint backup --out E:\\ --json\n"
            "  voxprint restore --from E:\\ --json\n"
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    n = sub.add_parser(
        "narrate", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Narrate a book with a trained voice",
        description="Narrate a TXT, Markdown, FB2, FB2.ZIP, EPUB or .vxbook file with a voice from the library.",
        epilog=(
            "Examples:\n"
            "  voxprint narrate book.epub --voice my-voice --out ./audiobooks\n"
            "  voxprint narrate book.vxbook --voice narrator --out ./audiobooks --dry-run --json\n"
            "  voxprint narrate book.fb2 --voice my-voice --out ./audiobooks --format mp3,m4b,flac,opus --json\n"
            "  voxprint narrate book.txt --voice my-voice --out ./audiobooks --pauses --ai-disclosure\n"
            "  voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speakers --out ./audiobooks --json\n"
            "  voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann --speaker-marks marks.txt --out ./audiobooks\n"
            "  python cli.py narrate book.md --voice narrator --male-voice tom --female-voice ann --character Ivan=tom --character Anna=ann --speaker-marks marks.txt --out ./audiobooks --format wav --json\n"
            "  voxprint narrate book.txt --voice levi --speakers --out ./out\n"
            "  voxprint narrate book.txt --voice levi --male-voice natan --male2-voice shimon --female-voice noa --speakers --out ./out\n"
            "  voxprint narrate book.txt --voice levi --male-voice natan --character David=shimon --speakers --out ./out\n"
        ),
    )
    n.add_argument("book", help="Path to a TXT, Markdown, FB2, FB2.ZIP, EPUB or .vxbook file")
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
    yo_flag = n.add_mutually_exclusive_group()
    yo_flag.add_argument("--yo", dest="yo", action="store_true", default=None,
                         help="Restore the Russian letter yo where a dictionary is sure (default: on). "
                              "A word that already contains yo, and a U+0301 stress mark the author wrote, stay")
    yo_flag.add_argument("--no-yo", dest="yo", action="store_false",
                         help="Leave the letter e as written; do not restore yo")
    n.set_defaults(yo=None)
    n.add_argument("--ai-disclosure", action="store_true",
                   help="Speak a short AI disclosure at the start (opt-in)")
    n.add_argument("--speakers", action="store_true",
                   help="Mark each paragraph narrator, male or female with the text model (Gemma), then narrate those voices. "
                        "Without --male-voice / --female-voice / --character the shipped cast is used when installed "
                        "(men Natan and Shimon, woman Miriam), else the first library voices of each gender")
    n.add_argument("--male-voice", default="", metavar="ID_OR_NAME",
                   help="Voice for paragraphs marked male (narrator is --voice)")
    n.add_argument("--female-voice", default="", metavar="ID_OR_NAME",
                   help="Voice for paragraphs marked female (narrator is --voice)")
    n.add_argument("--male2-voice", default="", metavar="ID_OR_NAME",
                   help="Second male voice: different male characters alternate between --male-voice and this one "
                        "in order of first appearance")
    n.add_argument("--female2-voice", default="", metavar="ID_OR_NAME",
                   help="Second female voice, alternating with --female-voice the same way")
    n.add_argument("--character", action="append", default=[], metavar="NAME=VOICE", dest="characters",
                   help="Pin one character (the name as written in the marks) to a voice. Repeatable; "
                        "wins over the male / female voices")
    n.add_argument("--speaker-marks", type=Path, default=None, metavar="FILE",
                   help="Narrate from this marks file instead of running Gemma (voxprint speakers writes it). "
                        "One line per paragraph: 'N. NARRATOR', 'N. MALE: Name', 'N. FEMALE: Name', "
                        "or the same lines without the numbers")
    n.add_argument("--no-soundscape", action="store_true",
                   help="Do not mix a soundscape on this run. Does not change the saved setting and does not download the model. "
                        "A soundscape plays only for a .vxbook that declares extension sound/1, and only when the setting is on")
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
            "  voxprint train ./clips --name MyVoice --language ru --consent commercial --speaker \"Reader Name\" --license CC0-1.0\n"
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
    t.set_defaults(_handler="train")

    v = sub.add_parser(
        "voices", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="List, export or download voices",
        description="List installed voices, export one as a zip, or list and download the voices of the online catalog.",
        epilog=(
            "Examples:\n"
            "  voxprint voices list\n"
            "  voxprint voices list --json\n"
            "  voxprint voices export my-voice --out my-voice.zip\n"
            "  voxprint voices catalog\n"
            "  voxprint voices download eitan --json\n"
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
    vsub.add_parser(
        "catalog", parents=[child], help="List the voices of the online voice catalog (installed or not)",
        description="Fetch the voice catalog (voices/index.json; the cached copy when offline) and list its voices.",
        epilog="Examples:\n  voxprint voices catalog\n  voxprint voices catalog --json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).set_defaults(_handler="voices_catalog")
    vd = vsub.add_parser(
        "download", parents=[child], help="Download a catalog voice into the library (hash-checked, resumable)",
        description="Download one voice of the catalog by its id or name, check its SHA-256 and import it. "
                    "A voice already in the library is skipped.",
        epilog="Examples:\n  voxprint voices download eitan --json\n  voxprint voices download Noa\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    vd.add_argument("voice", metavar="VOICE", help="Catalog id or name (see: voxprint voices catalog)")
    vd.set_defaults(_handler="voices_download")

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
    rv.add_argument("--language", default=None,
                    help="Recognition language hint: ISO code or name, any case (ru, en, de, Russian ...; default: automatic)")
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
    sp.add_argument("book", help="Path to a TXT, Markdown, FB2, FB2.ZIP or EPUB file")
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

    pr = sub.add_parser(
        "prepare", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Prepare a book's text for narration (rules, letter yo, typo fix; optional text-model rewrite)",
        description=(
            "The Narrate window's Prepare text step without narration: layout, footnotes, quotes, links, headings, "
            "numbers, abbreviations and (Russian) the letter yo, then the Russian typo model when it is downloaded. "
            "--llm adds the text model's narration rewrite. Writes plain text and a JSON report."
        ),
        epilog=(
            "Steps: " + ", ".join(STEP_KEYS) + "\n"
            "Examples:\n"
            "  voxprint prepare book.txt --out book.prepared.txt --json\n"
            "  voxprint prepare book.fb2 --out prepared.txt --report report.json --no-typos\n"
            "  voxprint prepare book.txt --out yo-only.txt --steps yo\n"
            "  voxprint prepare book.txt --out prepared.txt --llm --json\n"
        ),
    )
    pr.add_argument("book", help="Path to a TXT, Markdown, FB2, FB2.ZIP or EPUB file")
    pr.add_argument("--out", required=True, type=Path, metavar="FILE", help="Prepared text file to write (UTF-8)")
    pr.add_argument("--report", type=Path, default=None, metavar="FILE",
                    help="JSON report to write (default: <out stem>.prep_report.json next to --out)")
    pr.add_argument("--language", default="", metavar="CODE", help="Book language hint: ru, en, de (default: detected)")
    pr.add_argument("--steps", action="append", default=[], metavar="NAME",
                    help="Only these rule steps (repeatable or comma-separated; default: all)")
    pr.add_argument("--no-rules", action="store_true", help="Skip every rule step")
    pr_yo = pr.add_mutually_exclusive_group()
    pr_yo.add_argument("--yo", dest="yo", action="store_true", default=None, help="Restore the Russian letter yo (default: on)")
    pr_yo.add_argument("--no-yo", dest="yo", action="store_false", help="Leave the letter e as written")
    pr_typo = pr.add_mutually_exclusive_group()
    pr_typo.add_argument("--typos", dest="typos", action="store_true", default=None,
                         help="Require the Russian typo model (exit 4 when it is not downloaded)")
    pr_typo.add_argument("--no-typos", dest="typos", action="store_false",
                         help="Skip the typo model (default: used when downloaded, unless --steps is given)")
    pr.add_argument("--llm", "--markup", dest="llm", action="store_true",
                    help="Also run the text model's narration rewrite (Gemma; off by default)")
    pr.add_argument("--work-dir", type=Path, default=None, metavar="DIR",
                    help="Keep the text model's cache here (default: a temporary folder)")
    pr.set_defaults(_handler="prepare")

    tr_p = sub.add_parser(
        "translate", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Translate a book offline (Opus-MT; --literary uses the text model)",
        description="Translate a book with the offline models the Narrate window uses and write plain text. Does not narrate.",
        epilog=(
            "Examples:\n"
            "  voxprint translate book.epub --to ru --out book.ru.txt --json\n"
            "  voxprint translate book.txt --from de --to en --out book.en.txt\n"
            "  voxprint translate book.txt --to ru --out book.ru.txt --literary\n"
        ),
    )
    tr_p.add_argument("book", help="Path to a TXT, Markdown, FB2, FB2.ZIP or EPUB file")
    tr_p.add_argument("--to", required=True, choices=("en", "ru", "de"), help="Target language")
    tr_p.add_argument("--from", dest="source", default="", choices=("", "en", "ru", "de", "uk"),
                      help="Source language (default: detected)")
    tr_p.add_argument("--out", required=True, type=Path, metavar="FILE", help="Translated text file to write (UTF-8)")
    tr_p.add_argument("--literary", action="store_true",
                      help="Literary translation by the text model (Gemma), Opus-MT as the fallback")
    tr_p.add_argument("--work-dir", type=Path, default=None, metavar="DIR",
                      help="Keep the sentence cache here, so a second run is quick (default: a temporary folder)")
    tr_p.set_defaults(_handler="translate")

    se = sub.add_parser(
        "settings", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Show or change the saved settings",
        description="The values Settings shows: language, projects folder, reading speed and style, pauses, ordinals.",
        epilog=(
            "Examples:\n"
            "  voxprint settings list --json\n"
            "  voxprint settings get narration.speed\n"
            "  voxprint settings set narration.ordinals off\n"
            "  voxprint settings set narration.pause.sentence 0.8\n"
        ),
    )
    se_sub = se.add_subparsers(dest="settings_action", required=True)
    se_sub.add_parser("list", parents=[child], help="Every setting and its value")
    se_get = se_sub.add_parser("get", parents=[child], help="One setting")
    se_get.add_argument("key", metavar="KEY")
    se_set = se_sub.add_parser("set", parents=[child], help="Change one setting")
    se_set.add_argument("key", metavar="KEY")
    se_set.add_argument("value", metavar="VALUE")
    se.set_defaults(_handler="settings")

    bn = sub.add_parser(
        "bench", parents=[child], formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Speed test: batched generation vs CUDA Graphs (realtime factor, peak VRAM)",
        description="Narrate a fixed Russian text (12 phrases) once per mode with the same voice and print the realtime "
                    "factor and the peak video memory. 'batched' = several phrases per generate call (the default path); "
                    "'graphs' = one phrase at a time through faster-qwen3-tts with CUDA Graphs (optional module). "
                    "Each mode loads the model itself; nothing is saved unless --out is given.",
        epilog=(
            "Examples:\n"
            "  voxprint bench\n"
            "  voxprint bench --voice Levi --modes batched,graphs --out ./bench --json\n"
            "  voxprint bench --install-graphs\n"
        ),
    )
    bn.add_argument("--voice", default="Levi", metavar="VOICE", help="Voice id or name (default: Levi)")
    bn.add_argument("--modes", default="batched,graphs", metavar="LIST", help="Comma-separated: batched, graphs")
    bn.add_argument("--out", type=Path, default=None, metavar="DIR", help="Also save bench-<mode>.wav here to listen")
    bn.add_argument("--install-graphs", action="store_true",
                    help="First download faster-qwen3-tts 0.3.2 (MIT, 43 KB, SHA-256 checked) into the packages folder")
    bn.set_defaults(_handler="bench")
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
        # Windows folders are case-insensitive: get("Bob") opens the folder "bob" but names the record "Bob".
        # Return the library's own record so one voice has one id (one engine, one cast entry).
        listed = library.list_voices()
        same = [v for v in listed if v.id == rec.id] or [v for v in listed if v.id.lower() == rec.id.lower()]
        return same[0] if len(same) == 1 else rec
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
            extra: dict[str, Any] = {"done": p.done, "total": p.total}
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
                hint="Pass a TXT, Markdown, FB2, FB2.ZIP, EPUB or .vxbook file that exists. "
                     "Example: voxprint narrate book.epub --voice my-voice --out ./audiobooks",
            )
        book = load_book(book_path)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.work_dir is not None:
            ws.save_folder(Path(args.work_dir))
        formats = parse_formats(args.formats)
        lengths, pace = narration_shaping(args)
        yo_on = True if getattr(args, "yo", None) is None else bool(args.yo)
        if book.explicit_yo:
            yo_on = False
        options = NarrationOptions(
            formats=formats,
            pauses=pz.PauseProfile(lengths=lengths) if args.pauses else None,
            pause_lengths=lengths, pace=pace,
            ai_disclosure=bool(args.ai_disclosure),
            ordinals=ordinals.load_enabled() if getattr(args, "ordinals", None) is None else bool(args.ordinals),
            yo=yo_on,
            # Yo only. The rest of Prepare text stays a window switch; this plan writes the restored book to .debug.
            prep=PrepPlan(rules=PrepOptions(frozenset({STEP_YO}))) if yo_on else None,
            sound=(soundscape_mod.SoundRequest()
                   if soundscape_model.enabled() and not getattr(args, "no_soundscape", False)
                   and soundscape_mod.requested(book) else None),
        )
        cast, extra_voices = _speaker_job(args, voice, lib, plan_fn, book)
        if cast is not None:
            options.speakers = cast
        job = NarrationJob(book=book, voice=voice, out_dir=out_dir, options=options, extra_voices=extra_voices)
        with keep_awake.keep_awake():
            result = run_fn(job, _narration_progress(json_mode), CancelToken(), PauseToken())
        files = list(getattr(result, "files", None) or [])
        done = Path(getattr(result, "out_dir", out_dir))
        debug = done / ".debug"
        for name in ("speakers.txt", "speakers-raw.txt"):
            marks = debug / name
            if marks.is_file():
                files.append(marks)
        warnings = speaker_warning_lines(getattr(result, "speaker_warning", ""))
        human = [f"Done: {done}"] + [f"  {f}" for f in files] + [f"  ! {w}" for w in warnings]
        return _ok(
            json_mode, "narrate", started, outputs=files, warnings=warnings,
            human=human, extra={"out_dir": str(done)},
        )

    return _run("narrate", args, body)


SPEAKER_MISMATCH = "Speaker marks do not match the prepared text, so the narrator reads the whole book."
SPEAKER_UNPARSED = (
    "The text model's reply could not be read as speaker marks, so the narrator is used for every paragraph."
)
SPEAKER_NO_SPEAKERS = "The text has dialogue, but the text model marked every paragraph as the narrator."
_SPEAKER_WARNING_TEXT = {
    "mismatch": SPEAKER_MISMATCH,
    "unparsed": SPEAKER_UNPARSED,
    "no_speakers": SPEAKER_NO_SPEAKERS,
}


def speaker_warning_lines(code: str) -> list:
    """English warning sentences for a ``speaker_warning`` code (several codes are joined with ``;``)."""
    out = []
    for part in (code or "").split(";"):
        text = _SPEAKER_WARNING_TEXT.get(part.strip())
        if text and text not in out:
            out.append(text)
    return out


def _speaker_job(args, narrator, library: VoiceLibrary, plan_fn: Optional[Callable[[], object]], book=None):
    """Speaker cast and extra voices for ``narrate``, or ``(None, {})`` when multi-voice is off.

    The cast is the same object the Narrate window builds (:class:`core.speakers.SpeakerCast`). Narration applies it.
    A ``.vxbook`` already carries marks and a cast. ``--voice`` is the narrator. ``--character`` wins over ``cast.json``.
    ``--speakers`` does not call the text model for a book that already has marks.
    """
    vx_marks = tuple(getattr(book, "speaker_marks", ()) or ())
    vx = bool(vx_marks) and getattr(args, "speaker_marks", None) is None
    want = bool(getattr(args, "speakers", False)) or getattr(args, "speaker_marks", None) is not None or vx
    names = {flag: (getattr(args, attr, "") or "").strip() for flag, attr in (
        ("--male-voice", "male_voice"), ("--female-voice", "female_voice"),
        ("--male2-voice", "male2_voice"), ("--female2-voice", "female2_voice"))}
    raw_characters = list(getattr(args, "characters", None) or [])
    if not want and not any(names.values()) and not raw_characters:
        return None, {}
    if not want:
        raise CliError(
            EXIT_BAD_ARGS, "--male-voice, --female-voice, --male2-voice, --female2-voice and --character need "
                           "--speakers or --speaker-marks",
            hint="Example: voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann "
                 "--speakers --out ./audiobooks",
        )
    if args.speakers and args.speaker_marks is not None:
        raise CliError(
            EXIT_BAD_ARGS, "--speakers and --speaker-marks cannot be used together",
            hint="Use --speakers to ask Gemma, or --speaker-marks FILE to narrate marks you already edited.",
        )
    for second, first in (("--male2-voice", "--male-voice"), ("--female2-voice", "--female-voice")):
        if names[second] and not names[first]:
            raise CliError(
                EXIT_BAD_ARGS, f"{second} needs {first}",
                hint="Example: voxprint narrate book.txt --voice gideon --male-voice asher --male2-voice tom "
                     "--speakers --out ./out",
            )
    try:
        character_names = spk.parse_character_map(raw_characters)
    except ValueError as exc:
        raise CliError(EXIT_BAD_ARGS, f"--character: {exc}",
                       hint="Example: --character David=asher --character Hannah=noa") from exc
    if not any(names.values()) and not raw_characters and not getattr(book, "voice_cast", None):
        # --speakers alone: the shipped cast (Natan, Shimon, Miriam) when installed, else the first voices by gender
        from infra import bundled_voices

        offered = bundled_voices.offered_for_roles(library.list_voices())
        picks = spk.default_role_picks(offered, narrator.id, bundled_voices.preferred_ids(offered))
        names["--male-voice"], names["--male2-voice"], names["--female-voice"] = picks["male"], picks["male2"], picks["female"]
        if not names["--male-voice"] and names["--male2-voice"]:
            names["--male-voice"], names["--male2-voice"] = names["--male2-voice"], ""
    recs = {flag: (resolve_voice(library, name) if name else None) for flag, name in names.items()}
    characters = {who: resolve_voice(library, vname) for who, vname in character_names.items()}
    pinned: dict = {}
    if vx and book is not None:
        pinned = dict(book.voice_cast)
        for who, rec in characters.items():
            canon = book.alias_to_name.get(who, who)
            pinned[canon] = rec.id
            pinned[who] = rec.id
            for key in list(pinned):
                if key.casefold() == who.casefold() or key.casefold() == canon.casefold():
                    pinned[key] = rec.id
        for cast_id in set(pinned.values()):
            try:
                resolve_voice(library, cast_id)
            except CliError as exc:
                raise CliError(
                    EXIT_INPUT, f"cast voice not found: {cast_id}",
                    hint="Install that voice, or override it with --character Name=voice. "
                         "voxprint voices list --json",
                ) from exc
    elif characters:
        pinned = {who: rec.id for who, rec in characters.items()}
    lines = None
    tagger = None
    if vx and args.speakers:
        log.warning("speaker marks are already in the .vxbook; --speakers will not call the text model")
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
                hint="Each line is 'NARRATOR', 'MALE: Name' or 'FEMALE: Name'. A leading 'N. ' is optional.",
            ) from exc
    elif vx:
        lines = [spk.SpeakerLine(role, name) for role, name in vx_marks]
    else:
        plan = (plan_fn or llm_tool.make_plan)()
        if plan is None:
            raise CliError(
                EXIT_MISSING, "text model is not installed",
                hint="voxprint models download llm --json",
            )
        tagger = plan

    def vid(flag: str) -> str:
        rec = recs[flag]
        return rec.id if rec is not None else ""

    speaker_cast = spk.SpeakerCast(
        lines=lines, male_id=vid("--male-voice"), female_id=vid("--female-voice"),
        tagger=cast(Any, tagger), narrator_id=narrator.id,
        male2_id=vid("--male2-voice"), female2_id=vid("--female2-voice"),
        characters=pinned,
    )
    if not speaker_cast.uses_several(narrator.id):
        if vx:
            return None, {}
        raise CliError(
            EXIT_BAD_ARGS, "pick a male or female voice that is not the narrator",
            hint="Example: voxprint narrate book.txt --voice narrator --male-voice tom --female-voice ann "
                 "--speakers --out ./audiobooks",
        )
    extra = {}
    for maybe in [*recs.values(), *characters.values()]:
        if maybe is not None and maybe.id != narrator.id:
            extra[maybe.id] = maybe
    for pinned_id in pinned.values():
        if pinned_id and pinned_id != narrator.id and pinned_id not in extra:
            extra[pinned_id] = resolve_voice(library, pinned_id)
    return speaker_cast, extra


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


def _catalog_or_fail(fetch_fn=None):
    """The catalog voices (``(voices, offline)``); exit 4 when it can be neither fetched nor read from the cache."""
    from infra import voice_repository as repo

    res = (fetch_fn or repo.fetch_index)()
    if not res.voices:
        raise CliError(EXIT_MISSING, f"voice catalog not available ({res.error or 'empty'})",
                       hint="Check the internet connection, then: voxprint voices catalog", details=res.detail or "")
    return res.voices, bool(res.offline)


def cmd_voices_catalog(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None, fetch_fn=None) -> int:
    """``voices catalog``: id, name, gender, language, licence, size and whether it is installed."""

    from infra import voice_repository as repo

    def body(json_mode: bool, started: float) -> int:
        lib = library if library is not None else VoiceLibrary()
        voices, offline = _catalog_or_fail(fetch_fn)
        have = repo.installed_entry_ids(voices, lib.list_voices())   # by repo_id, else a local voice with the same id / name
        rows = [{"id": e.id, "name": e.name, "language": e.language, "gender": e.gender or "", "license": e.license,
                 "size_bytes": e.size_bytes, "bundled": bool(e.bundled), "installed": e.id in have}
                for e in voices if e.id in have or not e.hidden]   # retired voices only when still installed
        warnings = ["The catalog could not be fetched; this is the cached copy."] if offline else []
        human = [f"{r['id']}\t{r['name']}\t{r['gender'] or '-'}\t{r['language'] or '-'}\t{r['license']}\t"
                 f"{r['size_bytes'] / 1e6:.0f} MB\t{'installed' if r['installed'] else '-'}" for r in rows]
        return _ok(json_mode, "voices catalog", started, outputs=[], warnings=warnings,
                   human=human + [f"! {w}" for w in warnings], extra={"voices": rows})

    return _run("voices catalog", args, body)


def cmd_voices_download(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None, fetch_fn=None,
                        download_fn=None) -> int:
    """``voices download VOICE``: download a catalog voice (SHA-256 checked) and import it into the library."""
    from infra import voice_repository as repo

    def body(json_mode: bool, started: float) -> int:
        lib = library if library is not None else VoiceLibrary()
        voices, _offline = _catalog_or_fail(fetch_fn)
        want = str(args.voice).strip().lower()
        entry = next((e for e in voices if e.id.lower() == want), None) or \
            next((e for e in voices if e.name.strip().lower() == want), None)
        if entry is None:
            raise CliError(EXIT_INPUT, f"voice {args.voice!r} is not in the catalog",
                           hint="List the catalog: voxprint voices catalog")
        existing = repo.installed_record(entry, lib.list_voices())
        if existing is not None:
            return _ok(json_mode, "voices download", started, outputs=[], human=[f"Already installed: {existing.name} ({existing.id})"],
                       extra={"voice": _voice_payload(existing), "downloaded": False})
        if json_mode:
            _emit_progress("download", 0.0, f"Downloading {entry.name}")

        def progress(frac: float, name: str) -> None:
            if json_mode:
                _emit_progress("download", float(frac) * 100.0, f"Downloading {name}")

        rec = (download_fn or repo.download_voice)(entry, lib, progress=progress)
        return _ok(json_mode, "voices download", started, outputs=[], human=[f"Installed: {rec.name} ({rec.id})"],
                   extra={"voice": _voice_payload(rec), "downloaded": True})

    return _run("voices download", args, body)


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
            "requirement": gpu_raw.get("requirement"),
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


def revoice_language(value: Optional[str]) -> Optional[str]:
    """``--language`` of ``revoice`` as the English name Qwen3-ASR expects, or ``None`` for automatic detection.

    Accepts ISO codes in any case (``ru``, ``EN``, ``de-DE``, ``rus``), English names (``Russian``) and native names
    (``русский``); ``auto`` or empty means automatic. Build 702 passed the raw value through, so ``--language ru`` failed in
    the recogniser. Raises a usage error for a value that is not a language.
    """
    from core import languages

    raw = (value or "").strip()
    if not raw or raw.lower() == "auto":
        return None
    name = languages.language_name(raw)
    if not name or name == languages.language_code(raw):     # unknown code: language_name echoes the code back
        raise CliError(
            EXIT_BAD_ARGS, f"--language: unknown language {raw!r}",
            hint="Use an ISO code or name, e.g. --language ru, --language en, --language German, or leave it out (automatic).",
        )
    return name


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
        chapters = list(cast(Iterable[Any], fn(files, revoice_language(args.language), progress, CancelToken())))
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

        tagged = spk.tag_paragraphs(paras, detect_book_language(book) or "en", cast(Any, plan), progress)
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(spk.dump_marks(tagged.lines), encoding="utf-8")
        raw_path = dest.with_name(f"{dest.stem}.speakers-raw.txt")
        raw_path.write_text(tagged.raw, encoding="utf-8")
        warnings = speaker_warning_lines(tagged.warning)
        human = [f"Marks: {dest}", f"  {len(tagged)} paragraphs"] + [f"  ! {w}" for w in warnings]
        return _ok(
            json_mode, "speakers", started, outputs=[dest, raw_path], warnings=warnings,
            human=human, extra={"paragraphs": len(tagged)},
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

        report = cast(auto_repair.Report, (repair_fn or auto_repair.run)(progress))
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


# ----------------------------------------------------------------------------------------- prepare / translate / settings
def _load_book_or_fail(path_text: str, example: str):
    book_path = Path(path_text)
    if not book_path.is_file():
        raise CliError(EXIT_INPUT, f"book not found: {book_path}", hint=f"Pass a TXT, Markdown, FB2, FB2.ZIP or EPUB file. Example: {example}")
    return load_book(book_path)


def book_plain_text(book, like=None) -> str:
    """The book as plain text: chapters separated by two blank lines, a chapter title on its own line above its text.

    A chapter whose title equals the book title (a one-chapter TXT is titled after its file) gets no title line.
    ``like`` is the book before preparation or translation, whose titles decide that (the prepared title may differ).
    """
    src = like if like is not None else book
    book_title = (src.title or "").strip()
    parts = []
    for i, ch in enumerate(book.chapters):
        title = (ch.title or "").strip()
        orig = src.chapters[i].title.strip() if i < len(src.chapters) and src.chapters[i].title else title
        text = (ch.text or "").strip()
        if title and orig != book_title:
            parts.append(f"{title}\n\n{text}" if text else title)
        elif text:
            parts.append(text)
    return "\n\n\n".join(parts) + ("\n" if parts else "")


def _write_text(dest: Path, text: str) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return dest


def _prep_steps(args) -> frozenset:
    """Rule steps from ``--steps`` / ``--no-rules`` / ``--no-yo`` (default: all, as the Prepare text switch)."""
    from core import text_prep as tp

    steps: set[str] = set()
    if getattr(args, "no_rules", False):
        pass
    elif getattr(args, "steps", None):
        steps = set()
        for item in args.steps:
            for part in str(item).split(","):
                key = part.strip().lower()
                if not key:
                    continue
                if key not in tp.STEP_KEYS:
                    raise CliError(EXIT_BAD_ARGS, f"unknown step {part!r}",
                                   hint=f"Choose from: {', '.join(tp.STEP_KEYS)}")
                steps.add(key)
    else:
        steps = set(tp.STEP_KEYS)
    if getattr(args, "yo", None) is False:
        steps.discard(tp.STEP_YO)
    elif getattr(args, "yo", None) is True:
        steps.add(tp.STEP_YO)
    return frozenset(steps)


def _typo_state(language: str):
    """``(model, ready)`` of the Russian typo model for ``language`` (``(None, False)`` when no model covers it)."""
    model = text_models.for_step(text_models.STEP_SPELLFIX, language)
    if model is None or not model.integrated:
        return None, False
    return model, text_models.state(model) == text_models.STATE_READY


def cmd_prepare(args: argparse.Namespace, *,
                plan_fn: Optional[Callable[[], object]] = None,
                typo_state_fn: Optional[Callable[[str], tuple]] = None,
                cleanup_factory: Optional[Callable[[str], object]] = None) -> int:
    """``prepare BOOK --out FILE``: the Narrate window's Prepare text (rules, letter yo, Russian typo model) and,
    with ``--llm``, the text model's narration rewrite. Writes plain text and a JSON report; does not narrate."""
    from core import llm_text
    from core.book_prep import NEURAL_SPELLFIX, run_preparation
    from core.text_prep import resolve_language

    example = "voxprint prepare book.txt --out book.prepared.txt --json"

    def body(json_mode: bool, started: float) -> int:
        book = _load_book_or_fail(args.book, example)
        language = resolve_language(book, args.language or "")
        steps = _prep_steps(args)
        warnings: list = []
        neural: frozenset = frozenset()
        typo_model = None
        # The typo model runs on a full preparation (no --steps) when it is downloaded, and with --steps only when
        # --typos asks for it: "--steps yo" must change nothing but the letter yo.
        use_typos = args.typos is True or (args.typos is None and not getattr(args, "steps", None))
        if use_typos:
            model, ready = (typo_state_fn or _typo_state)(language)
            if model is not None and ready:
                neural = frozenset({NEURAL_SPELLFIX})
                typo_model = getattr(model, "key", str(model))
            elif args.typos is True:
                if model is None:
                    raise CliError(EXIT_BAD_ARGS, f"no typo model for language {language or '?'}",
                                   hint="The typo model covers Russian books only. Drop --typos.")
                raise CliError(EXIT_MISSING, "the Russian typo model is not installed",
                               hint=f"voxprint models download {getattr(model, 'key', 'sage-ru')} --json")
            elif model is not None:
                warnings.append(f"Typo fix skipped: the model {getattr(model, 'key', '')} is not downloaded "
                                f"(voxprint models download {getattr(model, 'key', '')}).")
        factory = cleanup_factory or (text_models.cleanup_engine_for if neural else None)
        plan = PrepPlan(PrepOptions(steps), neural, cast(Any, factory))
        if json_mode:
            _emit_progress("prepare", 0.0, "Preparing the text")

        def progress(frac: float, message: str) -> None:
            if json_mode:
                _emit_progress("prepare", float(frac) * 100.0, message or "Preparing the text")

        prepared, report = run_preparation(book, plan, language, progress=progress)
        report = dict(report)
        report["typo_model"] = typo_model or ""
        llm_tag = ""
        if args.llm:
            llm_plan = (plan_fn or llm_tool.make_plan)()
            if llm_plan is None:
                raise CliError(EXIT_MISSING, "text model is not installed", hint="voxprint models download llm --json")
            import tempfile

            with tempfile.TemporaryDirectory(prefix="voxprint-prepare-") as tmp:
                work = Path(args.work_dir) if args.work_dir else Path(tmp)
                prepared = llm_text.prepare_book(
                    prepared, cast(Any, llm_plan), report.get("language") or language or "en", work,
                    lambda f: progress(0.5 + 0.5 * float(f), "Text model: narration rewrite"))
            llm_tag = str(getattr(llm_plan, "tag", "") or "text model")
        report["llm"] = llm_tag
        dest = _write_text(Path(args.out), book_plain_text(prepared, book))
        report_path = Path(args.report) if args.report else dest.with_name(dest.stem + ".prep_report.json")
        report_doc = {"input": str(Path(args.book)), "output": str(dest), **report}
        _write_text(report_path, json.dumps(report_doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        counts = report.get("rule_counts") or {}
        human = [f"Prepared: {dest}", f"Report: {report_path}", f"  language: {report.get('language') or '?'}",
                 "  steps: " + (", ".join(report.get("rules") or []) or "none")]
        if counts:
            human.append("  changes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        human += [f"  ! {w}" for w in warnings]
        return _ok(json_mode, "prepare", started, outputs=[dest, report_path], warnings=warnings, human=human,
                   extra={"report": report_doc})

    return _run("prepare", args, body)


def _translate_ready(source: str, target: str) -> list:
    """Module ids of the Opus-MT models still missing for ``source -> target`` (``[]`` = ready)."""
    missing = []
    for model in text_models.translate_models(source, target):
        if text_models.state(model) != text_models.STATE_READY:
            missing.append(model.key)
    return missing


def cmd_translate(args: argparse.Namespace, *,
                  factory: Optional[Callable[[str, str], object]] = None,
                  plan_fn: Optional[Callable[[], object]] = None,
                  missing_fn: Optional[Callable[[str, str], list]] = None) -> int:
    """``translate BOOK --to LANG --out FILE``: the offline translation the Narrate window runs (Opus-MT; with
    ``--literary`` the text model translates paragraphs and Opus-MT is the fallback). Does not narrate."""
    from core import translate as tl

    example = "voxprint translate book.txt --to ru --out book.ru.txt --json"

    def body(json_mode: bool, started: float) -> int:
        book = _load_book_or_fail(args.book, example)
        target = args.to
        source = args.source or tl.detect_book_language(book)
        if not source:
            raise CliError(EXIT_INPUT, "the book language could not be detected", hint="Pass --from en|ru|de|uk")
        if source == target:
            raise CliError(EXIT_BAD_ARGS, f"the book is already in {target}", hint="Choose another --to language.")
        try:
            tl.route(source, target)
        except tl.TranslateError as exc:
            raise CliError(EXIT_BAD_ARGS, f"unsupported language pair {source} -> {target}",
                           hint="Supported: en, ru, de (and uk as a source).") from exc
        if factory is None:
            try:
                missing = (missing_fn or _translate_ready)(source, target)
            except ValueError as exc:
                raise CliError(EXIT_BAD_ARGS, str(exc)) from exc
            if missing:
                raise CliError(EXIT_MISSING, "translation model not installed: " + ", ".join(missing),
                               hint="voxprint models download " + missing[0] + " --json")
        llm = None
        if args.literary:
            llm = (plan_fn or llm_tool.make_plan)()
            if llm is None:
                raise CliError(EXIT_MISSING, "text model is not installed", hint="voxprint models download llm --json")
        plan = tl.TranslatePlan(target, source, cast(Any, factory or text_models.make_translator), llm=llm)
        if json_mode:
            _emit_progress("translate", 0.0, "Translating")

        def progress(frac: float, message: str) -> None:
            if json_mode:
                _emit_progress("translate", float(frac) * 100.0, message or "Translating")

        import tempfile

        with tempfile.TemporaryDirectory(prefix="voxprint-translate-") as tmp:
            work = Path(args.work_dir) if args.work_dir else Path(tmp)
            work.mkdir(parents=True, exist_ok=True)
            cache = tl.TranslationCache(work / "translation_cache.json")
            translated = tl.translate_book(book, plan, source, cache, progress,
                                           glossary_file=work / f"names_{target}.txt")
        dest = _write_text(Path(args.out), book_plain_text(translated, book))
        model = (str(getattr(llm, "tag", "") or "text model") + " literary, Opus-MT fallback") if llm else "Opus-MT"
        extra = {"source": source, "target": target, "chapters": len(translated.chapters), "model": model}
        human = [f"Translated {source} -> {target}: {dest}", f"  model: {model}"]
        return _ok(json_mode, "translate", started, outputs=[dest], human=human, extra=extra)

    return _run("translate", args, body)


# Settings -> Narration, Language and Projects, as ``key: (kind, help)``. ``kind`` is bool, float, choice or path.
SETTINGS_HELP = {
    "language": "UI language (en, de, ru, uk, lv)",
    "projects.folder": "Projects (working) folder for new jobs",
    "models.folder": "Models folder (read-only: set by the installer or by VOXPRINT_MODELS_DIR)",
    "narration.speed": "Reading speed 0.7-1.3 (1 = the voice's own)",
    "narration.style": "Reading style: auto, scripture, fiction, dialogue",
    "narration.pauses": "Explicit pauses between phrases (on/off)",
    "narration.ordinals": "Read ordinal numbers by context (on/off)",
    "narration.soundscape": "Soundscape under narration for a .vxbook that declares it (on/off, off by default; "
                            "turning it on downloads ACE-Step)",
    "narration.ai_disclosure": "Speak the AI disclosure at the start (on/off)",
    "theme": "Look shared by the Voxprint programs (glass-dark)",
    "gpu": "GPU shared by the Voxprint programs: auto or cuda:N",
    "gpu.vram_fraction": "Optional cap on video memory narration may plan for, 0.70-0.80 of the total (default off)",
    "gpu.fast_decode": "Fast decode with CUDA Graphs (off/graphs; experimental, see voxprint bench)",
    **{f"narration.pause.{k}": f"Pause after a {k} in seconds (0-{pz.MAX_PAUSE_MS / 1000:g})" for k in pz.DEFAULT_LENGTHS_MS},
}


def _parse_bool(value: str) -> bool:
    low = str(value).strip().lower()
    if low in ("1", "on", "true", "yes", "y"):
        return True
    if low in ("0", "off", "false", "no", "n"):
        return False
    raise CliError(EXIT_BAD_ARGS, f"expected on or off, got {value!r}")


def settings_values() -> dict:
    """Every setting of :data:`SETTINGS_HELP` with its current value."""
    from core import ai_disclosure, i18n
    from infra import gpu_prefs, suite_settings

    pace = pc.load()
    lengths = pz.load_lengths().to_dict()
    out = {
        "language": i18n.saved_language() or i18n.get_language(),
        "projects.folder": str(ws.load_folder(app_paths.default_results_dir())),
        "models.folder": str(app_paths.models_dir()),
        "narration.speed": float(pace.speed),
        "narration.style": pace.style,
        "narration.pauses": bool(pz.load_enabled()),
        "narration.ordinals": bool(ordinals.load_enabled()),
        "narration.soundscape": bool(soundscape_model.enabled()),
        "narration.ai_disclosure": bool(ai_disclosure.load_enabled()),
        "theme": suite_settings.theme(),
        "gpu": suite_settings.gpu(),
        "gpu.vram_fraction": "off" if (frac := gpu_prefs.vram_fraction()) is None else frac,
        "gpu.fast_decode": gpu_prefs.fast_decode(),
    }
    for kind, ms in lengths.items():
        out[f"narration.pause.{kind}"] = round(ms / 1000.0, 3)
    return out


def set_setting(key: str, value: str, *, persist: bool = True) -> object:
    """Validate and save one setting; returns the stored value.

    ``persist=False`` returns the value that would be stored and writes nothing.
    Turning the soundscape on does not download its model in that mode.
    """
    from core import ai_disclosure, i18n

    if key not in SETTINGS_HELP:
        raise CliError(EXIT_BAD_ARGS, f"unknown setting {key!r}", hint="voxprint settings list")
    if key == "models.folder":
        raise CliError(EXIT_BAD_ARGS, "models.folder is read-only here",
                       hint="The installer chooses it; for one session set the VOXPRINT_MODELS_DIR environment variable.")
    if key == "language":
        code = i18n.normalize_code(value)
        if code is None:
            raise CliError(EXIT_BAD_ARGS, f"unsupported language {value!r}", hint="Choose en, de, ru, uk or lv")
        if not persist:
            return code
        return i18n.set_language(code, persist=True)
    if key in ("theme", "gpu"):
        from infra import suite_settings

        if not persist:
            if key == "theme":
                if not str(value).strip():
                    raise CliError(EXIT_BAD_ARGS, f"invalid theme {value!r}")
                return str(value).strip()
            stored = suite_settings.normalize_gpu(value)
            if stored is None:
                raise CliError(EXIT_BAD_ARGS, f"gpu must be auto or cuda:N, got {value!r}")
            return stored
        try:
            return suite_settings.set_value(key, value)
        except ValueError as exc:
            raise CliError(EXIT_BAD_ARGS, str(exc)) from exc
    if key == "gpu.vram_fraction":
        from infra import gpu_prefs

        if not persist:
            if str(value).strip().lower() in ("off", "none", "false", "no"):
                return "off"
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise CliError(EXIT_BAD_ARGS, f"expected a number such as 0.75, or off, got {value!r}") from exc
            return vram_policy.clamp_fraction(value)
        try:
            fraction = gpu_prefs.set_vram_fraction(value)
        except ValueError as exc:
            raise CliError(EXIT_BAD_ARGS, f"expected a number such as 0.75, or off, got {value!r}") from exc
        return "off" if fraction is None else fraction
    if key == "gpu.fast_decode":
        from infra import gpu_prefs

        if not persist:
            try:
                return gpu_prefs.normalize_mode(value)
            except ValueError as exc:
                raise CliError(EXIT_BAD_ARGS, f"gpu.fast_decode must be off or graphs, got {value!r}") from exc
        try:
            return gpu_prefs.set_fast_decode(value)
        except ValueError as exc:
            raise CliError(EXIT_BAD_ARGS, f"gpu.fast_decode must be off or graphs, got {value!r}") from exc
    if key == "projects.folder":
        folder = Path(value).expanduser()
        if not folder.is_absolute():
            raise CliError(EXIT_BAD_ARGS, "projects.folder needs an absolute path")
        if not persist:
            return str(folder)
        folder.mkdir(parents=True, exist_ok=True)
        ws.save_folder(folder)
        return str(folder)
    if key in ("narration.speed", "narration.style"):
        pace = pc.load()
        if key == "narration.speed":
            try:
                speed = float(value)
            except ValueError as exc:
                raise CliError(EXIT_BAD_ARGS, f"expected a number, got {value!r}") from exc
            if not pc.MIN_SPEED <= speed <= pc.MAX_SPEED:
                raise CliError(EXIT_BAD_ARGS, f"narration.speed must be {pc.MIN_SPEED:g}-{pc.MAX_SPEED:g}")
            pace.speed = speed
        else:
            if value not in pc.STYLES:
                raise CliError(EXIT_BAD_ARGS, f"narration.style must be one of {', '.join(pc.STYLES)}")
            pace.style = value
        if persist:
            pc.save(pc.Pace(pace.speed, pace.style))
        return pace.speed if key == "narration.speed" else pace.style
    if key == "narration.pauses":
        on = _parse_bool(value)
        if persist:
            pz.save_enabled(on)
        return on
    if key == "narration.ordinals":
        on = _parse_bool(value)
        if persist:
            ordinals.save_enabled(on)
        return on
    if key == "narration.ai_disclosure":
        on = _parse_bool(value)
        if persist:
            ai_disclosure.save_enabled(on)
        return on
    if key == "narration.soundscape":
        on = _parse_bool(value)
        if not persist:
            return on
        if on:
            soundscape_model.enable()
        else:
            soundscape_model.disable()
        return soundscape_model.enabled()
    kind = key.rsplit(".", 1)[1]
    try:
        sec = float(value)
    except ValueError as exc:
        raise CliError(EXIT_BAD_ARGS, f"expected seconds, got {value!r}") from exc
    if not 0 <= sec <= pz.MAX_PAUSE_MS / 1000:
        raise CliError(EXIT_BAD_ARGS, f"{key} must be 0-{pz.MAX_PAUSE_MS / 1000:g} seconds")
    ms = round(sec * 1000)
    if persist:
        lengths = pz.load_lengths().to_dict()
        lengths[kind] = ms
        pz.save_lengths(pz.PauseLengths.from_dict(lengths))
    return round(ms / 1000.0, 3)


def cmd_bench(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None,
              make_engine: Optional[Callable[[str], object]] = None) -> int:
    """``bench``: the same voice and text through each mode (:mod:`core.bench`)."""
    from core import bench

    def body(json_mode: bool, started: float) -> int:
        modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
        bad = [m for m in modes if m not in bench.MODES]
        if not modes or bad:
            raise CliError(EXIT_BAD_ARGS, f"unknown mode {', '.join(bad) or '(none)'}", hint="--modes batched,graphs")
        if getattr(args, "install_graphs", False):
            from core import fast_decode

            where = fast_decode.install_wheel()
            if not json_mode:
                _write_stream(sys.stdout, f"faster-qwen3-tts {fast_decode.WHEEL['version']} installed into {where.parent}")
            import importlib

            importlib.invalidate_caches()
            if str(where.parent) not in sys.path:
                sys.path.insert(0, str(where.parent))
        factory = make_engine
        voice_name = ""
        if factory is None:
            from core import tts_engine

            lib = library if library is not None else VoiceLibrary()
            voice = resolve_voice(lib, args.voice)
            voice_name = voice.name

            def factory(mode: str):
                return tts_engine.make_engine_factory(voice, "", fast_decode="off" if mode == "batched" else "graphs")()

        def note(msg: str) -> None:
            if not json_mode:
                _write_stream(sys.stdout, msg)

        results = bench.run(modes, factory, out_dir=args.out, log=note)
        data = bench.summary(results)
        data["voice"] = voice_name
        human = [r.line() for r in results]
        if "graphs_vs_batched" in data:
            human.append(f"graphs / batched speed: {data['graphs_vs_batched']:.2f}x")
        outputs = [w for r in results for w in r.wavs]
        warnings = [f"{r.mode}: {r.reason}" for r in results if not r.ok]
        if not any(r.ok for r in results):
            raise CliError(EXIT_GPU, "no mode could run", hint="; ".join(warnings))
        return _ok(json_mode, "bench", started, outputs=outputs, warnings=warnings, human=human, extra={"bench": data})

    return _run("bench", args, body)


def cmd_settings(args: argparse.Namespace) -> int:
    """``settings list | get KEY | set KEY VALUE``: the values Settings shows, for scripts."""

    def body(json_mode: bool, started: float) -> int:
        action = args.settings_action
        if action == "list":
            values = settings_values()
            human = [f"{k} = {v}    # {SETTINGS_HELP[k]}" for k, v in values.items()]
            return _ok(json_mode, "settings", started, outputs=[], human=human, extra={"settings": values})
        key = args.key
        if key not in SETTINGS_HELP:
            raise CliError(EXIT_BAD_ARGS, f"unknown setting {key!r}", hint="voxprint settings list")
        if action == "get":
            value = settings_values()[key]
            return _ok(json_mode, "settings", started, outputs=[], human=[str(value)],
                       extra={"key": key, "value": value})
        stored = set_setting(key, args.value)
        return _ok(json_mode, "settings", started, outputs=[], human=[f"{key} = {stored}"],
                   extra={"key": key, "value": stored})

    return _run("settings", args, body)


_DRY_COMMAND = {
    "narrate": "narrate",
    "train": "train",
    "voices_list": "voices list",
    "voices_export": "voices export",
    "voices_catalog": "voices catalog",
    "voices_download": "voices download",
    "diag": "diag",
    "status": "status",
    "models_list": "models list",
    "models_download": "models download",
    "revoice": "revoice",
    "speakers": "speakers",
    "check": "check",
    "prepare": "prepare",
    "translate": "translate",
    "settings": "settings",
    "bench": "bench",
    "backup": "backup",
    "restore": "restore",
}
_DRY_MODEL_WARNING = "Dry-run does not load models, use the GPU, or download anything."
_STATUS_KEYS = (
    "version", "build", "codename", "platform", "gpu", "voices", "formats", "modules", "runtime",
)


def _dry_command_name(args: argparse.Namespace) -> str:
    handler = str(getattr(args, "_handler", "") or "")
    if handler == "status":
        return str(getattr(args, "command", None) or "status")
    return _DRY_COMMAND.get(handler, handler or "unknown")


def _dry_payload(command: str, *, ok: bool, code: int, started: float, error=None, hint=None,
                 warnings=None, extra=None) -> dict:
    payload = {
        "type": "result",
        "ok": ok,
        "command": command,
        "exit_code": int(code),
        "outputs": [],
        "warnings": list(warnings or []),
        "duration_s": round(time.perf_counter() - started, 3),
        "error": error,
        "hint": hint,
        "dry_run": True,
    }
    if extra:
        payload.update(extra)
    return payload


def _finish_dry(args: argparse.Namespace, payload: dict, *, extra_keys=(), human=()) -> int:
    """Print one result object (always for status) and return its exit code."""
    problem = schema_error(payload, extra_keys if payload.get("ok") else ())
    if problem:
        payload = _dry_payload(
            str(payload.get("command") or ""), ok=False, code=EXIT_INTERNAL, started=time.perf_counter(),
            error=problem, hint="See docs/CLI.md",
        )
    json_mode = bool(getattr(args, "json_output", False)) or payload["command"] in ("status", "capabilities")
    if json_mode:
        _write_stream(sys.stdout, json.dumps(payload, ensure_ascii=False))
    elif human:
        for line in human:
            _write_stream(sys.stdout, line)
    else:
        state = "ok" if payload["ok"] else "failed"
        _write_stream(sys.stdout, f"dry-run {state}: {payload['command']}")
    if not payload["ok"]:
        _write_stream(sys.stderr, f"ERROR: {payload['error']}")
        if payload.get("hint"):
            _write_stream(sys.stderr, f"Fix: {payload['hint']}")
    return int(payload["exit_code"])


def _dry_status_extra() -> dict:
    """The status object without probing the GPU or loading a model."""
    info = version_payload()
    requirement = {
        "ok": False,
        "min_compute": "8.9",
        "detected": "none",
        "compute_cap": None,
        "source": "dry-run",
        "override": False,
    }
    return {
        "version": info["version"],
        "build": info["build"],
        "codename": info["codename"],
        "platform": sys.platform,
        "gpu": {
            "cuda_available": False,
            "name": None,
            "vram_total_gb": None,
            "vram_free_gb": None,
            "torch": None,
            "driver": None,
            "requirement": requirement,
        },
        "voices": [_voice_payload(v) for v in VoiceLibrary().list_voices()],
        "formats": list(ex.ALL_FORMATS),
        "format_aliases": dict(FORMAT_ALIASES),
        "modules": module_catalog(),
        "runtime": _runtime_status(),
    }


def _dry_body(args: argparse.Namespace, handler: str):
    """``(extra fields, required keys, warnings, human lines)`` for one dry-run. Raises on a bad input."""
    if handler == "narrate":
        formats = sorted(parse_formats(args.formats))
        _book, fmt = load_book_input(Path(args.book))
        output = check_output(Path(args.out), kind="dir")
        extra = {
            "input": {"path": str(Path(args.book)), "format": fmt},
            "output": output,
            "voice": args.voice,
            "formats": formats,
        }
        return extra, ("input", "output", "voice", "formats"), [_DRY_MODEL_WARNING], [
            f"dry-run narrate {fmt}: {args.book} -> {output}",
        ]
    if handler == "speakers":
        book, fmt = load_book_input(Path(args.book))
        paragraphs = [text for _ci, text in spk.paragraphs(book)]
        if not paragraphs:
            raise DryRunFailure(EXIT_INPUT, "book has no paragraphs",
                                hint="The file needs at least one paragraph of text.")
        planned = check_output(Path(args.out), kind="file")
        extra = {
            "input": {"path": str(Path(args.book)), "format": fmt},
            "paragraphs": len(paragraphs),
            "planned_output": planned,
        }
        return extra, ("input", "paragraphs", "planned_output"), [_DRY_MODEL_WARNING], [
            f"dry-run speakers: {len(paragraphs)} paragraphs -> {planned}",
        ]
    if handler == "prepare":
        steps = sorted(_prep_steps(args))
        book, fmt = load_book_input(Path(args.book))
        language = resolve_language(book, getattr(args, "language", "") or "")
        planned = check_output(Path(args.out), kind="file")
        extra = {
            "input": {"path": str(Path(args.book)), "format": fmt},
            "planned_output": planned,
            "language": language,
            "steps": steps,
        }
        return extra, ("input", "planned_output", "language", "steps"), [_DRY_MODEL_WARNING], [
            f"dry-run prepare {fmt} -> {planned}",
        ]
    if handler == "translate":
        book, fmt = load_book_input(Path(args.book))
        target = args.to
        source = args.source or detect_book_language(book)
        if not source:
            raise DryRunFailure(EXIT_INPUT, "the book language could not be detected",
                                hint="Pass --from en|ru|de|uk")
        if source == target:
            raise CliError(EXIT_BAD_ARGS, f"the book is already in {target}", hint="Choose another --to language.")
        try:
            route(source, target)
        except TranslateError as exc:
            raise CliError(EXIT_BAD_ARGS, f"unsupported language pair {source} -> {target}",
                           hint="Supported: en, ru, de (and uk as a source).") from exc
        planned = check_output(Path(args.out), kind="file")
        extra = {
            "input": {"path": str(Path(args.book)), "format": fmt},
            "source": source,
            "target": target,
            "planned_output": planned,
        }
        return extra, ("input", "source", "target", "planned_output"), [_DRY_MODEL_WARNING], [
            f"dry-run translate {source} -> {target}: {planned}",
        ]
    if handler == "check":
        extra = {"items": [], "checked": 0, "fixed": 0, "failed": 0}
        return extra, ("items", "checked", "fixed", "failed"), [_DRY_MODEL_WARNING], [
            "dry-run check: no files were read or changed",
        ]
    if handler == "status":
        return _dry_status_extra(), _STATUS_KEYS, [], ["dry-run status"]
    if handler == "voices_catalog":
        return {"voices": []}, ("voices",), ["Dry-run does not fetch the voice catalog."], ["dry-run voices catalog"]
    if handler == "settings":
        action = args.settings_action
        if action == "list":
            values = settings_values()
            return {"settings": values}, ("settings",), [], [f"{k} = {v}" for k, v in values.items()]
        key = args.key
        if key not in SETTINGS_HELP:
            raise CliError(EXIT_BAD_ARGS, f"unknown setting {key!r}", hint="voxprint settings list")
        if action == "get":
            value = settings_values()[key]
            return {"key": key, "value": value}, ("key", "value"), [], [str(value)]
        stored = set_setting(key, args.value, persist=False)
        return {"key": key, "value": stored}, ("key", "value"), [], [f"{key} = {stored}"]
    if handler == "models_download":
        key = canonical_module(args.module)
        return {"module": key, "downloaded": False}, ("module", "downloaded"), [_DRY_MODEL_WARNING], [
            f"dry-run models download {key}",
        ]
    if handler == "models_list":
        return {"modules": module_catalog()}, ("modules",), [], ["dry-run models list"]
    if handler == "voices_list":
        rows = [_voice_payload(v) for v in VoiceLibrary().list_voices()]
        return {"voices": rows}, ("voices",), [], ["dry-run voices list"]
    if handler == "voices_export":
        resolve_voice(VoiceLibrary(), args.voice)
        planned = check_output(Path(args.out), kind="file")
        return {"planned_output": planned}, ("planned_output",), [_DRY_MODEL_WARNING], [
            f"dry-run voices export -> {planned}",
        ]
    if handler == "voices_download":
        return {"voice": args.voice, "downloaded": False}, ("voice", "downloaded"), [
            "Dry-run does not fetch or download voices.",
        ], [f"dry-run voices download {args.voice}"]
    if handler == "revoice":
        files = _expand_audio(list(args.audio))
        output = check_output(Path(args.out), kind="dir") if args.out else ""
        return {"inputs": [str(p) for p in files], "output": output}, (), [_DRY_MODEL_WARNING], [
            f"dry-run revoice: {len(files)} file(s)",
        ]
    if handler == "train":
        existing_input(Path(args.audio), "audio")
        if args.text is not None:
            existing_input(Path(args.text), "text")
        output = check_output(Path(args.out), kind="dir") if args.out else ""
        return {"output": output}, (), [_DRY_MODEL_WARNING], ["dry-run train"]
    if handler == "backup":
        output = check_output(Path(args.out), kind="dir")
        return {"output": output}, (), [_DRY_MODEL_WARNING], [f"dry-run backup -> {output}"]
    if handler == "restore":
        source = existing_input(Path(args.src), "backup")
        return {"source": source}, (), [_DRY_MODEL_WARNING], [f"dry-run restore from {source}"]
    if handler == "diag":
        planned = check_output(Path(args.out), kind="file") if args.out else ""
        return {"planned_output": planned}, (), [], ["dry-run diag"]
    if handler == "bench":
        output = check_output(Path(args.out), kind="dir") if args.out else ""
        return {"output": output}, (), [_DRY_MODEL_WARNING], ["dry-run bench"]
    raise CliError(EXIT_BAD_ARGS, f"unknown command: {handler}", hint="See: voxprint --help")


def run_dry(args: argparse.Namespace) -> int:
    """Validate ``args`` and print the result JSON. Does not call the command's runner."""
    started = time.perf_counter()
    command = _dry_command_name(args)
    handler = str(getattr(args, "_handler", "") or "")
    try:
        extra, keys, warnings, human = _dry_body(args, handler)
    except DryRunFailure as exc:
        payload = _dry_payload(command, ok=False, code=exc.code, started=started,
                               error=exc.message, hint=exc.hint or None)
        return _finish_dry(args, payload)
    except CliError as exc:
        payload = _dry_payload(command, ok=False, code=exc.code, started=started,
                               error=exc.message, hint=exc.hint or None)
        return _finish_dry(args, payload)
    payload = _dry_payload(command, ok=True, code=EXIT_OK, started=started, warnings=warnings, extra=extra)
    return _finish_dry(args, payload, extra_keys=keys, human=human)


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
    guard_stdio()
    previous_hook = sys.excepthook
    install_cli_excepthook()
    try:
        return _dispatch(argv, run_narration_fn=run_narration_fn, run_task_fn=run_task_fn, library=library,
                         transcribe_fn=transcribe_fn, download_fn=download_fn, installed_fn=installed_fn,
                         plan_fn=plan_fn, repair_fn=repair_fn)
    finally:
        # A frozen windowed exe keeps the hook so a late exception does not open a traceback window.
        # Tests and a developer ``python`` put the previous hook back.
        if not getattr(sys, "frozen", False):
            sys.excepthook = previous_hook


def _dispatch(argv: Optional[Sequence[str]], *, run_narration_fn, run_task_fn, library, transcribe_fn, download_fn,
              installed_fn, plan_fn, repair_fn) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    if _flag_present(raw, "--version", "-V") and not _flag_present(raw, "--help", "-h"):
        return cmd_version(_flag_present(raw, "--json"))
    if startup_requires_gpu(["voxprint", *raw]):
        blocked = enforce(gui=False)
        if blocked:
            return blocked
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
    if getattr(args, "dry_run", False):
        return run_dry(args)
    if handler == "narrate":
        return cmd_narrate(args, run_fn=run_narration_fn, library=library, plan_fn=plan_fn)
    if handler == "train":
        return cmd_train(args, run_fn=run_task_fn)
    if handler == "voices_list":
        return cmd_voices_list(args, library=library)
    if handler == "voices_export":
        return cmd_voices_export(args, library=library)
    if handler == "voices_catalog":
        return cmd_voices_catalog(args, library=library)
    if handler == "voices_download":
        return cmd_voices_download(args, library=library)
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
    if handler == "prepare":
        return cmd_prepare(args, plan_fn=plan_fn)
    if handler == "translate":
        return cmd_translate(args, plan_fn=plan_fn)
    if handler == "settings":
        return cmd_settings(args)
    if handler == "bench":
        return cmd_bench(args, library=library)
    if handler == "backup":
        return cmd_backup(args)
    if handler == "restore":
        return cmd_restore(args)
    ap.error(f"unknown command: {handler}")
    return EXIT_BAD_ARGS


def is_user_cli(argv: Sequence[str]) -> bool:
    """True when ``argv`` (full ``sys.argv``-style, with program name) starts a user subcommand.

    Global flags (``--json``, ``--yes``, ``--dry-run``, ``--version``, ``--help``) may come first.
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
