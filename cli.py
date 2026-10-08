"""User-facing command line for headless servers and agents.

Wired from ``main.py`` so both ``python main.py narrate ...`` and a packaged
``Voxprint.exe narrate ...`` work.  Subcommands call the same Qt-free runners the
GUI uses (:func:`workers.narration_runner.run_narration`,
:func:`workers.pipeline_runner.run_task`, :class:`core.voice_library.VoiceLibrary`).

Examples::

    python main.py narrate book.epub --voice [model_voice] --out ./audiobooks
    python main.py train recording.wav --text script.txt --name Anna --type female
    python main.py voices list
    python main.py voices export [model_voice] --out anna.zip

The developer dataset CLI remains ``python -m core.cli`` (see ``core/cli.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Set

from core import audiobook_export as ex
from core import pauses as pz
from core.book_parsers import load_book
from core.errors import BackupError, CancelledByUser, DatasetMakerError
from core.events import CancelToken, Stage
from core.narration import NarrationOptions, NarrationProgress, PauseToken
from core.voice_info import VOICE_TYPES, normalize_voice_type
from core.voice_library import VoiceLibrary, VoiceRecord
from infra import backup as backup_mod
from workers.backup_runner import collect as collect_backup
from workers.narration_runner import NarrationJob, run_narration
from workers.pipeline_runner import KIND_LORA, TaskRequest, run_task

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

USER_COMMANDS = frozenset({"narrate", "train", "voices", "diag", "backup", "restore"})


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree for ``narrate`` / ``train`` / ``voices``."""
    ap = argparse.ArgumentParser(
        prog="voxprint",
        description="Voxprint AI Audiobook Builder - headless CLI (narrate, train, voices).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    n = sub.add_parser("narrate", help="Narrate a book with a trained voice")
    n.add_argument("book", help="Path to a TXT, FB2, FB2.ZIP or EPUB file")
    n.add_argument("--voice", required=True, metavar="ID_OR_NAME",
                   help="Voice library id or display name (example: [model_voice])")
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
    n.add_argument("--ai-disclosure", action="store_true",
                   help="Speak a short AI disclosure at the start (opt-in)")
    n.add_argument("--work-dir", type=Path, default=None, metavar="DIR",
                   help="Remember this folder as the app working folder (also used when --out is omitted in tools)")
    n.set_defaults(_handler="narrate")

    t = sub.add_parser("train", help="Train a voice (LoRA) from a recording")
    t.add_argument("audio", type=Path, help="Path to the recording (WAV/MP3/...); folder of clips with --no text")
    t.add_argument("--text", type=Path, default=None, metavar="SCRIPT",
                   help="Transcript of the recording; omit to use no-transcript (ASR) mode")
    t.add_argument("--name", default="", help="Display name for the new voice")
    t.add_argument("--type", dest="voice_type", default="", choices=("",) + VOICE_TYPES,
                   help="Voice type stored in voice.json: male | female | child | other")
    t.add_argument("--out", type=Path, default=None, metavar="DIR",
                   help="Result root folder (default: next to the recording)")
    t.add_argument("--force-cpu", action="store_true", help="Train on CPU even when a GPU is present")
    t.set_defaults(_handler="train")

    v = sub.add_parser("voices", help="List or export voices from the library")
    vsub = v.add_subparsers(dest="voices_command", required=True)
    vsub.add_parser("list", help="List installed voices").set_defaults(_handler="voices_list")
    ve = vsub.add_parser("export", help="Export a voice as a small zip (adapter + voice.json)")
    ve.add_argument("voice", metavar="VOICE", help="Voice id or display name (example: [model_voice])")
    ve.add_argument("--out", required=True, type=Path, metavar="ZIP", help="Destination .zip path")
    ve.set_defaults(_handler="voices_export")

    d = sub.add_parser("diag", help="Save a diagnostic report (logs + system information) as a zip")
    d.add_argument("--out", type=Path, default=None, metavar="ZIP",
                   help="Destination .zip (default: voxprint-diagnostics-<date>.zip in the current folder)")
    d.set_defaults(_handler="diag")

    b = sub.add_parser("backup", help="Copy models and voices to a folder (resumable, with a manifest)")
    b.add_argument("--out", required=True, type=Path, metavar="DIR", help="Folder the backup is written into")
    b.add_argument("--no-models", action="store_true", help="Leave models out")
    b.add_argument("--no-voices", action="store_true", help="Leave the voice library out")
    b.add_argument("--json", action="store_true", help="Print the result as one JSON object")
    b.set_defaults(_handler="backup")

    r = sub.add_parser("restore", help="Restore a backup into the normal folders, or use its models in place")
    r.add_argument("--from", dest="src", required=True, type=Path, metavar="DIR",
                   help="Folder that contains the backup (or the Voxprint-backup folder itself)")
    r.add_argument("--link", action="store_true",
                   help="Use models from this folder (don't copy). The drive must stay connected.")
    r.add_argument("--json", action="store_true", help="Print the result as one JSON object")
    r.set_defaults(_handler="restore")

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
                raise SystemExit(f"ERROR: unknown format {part!r}; choose from: {', '.join(ex.ALL_FORMATS)}")
            out.add(mapped)
    return out or set(ex.DEFAULT_FORMATS)


def resolve_voice(library: VoiceLibrary, id_or_name: str) -> VoiceRecord:
    """Resolve a voice by folder id or display name (case-insensitive); raises SystemExit on failure."""
    needle = (id_or_name or "").strip()
    if not needle:
        raise SystemExit("ERROR: voice id or name is required")
    rec = library.get(needle)
    if rec is not None:
        return rec
    lower = needle.lower()
    matches = [v for v in library.list_voices()
               if v.id.lower() == lower or v.name.lower() == lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(m.id for m in matches)
        raise SystemExit(f"ERROR: voice {needle!r} matches several entries: {ids}")
    raise SystemExit(f"ERROR: voice not found: {needle}")


def _print_progress_stage(stage: Stage, frac: float, msg: str) -> None:
    """Training progress line for the console."""
    print(f"[{stage.label}] {frac * 100:5.1f}%  {msg}", flush=True)


def _print_narration_progress(p: NarrationProgress) -> None:
    """Narration progress line for the console."""
    eta = ""
    if p.eta is not None and p.eta >= 0:
        from workers.narration_runner import format_eta
        eta = f"  ETA {format_eta(p.eta)}"
    print(f"[{p.phase}] {p.done}/{p.total}{eta}  {p.message}", flush=True)


def cmd_narrate(args: argparse.Namespace, *,
                run_fn: Callable[..., object] = run_narration,
                library: Optional[VoiceLibrary] = None) -> int:
    """``narrate BOOK --voice ... --out DIR``: load the book, resolve the voice, call ``run_narration``."""
    from core import workspace as ws
    from infra import keep_awake

    lib = library if library is not None else VoiceLibrary()
    voice = resolve_voice(lib, args.voice)
    book_path = Path(args.book)
    if not book_path.is_file():
        print(f"ERROR: book not found: {book_path}", file=sys.stderr)
        return 2
    try:
        book = load_book(book_path)
    except DatasetMakerError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir is not None:
        ws.save_folder(Path(args.work_dir))

    formats = parse_formats(args.formats)
    options = NarrationOptions(
        formats=formats,
        pauses=pz.PauseProfile() if args.pauses else None,
        ai_disclosure=bool(args.ai_disclosure),
    )
    job = NarrationJob(book=book, voice=voice, out_dir=out_dir, options=options)
    try:
        with keep_awake.keep_awake():
            result = run_fn(job, _print_narration_progress, CancelToken(), PauseToken())
    except DatasetMakerError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return 2
    files = getattr(result, "files", None) or []
    print(f"Done: {getattr(result, 'out_dir', out_dir)}")
    for f in files:
        print(f"  {f}")
    return 0


def cmd_train(args: argparse.Namespace, *,
              run_fn: Callable[..., object] = run_task) -> int:
    """``train AUDIO [--text SCRIPT]``: build a LoRA voice via ``run_task``."""
    from infra import keep_awake

    audio = Path(args.audio)
    if not audio.exists():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 2
    text = Path(args.text) if args.text else None
    if text is not None and not text.is_file():
        print(f"ERROR: text not found: {text}", file=sys.stderr)
        return 2
    vtype = normalize_voice_type(args.voice_type)
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
            consent_mode="none",
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
            consent_mode="none",
        )
    try:
        with keep_awake.keep_awake():
            result = run_fn(req, _print_progress_stage, CancelToken())
    except DatasetMakerError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return 2
    print(f"Done: voice_id={getattr(result, 'voice_id', '') or '(none)'}  "
          f"adapter={getattr(result, 'adapter_path', None)}  root={getattr(result, 'root_dir', '')}")
    for w in getattr(result, "warnings", None) or []:
        print(f"  ! {w}")
    return 0


def cmd_voices_list(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None) -> int:
    """``voices list``: print id, name, type, language, licence, gender, age group (tab-separated; "-" = not set)."""
    lib = library if library is not None else VoiceLibrary()
    voices = lib.list_voices()
    if not voices:
        print("No voices installed.")
        return 0
    for v in voices:
        vtype = v.info.get("voice_type") or "-"
        gender, age = v.info.get("gender") or "-", v.info.get("age_group") or "-"     # appended: older columns keep their places
        print(f"{v.id}\t{v.name}\t{vtype}\t{v.language or '-'}\t{v.license}\t{gender}\t{age}")
    return 0


def cmd_voices_export(args: argparse.Namespace, *, library: Optional[VoiceLibrary] = None) -> int:
    """``voices export VOICE --out ZIP``: pack the voice for transfer."""
    lib = library if library is not None else VoiceLibrary()
    voice = resolve_voice(lib, args.voice)
    dest = Path(args.out)
    if dest.suffix.lower() != ".zip":
        dest = dest.with_suffix(dest.suffix + ".zip") if dest.suffix else dest.with_suffix(".zip")
    try:
        path = lib.export_zip(voice.id, dest)
    except DatasetMakerError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        if exc.details:
            print(f"  details: {exc.details}", file=sys.stderr)
        return 2
    print(f"Exported: {path}")
    return 0


def _report_payload(report, ok: bool = True) -> dict:
    """JSON-friendly summary of a backup or restore report."""
    return {
        "ok": ok,
        "target": str(report.target),
        "copied_files": report.copied_files,
        "skipped_files": report.skipped_files,
        "copied_bytes": report.copied_bytes,
        "problems": list(getattr(report, "problems", ()) or ()),
        "conflicts": list(report.conflicts),
        "external_models": getattr(report, "external", "") or "",
    }


def _emit_report(args, report, ok: bool = True) -> int:
    """Print a report as JSON or as a short text line. Returns 0, or 1 when ``ok`` is false."""
    payload = _report_payload(report, ok)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    elif payload["external_models"]:
        print(f"Models stay in {payload['external_models']} (not copied).")
        if payload["problems"]:
            print("Damaged or missing: " + ", ".join(payload["problems"]))
        print("The drive must stay connected. If it is missing, models are downloaded into the usual folder.")
    else:
        print(f"Copied {payload['copied_files']} files, skipped {payload['skipped_files']}.")
        if payload["problems"]:
            print("Damaged or missing (downloaded as usual): " + ", ".join(payload["problems"]))
        if payload["conflicts"]:
            print("Not overwritten: " + ", ".join(payload["conflicts"]))
    return 0 if ok else 1


def cmd_backup(args) -> int:
    """``backup --out DIR [--no-models|--no-voices] [--json]``."""
    try:
        items = collect_backup(include_voices=not args.no_voices, include_models=not args.no_models)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not items:
        print("ERROR: nothing to back up", file=sys.stderr)
        return 1
    try:
        report = backup_mod.run_backup(items, args.out, cancel=CancelToken())
    except CancelledByUser:
        print("ERROR: cancelled", file=sys.stderr)
        return 1
    except BackupError as exc:
        print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1
    return _emit_report(args, report)


def cmd_restore(args) -> int:
    """``restore --from DIR [--link] [--json]``. Damaged files are reported; the app downloads those later."""
    try:
        if args.link:
            report = backup_mod.run_link(args.src, cancel=CancelToken())
        else:
            report = backup_mod.run_restore(args.src, strict=False, cancel=CancelToken())
    except BackupError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": exc.code, "message": exc.user_message}, ensure_ascii=False))
        else:
            print(f"ERROR: {exc.user_message}", file=sys.stderr)
        return 1
    return _emit_report(args, report, ok=True)


def cmd_diag(args) -> int:
    """``diag``: zip the logs, system information and a settings snapshot (infra/diagnostics.py)."""
    import time

    from infra import diagnostics

    out = args.out or Path.cwd() / time.strftime("voxprint-diagnostics-%Y%m%d-%H%M%S.zip")
    info = diagnostics.system_info(import_torch=True, full=True)
    for line in diagnostics.summary_lines(info):
        print(line)
    try:
        path = diagnostics.write_report(out, info=lambda: info)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Diagnostic report: {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None, *,
         run_narration_fn: Callable[..., object] = run_narration,
         run_task_fn: Callable[..., object] = run_task,
         library: Optional[VoiceLibrary] = None) -> int:
    """Parse ``argv`` and run the matching subcommand.  Returns the process exit code."""
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "_handler", "")
    if handler == "narrate":
        return cmd_narrate(args, run_fn=run_narration_fn, library=library)
    if handler == "train":
        return cmd_train(args, run_fn=run_task_fn)
    if handler == "voices_list":
        return cmd_voices_list(args, library=library)
    if handler == "voices_export":
        return cmd_voices_export(args, library=library)
    if handler == "diag":
        return cmd_diag(args)
    if handler == "backup":
        return cmd_backup(args)
    if handler == "restore":
        return cmd_restore(args)
    ap.error(f"unknown command: {handler}")
    return 2


def is_user_cli(argv: Sequence[str]) -> bool:
    """True when ``argv`` (full ``sys.argv``-style, with program name) starts a user subcommand."""
    rest = list(argv[1:] if argv else [])
    return bool(rest) and not rest[0].startswith("-") and rest[0] in USER_COMMANDS


if __name__ == "__main__":
    raise SystemExit(main())
