"""The automatic "prepare book for narration" pipeline: rules -> neural clean-up -> (synthesis, elsewhere).

:func:`run_preparation` takes a parsed :class:`~core.book_parsers.Book` and a :class:`PrepPlan` and returns the prepared
book.  The user never edits or reviews the text, so the result is also written next to the job cache for debugging:
``<job>/.debug/prepared_text.txt`` (all chapters, titles included) and ``prep_report.json`` (language, steps, counts,
validator statistics, model tags).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from core import text_cleanup
from core.book_parsers import Book
from core.events import CancelToken
from core.text_prep import PrepOptions, PrepReport, prepare_book, resolve_language

log = logging.getLogger("voxprint.prep")

#: Name of the neural spelling / comma step (the UI checkbox key, see :mod:`infra.text_models`).
NEURAL_SPELLFIX = "spellfix"


@dataclass
class PrepPlan:
    """What to do with the text before synthesis."""
    rules: PrepOptions = field(default_factory=PrepOptions.none)
    neural: frozenset = frozenset()                                # neural step keys, e.g. {"spellfix"}
    #: Creates the clean-up engine for a language code, or returns ``None`` if no model is available for it.
    engine_factory: Optional[Callable[[str], Optional[text_cleanup.CleanupEngine]]] = None

    @property
    def enabled(self) -> bool:
        """True if anything would run."""
        return bool(self.rules.steps or self.neural)

    @property
    def spells_out_numbers(self) -> bool:
        """True if digits are already spelled out (the engine-side normalizer is then unnecessary)."""
        return "numbers" in self.rules.steps


def run_preparation(book: Book, plan: PrepPlan, language_hint: str = "", debug_dir: Optional[Path] = None,
                    cache_dir: Optional[Path] = None, progress: Optional[Callable[[float, str], None]] = None,
                    cancel: Optional[CancelToken] = None) -> Tuple[Book, Dict[str, Any]]:
    """Prepare ``book``; returns ``(prepared_book, report_dict)`` and writes the debug files when ``debug_dir`` is given."""
    cancel = cancel or CancelToken()
    prepared, rep = prepare_book(book, plan.rules, language_hint) if plan.rules.steps else (
        book, PrepReport(resolve_language(book, language_hint)))
    report: Dict[str, Any] = {"language": rep.language, "rules": rep.steps, "rule_counts": rep.counts,
                              "rules_skipped": rep.skipped, "neural": []}
    cancel.check()
    if NEURAL_SPELLFIX in plan.neural and plan.engine_factory is not None:
        engine = plan.engine_factory(rep.language)
        if engine is None:
            report["neural_skipped"] = {NEURAL_SPELLFIX: "no model for language " + (rep.language or "?")}
        else:
            try:
                prepared, stats = text_cleanup.cleanup_book(
                    prepared, engine, (cache_dir / "cleanup.json") if cache_dir else None, progress, cancel)
                report["neural"].append({"step": NEURAL_SPELLFIX, "model": engine.tag, **stats.as_dict()})
            finally:
                engine.close()
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for i, ch in enumerate(prepared.chapters):
            parts.append(f"=== [{i + 1}] {ch.title}\n\n{ch.text}\n")
        (debug_dir / "prepared_text.txt").write_text("\n".join(parts), encoding="utf-8")
        (debug_dir / "prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                                    encoding="utf-8")
    return prepared, report
