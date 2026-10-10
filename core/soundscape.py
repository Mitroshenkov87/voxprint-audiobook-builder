"""Soundscape for a ``.vxbook`` that declares extension ``sound/1``.

The module plays nothing unless the manifest lists ``sound/1`` and ``sound.json`` parsed. A plain book, and a
``.vxbook`` without that pair, is left untouched. ``sfx`` cues use the same one-shot path as accents. A cue that
cannot be rendered is skipped and a warning is logged.

Beds are one 30-60 s clip (45 s here), looped with a crossfade for the whole scene. Levels are relative to the
narration's own loudness: bed -22 dB, accent -18 dB, sfx and transition -16 dB, and an extra 8 dB of ducking while
speech is present. The mixed chapter is then scaled so the speech keeps that same loudness, and peak-limited.

``at_text`` starts at the synthesis chunk that contains the snippet. ``sound-cast.json`` overrides a cue by id
(``prompt``, ``gain_db``, ``disabled``). An ``asset`` field is ignored: sound/1 is generated from the prompt.
``sound.json`` wins when an inline ``vx:sound`` comment disagrees.

Example::

    plan = plan_for(book)
    if plan is not None:
        mix_chapter_file(chapter, spans, plan, 0, SoundRequest(generate=my_tones), cache)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from core.audiobook_export import ChapterAudio
from core.book_parsers import Book
from infra import soundscape_model

log = logging.getLogger("voxprint.soundscape")

EXTENSION = "sound/1"
BED_SECONDS = 45.0
SHOT_SECONDS = {"accent": 3.0, "sfx": 4.0, "transition": 3.0}
DEFAULT_GAINS = {"bed": -22.0, "accent": -18.0, "sfx": -16.0, "transition": -16.0}
DEFAULT_DUCK_DB = 8.0
DEFAULT_FADE_IN_MS = 2000
DEFAULT_FADE_OUT_MS = 3000
DEFAULT_CROSSFADE_MS = 2000
MOODS = frozenset({
    "neutral", "calm", "warm", "joyful", "playful", "romantic", "melancholic",
    "mysterious", "tense", "ominous", "horror", "action", "triumphant",
})
_CUE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
Generator = Callable[[str, float, int, str], np.ndarray]


@dataclass
class SoundRequest:
    """How to render cue audio. ``generate`` is injected by tests. The default calls the optional model."""
    generate: Optional[Generator] = None


@dataclass
class Cue:
    """One cue after overrides and re-anchoring."""
    id: str
    kind: str
    prompt: str
    start_paragraph: int
    end_paragraph: int
    chapter_index: int
    gain_db: float
    duck_db: float
    position: str = "before"
    at_text: str = ""
    layer: str = ""
    loop: bool = True
    fade_in_ms: int = DEFAULT_FADE_IN_MS
    fade_out_ms: int = DEFAULT_FADE_OUT_MS
    crossfade_ms: int = DEFAULT_CROSSFADE_MS
    conf: float = 0.0


@dataclass
class SoundPlan:
    """Cues to mix, plus warnings already logged."""
    cues: List[Cue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    master_gain_db: float = 0.0


def requested(book: Book) -> bool:
    """True when this book both declares ``sound/1`` and carried a parsed ``sound.json``."""
    return EXTENSION in getattr(book, "extensions", ()) and isinstance(getattr(book, "sound_document", None), dict)


def _warn(warnings: List[str], message: str) -> None:
    warnings.append(message)
    log.warning("soundscape: %s", message)


def _num(value: object, lo: float, hi: float, default: float) -> Optional[float]:
    """A number inside ``lo..hi``, or None when the value is outside that range."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < lo or number > hi:
        return None
    return number


def _paragraphs(book: Book) -> List[Tuple[int, str, int, str, str]]:
    """``(index, chapter id, chapter index, fingerprint, plain text)`` from the unread ``book.md``."""
    return list(getattr(book, "sound_paragraphs", ()) or ())


def _reanchor(paras: Sequence[Tuple[int, str, int, str, str]], chapter: str, index: object, fingerprint: str,
              cue_id: str, warnings: List[str]) -> Optional[int]:
    """Paragraph index. On a mismatch, search the same chapter, then the rest of the book."""
    if not isinstance(index, int) or isinstance(index, bool):
        _warn(warnings, f"cue {cue_id} dropped: start paragraph is missing")
        return None
    by_index = {row[0]: row for row in paras}
    row = by_index.get(index)
    chapter_ok = (not chapter) or (row is not None and row[1] == chapter)
    fp_ok = (not fingerprint) or (row is not None and row[3] == fingerprint)
    if row is not None and chapter_ok and fp_ok:
        return index
    found: List[Tuple[int, str, int, str, str]] = []
    if fingerprint:
        in_chapter = [item for item in paras if item[3] == fingerprint and chapter and item[1] == chapter]
        if len(in_chapter) == 1:
            found = in_chapter
        else:
            anywhere = [item for item in paras if item[3] == fingerprint]
            if len(anywhere) == 1:
                found = anywhere
    if found:
        _warn(warnings, f"cue {cue_id} re-anchored from paragraph {index} to {found[0][0]}")
        return found[0][0]
    _warn(warnings, f"cue {cue_id} dropped: paragraph fingerprint not found")
    return None


def _apply_cast(cue: Cue, override: Dict[str, Any], warnings: List[str]) -> Optional[Cue]:
    """One ``sound-cast.json`` object. Unknown keys are ignored. A disabled cue is dropped."""
    if not isinstance(override, dict):
        return cue
    if override.get("disabled") is True:
        return None
    gain = override.get("gain_db", None)
    if gain is not None:
        number = _num(gain, -60.0, 6.0, cue.gain_db)
        if number is None:
            _warn(warnings, f"cue {cue.id} override gain was ignored")
        else:
            cue.gain_db = number
    prompt = override.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        cue.prompt = prompt.strip()[:300]
    # sound/1 is all-generated. ``asset`` is reserved for a later minor version and is ignored.
    return cue


def _enforce_limits(cues: List[Cue], warnings: List[str]) -> List[Cue]:
    """Drop lower-confidence cues that break the layering limits."""
    cues = sorted(cues, key=lambda c: (-c.conf, c.id))
    kept: List[Cue] = []
    per_chapter: Dict[int, int] = {}
    for cue in cues:
        per_chapter[cue.chapter_index] = per_chapter.get(cue.chapter_index, 0) + 1
        if per_chapter[cue.chapter_index] > 200 or len(kept) >= 5000:
            _warn(warnings, f"cue {cue.id} dropped: too many cues")
            continue
        if cue.kind == "bed":
            clash = False
            for other in kept:
                if other.kind != "bed" or other.layer != cue.layer:
                    continue
                overlap = min(cue.end_paragraph, other.end_paragraph) - max(cue.start_paragraph, other.start_paragraph)
                if overlap > 0:
                    clash = True
                    break
            if clash:
                _warn(warnings, f"cue {cue.id} dropped: another {cue.layer} bed already covers those paragraphs")
                continue
        else:
            same = [other for other in kept if other.kind != "bed" and other.start_paragraph == cue.start_paragraph]
            if len(same) >= 2 or any(other.position == cue.position for other in same):
                _warn(warnings, f"cue {cue.id} dropped: too many one-shots on paragraph {cue.start_paragraph}")
                continue
        kept.append(cue)
    return sorted(kept, key=lambda c: (c.start_paragraph, c.id))


def plan_for(book: Book) -> Optional[SoundPlan]:
    """The mix for ``book``, or None when the soundscape must not run.

    A fingerprint that no longer matches is searched in the same chapter and the cue is moved. A cue that cannot
    be found is dropped. ``sound-cast.json`` with ``enabled: false`` turns the whole plan off.
    """
    if not requested(book):
        return None
    doc = book.sound_document or {}
    warnings: List[str] = []
    if doc.get("extension") != EXTENSION:
        _warn(warnings, "sound.json extension is not sound/1")
        return None
    version = str(doc.get("version") or "")
    major = version.split(".", 1)[0]
    if major != "1":
        _warn(warnings, f"sound extension version {version or '(missing)'} is not supported")
        return None
    cast = book.sound_cast_document if isinstance(book.sound_cast_document, dict) else {}
    if cast.get("enabled") is False:
        _warn(warnings, "sound-cast.json turns the soundscape off for this book")
        return None
    master = _num(cast.get("master_gain_db", 0), -60.0, 12.0, 0.0)
    if master is None:
        _warn(warnings, "master gain was ignored")
        master = 0.0
    defaults = doc.get("defaults") if isinstance(doc.get("defaults"), dict) else {}
    gains = dict(DEFAULT_GAINS)
    for key, kind in (("bed_gain_db", "bed"), ("accent_gain_db", "accent"),
                      ("sfx_gain_db", "sfx"), ("transition_gain_db", "transition")):
        number = _num(defaults.get(key), -60.0, 0.0, gains[kind])
        if number is None:
            _warn(warnings, f"default {key} is out of range and was ignored")
        else:
            gains[kind] = number
    duck_default = _num(defaults.get("duck_db"), 0.0, 30.0, DEFAULT_DUCK_DB)
    if duck_default is None:
        _warn(warnings, "default duck_db is out of range and was ignored")
        duck_default = DEFAULT_DUCK_DB
    paras = _paragraphs(book)
    by_index = {row[0]: row for row in paras}
    raw_cues = doc.get("cues") if isinstance(doc.get("cues"), list) else []
    overrides = cast.get("cues") if isinstance(cast.get("cues"), dict) else {}
    for key in overrides:
        if not any(isinstance(row, dict) and row.get("id") == key for row in raw_cues):
            _warn(warnings, f"sound-cast override {key} does not match a cue and was dropped")
    scenes = doc.get("scenes") if isinstance(doc.get("scenes"), list) else []
    for scene in scenes:
        if isinstance(scene, dict) and scene.get("mood") not in MOODS and scene.get("mood") is not None:
            _warn(warnings, f"scene {scene.get('id')} has an unknown mood and is treated as neutral")
    inline = list(getattr(book, "sound_inline", ()) or ())
    built: List[Cue] = []
    seen = set()
    for row in raw_cues:
        if not isinstance(row, dict):
            continue
        cid = row.get("id")
        kind = row.get("kind")
        if not isinstance(cid, str) or not _CUE_ID.match(cid) or cid in seen:
            _warn(warnings, f"cue {cid!r} dropped: bad or repeated id")
            continue
        seen.add(cid)
        if kind not in ("bed", "accent", "transition", "sfx"):
            _warn(warnings, f"cue {cid} dropped: unknown kind")
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 300:
            _warn(warnings, f"cue {cid} dropped: prompt is missing or too long")
            continue
        chapter = str(row.get("chapter") or "")
        start = _reanchor(paras, chapter, row.get("start_paragraph"), str(row.get("para_fp") or ""), cid, warnings)
        if start is None:
            continue
        end = start
        layer = ""
        if kind == "bed":
            layer = str(row.get("layer") or "")
            if layer not in ("music", "ambience"):
                _warn(warnings, f"cue {cid} dropped: a bed needs a music or ambience layer")
                continue
            end = _reanchor(paras, chapter, row.get("end_paragraph"), str(row.get("end_para_fp") or ""), cid, warnings)
            if end is None or end < start or by_index[end][2] != by_index[start][2]:
                _warn(warnings, f"cue {cid} dropped: the bed does not stay inside one chapter")
                continue
        position = str(row.get("position") or ("before" if kind == "transition" else "before"))
        at_text = str(row.get("at_text") or "")
        if kind != "bed":
            if position not in ("before", "after", "at_text"):
                _warn(warnings, f"cue {cid} dropped: position is not before, after or at_text")
                continue
            if position == "at_text":
                plain = by_index[start][4]
                if not (3 <= len(at_text) <= 60) or at_text not in plain:
                    _warn(warnings, f"cue {cid} at_text was not found; it will play at the paragraph start")
                    position = "before"
                    at_text = ""
        gain = _num(row.get("gain_db"), -60.0, 0.0, gains[kind])
        if gain is None:
            _warn(warnings, f"cue {cid} dropped: gain is out of range")
            continue
        duck = _num(row.get("duck_db"), 0.0, 30.0, duck_default) if kind == "bed" else 0.0
        if duck is None:
            _warn(warnings, f"cue {cid} duck was ignored")
            duck = duck_default
        conf = _num(row.get("conf"), 0.0, 1.0, 0.0)
        if conf is None:
            conf = 0.0
        cue = Cue(
            id=cid, kind=kind, prompt=prompt.strip(), start_paragraph=start, end_paragraph=end,
            chapter_index=by_index[start][2], gain_db=gain, duck_db=float(duck), position=position,
            at_text=at_text, layer=layer, loop=bool(row.get("loop", True)),
            fade_in_ms=int(_num(row.get("fade_in_ms"), 0, 60000, DEFAULT_FADE_IN_MS) or DEFAULT_FADE_IN_MS),
            fade_out_ms=int(_num(row.get("fade_out_ms"), 0, 60000, DEFAULT_FADE_OUT_MS) or DEFAULT_FADE_OUT_MS),
            crossfade_ms=int(_num(row.get("crossfade_ms"), 0, 60000, DEFAULT_CROSSFADE_MS) or DEFAULT_CROSSFADE_MS),
            conf=float(conf),
        )
        for para, inline_id in inline:
            if inline_id == cid and para != start:
                _warn(warnings, f"inline vx:sound for {cid} is on paragraph {para}; sound.json paragraph {start} wins")
        override = overrides.get(cid)
        if isinstance(override, dict):
            updated = _apply_cast(cue, override, warnings)
            if updated is None:
                continue
            cue = updated
        built.append(cue)
    return SoundPlan(_enforce_limits(built, warnings), warnings, float(master))


def _renderer(request: SoundRequest) -> Generator:
    """The injected generator, or the optional ACE-Step model."""
    if request.generate is not None:
        return request.generate
    return soundscape_model.generate


def _rms(samples: np.ndarray) -> float:
    """Root-mean-square of a mono buffer. Empty audio is silent."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _speech_mask(narr: np.ndarray, sr: int) -> np.ndarray:
    """1 where the narration is loud enough to count as speech, else 0."""
    mask = np.zeros(len(narr), dtype=np.float32)
    if narr.size == 0:
        return mask
    frame = max(1, int(sr * 0.02))
    hop = max(1, frame // 2)
    thresh = max(1e-4, _rms(narr) * 0.2)
    for i in range(0, len(narr), hop):
        j = min(len(narr), i + frame)
        if _rms(narr[i:j]) >= thresh:
            mask[i:j] = 1.0
    return mask


def _fade_edges(clip: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    """Linear fades. The lengths are clamped so they cannot eat the whole clip."""
    out = clip.astype(np.float32, copy=True)
    n = len(out)
    fi = min(max(fade_in, 0), n // 2)
    fo = min(max(fade_out, 0), n // 2)
    if fi:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def loop_crossfade(clip: np.ndarray, n_out: int, crossfade: int) -> np.ndarray:
    """Repeat ``clip`` to ``n_out`` samples. Each join overlaps by ``crossfade`` samples."""
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if n_out <= 0 or clip.size == 0:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    if clip.size >= n_out:
        return clip[:n_out].copy()
    cf = min(max(crossfade, 0), clip.size // 3)
    if cf <= 0:
        reps = int(np.ceil(n_out / clip.size))
        return np.tile(clip, reps)[:n_out]
    step = clip.size - cf
    out = np.zeros(n_out, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
    pos = 0
    first = True
    guard = 0
    while pos < n_out and guard < n_out:
        piece = clip.copy()
        if not first:
            piece[:cf] *= fade_in
        piece[-cf:] *= fade_out
        n = min(piece.size, n_out - pos)
        out[pos:pos + n] += piece[:n]
        if first:
            first = False
        pos += step
        guard += 1
    return out


def _normalise_clip(clip: np.ndarray) -> np.ndarray:
    """Scale a generated clip to RMS 1 so later gains are relative to the narration, not to the generator."""
    audio = np.asarray(clip, dtype=np.float32).reshape(-1)
    level = _rms(audio)
    if level < 1e-8:
        return audio
    return audio / np.float32(level)


def _place_sample(cue: Cue, spans: Sequence[Tuple[int, str, int, int, int]]) -> Optional[int]:
    """Sample index where a one-shot starts. ``spans`` are ``(paragraph, text, start, speech_end, pause_end)``.

    A transition with ``before`` starts at the beginning of the pause in front of the paragraph (the scene or
    chapter gap). ``at_text`` starts at the synthesis chunk that contains the snippet. A missing snippet uses
    the paragraph start.
    """
    rows = [row for row in spans if row[0] == cue.start_paragraph]
    if not rows:
        return None
    if cue.position == "after":
        return rows[-1][3]
    if cue.kind == "transition" and cue.position == "before":
        start = rows[0][2]
        earlier = [row[3] for row in spans if row[3] <= start]
        return max(earlier) if earlier else start
    if cue.position == "at_text" and cue.at_text:
        for _para, text, start, _speech_end, _pause_end in rows:
            if cue.at_text in text.replace("\u0301", ""):
                return start
        return rows[0][2]
    return rows[0][2]


def _bed_range(cue: Cue, spans: Sequence[Tuple[int, str, int, int, int]]) -> Optional[Tuple[int, int]]:
    """``(start, end)`` sample range of a bed, from the first paragraph through the pause after the last."""
    rows = [row for row in spans if cue.start_paragraph <= row[0] <= cue.end_paragraph and row[0] > 0]
    if not rows:
        return None
    return rows[0][2], rows[-1][4]


def _cache_key(kind: str, prompt: str, seconds: float, revision: str) -> str:
    """Stable name for one generated clip."""
    raw = f"{kind}\x00{prompt}\x00{seconds:.3f}\x00{revision}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _load_cached(folder, key: str) -> Optional[np.ndarray]:
    path = folder / f"{key}.wav"
    if not path.is_file():
        return None
    try:
        data, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError):
        return None
    return data if getattr(data, "ndim", 1) == 1 else data.mean(axis=1)


def _save_cached(folder, key: str, audio: np.ndarray, sr: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / f"{key}.wav"), np.asarray(audio, dtype=np.float32), sr)


def _render(cue: Cue, seconds: float, sr: int, generate: Generator, cache, revision: str) -> Optional[np.ndarray]:
    """One clip, from the cache when the same prompt was already rendered. A failure skips the cue."""
    key = _cache_key(cue.kind, cue.prompt, seconds, revision)
    if cache is not None:
        cached = _load_cached(cache, key)
        if cached is not None and cached.size:
            return cached
    try:
        audio = generate(cue.prompt, seconds, sr, cue.kind)
    except Exception as exc:  # noqa: BLE001 - a model failure must not stop the chapter
        log.warning("soundscape: cue %s (%s) could not be rendered: %s", cue.id, cue.kind, exc)
        return None
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        log.warning("soundscape: cue %s (%s) produced no audio and was skipped", cue.id, cue.kind)
        return None
    if cache is not None:
        _save_cached(cache, key, audio, sr)
    return audio


def match_speech_loudness(narr: np.ndarray, mixed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale ``mixed`` so speech stays at the narration's loudness, then peak-limit at 0.99.

    The narration as synthesized is the loudness target. There is no separate LUFS pass.
    """
    speech = mask > 0.5
    target = _rms(narr[speech]) if np.any(speech) else _rms(narr)
    current = _rms(mixed[speech]) if np.any(speech) else _rms(mixed)
    out = mixed.astype(np.float32, copy=True)
    if target > 1e-8 and current > 1e-8:
        out *= np.float32(target / current)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= np.float32(0.99 / peak)
    return out


def _mix_buffers(narr: np.ndarray, sr: int, spans: Sequence[Tuple[int, str, int, int, int]], plan: SoundPlan,
                 chapter_index: int, generate: Generator, cache, revision: str,
                 thermal=None, cancel=None, pause=None) -> np.ndarray:
    """Narration plus every cue of this chapter."""
    mix = narr.astype(np.float32, copy=True)
    mask = _speech_mask(narr, sr)
    speech = mask > 0.5
    reference = _rms(narr[speech]) if np.any(speech) else _rms(narr)
    if reference < 1e-8:
        reference = 0.1
    for cue in plan.cues:
        if cue.chapter_index != chapter_index:
            continue
        if thermal is not None and thermal.should_cool():
            thermal.cool(cancel, pause)
        if cancel is not None:
            cancel.check()
        level = reference * (10.0 ** ((cue.gain_db + plan.master_gain_db) / 20.0))
        if cue.kind == "bed":
            span = _bed_range(cue, spans)
            if span is None:
                log.warning("soundscape: cue %s has no audio in this chapter and was skipped", cue.id)
                continue
            start, end = span
            need = max(0, end - start)
            clip = _render(cue, BED_SECONDS, sr, generate, cache, revision)
            if clip is None:
                continue
            tiled = loop_crossfade(_normalise_clip(clip), need, int(sr * cue.crossfade_ms / 1000))
            tiled = _fade_edges(tiled, int(sr * cue.fade_in_ms / 1000), int(sr * cue.fade_out_ms / 1000))
            region = mask[start:start + tiled.size]
            duck = 10.0 ** (-cue.duck_db / 20.0)
            gain = np.float32(level) * (1.0 - region * np.float32(1.0 - duck))
            mix[start:start + tiled.size] += tiled * gain
            continue
        seconds = SHOT_SECONDS.get(cue.kind, 3.0)
        at = _place_sample(cue, spans)
        if at is None:
            log.warning("soundscape: cue %s has no audio in this chapter and was skipped", cue.id)
            continue
        clip = _render(cue, seconds, sr, generate, cache, revision)
        if clip is None:
            continue
        shot = _fade_edges(_normalise_clip(clip), int(0.02 * sr), int(0.05 * sr))
        # A one-shot longer than the requested duration is trimmed. sfx uses this same path.
        n = min(shot.size, int(seconds * sr), mix.size - at)
        if n <= 0:
            continue
        mix[at:at + n] += shot[:n] * np.float32(level)
    return match_speech_loudness(narr, mix, mask)


def mix_chapter_file(chapter: ChapterAudio, spans: Sequence[Tuple[int, str, int, int, int]], plan: SoundPlan,
                     chapter_index: int, request: SoundRequest, cache, thermal=None, cancel=None,
                     pause=None, revision: str = "") -> ChapterAudio:
    """Mix one chapter file. ``chapter_index`` is the book's chapter number."""
    try:
        narr, sr = sf.read(str(chapter.wav), dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        log.warning("soundscape: chapter %s could not be read: %s", chapter_index, exc)
        return chapter
    if getattr(narr, "ndim", 1) != 1:
        narr = narr.mean(axis=1)
    if not revision:
        revision = soundscape_model.source_revision()
    mixed = _mix_buffers(
        np.asarray(narr, dtype=np.float32), int(sr), spans, plan, chapter_index,
        _renderer(request), cache, revision, thermal, cancel, pause)
    sf.write(str(chapter.wav), mixed, int(sr))
    return chapter
