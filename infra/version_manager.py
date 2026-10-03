"""Версии пакетов (PyPI) и моделей (HF Hub): сравнение, поиск последней совместимой версии.

Два канала обновлений (канал по умолчанию - «verified»):
  * verified - целевые версии берутся из манифеста «проверено Voxprint» (infra/verified_manifest.py):
    ставим именно проверенные версии (в том числе откат вниз, если установлено что-то другое);
  * latest   - последняя стабильная версия на PyPI, совместимая с ограничениями (TRACKED_PACKAGES).
Последняя стабильная версия PyPI проверяется всегда, но в канале verified она только информирует
(`PackageStatus.newer_unverified`) и сама не устанавливается.

Сетевые функции принимают `fetch_json` - подставляемый загрузчик, поэтому всё тестируется без сети.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

log = logging.getLogger("voxprint.versions")

#: Пакеты, которые обновляем, и ограничения совместимости.
#: transformers < 5: qwen-asr 0.0.6 и qwen-tts 0.1.1 написаны под transformers 4.57.x (проверено по METADATA).
TRACKED_PACKAGES: Dict[str, str] = {
    "qwen-asr": "",
    "qwen-tts": "",
    "transformers": ">=4.57.6,<5",
    "peft": "",
    "accelerate": "",
    "bitsandbytes": "",
    "huggingface_hub": "<1.0",  # transformers 4.57.x требует huggingface-hub<1.0
    "ru-normalizr": "",
    "rutextnorm": "",
    "ctc-forced-aligner": "",
}
#: Необязательные пакеты: если не установлены, обновлятор их не ставит (приложение работает и без них).
OPTIONAL_PACKAGES = frozenset({"bitsandbytes", "ru-normalizr", "rutextnorm", "ctc-forced-aligner"})
#: Модели HF, за которыми следим (аудио-токенайзер лежит внутри Base-моделей).
TRACKED_MODELS = (
    "Qwen/Qwen3-ForcedAligner-0.6B",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
CHANNEL_VERIFIED = "verified"
CHANNEL_LATEST = "latest"

FetchJson = Callable[[str], Any]

# ---- решение по компоненту: использовать готовое / обновить / поставить
ACTION_REUSE = "reuse"        # установлено и подходит - ничего не трогаем
ACTION_UPGRADE = "upgrade"    # установлено, но старее целевой версии - обновляем в СОБСТВЕННОМ окружении Voxprint
ACTION_INSTALL = "install"    # нет (или несовместимо) - ставим целевую версию в собственное окружение
ACTION_IGNORE = "ignore"      # не используем (например Unsloth: Qwen3-TTS он пока не поддерживает)
#: Версии, прогнанные вместе с Voxprint, но новее закреплённой в манифесте: их тоже можно использовать как есть.
#: peft: закреплён 0.18.1 (как у Alexandria), на CPU с крошечной моделью проверена и 0.21.2.
TESTED_COMPATIBLE: Dict[str, frozenset] = {"peft": frozenset({"0.18.1", "0.21.2"})}
#: Известные пакеты, которые Voxprint намеренно не использует, и причина (стабильный код).
IGNORED_PACKAGES: Dict[str, str] = {"unsloth": "no_qwen3_tts_training"}


@dataclass(frozen=True)
class Decision:
    name: str
    installed: Optional[str]
    target: Optional[str]
    action: str
    reason: str           # стабильный код: current | compatible_newer | outdated | missing | incompatible | ...


def decide_package(name: str, installed: Optional[str], target: Optional[str], constraint: str = "") -> Decision:
    """Правило: готовое используем, если оно актуально или проверенно совместимо; старее - обновляем (в окружении
    Voxprint, чужие окружения не трогаем); нет - ставим. target - закреплённая (или новейшая стабильная) версия."""
    if name in IGNORED_PACKAGES:
        return Decision(name, installed, None, ACTION_IGNORE, IGNORED_PACKAGES[name])
    if target is None:
        return Decision(name, installed, None, ACTION_REUSE if installed else ACTION_IGNORE,
                        "no_target" if installed else "optional_absent")
    if installed is None:
        return Decision(name, None, target, ACTION_INSTALL, "missing")
    iv = parse_version(installed)
    if constraint and iv is not None and iv not in SpecifierSet(constraint):
        return Decision(name, installed, target, ACTION_INSTALL, "incompatible")
    c = compare_versions(installed, target)
    if c == 0:
        return Decision(name, installed, target, ACTION_REUSE, "current")
    if c < 0:
        return Decision(name, installed, target, ACTION_UPGRADE, "outdated")
    if installed in TESTED_COMPATIBLE.get(name, ()):
        return Decision(name, installed, target, ACTION_REUSE, "compatible_newer")
    return Decision(name, installed, target, ACTION_INSTALL, "unverified_newer")


def parse_version(s: str) -> Optional[Version]:
    try:
        return Version(s)
    except InvalidVersion:
        return None


def compare_versions(a: str, b: str) -> int:
    """-1 если a<b, 0 если равны, 1 если a>b. Невалидные версии считаются меньше валидных."""
    va, vb = parse_version(a), parse_version(b)
    if va is None and vb is None:
        return (a > b) - (a < b)
    if va is None:
        return -1
    if vb is None:
        return 1
    return (va > vb) - (va < vb)


def is_newer(candidate: str, installed: Optional[str]) -> bool:
    return installed is None or compare_versions(candidate, installed) > 0


def installed_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def http_fetch_json(url: str, timeout: float = 6.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Voxprint-updater"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - только https к PyPI/HF
        return json.loads(r.read().decode("utf-8"))


def latest_compatible_pypi(name: str, constraint: str = "", fetch_json: FetchJson = http_fetch_json) -> Optional[str]:
    """Самая свежая стабильная (не pre-release, не yanked) версия, удовлетворяющая ограничению."""
    data = fetch_json(f"https://pypi.org/pypi/{name}/json")
    spec = SpecifierSet(constraint) if constraint else SpecifierSet()
    best: Optional[Version] = None
    for ver, files in (data.get("releases") or {}).items():
        v = parse_version(ver)
        if v is None or v.is_prerelease or v.is_devrelease:
            continue
        if files and all(f.get("yanked") for f in files):
            continue
        if not files:
            continue
        if v not in spec:
            continue
        if best is None or v > best:
            best = v
    return str(best) if best else None


def hf_model_revision(repo_id: str, fetch_json: FetchJson = http_fetch_json) -> Optional[Dict[str, str]]:
    """{'sha': ..., 'last_modified': ...} с HF Hub API (у моделей нет semver - версией служит коммит)."""
    data = fetch_json(f"https://huggingface.co/api/models/{repo_id}")
    sha = data.get("sha")
    return {"sha": sha, "last_modified": data.get("lastModified", "")} if sha else None


@dataclass
class PackageStatus:
    name: str
    installed: Optional[str]
    latest_stable: Optional[str]      # последняя стабильная на PyPI (с учётом ограничений)
    constraint: str = ""
    target: Optional[str] = None      # что нужно установить в выбранном канале (None - ничего)
    pinned: bool = False              # target взят из манифеста «проверено Voxprint»
    error: str = ""

    @property
    def update_available(self) -> bool:
        """Нужно менять установленную версию (в т.ч. откатить к проверенной)."""
        return self.decision.action in (ACTION_UPGRADE, ACTION_INSTALL)

    @property
    def decision(self) -> Decision:
        return decide_package(self.name, self.installed, self.target, self.constraint)

    @property
    def newer_unverified(self) -> bool:
        """На PyPI есть версия новее целевой, но Voxprint её пока не проверял."""
        ref = self.target or self.installed
        return bool(self.latest_stable) and ref is not None and is_newer(self.latest_stable, ref)  # type: ignore[arg-type]


@dataclass
class ModelStatus:
    repo_id: str
    local_sha: Optional[str]
    remote_sha: Optional[str]         # HEAD репозитория на HF (информативно)
    target_sha: Optional[str] = None  # проверенная ревизия из манифеста / HEAD в канале latest
    error: str = ""

    @property
    def update_available(self) -> bool:
        return bool(self.target_sha) and self.local_sha is not None and self.local_sha != self.target_sha

    @property
    def newer_unverified(self) -> bool:
        return bool(self.remote_sha) and bool(self.target_sha) and self.remote_sha != self.target_sha


@dataclass
class VersionReport:
    packages: List[PackageStatus] = field(default_factory=list)
    models: List[ModelStatus] = field(default_factory=list)
    network_ok: bool = True
    channel: str = CHANNEL_VERIFIED
    manifest_date: str = ""

    @property
    def outdated_packages(self) -> List[PackageStatus]:
        return [p for p in self.packages if p.update_available]

    @property
    def outdated_models(self) -> List[ModelStatus]:
        return [m for m in self.models if m.update_available]

    @property
    def unverified_newer(self) -> List[str]:
        """Для информации: пакеты, у которых на PyPI вышла версия новее проверенной."""
        return [f"{p.name} {p.latest_stable}" for p in self.packages if p.newer_unverified]

    @property
    def has_updates(self) -> bool:
        return bool(self.outdated_packages or self.outdated_models)


def check_versions(
    local_model_shas: Optional[Dict[str, str]] = None,
    fetch_json: FetchJson = http_fetch_json,
    installed_fn: Callable[[str], Optional[str]] = installed_version,
    packages: Optional[Dict[str, str]] = None,
    models: Optional[tuple] = None,
    manifest: Optional[Any] = None,
    channel: str = CHANNEL_VERIFIED,
) -> VersionReport:
    """manifest - infra.verified_manifest.Manifest (или None: pins нет). channel: verified | latest."""
    local_model_shas = local_model_shas or {}
    pins = dict(getattr(manifest, "packages", {}) or {})
    model_pins = dict(getattr(manifest, "models", {}) or {})
    rep = VersionReport(channel=channel, manifest_date=getattr(manifest, "date", "") or "")
    errors = 0
    for name, constraint in (packages if packages is not None else TRACKED_PACKAGES).items():
        inst = installed_fn(name)
        st = PackageStatus(name, inst, None, constraint)
        try:
            st.latest_stable = latest_compatible_pypi(name, constraint, fetch_json)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            st.error = str(exc)
            errors += 1
        if channel == CHANNEL_LATEST:
            st.target = st.latest_stable
        elif name in pins:
            st.target, st.pinned = pins[name], True
        elif inst is None and name not in OPTIONAL_PACKAGES:
            st.target = st.latest_stable        # обязательный пакет отсутствует - ставим последнюю совместимую
        if inst is None and name in OPTIONAL_PACKAGES:
            st.target = None                    # необязательное не ставим сами
        rep.packages.append(st)
    for repo in (models if models is not None else TRACKED_MODELS):
        ms = ModelStatus(repo, local_model_shas.get(repo), None)
        try:
            rev = hf_model_revision(repo, fetch_json)
            ms.remote_sha = rev["sha"] if rev else None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            ms.error = str(exc)
            errors += 1
        ms.target_sha = ms.remote_sha if channel == CHANNEL_LATEST else (model_pins.get(repo) or ms.remote_sha)
        rep.models.append(ms)
    total = len(rep.packages) + len(rep.models)
    rep.network_ok = errors < total
    return rep
