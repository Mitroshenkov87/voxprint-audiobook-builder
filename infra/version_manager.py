"""Versions of packages (PyPI) and models (HF Hub): comparison and finding the latest compatible version.

Two update channels (the default is "verified"):

* ``verified`` - target versions come from the "verified by Voxprint" manifest (``infra/verified_manifest.py``):
  exactly the tested versions are installed (including a downgrade if something else is installed);
* ``latest`` - the newest stable PyPI release that satisfies the constraints (``TRACKED_PACKAGES``).

The newest stable PyPI release is always looked up, but in the ``verified`` channel it is informational only
(``PackageStatus.newer_unverified``) and is never installed by itself.

Network functions take a ``fetch_json`` callable, so everything is testable without a network.
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

#: Packages we keep up to date, with their compatibility constraints.
#: transformers < 5: qwen-asr 0.0.6 and qwen-tts 0.1.1 are written for transformers 4.57.x (checked in their METADATA).
TRACKED_PACKAGES: Dict[str, str] = {
    "qwen-asr": "",
    "qwen-tts": "",
    "transformers": ">=4.57.6,<5",
    "peft": "",
    "accelerate": "",
    "bitsandbytes": "",
    "huggingface_hub": "<1.0",  # transformers 4.57.x requires huggingface-hub<1.0
    "ru-normalizr": "",
    "rutextnorm": "",
    "ctc-forced-aligner": "",
}
#: Optional packages: if absent, the updater does not install them (the app works without them).
OPTIONAL_PACKAGES = frozenset({"bitsandbytes", "ru-normalizr", "rutextnorm", "ctc-forced-aligner"})
#: Hugging Face models we track (the audio tokenizer lives inside the Base models).
TRACKED_MODELS = (
    "Qwen/Qwen3-ForcedAligner-0.6B",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
CHANNEL_VERIFIED = "verified"
CHANNEL_LATEST = "latest"

FetchJson = Callable[[str], Any]

# ---- decision per component: reuse what is installed / upgrade / install
ACTION_REUSE = "reuse"        # installed and suitable - leave it alone
ACTION_UPGRADE = "upgrade"    # installed but older than the target - upgrade it in Voxprint's OWN environment
ACTION_INSTALL = "install"    # missing (or incompatible) - install the target version into Voxprint's own environment
ACTION_OFFER = "offer_upgrade"  # outdated in the USER's environment: touch nothing, ask first
ACTION_IGNORE = "ignore"      # deliberately not used (e.g. Unsloth: it does not support Qwen3-TTS yet)
#: Versions that were run with Voxprint but are newer than the manifest pin: they can be used as they are.
#: peft: 0.18.1 is pinned (as in Alexandria); 0.21.2 was also tested on the CPU with a tiny model.
TESTED_COMPATIBLE: Dict[str, frozenset] = {"peft": frozenset({"0.18.1", "0.21.2"})}
#: Known packages Voxprint deliberately does not use, with the reason (a stable code).
IGNORED_PACKAGES: Dict[str, str] = {"unsloth": "no_qwen3_tts_training"}


@dataclass(frozen=True)
class Decision:
    """What to do with one component: the versions involved, an ``ACTION_*`` and a stable ``reason`` code."""
    name: str
    installed: Optional[str]
    target: Optional[str]
    action: str
    reason: str           # stable code: current | compatible_newer | outdated | missing | incompatible | ...
    outdated: bool = False       # the installed version is older than the target
    compatible: bool = True      # the installed version is still acceptable (within the constraints)


def decide_package(name: str, installed: Optional[str], target: Optional[str], constraint: str = "",
                   external: bool = False) -> Decision:
    """Decide reuse / upgrade / offer / install for one package.

    ``external=True`` means the component lives in the USER's environment (not in Voxprint's own).  An outdated component
    there is never updated silently: the decision is ``ACTION_OFFER`` (the UI asks; for a refusal see ``Updater.apply``).

    Rule: what is installed is used if it is current or verified-compatible; older -> upgrade (in Voxprint's environment -
    foreign environments are not touched); missing -> install.  ``target`` is the pinned (or newest stable) version.
    """
    if name in IGNORED_PACKAGES:
        return Decision(name, installed, None, ACTION_IGNORE, IGNORED_PACKAGES[name])
    if target is None:
        return Decision(name, installed, None, ACTION_REUSE if installed else ACTION_IGNORE,
                        "no_target" if installed else "optional_absent")
    if installed is None:
        return Decision(name, None, target, ACTION_INSTALL, "missing")
    iv = parse_version(installed)
    c = compare_versions(installed, target)
    if constraint and iv is not None and iv not in SpecifierSet(constraint):
        older = c < 0
        return Decision(name, installed, target, ACTION_OFFER if (external and older) else ACTION_INSTALL,
                        "incompatible", outdated=older, compatible=False)
    if c == 0:
        return Decision(name, installed, target, ACTION_REUSE, "current")
    if c < 0:
        return Decision(name, installed, target, ACTION_OFFER if external else ACTION_UPGRADE, "outdated",
                        outdated=True, compatible=True)
    if installed in TESTED_COMPATIBLE.get(name, ()):
        return Decision(name, installed, target, ACTION_REUSE, "compatible_newer")
    return Decision(name, installed, target, ACTION_INSTALL, "unverified_newer")


def parse_version(s: str) -> Optional[Version]:
    """Parse a PEP 440 version string; None if it is invalid."""
    try:
        return Version(s)
    except InvalidVersion:
        return None


def compare_versions(a: str, b: str) -> int:
    """-1 if a<b, 0 if equal, 1 if a>b.  Invalid versions sort below valid ones."""
    va, vb = parse_version(a), parse_version(b)
    if va is None and vb is None:
        return (a > b) - (a < b)
    if va is None:
        return -1
    if vb is None:
        return 1
    return (va > vb) - (va < vb)


def is_newer(candidate: str, installed: Optional[str]) -> bool:
    """True if ``candidate`` is newer than ``installed`` (or nothing is installed)."""
    return installed is None or compare_versions(candidate, installed) > 0


def installed_version(name: str) -> Optional[str]:
    """Installed version of a distribution via importlib.metadata, or None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def http_fetch_json(url: str, timeout: float = 6.0) -> Any:
    """GET a JSON document (PyPI / HF API) with a short timeout; the production ``fetch_json``."""
    req = urllib.request.Request(url, headers={"User-Agent": "Voxprint-updater"})
    from infra import net

    with net.urlopen(req, timeout) as r:  # noqa: S310 - https to PyPI/HF only
        return json.loads(r.read().decode("utf-8"))


def latest_compatible_pypi(name: str, constraint: str = "", fetch_json: FetchJson = http_fetch_json) -> Optional[str]:
    """Newest stable release (not a pre-release, not yanked) that satisfies ``constraint``, or None."""
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
    """``{'sha': ..., 'last_modified': ...}`` from the HF Hub API (models have no semver - the commit is the version)."""
    data = fetch_json(f"https://huggingface.co/api/models/{repo_id}")
    sha = data.get("sha")
    return {"sha": sha, "last_modified": data.get("lastModified", "")} if sha else None


@dataclass
class PackageStatus:
    """Installed/target/latest information about one tracked package, and the derived decision."""
    name: str
    installed: Optional[str]
    latest_stable: Optional[str]      # latest stable on PyPI (within the constraints)
    constraint: str = ""
    target: Optional[str] = None      # what should be installed in the selected channel (None = nothing)
    pinned: bool = False              # target comes from the "verified by Voxprint" manifest
    error: str = ""
    external: bool = False            # installed in the user's environment rather than Voxprint's own

    @property
    def update_available(self) -> bool:
        """The installed version must be changed automatically (including rolling back to the verified one)."""
        return self.decision.action in (ACTION_UPGRADE, ACTION_INSTALL)

    @property
    def offer_available(self) -> bool:
        """The user must be asked (an outdated component in their own environment)."""
        return self.decision.action == ACTION_OFFER

    @property
    def decision(self) -> Decision:
        """The :class:`Decision` for this package."""
        return decide_package(self.name, self.installed, self.target, self.constraint, self.external)

    @property
    def newer_unverified(self) -> bool:
        """PyPI has a version newer than the target that Voxprint has not verified yet."""
        ref = self.target or self.installed
        return bool(self.latest_stable) and ref is not None and is_newer(self.latest_stable, ref)  # type: ignore[arg-type]


@dataclass
class ModelStatus:
    """Local vs. remote vs. target commit of one tracked model."""
    repo_id: str
    local_sha: Optional[str]
    remote_sha: Optional[str]         # HEAD of the repository on HF (informational)
    target_sha: Optional[str] = None  # verified revision from the manifest / HEAD in the latest channel
    error: str = ""

    @property
    def update_available(self) -> bool:
        """A local copy exists and differs from the target revision."""
        return bool(self.target_sha) and self.local_sha is not None and self.local_sha != self.target_sha

    @property
    def newer_unverified(self) -> bool:
        """The remote HEAD differs from the verified target revision."""
        return bool(self.remote_sha) and bool(self.target_sha) and self.remote_sha != self.target_sha


@dataclass
class VersionReport:
    """Result of :func:`check_versions`: per-package and per-model status plus channel and network health."""
    packages: List[PackageStatus] = field(default_factory=list)
    models: List[ModelStatus] = field(default_factory=list)
    network_ok: bool = True
    channel: str = CHANNEL_VERIFIED
    manifest_date: str = ""

    @property
    def outdated_packages(self) -> List[PackageStatus]:
        """Packages that must be changed automatically."""
        return [p for p in self.packages if p.update_available]

    @property
    def outdated_models(self) -> List[ModelStatus]:
        """Models whose local revision differs from the target revision."""
        return [m for m in self.models if m.update_available]

    @property
    def unverified_newer(self) -> List[str]:
        """For information: packages for which PyPI has a version newer than the verified one."""
        return [f"{p.name} {p.latest_stable}" for p in self.packages if p.newer_unverified]

    @property
    def offered_packages(self) -> List[PackageStatus]:
        """Packages to offer to the user (outdated in their own environment)."""
        return [p for p in self.packages if p.offer_available]

    @property
    def has_updates(self) -> bool:
        """True if anything needs changing or offering."""
        return bool(self.outdated_packages or self.outdated_models or self.offered_packages)


def check_versions(
    local_model_shas: Optional[Dict[str, str]] = None,
    fetch_json: FetchJson = http_fetch_json,
    installed_fn: Callable[[str], Optional[str]] = installed_version,
    packages: Optional[Dict[str, str]] = None,
    models: Optional[tuple] = None,
    manifest: Optional[Any] = None,
    channel: str = CHANNEL_VERIFIED,
    external_env: bool = False,
) -> VersionReport:
    """Compare installed packages/models with PyPI / HF and the manifest pins.

    ``manifest`` is an :class:`infra.verified_manifest.Manifest` (or None: no pins).  ``channel``: ``verified`` | ``latest``.
    ``network_ok`` is False only when every single lookup failed.
    """
    local_model_shas = local_model_shas or {}
    pins = dict(getattr(manifest, "packages", {}) or {})
    model_pins = dict(getattr(manifest, "models", {}) or {})
    rep = VersionReport(channel=channel, manifest_date=getattr(manifest, "date", "") or "")
    errors = 0
    for name, constraint in (packages if packages is not None else TRACKED_PACKAGES).items():
        inst = installed_fn(name)
        st = PackageStatus(name, inst, None, constraint, external=external_env)
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
            st.target = st.latest_stable        # a required package is missing - install the latest compatible one
        if inst is None and name in OPTIONAL_PACKAGES:
            st.target = None                    # optional packages are never installed on our own
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


@dataclass(frozen=True)
class Offer:
    """What we offer to update in the user's environment (shown by the upgrade dialog)."""
    name: str
    installed: str
    target: str
    compatible: bool      # True: if declined we use it as it is; False: if declined Voxprint's own copy is used
    env: str              # which environment would change (path)


def make_offers(report: "VersionReport", env: str) -> List[Offer]:
    """Build the :class:`Offer` list for the packages that need the user's consent in environment ``env``."""
    return [Offer(p.name, p.installed or "", p.target or "", p.decision.compatible, env)
            for p in report.offered_packages if p.target]
