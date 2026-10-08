"""infra/runtime_lock.json: every third-party file the thin installer downloads is pinned and comes from its upstream site."""
import json
import re
import urllib.parse
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools import make_runtime_lock as mk

ROOT = Path(__file__).resolve().parent.parent
LOCK = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
UPSTREAM_HOSTS = {"files.pythonhosted.org", "download.pytorch.org"}


def test_every_file_is_pinned_and_comes_from_its_upstream_site():
    assert LOCK["schema"] == 1 and LOCK["platform"] == "win_amd64" and LOCK["python"] == "3.11"
    seen = set()
    for w in LOCK["wheels"]:
        assert urllib.parse.urlparse(w["url"]).scheme == "https" and urllib.parse.urlparse(w["url"]).hostname in UPSTREAM_HOSTS, w
        assert re.fullmatch(r"[0-9a-f]{64}", w["sha256"]) and w["size"] > 0 and w["file"] and w["version"]
        assert w["url"].rsplit("/", 1)[-1].replace("%2B", "+") == w["file"]
        key = (w["dist"], w.get("flavor"))
        assert key not in seen, key
        seen.add(key)


def test_torch_has_all_flavors_with_one_version_and_nothing_else_is_torch():
    for pkg in mk.TORCH:
        flavors = {w["flavor"]: w for w in LOCK["wheels"] if w["dist"] == pkg}
        assert set(flavors) == set(LOCK["flavors"]) == {"cu128", "cu126", "cpu"}
        for fl, w in flavors.items():
            assert w["version"] == f"{LOCK['torch_version']}+{fl}" and w["url"].startswith(f"https://download.pytorch.org/whl/{fl}/")
            assert w["file"].endswith("cp311-cp311-win_amd64.whl") and w["group"] == "torch"
    assert not [w for w in LOCK["wheels"] if w["group"] == "torch" and w["dist"] not in mk.TORCH]


def test_nothing_the_shell_bundles_is_downloaded_again():
    names = {mk.norm(w["dist"]) for w in LOCK["wheels"]}
    assert not names & mk.SHELL_PROVIDED
    assert set(LOCK["shell"]) <= mk.SHELL_PROVIDED and {"pyside6", "numpy", "soundfile"} <= set(LOCK["shell"])


def test_sdist_only_packages_are_few_and_known():
    assert {w["dist"] for w in LOCK["wheels"] if w.get("sdist")} <= {"docopt", "eng-to-ipa", "sox"}


def test_the_lock_covers_requirements_txt():
    # build tooling (setuptools etc.) is pinned in requirements.txt for dev venvs but never shipped
    have = {mk.norm(w["dist"]) for w in LOCK["wheels"]} | set(LOCK["shell"]) | set(mk.TORCH) | mk.SKIP
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        r = Requirement(line)
        if r.marker is not None and not r.marker.evaluate({"python_version": "3.11"}):
            continue
        assert mk.norm(r.name) in have, f"{r.name} is in requirements.txt but not in runtime_lock.json (run tools/make_runtime_lock.py)"


@pytest.mark.parametrize("fn,ok", [
    ("numpy-2.4.6-cp311-cp311-win_amd64.whl", True),
    ("numpy-2.4.6-cp312-cp312-win_amd64.whl", False),
    ("numpy-2.4.6-cp311-cp311-manylinux_2_28_x86_64.whl", False),
    ("scipy-1.1-cp39-abi3-win_amd64.whl", True),
    ("x-1-py3-none-any.whl", True),
    ("bitsandbytes-0.50-py3-none-win_amd64.whl", True),
    ("bitsandbytes-0.50-py3-none-win_arm64.whl", False),
])
def test_wheel_selection(fn, ok):
    assert (mk.wheel_score(fn, "cp311") is not None) is ok


def test_the_platform_specific_wheel_wins_over_a_generic_one():
    assert mk.wheel_score("a-1-cp311-cp311-win_amd64.whl", "cp311") > mk.wheel_score("a-1-py3-none-any.whl", "cp311")
