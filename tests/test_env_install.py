"""Environment probe (reuse / upgrade / install), install manifest + health check, pinned asset installer."""
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from core import i18n
from infra import assets, env_probe as ep, install_state as ist, paths
from infra.updater import Updater
from infra.verified_manifest import Manifest
from infra.version_manager import (ACTION_IGNORE, ACTION_INSTALL, ACTION_OFFER, ACTION_REUSE, ACTION_UPGRADE, PackageStatus,
                                   check_versions, decide_package)


# ------------------------------------------------------------------------------------- decisions
def test_reuse_when_current_upgrade_when_outdated_install_when_missing():
    assert decide_package("qwen-asr", "0.0.6", "0.0.6").action == ACTION_REUSE
    assert decide_package("qwen-asr", "0.0.6", "0.0.6").reason == "current"
    d = decide_package("accelerate", "1.12.0", "1.15.0")
    assert (d.action, d.reason) == (ACTION_UPGRADE, "outdated")
    d = decide_package("peft", None, "0.18.1")
    assert (d.action, d.reason) == (ACTION_INSTALL, "missing")


def test_newer_than_verified_is_reused_only_if_tested():
    assert decide_package("peft", "0.21.2", "0.18.1").action == ACTION_REUSE      # tested together with Voxprint
    assert decide_package("peft", "0.21.2", "0.18.1").reason == "compatible_newer"
    d = decide_package("accelerate", "1.99.0", "1.15.0")                            # newer but never verified
    assert (d.action, d.reason) == (ACTION_INSTALL, "unverified_newer")


def test_constraint_violation_is_replaced_even_if_newer():
    d = decide_package("transformers", "5.1.0", "4.57.6", ">=4.57.6,<5")
    assert (d.action, d.reason) == (ACTION_INSTALL, "incompatible")
    assert decide_package("transformers", "4.57.3", "4.57.6", ">=4.57.6,<5").action == ACTION_INSTALL


def test_unsloth_is_never_used():
    d = decide_package("unsloth", "2026.9.11", "2026.9.11")
    assert d.action == ACTION_IGNORE and d.reason == "no_qwen3_tts_training"


def test_package_status_follows_the_decision():
    assert not PackageStatus("peft", "0.21.2", "0.21.2", "", target="0.18.1", pinned=True).update_available
    assert PackageStatus("peft", "0.17.0", "0.21.2", "", target="0.18.1", pinned=True).update_available
    assert PackageStatus("peft", None, "0.21.2", "", target="0.18.1").update_available
    assert not PackageStatus("peft", "0.18.1", "0.21.2", "", target="0.18.1").update_available


def test_check_versions_counts_actions():
    m = Manifest(date="2026-10-02", packages={"a": "2.0", "b": "1.0", "c": "3.0"})
    rep = check_versions({}, lambda u: {"releases": {"9.9": [{}]}}, {"a": "1.0", "b": "1.0"}.get,
                         packages={"a": "", "b": "", "c": ""}, models=(), manifest=m)
    acts = {p.name: p.decision.action for p in rep.packages}
    assert acts == {"a": ACTION_UPGRADE, "b": ACTION_REUSE, "c": ACTION_INSTALL}


# ------------------------------------------------------------------------------------- torch flavor
@pytest.mark.parametrize("smi,flavor", [
    ("| NVIDIA-SMI 581.15  Driver Version: 581.15  CUDA Version: 13.0 |", "cu130"),
    ("NVIDIA-SMI 610.88  KMD Version: 610.88  CUDA UMD Version: 13.3", "cu130"),     # drivers 6xx: "UMD"
    ("CUDA Version: 12.9", "cu128"), ("CUDA Version: 12.6", "cu126"), ("CUDA Version: 12.4", "cu124"),
    ("CUDA Version: 11.8", "cu118"), ("CUDA Version: 11.2", "cpu"), ("no gpu here", "cpu")])
def test_torch_flavor_from_driver(smi, flavor):
    assert ep.torch_flavor_for_driver(ep.parse_nvidia_smi_cuda(smi)) == flavor


def test_torch_decisions():
    assert ep.decide_torch(None, "cu128").action == ACTION_INSTALL
    assert ep.decide_torch("2.8.0+cu128", "cu128").action == ACTION_REUSE
    d = ep.decide_torch("2.8.0+cpu", "cu128")
    assert (d.action, d.reason) == (ACTION_UPGRADE, "torch_flavor_changed")      # CPU build on a GPU machine
    assert ep.decide_torch("2.8.0+cu124", "cu128").action == ACTION_REUSE        # older CUDA build runs on a newer driver
    assert ep.decide_torch("2.8.0+cu130", "cu126").action == ACTION_UPGRADE      # driver too old for this build
    assert ep.decide_torch("2.8.0+cu128", "cpu").action == ACTION_REUSE          # no GPU: harmless
    assert ep.decide_torch("2.8.0", "cu128").reason == "flavor_unknown"


# ------------------------------------------------------------------------------------- probe with fake machine
def fake_machine(smi=None, ffmpeg_ok=True, pythons=None):
    calls = []

    def run(args):
        calls.append(args)
        exe = Path(args[0]).name
        if exe.startswith("nvidia-smi"):
            return (0, smi) if smi else (1, "")
        if exe.startswith("ffmpeg"):
            return (0, "ffmpeg version 7.1.1 Copyright") if ffmpeg_ok else (1, "boom")
        if args[1:2] == ["-I"] and pythons and args[0] in pythons:
            return 0, "noise\nVXPROBE " + json.dumps(pythons[args[0]])
        return 1, ""

    def which(n):
        return {"nvidia-smi": "/usr/bin/nvidia-smi" if smi else None, "ffmpeg": "/usr/bin/ffmpeg"}.get(n)

    return run, which, calls


def test_probe_environment_reuse_upgrade_install_and_report():
    run, which, calls = fake_machine(smi="CUDA Version: 12.8")
    inst = {"qwen-asr": "0.0.6", "qwen-tts": "0.1.1", "transformers": "4.57.6", "peft": "0.21.2",
            "accelerate": "1.12.0", "torch": "2.8.0+cpu", "unsloth": "2026.9.11"}.get
    pins = {"qwen-asr": "0.0.6", "qwen-tts": "0.1.1", "transformers": "4.57.6", "peft": "0.18.1",
            "accelerate": "1.15.0", "huggingface_hub": "0.36.2"}
    rep = ep.probe_environment(pins, inst, run, which, scan_other_pythons=False,
                               packages={"qwen-asr": "", "qwen-tts": "", "transformers": ">=4.57.6,<5", "peft": "",
                                         "accelerate": "", "huggingface_hub": "<1.0", "bitsandbytes": ""})
    act = {d.name: (d.action, d.reason) for d in rep.decisions}
    assert act["qwen-asr"] == (ACTION_REUSE, "current")
    assert act["peft"] == (ACTION_REUSE, "compatible_newer")
    assert act["accelerate"] == (ACTION_UPGRADE, "outdated")
    assert act["huggingface_hub"] == (ACTION_INSTALL, "missing")
    assert "bitsandbytes" not in act                                         # optional and absent: left alone
    assert act["torch"] == (ACTION_UPGRADE, "torch_flavor_changed") and rep.wanted_torch_flavor == "cu128"
    assert rep.ffmpeg and rep.ffmpeg.ok and rep.ffmpeg.version == "7.1.1"
    assert rep.ignored == {"unsloth": "no_qwen3_tts_training"}
    assert rep.counts() == {ACTION_REUSE: 4, ACTION_UPGRADE: 2, ACTION_OFFER: 0, ACTION_INSTALL: 1}


def test_broken_ffmpeg_is_not_trusted():
    run, which, _ = fake_machine(ffmpeg_ok=False)
    info = ep.probe_ffmpeg(which, run)
    assert info and not info.ok
    assert ep.probe_ffmpeg(lambda n: None, run) is None


def test_other_pythons_are_only_asked_never_modified(tmp_path, monkeypatch):
    home = tmp_path / "home"
    exe_dir = home / "pinokio" / "api" / "alexandria-audiobook.git" / "app" / "env" / ("Scripts" if sys.platform == "win32" else "bin")
    exe_dir.mkdir(parents=True)
    exe = exe_dir / ("python.exe" if sys.platform == "win32" else "python")
    exe.write_text("")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("PINOKIO_HOME", raising=False)
    before = sorted(str(p) for p in home.rglob("*"))
    cands = ep.candidate_pythons(lambda n: None, {})
    assert cands == [(str(exe), "pinokio")]
    run, which, calls = fake_machine(pythons={str(exe): {"python": [3, 11, 9], "packages": {
        "torch": "2.7.0+cu126", "peft": "0.18.1", "unsloth": "2026.9.11"}}})
    rep = ep.probe_environment({}, lambda n: None, run, which, scan_other_pythons=True, packages={},
                               environ={})
    assert [(e.version, e.source) for e in rep.pythons] == [((3, 11, 9), "pinokio")]
    assert rep.pythons[0].packages["peft"] == "0.18.1" and "unsloth" in rep.ignored
    assert all(c[1] == "-I" for c in calls if Path(c[0]).name.startswith("python"))     # isolated, metadata only
    assert sorted(str(p) for p in home.rglob("*")) == before
    assert ep.pick_pip_python(rep.pythons, (3, 11)) == str(exe)
    assert ep.pick_pip_python(rep.pythons, (3, 12)) is None


def test_user_messages_in_all_languages():
    run, which, _ = fake_machine()
    rep = ep.probe_environment({"a": "2.0", "b": "1.0"}, {"a": "1.0", "b": "1.0", "unsloth": "1"}.get, run, which,
                               scan_other_pythons=False, packages={"a": "", "b": "", "c": ""})
    rep.decisions.append(ep.Decision("c", None, "3.0", ACTION_INSTALL, "missing"))
    seen = set()
    for lang in i18n.LANGS:
        i18n.set_language(lang)
        msgs = ep.user_messages(rep)
        assert len(msgs) >= 4 and all("{" not in m for m in msgs)
        seen.add(msgs[0])
    assert len(seen) == len(i18n.LANGS)


def test_updater_reports_environment_and_upgrades_only_own_overlay(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXPRINT_NO_ENV_PROBE")
    pip_calls = []

    def pip(cmd):
        pip_calls.append(cmd)
        return 0, ""

    def smoke(cmd):
        return 0, 'SMOKE_OK {"accelerate": "1.15.0"}'

    def fetch(url):
        return {"releases": {"1.15.0": [{}]}} if "pypi" in url else {"sha": "x"}

    msgs = []
    u = Updater(fetch_json=fetch, pip_runner=pip, smoke_runner=smoke, python_exe="python", now=lambda: 5.0,
                installed_fn={"accelerate": "1.12.0", "peft": "0.21.2"}.get,
                packages={"accelerate": "", "peft": "", "huggingface_hub": ""}, models=(),
                manifest=Manifest(date="d", packages={"accelerate": "1.15.0", "peft": "0.18.1", "huggingface_hub": "0.36.2"}),
                probe_env=lambda: ep.probe_environment(
                    {"accelerate": "1.15.0", "peft": "0.18.1", "huggingface_hub": "0.36.2"},
                    {"accelerate": "1.12.0", "peft": "0.21.2"}.get, lambda a: (1, ""), lambda n: None,
                    scan_other_pythons=False, packages={"accelerate": "", "peft": "", "huggingface_hub": ""}))
    rep, res = u.check_and_apply(lambda s, f, m: msgs.append(m))
    assert any("accelerate" in m and "1.12.0" in m for m in msgs)              # upgrade announced
    assert any("huggingface_hub" in m for m in msgs)                           # install announced
    specs = [a for a in pip_calls[0] if "==" in a]
    assert sorted(specs) == ["accelerate==1.15.0", "huggingface_hub==0.36.2"]  # peft 0.21.2 reused, not touched
    assert "--target" in pip_calls[0]                                          # only into Voxprint's own staging/overlay
    target = pip_calls[0][pip_calls[0].index("--target") + 1]
    assert str(paths.app_home()) in target and "--user" not in pip_calls[0]


def test_nothing_to_do_when_everything_is_current(tmp_path):
    called = []
    u = Updater(fetch_json=lambda url: {"releases": {"1.0": [{}]}}, pip_runner=lambda c: called.append(c) or (0, ""),
                python_exe="python", installed_fn={"a": "1.0"}.get, packages={"a": ""}, models=(),
                manifest=Manifest(date="d", packages={"a": "1.0"}))
    rep, res = u.check_and_apply()
    assert not rep.has_updates and called == []


# ------------------------------------------------------------------------------------- install manifest / health
def test_manifest_lifecycle_atomic_and_last(tmp_path):
    assert ist.verify_install(has_module=lambda m: True).codes == ["no_manifest"]
    assert ist.write_manifest("cu128")
    assert ist.verify_install(has_module=lambda m: True, current_torch_flavor="cu128").ok
    ist.begin_install()                                      # install starts: manifest disappears (retired)
    assert not ist.manifest_path().exists() and ist.manifest_path().with_suffix(".previous.json").exists()
    assert ist.verify_install(has_module=lambda m: True).codes == ["no_manifest"]
    assert not list(paths.app_home().glob("*.tmp"))          # atomic write leaves no temp files


def test_health_reason_codes():
    ist.write_manifest("cu128", req_root=tmp_req({"requirements.txt": "a"}))
    m = ist.read_manifest()
    m["app_version"], m["python"] = "0.0.1", "2.7.1"
    ist.manifest_path().write_text(json.dumps(m))
    rep = ist.verify_install(has_module=lambda mod: mod != "peft", current_torch_flavor="cpu",
                             req_root=tmp_req({"requirements.txt": "CHANGED"}))
    assert set(rep.codes) == {"app_version_changed", "python_changed", "requirements_changed",
                              "torch_flavor_changed", "package_missing"}
    assert not rep.ok and "peft" in " ".join(rep.reasons)
    ist.manifest_path().write_text("{broken")
    assert ist.verify_install(has_module=lambda m: True).codes == ["manifest_unreadable"]


_req_counter = [0]


def tmp_req(files):
    _req_counter[0] += 1
    d = paths.app_home() / f"req{_req_counter[0]}"
    d.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (d / n).write_text(t)
    return d


def test_interrupted_pip_leftovers_flagged():
    site = paths.packages_dir()
    (site / "~eta-1.0.dist-info").mkdir(parents=True)
    (site / "peft-0.18.1.dist-info").mkdir()
    (site / "peft-0.21.2.dist-info").mkdir()
    rep = ist.verify_install(has_module=lambda m: True, require_manifest=False)
    assert rep.codes == ["dist_info_inconsistent"]


def test_deep_check_reports_class_name_not_traceback():
    def importer(name):
        if name == "torch":
            raise OSError("DLL load failed ... very long traceback text")
    ist.write_manifest("cpu")
    rep = ist.verify_install(has_module=lambda m: True, deep=True, importer=importer, current_torch_flavor="cpu")
    assert rep.reasons == ["import_failed:torch/OSError"]


def test_health_text_in_all_languages_for_every_code():
    rep = ist.HealthReport()
    for c in ist.ALL_CODES:
        rep.add(c, "x")
    seen = set()
    for lang in i18n.LANGS:
        i18n.set_language(lang)
        lines = ist.describe_reasons(rep)
        assert len(lines) == len(ist.ALL_CODES) and all("health." not in l and "{" not in l for l in lines)
        seen.add(lines[0])
    assert len(seen) == len(i18n.LANGS)


def test_cli_verify_and_repair_messages():
    out = []
    assert ist.cli_verify(deep=False, print_fn=out.append) == 1
    assert any("--repair" in l for l in out)
    out.clear()
    assert ist.cli_repair(lambda c: (0, ""), lambda n: None, out.append) == 2        # no uv: clear message
    assert "uv" in out[0]


# ------------------------------------------------------------------------------------- uv venv plan / repair
def test_plan_uses_uv_with_explicit_python_and_torch_index(tmp_path):
    venv = tmp_path / "venv"
    cmds = ist.plan_commands("uv", "cu128", "3.12", tmp_path, venv)
    py = str(ist.venv_python(venv))
    assert cmds[0][:3] == ["uv", "venv", "--python"]
    assert all(c[:3] == ["uv", "pip", "install"] and c[3:5] == ["--python", py] for c in cmds[1:])
    assert "https://download.pytorch.org/whl/cu128" in cmds[1] and "torch" in cmds[1]
    assert cmds[3][5] == "--no-deps"


def test_run_install_writes_manifest_last_and_skips_unchanged_torch(tmp_path):
    venv = tmp_path / "venv"
    steps = []

    def run(cmd):
        steps.append(cmd)
        assert not ist.manifest_path().exists()             # never present while the install is running
        if cmd[1] == "venv":
            ist.venv_python(venv).parent.mkdir(parents=True)
            ist.venv_python(venv).write_text("")
        return 0, ""

    res = ist.run_install("uv", "cu128", run, "3.12", tmp_path, venv)
    assert res.ok and len(steps) == 4 and ist.manifest_path().exists()
    assert ist.read_manifest()["expected_torch_tag"] == "cu128" and (venv / ist.OWNER_MARKER).exists()
    steps.clear()
    res = ist.run_install("uv", "cu128", run, "3.12", tmp_path, venv, current_flavor="cu128")   # repair, same flavor
    assert res.ok and len(steps) == 2 and not any("torch" in c for c in steps)                   # torch not re-downloaded
    steps.clear()
    res = ist.run_install("uv", "cu130", run, "3.12", tmp_path, venv, current_flavor="cu128")   # driver updated
    assert res.ok and any("whl/cu130" in " ".join(c) for c in steps)


def test_failed_step_keeps_no_manifest_and_foreign_folder_is_protected(tmp_path):
    venv = tmp_path / "venv"
    ist.write_manifest("cpu")
    res = ist.run_install("uv", "cpu", lambda c: (1, "network down"), "3.12", tmp_path, venv)
    assert not res.ok and res.failed_step == 0 and not ist.manifest_path().exists()   # "ready" is never claimed
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine")
    called = []
    res = ist.run_install("uv", "cpu", lambda c: called.append(c) or (0, ""), "3.12", tmp_path, foreign)
    assert not res.ok and called == [] and (foreign / "keep.txt").read_text() == "mine"


# ------------------------------------------------------------------------------------- previous Voxprint installs
def test_previous_install_settings_and_models_are_reused(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    old = tmp_path / "oldvox"
    (old / "state").mkdir(parents=True)
    (old / "state" / "language").write_text("de")
    (old / "state" / "updater_state.json").write_text(json.dumps({"channel": "latest"}))
    (old / "state" / "secret.txt").write_text("not adopted")
    monkeypatch.setenv("VOXPRINT_PREVIOUS_HOMES", str(old))
    assert paths.previous_homes() == [old]
    before = sorted((p.name, p.read_text()) for p in old.rglob("*") if p.is_file())
    adopted = paths.adopt_previous_settings()
    assert sorted(adopted) == ["language", "updater_state.json"]
    assert (paths.state_dir() / "language").read_text() == "de" and not (paths.state_dir() / "secret.txt").exists()
    assert sorted((p.name, p.read_text()) for p in old.rglob("*") if p.is_file()) == before     # old install untouched
    (paths.state_dir() / "language").write_text("ru")
    paths.adopt_previous_settings()
    assert (paths.state_dir() / "language").read_text() == "ru"                                  # never overwrites
    # models of the old install are found by the locator
    monkeypatch.delenv("VOXPRINT_NO_EXTERNAL_MODELS")
    monkeypatch.setenv("HOME", str(home))
    from tests.test_model_locator import make_model
    from core import model_locator as ml
    make_model(old / "models" / "Org--Tiny-1.7B")
    f = ml.find_model("Org/Tiny-1.7B")
    assert f and f.kind == "voxprint" and f.path == old / "models" / "Org--Tiny-1.7B"


def _net_down(**k):
    raise OSError("network down")


# ------------------------------------------------------------------------------------- per-model state
def test_model_states_missing_partial_ready(tmp_path, monkeypatch):
    from infra import model_downloader as md
    from tests.test_model_locator import make_hf_cache, make_model, SHA_A
    monkeypatch.delenv("VOXPRINT_NO_EXTERNAL_MODELS")
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for k in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "PINOKIO_HOME", "MODELSCOPE_CACHE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    assert md.model_state("Org/A") == "missing"
    part = md.local_dir_for("Org/A").with_name(md.local_dir_for("Org/A").name + ".partial")
    part.mkdir(parents=True)
    (part / "model.safetensors.incomplete").write_bytes(b"x")
    assert md.model_state("Org/A") == "partial"
    msgs = []
    with pytest.raises(Exception):
        md.ensure_model("Org/A", lambda s, f, m: msgs.append(m), snapshot_download=_net_down, revision=SHA_A)
    assert i18n.tr("progress.model_resume", short="A") in msgs and part.exists()       # kept for resuming
    make_model(md.local_dir_for("Org/B"))
    assert md.model_state("Org/B") == "ready"
    hub = home / ".cache" / "huggingface" / "hub"
    make_hf_cache(hub, "Org/C", SHA_A)
    assert md.model_state("Org/C") == "ready"
    blobs = hub / "models--Org--D" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x")
    assert md.model_state("Org/D") == "partial"                                         # reported, never touched
    assert (blobs / "abc.incomplete").exists()
    assert md.model_states(["Org/A", "Org/B", "Org/C", "Org/D", "Org/E"]) == {
        "Org/A": "partial", "Org/B": "ready", "Org/C": "ready", "Org/D": "partial", "Org/E": "missing"}


# ------------------------------------------------------------------------------------- pinned asset installer
def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    return buf.getvalue()


class FakeOpener:
    def __init__(self, data: bytes, drop_first=None):
        self.data, self.requests, self.drop_first = data, [], drop_first

    def __call__(self, req, timeout):
        rng = req.headers.get("Range")
        self.requests.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = self.data[start:]
        if self.drop_first and not rng:
            body = body[:self.drop_first]
            r = _Resp(body, 200)
            r.drop = True
            return r
        return _Resp(body, 206 if rng else 200)


class _Resp:
    drop = False

    def __init__(self, b, status):
        self._b, self.status = io.BytesIO(b), status

    def read(self, n=-1):
        d = self._b.read(n)
        if not d and self.drop:
            raise OSError("reset")
        return d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def tool_spec(blob: bytes, **kw):
    spec = {"name": "ffmpeg", "version": "n1", "url": "https://example.invalid/ff.zip",
            "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob), "archive": "zip",
            "extract": {"top/bin/ffmpeg.exe": "ffmpeg.exe", "top/LICENSE.txt": "LICENSE.txt"},
            "exe": "ffmpeg.exe", "smoke": ["-version"], "smoke_expect": "ffmpeg version"}
    spec.update(kw)
    return spec


GOOD_RUN = lambda args: (0, "ffmpeg version n1 LGPL")


def test_bundled_ffmpeg_pin_is_complete():
    spec = assets.spec_for("ffmpeg", "win32")
    assert spec and len(spec["sha256"]) == 64 and spec["url"].startswith("https://github.com/BtbN/")
    assert "lgpl" in spec["url"] and spec["exe"] == "ffmpeg.exe" and spec["extract"]
    assert assets.spec_for("ffmpeg", "linux") is None


def test_asset_install_verifies_stages_smokes_and_swaps(tmp_path):
    blob = make_zip({"top/bin/ffmpeg.exe": b"EXE", "top/LICENSE.txt": b"lic", "top/evil": b"x"})
    spec = tool_spec(blob)
    exe = assets.install_asset(spec, tmp_path, FakeOpener(blob), GOOD_RUN)
    assert exe == tmp_path / "ffmpeg" / "ffmpeg.exe" and exe.read_bytes() == b"EXE"
    assert not (tmp_path / "ffmpeg" / "evil").exists()                      # only whitelisted members extracted
    assert (tmp_path / "ffmpeg" / assets.OWNER_MARKER).exists()
    assert assets.installed_path(spec, tmp_path) == exe
    op = FakeOpener(b"")
    assert assets.install_asset(spec, tmp_path, op, GOOD_RUN) == exe and op.requests == []   # already current: no download


def test_asset_sha_mismatch_is_rejected_and_old_copy_kept(tmp_path):
    blob = make_zip({"top/bin/ffmpeg.exe": b"OLD", "top/LICENSE.txt": b"l"})
    spec = tool_spec(blob)
    exe = assets.install_asset(spec, tmp_path, FakeOpener(blob), GOOD_RUN)
    evil = make_zip({"top/bin/ffmpeg.exe": b"EVIL", "top/LICENSE.txt": b"l"})
    spec2 = tool_spec(blob, sha256=hashlib.sha256(b"something else").hexdigest(), version="n2")
    with pytest.raises(assets.AssetError) as ei:
        assets.install_asset(spec2, tmp_path, FakeOpener(evil), GOOD_RUN)
    assert ei.value.code == "sha256_mismatch" and exe.read_bytes() == b"OLD"


def test_asset_failing_smoke_test_keeps_old_copy(tmp_path):
    old = make_zip({"top/bin/ffmpeg.exe": b"OLD", "top/LICENSE.txt": b"l"})
    exe = assets.install_asset(tool_spec(old), tmp_path, FakeOpener(old), GOOD_RUN)
    new = make_zip({"top/bin/ffmpeg.exe": b"NEW", "top/LICENSE.txt": b"l"})
    with pytest.raises(assets.AssetError) as ei:
        assets.install_asset(tool_spec(new, version="n2"), tmp_path, FakeOpener(new), lambda a: (1, "crash"))
    assert ei.value.code == "smoke_failed" and exe.read_bytes() == b"OLD"
    assert not (tmp_path / "ffmpeg.bak").exists()


def test_asset_upgrade_swaps_atomically_and_foreign_folder_is_protected(tmp_path):
    old = make_zip({"top/bin/ffmpeg.exe": b"OLD", "top/LICENSE.txt": b"l"})
    assets.install_asset(tool_spec(old), tmp_path, FakeOpener(old), GOOD_RUN)
    new = make_zip({"top/bin/ffmpeg.exe": b"NEW", "top/LICENSE.txt": b"l"})
    exe = assets.install_asset(tool_spec(new, version="n2"), tmp_path, FakeOpener(new), GOOD_RUN)
    assert exe.read_bytes() == b"NEW" and not (tmp_path / "ffmpeg.bak").exists()
    foreign = tmp_path / "root2"
    (foreign / "ffmpeg").mkdir(parents=True)
    (foreign / "ffmpeg" / "mine.txt").write_text("x")
    with pytest.raises(assets.AssetError) as ei:
        assets.install_asset(tool_spec(new), foreign, FakeOpener(new), GOOD_RUN)
    assert ei.value.code == "swap_failed" and (foreign / "ffmpeg" / "mine.txt").exists()


def test_asset_download_resumes(tmp_path):
    blob = make_zip({"top/bin/ffmpeg.exe": b"E" * 5000, "top/LICENSE.txt": b"l"})
    spec = tool_spec(blob)
    op = FakeOpener(blob, drop_first=100)
    with pytest.raises(assets.AssetError) as ei:
        assets.install_asset(spec, tmp_path, op, GOOD_RUN)
    assert ei.value.code == "download_failed"
    assert (tmp_path / ".staging" / "ffmpeg" / "archive.part").stat().st_size == 100       # kept
    exe = assets.install_asset(spec, tmp_path, op, GOOD_RUN)
    assert op.requests[-1] == "bytes=100-" and exe.exists()


def test_bad_archive_member_missing(tmp_path):
    blob = make_zip({"top/LICENSE.txt": b"l"})
    with pytest.raises(assets.AssetError) as ei:
        assets.install_asset(tool_spec(blob), tmp_path, FakeOpener(blob), GOOD_RUN)
    assert ei.value.code == "bad_archive"


def test_ensure_ffmpeg_tool_prefers_working_system_ffmpeg_then_pinned(tmp_path):
    msgs = []
    sysrun = lambda a: (0, "ffmpeg version 7.0 x")
    p = assets.ensure_ffmpeg_tool(lambda f, m: msgs.append(m), FakeOpener(b""), sysrun, lambda n: "/usr/bin/ffmpeg",
                                  "win32", tmp_path)
    assert p == Path("/usr/bin/ffmpeg") and i18n.tr("env.ffmpeg_reused", version="7.0") in msgs
    # system ffmpeg broken -> the pinned build is downloaded; patch the manifest spec for the fake archive
    blob = make_zip({"top/bin/ffmpeg.exe": b"E", "top/LICENSE.txt": b"l"})
    spec = tool_spec(blob)
    orig = assets.spec_for
    assets.spec_for = lambda name, platform=None, manifest=None: spec
    try:
        def run(a):
            return (1, "broken") if a[0] == "/usr/bin/ffmpeg" else (0, "ffmpeg version n1")
        p = assets.ensure_ffmpeg_tool(lambda f, m: msgs.append(m), FakeOpener(blob), run, lambda n: "/usr/bin/ffmpeg",
                                      "win32", tmp_path)
        assert p == tmp_path / "ffmpeg" / "ffmpeg.exe"
        assert i18n.tr("progress.ffmpeg_ready") in msgs
        # a failure never raises: the bundled imageio-ffmpeg stays the fallback
        spec2 = tool_spec(blob, sha256="0" * 64)
        assets.spec_for = lambda name, platform=None, manifest=None: spec2
        assert assets.ensure_ffmpeg_tool(None, FakeOpener(blob), run, lambda n: None, "win32", tmp_path / "other") is None
    finally:
        assets.spec_for = orig


def test_ensure_ffmpeg_in_audio_utils_requires_smoke_test(monkeypatch, tmp_path):
    from core import audio_utils
    from infra import env_probe
    monkeypatch.setattr(env_probe, "probe_ffmpeg", lambda *a, **k: env_probe.FfmpegInfo("/broken/ffmpeg", "", False))
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "x.py")])
    got = audio_utils.ensure_ffmpeg()
    assert got != "/broken/ffmpeg"                                   # falls through to the next source
    monkeypatch.setattr(env_probe, "probe_ffmpeg", lambda *a, **k: env_probe.FfmpegInfo("/good/ffmpeg", "7", True))
    assert audio_utils.ensure_ffmpeg() == "/good/ffmpeg"


def test_default_venv_python_matches_supported_runtime():
    # --verify-install compares the manifest's Python version with the running interpreter (3.11): the default venv is 3.11 too
    assert ist.PYTHON_VERSION_DEFAULT == "3.11"
