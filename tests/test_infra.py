"""Infrastructure without network: version logic, the verified manifest, model downloader, updater, VRAM planning."""
import json
from pathlib import Path

import pytest

from core.errors import ModelDownloadError
from infra import model_downloader as md
from infra import paths
from infra.verified_manifest import Manifest, load_bundled, load_manifest, parse_manifest, requirements_lines
from infra.version_manager import (CHANNEL_LATEST, PackageStatus, TRACKED_MODELS, TRACKED_PACKAGES, check_versions,
                                   compare_versions, is_newer, latest_compatible_pypi)
from infra.vram_optimizer import (MODEL_0_6B, MODEL_1_7B, GpuInfo, compute_epochs, compute_lr, plan_training,
                                  reduce_after_oom)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


# ---------------- versions
def test_compare_versions():
    assert compare_versions("1.2.0", "1.10.0") == -1
    assert compare_versions("2.0", "2.0.0") == 0
    assert compare_versions("4.57.6", "4.57.3") == 1
    assert compare_versions("1.0rc1", "1.0") == -1
    assert is_newer("0.0.7", "0.0.6") and not is_newer("0.0.6", "0.0.6") and is_newer("1", None)


def _pypi(releases):
    return {"releases": releases}


def test_latest_compatible_respects_constraint_prerelease_yanked():
    f = lambda url: _pypi({
        "4.57.6": [{}], "4.58.0": [{}], "5.0.0": [{}], "5.1.0rc1": [{}],
        "3.0.0": [{}], "4.99.0": [{"yanked": True}], "4.60.0": []})
    assert latest_compatible_pypi("transformers", ">=4.57.6,<5", f) == "4.58.0"
    assert latest_compatible_pypi("transformers", "", f) == "5.0.0"
    assert latest_compatible_pypi("x", ">=9", f) is None


def test_check_versions_report_latest_channel():
    def fetch(url):
        if "pypi.org" in url:
            return _pypi({"0.0.6": [{}], "0.0.7": [{}]})
        return {"sha": "NEW", "lastModified": "2026"}
    inst = {"qwen-asr": "0.0.6", "peft": "0.0.7"}.get
    rep = check_versions({"M/a": "OLD", "M/b": "NEW"}, fetch, inst, packages={"qwen-asr": "", "peft": "", "zzz": ""},
                         models=("M/a", "M/b", "M/c"), channel=CHANNEL_LATEST)
    names = {p.name: p.update_available for p in rep.packages}
    assert names == {"qwen-asr": True, "peft": False, "zzz": True}  # zzz is not installed -> can be installed
    mod = {m.repo_id: m.update_available for m in rep.models}
    assert mod == {"M/a": True, "M/b": False, "M/c": False}
    assert rep.has_updates and rep.network_ok and rep.channel == "latest"


def test_verified_channel_uses_manifest_pins_and_only_reports_newer():
    def fetch(url):
        if "pypi.org" in url:
            return _pypi({"0.0.6": [{}], "0.0.7": [{}], "0.0.8": [{}]})
        return {"sha": "HEAD"}
    man = Manifest(date="2026-10-02", packages={"qwen-asr": "0.0.7", "peft": "0.0.6"}, models={"M/a": "P" * 40})
    inst = {"qwen-asr": "0.0.6", "peft": "0.0.8", "other": "0.0.6", "bitsandbytes": None}.get
    rep = check_versions({"M/a": "OLD"}, fetch, inst,
                         packages={"qwen-asr": "", "peft": "", "other": "", "bitsandbytes": ""},
                         models=("M/a",), manifest=man)
    st = {p.name: p for p in rep.packages}
    assert st["qwen-asr"].target == "0.0.7" and st["qwen-asr"].update_available and st["qwen-asr"].pinned
    assert st["peft"].target == "0.0.6" and st["peft"].update_available       # roll back down to the verified version
    assert st["other"].target is None and not st["other"].update_available     # no pin - leave it alone...
    assert st["other"].newer_unverified                                        # ...but report 0.0.8
    assert st["bitsandbytes"].target is None                                   # optional and not installed
    assert rep.models[0].target_sha == "P" * 40 and rep.models[0].update_available
    assert rep.models[0].newer_unverified                                      # HEAD != verified revision
    assert "qwen-asr 0.0.8" in rep.unverified_newer and rep.manifest_date == "2026-10-02"


def test_check_versions_offline():
    def fetch(url):
        raise OSError("offline")
    rep = check_versions({}, fetch, lambda n: "1", packages={"a": ""}, models=("M/a",))
    assert not rep.network_ok and not rep.has_updates


# ---------------- VRAM
def test_plan_16gb_follows_alexandria_recipe():
    p = plan_training(GpuInfo(True, "RTX 4090 Laptop", 16.0, 15.0), n_items=61)
    assert p.device == "cuda:0" and p.base_model == MODEL_1_7B
    assert p.batch_size == 1 and p.grad_accum in (4, 5, 6, 7, 8) and p.gradient_checkpointing
    assert p.dtype == "bfloat16" and p.lora_r == 32 and p.lora_alpha == 128
    assert p.language == "russian" and p.attn_implementation == "eager"
    assert p.lr == 1e-6                      # < 90 examples
    assert 250 <= 61 * p.epochs <= 400       # the lora.md rule


def test_hyperparameters_auto_rules():
    assert compute_lr(61) == 1e-6 and compute_lr(121) == 2e-6
    assert compute_epochs(61) == 5 and compute_epochs(121) == 3 and compute_epochs(10) == 15 and compute_epochs(1000) == 2
    for n in (40, 61, 80, 100, 121, 160):
        p = plan_training(GpuInfo(True, "x", 16.0), n)
        assert 1e-6 <= p.lr <= 2e-6 and 4 <= p.grad_accum <= 8
        if 20 <= n <= 160:
            assert 250 <= n * p.epochs <= 450, (n, p.epochs)


def test_plan_variants_and_cpu_fallback():
    assert plan_training(GpuInfo(True, "x", 24.0), 100).batch_size == 1
    p12 = plan_training(GpuInfo(True, "x", 12.0), 100)
    assert p12.base_model == MODEL_1_7B and p12.use_8bit_adam and p12.warnings
    p8 = plan_training(GpuInfo(True, "x", 8.0), 100)
    assert p8.base_model == MODEL_0_6B and p8.warnings
    for plan in (plan_training(GpuInfo(False), 100), plan_training(GpuInfo(True, "x", 16.0), 100, force_cpu=True),
                 plan_training(GpuInfo(True, "x", 3.0), 100)):
        assert plan.device == "cpu" and plan.dtype == "float32" and not plan.use_8bit_adam and plan.warnings
    assert any("мало" in w for w in plan_training(GpuInfo(True, "x", 16.0), 10).warnings)


def test_reduce_after_oom_chain():
    p = plan_training(GpuInfo(True, "x", 16.0), 100)
    assert not p.use_8bit_adam
    p2 = reduce_after_oom(p)
    assert p2.use_8bit_adam and p2.base_model == MODEL_1_7B and p2.batch_size == 1
    p3 = reduce_after_oom(p2)
    assert p3.base_model == MODEL_0_6B
    assert reduce_after_oom(p3) is None
    assert reduce_after_oom(plan_training(GpuInfo(False), 10)) is None


# ---------------- the "verified by Voxprint" manifest
def test_bundled_manifest_is_valid_and_consistent():
    m = load_bundled()
    assert m.packages["transformers"].startswith("4.57") and m.packages["peft"] == "0.18.1"
    assert set(m.packages) <= set(TRACKED_PACKAGES) and set(m.models) == set(TRACKED_MODELS)
    # requirements-verified.txt and requirements-nodeps.txt are derived from the manifest
    root = Path(__file__).resolve().parent.parent
    ver = {ln.strip() for ln in (root / "requirements-verified.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")}
    assert ver == set(requirements_lines(m))
    nod = {ln.strip() for ln in (root / "requirements-nodeps.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")}
    assert nod == set(requirements_lines(m, nodeps=True))


def test_parse_manifest_rejects_unknown_and_malformed():
    ok = {"schema": 1, "packages": {"peft": "0.18.1"}, "models": {"Qwen/Qwen3-ForcedAligner-0.6B": "a" * 40}}
    assert parse_manifest(ok, list(TRACKED_PACKAGES), list(TRACKED_MODELS)).packages == {"peft": "0.18.1"}
    for bad in ({**ok, "schema": 2}, {**ok, "packages": {"evil-pkg": "1.0"}}, {**ok, "packages": {"peft": "x.y"}},
                {**ok, "models": {"Qwen/Qwen3-ForcedAligner-0.6B": "main"}}, {**ok, "models": {"Evil/Repo": "a" * 40}}, 5):
        with pytest.raises(ValueError):
            parse_manifest(bad, list(TRACKED_PACKAGES), list(TRACKED_MODELS))


def test_remote_manifest_used_only_if_valid_and_not_older():
    newer = {"schema": 1, "date": "2027-01-01", "packages": {"peft": "0.19.0"}, "models": {}}
    m = load_manifest(lambda u: newer, url="https://example.invalid/m.json")
    assert m.source == "remote" and m.packages == {"peft": "0.19.0"}
    # the result is cached and used offline
    off = load_manifest(lambda u: (_ for _ in ()).throw(OSError("offline")), url="https://example.invalid/m.json")
    assert off.source == "cache" and off.packages == {"peft": "0.19.0"}
    # older than the bundled one / corrupt - ignored
    older = {"schema": 1, "date": "2020-01-01", "packages": {"peft": "0.1.0"}, "models": {}}
    paths_cache = paths.state_dir() / "verified_manifest.json"
    paths_cache.unlink()
    assert load_manifest(lambda u: older, url="https://example.invalid/m.json").source == "bundled"
    paths_cache.unlink(missing_ok=True)
    assert load_manifest(lambda u: {"schema": 1, "packages": {"evil": "1"}}, url="https://x").source == "bundled"


# ---------------- model downloader
def _fake_snapshot(files=("config.json", "model.safetensors"), fail=False):
    def sd(repo_id, local_dir, tqdm_class=None, **kw):
        if fail:
            raise ConnectionError("no internet")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for f in files:
            (Path(local_dir) / f).write_bytes(b"{}")
        if tqdm_class is not None:
            bar = tqdm_class(total=100, unit="B")
            bar.update(50)
            bar.update(50)
        return local_dir
    return sd


def test_ensure_model_downloads_then_reuses(tmp_path):
    calls = []
    prog = lambda s, f, m: calls.append((f, m))
    p = md.ensure_model("Org/Model", prog, root=tmp_path, snapshot_download=_fake_snapshot(),
                        get_remote_sha=lambda r: "abc123")
    assert md.verify_local_model(p) and md.local_revision("Org/Model", tmp_path) == "abc123"
    assert not (tmp_path / "Org--Model.partial").exists()
    assert any(f == 1.0 for f, _ in calls) and any("Первый запуск" in m for _, m in calls)
    # a second time - without touching the network
    p2 = md.ensure_model("Org/Model", prog, root=tmp_path, snapshot_download=_fake_snapshot(fail=True))
    assert p2 == p


def test_download_failure_gives_hf_link(tmp_path):
    with pytest.raises(ModelDownloadError) as ei:
        md.ensure_model("Org/Model", root=tmp_path, snapshot_download=_fake_snapshot(fail=True),
                        get_remote_sha=lambda r: None)
    assert ei.value.url == "https://huggingface.co/Org/Model"
    assert "https://huggingface.co/Org/Model" in ei.value.user_message
    assert not list(tmp_path.glob("*.partial"))


def test_incomplete_download_rejected(tmp_path):
    with pytest.raises(ModelDownloadError):
        md.ensure_model("Org/Model", root=tmp_path, snapshot_download=_fake_snapshot(files=("config.json",)),
                        get_remote_sha=lambda r: None)


# ---------------- updater
def _updater(tmp_path, pip_ok=True, smoke_ok=True, now=1_000_000.0, **kw):
    from infra.updater import Updater
    log = {"pip": [], "smoke": []}

    def pip(cmd):
        log["pip"].append(cmd)
        if pip_ok:
            site = Path(cmd[cmd.index("--target") + 1])
            (site / "qwen_asr").mkdir(parents=True, exist_ok=True)
            (site / "qwen_asr" / "NEW").write_text("new")
        return (0 if pip_ok else 1), "pip output"

    def smoke(cmd):
        log["smoke"].append(cmd)
        return (0, 'SMOKE_OK {"qwen-asr": "0.0.7"}') if smoke_ok else (1, "ImportError")

    def fetch(url):
        if "pypi.org" in url:
            return {"releases": {"0.0.6": [{}], "0.0.7": [{}]}}
        return {"sha": "NEWSHA"}

    u = Updater(fetch_json=fetch, pip_runner=pip, smoke_runner=smoke, python_exe="python", now=lambda: now,
                installed_fn={"qwen-asr": "0.0.6"}.get, packages={"qwen-asr": ""}, models=(),
                manifest=Manifest(date="2026-10-02", packages={"qwen-asr": "0.0.7"}), **kw)
    return u, log


def test_autocheck_interval(tmp_path):
    u, _ = _updater(tmp_path, now=1_000_000.0)
    assert u.should_autocheck()
    u.check()
    assert not u.should_autocheck()
    u.now = lambda: 1_000_000.0 + 6 * 86400
    assert not u.should_autocheck()
    u.now = lambda: 1_000_000.0 + 7 * 86400 + 1
    assert u.should_autocheck()


def test_update_success_swaps_and_reports(tmp_path):
    (paths.packages_dir() / "old_pkg").mkdir(parents=True)
    (paths.packages_dir() / "old_pkg" / "f").write_text("1")
    u, log = _updater(tmp_path)
    rep, res = u.check_and_apply()
    assert res.after == {"qwen-asr": "0.0.7"} and res.before == {"qwen-asr": "0.0.6"}
    assert res.needs_restart and not res.rolled_back
    assert (paths.packages_dir() / "qwen_asr" / "NEW").exists()
    assert (paths.packages_dir() / "old_pkg" / "f").exists()  # earlier updates are preserved
    assert list(paths.app_home().glob("packages.bak-*"))
    assert "0.0.6 → 0.0.7" in res.summary()
    assert "--no-deps" in log["pip"][0] and "qwen-asr==0.0.7" in log["pip"][0]
    assert not list(paths.staging_dir().glob("pkgs-*"))
    text = (paths.logs_dir() / "updater.log").read_text(encoding="utf-8")
    assert "swap done" in text
    assert u.rollback() and not (paths.packages_dir() / "qwen_asr").exists()


def test_verified_channel_ignores_newer_pypi_but_reports_it_and_restore_downgrades(tmp_path):
    u, log = _updater(tmp_path)
    # PyPI knows 0.0.7, 0.0.7 is verified; 0.0.9 is installed -> restore_verified rolls back to the verified one
    u.installed_fn = {"qwen-asr": "0.0.9"}.get
    u.fetch_json = lambda url: {"releases": {"0.0.7": [{}], "0.0.9": [{}]}} if "pypi" in url else {"sha": "x"}
    res = u.restore_verified()
    assert "qwen-asr==0.0.7" in log["pip"][0] and res.after == {"qwen-asr": "0.0.7"}
    # while a normal check in the verified channel without a pin leaves the package alone
    u2, log2 = _updater(tmp_path)
    u2._manifest = Manifest(date="2026-10-02", packages={})
    u2.fetch_json = lambda url: {"releases": {"0.0.6": [{}], "0.0.9": [{}]}} if "pypi" in url else {"sha": "x"}
    rep, res2 = u2.check_and_apply()
    assert not log2["pip"] and res2.unverified_newer == ["qwen-asr 0.0.9"] and not rep.has_updates


def test_channel_latest_via_env(monkeypatch, tmp_path):
    u, _ = _updater(tmp_path)
    assert u.channel == "verified"
    monkeypatch.setenv("VOXPRINT_CHANNEL", "latest")
    assert u.channel == "latest"


def test_pinned_model_revision_is_requested(tmp_path):
    seen = {}

    def sd(repo_id, local_dir, tqdm_class=None, **kw):
        seen.update(kw)
        return _fake_snapshot()(repo_id, local_dir, tqdm_class)
    md.ensure_model("Org/M2", root=tmp_path, snapshot_download=sd, revision="b" * 40)
    assert seen["revision"] == "b" * 40 and md.local_revision("Org/M2", tmp_path) == "b" * 40
    # the pinned revision is unavailable -> one retry without a revision
    calls = []

    def sd2(repo_id, local_dir, tqdm_class=None, **kw):
        calls.append(kw.get("revision"))
        if kw.get("revision"):
            raise ConnectionError("revision gone")
        return _fake_snapshot()(repo_id, local_dir, tqdm_class)
    md.ensure_model("Org/M3", root=tmp_path, snapshot_download=sd2, revision="c" * 40, get_remote_sha=lambda r: "HEADSHA")
    assert calls == ["c" * 40, None] and md.verify_local_model(md.local_dir_for("Org/M3", tmp_path))


def test_update_rolls_back_on_failed_compat(tmp_path):
    (paths.packages_dir() / "old_pkg").mkdir(parents=True)
    u, _ = _updater(tmp_path, smoke_ok=False)
    _, res = u.check_and_apply()
    assert res.rolled_back and not res.needs_restart and res.after == {}
    assert (paths.packages_dir() / "old_pkg").exists() and not (paths.packages_dir() / "qwen_asr").exists()
    assert not list(paths.staging_dir().glob("pkgs-*"))
    assert "прежняя" in res.summary() or "прежн" in res.summary()


def test_update_pip_failure(tmp_path):
    u, _ = _updater(tmp_path, pip_ok=False)
    _, res = u.check_and_apply()
    assert res.rolled_back and not paths.packages_dir().exists()


def test_model_update_swap(tmp_path):
    from infra.updater import Updater
    md.ensure_model("Org/M", snapshot_download=_fake_snapshot(), get_remote_sha=lambda r: "OLD")
    u = Updater(fetch_json=lambda url: {"sha": "NEW"} if "huggingface" in url else {"releases": {}},
                pip_runner=lambda c: (0, ""), python_exe="python", installed_fn=lambda n: None,
                snapshot_download=_fake_snapshot(files=("config.json", "model.safetensors", "extra.json")),
                get_remote_sha=lambda r: "NEW", packages={}, models=("Org/M",))
    rep, res = u.check_and_apply()
    assert res.models_updated == ["Org/M"]
    assert md.local_revision("Org/M") == "NEW" and (md.local_dir_for("Org/M") / "extra.json").exists()


def test_model_update_failure_keeps_old(tmp_path):
    from infra.updater import Updater
    md.ensure_model("Org/M", snapshot_download=_fake_snapshot(), get_remote_sha=lambda r: "OLD")
    u = Updater(fetch_json=lambda url: {"sha": "NEW"} if "huggingface" in url else {"releases": {}},
                python_exe="python", installed_fn=lambda n: None, snapshot_download=_fake_snapshot(fail=True),
                get_remote_sha=lambda r: "NEW", packages={}, models=("Org/M",))
    _, res = u.check_and_apply()
    assert res.models_updated == [] and md.local_revision("Org/M") == "OLD"
    assert md.verify_local_model(md.local_dir_for("Org/M"))


def test_activate_overlay(tmp_path):
    import sys
    from infra.updater import activate_overlay
    assert activate_overlay() is None
    (paths.packages_dir()).mkdir(parents=True)
    (paths.packages_dir() / "x").mkdir()
    p = activate_overlay()
    try:
        assert sys.path[0] == str(p)
    finally:
        sys.path.remove(str(p))


def test_offline_check_and_apply(tmp_path):
    from infra.updater import Updater

    def fetch(url):
        raise OSError("offline")
    u = Updater(fetch_json=fetch, python_exe="python", installed_fn=lambda n: "1", packages={"a": ""}, models=())
    rep, res = u.check_and_apply()
    assert not rep.network_ok and "интернет" in res.summary()
    assert u.should_autocheck()  # a failed check does not reset the timer


def test_net_urlopen_retries_with_certifi_on_cert_error(monkeypatch):
    import ssl
    import urllib.error
    from infra import net, netroute

    calls = []

    class Op:
        def __init__(self, ctx):
            self.ctx = ctx

        def open(self, req, timeout=None):
            calls.append(self.ctx)
            if self.ctx is None:
                raise urllib.error.URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))
            return "ok"

    monkeypatch.setattr(netroute, "_opener", lambda route, ctx, ct: Op(ctx))
    monkeypatch.setattr(netroute, "candidates", lambda host: [netroute.DEFAULT])
    assert net.urlopen("https://example.org", 3) == "ok"
    assert calls[0] is None and calls[1] is not None             # second attempt carries the certifi context

    class Down:
        def open(self, req, timeout=None):
            raise urllib.error.URLError(OSError("network down"))

    monkeypatch.setattr(netroute, "_opener", lambda route, ctx, ct: Down())
    import pytest
    with pytest.raises(urllib.error.URLError):
        net.urlopen("https://example.org", 3)
