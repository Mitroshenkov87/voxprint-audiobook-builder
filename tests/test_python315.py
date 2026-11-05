"""Suite rule 2026-11-05: this repo requires Python 3.15 and bundles 3.15.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requires_python_is_at_least_3_15():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.15"' in text


def test_ci_uses_setup_python_3_15_only():
    versions = []
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "python-version:" in line:
                versions.append((path.name, line.strip()))
    assert versions, "expected setup-python pins"
    assert {line.split(":", 1)[1].strip().strip('"') for _, line in versions} == {"3.15"}


def test_installer_bundles_python_3_15_2_and_drops_russian():
    from infra import install_state

    assert install_state.PYTHON_VERSION_DEFAULT == "3.15.2"
    linux = (ROOT / "installer" / "linux" / "install-voxprint-linux.sh").read_text(encoding="utf-8")
    assert "--python 3.15.2" in linux and "Python 3.14 is required" not in linux
    assert "torch==2.14.1" in linux and "torchaudio==" not in linux
    iss = (ROOT / "installer" / "Voxprint.iss").read_text(encoding="utf-8-sig")
    assert 'Name: "russian"' not in iss and "Russian.isl" not in iss
    assert 'Name: "english"' in iss and 'Name: "german"' in iss
    thin = (ROOT / "build_thin.bat").read_text(encoding="utf-8")
    assert thin.count("--collect-all cryptography") == 2 and thin.count('--add-binary "%PY3DLL%;."') == 2
    assert '_internal\\python3.dll' in thin and "bindings\\python3.dll" not in thin


def test_lock_records_the_packages_with_no_3_15_wheel():
    import json

    lock = json.loads((ROOT / "infra" / "runtime_lock.json").read_text(encoding="utf-8"))
    assert lock["python"] == "3.15" and lock["python_full"] == "3.15.2"
    assert lock["torch_version"] == "2.14.1"
    blocked = {row["dist"]: row["newest"] for row in lock["blocked"]}
    assert blocked["torchaudio"] == "2.11.1"
    assert blocked["onnxruntime"] == "1.31.1"
    assert "torchaudio" not in {row["dist"] for row in lock["wheels"]}
    assert "onnxruntime" not in {row["dist"] for row in lock["wheels"]}
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "voxprint_common-0.6.4-py3-none-any.whl" in req
    assert 'onnxruntime; python_version < "3.15"' in req
