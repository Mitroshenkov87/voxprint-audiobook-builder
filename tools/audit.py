"""One-shot code audit: ruff, mypy (basic), bandit, pip-audit, vulture (high confidence), radon (rank D or worse) and
an optional coverage run, printed as ONE short summary; the full output goes to ``audit-report.txt``.

    python -m pip install -r requirements-audit.txt          # the tools (any venv)
    python tools/audit.py --python <app venv python> [--coverage]

``--python``: the interpreter with the app's dependencies (its installed packages are checked by pip-audit, and the test
suite runs under coverage with it); default: the interpreter running this script.  Exit code 0, or 1 with ``--strict``
when bandit reports a high-severity issue or pip-audit a vulnerable package.  Each tool that is not installed is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ["core", "infra", "ui", "workers", "tools", "cli.py", "main.py"]
REPORT = ROOT / "audit-report.txt"


def _run(cmd, env=None, timeout=3600):
    env = dict(env or os.environ, NO_COLOR="1")
    env.pop("FORCE_COLOR", None)
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


def _missing(err: str) -> bool:
    return "No module named" in err


def ruff(py):
    # real bugs only: pyflakes (F), syntax (E9) and bugbear (B); no style rules
    rc, out, err = _run([py, "-m", "ruff", "check", "--select", "F,E9,B", "--ignore", "B008,B904,B905",
                         "--output-format", "concise", "--no-cache", *SRC])
    if _missing(err):
        return None, err
    lines = [x for x in out.splitlines() if x.count(":") >= 3]
    return len(lines), out


def mypy(py):
    rc, out, err = _run([py, "-m", "mypy", "--ignore-missing-imports", "--follow-imports=silent", "--no-error-summary",
                         "--hide-error-context", "--no-color-output", "--cache-dir", os.path.join(tempfile.gettempdir(), "vx-mypy"),
                         *SRC])
    if _missing(err):
        return None, err
    lines = [x for x in out.splitlines() if ": error:" in x]
    return len(lines), out


def bandit(py):
    rc, out, err = _run([py, "-m", "bandit", "-r", *SRC, "-q", "-f", "json", "-ll", "-ii", "-x", "tests"])
    if _missing(err):
        return None, err
    try:
        res = json.loads(out).get("results", [])
    except ValueError:
        return None, out + err
    high = sum(1 for r in res if r.get("issue_severity") == "HIGH")
    text = "\n".join(f"{r['filename']}:{r['line_number']}: {r['issue_severity']}/{r['issue_confidence']} {r['test_id']} "
                     f"{r['issue_text']}" for r in res)
    return (len(res), high), text


def pip_audit(py, app_py):
    # the app's installed distributions (no pip needed in the app venv), audited without resolving again
    rc, frozen, err = _run([app_py, "-c", "import importlib.metadata as m\n"
                            "seen=set()\n"
                            "for d in m.distributions():\n"
                            "  n=(d.metadata['Name'] or '').lower()\n"
                            "  if n and n not in seen: seen.add(n); print(f'{n}=={d.version}')"])
    if rc:
        return None, err
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(frozen)
    try:
        rc, out, err = _run([py, "-m", "pip_audit", "-r", f.name, "--no-deps", "--disable-pip", "-f", "json",
                             "--progress-spinner", "off"])
    finally:
        os.unlink(f.name)
    if _missing(err):
        return None, err
    try:
        deps = json.loads(out).get("dependencies", [])
    except ValueError:
        return None, out + err
    vul = [d for d in deps if d.get("vulns")]
    text = "\n".join(f"{d['name']}=={d['version']}: " + ", ".join(
        f"{v['id']} (fix: {','.join(v.get('fix_versions') or []) or '-'})" for v in d["vulns"]) for d in vul)
    return len(vul), text


def vulture(py):
    rc, out, err = _run([py, "-m", "vulture", *SRC, "--min-confidence", "90"])
    if _missing(err):
        return None, err
    lines = [x for x in out.splitlines() if "% confidence" in x]
    return len(lines), out


def radon(py):
    rc, out, err = _run([py, "-m", "radon", "cc", "-n", "D", "-s", "-j", *SRC])
    if _missing(err):
        return None, err
    try:
        data = json.loads(out or "{}")
    except ValueError:
        return None, out + err
    items = [f"{f}:{b['lineno']} {b['name']} {b['rank']} ({b['complexity']})" for f, blocks in data.items()
             if isinstance(blocks, list) for b in blocks]
    return len(items), "\n".join(items)


def coverage(app_py, run_tests):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    if run_tests:
        rc, out, err = _run([app_py, "-m", "coverage", "run", "--source=core,infra,ui,workers,tools", "-m", "pytest", "-q",
                             "-p", "no:cacheprovider"], env=env, timeout=7200)
        if _missing(err):
            return None, err
        tests = (out.strip().splitlines() or [""])[-1]
    elif not (ROOT / ".coverage").is_file():
        return None, "no .coverage data (run with --coverage)"
    else:
        tests = ""
    rc, total, err = _run([app_py, "-m", "coverage", "report", "--format=total"], env=env)
    rc2, full, _ = _run([app_py, "-m", "coverage", "report", "--skip-covered", "--sort=cover"], env=env)
    return (total.strip() or "?"), f"{tests}\n{full}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default=sys.executable, help="interpreter with the app's dependencies")
    ap.add_argument("--coverage", action="store_true", help="run the test suite under coverage (slow)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on a high-severity bandit issue or a vulnerable package")
    a = ap.parse_args(argv)
    py = sys.executable
    results = {
        "ruff (F,E9,B)": ruff(py), "mypy (basic)": mypy(py), "bandit (medium+)": bandit(py),
        "pip-audit": pip_audit(py, a.python), "vulture (>=90%)": vulture(py), "radon (rank D+)": radon(py),
        "coverage": coverage(a.python, a.coverage),
    }
    summary, report = [], []
    for name, (n, text) in results.items():
        if n is None:
            shown = "skipped (" + (text.strip().splitlines() or ["not installed"])[-1][:80] + ")"
        elif name.startswith("bandit"):
            shown = f"{n[0]} issue(s), {n[1]} high"
        elif name == "coverage":
            shown = f"{n}% total"
        else:
            shown = f"{n} finding(s)"
        summary.append(f"  {name:18s} {shown}")
        report.append(f"===== {name}: {shown}\n{text.strip()}\n")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("Voxprint audit\n" + "\n".join(summary) + f"\n  details: {REPORT.name}")
    b, pa = results["bandit (medium+)"][0], results["pip-audit"][0]
    bad = (b is not None and b[1] > 0) or (pa is not None and pa > 0)
    return 1 if a.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
