# Code audit

Automated pass used by pull requests, pushes to `main`, a weekly schedule, and the installer workflow.
The script is `tools/audit.py`. It writes one summary to the GitHub job summary and the full text to
`audit-report.txt` (uploaded as the `audit-report` artifact).

## What runs

| Tool | Scope | Release gate |
| --- | --- | --- |
| ruff | Pyflakes (`F`, including undefined names) and syntax (`E9`). Line length (`E501`) and pyupgrade (`UP`) are not selected. | Fails on any error |
| mypy | Non-strict (`mypy.ini`): `core`, `infra`, `ui`, `workers`, `tools`, `cli.py`, `main.py`. Tests are not part of the gate. Untyped functions are allowed. No global `ignore_errors`. | Fails on any error |
| bandit | Medium and high severity, medium and high confidence. | Fails on high only |
| pip-audit | Packages installed from `requirements.txt`, `requirements-verified.txt`, `requirements-dev.txt`, `requirements-nodeps.txt`, and CPU PyTorch. A local version tag such as `+cpu` is removed so PyPI can see the release. | Fails on high or critical that are not in `tools/audit-allowlist.toml`. An advisory with no readable severity counts as high. |
| gitleaks | Secrets. Full git history on the weekly schedule, on a manual run, and when the installer workflow calls the audit. The commits of a push or pull request only, otherwise. | Fails on any finding |
| vulture | Dead code, confidence 80 and above. | Report only |
| radon | Cyclomatic complexity, rank D and worse. | Report only |

When an advisory has a GitHub id (`GHSA`), that reviewed severity is the one that counts.
`MODERATE` is medium and does not block. Another database's label is used only when there is no GHSA.
When there is no reviewed label at all, the CVSS v3 base score is used (high is 7.0 and above, critical is 9.0 and above).

## How to run it locally

```
python -m pip install -r requirements-audit.txt
python tools/audit.py --python <interpreter that has the app dependencies>
```

`--gitleaks-mode diff --gitleaks-range BASE..HEAD` matches a pull request. The default is a full-history secret scan.
`gitleaks` must be on `PATH` for that check. A missing blocking tool fails the run.

The installer workflow (`.github/workflows/build-installer.yml`) calls `.github/workflows/audit.yml` as its first job.
`install-script`, `online-smoke`, `thin-smoke`, and `build-thin` do not start when that job fails.
CodeQL (Python, weekly and manual) is `.github/workflows/codeql.yml`. Dependabot opens weekly pull requests for pip
and monthly ones for GitHub Actions. `torch`, `torchaudio`, `transformers`, `qwen-tts`, and `qwen-asr` are ignored
there on purpose.

## Fixed in this pass

- `tools/pin_hf_hashes.py`: SHA-1 is the git blob id used to check a small file against the Hugging Face listing.
  Marked `usedforsecurity=False` (bandit B324, high).
- mypy, under `mypy.ini`, reported errors in annotated code (optional values, JSON `object` values passed to `int`,
  Windows-only attributes, PySide6 constructors the stubs type as `None`, unused `# type: ignore` comments).
  Fixed with annotations, locals that mypy can narrow, and `cast` where the existing call already accepts the value.
  No file-wide ignore was added. One of those edits named a local `cast` in `cli.py` and shadowed `typing.cast`
  (ruff F823). The local is `speaker_cast`.
- gitleaks `generic-api-key` matched the public SHA-256 pins in `infra/text_models.py` (the word `token` inside
  `tokenizer_config.json`, then a 64-hex digest). `.gitleaks.toml` allowlists that rule only on that file, and only
  on a line that contains a 64-hex digest. An AWS-shaped key on the next line still fails the scan.

## Accepted (not changed)

| Tool | Item | Why it stays |
| --- | --- | --- |
| bandit B310 | `urllib.request.urlopen` in `tools/pin_hf_hashes.py` and `tools/audit.py` | Medium. The calls go to `https://huggingface.co` and `https://api.osv.dev`, hosts the tool builds, not to a `file:` URL taken from a user. |
| pip-audit | `transformers==4.57.6` high advisories, allowlisted in `tools/audit-allowlist.toml`: PYSEC-2025-217 / CVE-2025-14929 (X-CLIP conversion; last affected 5.0.0rc0), GHSA-29pf-2h5f-8g72 / CVE-2026-4372 (fix 5.3.0), GHSA-fgcw-684q-jj6r / CVE-2026-5241 (fix 5.5.0), GHSA-xrqw-3rrv-vx5w / CVE-2026-9856 (fix 5.10.0), GHSA-x9r9-c232-4q39 / CVE-2026-80047 (affected through 5.8.1) | The published fixes are on the 5.x line. There is no 4.57.x fix. qwen-asr 0.0.6 and qwen-tts 0.1.1 pin transformers 4.57.x (their METADATA says `==4.57.6` and `==4.57.3`), and transformers 5.x breaks them. `requirements-verified.txt` pins 4.57.6 for that stack. |
| pip-audit | `transformers==4.57.6` — PYSEC-2026-2288 / GHSA-69w3-r845-3855 (`Trainer` arbitrary code execution). GitHub rates it moderate. The fix is transformers 5.0.0. | Moderate does not block a release and is not on the allowlist. The same 4.57 pin applies. |
| vulture | `ui/std_buttons.py` `ButtonTranslator.translate` argument `disambiguation` (100% confidence) | The name is part of Qt's `QTranslator.translate` signature. The override has to accept it. The method does not use the value; returning `None` means "not my string." |
| radon | 55 functions at rank D or worse. Rank F: `online_fetch.run`, `model_downloader._ensure_model`, `main.main`, `narration.narrate_book`, `text_cleanup.validate`, `book_parsers.parse_fb2`, `dataset_builder.run`, `runtime_reuse.candidate_prefixes`, `book_parsers.parse_epub`, `backup.collect_items`. Rank E and D are in the audit report. | Report only. These are the long pipeline, download, UI, and tool functions. Splitting them is not part of this pass. |
| ruff E501 / UP | Style and `pyupgrade` rewrites | The tree uses long lines and the existing typing style. Selecting those rules would bury the real errors. |
| mypy | `ignore_missing_imports` | Third-party libraries in this stack (torch, PySide6, and others) ship no stubs. Our own modules are still type-checked. There is no `ignore_errors`. |
| gitleaks | 64-hex lines in `infra/text_models.py` for `generic-api-key` only | Public SHA-256 pins of Hugging Face model files, not credentials. See `.gitleaks.toml`. |

## Tests

`python -m pytest -q` (the suite in `tests/`, including `tests/test_audit.py` for the gate rules).
