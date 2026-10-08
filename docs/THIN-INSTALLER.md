# Online installer (thin): design and status

**Goal.** `Voxprint-Setup-online.exe` is a small file (a few tens of MB). It installs only our own program shell (Qt + the Voxprint code)
and then downloads whatever is missing **from the developers' own sites** — PyTorch from `download.pytorch.org`, every other library from
PyPI — pinned by version and SHA-256. Before downloading anything it looks at what the PC already has and reuses it. Nothing third-party
is repacked or uploaded by us, so a CI build takes minutes instead of an hour and the release holds only the shell and a manifest.

## What comes from where

| Part | Source | Pinned by | Fallback |
|---|---|---|---|
| Voxprint shell (Qt, numpy, soundfile, our code) | our GitHub release (`Voxprint-shell-01.zip`) | SHA-256 in the manifest | — |
| PyTorch + torchaudio (2.11.0; `cu128`, `cu126` or `cpu`) | `download.pytorch.org/whl/<flavor>/` | `infra/runtime_lock.json` | optional mirror (`urls`) |
| ~70 other libraries (transformers, scipy, librosa, onnxruntime, peft ...) | PyPI (`files.pythonhosted.org`) | `infra/runtime_lock.json` | optional mirror |
| 3 sdist-only pure-Python packages (`eng-to-ipa`, `sox`, `docopt`) | built from the hash-checked PyPI sdist in CI, shipped as small wheels in our release | SHA-256 in the manifest | — |
| Visual C++ runtime | Microsoft (`aka.ms/vs/17/release/vc_redist.x64.exe`, fetched by CI and embedded in the setup) — installed only if the PC lacks 14.29 or newer | — | — |
| ffmpeg | the PC's own, else the pinned LGPL build of BtbN/FFmpeg-Builds (`infra/assets_manifest.json`, SHA-256) — on first use, as before | SHA-256 | bundled `imageio-ffmpeg` |
| Models (Qwen3 TTS / ASR / aligner, translation) | Hugging Face → ModelScope → our HF backup mirror (`infra/model_downloader.py`) | pinned revisions + sizes/SHA-256 | see `docs/MODELS.md` |

Every download: HTTP `Range` resume, retries, size + SHA-256 check, zip-slip-safe unpack, **network interface hopping** (`infra/netroute.py`,
Settings → Network interface) and the stall watchdog; a component with several addresses (`urls`) tries the upstream one first, then the fallbacks.
The lock is regenerated with `python tools/make_runtime_lock.py` (uses `uv pip compile`; review the diff; `tools/check_lock_urls.py` verifies that
every file is still at its address with the pinned size). CI only reads the committed lock.

## What is detected before anything is downloaded

1. **Our own runtime folder** (`%LOCALAPPDATA%\Voxprint\runtime`, survives uninstall/reinstall): every component whose SHA-256 is recorded is skipped.
2. **PyTorch of another program** (system Python, conda, a project venv, a Pinokio app, `VIRTUAL_ENV`): `infra/runtime_reuse.py` reads the
   `torch-*.dist-info` folders it finds (nothing is executed from foreign environments), applies the rules below and then **runs a check in a child
   process of Voxprint** (`Voxprint.exe --probe-torch`: import torch/torchaudio, a tensor computation, a NumPy round trip, and a CUDA computation if
   the PC has an NVIDIA GPU). Only if it passes is that `site-packages` appended to the **end** of `sys.path` (our own pinned libraries always win; the
   foreign environment only supplies `torch` and `torchaudio`) and remembered in `runtime\reuse.json`. If a remembered copy changes or disappears it is
   forgotten and the module is downloaded. `Voxprint.exe --install-modules --own-torch` forces our own copy; `VOXPRINT_NO_REUSE=1` switches the search off.
3. **ffmpeg** on `PATH` (smoke-tested) is used before the pinned download (`core/audio_utils.ensure_ffmpeg`, `infra/assets.py`).
4. **Visual C++ runtime**: the setup reads `HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64` and skips the redistributable when 14.29+ is present.

### Compatibility rules for a PyTorch found on the PC (all must hold; each rejection is logged with its reason)

| # | Rule |
|---|---|
| 1 | Same CPython minor version as Voxprint (the `Tag:` of the installed wheel, now `cp311`) and `win_amd64` |
| 2 | `torch` inside `compat.torch` of the lock (now `>=2.8,<2.13`; we tested 2.11.0) |
| 3 | `torchaudio` in the same `site-packages`, same release number as `torch` |
| 4 | Flavor: a CPU build only on a PC without an NVIDIA GPU; a CUDA build must not need a newer driver than installed (older CUDA builds are fine); a build without a local tag on Windows counts as CPU |
| 5 | `torch/lib`, `torch/__init__.py`, `torchaudio/__init__.py` exist |
| 6 | The child-process check passes (150 s limit) |

The flavor we download follows the driver (`nvidia-smi`): CUDA ≥ 12.8 → `cu128`, ≥ 12.6 → `cu126`, otherwise `cpu` (`VOXPRINT_TORCH_FLAVOR` forces one).

## Start-up flow

1. `Voxprint-Setup-online.exe` (Inno, `/DONLINE /DTHIN`) runs `voxprint-fetch --role core`: downloads the shell, writes `_internal\modules.json`
   (manifest address `manifest-thin-<channel>.json`), installs the Visual C++ runtime if needed.
2. The app starts, activates what is in the runtime folder (`infra/modules.activate()`), and opens the **Components** window (also Settings → Components).
3. The window reads the manifest (last good copy cached offline), looks for a reusable PyTorch, and downloads the missing modules (`libs`, `text`,
   `audio`, `torch`) without a click, followed by the SAGE text clean-up model (step 1). One live line names what is downloaded right now
   (percentage, speed), says when a file is being verified (SHA-256) or shows the error; a bar shows the overall progress. No restart is needed
   afterwards; features that need a module work as soon as it is ready.
4. Step 2 follows at once: the first-run download of ALL models (TTS, aligner, Qwen3-ASR, SAGE if step 1 could not get it) into the chosen models
   folder, mirrored on the same live line (backup restore and existing copies first; no "maximum quality" button). Closing the window does not stop it.
5. CLI for scripts and CI: `Voxprint.exe --modules-status`, `--install-modules [ids] [--own-torch]`, `--probe-torch` (results also in `logs\modules.txt`, `logs\probe_torch.txt`).

## Failure and resume

Finished components are recorded (SHA-256) and skipped next time; a broken one is deleted and fetched again; closing the window or going offline just
stops (press Download again); the runtime folder can be deleted at any time. A new release has a new manifest — changed files get new hashes, the rest stays.

## Network probe

Before the first download of a run, `voxprint-fetch` (installer and Components window) fetches the first 256 KB of a component (`Range` request) through every network route at once (default, no-proxy, each local interface; `infra/netroute.pick_fastest`, at most 4 s) and remembers the fastest; the others stay the connect fallback. Only with the *Auto* network setting. A component with several sources (`url` + `urls`) has its sources ranked the same way, once per host (`online_fetch.ranked`); equal speeds keep the manifest order.

## Pinned vs latest components

Every release's manifest is *pinned*: the exact files (size + SHA-256) this release was built and tested with. `modules.json` and the online installer also know the newest release's manifest (`releases/latest/download/...`; `latest_manifest_url`, installer define `LatestManifestUrl`, set by `build_online.ps1` for GitHub builds).

- Default: pinned. If a pinned download fails or its SHA-256 does not match (an upstream file disappeared), the whole latest manifest is used instead, so an online install does not brick.
- Optional "Try the latest component versions" (installer checkbox on the models-folder page, silent `/Latest=1`; Components window check box): the latest manifest first, the pinned one as the fallback.
- A manifest is always used as a whole (parts of two releases are never mixed); runtime modules are selected by module id, so a different number of parts is fine. The app remembers the manifest it installed from (`state/modules_channel.json`) and reads the module list from it.
- `voxprint-fetch --manifest <pinned> --latest-manifest <latest> [--prefer latest]`.

## CI (`build-thin` job of `.github/workflows/build-installer.yml`)

Installs only the pinned shell packages (Qt, numpy, soundfile ...) into a venv, runs the unit tests that need no PyTorch, builds the shell with
PyInstaller (`build_thin.bat`), writes `Voxprint-shell-01.zip` + `manifest-thin-<channel>.json` and compiles `Voxprint-Setup-online.exe`
(`installer/build_online.ps1 -RuntimeLock infra\runtime_lock.json`). It then **really runs the product on the runner**: the frozen shell downloads the whole
runtime from upstream (CPU PyTorch), imports it, and a second run proves that a PyTorch lying around on the PC is found, checked and reused instead of
downloaded. Only then are the installer, its hash and the manifest attached to the release (older assets are never touched).

## Portable setup folder (foundation; the installer checkbox comes later)

Planned: an optional **"Keep a portable setup folder"** checkbox. The installer then downloads **all** components — and later the models — into one user
folder (e.g. `Voxprint Portable`) even if the PC already has some, installs from it, and the folder can be reused for an offline re-install or copied to
another PC. On the next run the installer compares the manifest (new installer/app version) and fetches only the components whose hash changed. This replaces
the idea of a big offline installer.

Already in place: the layout and the downloader switch `voxprint-fetch --portable <folder>`:

```
Voxprint Portable/
  manifest.json          the manifest the folder was made from
  SHA256SUMS.txt         <sha256> *components/<id>/<file>
  components/<id>/<file> one folder per component (shell part, wheel, later model archives)
```

With `--portable`, every selected component is kept in the folder (also those already installed); a file in the folder that passes the SHA-256 check is used
instead of the network. Test: `tests/test_upstream_fetch.py`. Missing: the installer checkbox/page, models in the folder, "newer version available" check,
a launcher that installs from the folder without any manifest URL.

## Decisions to know about

* PyTorch 2.11.0 is the version tested so far; reusing 2.8–2.12 from another program is allowed by the rules above but only the child-process check proves it works.
* The mirror fallback (`urls`) is implemented but no mirror is populated yet: the CUDA wheel of PyTorch (2.6 GB) is above GitHub's 2 GiB asset limit, so a mirror
  would have to live on Hugging Face (or be split). If upstream removes an old wheel, the lock must be regenerated (`tools/check_lock_urls.py` warns in CI).
* Python 3.11 / Windows x64 only (the lock is for `cp311-win_amd64`). Linux keeps its own installer.
