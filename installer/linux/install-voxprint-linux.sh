#!/usr/bin/env bash
# Voxprint AI Audiobook Builder - Linux installer (EXPERIMENTAL).
#
#   curl -fsSL https://github.com/@REPO@/releases/download/@TAG@/install-voxprint-linux.sh | bash
#   or, from the unpacked Voxprint-linux-experimental.tar.gz:   ./install-voxprint-linux.sh
#
# Installs for the current user only (no root, except the optional "apt-get install" of system libraries):
#   ~/.local/share/voxprint/app    the program (Python source, same code as the Windows build)
#   ~/.local/share/voxprint/venv   its Python environment (PySide6, PyTorch, transformers ...)
#   ~/.local/share/voxprint/       models, voices, settings, logs (created by the program)
#   ~/.local/bin/voxprint          the launcher;  ~/.local/share/applications/voxprint.desktop  the menu entry
# PyTorch: torch 2.14.1 cu130 (NVIDIA driver 600+) with TorchAudio 2.11.0. --cpu installs the CPU wheels (CI).
# Re-running the script updates the program (the environment is reused).  `--uninstall` removes it again.
set -u
set -o pipefail

REPO="${VOXPRINT_REPO:-@REPO@}"
TAG="${VOXPRINT_TAG:-@TAG@}"
BASE_URL="${VOXPRINT_BASE_URL:-https://github.com/$REPO/releases/download/$TAG}"

DATA="${VOXPRINT_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/voxprint}"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/voxprint"
APP="$DATA/app"
VENV="$DATA/venv"
LAUNCHER="$BIN_DIR/voxprint"

# system packages (Debian / Ubuntu / AnduinOS / Mint names): Qt xcb platform plugin, OpenGL, ffmpeg, venv
APT_PACKAGES="python3-venv ffmpeg libegl1 libgl1 libdbus-1-3 libfontconfig1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libpulse0"

MODE=install; ASSUME_YES=0; INSTALL_DEPS=0; TORCH=cu130; FROM_DIR=""; NO_DESKTOP=0; PURGE=0; RECREATE=0; SKIP_CHECK=0; NO_SYSCHECK=0

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
Voxprint for Linux (experimental) - installer

  install-voxprint-linux.sh [options]
    --check            only report what is missing (system libraries, Python), change nothing
    --install-deps     install the missing system packages with "sudo apt-get install" (Debian/Ubuntu family)
    --yes, -y          answer yes to questions
    --cpu              install the CPU build of PyTorch (CI and machines without an NVIDIA GPU)
    --torch-backend B  wheel index: cu130 (default), cpu. "auto" means cu130
    --from-dir DIR     take the program from a checkout / unpacked folder (DIR contains main.py) instead of downloading
    --recreate         delete and recreate the Python environment
    --no-desktop       do not create the menu entry
    --skip-check       do not run the self-tests at the end
    --no-system-check  do not check the system packages (other distributions, containers)
    --uninstall        remove the program, the environment, the launcher and the menu entry (models/voices are kept)
    --purge            with --uninstall: also delete models, voices and settings (${DATA})
    -h, --help         this text
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check ;;
    --install-deps) INSTALL_DEPS=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --cpu) TORCH=cpu ;;
    --torch-backend) shift; [ $# -gt 0 ] || die "--torch-backend needs a value"; if [ "$1" = auto ]; then TORCH=cu130; else TORCH="$1"; fi ;;
    --from-dir) shift; [ $# -gt 0 ] || die "--from-dir needs a folder"; FROM_DIR="$1" ;;
    --recreate) RECREATE=1 ;;
    --no-desktop) NO_DESKTOP=1 ;;
    --skip-check) SKIP_CHECK=1 ;;
    --no-system-check) NO_SYSCHECK=1 ;;
    --uninstall) MODE=uninstall ;;
    --purge) PURGE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

# ----------------------------------------------------------------------------------------------- uninstall
if [ "$MODE" = uninstall ]; then
  say "Removing Voxprint (program, environment, launcher, menu entry)"
  rm -rf "$APP" "$APP.old" "$APP.new" "$VENV" "$DATA/install.json"
  rm -f "$LAUNCHER" "$APPS_DIR/voxprint.desktop" "$ICON_DIR/voxprint.png"
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
  if [ "$PURGE" = 1 ]; then
    say "Deleting $DATA (models, voices, settings, logs)"
    rm -rf "$DATA"
  else
    echo "Kept: $DATA (models, voices, settings). Use --purge to delete them too."
  fi
  echo "Done."
  exit 0
fi

[ "$(uname -s)" = Linux ] || die "this installer is for Linux (Windows has Voxprint-Setup-*.exe)"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ;;
  aarch64|arm64) warn "ARM64 is untested (PyTorch CPU wheels exist, some dependencies may not install)." ;;
  *) die "unsupported CPU architecture: $ARCH" ;;
esac

# Best effort from os-release. A year we cannot read is not a warning, and a warning never blocks.
warn_if_old_distro() {
  local file id ver pretty year yy q sq
  q='"'; sq="'"
  file="${VOXPRINT_OS_RELEASE:-/etc/os-release}"
  [ -r "$file" ] || return 0
  id=$(sed -n -- 's/^VERSION_ID=//p' "$file" | head -n 1)
  ver=$(sed -n -- 's/^VERSION=//p' "$file" | head -n 1)
  pretty=$(sed -n -- 's/^PRETTY_NAME=//p' "$file" | head -n 1)
  id=${id#"$q"}; id=${id%"$q"}; id=${id#"$sq"}; id=${id%"$sq"}
  ver=${ver#"$q"}; ver=${ver%"$q"}; ver=${ver#"$sq"}; ver=${ver%"$sq"}
  pretty=${pretty#"$q"}; pretty=${pretty%"$q"}; pretty=${pretty#"$sq"}; pretty=${pretty%"$sq"}
  year=""
  if printf '%s' "$id" | grep -Eq '^[0-9]{2}\.[0-9]{2}([^0-9]|$)'; then
    yy=$(printf '%s' "$id" | cut -c 1-2)
    year=$((2000 + 10#$yy))
  else
    year=$(printf '%s\n%s\n%s\n' "$id" "$ver" "$pretty" | grep -Eo '20[0-9]{2}|19[0-9]{2}' | head -n 1 || true)
  fi
  case "$year" in
    ''|*[!0-9]*) return 0 ;;
  esac
  if [ "$year" -lt 2025 ]; then
    warn "This Linux distribution looks older than 2025 (${pretty:-${id:-unknown}}). Supported systems: current and previous year OS releases (Linux distributions released from 2025). Older systems are not a goal. Installation continues."
  fi
}
warn_if_old_distro

# ----------------------------------------------------------------------------------------------- system checks
say "Voxprint for Linux (EXPERIMENTAL) - checking the system"
PY="${VOXPRINT_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || die "python3 not found. Debian/Ubuntu: sudo apt install python3 python3-venv"
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)'; then
  die "Python 3.14 is required (found: $("$PY" -c 'import sys; print(sys.version.split()[0])')). Set VOXPRINT_PYTHON=/path/to/python3.14 to use another one."
fi
echo "Python: $("$PY" -c 'import sys; print(sys.version.split()[0])') ($(command -v "$PY"))"

MISSING=""
if [ "$NO_SYSCHECK" = 1 ]; then
  echo "System packages: check skipped"
elif command -v dpkg-query >/dev/null 2>&1; then
  for p in $APT_PACKAGES; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || MISSING="$MISSING $p"
  done
  # "ffmpeg" may be provided by another package (e.g. a snap / ffmpeg-free): the command is what counts
  case " $MISSING " in *" ffmpeg "*) command -v ffmpeg >/dev/null 2>&1 && MISSING="${MISSING/ ffmpeg/}" ;; esac
else
  warn "no dpkg found (not Debian/Ubuntu): install the equivalents of: $APT_PACKAGES"
fi
if [ -n "$MISSING" ]; then
  echo "Missing system packages:$MISSING"
  if [ "$MODE" = check ]; then echo "Install with: sudo apt-get install -y$MISSING"; exit 1; fi
  DO=$INSTALL_DEPS
  if [ "$DO" = 0 ] && [ "$ASSUME_YES" = 1 ]; then DO=1; fi
  if [ "$DO" = 0 ] && [ -r /dev/tty ]; then
    printf 'Install them now with sudo apt-get? [Y/n] ' >/dev/tty; read -r ans </dev/tty || ans=n
    case "$ans" in ""|y|Y|yes|Yes) DO=1 ;; esac
  fi
  if [ "$DO" = 1 ]; then
    SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
    # shellcheck disable=SC2086
    $SUDO apt-get update && $SUDO apt-get install -y $MISSING || die "apt-get failed. Install manually: sudo apt-get install -y$MISSING"
  else
    die "install the missing packages first:  sudo apt-get install -y$MISSING   (or re-run with --install-deps)"
  fi
else
  echo "System packages: OK"
fi
if [ "$MODE" = check ]; then
  command -v nvidia-smi >/dev/null 2>&1 && echo "NVIDIA driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)" || echo "NVIDIA driver: none found (the CPU build of PyTorch will be used)"
  echo "Check finished."; exit 0
fi
"$PY" -c 'import venv, ensurepip' 2>/dev/null || die "python3-venv is missing: sudo apt-get install python3-venv"

# ----------------------------------------------------------------------------------------------- the program
mkdir -p "$DATA" "$CACHE" "$BIN_DIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo .)"
SRC=""
if [ -n "$FROM_DIR" ]; then SRC="$FROM_DIR"
elif [ -f "$HERE/app/main.py" ]; then SRC="$HERE/app"        # unpacked Voxprint-linux-experimental.tar.gz
fi
rm -rf "$APP.new"
if [ -n "$SRC" ]; then
  [ -f "$SRC/main.py" ] || die "$SRC does not contain main.py"
  say "Installing the program from $SRC"
  mkdir -p "$APP.new"
  for item in main.py credits.json requirements.txt requirements-verified.txt requirements-nodeps.txt requirements-torch.txt \
              LICENSE NOTICE THIRD_PARTY_NOTICES.md core infra ui workers locales prompts licenses assets installer; do
    [ -e "$SRC/$item" ] && cp -a "$SRC/$item" "$APP.new/"
  done
  find "$APP.new" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  [ -d "$SRC/docs" ] && mkdir -p "$APP.new/docs" && cp -a "$SRC"/docs/LINUX-TEST-CHECKLIST.md "$APP.new/docs/" 2>/dev/null
else
  say "Downloading the program ($TAG)"
  FETCH="$CACHE/voxprint-fetch.py"
  "$PY" - "$BASE_URL/voxprint-fetch.py" "$FETCH" <<'PYDL' || die "could not download voxprint-fetch.py from $BASE_URL"
import sys, urllib.request
url, dst = sys.argv[1:3]
req = urllib.request.Request(url, headers={"User-Agent": "voxprint-install/1"})
with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
    f.write(r.read())
PYDL
  "$PY" "$FETCH" --manifest "$BASE_URL/manifest-linux.json" --dest "$APP.new" --cache "$CACHE" || die "download of the program failed (see the message above)"
fi
[ -f "$APP.new/main.py" ] || die "the program files are incomplete"
rm -rf "$APP.old"
[ -d "$APP" ] && mv "$APP" "$APP.old"
mv "$APP.new" "$APP"
rm -rf "$APP.old"

# ----------------------------------------------------------------------------------------------- the Python environment
if [ "$RECREATE" = 1 ]; then rm -rf "$VENV"; fi
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys' 2>/dev/null; then rm -rf "$VENV"; fi
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating the Python environment in $VENV"
  "$PY" -m venv "$VENV" || die "could not create the virtual environment"
fi
VPY="$VENV/bin/python"
say "Installing the installer tools (pip, uv)"
"$VPY" -m pip install --quiet --upgrade pip uv || die "pip could not install uv (is the network reachable?)"
UV="$VENV/bin/uv"
export UV_LINK_MODE=copy UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE/uv}"

say "Installing PyTorch $TORCH (torch 2.14.1, torchaudio 2.11.0)"
install_torch() {
  "$UV" pip install --python "$VPY" "torch==2.14.1" "torchaudio==2.11.0" --index-url "https://download.pytorch.org/whl/$1"
}
if ! install_torch "$TORCH"; then
  if [ "$TORCH" != cpu ]; then
    warn "PyTorch ($TORCH) failed - falling back to the CPU build"
    install_torch cpu || die "PyTorch could not be installed"
    TORCH=cpu
  else
    die "PyTorch could not be installed"
  fi
fi
say "Installing the other Python packages (this takes a few minutes)"
"$UV" pip install --python "$VPY" -r "$APP/requirements.txt" -r "$APP/requirements-verified.txt" || die "installing requirements.txt failed"
"$UV" pip install --python "$VPY" --no-deps -r "$APP/requirements-nodeps.txt" || die "installing requirements-nodeps.txt failed"

# ----------------------------------------------------------------------------------------------- launcher, menu entry
say "Creating the launcher $LAUNCHER"
cat > "$LAUNCHER" <<LAUNCH
#!/usr/bin/env bash
# Voxprint launcher (written by install-voxprint-linux.sh).  Extra Qt hints: QT_QPA_PLATFORM=xcb voxprint
export VOXPRINT_HOME="\${VOXPRINT_HOME:-$DATA}"
export PYTHONNOUSERSITE=1
exec "$VPY" "$APP/main.py" "\$@"
LAUNCH
chmod +x "$LAUNCHER"
if [ "$NO_DESKTOP" = 0 ]; then
  mkdir -p "$APPS_DIR" "$ICON_DIR"
  [ -f "$APP/installer/linux/voxprint-256.png" ] && cp "$APP/installer/linux/voxprint-256.png" "$ICON_DIR/voxprint.png"
  if [ -f "$APP/installer/linux/voxprint.desktop" ]; then
    sed "s|@LAUNCHER@|$LAUNCHER|g" "$APP/installer/linux/voxprint.desktop" > "$APPS_DIR/voxprint.desktop"
    chmod 644 "$APPS_DIR/voxprint.desktop"
  fi
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
  command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "${ICON_DIR%/hicolor/*}/hicolor" >/dev/null 2>&1 || true
fi
GPU="none"
command -v nvidia-smi >/dev/null 2>&1 && GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
printf '{"torch_backend": "%s", "gpu": "%s", "tag": "%s", "installed": "%s"}\n' "$TORCH" "$GPU" "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$DATA/install.json"

# ----------------------------------------------------------------------------------------------- self-tests
RC=0
if [ "$SKIP_CHECK" = 0 ]; then
  say "Self-tests (no GPU and no models needed)"
  export QT_QPA_PLATFORM=offscreen VOXPRINT_HOME="$DATA"
  "$VPY" "$APP/main.py" --selftest-imports || { warn "some libraries failed to import (see above)"; RC=1; }
  "$VPY" "$APP/main.py" --selftest-text || { warn "the text pipeline self-test failed"; RC=1; }
  "$VPY" "$APP/main.py" --selftest || { warn "the window did not start (Qt platform plugin?)"; RC=1; }
fi

echo
if [ "$RC" = 0 ]; then say "Voxprint is installed."; else warn "Voxprint is installed, but a self-test failed - see docs/LINUX-TEST-CHECKLIST.md"; fi
echo "Start it from the application menu or with:  voxprint"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "$BIN_DIR is not in PATH - log out and in again, or run: $LAUNCHER" ;; esac
echo "Models are downloaded on the first run into $DATA/models.  Update: run this script again.  Remove: install-voxprint-linux.sh --uninstall"
exit "$RC"
