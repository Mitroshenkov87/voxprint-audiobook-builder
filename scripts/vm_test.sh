#!/usr/bin/env bash
# Non-interactive GPU check for an installed or source-tree Voxprint.
#
#   scripts/vm_test.sh --smoke --json --tag v1.0.0-rc.3
#   scripts/vm_test.sh --full --json --results ./vm-results/audiobook-builder/v1.0.0-rc.3
#
# Exit codes: 0 pass, 1 a test failed, 2 setup error, 3 no NVIDIA GPU.
# Successful steps leave a marker in ~/.done/<step> (or $VOXPRINT_VM_DONE) and are skipped on the next run.
# Results land in ./vm-results/audiobook-builder/<tag>/ unless --results or $VOXPRINT_VM_RESULTS is set.
# Sync that folder to /workspace/vm-results/audiobook-builder/<tag>/.
set -eu
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Prefer python. On Windows, python3 is often the Store alias (WindowsApps), which does not run.
PY=""
for cand in python python3; do
  if "$cand" -X utf8 -c "import sys; p=sys.executable.replace(chr(92), '/').lower(); raise SystemExit(0 if 'windowsapps' not in p else 1)" >/dev/null 2>&1; then
    PY=$cand
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "python was not found" >&2
  exit 2
fi
WANT_JSON=0
SMOKE_FLAG=0
FULL_FLAG=0
TAG_OPT=""
RESULTS_OPT=""
BIN_OPT=""
VOX=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE_FLAG=1 ;;
    --full) FULL_FLAG=1 ;;
    --json) WANT_JSON=1 ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "missing value for --tag" >&2; exit 2; }
      TAG_OPT=$2
      shift
      ;;
    --results)
      [[ $# -ge 2 ]] || { echo "missing value for --results" >&2; exit 2; }
      RESULTS_OPT=$2
      shift
      ;;
    --bin)
      [[ $# -ge 2 ]] || { echo "missing value for --bin" >&2; exit 2; }
      BIN_OPT=$2
      shift
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$FULL_FLAG" -eq 1 ]]; then
  MODE="full"
elif [[ "$SMOKE_FLAG" -eq 1 ]]; then
  MODE="smoke"
else
  echo "pass --smoke or --full" >&2
  exit 2
fi

resolve_bin() {
  if [[ -n "$BIN_OPT" ]]; then
    [[ -x "$BIN_OPT" ]] || return 1
    VOX=("$BIN_OPT")
    return 0
  fi
  if [[ -n "${VOXPRINT_BIN:-}" ]]; then
    [[ -x "$VOXPRINT_BIN" ]] || return 1
    VOX=("$VOXPRINT_BIN")
    return 0
  fi
  if command -v voxprint-cli >/dev/null 2>&1; then
    VOX=("voxprint-cli")
    return 0
  fi
  if command -v voxprint >/dev/null 2>&1; then
    VOX=("voxprint")
    return 0
  fi
  if [[ -f "$ROOT/main.py" ]]; then
    VOX=("$PY" "$ROOT/main.py")
    return 0
  fi
  return 1
}

if ! resolve_bin; then
  echo "voxprint was not found (set --bin or VOXPRINT_BIN)" >&2
  exit 2
fi

sanitize_tag() {
  local raw="$1"
  raw=${raw//\//-}
  raw=${raw//\\/-}
  raw=${raw// /-}
  printf '%s\n' "$raw"
}

TAG=$(sanitize_tag "${TAG_OPT:-unknown}")
if [[ -n "$RESULTS_OPT" ]]; then
  RESULTS=$RESULTS_OPT
elif [[ -n "${VOXPRINT_VM_RESULTS:-}" ]]; then
  RESULTS=$VOXPRINT_VM_RESULTS
else
  RESULTS="$PWD/vm-results/audiobook-builder/$TAG"
fi
DONE=${VOXPRINT_VM_DONE:-$HOME/.done}
PEAK="$RESULTS/vram.peak"
WORK="$RESULTS/work"
POLLER=""

cleanup() {
  if [[ -n "${POLLER}" ]]; then
    kill "$POLLER" 2>/dev/null || true
    wait "$POLLER" 2>/dev/null || true
    POLLER=""
  fi
}
trap cleanup EXIT

mkdir -p "$RESULTS" "$DONE" "$WORK" || exit 2

note() {
  local id="$1" status="$2" rc="$3" line
  line=$(STEP_ID="$id" STEP_STATUS="$status" STEP_RC="$rc" "$PY" -X utf8 - <<'PY'
import json, os
print(json.dumps({
    "type": "step",
    "id": os.environ["STEP_ID"],
    "status": os.environ["STEP_STATUS"],
    "exit": int(os.environ["STEP_RC"]),
}, ensure_ascii=False))
PY
)
  printf '%s\n' "$line" >> "$RESULTS/steps.jsonl"
  if [[ "$WANT_JSON" -eq 1 ]]; then
    printf '%s\n' "$line"
  else
    printf '%s %s\n' "$id" "$status"
  fi
}

finish() {
  local code="$1" reason="$2"
  cleanup
  set +e
  CODE="$code" REASON="$reason" MODE="$MODE" TAG="$TAG" RESULTS="$RESULTS" PEAK_FILE="$PEAK" WANT_JSON="$WANT_JSON" \
    "$PY" -X utf8 - <<'PY'
import json, os
from pathlib import Path
results = Path(os.environ["RESULTS"])
steps = []
log = results / "steps.jsonl"
if log.is_file():
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(json.loads(line))
peak_raw = ""
peak_path = Path(os.environ["PEAK_FILE"])
if peak_path.is_file():
    peak_raw = "".join(ch for ch in peak_path.read_text(encoding="utf-8") if ch.isdigit())
payload = {
    "ok": os.environ["CODE"] == "0",
    "exit": int(os.environ["CODE"]),
    "mode": os.environ["MODE"],
    "tag": os.environ["TAG"],
    "reason": os.environ["REASON"],
    "steps": steps,
    "peak_vram_mib": int(peak_raw) if peak_raw else None,
    "results": str(results),
}
text = json.dumps(payload, ensure_ascii=False)
results.mkdir(parents=True, exist_ok=True)
(results / "result.json").write_text(text + "\n", encoding="utf-8")
if os.environ["WANT_JSON"] == "1":
    print(text)
else:
    print(os.environ["REASON"] or ("pass" if payload["ok"] else "fail"))
PY
  exit "$code"
}

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  finish 3 "no GPU"
fi

sample_vram() {
  local v cur=""
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -cd '0-9' || true)
  [[ -n "$v" ]] || return 0
  if [[ -f "$PEAK" ]]; then
    cur=$(tr -cd '0-9' < "$PEAK" || true)
  fi
  if [[ -z "$cur" || "$v" -gt "$cur" ]]; then
    printf '%s\n' "$v" > "$PEAK"
  fi
}

start_poll() {
  sample_vram
  (
    while true; do
      sample_vram
      sleep 2
    done
  ) >/dev/null 2>&1 &
  POLLER=$!
}

stop_poll() {
  cleanup
  sample_vram
}

skipped() {
  [[ -f "$DONE/$1" ]]
}

mark_done() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$DONE/$1"
}

require_json_dry_run() {
  local log="$1"
  "$PY" -X utf8 - "$log" <<'PY'
import json, sys
lines = [ln for ln in open(sys.argv[1], encoding="utf-8").read().splitlines() if ln.strip()]
if not lines:
    sys.exit(1)
objs = [json.loads(ln) for ln in lines]
result = next(obj for obj in reversed(objs) if obj.get("type") == "result")
durations = result.get("durations") or {}
if durations.get("audio_s") is not None or result.get("outputs") != [] or result.get("dry_run") is not True:
    sys.exit(1)
PY
}

require_narration_result() {
  local log="$1"
  "$PY" -X utf8 - "$log" <<'PY'
import json, sys
lines = [ln for ln in open(sys.argv[1], encoding="utf-8").read().splitlines() if ln.strip()]
objs = [json.loads(ln) for ln in lines]
result = next(obj for obj in reversed(objs) if obj.get("type") == "result")
audio = (result.get("durations") or {}).get("audio_s")
if not isinstance(audio, (int, float)):
    sys.exit(1)
PY
}

step_version() {
  local id="version"
  if skipped "$id"; then
    note "$id" skipped 0
    return 0
  fi
  set +e
  "${VOX[@]}" --json --version >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    set +e
    "${VOX[@]}" --version >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
    rc=$?
    set -e
  fi
  if [[ "$rc" -ne 0 ]]; then
    note "$id" fail "$rc"
    finish 1 "version failed"
  fi
  if [[ -z "$TAG_OPT" ]]; then
    local discovered
    discovered=$("$PY" -X utf8 - "$RESULTS/$id.log" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ver = ""
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        match = re.search(r"\b(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)", line)
        if match:
            ver = match.group(1)
        continue
    if isinstance(obj, dict) and obj.get("version"):
        ver = str(obj["version"])
if ver and not ver.startswith("v"):
    ver = "v" + ver
print(ver)
PY
)
    if [[ -n "$discovered" && -z "$RESULTS_OPT" && -z "${VOXPRINT_VM_RESULTS:-}" ]]; then
      local new_tag new_results
      new_tag=$(sanitize_tag "$discovered")
      new_results="$PWD/vm-results/audiobook-builder/$new_tag"
      if [[ "$new_results" != "$RESULTS" ]]; then
        mkdir -p "$(dirname "$new_results")"
        rm -rf "$new_results"
        mv "$RESULTS" "$new_results"
        RESULTS=$new_results
        TAG=$new_tag
        PEAK="$RESULTS/vram.peak"
        WORK="$RESULTS/work"
      fi
    elif [[ -n "$discovered" ]]; then
      TAG=$(sanitize_tag "$discovered")
    fi
  fi
  mark_done "$id"
  note "$id" pass 0
}

step_selftest() {
  local id="selftest"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  set +e
  "${VOX[@]}" --selftest >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then note "$id" fail "$rc"; finish 1 "selftest failed"; fi
  mark_done "$id"
  note "$id" pass 0
}

step_dry_run() {
  local id="dry-run"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  mkdir -p "$WORK/dry"
  printf '%s\n' "Привет. Это короткая проверка голоса." > "$WORK/smoke.txt"
  set +e
  "${VOX[@]}" --dry-run --json narrate "$WORK/smoke.txt" --voice levi --out "$WORK/dry/narration" \
    >"$RESULTS/dry-narrate.log" 2>"$RESULTS/dry-narrate.err"
  local rc1=$?
  "${VOX[@]}" --dry-run --json prepare "$WORK/smoke.txt" --out "$WORK/dry/prepared.txt" --language ru \
    >"$RESULTS/dry-prepare.log" 2>"$RESULTS/dry-prepare.err"
  local rc2=$?
  set -e
  if [[ "$rc1" -ne 0 || "$rc2" -ne 0 ]]; then
    note "$id" fail 1
    finish 1 "dry-run failed"
  fi
  if ! require_json_dry_run "$RESULTS/dry-narrate.log" || ! require_json_dry_run "$RESULTS/dry-prepare.log"; then
    note "$id" fail 1
    finish 1 "dry-run JSON was not a result with null audio"
  fi
  mark_done "$id"
  note "$id" pass 0
}

step_smoke_narrate() {
  local id="smoke-narrate"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  mkdir -p "$WORK/smoke-out"
  printf '%s\n' "Привет. Это короткая проверка голоса." > "$WORK/smoke.txt"
  start_poll
  set +e
  "${VOX[@]}" narrate "$WORK/smoke.txt" --voice levi --format mp3 --json --out "$WORK/smoke-out" \
    >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc=$?
  set -e
  stop_poll
  if [[ "$rc" -ne 0 ]]; then note "$id" fail "$rc"; finish 1 "smoke narration failed"; fi
  if ! require_narration_result "$RESULTS/$id.log"; then
    note "$id" fail 1
    finish 1 "smoke narration did not report audio_s"
  fi
  mark_done "$id"
  note "$id" pass 0
}

step_vram() {
  local id="vram"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  sample_vram
  local n=""
  if [[ -f "$PEAK" ]]; then
    n=$(tr -cd '0-9' < "$PEAK" || true)
  fi
  if [[ -z "$n" ]]; then
    note "$id" fail 1
    finish 1 "nvidia-smi did not report memory.used"
  fi
  printf '%s\n' "$n" > "$RESULTS/vram.txt"
  mark_done "$id"
  note "$id" pass 0
}

step_dialogue() {
  local id="dialogue"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  mkdir -p "$WORK/dialogue-out"
  printf '%s\n' "Иван вошёл в комнату." "" "— Здравствуй, — сказал Иван." "" "— Здравствуй, — ответила Анна." \
    > "$WORK/dialogue.txt"
  set +e
  "${VOX[@]}" speakers "$WORK/dialogue.txt" --out "$WORK/marks.txt" --json \
    >"$RESULTS/speakers.log" 2>"$RESULTS/speakers.err"
  local rc1=$?
  set -e
  if [[ "$rc1" -ne 0 ]]; then note "$id" fail "$rc1"; finish 1 "speakers failed"; fi
  if ! "$PY" -X utf8 - "$WORK/marks.txt" <<'PY'
import sys
text = open(sys.argv[1], encoding="utf-8").read().casefold()
need = ("male", "female", "муж", "жен")
sys.exit(0 if any(tok in text for tok in need) else 1)
PY
  then
    note "$id" fail 1
    finish 1 "speaker marks have no second voice"
  fi
  set +e
  "${VOX[@]}" narrate "$WORK/dialogue.txt" --voice levi --male-voice natan --male2-voice shimon \
    --female-voice miriam --speaker-marks "$WORK/marks.txt" --format mp3 --json --out "$WORK/dialogue-out" \
    >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc2=$?
  set -e
  if [[ "$rc2" -ne 0 ]]; then note "$id" fail "$rc2"; finish 1 "dialogue narration failed"; fi
  mark_done "$id"
  note "$id" pass 0
}

step_yo() {
  local id="yo"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  printf '%s\n' "Еще одно слово в этом предложении." > "$WORK/yo-src.txt"
  set +e
  "${VOX[@]}" prepare "$WORK/yo-src.txt" --out "$WORK/yo-only.txt" --steps yo --language ru \
    >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then note "$id" fail "$rc"; finish 1 "yo prepare failed"; fi
  if ! "$PY" -X utf8 - "$WORK/yo-only.txt" <<'PY'
import sys
text = open(sys.argv[1], encoding="utf-8").read()
sys.exit(0 if ("\u0451" in text or "\u0401" in text) else 1)
PY
  then
    note "$id" fail 1
    finish 1 "prepared text has no yo"
  fi
  mark_done "$id"
  note "$id" pass 0
}

step_vxbook() {
  local id="vxbook"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  if ! "$PY" -X utf8 "$ROOT/scripts/vm_vxbook.py" "$WORK/story.vxbook" >"$RESULTS/$id-pack.log" 2>"$RESULTS/$id-pack.err"; then
    note "$id" fail 2
    finish 2 "could not write the vxbook"
  fi
  mkdir -p "$WORK/vxbook-out"
  set +e
  "${VOX[@]}" narrate "$WORK/story.vxbook" --voice levi --format mp3 --no-soundscape --json --out "$WORK/vxbook-out" \
    >"$RESULTS/$id.log" 2>"$RESULTS/$id.err"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then note "$id" fail "$rc"; finish 1 "vxbook narration failed"; fi
  mark_done "$id"
  note "$id" pass 0
}

step_mp3_labels() {
  local id="mp3-labels"
  if skipped "$id"; then note "$id" skipped 0; return 0; fi
  if ! "$PY" -X utf8 - "$WORK/vxbook-out" <<'PY'
import os, re, sys
pat = re.compile(r"^\d{2,} - .+\.mp3$", re.IGNORECASE)
found = []
for root, _dirs, files in os.walk(sys.argv[1]):
    for name in files:
        if pat.match(name):
            found.append(os.path.join(root, name))
sys.exit(0 if found else 1)
PY
  then
    note "$id" fail 1
    finish 1 "no labelled chapter mp3"
  fi
  mark_done "$id"
  note "$id" pass 0
}

step_version
step_selftest
step_dry_run
step_smoke_narrate
step_vram
if [[ "$MODE" == "full" ]]; then
  step_dialogue
  step_yo
  step_vxbook
  step_mp3_labels
fi
finish 0 "pass"
