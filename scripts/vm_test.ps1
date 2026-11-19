# Non-interactive GPU check for an installed or source-tree Voxprint. Windows twin of scripts/vm_test.sh.
#
#   powershell -File scripts/vm_test.ps1 --smoke --json --tag v1.0.0-rc.3
#   powershell -File scripts/vm_test.ps1 --full --json --results .\vm-results\audiobook-builder\v1.0.0-rc.3
#
# Exit codes: 0 pass, 1 a test failed, 2 setup error, 3 no NVIDIA GPU.
# Successful steps leave a marker in $env:USERPROFILE\.done\<step> (or $env:VOXPRINT_VM_DONE).
# The model download at the start always runs and does not write a marker.
# Results land in .\vm-results\audiobook-builder\<tag>\ unless --results or $env:VOXPRINT_VM_RESULTS is set.
# Sync that folder to /workspace/vm-results/audiobook-builder/<tag>/.

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$WantJson = $false
$SmokeFlag = $false
$FullFlag = $false
$TagOpt = ""
$ResultsOpt = ""
$BinOpt = ""

function Die-Usage([string]$Message) {
    Write-Error $Message
    exit 2
}

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--smoke" { $SmokeFlag = $true }
        "--full" { $FullFlag = $true }
        "--json" { $WantJson = $true }
        "--tag" {
            if ($i + 1 -ge $args.Count) { Die-Usage "missing value for --tag" }
            $i++; $TagOpt = $args[$i]
        }
        "--results" {
            if ($i + 1 -ge $args.Count) { Die-Usage "missing value for --results" }
            $i++; $ResultsOpt = $args[$i]
        }
        "--bin" {
            if ($i + 1 -ge $args.Count) { Die-Usage "missing value for --bin" }
            $i++; $BinOpt = $args[$i]
        }
        default { Die-Usage "unknown argument: $($args[$i])" }
    }
}

if ($FullFlag) { $Mode = "full" }
elseif ($SmokeFlag) { $Mode = "smoke" }
else { Die-Usage "pass --smoke or --full" }

function Find-Python {
    foreach ($name in @("python", "py", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

$Python = Find-Python
if (-not $Python) { Die-Usage "python was not found" }

$Vox = @()
if ($BinOpt) {
    if (-not (Test-Path -LiteralPath $BinOpt)) { Die-Usage "voxprint was not found (set --bin or VOXPRINT_BIN)" }
    $Vox = @($BinOpt)
} elseif ($env:VOXPRINT_BIN) {
    if (-not (Test-Path -LiteralPath $env:VOXPRINT_BIN)) { Die-Usage "voxprint was not found (set --bin or VOXPRINT_BIN)" }
    $Vox = @($env:VOXPRINT_BIN)
} else {
    $known = Join-Path $env:ProgramFiles "VOXPrint AI Suite\Audiobook Builder\voxprint-audiobook-cli.exe"
    if (Test-Path -LiteralPath $known) { $Vox = @($known) }
    else {
    foreach ($name in @("voxprint-audiobook-cli", "voxprint-cli", "voxprint")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $Vox = @($cmd.Source); break }
    }
    }
    if (-not $Vox -and (Test-Path -LiteralPath (Join-Path $Root "main.py"))) { $Vox = @($Python, (Join-Path $Root "main.py")) }
    if (-not $Vox) { Die-Usage "voxprint was not found (set --bin or VOXPRINT_BIN)" }
}

function Sanitize-Tag([string]$Raw) {
    return ($Raw -replace '[\\/ ]', '-')
}

$Tag = if ($TagOpt) { Sanitize-Tag $TagOpt } else { "unknown" }
if ($ResultsOpt) { $Results = $ResultsOpt }
elseif ($env:VOXPRINT_VM_RESULTS) { $Results = $env:VOXPRINT_VM_RESULTS }
else { $Results = Join-Path (Join-Path (Join-Path (Get-Location) "vm-results") "audiobook-builder") $Tag }

if ($env:VOXPRINT_VM_DONE) { $Done = $env:VOXPRINT_VM_DONE }
else { $Done = Join-Path $env:USERPROFILE ".done" }
$Work = Join-Path $Results "work"
$Peak = Join-Path $Results "vram.peak"
New-Item -ItemType Directory -Force -Path $Results, $Done, $Work | Out-Null

function Invoke-Vox([string[]]$More) {
    $tail = @()
    if ($script:Vox.Count -gt 1) { $tail = @($script:Vox[1..($script:Vox.Count - 1)]) }
    & $script:Vox[0] @($tail + $More)
    return $LASTEXITCODE
}

function Write-Utf8([string]$Path, [string]$Text) {
    $enc = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Write-Note([string]$Id, [string]$Status, [int]$Rc) {
    $line = & $Python -c "import json,sys; print(json.dumps({'type':'step','id':sys.argv[1],'status':sys.argv[2],'exit':int(sys.argv[3])}, ensure_ascii=False))" $Id $Status $Rc
    $enc = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText((Join-Path $Results "steps.jsonl"), ($line + "`n"), $enc)
    if ($WantJson) { Write-Output $line } else { Write-Output "$Id $Status" }
}

function Finish([int]$Code, [string]$Reason) {
    $env:CODE = "$Code"
    $env:REASON = $Reason
    $env:MODE = $Mode
    $env:TAG = $Tag
    $env:RESULTS = $Results
    $env:PEAK_FILE = $Peak
    $env:WANT_JSON = $(if ($WantJson) { "1" } else { "0" })
    & $Python -c @"
import json, os
from pathlib import Path
results = Path(os.environ['RESULTS'])
steps = []
log = results / 'steps.jsonl'
if log.is_file():
    for line in log.read_text(encoding='utf-8').splitlines():
        if line.strip():
            steps.append(json.loads(line))
peak_raw = ''
peak_path = Path(os.environ['PEAK_FILE'])
if peak_path.is_file():
    peak_raw = ''.join(ch for ch in peak_path.read_text(encoding='utf-8') if ch.isdigit())
payload = {
    'ok': os.environ['CODE'] == '0',
    'exit': int(os.environ['CODE']),
    'mode': os.environ['MODE'],
    'tag': os.environ['TAG'],
    'reason': os.environ['REASON'],
    'steps': steps,
    'peak_vram_mib': int(peak_raw) if peak_raw else None,
    'results': str(results),
}
text = json.dumps(payload, ensure_ascii=False)
results.mkdir(parents=True, exist_ok=True)
(results / 'result.json').write_text(text + '\n', encoding='utf-8')
print(text if os.environ['WANT_JSON'] == '1' else (os.environ['REASON'] or ('pass' if payload['ok'] else 'fail')))
"@
    exit $Code
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Finish 3 "no GPU" }
& nvidia-smi *> $null
if ($LASTEXITCODE -ne 0) { Finish 3 "no GPU" }

function Sample-Vram {
    $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    $digits = ([string]$raw) -replace '\D', ''
    if (-not $digits) { return }
    $cur = ""
    if (Test-Path -LiteralPath $Peak) {
        $cur = ([IO.File]::ReadAllText($Peak)) -replace '\D', ''
    }
    if (-not $cur -or [int]$digits -gt [int]$cur) {
        Set-Content -LiteralPath $Peak -Value $digits -Encoding ascii
    }
}

function Test-Skipped([string]$Id) { Test-Path -LiteralPath (Join-Path $Done $Id) }
function Mark-Done([string]$Id) {
    Set-Content -LiteralPath (Join-Path $Done $Id) -Value ((Get-Date).ToUniversalTime().ToString("o")) -Encoding ascii
}

function Get-AppManifest {
    $beside = if ($script:Vox.Count -eq 1) {
        Join-Path (Split-Path -Parent $script:Vox[0]) "voxprint-app-models.json"
    } else {
        Join-Path $Root "voxprint-app-models.json"
    }
    if (Test-Path -LiteralPath $beside) { return $beside }
    return (Join-Path $Root "voxprint-app-models.json")
}

function Step-Download {
    # Unconditional. mdl.ready() may trust a status file; download refills gaps and is not skipped.
    $manifest = Get-AppManifest
    $log = Join-Path $Results "download.log"
    $err = Join-Path $Results "download.err"
    Invoke-Vox @(
        "--run-mdl", "download", "--app", "audiobook-builder",
        "--manifest", $manifest, "--json"
    ) 1> $log 2> $err
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { Write-Note "download" "fail" $rc; Finish 1 "model download failed" }
    Write-Note "download" "pass" 0
}

function Step-Version {
    if (Test-Skipped "version") { Write-Note "version" "skipped" 0; return }
    $log = Join-Path $Results "version.log"
    $err = Join-Path $Results "version.err"
    Invoke-Vox @("--json", "--version") 1> $log 2> $err
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        Invoke-Vox @("--version") 1> $log 2> $err
        $rc = $LASTEXITCODE
    }
    if ($rc -ne 0) { Write-Note "version" "fail" $rc; Finish 1 "version failed" }
    if (-not $TagOpt) {
        $env:VERLOG = $log
        $discovered = & $Python -c @"
import json, os, re
text = open(os.environ['VERLOG'], encoding='utf-8', errors='replace').read()
ver = ''
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        match = re.search(r'\b(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)', line)
        if match:
            ver = match.group(1)
        continue
    if isinstance(obj, dict) and obj.get('version'):
        ver = str(obj['version'])
if ver and not ver.startswith('v'):
    ver = 'v' + ver
print(ver)
"@
        if ($discovered) { $script:Tag = Sanitize-Tag $discovered }
    }
    Mark-Done "version"
    Write-Note "version" "pass" 0
}

function Step-Selftest {
    if (Test-Skipped "selftest") { Write-Note "selftest" "skipped" 0; return }
    Invoke-Vox @("--selftest") 1> (Join-Path $Results "selftest.log") 2> (Join-Path $Results "selftest.err")
    if ($LASTEXITCODE -ne 0) { Write-Note "selftest" "fail" $LASTEXITCODE; Finish 1 "selftest failed" }
    Mark-Done "selftest"
    Write-Note "selftest" "pass" 0
}

function Step-DryRun {
    if (Test-Skipped "dry-run") { Write-Note "dry-run" "skipped" 0; return }
    New-Item -ItemType Directory -Force -Path (Join-Path $Work "dry") | Out-Null
    $book = Join-Path $Work "smoke.txt"
    Write-Utf8 $book "Привет. Это короткая проверка голоса."
    Invoke-Vox @("--dry-run", "--json", "narrate", $book, "--voice", "levi", "--out", (Join-Path $Work "dry\narration")) `
        1> (Join-Path $Results "dry-narrate.log") 2> (Join-Path $Results "dry-narrate.err")
    $rc1 = $LASTEXITCODE
    Invoke-Vox @("--dry-run", "--json", "prepare", $book, "--out", (Join-Path $Work "dry\prepared.txt"), "--language", "ru") `
        1> (Join-Path $Results "dry-prepare.log") 2> (Join-Path $Results "dry-prepare.err")
    $rc2 = $LASTEXITCODE
    if ($rc1 -ne 0 -or $rc2 -ne 0) { Write-Note "dry-run" "fail" 1; Finish 1 "dry-run failed" }
    foreach ($logName in @("dry-narrate.log", "dry-prepare.log")) {
        $env:DRYLOG = Join-Path $Results $logName
        & $Python -c @"
import json, os, sys
lines = [ln for ln in open(os.environ['DRYLOG'], encoding='utf-8').read().splitlines() if ln.strip()]
objs = [json.loads(ln) for ln in lines]
result = next(obj for obj in reversed(objs) if obj.get('type') == 'result')
durations = result.get('durations') or {}
bad = durations.get('audio_s') is not None or result.get('outputs') != [] or result.get('dry_run') is not True
sys.exit(1 if bad or not lines else 0)
"@
        if ($LASTEXITCODE -ne 0) { Write-Note "dry-run" "fail" 1; Finish 1 "dry-run JSON was not a result with null audio" }
    }
    Mark-Done "dry-run"
    Write-Note "dry-run" "pass" 0
}

function Step-SmokeNarrate {
    if (Test-Skipped "smoke-narrate") { Write-Note "smoke-narrate" "skipped" 0; return }
    $out = Join-Path $Work "smoke-out"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $book = Join-Path $Work "smoke.txt"
    Write-Utf8 $book "Привет. Это короткая проверка голоса."
    Sample-Vram
    $poll = Start-Job -ScriptBlock {
        param($PeakPath)
        while ($true) {
            $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            $digits = ([string]$raw) -replace '\D', ''
            if ($digits) { Set-Content -LiteralPath $PeakPath -Value $digits -Encoding ascii }
            Start-Sleep -Seconds 2
        }
    } -ArgumentList $Peak
    Invoke-Vox @("narrate", $book, "--voice", "levi", "--format", "mp3", "--json", "--out", $out) `
        1> (Join-Path $Results "smoke-narrate.log") 2> (Join-Path $Results "smoke-narrate.err")
    $rc = $LASTEXITCODE
    Stop-Job $poll -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $poll -Force -ErrorAction SilentlyContinue | Out-Null
    Sample-Vram
    if ($rc -ne 0) { Write-Note "smoke-narrate" "fail" $rc; Finish 1 "smoke narration failed" }
    $env:NLOG = Join-Path $Results "smoke-narrate.log"
    & $Python -c @"
import json, os, sys
lines = [ln for ln in open(os.environ['NLOG'], encoding='utf-8').read().splitlines() if ln.strip()]
objs = [json.loads(ln) for ln in lines]
result = next(obj for obj in reversed(objs) if obj.get('type') == 'result')
audio = (result.get('durations') or {}).get('audio_s')
sys.exit(0 if isinstance(audio, (int, float)) else 1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Note "smoke-narrate" "fail" 1; Finish 1 "smoke narration did not report audio_s" }
    Mark-Done "smoke-narrate"
    Write-Note "smoke-narrate" "pass" 0
}

function Step-Vram {
    if (Test-Skipped "vram") { Write-Note "vram" "skipped" 0; return }
    Sample-Vram
    if (-not (Test-Path -LiteralPath $Peak)) { Write-Note "vram" "fail" 1; Finish 1 "nvidia-smi did not report memory.used" }
    $n = ([IO.File]::ReadAllText($Peak)) -replace '\D', ''
    if (-not $n) { Write-Note "vram" "fail" 1; Finish 1 "nvidia-smi did not report memory.used" }
    Set-Content -LiteralPath (Join-Path $Results "vram.txt") -Value $n -Encoding ascii
    Mark-Done "vram"
    Write-Note "vram" "pass" 0
}

function Step-Dialogue {
    if (Test-Skipped "dialogue") { Write-Note "dialogue" "skipped" 0; return }
    $book = Join-Path $Work "dialogue.txt"
    $marks = Join-Path $Work "marks.txt"
    $out = Join-Path $Work "dialogue-out"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Utf8 $book "Иван вошёл в комнату.`n`n— Здравствуй, — сказал Иван.`n`n— Здравствуй, — ответила Анна.`n"
    Invoke-Vox @("speakers", $book, "--out", $marks, "--json") 1> (Join-Path $Results "speakers.log") 2> (Join-Path $Results "speakers.err")
    if ($LASTEXITCODE -ne 0) { Write-Note "dialogue" "fail" $LASTEXITCODE; Finish 1 "speakers failed" }
    $env:MARKS = $marks
    & $Python -c @"
import os, sys
text = open(os.environ['MARKS'], encoding='utf-8').read().casefold()
sys.exit(0 if any(tok in text for tok in ('male', 'female', 'муж', 'жен')) else 1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Note "dialogue" "fail" 1; Finish 1 "speaker marks have no second voice" }
    Invoke-Vox @("narrate", $book, "--voice", "levi", "--male-voice", "natan", "--male2-voice", "shimon",
        "--female-voice", "miriam", "--speaker-marks", $marks, "--format", "mp3", "--json", "--out", $out) `
        1> (Join-Path $Results "dialogue.log") 2> (Join-Path $Results "dialogue.err")
    if ($LASTEXITCODE -ne 0) { Write-Note "dialogue" "fail" $LASTEXITCODE; Finish 1 "dialogue narration failed" }
    Mark-Done "dialogue"
    Write-Note "dialogue" "pass" 0
}

function Step-Yo {
    if (Test-Skipped "yo") { Write-Note "yo" "skipped" 0; return }
    $src = Join-Path $Work "yo-src.txt"
    $out = Join-Path $Work "yo-only.txt"
    Write-Utf8 $src "Еще одно слово в этом предложении."
    Invoke-Vox @("prepare", $src, "--out", $out, "--steps", "yo", "--language", "ru") `
        1> (Join-Path $Results "yo.log") 2> (Join-Path $Results "yo.err")
    if ($LASTEXITCODE -ne 0) { Write-Note "yo" "fail" $LASTEXITCODE; Finish 1 "yo prepare failed" }
    $env:YOOUT = $out
    & $Python -c @"
import os, sys
text = open(os.environ['YOOUT'], encoding='utf-8').read()
sys.exit(0 if ('\u0451' in text or '\u0401' in text) else 1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Note "yo" "fail" 1; Finish 1 "prepared text has no yo" }
    Mark-Done "yo"
    Write-Note "yo" "pass" 0
}

function Step-Vxbook {
    if (Test-Skipped "vxbook") { Write-Note "vxbook" "skipped" 0; return }
    $book = Join-Path $Work "story.vxbook"
    $out = Join-Path $Work "vxbook-out"
    & $Python (Join-Path $Root "scripts\vm_vxbook.py") $book
    if ($LASTEXITCODE -ne 0) { Write-Note "vxbook" "fail" 2; Finish 2 "could not write the vxbook" }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Invoke-Vox @("narrate", $book, "--voice", "levi", "--format", "mp3", "--no-soundscape", "--json", "--out", $out) `
        1> (Join-Path $Results "vxbook.log") 2> (Join-Path $Results "vxbook.err")
    if ($LASTEXITCODE -ne 0) { Write-Note "vxbook" "fail" $LASTEXITCODE; Finish 1 "vxbook narration failed" }
    Mark-Done "vxbook"
    Write-Note "vxbook" "pass" 0
}

function Step-Mp3Labels {
    if (Test-Skipped "mp3-labels") { Write-Note "mp3-labels" "skipped" 0; return }
    $env:VXOUT = Join-Path $Work "vxbook-out"
    & $Python -c @"
import os, re, sys
pat = re.compile(r'^\d{2,} - .+\.mp3$', re.IGNORECASE)
found = []
for root, _dirs, files in os.walk(os.environ['VXOUT']):
    for name in files:
        if pat.match(name):
            found.append(name)
sys.exit(0 if found else 1)
"@
    if ($LASTEXITCODE -ne 0) { Write-Note "mp3-labels" "fail" 1; Finish 1 "no labelled chapter mp3" }
    Mark-Done "mp3-labels"
    Write-Note "mp3-labels" "pass" 0
}

Step-Download
Step-Version
Step-Selftest
Step-DryRun
Step-SmokeNarrate
Step-Vram
if ($Mode -eq "full") {
    Step-Dialogue
    Step-Yo
    Step-Vxbook
    Step-Mp3Labels
}
Finish 0 "pass"
