# Builds the ONLINE installer from a PyInstaller --onedir folder:
#   1. voxprint-fetch.exe  (tools/online_fetch.py, standard library only)  -> build\online\
#   2. the payload zips + manifest-<channel>.json                           -> build\online-release\
#   3. Voxprint-Setup-online.exe (installer\Voxprint.iss /DONLINE) + its .sha256 -> build\online-release\
# Used by .github/workflows/build-installer.yml (job "build" and the quick "online-smoke" job).
param(
    [string]$Dist = "dist\Voxprint",
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Repo = "Mitroshenkov87/voxprint-audiobook-builder",
    [string]$BaseUrl = "",          # where the payload zips will be served (default: the release assets of $Tag)
    [string]$ManifestUrl = "",      # baked into the installer (default: the manifest asset of the release $Tag)
    [int]$LimitMib = 1800,          # part size limit: a release asset must be below 2 GiB
    [string]$Python = ".venv\Scripts\python.exe",
    [switch]$Thin,                  # THIN installer: $Dist is the thin shell (build_thin.bat); the heavy libraries become runtime modules
    [string]$RuntimeSite = ""       # site-packages with those libraries (default: that of $Python's environment)
)
$ErrorActionPreference = "Stop"
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

New-Item -ItemType Directory -Force build\online, build\online-release | Out-Null
Remove-Item build\online-release\* -Force -ErrorAction SilentlyContinue

& $Python -m PyInstaller --onefile --console --name voxprint-fetch --distpath build\online --workpath build\online-work `
    --specpath build\online-work --noconfirm --log-level WARN --paths infra --hidden-import netroute `
    --exclude-module torch --exclude-module torchaudio --exclude-module transformers --exclude-module huggingface_hub --exclude-module requests `
    --exclude-module numpy --exclude-module scipy --exclude-module PySide6 --exclude-module psutil --exclude-module tkinter `
    tools\online_fetch.py
Check "PyInstaller (voxprint-fetch)"
# guard: the downloader must stay small (it once pulled PyTorch in through an optional import: 3 GB installer)
$fetchMb = (Get-Item build\online\voxprint-fetch.exe).Length / 1MB
"voxprint-fetch.exe: {0:N1} MB" -f $fetchMb
if ($fetchMb -gt 60) { throw "voxprint-fetch.exe is $([int]$fetchMb) MB - an import pulled heavy packages into it" }

$pargs = @("tools\make_online_payload.py", "--dist", $Dist, "--out", "build\online-release", "--tag", $Tag, "--repo", $Repo,
           "--limit-mib", "$LimitMib")
if ($BaseUrl) { $pargs += @("--base-url", $BaseUrl) }
if ($Thin) {
    if (-not $RuntimeSite) { $RuntimeSite = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim() }
    $pargs += @("--runtime-site", $RuntimeSite)
}
& $Python @pargs
Check "make_online_payload"

$channel = if ($Tag -match '^[vV]?[0-9][0-9.]*-') { "beta" } else { "stable" }
$manifest = "manifest-$channel.json"
if (-not (Test-Path "build\online-release\$manifest")) { throw "$manifest was not written" }
if (-not $ManifestUrl) { $ManifestUrl = "https://github.com/$Repo/releases/download/$Tag/$manifest" }

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "$env:ProgramFiles\Inno Setup 6\ISCC.exe" }
$setupName = if ($Thin) { "Voxprint-Setup-thin.exe" } else { "Voxprint-Setup-online.exe" }
$isArgs = @("/DONLINE", "/DONEDIR", "/DManifestUrl=$ManifestUrl")
if ($Thin) { $isArgs += "/DTHIN" }
& $iscc @isArgs installer\Voxprint.iss
Check "Inno Setup (online)"

$setupMb = (Get-Item "installer\Output\$setupName").Length / 1MB
if ($setupMb -gt 200) { throw "$setupName is $([int]$setupMb) MB (expected a few tens of MB)" }
Move-Item "installer\Output\$setupName" "build\online-release\$setupName" -Force
$h = (Get-FileHash "build\online-release\$setupName" -Algorithm SHA256).Hash.ToLower()
"$h  $setupName" | Out-File -Encoding ascii "build\online-release\$setupName.sha256"
Get-ChildItem build\online-release | Format-Table Name, Length
"manifest: $manifest   installer manifest url: $ManifestUrl"
