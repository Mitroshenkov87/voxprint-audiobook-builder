# Builds the ONLINE installer from a PyInstaller --onedir folder:
#   1. voxprint-fetch.exe  (tools/online_fetch.py, standard library only)  -> build\online\
#   2. the payload zips + manifest-<channel>.json                           -> build\online-release\
#   3. Voxprint-Setup-online.exe (installer\Voxprint.iss /DONLINE) + its .sha256 -> build\online-release\
# Used by .github/workflows/build-installer.yml (job "build-thin" with -RuntimeLock, and the quick smoke jobs with a local -BaseUrl).
param(
    [string]$Dist = "dist\Voxprint",
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Repo = "Mitroshenkov87/voxprint-audiobook-builder",
    [string]$BaseUrl = "",          # where the payload zips will be served (default: the release assets of $Tag)
    [string]$ManifestUrl = "",      # baked into the installer (default: the manifest asset of the release $Tag)
    [int]$LimitMib = 1800,          # part size limit: a release asset must be below 2 GiB
    [string]$Python = ".venv\Scripts\python.exe",
    [switch]$Thin,                  # THIN installer: $Dist is the thin shell (build_thin.bat); the heavy libraries become runtime modules
    [string]$RuntimeSite = "",      # site-packages with those libraries (default: that of $Python's environment)
    [string]$RuntimeLock = "",      # THIN v2: infra\runtime_lock.json - the libraries stay at their upstream sites (PyPI, download.pytorch.org);
                                    #   the release gets only the shell + manifest-thin-<channel>.json; the installer is named Voxprint-Setup-online.exe
    [switch]$Portable,              # compile the wizard page "Keep a portable setup folder" (installer\Voxprint.iss /DPORTABLE; not in the default build yet)
    [string]$MirrorBase = "",       # optional fallback address prefix for the locked files (our mirror)
    [string]$Build = $env:VOXPRINT_BUILD,      # CI build number (tools/build_number.py); also in the file name of a copy of the installer
    [string]$Codename = $env:VOXPRINT_CODENAME,
    [switch]$LegacyPayload          # allow the OLD multi-GB payload parts for a real release (deleted on purpose; never needed now)
)
$ErrorActionPreference = "Stop"
function Check([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

# Guard: without -RuntimeLock this script packs the whole program into multi-GB payload zips.  That is allowed only for local
# test servers (the smoke jobs use -BaseUrl http://127.0.0.1:...), never for release assets, unless -LegacyPayload is given.
if (-not $RuntimeLock -and -not $LegacyPayload -and ($BaseUrl -notmatch '^(https?://(127\.0\.0\.1|localhost)[:/]|file:)')) {
    throw "Refusing to build multi-GB payload parts for a release: use -RuntimeLock infra\runtime_lock.json (the online installer)"
}

New-Item -ItemType Directory -Force build\online, build\online-release | Out-Null
Remove-Item build\online-release\* -Force -ErrorAction SilentlyContinue

# PyInstaller 6.x resolves relative --add-data / --paths sources against --specpath (build\online-work), not the
# current directory, so pass absolute paths based on the repo root (the parent of this script's folder).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$infraDir = Join-Path $RepoRoot "infra"
& $Python -m PyInstaller --onefile --console --name voxprint-fetch --distpath build\online --workpath build\online-work `
    --specpath build\online-work --noconfirm --log-level WARN --paths "$infraDir" --hidden-import netroute `
    --hidden-import infra.portable --hidden-import infra.model_release --hidden-import infra.model_mirrors `
    --hidden-import infra.modelscope_mirror --hidden-import infra.download_watch --hidden-import infra.net --hidden-import infra.paths `
    --add-data "$infraDir\model_mirrors.json;infra" --add-data "$infraDir\model_release.json;infra" `
    --exclude-module torch --exclude-module torchaudio --exclude-module transformers --exclude-module huggingface_hub --exclude-module requests `
    --exclude-module numpy --exclude-module scipy --exclude-module PySide6 --exclude-module psutil --exclude-module tkinter `
    (Join-Path $RepoRoot "tools\online_fetch.py")
Check "PyInstaller (voxprint-fetch)"
# guard: the downloader must stay small (it once pulled PyTorch in through an optional import: 3 GB installer)
$fetchMb = (Get-Item build\online\voxprint-fetch.exe).Length / 1MB
"voxprint-fetch.exe: {0:N1} MB" -f $fetchMb
if ($fetchMb -gt 60) { throw "voxprint-fetch.exe is $([int]$fetchMb) MB - an import pulled heavy packages into it" }

$pargs = @("tools\make_online_payload.py", "--dist", $Dist, "--out", "build\online-release", "--tag", $Tag, "--repo", $Repo,
           "--limit-mib", "$LimitMib")
if ($BaseUrl) { $pargs += @("--base-url", $BaseUrl) }
if ($RuntimeLock) {
    $Thin = $true
    $pargs += @("--runtime-lock", $RuntimeLock)
    if ($MirrorBase) { $pargs += @("--mirror-base", $MirrorBase) }
} elseif ($Thin) {
    if (-not $RuntimeSite) { $RuntimeSite = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim() }
    $pargs += @("--runtime-site", $RuntimeSite)
}
& $Python @pargs
Check "make_online_payload"

$channel = if ($Tag -match '^[vV]?[0-9][0-9.]*-') { "beta" } else { "stable" }
$manifest = if ($RuntimeLock) { "manifest-thin-$channel.json" } else { "manifest-$channel.json" }
if (-not (Test-Path "build\online-release\$manifest")) { throw "$manifest was not written" }
if (-not $ManifestUrl) { $ManifestUrl = "https://github.com/$Repo/releases/download/$Tag/$manifest" }
# the newest release's manifest: fallback of the pinned one and the "try the latest versions" option (local test builds: none)
$LatestManifestUrl = if ($ManifestUrl -like "https://github.com/*") { "https://github.com/$Repo/releases/latest/download/$manifest" } else { "" }

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "$env:ProgramFiles\Inno Setup 6\ISCC.exe" }
$setupName = if ($Thin) { "Voxprint-Setup-thin.exe" } else { "Voxprint-Setup-online.exe" }
$finalName = if ($RuntimeLock) { "Voxprint-Setup-online.exe" } else { $setupName }
$isArgs = @("/DONLINE", "/DONEDIR", "/DManifestUrl=$ManifestUrl")
if ($LatestManifestUrl) { $isArgs += "/DLatestManifestUrl=$LatestManifestUrl" }
if ($Thin) { $isArgs += "/DTHIN" }
if ($Portable) { $isArgs += "/DPORTABLE" }
if ($Build -match '^\d+$') { $isArgs += "/DAppBuild=$Build" }
if ($Codename -match '^[A-Za-z]+$') { $isArgs += "/DAppCodename=$Codename" }
& $iscc @isArgs installer\Voxprint.iss
Check "Inno Setup (online)"

$setupMb = (Get-Item "installer\Output\$setupName").Length / 1MB
if ($setupMb -gt 200) { throw "$setupName is $([int]$setupMb) MB (expected a few tens of MB)" }
Move-Item "installer\Output\$setupName" "build\online-release\$finalName" -Force
$h = (Get-FileHash "build\online-release\$finalName" -Algorithm SHA256).Hash.ToLower()
"$h  $finalName" | Out-File -Encoding ascii "build\online-release\$finalName.sha256"
if ($Build -match '^\d+$' -and [int]$Build -gt 0) {
    # the same installer under a name with the version and build number (the plain name stays for stable links)
    $ver = (Get-Content credits.json -Raw | ConvertFrom-Json).app.version
    $numbered = "Voxprint-Setup-online-$ver-build$Build.exe"
    Copy-Item "build\online-release\$finalName" "build\online-release\$numbered" -Force
    "$h  $numbered" | Out-File -Encoding ascii "build\online-release\$numbered.sha256"
}
Get-ChildItem build\online-release | Format-Table Name, Length
"manifest: $manifest   installer manifest url: $ManifestUrl"
