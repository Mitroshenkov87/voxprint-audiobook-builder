# Builds the ONLINE installer from a PyInstaller --onedir folder:
#   1. voxprint-fetch.exe  (tools/online_fetch.py, standard library only)  -> build\online\
#   2. the payload zips + manifest-<channel>.json                           -> build\online-release\
#   3. voxprint-audiobook-builder-setup-<version>-win-x64.exe and the alias
#      voxprint-audiobook-builder-setup-win-x64.exe, each with a .sha256 -> build\online-release\
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
                                    #   the release gets only the shell + manifest-thin-<channel>.json; the installer uses the versioned setup name
    [switch]$Portable,              # compile the wizard page "Keep a portable setup folder" (installer\Voxprint.iss /DPORTABLE; not in the default build yet)
    [string]$MirrorBase = "",       # optional fallback address prefix for the locked files (our mirror)
    [string]$Build = $env:VOXPRINT_BUILD,      # CI build number (tools/build_number.py); written into the installer's version resource
    [string]$Codename = $env:VOXPRINT_CODENAME,
    [switch]$LegacyPayload,         # allow the OLD multi-GB payload parts for a real release (deleted on purpose; never needed now)
    [switch]$LocalAssets            # shell zip and sdist wheels are file names next to the setup exe; the manifest is embedded
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
# The fetch exe imports infra modules that use voxprint-common. Smoke jobs install only PyInstaller.
# A draft UI pin lives in runtime_lock.json "git". With no git pin, pip installs the released
# wheel and checks the sha256 fragment.
& $Python -c "import voxprint_common" 2>$null
if ($LASTEXITCODE -ne 0) {
    $lock = Get-Content -Raw -Encoding utf8 (Join-Path $RepoRoot "infra\runtime_lock.json") | ConvertFrom-Json
    $git = @()
    if ($null -ne $lock.git) { $git = @($lock.git) }
    if ($git.Count -gt 0) {
        foreach ($line in $git) {
            & $Python -m pip install --quiet $line
            Check "voxprint-common"
        }
    } else {
        $suite = $lock.suite
        if (-not $suite.url -or -not $suite.sha256) { throw "runtime_lock.json has no suite wheel pin" }
        & $Python -m pip install --quiet "voxprint-common @ $($suite.url)#sha256=$($suite.sha256)"
        Check "voxprint-common"
    }
}
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
if ($LocalAssets) { $pargs += "--local-assets" }
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
if ($LocalAssets) {
    Copy-Item "build\online-release\$manifest" "build\online-release\embedded-manifest.json" -Force
    $ManifestUrl = ""
    $LatestManifestUrl = ""
} else {
    if (-not $ManifestUrl) { $ManifestUrl = "https://github.com/$Repo/releases/download/$Tag/$manifest" }
    # the newest release's manifest: fallback of the pinned one and the "try the latest versions" option (local test builds: none)
    $LatestManifestUrl = if ($ManifestUrl -like "https://github.com/*") { "https://github.com/$Repo/releases/latest/download/$manifest" } else { "" }
}

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "$env:ProgramFiles\Inno Setup 6\ISCC.exe" }
$ver = (Get-Content credits.json -Raw | ConvertFrom-Json).app.version
$setupName = "voxprint-audiobook-builder-setup-$ver-win-x64.exe"
$aliasName = "voxprint-audiobook-builder-setup-win-x64.exe"
if ($LocalAssets) {
    $isArgs = @("/DONLINE", "/DONEDIR", "/DEMBEDMANIFEST")
} else {
    $isArgs = @("/DONLINE", "/DONEDIR", "/DManifestUrl=$ManifestUrl")
    if ($LatestManifestUrl) { $isArgs += "/DLatestManifestUrl=$LatestManifestUrl" }
}
if ($Thin) { $isArgs += "/DTHIN" }
if ($Portable) { $isArgs += "/DPORTABLE" }
if ($Build -match '^\d+$') { $isArgs += "/DAppBuild=$Build" }
if ($Codename -match '^[A-Za-z]+$') { $isArgs += "/DAppCodename=$Codename" }
& $iscc @isArgs installer\Voxprint.iss
Check "Inno Setup (online)"

$setupMb = (Get-Item "installer\Output\$setupName").Length / 1MB
if ($setupMb -gt 200) { throw "$setupName is $([int]$setupMb) MB (expected a few tens of MB)" }
Move-Item "installer\Output\$setupName" "build\online-release\$setupName" -Force
$h = (Get-FileHash "build\online-release\$setupName" -Algorithm SHA256).Hash.ToLower()
"$h  $setupName" | Out-File -Encoding ascii "build\online-release\$setupName.sha256"
Copy-Item "build\online-release\$setupName" "build\online-release\$aliasName" -Force
"$h  $aliasName" | Out-File -Encoding ascii "build\online-release\$aliasName.sha256"
Get-ChildItem build\online-release | Format-Table Name, Length
"manifest: $manifest   installer manifest url: $ManifestUrl"
