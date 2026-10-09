# Download Voxprint-Setup-online.exe from the official GitHub releases and start it.
# Read this file before you run it. The one-line form is:
#   irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-audiobook-builder/main/install.ps1 | iex
#
# The installer and its .sha256 file come only from
# https://github.com/Mitroshenkov87/voxprint-audiobook-builder/releases
# A real install refuses every other host. -SourceUrl is for a local dry run only.
# Requires administrator rights (a real install re-launches itself elevated).
# -Silent adds /VERYSILENT. -Version picks a tag; the default is the latest pre-release.
[CmdletBinding()]
param(
    [string]$Version = "",
    [switch]$Silent,
    [switch]$DryRun,
    [string]$SourceUrl = ""
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$OfficialRepo = "Mitroshenkov87/voxprint-audiobook-builder"
$AssetName = "Voxprint-Setup-online.exe"
$ScriptUrl = "https://raw.githubusercontent.com/$OfficialRepo/main/install.ps1"
$ReleaseApi = "https://api.github.com/repos/$OfficialRepo/releases"

function Test-IsAdmin {
    if ($env:OS -ne "Windows_NT") { return $false }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-OfficialAsset([string]$Url) {
    $pattern = '^https://github\.com/Mitroshenkov87/voxprint-audiobook-builder/releases/download/[^/]+/Voxprint-Setup-online\.exe(\.sha256)?$'
    if ($Url -notmatch $pattern) {
        throw "Refusing $Url. This script downloads only $AssetName and its .sha256 file from https://github.com/$OfficialRepo/releases"
    }
}

function Assert-LoopbackSource([string]$Url) {
    $parsed = $null
    if (-not [Uri]::TryCreate($Url, [UriKind]::Absolute, [ref]$parsed)) {
        throw "Dry-run -SourceUrl must be an absolute http://127.0.0.1 URL."
    }
    $hostOk = ($parsed.Host -eq "127.0.0.1" -or $parsed.Host -eq "localhost")
    $schemeOk = ($parsed.Scheme -eq "http" -or $parsed.Scheme -eq "https")
    if (-not $hostOk -or -not $schemeOk) {
        throw "Dry-run -SourceUrl must be http://127.0.0.1. A real install uses only the official GitHub releases."
    }
}

function Get-HashToken([string]$Text) {
    if ($Text -match '([A-Fa-f0-9]{64})') { return $Matches[1].ToLower() }
    throw "The .sha256 file does not contain a SHA-256 hash. The installer was not started."
}

function Get-OfficialUrls {
    $headers = @{
        "User-Agent" = "VoxprintInstall"
        "Accept"     = "application/vnd.github+json"
    }
    if ($Version) {
        $rel = Invoke-RestMethod -Headers $headers -Uri "$ReleaseApi/tags/$Version"
    } else {
        $rels = @(Invoke-RestMethod -Headers $headers -Uri "$($ReleaseApi)?per_page=20")
        $rel = $rels | Where-Object { $_.prerelease } | Select-Object -First 1
        if (-not $rel) {
            throw "No pre-release was found. Pass -Version with a tag from https://github.com/$OfficialRepo/releases"
        }
    }
    $exeUrl = ""
    $shaUrl = ""
    foreach ($asset in @($rel.assets)) {
        if ($asset.name -eq $AssetName) { $exeUrl = [string]$asset.browser_download_url }
        if ($asset.name -eq "$AssetName.sha256") { $shaUrl = [string]$asset.browser_download_url }
    }
    if (-not $exeUrl -or -not $shaUrl) {
        throw "Release $($rel.tag_name) does not publish $AssetName and its .sha256 file."
    }
    Assert-OfficialAsset $exeUrl
    Assert-OfficialAsset $shaUrl
    return @{ Exe = $exeUrl; Sha = $shaUrl; Tag = [string]$rel.tag_name }
}

function Receive-File([string]$Url, [string]$Dest) {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

function Install-Checked {
    param([string]$ExeUrl, [string]$ShaUrl)
    $root = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
    $dir = Join-Path $root ("voxprint-setup-" + [guid]::NewGuid().ToString("n"))
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $exe = Join-Path $dir $AssetName
    $shaFile = Join-Path $dir "$AssetName.sha256"
    try {
        Write-Host "Downloading $ExeUrl"
        Receive-File $ExeUrl $exe
        Write-Host "Downloading $ShaUrl"
        Receive-File $ShaUrl $shaFile
        $expected = Get-HashToken (Get-Content -Raw -Path $shaFile)
        $actual = (Get-FileHash -Algorithm SHA256 -Path $exe).Hash.ToLower()
        if ($actual -ne $expected) {
            throw "SHA-256 does not match the .sha256 file (got $actual, expected $expected). The installer was not started."
        }
        Write-Host "SHA-256 matches."
        if ($DryRun) {
            Write-Host "Dry run: the installer was not started."
            return
        }
        $argList = @()
        if ($Silent) { $argList = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") }
        $proc = Start-Process -FilePath $exe -ArgumentList $argList -Wait -PassThru
        if ($null -eq $proc.ExitCode -or $proc.ExitCode -ne 0) {
            throw "The installer exited with code $($proc.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Start-Elevated {
    $scriptPath = $PSCommandPath
    if (-not $scriptPath) {
        $root = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
        $scriptPath = Join-Path $root "voxprint-install.ps1"
        Write-Host "Downloading the installer script from $ScriptUrl"
        Receive-File $ScriptUrl $scriptPath
    }
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath)
    if ($Version) { $argList += @("-Version", $Version) }
    if ($Silent) { $argList += "-Silent" }
    $exe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $proc = Start-Process -FilePath $exe -ArgumentList $argList -Verb RunAs -Wait -PassThru
    if (-not $proc -or $null -eq $proc.ExitCode) {
        throw "Administrator permission was not granted. The installer was not started."
    }
    exit $proc.ExitCode
}

if ($SourceUrl -and -not $DryRun) {
    throw "-SourceUrl is only allowed with -DryRun. A real install downloads only from https://github.com/$OfficialRepo/releases"
}

if ($DryRun -and $SourceUrl) {
    Assert-LoopbackSource $SourceUrl
    $base = $SourceUrl.TrimEnd("/")
    Install-Checked -ExeUrl "$base/$AssetName" -ShaUrl "$base/$AssetName.sha256"
    return
}

if ($DryRun) {
    $urls = Get-OfficialUrls
    Write-Host "Dry run: would download $($urls.Tag)"
    Write-Host $urls.Exe
    Write-Host $urls.Sha
    return
}

if (-not (Test-IsAdmin)) {
    Start-Elevated
}

$urls = Get-OfficialUrls
Write-Host "Release $($urls.Tag)"
Install-Checked -ExeUrl $urls.Exe -ShaUrl $urls.Sha
