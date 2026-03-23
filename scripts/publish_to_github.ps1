param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteUrl,
    [string]$Branch = "main"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if (-not (Test-Path ".git")) {
        git init
    }

    git add .
    git commit -m "Initial public reproducibility package" 2>$null

    $hasOrigin = git remote | Select-String -SimpleMatch "origin"
    if (-not $hasOrigin) {
        git remote add origin $RemoteUrl
    }

    git branch -M $Branch
    git push -u origin $Branch

    Write-Output "Repository published to $RemoteUrl on branch $Branch"
}
finally {
    Pop-Location
}
