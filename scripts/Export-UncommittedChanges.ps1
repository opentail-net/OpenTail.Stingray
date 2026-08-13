<#
.SYNOPSIS
    Exports all uncommitted files (staged, unstaged, untracked) into a single aggregated log file
    with clear header comment lines for external AI code review.

.PARAMETER OutputPath
    Target output log file path. Defaults to 'docs/uncommitted-changes.log'.
#>
[CmdletBinding()]
param (
    [string]$OutputPath = "docs/uncommitted-changes.log"
)

$ErrorActionPreference = "Stop"

# Get repository root
$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    Write-Error "Not inside a git repository."
    exit 1
}

Set-Location $repoRoot

# Ensure output directory exists
$fullOutputPath = [System.IO.Path]::Combine($repoRoot, $OutputPath)
$outputDir = [System.IO.Path]::GetDirectoryName($fullOutputPath)
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Collect modified and untracked files from git status
$gitFiles = & git status --porcelain
if (-not $gitFiles) {
    Write-Host "No uncommitted changes found."
    exit 0
}

$fileList = @()
foreach ($line in $gitFiles) {
    if ($line.Length -ge 4) {
        $filePath = $line.Substring(3).Trim().Trim('"')
        # Skip the output file itself if present
        $normalizedRel = $filePath -replace '\\','/'
        $normalizedOut = $OutputPath -replace '\\','/'
        if ($normalizedRel -ne $normalizedOut) {
            if (Test-Path $filePath -PathType Leaf) {
                $fileList += $filePath
            }
        }
    }
}

$fileList = $fileList | Select-Object -Unique | Sort-Object

Write-Host "Found $($fileList.Count) uncommitted files to export."

$builder = [System.Text.StringBuilder]::new()

$header = @"
// ============================================================================
// OpenTail.Stingray — Aggregated Uncommitted Changes Export
// Generated At: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss K")
// Total Files: $($fileList.Count)
// ============================================================================

"@
[void]$builder.AppendLine($header)

foreach ($relPath in $fileList) {
    Write-Host "Exporting: $relPath"
    
    $banner = @"
// ============================================================================
// File: $relPath
// ============================================================================
"@
    [void]$builder.AppendLine($banner)
    
    $content = Get-Content -Path $relPath -Raw -ErrorAction SilentlyContinue
    if ($content) {
        [void]$builder.AppendLine($content)
    } else {
        [void]$builder.AppendLine("// (Empty or unreadable file)")
    }
    [void]$builder.AppendLine("")
}

[System.IO.File]::WriteAllText($fullOutputPath, $builder.ToString(), [System.Text.Encoding]::UTF8)

Write-Host "Successfully exported $($fileList.Count) files to $fullOutputPath"
