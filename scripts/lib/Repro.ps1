# Reproducibility header for benchmark scripts (plan 2, P9).
#
# Dot-source from a script in scripts/:
#     . (Join-Path $PSScriptRoot 'lib/Repro.ps1')
#
# Get-ModelIdentity   size and modified time always; SHA-256 only with -Hash, computed at most once per
#                     (path, size, mtime) per experiment and kept in -CacheFile. A 12 GB checkpoint is never
#                     rehashed per run.
# Get-ReproHeader     the Markdown lines a benchmark writes at the top of its results file: commit and dirty state,
#                     binary and build configuration, backend and GPU names, settings, model identities, run count, date.

$script:ReproHashMemo = @{}

function Get-ModelIdentity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$Hash,
        [string]$CacheFile = ''
    )

    $fi = Get-Item -LiteralPath $Path -ErrorAction Stop
    $key = '{0}|{1}|{2}' -f $fi.FullName, $fi.Length, $fi.LastWriteTimeUtc.Ticks
    $sha = $null
    $source = 'not hashed'

    if ($Hash) {
        if ($script:ReproHashMemo.ContainsKey($key)) {
            $sha = $script:ReproHashMemo[$key]; $source = 'memory'
        }
        if (-not $sha -and $CacheFile -and (Test-Path -LiteralPath $CacheFile)) {
            $cache = Get-Content -LiteralPath $CacheFile -Raw | ConvertFrom-Json -AsHashtable
            if ($cache -and $cache.ContainsKey($key)) { $sha = [string]$cache[$key]; $source = 'cache file' }
        }
        if (-not $sha) {
            $sha = (Get-FileHash -LiteralPath $fi.FullName -Algorithm SHA256).Hash
            $source = 'computed now'
            if ($CacheFile) {
                $cache = @{}
                if (Test-Path -LiteralPath $CacheFile) {
                    $cache = Get-Content -LiteralPath $CacheFile -Raw | ConvertFrom-Json -AsHashtable
                }
                $cache[$key] = $sha
                $cache | ConvertTo-Json | Set-Content -LiteralPath $CacheFile -Encoding utf8
            }
        }
        $script:ReproHashMemo[$key] = $sha
    }

    [pscustomobject]@{
        File     = $fi.Name
        SizeGB   = [Math]::Round($fi.Length / 1GB, 2)
        Modified = $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
        Sha256   = $sha
        HashSource = $source
    }
}

function Get-ReproHeader {
    [CmdletBinding()]
    param(
        [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
        [string]$BinaryPath = '',
        [string]$Backend = '',
        [string[]]$ModelPaths = @(),
        [hashtable]$Settings = @{},
        [int]$RunCount = 0,
        [switch]$Hash,
        [string]$CacheFile = ''
    )

    $lines = New-Object System.Collections.Generic.List[string]

    # Commit and dirty state. Dirty lists the changed paths, so two runs from one commit with different
    # uncommitted edits are distinguishable.
    $commit = (git -C $RepoRoot rev-parse --short HEAD 2>$null)
    $status = @(git -C $RepoRoot status --porcelain 2>$null)
    $dirty = if ($status.Count -gt 0) { "dirty: $($status.Count) path(s)" } else { 'clean' }
    $lines.Add("- **Commit:** ``$commit`` ($dirty)")

    # Binary and build configuration, taken from the file that actually runs.
    if ($BinaryPath) {
        $bi = Get-Item -LiteralPath $BinaryPath -ErrorAction SilentlyContinue
        if ($bi) {
            $config = if ($bi.FullName -match '[\\/]bin[\\/](Release|Debug)[\\/]') { $Matches[1] } else { 'unknown' }
            $lines.Add("- **Binary:** ``$($bi.Name)`` ($config), $([Math]::Round($bi.Length / 1MB, 1)) MB, built $($bi.LastWriteTime.ToString('yyyy-MM-dd HH:mm')), version $($bi.VersionInfo.ProductVersion)")
        } else {
            $lines.Add("- **Binary:** NOT FOUND at ``$BinaryPath``")
        }
    }

    # Backend and device names. The iGPU caveat (CLAUDE.md rule 13) is printed with the names, not implied.
    $gpus = try { (Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name }) -join '; ' } catch { 'unavailable' }
    $lines.Add("- **Backend:** $(if ($Backend) { $Backend } else { 'not set' }); **GPUs visible to Windows:** $gpus")

    # Settings: the caller's generation and context settings, sorted for stable diffs.
    if ($Settings.Count -gt 0) {
        $pairs = $Settings.GetEnumerator() | Sort-Object Key | ForEach-Object { "$($_.Key)=$($_.Value)" }
        $lines.Add("- **Settings:** $($pairs -join ', ')")
    }

    if ($RunCount -gt 0) { $lines.Add("- **Runs per row:** $RunCount") }

    # Model identities: size and modified time always; SHA-256 only when asked for.
    foreach ($mp in $ModelPaths) {
        if (-not (Test-Path -LiteralPath $mp)) { $lines.Add("- **Model:** NOT FOUND ``$mp``"); continue }
        $id = Get-ModelIdentity -Path $mp -Hash:$Hash -CacheFile $CacheFile
        $shaText = if ($id.Sha256) { ", sha256 ``$($id.Sha256.Substring(0, 16))...`` ($($id.HashSource))" } else { '' }
        $lines.Add("- **Model:** ``$($id.File)``, $($id.SizeGB) GB, modified $($id.Modified)$shaText")
    }

    $lines.Add("- **Date:** $((Get-Date).ToString('yyyy-MM-dd HH:mm zzz'))")
    return $lines.ToArray()
}
