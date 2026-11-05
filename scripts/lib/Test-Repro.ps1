# Checks for lib/Repro.ps1 (plan 2, P9). Run: pwsh -File scripts/lib/Test-Repro.ps1
# Exits non-zero on the first failed assertion. Uses a small temp file, never a model.

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Repro.ps1')

$fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { Write-Output "PASS  $name" } else { Write-Output "FAIL  $name  $detail"; $script:fail++ }
}

$dir = Join-Path ([System.IO.Path]::GetTempPath()) ("repro-check-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $dir | Out-Null
try {
    $model = Join-Path $dir 'fake-model.gguf'
    [System.IO.File]::WriteAllBytes($model, [byte[]]::new(2048))
    $cache = Join-Path $dir 'hashes.json'

    # 1. Default: size and modified time, no hash.
    $id = Get-ModelIdentity -Path $model
    Check 'no -Hash gives no sha256' ($null -eq $id.Sha256 -and $id.HashSource -eq 'not hashed') "got sha=$($id.Sha256) src=$($id.HashSource)"
    Check 'size and mtime reported' ($id.File -eq 'fake-model.gguf' -and $id.Modified.Length -gt 0)

    # 2. First -Hash call computes and writes the cache file.
    $first = Get-ModelIdentity -Path $model -Hash -CacheFile $cache
    Check 'first -Hash computes' ($first.HashSource -eq 'computed now' -and $first.Sha256.Length -eq 64) "src=$($first.HashSource)"
    Check 'cache file written' (Test-Path -LiteralPath $cache)

    # 3. Same process again: served from memory, not recomputed.
    $second = Get-ModelIdentity -Path $model -Hash -CacheFile $cache
    Check 'second call uses memory' ($second.HashSource -eq 'memory' -and $second.Sha256 -eq $first.Sha256) "src=$($second.HashSource)"

    # 4. Fresh process state (memory cleared): served from the cache file, with no rehash.
    #    The cache entry is replaced by a sentinel. If the helper rehashed, the sentinel would disappear.
    $raw = Get-Content -LiteralPath $cache -Raw | ConvertFrom-Json -AsHashtable
    $key = @($raw.Keys)[0]
    $raw[$key] = 'SENTINEL-NOT-A-REAL-HASH'
    $raw | ConvertTo-Json | Set-Content -LiteralPath $cache -Encoding utf8
    $script:ReproHashMemo.Clear()
    $third = Get-ModelIdentity -Path $model -Hash -CacheFile $cache
    Check 'cache file used without rehash' ($third.HashSource -eq 'cache file' -and $third.Sha256 -eq 'SENTINEL-NOT-A-REAL-HASH') "src=$($third.HashSource) sha=$($third.Sha256)"

    # 5. A changed file (new size) is a new key, so the stale cache entry is not reused.
    [System.IO.File]::WriteAllBytes($model, [byte[]]::new(4096))
    $script:ReproHashMemo.Clear()
    $fourth = Get-ModelIdentity -Path $model -Hash -CacheFile $cache
    Check 'changed file is rehashed' ($fourth.HashSource -eq 'computed now' -and $fourth.Sha256 -ne 'SENTINEL-NOT-A-REAL-HASH') "src=$($fourth.HashSource)"

    # 6. Header block: has a commit line, a model line, and no hash without -Hash.
    $header = Get-ReproHeader -RepoRoot (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) -BinaryPath $model -Backend 'cpu' -ModelPaths @($model) -Settings @{ scenarios = '500' } -RunCount 2
    $text = $header -join "`n"
    Check 'header has commit line' ($text -match '\*\*Commit:\*\* `[0-9a-f]+`')
    Check 'header has model line without hash' ($text -match 'fake-model\.gguf' -and $text -notmatch 'sha256')
    Check 'header has settings and runs' ($text -match 'scenarios=500' -and $text -match 'Runs per row:\*\* 2')
}
finally {
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
}

if ($fail -gt 0) { Write-Output "$fail check(s) failed"; exit 1 }
Write-Output 'All Repro checks passed'
exit 0
