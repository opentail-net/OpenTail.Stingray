<#
.SYNOPSIS
    Reports which GGUF models the test suite gates on, and which are missing.

.DESCRIPTION
    Most model-dependent tests call a local FindModelPath helper and `return` early when the
    file is absent. An early return is indistinguishable from a pass in the runner summary, so
    a green suite can hide arbitrarily many real defects — on 2026-08-03 exactly one missing
    model (gemma-4-E4B_q4_0-it.gguf) was downloaded and immediately surfaced three genuine
    Vulkan failures that had been "passing" for as long as the file was absent.

    This script makes that invisible skipping legible. It scans the test sources for .gguf
    filename literals, checks each against the models directory, and prints a present/absent
    table with a coverage percentage.

    Gating is FILENAME-EXACT: holding Qwen3-0.6B-Q8_0.gguf does not satisfy a test that looks
    for Qwen3-0.6B-Instruct-Q4_K_M.gguf. Near-duplicate names in the list are therefore real
    coverage gaps, not bookkeeping noise.

.PARAMETER FailOnMissing
    Exit non-zero if any gated model is absent. Off by default: on a dev box most models are
    legitimately absent, and a script that always fails gets ignored. Use in CI where the full
    model set is expected.

.EXAMPLE
    .\scripts\check-test-model-coverage.ps1
    .\scripts\check-test-model-coverage.ps1 -FailOnMissing
#>
param(
    [switch]$FailOnMissing
)

$ErrorActionPreference = "Stop"
$repoRoot  = Resolve-Path "$PSScriptRoot\.."
$modelsDir = Join-Path $repoRoot "models"
$testsDir  = Join-Path $repoRoot "tests"

if (-not (Test-Path $testsDir)) { throw "Tests directory not found: $testsDir" }

# Collect .gguf literals from every test source, with the files that reference each.
$refs = @{}
Get-ChildItem -Path $testsDir -Filter *.cs -Recurse -File | ForEach-Object {
    $file = $_
    foreach ($m in [regex]::Matches($(Get-Content $file.FullName -Raw), '"([A-Za-z0-9._\-]+\.gguf)"')) {
        $name = $m.Groups[1].Value
        if (-not $refs.ContainsKey($name)) { $refs[$name] = [System.Collections.Generic.HashSet[string]]::new() }
        [void]$refs[$name].Add($file.Name)
    }
}

if ($refs.Count -eq 0) { Write-Host "No .gguf literals found under $testsDir"; exit 0 }

# Separate real gated models from test fixtures. Tests create throwaway GGUFs (a.gguf, smol.gguf,
# broken.gguf, ...) whose absence means nothing — counting those as missing coverage would make
# this report cry wolf, and a checker that over-reports gets ignored.
#
# Authority for "real model" is download-model.ps1's own file list, plus anything actually on disk
# in models/. That keeps the two in sync automatically: adding a preset there makes it count here.
$known = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$dl = Join-Path $PSScriptRoot "download-model.ps1"
if (Test-Path $dl) {
    foreach ($m in [regex]::Matches($(Get-Content $dl -Raw), '"([A-Za-z0-9._\-]+\.gguf)"')) {
        [void]$known.Add($m.Groups[1].Value)
    }
}
if (Test-Path $modelsDir) {
    Get-ChildItem -Path $modelsDir -Filter *.gguf -File | ForEach-Object { [void]$known.Add($_.Name) }
}

# download-model.ps1 alone is NOT sufficient: several real models the tests gate on are not named
# literally in it (gemma-4-E4B-it-Q8_0.gguf backs 11 test files yet never appears there). Filtering
# on it alone hid the single largest gap — a false negative, which is worse than the cry-wolf it
# was meant to cure, because it silently under-reports.
#
# Second signal: a quant marker in the filename (Q4_K_M, Q8_0, q4_0, q4km ...). Every real GGUF here
# carries one; the throwaway fixtures (a, b, c, x, smol, model, broken, from-*) carry none.
$quantMarker = '[Qq]\d[_KkMm0-9]'
$isRealModel = { param($n) $known.Contains($n) -or ($n -cmatch $quantMarker) }

$fixtures = @($refs.Keys | Where-Object { -not (& $isRealModel $_) } | Sort-Object)
$gated    = @($refs.Keys | Where-Object {      (& $isRealModel $_) } | Sort-Object)

if ($gated.Count -eq 0) {
    Write-Host "No recognised gated models referenced (checked against download-model.ps1 and models/)."
    exit 0
}

$rows = foreach ($name in $gated) {
    $path = Join-Path $modelsDir $name
    $have = Test-Path $path
    [pscustomobject]@{
        Present   = if ($have) { "yes" } else { "NO" }
        Model     = $name
        SizeGB    = if ($have) { "{0:N2}" -f ((Get-Item $path).Length / 1GB) } else { "" }
        TestFiles = $refs[$name].Count
    }
}

$present = @($rows | Where-Object Present -eq "yes").Count
$total   = $rows.Count
$absent  = $total - $present

# Absent first, and within that the ones blocking the most test files — that is the download
# priority order. Size of the model is not the right sort key; tests unblocked is.
$rows | Sort-Object @{ e = { $_.Present -eq "yes" } }, @{ e = { -$_.TestFiles } } |
    Format-Table -AutoSize | Out-String | Write-Host

$pct = if ($total -gt 0) { [math]::Round(100.0 * $present / $total, 1) } else { 0 }
Write-Host "Gated models: $present/$total present ($pct%), $absent absent."

if ($fixtures.Count -gt 0) {
    Write-Host "Ignored $($fixtures.Count) test-fixture name(s) not in download-model.ps1 or models/: $($fixtures -join ', ')"
}

if ($absent -gt 0) {
    $blocked = ($rows | Where-Object Present -eq "NO" | Measure-Object -Property TestFiles -Sum).Sum
    Write-Host ""
    Write-Host "WARNING: $absent gated model(s) absent, affecting up to $blocked test-file reference(s)."
    Write-Host "Tests needing them return early and are counted as PASSED, not skipped —"
    Write-Host "so the suite's pass count overstates what was actually verified."
}

if ($FailOnMissing -and $absent -gt 0) { exit 1 }
