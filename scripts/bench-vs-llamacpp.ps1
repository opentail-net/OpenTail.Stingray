<#
.SYNOPSIS
    One-command CPU prefill/decode comparison: OpenTail.Stingray vs llama.cpp vs the pristine
    pre-rebrand SharpInference snapshot (regression check for "did our changes make it worse").
.DESCRIPTION
    Runs the same GGUF model through three CLIs at matched prompt lengths and thread counts,
    parses each tool's own reported t/s, and prints one side-by-side table. Existing bench
    scripts (bench-allrows-1k.ps1 etc.) sweep many models/backends for the design-doc tables;
    this one is deliberately narrow — a fast, repeatable "are we closing the gap" check for the
    CPU prefill work tracked in docs/cpu-prefill-plan.md.
.PARAMETER Model
    Path to a GGUF model. Default: models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf (relative to this
    script's ..\ root), the model the prefill investigation has been using throughout.
.PARAMETER PromptTokens
    Approximate prompt lengths to test, in tokens. Default matches the plan doc: 87, 261, 903.
.PARAMETER DecodeTokens
    Tokens to decode for the tg (decode) rate. Default: 64.
.PARAMETER Threads
    Thread count passed to llama-bench (-t) for a fair comparison; OpenTail.Stingray always uses
    Environment.ProcessorCount. Default: 16.
.PARAMETER Runs
    Repeats per configuration for both llama-bench (-r) and the OpenTail-family CLIs.
.PARAMETER SkipDome
    Skip the pristine SharpInference snapshot (examples/cpp/_dome/SharpInference). Use this if
    that checkout has been removed or moved.
.PARAMETER SkipLlamaCpp
    Skip the llama.cpp reference. Use this for a quick OpenTail-only before/after check.
.EXAMPLE
    .\bench-vs-llamacpp.ps1
    .\bench-vs-llamacpp.ps1 -PromptTokens 900 -Runs 3
    .\bench-vs-llamacpp.ps1 -SkipDome
#>
param(
    [string]$Model = "",
    [int[]]$PromptTokens = @(87, 261, 903),
    [int]$DecodeTokens = 64,
    [int]$Threads = 16,
    [int]$Runs = 2,
    [switch]$SkipDome,
    [switch]$SkipLlamaCpp
)

$ErrorActionPreference = "Stop"
$RepoRoot   = Resolve-Path (Join-Path $PSScriptRoot "..")
$DomeRoot   = Join-Path $RepoRoot "..\..\examples\cpp\_dome\SharpInference"
$LlamaBench = Join-Path $RepoRoot "tools\llama.cpp\llama-bench.exe"

if ($Model -eq "") {
    $Model = Join-Path $RepoRoot "models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
}
if (-not (Test-Path $Model)) {
    Write-Error "Model not found: $Model`nFetch one first, e.g.:`n  .\scripts\download-model.ps1 -Model smollm2"
    exit 1
}
$ModelFull = (Resolve-Path $Model).Path

# ── Prompt fixtures ──────────────────────────────────────────────────────────
# Built once per run from repeated natural-language text (not random bytes), so both the
# OpenTail tokenizer and llama.cpp's tokenizer land close to the requested token count without
# needing either CLI's own tokenizer to size the file up front.
$Filler = "The quick brown fox jumps over the lazy dog near the riverbank at dawn while several " +
          "birds circle overhead searching for food among the tall reeds and scattered stones. "
$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "opentail-bench-vs-llamacpp"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

function New-PromptFile([int]$targetTokens) {
    # ~4.7 chars/token for this filler text, empirically (matches the 87/261/903-token fixtures
    # used throughout the plan doc's own measurements).
    $chars = [int]([Math]::Ceiling($targetTokens * 4.7))
    $text = ""
    while ($text.Length -lt $chars) { $text += $Filler }
    $path = Join-Path $TmpDir "p_$targetTokens.txt"
    [System.IO.File]::WriteAllText($path, $text.Substring(0, $chars))
    return $path
}

# ── Run one OpenTail-family CLI (current or the pristine _dome snapshot) ────
function Invoke-OpenTailCli([string]$ProjectDir, [string]$PromptFile, [int]$DecodeN) {
    # DecodeN must be > ~30-40 or the reported decode t/s is dominated by process
    # startup/JIT-tiering noise rather than steady-state throughput (measured: -n 1 reads
    # ~20% low and noisier vs -n 64/128 back-to-back on identical input).
    $out = & dotnet run --project $ProjectDir -c Release --no-build -- `
        -m $ModelFull -f $PromptFile --temp 0 -n $DecodeN --no-display-prompt 2>&1
    $line = $out | Select-String -Pattern "Prefill: (\d+) tokens, ([\d.]+) t/s \| Decode: \d+ tokens, ([\d.]+) t/s"
    if (-not $line) { return $null }
    $m = $line.Matches[0]
    return [pscustomobject]@{
        PromptTokens = [int]$m.Groups[1].Value
        PrefillTps   = [double]$m.Groups[2].Value
        DecodeTps    = [double]$m.Groups[3].Value
    }
}

function Measure-OpenTailCli([string]$Label, [string]$ProjectDir, [int[]]$Tokens, [int]$Runs, [int]$DecodeN) {
    Write-Output "--- building $Label ($ProjectDir) ---"
    dotnet build $ProjectDir -c Release 2>&1 | Select-String -Pattern "error|Build succeeded" | Out-Host

    $rows = @()
    foreach ($t in $Tokens) {
        $promptFile = New-PromptFile $t
        $prefillSamples = @()
        $decodeSamples = @()
        for ($r = 0; $r -lt $Runs; $r++) {
            $result = Invoke-OpenTailCli -ProjectDir $ProjectDir -PromptFile $promptFile -DecodeN $DecodeN
            if ($result) {
                $prefillSamples += $result.PrefillTps
                $decodeSamples += $result.DecodeTps
            }
        }
        if ($prefillSamples.Count -eq 0) {
            Write-Warning "$Label produced no parseable output at ~$t tokens"
            continue
        }
        $rows += [pscustomobject]@{
            Label      = $Label
            Tokens     = $t
            PrefillTps = ($prefillSamples | Measure-Object -Average).Average
            DecodeTps  = ($decodeSamples  | Measure-Object -Average).Average
        }
    }
    return $rows
}

# ── Run llama.cpp's own bench tool ──────────────────────────────────────────
function Measure-LlamaCpp([int[]]$Tokens, [int]$DecodeTokens, [int]$Threads, [int]$Runs) {
    if (-not (Test-Path $LlamaBench)) {
        Write-Warning "llama-bench.exe not found at $LlamaBench`nFetch it first: .\scripts\setup-llamacpp.ps1 -Variant cpu"
        return @()
    }

    $promptArg = ($Tokens -join ",")
    $raw = & $LlamaBench -m $ModelFull -t $Threads -p $promptArg -n $DecodeTokens -r $Runs 2>&1

    $rows = @()
    foreach ($line in $raw) {
        # | model | size | params | backend | threads | test | t/s |
        # The separator between the mean and stddev is "±", but PowerShell's native-command
        # capture mangles it into other bytes depending on console codepage (observed: U+2534
        # U+2592 instead of U+00B1) -- match on "some non-digit run" instead of the literal
        # glyph so this doesn't silently parse zero rows again.
        if ($line -match '\|\s*pp(\d+)\s*\|\s*([\d.]+)\s*\S') {
            $rows += [pscustomobject]@{ Label = "llama.cpp"; Tokens = [int]$Matches[1]; PrefillTps = [double]$Matches[2]; DecodeTps = $null }
        } elseif ($line -match '\|\s*tg(\d+)\s*\|\s*([\d.]+)\s*\S') {
            $rows += [pscustomobject]@{ Label = "llama.cpp"; Tokens = $null; PrefillTps = $null; DecodeTps = [double]$Matches[2] }
        }
    }
    return $rows
}

# ── Run all three ────────────────────────────────────────────────────────────
Write-Output "Model: $ModelFull"
Write-Output "Prompt sizes: $($PromptTokens -join ', ') tokens | decode: $DecodeTokens tokens | threads (llama.cpp): $Threads | runs: $Runs"
Write-Output ""

$current = Measure-OpenTailCli -Label "OpenTail.Stingray (current)" `
    -ProjectDir (Join-Path $RepoRoot "src\OpenTail.Stingray.Cli") -Tokens $PromptTokens -Runs $Runs -DecodeN $DecodeTokens

$dome = @()
if (-not $SkipDome) {
    if (Test-Path $DomeRoot) {
        $dome = Measure-OpenTailCli -Label "SharpInference (pristine _dome)" `
            -ProjectDir (Join-Path $DomeRoot "src\SharpInference.Cli") -Tokens $PromptTokens -Runs $Runs -DecodeN $DecodeTokens
    } else {
        Write-Warning "Pristine snapshot not found at $DomeRoot -- skipping. Use -SkipDome to silence this."
    }
}

$llamaRows = @()
$llamaDecode = $null
if (-not $SkipLlamaCpp) {
    $llamaRaw = Measure-LlamaCpp -Tokens $PromptTokens -DecodeTokens $DecodeTokens -Threads $Threads -Runs $Runs
    $llamaRows = $llamaRaw | Where-Object { $_.Tokens -ne $null }
    $llamaDecodeRow = $llamaRaw | Where-Object { $_.DecodeTps -ne $null } | Select-Object -First 1
    if ($llamaDecodeRow) { $llamaDecode = $llamaDecodeRow.DecodeTps }
}

# ── Report ───────────────────────────────────────────────────────────────────
Write-Output ""
Write-Output "=== Prefill (t/s) ==="
$table = @()
foreach ($t in $PromptTokens) {
    $row = [ordered]@{ Tokens = $t }
    $c = $current | Where-Object { $_.Tokens -eq $t } | Select-Object -First 1
    if ($c) { $row["OpenTail.Stingray"] = [Math]::Round($c.PrefillTps, 1) }
    if ($dome.Count -gt 0) {
        $d = $dome | Where-Object { $_.Tokens -eq $t } | Select-Object -First 1
        if ($d) { $row["SharpInference (dome)"] = [Math]::Round($d.PrefillTps, 1) }
    }
    if ($llamaRows.Count -gt 0) {
        $l = $llamaRows | Where-Object { $_.Tokens -eq $t } | Select-Object -First 1
        if ($l) { $row["llama.cpp"] = [Math]::Round($l.PrefillTps, 1) }
    }
    if ($c -and $llamaRows.Count -gt 0) {
        $l = $llamaRows | Where-Object { $_.Tokens -eq $t } | Select-Object -First 1
        if ($l -and $c.PrefillTps -gt 0) { $row["Gap (llama/OpenTail)"] = [Math]::Round($l.PrefillTps / $c.PrefillTps, 1).ToString() + "x" }
    }
    $table += [pscustomobject]$row
}
$table | Format-Table -AutoSize

Write-Output "=== Decode (t/s) ==="
$decodeTable = [ordered]@{}
if ($current.Count -gt 0) { $decodeTable["OpenTail.Stingray"] = [Math]::Round(($current | Measure-Object -Property DecodeTps -Average).Average, 1) }
if ($dome.Count -gt 0)    { $decodeTable["SharpInference (dome)"] = [Math]::Round(($dome | Measure-Object -Property DecodeTps -Average).Average, 1) }
if ($llamaDecode)         { $decodeTable["llama.cpp"] = [Math]::Round($llamaDecode, 1) }
[pscustomobject]$decodeTable | Format-Table -AutoSize

Write-Output "(Full context and plan: docs/cpu-prefill-plan.md)"
