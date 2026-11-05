<#
.SYNOPSIS
  Local CPU performance gate (E2). Measures decode and prefill tokens per second for a fixed set of
  chat models and fails when a model falls more than the tolerance below its recorded baseline.

.DESCRIPTION
  The baseline is per machine (scripts/perf-baseline.json, keyed by computer name). Run with -Update
  once on a quiet machine to record the baseline, then run without it before a release or after a
  kernel change. Hosted CI runners have no model files, so this gate is local by design.

.PARAMETER Update    Record the measured medians as the new baseline for this machine.
.PARAMETER Tolerance Allowed fractional drop below the baseline before failing (default 0.15).
.PARAMETER Repeats   Runs per model; the median is used (default 3).
.PARAMETER Baseline  Path to the baseline JSON (default scripts/perf-baseline.json).
#>
[CmdletBinding()]
param(
  [switch]$Update,
  [double]$Tolerance = 0.15,
  [int]$Repeats = 3,
  [string]$Baseline = (Join-Path $PSScriptRoot "perf-baseline.json"),
  [string]$Cli = (Join-Path $PSScriptRoot "..\src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe")
)

$ErrorActionPreference = "Stop"

# Fixed set: catalog id, model file, prompt, decode tokens.
$models = @(
  @{ id = "qwen2.5-0.5b"; file = "F:\_models\qwen2.5-0.5b-instruct-q4_k_m.gguf" },
  @{ id = "qwen2.5-1.5b"; file = "F:\_models\qwen2.5-1.5b-instruct-q4_k_m.gguf" }
)
$prompt = '"Write a short story about the sea."'
$decodeTokens = 256

if (-not (Test-Path $Cli)) { throw "CLI not built: $Cli (run: dotnet build src/OpenTail.Stingray.Cli -c Release)" }

function Measure-Once([string]$file) {
  $out = [System.IO.Path]::GetTempFileName()
  $err = [System.IO.Path]::GetTempFileName()
  $p = Start-Process -FilePath $Cli -ArgumentList @("-m", $file, "-p", $prompt, "-n", "$decodeTokens", "--temp", "0", "-g", "0") `
        -NoNewWindow -PassThru -Wait -RedirectStandardOutput $out -RedirectStandardError $err
  if ($p.ExitCode -ne 0) { throw "CLI exited $($p.ExitCode) for $file" }
  $line = Select-String -Path $out -Pattern 'Prefill: .*? ([\d.]+) t/s \| Decode: .*? ([\d.]+) t/s' | Select-Object -Last 1
  Remove-Item $out, $err -ErrorAction SilentlyContinue
  if (-not $line) { throw "no timing line for $file" }
  $m = [regex]::Match($line.Line, 'Prefill: .*? ([\d.]+) t/s \| Decode: .*? ([\d.]+) t/s')
  return [pscustomobject]@{ prefill = [double]$m.Groups[1].Value; decode = [double]$m.Groups[2].Value }
}

function Median([double[]]$v) { $s = $v | Sort-Object; $s[[int][Math]::Floor($s.Count / 2)] }

$machine = $env:COMPUTERNAME
$measured = [ordered]@{}
foreach ($m in $models) {
  if (-not (Test-Path $m.file)) { throw "model missing: $($m.file)" }
  $runs = 1..$Repeats | ForEach-Object { Measure-Once $m.file }
  $measured[$m.id] = [ordered]@{
    prefill_tps = [Math]::Round((Median ($runs.prefill)), 1)
    decode_tps  = [Math]::Round((Median ($runs.decode)), 1)
  }
  "{0,-14} prefill {1,7} t/s   decode {2,7} t/s   (median of {3})" -f $m.id, $measured[$m.id].prefill_tps, $measured[$m.id].decode_tps, $Repeats
}

$all = @{}
if (Test-Path $Baseline) { $all = Get-Content $Baseline -Raw | ConvertFrom-Json -AsHashtable }
if ($Update) {
  $all[$machine] = $measured
  ($all | ConvertTo-Json -Depth 6) | Set-Content -Path $Baseline -Encoding utf8
  "baseline recorded for $machine in $Baseline"
  exit 0
}

if (-not $all.ContainsKey($machine)) { throw "no baseline for $machine; run with -Update first" }
$base = $all[$machine]
$failed = @()
foreach ($id in $measured.Keys) {
  foreach ($metric in @("prefill_tps", "decode_tps")) {
    $b = [double]$base[$id][$metric]
    $now = [double]$measured[$id][$metric]
    $floor = $b * (1 - $Tolerance)
    if ($now -lt $floor) { $failed += "{0} {1}: {2} t/s < {3} (baseline {4}, tolerance {5:P0})" -f $id, $metric, $now, [Math]::Round($floor,1), $b, $Tolerance }
  }
}
if ($failed.Count -gt 0) { "PERF GATE FAILED"; $failed | ForEach-Object { "  $_" }; exit 1 }
"PERF GATE PASSED (tolerance {0:P0})" -f $Tolerance
exit 0
