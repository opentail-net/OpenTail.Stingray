<#
.SYNOPSIS
  Reads the machine-readable result file a CLI run writes when STINGRAY_STATS_JSON=<path> is set.

.DESCRIPTION
  The structured alternative to regex-matching the printed "Prefill: ... | Decode: ..." lines. Validates schema_version
  and refuses a run taken with a diagnostic profile switch on (its numbers are perturbed by the profiling), unless
  -AllowPerturbed is passed. Returns the parsed object (load_ms, prompt_tokens, prefill_ms, prefill_tokens_per_second,
  decode_tokens, visible_tokens, decode_ms, decode_tokens_per_second, speculative_* ...). Contract: docs/guides/observability.md.

.EXAMPLE
  $env:STINGRAY_STATS_JSON = "$env:TEMP\run.json"
  stingray -m model.gguf -p "Hello" --temp 0 -g 64
  $s = ./scripts/Get-StingrayRunStats.ps1 -Path $env:STINGRAY_STATS_JSON
  "{0:F1} tok/s decode" -f $s.decode_tokens_per_second
#>
param(
  [Parameter(Mandatory)][string]$Path,
  [switch]$AllowPerturbed
)
if (-not (Test-Path -LiteralPath $Path)) { throw "No stats file at '$Path' (was STINGRAY_STATS_JSON set for the run, and did it exit normally?)" }
$stats = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
if ($stats.schema_version -ne 1) { throw "Unsupported stats schema_version '$($stats.schema_version)' (this reader understands 1)." }
if ($stats.perturbed -and -not $AllowPerturbed) {
  throw "This run was taken with a diagnostic profile switch on (STINGRAY_MOE_PHASE_TIMING / STINGRAY_PROFILE_GPU_SPLIT); its timings are perturbed and must not be compared. Pass -AllowPerturbed to read it anyway."
}
$stats
