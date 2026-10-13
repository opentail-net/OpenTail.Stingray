<#
.SYNOPSIS
  Publishes the real NativeAOT CLI and proves engine metrics and spans reach an in-process listener (no model needed).

.DESCRIPTION
  Compiling is not proof that telemetry works in a published binary: with MetricsSupport=false (or trimmed instruments)
  a Meter listener receives nothing. This runs `stingray telemetry-selftest` from the published executable and fails
  if anything is missing. Observability plan D9/D10.

.EXAMPLE
  ./scripts/check-aot-telemetry.ps1            # win-x64 by default
  ./scripts/check-aot-telemetry.ps1 -Rid linux-x64
#>
param(
  [string]$Rid = "win-x64",
  [string]$OutDir = (Join-Path ([System.IO.Path]::GetTempPath()) "stingray-aot-telemetry-check")
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
  if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
  dotnet publish src/OpenTail.Stingray.Cli -c Release -r $Rid -o $OutDir
  if ($LASTEXITCODE -ne 0) { throw "AOT publish failed (exit $LASTEXITCODE)" }
  $exe = Join-Path $OutDir ($(if ($Rid -like "win-*") { "stingray.exe" } else { "stingray" }))
  & $exe telemetry-selftest
  if ($LASTEXITCODE -ne 0) { throw "telemetry-selftest failed in the published binary (exit $LASTEXITCODE)" }
  Write-Host "ok: the published $Rid binary delivers engine metrics and spans to an in-process listener"
} finally { Pop-Location }
