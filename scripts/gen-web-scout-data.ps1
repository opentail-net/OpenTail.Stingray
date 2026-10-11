# Generates a JS data file for the public website's feasibility table from REAL `stingray scout` runs (header-only, loads no weights).
# Usage: pwsh scripts/gen-web-scout-data.ps1 -Out <path>\stingray-scout-data.js [-Models a.gguf,b.gguf]
param([string]$Out, [string[]]$Models = @(), [int]$Ctx = 4096, [int]$ReserveGB = 2, [int[]]$BudgetsGB = @(8, 16, 32, 64))
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe'
$Models = @($Models | ForEach-Object { $_ -split ";" } | Where-Object { $_ })
if (-not $Models) { throw "pass -Models with GGUF paths (separate with ;)" }
$rows = @()
foreach ($m in $Models) {
    $entry = [ordered]@{ file = (Split-Path $m -Leaf); budgets = [ordered]@{} }
    foreach ($b in $BudgetsGB) {
        $j = (& $exe scout -m $m --budget "${b}G" --reserve "${ReserveGB}G" -c $Ctx --format json 2>$null) | ConvertFrom-Json
        if ($b -eq $BudgetsGB[0]) {
            $entry.arch = $j.architecture.declared; $entry.status = "$($j.architecture.status)"; $entry.backends = "$($j.architecture.supported_backends)"
            $entry.fileGiB = [math]::Round($j.artifact.file_bytes / 1GB, 2)
            $entry.workingSetGiB = [math]::Round($j.resources.host_working_set.bytes / 1GB, 2)
            $entry.certainty = "$($j.resources.host_working_set.certainty)"
            $entry.scoutBuild = $j.scout_build
        }
        $entry.budgets["$b"] = "$($j.resources.execution_decision)"
    }
    $rows += $entry
}
$payload = [ordered]@{ generated = (Get-Date -Format 'yyyy-MM-dd'); context = $Ctx; reserveGiB = $ReserveGB; note = 'Upper-bound CPU working-set estimates from stingray scout (header only, no weights loaded). An estimate, not a guarantee, and not a speed.'; models = $rows }
"window.STINGRAY_SCOUT = $($payload | ConvertTo-Json -Depth 6 -Compress);" | Set-Content -Path $Out -Encoding utf8
"wrote $Out ($($rows.Count) models)"
# The admitted-architecture list in the data file is produced separately: a 3-line console project that prints
#   ArchitectureRegistry.All.Where(d => d.Status == AdmissionStatus.Admitted).SelectMany(d => d.Aliases.Prepend(d.Id)).Distinct().Order()
# and is appended as `window.STINGRAY_ADMITTED = [...]`. TODO: fold into this script (or a CLI command) so the list cannot go stale.
