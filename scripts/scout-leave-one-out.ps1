<#
.SYNOPSIS
    Leave-one-ARCHITECTURE-out check of the scout ranking.
.DESCRIPTION
    For every shipped signature, scouts one of its own origin files with that architecture's signatures removed and records the nearest remaining
    admitted parent. Answers the hard question: does a family the reference set has never seen still land on a sensible relative, and with how many
    differences? Index only; no weights are read. Build the CLI in Release first. Origin files are looked up by name under -ModelRoots; a signature whose
    file is not found is reported, not skipped. Re-run after changing the shipped signatures or the ranking, and record the result in the scout plan.
.PARAMETER ModelRoots
    Folders searched (recursively) for the origin files.
#>
param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string[]]$ModelRoots = @()
)
$ErrorActionPreference = 'Stop'
$cli = "$Repo\src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe"
$sigDir = "$Repo\src\OpenTail.Stingray.Cli\Scout\Signatures"
$roots = @("$Repo\models") + $ModelRoots
$all = Get-ChildItem $sigDir -Filter *.signature.json | ForEach-Object { [pscustomobject]@{ File = $_; Json = (Get-Content $_.FullName -Raw | ConvertFrom-Json) } }
$rows = foreach ($s in $all) {
    $arch = $s.Json.architecture_id
    $tmp = Join-Path $env:TEMP ("loo-" + [guid]::NewGuid().ToString('N')); New-Item -ItemType Directory $tmp | Out-Null
    $all | Where-Object { $_.Json.architecture_id -ne $arch } | ForEach-Object { Copy-Item $_.File.FullName $tmp }
    $name = $s.Json.origins[0].file_name
    $model = $roots | ForEach-Object { Get-ChildItem $_ -Recurse -Filter $name -ErrorAction SilentlyContinue } | Select-Object -First 1
    if (-not $model) { [pscustomobject]@{ Arch = $arch; Structure = $s.Json.structure_id.Substring(0,8); File = $name; Parent = '(file not found)'; Diffs = $null; Kind = ''; Top3 = ''; Detail = '' }; continue }
    $j = & $cli scout -m $model.FullName --no-builtin-signatures --signatures $tmp --format json | ConvertFrom-Json
    $c = @($j.architecture.candidates)
    $top = if ($c.Count) { $c[0] } else { $null }
    [pscustomobject]@{
        Arch = $arch; Structure = $s.Json.structure_id.Substring(0,8); File = $name
        Parent = if ($top) { $top.id } else { '(none)' }
        Diffs = if ($top) { $top.difference_count } else { $null }
        Kind = if ($top) { $top.kind } else { '' }
        Top3 = ($c | ForEach-Object { "$($_.id):$($_.difference_count)" }) -join ' '
        Detail = if ($top) { ($top.differing | Select-Object -First 4) -join ' | ' } else { '' }
        Findings = (($j.findings | Where-Object { $_.id -like 'arch.*' } | ForEach-Object id) -join ',')
    }
}
$rows | Format-Table Arch, Structure, Parent, Diffs, Kind, Top3, Findings -AutoSize -Wrap | Out-String -Width 220
"--- details"
$rows | ForEach-Object { "{0,-10} {1,-9} {2,-40} -> {3} ({4}): {5}" -f $_.Arch, $_.Structure, $_.File.Substring(0, [Math]::Min(40, $_.File.Length)), $_.Parent, $_.Diffs, $_.Detail }
