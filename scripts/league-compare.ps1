# Compare a bench-league-recheck.ps1 results file with the existing C# rows of PerformanceLeague.md.
# A League row is matched to a result when: the model names agree after normalisation, the backend agrees (CPU / Vulkan), the League scenario
# is a prefill or decode row with a prompt of 400-650 tokens (the recheck's '500' scenario is a ~490-580 token prompt), and it carries a t/s number.
# Output: one line per match: model, backend, metric, old (date), new, change. Rows with no match are listed separately (new coverage).
# Usage: pwsh scripts/league-compare.ps1 -Results <results.md> [-League PerformanceLeague.md] [-Csv out.csv]
param(
    [Parameter(Mandatory)][string]$Results,
    [string]$League = (Join-Path (Split-Path -Parent $PSScriptRoot) 'PerformanceLeague.md'),
    [string]$Csv = ''
)

function Norm([string]$s) { ($s.ToLowerInvariant() -replace '[^a-z0-9]', '') }

# --- new results: model | GB | arch | backend | scenario | tok | prefill | decode
$new = foreach ($l in (Get-Content $Results)) {
    if ($l -match '^\| (?<m>[^|]+?) \| (?<gb>[\d.]+) \| (?<a>[^|]*?) \| (?<b>cpu|vulkan) \| (?<s>500) \| (?<t>\d+) \| (?<p>[\d.]+) \| (?<d>[\d.]+) \|') {
        [pscustomobject]@{ model = $Matches.m; key = (Norm $Matches.m); backend = $Matches.b; tok = [int]$Matches.t; prefill = [double]$Matches.p; decode = [double]$Matches.d }
    }
}

# --- League rows (model-first tables and the per-section SmolLM2-style tables)
$rows = [System.Collections.Generic.List[object]]::new()
$section = ''; $cols = @()
foreach ($l in (Get-Content $League)) {
    if ($l -match '^##+ (.+)$') { $section = $Matches[1]; $cols = @(); continue }
    if ($l -notmatch '^\|') { continue }
    $cells = ($l.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
    if ($cells.Count -lt 5) { continue }
    if ($l -match '^\|[-: |]+\|$') { continue }
    if ($cells[0] -match '^(Model|Scenario)' ) { $cols = $cells; continue }
    if ($cols.Count -eq 0) { continue }
    $modelCol = [array]::IndexOf($cols, ($cols | Where-Object { $_ -match '^Model' } | Select-Object -First 1))
    $scIdx = [array]::IndexOf($cols, ($cols | Where-Object { $_ -match '^Scenario' } | Select-Object -First 1))
    $beIdx = [array]::IndexOf($cols, ($cols | Where-Object { $_ -match '^Backend' } | Select-Object -First 1))
    $csIdx = $cols.IndexOf(($cols | Where-Object { $_ -match 'C#|OT, t/s|Ours|stingray' } | Select-Object -First 1))
    $cppIdx = $cols.IndexOf(($cols | Where-Object { $_ -match 'C\+\+' } | Select-Object -First 1))
    $dtIdx = $cols.IndexOf(($cols | Where-Object { $_ -match 'Performance Check|Date' } | Select-Object -First 1))
    if ($beIdx -lt 0 -or $csIdx -lt 0) { continue }
    $model = if ($modelCol -ge 0) { $cells[$modelCol] } else { $section }
    $scen = if ($scIdx -ge 0 -and $scIdx -lt $cells.Count) { $cells[$scIdx] } else { ($cells -join ' ') }
    if ($scen -notmatch 'prefill|decode') { continue }
    $be = $cells[$beIdx]; $backend = if ($be -match 'Vulkan') { 'vulkan' } elseif ($be -match '^CPU') { 'cpu' } else { '' }
    if (-not $backend) { continue }
    if ($scen -notmatch '(\d{3,4})\s*tok') { continue }
    $tok = [int]$Matches[1]; if ($tok -lt 400 -or $tok -gt 650) { continue }
    if ($csIdx -ge $cells.Count -or $cells[$csIdx] -notmatch 't/s') { continue }
    # A cell is either one value or a "prefill / decode" pair ("32.3-32.4 / 10.9-11.0 t/s"); take the first number of each part.
    function Parts([string]$cell) {
        $c = ($cell -replace 't/s', '') -replace '\*', '' -replace '\(.*?\)', ''
        $nums = @(); foreach ($p in ($c -split '/')) { if ($p -match '([\d.]+)') { $nums += [double]$Matches[1] } else { $nums += $null } }
        return , $nums
    }
    $isPair = $scen -match 'prefill.*/.*decode'
    $csVals = Parts $cells[$csIdx]
    $cppVals = if ($cppIdx -ge 0 -and $cppIdx -lt $cells.Count) { Parts $cells[$cppIdx] } else { @() }
    $date = if ($dtIdx -ge 0 -and $dtIdx -lt $cells.Count) { $cells[$dtIdx] } else { '' }
    $names = @(); $idxs = @()
    if ($isPair) { $names = @('prefill', 'decode'); $idxs = @(0, 1) }
    elseif ($scen -match '^prefill|prefill \(') { $names = @('prefill'); $idxs = @(0) }
    elseif ($scen -match 'decode') { $names = @('decode'); $idxs = @(0) }
    for ($k = 0; $k -lt $names.Count; $k++) {
        $i = [int]$idxs[$k]
        if ($i -ge $csVals.Count -or $null -eq $csVals[$i]) { continue }
        $cpp = 0.0; if ($i -lt $cppVals.Count -and $null -ne $cppVals[$i]) { $cpp = [double]$cppVals[$i] }
        $rows.Add([pscustomobject]@{ model = $model; key = (Norm $model); backend = $backend; metric = $names[$k]; tok = $tok; old = [double]$csVals[$i]; cpp = $cpp; date = $date; scen = $scen })
    }
}

$out = foreach ($n in $new) {
    foreach ($metric in 'prefill', 'decode') {
        $cand = $rows | Where-Object { $_.backend -eq $n.backend -and $_.metric -eq $metric -and ($n.key.Contains($_.key) -or $_.key.Contains($n.key)) -and $_.key.Length -ge 6 }
        if (-not $cand) { continue }
        # prefer the row with the closest prompt length, then the latest date string
        $best = $cand | Sort-Object { [Math]::Abs($_.tok - $n.tok) }, date -Descending | Select-Object -First 1
        $newv = if ($metric -eq 'prefill') { $n.prefill } else { $n.decode }
        [pscustomobject]@{ model = $n.model; backend = $n.backend; metric = $metric; oldTok = $best.tok; newTok = $n.tok; old = $best.old; oldDate = $best.date; cpp = $best.cpp; estRatio = $(if ($best.cpp -gt 0) { [Math]::Round($newv / $best.cpp, 2) } else { "" }); oldRatio = $(if ($best.cpp -gt 0) { [Math]::Round($best.old / $best.cpp, 2) } else { "" }); new = $newv; change = [Math]::Round($newv / $best.old, 2) }
    }
}
$matchedKeys = $out | ForEach-Object { Norm $_.model } | Select-Object -Unique
$unmatched = $new | Where-Object { $matchedKeys -notcontains $_.key } | Select-Object -ExpandProperty model -Unique

$out | Sort-Object change -Descending | Format-Table model, backend, metric, oldTok, newTok, old, oldDate, new, change -AutoSize | Out-String -Width 220
"matched $($out.Count) metric rows over $(($out | Select-Object -ExpandProperty model -Unique).Count) models; $(($unmatched | Measure-Object).Count) measured models had no comparable League row (new coverage):"
$unmatched -join ', '
if ($Csv) { $out | Export-Csv -Path $Csv -NoTypeInformation -Encoding utf8 }
