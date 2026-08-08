<#
.SYNOPSIS
    Regenerates the per-command option tables in docs/cli-option-inventory.md from source.

.DESCRIPTION
    Scans [CommandOption(...)] / [Description(...)] pairs under src/OpenTail.Stingray.Cli and
    rewrites each "## <Command>" table. The inventory had drifted to 96 rows against 149 declared
    options, and a hand-maintained table cannot track a surface that size — the doc itself asked
    for "a checked-in generator or test-backed extraction".

    The Class column is HAND classification, so existing values are preserved by option name and
    command. New rows arrive blank for a human to fill in; that is the point of the column.

.PARAMETER Check
    Do not write. Exit 1 if the file would change. For CI / a release gate.
#>
[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cliDir = Join-Path $root 'src\OpenTail.Stingray.Cli'
$docPath = Join-Path $root 'docs\cli-option-inventory.md'

# Command name comes from the settings class's enclosing command type, e.g. RunCommand.Settings.
$rows = [System.Collections.Generic.List[object]]::new()
foreach ($file in Get-ChildItem -Path $cliDir -Filter *.cs -Recurse |
         Where-Object { $_.FullName -notlike '*obj*' -and $_.FullName -notlike '*bin*' }) {
    $lines = Get-Content -LiteralPath $file.FullName
    $command = [IO.Path]::GetFileNameWithoutExtension($file.Name)
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $m = [regex]::Match($lines[$i], '\[CommandOption\("([^"]+)"\)\]')
        if (-not $m.Success) { continue }
        # Longest alias is the readable one; Spectre lists them short-first.
        $names = $m.Groups[1].Value -split '\|'
        # @() is load-bearing: PowerShell unwraps a single-element Sort-Object result to a
        # scalar string, and [0] on a string yields its first CHARACTER — every single-alias
        # option came out named "-".
        $name = @($names | Sort-Object -Property Length -Descending)[0]
        $desc = ''
        for ($j = $i + 1; $j -lt [Math]::Min($i + 4, $lines.Count); $j++) {
            $d = [regex]::Match($lines[$j], '\[Description\("(.*)"\)\]')
            if ($d.Success) { $desc = $d.Groups[1].Value; break }
        }
        $desc = $desc -replace '\\"', '"' -replace '\|', '\|'
        $rows.Add([pscustomobject]@{ Command = $command; Name = $name; Description = $desc })
    }
}

# Preserve hand-written Class values, keyed on command + option name.
$existing = @{}
$doc = Get-Content -LiteralPath $docPath -Raw
$currentCmd = ''
foreach ($line in ($doc -split "`r?`n")) {
    if ($line -match '^##\s+(\S+)') { $currentCmd = $Matches[1]; continue }
    $r = [regex]::Match($line, '^\|\s*`([^`]+)`\s*\|([^|]*)\|')
    if ($r.Success) { $existing["$currentCmd|$($r.Groups[1].Value)"] = $r.Groups[2].Value.Trim() }
}

$sb = [System.Text.StringBuilder]::new()
foreach ($group in $rows | Group-Object Command | Sort-Object Name) {
    [void]$sb.AppendLine("## $($group.Name)")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine('| Option | Class | Description |')
    [void]$sb.AppendLine('|---|---|---|')
    foreach ($row in $group.Group | Sort-Object -Property Name -Unique) {
        $cls = $existing["$($group.Name)|$($row.Name)"]
        [void]$sb.AppendLine("| ``$($row.Name)`` | $cls | $($row.Description) |")
    }
    [void]$sb.AppendLine()
}

Write-Host "options found: $($rows.Count) across $(($rows | Group-Object Command).Count) command files"
# Compare on normalised line endings — a CRLF/LF difference is not staleness, and comparing raw
# text made -Check report stale immediately after a successful regeneration.
$normalise = { param($t) ($t -replace "`r`n", "`n").TrimEnd() }
$expected = & $normalise $sb.ToString()
$actual   = & $normalise $doc

if ($Check) {
    if (-not $actual.Contains($expected)) {
        Write-Host 'cli-option-inventory.md tables are stale - run scripts/gen-cli-option-inventory.ps1'
        exit 1
    }
    Write-Host 'tables are current.'
    exit 0
}
$sb.ToString() | Set-Content -LiteralPath (Join-Path $root 'docs\cli-option-inventory.generated.md') -NoNewline
Write-Host "wrote docs/cli-option-inventory.generated.md"
