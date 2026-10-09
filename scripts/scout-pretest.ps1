<#
.SYNOPSIS
    Opt-in, non-destructive pretest wrapper around `stingray scout`: static report, memory gate, then (only with -Run) a bounded smoke run and an optional golden check, with one receipt.

.DESCRIPTION
    Chains EXISTING commands; it implements no inference and decides nothing about admission.

      0 artifact             `scout` reads the GGUF index (no weights).
      1 static_contract      confirmed blockers from scout.
      2 feasibility          scout's memory gate for -Budget / -Reserve / -ContextSize. Anything but `allowed` stops here: Blocked, nothing is run.
      3 smoke                -Run only. Admitted architecture: a real CPU generation (8 tokens, greedy). Otherwise `admit-arch` (explicit, unverified bypass run).
                             Passed only with weight-load evidence in the output, never on exit code alone (CLAUDE.md rule 12).
      4 internal_consistency NotRun (no command exposes it).
      5 independent_reference -Run with -Golden: `admit-arch --golden`. Otherwise NotRun.
      6 admission_readiness  Always NotRun: a person decides, from the receipt plus the playbook.

    Guarantees:
      * Never writes to, moves or deletes a checkpoint, never downloads, never edits the repo, never changes admission or status. It writes ONE file: the receipt in -OutDir.
        (-ComputeHash additionally lets `stingray hash` write its usual <file>.sha256 cache beside the model; without it an existing cache is only read.)
      * Takes the repo's shared heavy-run gate (named mutex Global\OpenTailStingray.HeavyTests) before any run, so it cannot overlap the heavy test suites or capture-golden.
      * A run that exceeds -TimeoutSeconds, or is interrupted, has its whole process tree killed; the gate is always released.
      * Receipts carry file names only: no absolute paths, user names or drive letters.

    Exit codes: 0 = everything attempted passed (or nothing was attempted), 1 = a stage Failed, 2 = Blocked (gate, budget, or another heavy run), 64 = bad arguments, 66 = model missing.

    Documented in docs/reference/061-coverage-tooling.md. Plan: docs/3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md section 8.2.

.PARAMETER Model          GGUF file to examine.
.PARAMETER Budget         Host RAM budget for the gate (e.g. 64G). Default 64G.
.PARAMETER Reserve        Headroom kept free under the budget. Default 8G.
.PARAMETER ContextSize    Context the memory estimate and the runs use. Default 2048.
.PARAMETER Golden         Golden reference JSON (from capture-golden) for stage 5.
.PARAMETER Run            Actually execute stages 3 and 5. Without it the script only plans: scout, gate, and what it WOULD run.
.PARAMETER TimeoutSeconds Per-stage wall-clock limit; the process tree is killed past it. Default 900.
.PARAMETER GateWaitSeconds How long to wait for the shared heavy-run gate before reporting Blocked. Default 600.
.PARAMETER Cli            Path to stingray.exe. Default: the Release build in this repo.
.PARAMETER OutDir         Where the receipt is written. Default: <temp>/scout-pretest.
.PARAMETER ComputeHash    Also compute the SHA-256 (slow for big files; writes the usual .sha256 cache beside the model).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Budget = '64G',
    [string]$Reserve = '8G',
    [int]$ContextSize = 2048,
    [string]$Golden,
    [switch]$Run,
    [int]$TimeoutSeconds = 900,
    [int]$GateWaitSeconds = 600,
    [string]$Cli,
    [string]$OutDir = (Join-Path ([IO.Path]::GetTempPath()) 'scout-pretest'),
    [switch]$ComputeHash
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Cli) { $Cli = Join-Path $repo 'src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe' }

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { Write-Error "Model file not found: $(Split-Path $Model -Leaf)" -ErrorAction Continue; exit 66 }
if (-not (Test-Path -LiteralPath $Cli -PathType Leaf)) { Write-Error "CLI not found. Build it first: dotnet build src/OpenTail.Stingray.Cli -c Release" -ErrorAction Continue; exit 64 }
if ($Golden -and -not (Test-Path -LiteralPath $Golden -PathType Leaf)) { Write-Error "Golden file not found: $(Split-Path $Golden -Leaf)" -ErrorAction Continue; exit 66 }
if ($ContextSize -le 0 -or $TimeoutSeconds -le 0) { Write-Error 'ContextSize and TimeoutSeconds must be positive.' -ErrorAction Continue; exit 64 }
if ($Golden -and -not $Run) { Write-Warning '-Golden has no effect without -Run (planning only).' }

$modelFull = (Resolve-Path -LiteralPath $Model).Path
$modelName = Split-Path $modelFull -Leaf
$modelInfo = Get-Item -LiteralPath $modelFull

function Now { [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ') }

# ── child-process runner: timeout, whole-tree kill, captured output, peak working set ──────────
function Invoke-Bounded {
    param([string[]]$Arguments, [int]$Seconds, [hashtable]$Env = @{})
    $psi = [Diagnostics.ProcessStartInfo]::new($Cli)
    foreach ($a in $Arguments) { $psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8; $psi.StandardErrorEncoding = [Text.Encoding]::UTF8
    foreach ($k in $Env.Keys) { $psi.Environment[$k] = [string]$Env[$k] }
    $p = [Diagnostics.Process]::new(); $p.StartInfo = $psi
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $null = $p.Start()
    $stdout = $p.StandardOutput.ReadToEndAsync(); $stderr = $p.StandardError.ReadToEndAsync()
    $peak = 0L; $timedOut = $false
    try {
        while (-not $p.WaitForExit(500)) {
            try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch { }
            if ($sw.Elapsed.TotalSeconds -gt $Seconds) { $timedOut = $true; break }
        }
    }
    finally {
        if (-not $p.HasExited) { try { $p.Kill($true) } catch { } ; try { $null = $p.WaitForExit(10000) } catch { } }
    }
    $sw.Stop()
    $code = if ($timedOut) { $null } else { $p.ExitCode }
    [pscustomobject]@{ ExitCode = $code; TimedOut = $timedOut; Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); PolledPeakMiB = [math]::Round($peak / 1MB); Stdout = $stdout.GetAwaiter().GetResult(); Stderr = $stderr.GetAwaiter().GetResult() }
}

$stages = [Collections.Generic.List[object]]::new()
function Add-Stage($name, $state, $detail, $extra = @{}) {
    $o = [ordered]@{ stage = $name; state = $state; detail = $detail }
    foreach ($k in $extra.Keys) { $o[$k] = $extra[$k] }
    $stages.Add([pscustomobject]$o)
}

# ── checkpoint identity (file name, size, hash state) ──────────────────────────────────────────
$sha = $null; $shaState = 'not_computed'
$sidecar = "$modelFull.sha256"
if (Test-Path -LiteralPath $sidecar) {
    $parts = (Get-Content -LiteralPath $sidecar -Raw).Trim() -split '\s+'
    if ($parts.Count -ge 3 -and $parts[0].Length -eq 64 -and [int64]$parts[1] -eq $modelInfo.Length -and [int64]$parts[2] -eq $modelInfo.LastWriteTimeUtc.Ticks) { $sha = $parts[0].ToLowerInvariant(); $shaState = 'cached' }
}
if (-not $sha -and $ComputeHash) {
    $h = Invoke-Bounded -Arguments @('hash', '-m', $modelFull) -Seconds ([Math]::Max($TimeoutSeconds, 3600))
    if ($h.Stdout -match '\b([0-9a-fA-F]{64})\b') { $sha = $Matches[1].ToLowerInvariant(); $shaState = 'computed' } else { $shaState = 'failed' }
}

$cliVersion = ((Invoke-Bounded -Arguments @('--version') -Seconds 60).Stdout).Trim()

# ── stage 0/1/2: scout (static, no weights) ────────────────────────────────────────────────────
$t0 = Now
$s = Invoke-Bounded -Arguments @('scout', '-m', $modelFull, '--format', 'json', '--budget', $Budget, '--reserve', $Reserve, '-c', "$ContextSize") -Seconds 300
$scout = $null
if ($s.ExitCode -eq 0) { try { $scout = $s.Stdout | ConvertFrom-Json } catch { } }
if (-not $scout) {
    Add-Stage '0_artifact' 'Failed' "scout did not produce a report (exit $($s.ExitCode))" @{ started_utc = $t0; ended_utc = (Now) }
}
else {
    Add-Stage '0_artifact' 'Passed' 'GGUF header, metadata and tensor index read by scout (no weights).' @{ started_utc = $t0; ended_utc = (Now); seconds = $s.Seconds }
    $confirmed = @($scout.blockers | Where-Object { $_.kind -eq 'Confirmed' })
    Add-Stage '1_static_contract' $(if ($confirmed.Count) { 'Failed' } else { 'Passed' }) $(if ($confirmed.Count) { 'Confirmed blockers: ' + (($confirmed | ForEach-Object id) -join ', ') } else { 'No confirmed blockers. This is not admission.' })
    $res = $scout.resources
    $decision = $res.execution_decision
    $gateState = switch ($decision) { 'allowed' { 'Passed' } 'blocked' { 'Blocked' } default { 'NotRun' } }
    Add-Stage '2_feasibility' $gateState $res.reason @{ execution_decision = $decision; estimate_bytes = $res.host_working_set.bytes; estimate_certainty = $res.host_working_set.certainty; context_tokens = $res.context_tokens }
}

# ── stages 3-6 ─────────────────────────────────────────────────────────────────────────────────
$arch = $scout.architecture
$admitted = $scout -and $arch.resolved -and $arch.status -eq 'Admitted'
$gate = $null; $ownsGate = $false

try {
    $feasible = $scout -and ($stages | Where-Object stage -eq '2_feasibility').state -eq 'Passed'
    if (-not $scout) { Add-NotRun 3 'No scout report.' }
    elseif (-not $feasible) {
        $why = if (($stages | Where-Object stage -eq '2_feasibility').state -eq 'Blocked') { 'Blocked by the memory gate (see 2_feasibility); nothing was run.' } else { 'Memory gate not assessed; nothing was run.' }
        $st = if (($stages | Where-Object stage -eq '2_feasibility').state -eq 'Blocked') { 'Blocked' } else { 'NotRun' }
        foreach ($n in '3_smoke') { Add-Stage $n $st $why }
        Add-Stage '4_internal_consistency' 'NotRun' 'No command exposes this check.'
        Add-Stage '5_independent_reference' $(if ($Golden) { $st } else { 'NotRun' }) $(if ($Golden) { $why } else { 'No golden supplied.' })
        Add-Stage '6_admission_readiness' 'NotRun' 'A person decides from the receipt and the playbook.'
    }
    elseif (-not $Run) {
        $cmd = if ($admitted) { 'a real CPU generation (8 tokens, greedy)' } else { 'admit-arch (bypassed, unverified run)' }
        Add-Stage '3_smoke' 'NotRun' "Planned only (-Run not given). With -Run: $cmd."
        Add-Stage '4_internal_consistency' 'NotRun' 'No command exposes this check.'
        Add-Stage '5_independent_reference' 'NotRun' $(if ($Golden) { 'Planned only (-Run not given). With -Run: admit-arch --golden.' } else { 'No golden supplied.' })
        Add-Stage '6_admission_readiness' 'NotRun' 'A person decides from the receipt and the playbook.'
    }
    else {
        # shared heavy-run gate
        $gate = [Threading.Mutex]::new($false, 'Global\OpenTailStingray.HeavyTests')
        try { $ownsGate = $gate.WaitOne([TimeSpan]::FromSeconds($GateWaitSeconds)) } catch [Threading.AbandonedMutexException] { $ownsGate = $true }
        if (-not $ownsGate) {
            $why = "Another heavy run holds the shared gate (Global\OpenTailStingray.HeavyTests); waited ${GateWaitSeconds}s."
            Add-Stage '3_smoke' 'Blocked' $why
            Add-Stage '4_internal_consistency' 'NotRun' 'No command exposes this check.'
            Add-Stage '5_independent_reference' $(if ($Golden) { 'Blocked' } else { 'NotRun' }) $(if ($Golden) { $why } else { 'No golden supplied.' })
            Add-Stage '6_admission_readiness' 'NotRun' 'A person decides from the receipt and the playbook.'
        }
        else {
            # stage 3: smoke
            $t = Now
            $env3 = @{ STINGRAY_GC_STATS = '1' }
            if ($admitted) { $args3 = @('-m', $modelFull, '-p', 'The capital of France is', '-n', '8', '--temp', '0', '-g', '0', '-c', "$ContextSize"); $what = 'real CPU generation' }
            else { $args3 = @('admit-arch', '-m', $modelFull, '-n', '8', '--ctx-size', "$ContextSize"); $what = 'admit-arch bypassed run' }
            $r = Invoke-Bounded -Arguments $args3 -Seconds $TimeoutSeconds -Env $env3
            $text = $r.Stdout + "`n" + $r.Stderr
            $gc = if ($text -match 'peakWorkingSet=([0-9]+) MiB') { [int]$Matches[1] } else { $null }
            $loaded = $text -match 'Pre-faulted' -or $text -match 'Ran cleanly'
            $peak = if ($gc) { $gc } else { $r.PolledPeakMiB }
            $extra = @{ started_utc = $t; ended_utc = (Now); seconds = $r.Seconds; exit_code = $r.ExitCode; peak_working_set_mib = $peak; command = $what }
            if ($scout.resources.host_working_set.bytes) {
                $est = [math]::Round($scout.resources.host_working_set.bytes / 1MB)
                $extra.estimate_mib = $est
                $extra.estimate_exceeded = ($peak -gt $est)
            }
            if ($r.TimedOut) { Add-Stage '3_smoke' 'Failed' "Timed out after ${TimeoutSeconds}s; process tree killed." $extra }
            elseif ($r.ExitCode -ne 0) { Add-Stage '3_smoke' 'Failed' ("$what exited $($r.ExitCode): " + (($text -split "`n" | Where-Object { $_ -match 'rror|REJECT|xception' } | Select-Object -First 1))) $extra }
            elseif (-not $loaded) { Add-Stage '3_smoke' 'Failed' "$what exited 0 but printed no weight-load evidence (no 'Pre-faulted' / 'Ran cleanly' line): not counted as a real run." $extra }
            else { Add-Stage '3_smoke' 'Passed' "$what completed with weight-load evidence. Smoke only: this is not admission." $extra }

            Add-Stage '4_internal_consistency' 'NotRun' 'No command exposes this check.'

            # stage 5: golden
            if (-not $Golden) { Add-Stage '5_independent_reference' 'NotRun' 'No golden supplied.' }
            elseif (($stages | Where-Object stage -eq '3_smoke').state -ne 'Passed') { Add-Stage '5_independent_reference' 'NotRun' 'Skipped: the smoke stage did not pass.' }
            else {
                $t = Now
                $g = Invoke-Bounded -Arguments @('admit-arch', '-m', $modelFull, '--golden', (Resolve-Path -LiteralPath $Golden).Path, '--ctx-size', "$ContextSize") -Seconds $TimeoutSeconds -Env $env3
                $gt = $g.Stdout + "`n" + $g.Stderr
                $verdict = ($gt -split "`n" | Where-Object { $_ -match 'GOLDEN MATCH|NOT YET ADMISSIBLE|UNPINNED|LEGACY UNPINNED|Hyperparameter guard failed|Error:' } | Select-Object -First 3) -join ' | '
                $pin = ($gt -split "`n" | Where-Object { $_ -match '^(pinned|UNPINNED|pin):' } | Select-Object -First 1)
                $extra = @{ started_utc = $t; ended_utc = (Now); seconds = $g.Seconds; exit_code = $g.ExitCode; golden_file = (Split-Path $Golden -Leaf); verdict_lines = $verdict; pin = "$pin".Trim() }
                if ($g.TimedOut) { Add-Stage '5_independent_reference' 'Failed' "Timed out after ${TimeoutSeconds}s; process tree killed." $extra }
                elseif ($g.ExitCode -eq 0) { Add-Stage '5_independent_reference' 'Passed' "Golden check passed ($verdict). Evidence for a person to weigh, not admission." $extra }
                elseif ($verdict) { Add-Stage '5_independent_reference' 'Failed' "Golden check failed: $verdict" $extra }
                else { Add-Stage '5_independent_reference' 'Failed' "admit-arch exited $($g.ExitCode) with no verdict line: a tool error, not a parity result." $extra }
            }
            Add-Stage '6_admission_readiness' 'NotRun' 'A person decides from the receipt and the playbook.'
        }
    }
}
finally {
    if ($gate) { if ($ownsGate) { try { $gate.ReleaseMutex() } catch { } } ; $gate.Dispose() }
}

# ── receipt ────────────────────────────────────────────────────────────────────────────────────
$receipt = [ordered]@{
    schema_version = 1
    kind           = 'scout_pretest_receipt'
    generated_utc  = (Now)
    cli_version    = $cliVersion
    scout_build    = $scout.scout_build
    checkpoint     = [ordered]@{ file_name = $modelName; file_bytes = $modelInfo.Length; sha256 = $sha; sha256_state = $shaState }
    parameters     = [ordered]@{ budget = $Budget; reserve = $Reserve; context_tokens = $ContextSize; run = [bool]$Run; golden_file = $(if ($Golden) { Split-Path $Golden -Leaf } else { $null }); timeout_seconds = $TimeoutSeconds }
    architecture   = if ($scout) { [ordered]@{ declared = $arch.declared; resolved = $arch.resolved; descriptor_id = $arch.descriptor_id; status = $arch.status; forward_pass_family = $arch.forward_pass_family } } else { $null }
    stages         = $stages
    admission      = 'not_decided'
    note           = 'This receipt is evidence. It never admits an architecture; see docs/reference/architecture-admission-agent-playbook.md.'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$outFile = Join-Path $OutDir (($modelName -replace '[^A-Za-z0-9._-]', '_') + ".pretest.$stamp.json")
($receipt | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $outFile -Encoding utf8

foreach ($st in $stages) { '{0,-26} {1,-8} {2}' -f $st.stage, $st.state, $st.detail }
"Receipt: $outFile"

$states = @($stages | ForEach-Object state)
if ($states -contains 'Failed') { exit 1 }
if ($states -contains 'Blocked') { exit 2 }
exit 0
