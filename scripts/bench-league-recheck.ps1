# Re-check OUR side (C#, t/s) of PerformanceLeague.md for the LLM text rows: one results file, one model at a time.
#
# For every model it runs three prompt sizes on each backend and parses the CLI's
#   "Prefill: <N> tokens, <X> t/s | Decode: <M> tokens, <Y> t/s" line:
#     short  ~15-token prompt, 64 new tokens -> "decode (short ctx)" and a small-prompt prefill
#     500    docs/reference/benchmark-prompt.txt (~490 tokens), 16 new tokens -> "prefill (~500 tok)"
#     2k     that prompt x4 (~2,000 tokens), 24 new tokens -> "prefill (~2k tok)" and "decode @2k ctx"
# on CPU (-g 0) and Vulkan (-g -1). The CPU 2k scenario is off by default (it takes minutes per run on 7B+ models): -CpuLong.
#
# Results stream into -OutFile (markdown table rows, one per scenario) and the LAST thing written is a marker file
# "<OutFile>.done" containing one summary line, so a watcher can wait for that file instead of waking per result:
#     until [ -f "<OutFile>.done" ]; do sleep 30; done
#
# Rules this follows: never two model jobs at once (it refuses to start while a stingray / test process is running and
# runs strictly sequentially); default JIT settings (no DOTNET_TC_QuickJitForLoops=0, per the no-TC=0 rule); best-of-N
# because interference on this machine can only slow a run down; one failed run is recorded and the script moves on.
#
# Usage:
#   pwsh scripts/bench-league-recheck.ps1                          # -Set quick, cpu+vulkan, 2 runs each
#   pwsh scripts/bench-league-recheck.ps1 -Set full -Runs 3
#   pwsh scripts/bench-league-recheck.ps1 -Models SmolLM2-1.7B-Instruct-Q4_K_M,Qwen3-4B-Q4_K_M -Backends vulkan
#
# The numbers go into PerformanceLeague.md BY HAND, as dated rows, after reading them (rule 12: check the timing, not just
# that a row exists; a model that is missing here is reported as SKIP, never as a result).
param(
    [ValidateSet('quick', 'full')][string]$Set = 'quick',
    [string[]]$Models = @(),                      # file stems (without .gguf); overrides -Set
    [ValidateSet('cpu', 'vulkan')][string[]]$Backends = @('cpu', 'vulkan'),
    [int]$Runs = 2,
    [switch]$CpuLong,                             # also run the ~2k scenario on CPU
    [int]$TimeoutMinutes = 40,                    # per run
    [string]$OutFile = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$exe = Join-Path $repo 'src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe'
if (-not (Test-Path $exe)) { throw "Build the CLI in Release first: dotnet build src/OpenTail.Stingray.Cli -c Release" }

$quick = @('SmolLM2-1.7B-Instruct-Q4_K_M', 'Qwen3-4B-Q4_K_M', 'Qwen3.5-4B-Q4_K_M', 'gemma-4-E4B-it-Q4_K_M', 'Mistral-7B-Instruct-v0.3-Q4_K_M')
$full = $quick + @('SmolLM2-135M-Instruct-Q4_K_M', 'SmolLM2-360M-Instruct-Q4_K_M', 'qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-3b-instruct-q4_k_m',
    'Qwen3-0.6B-Q8_0', 'Qwen3-8B-Q4_K_M', 'Qwen3.5-9B-Q4_K_M', 'gemma-3-4b-it-Q4_K_M', 'gemma-4-12b-it-Q4_K_M',
    'OLMoE-1B-7B-0924-Instruct-Q4_K_M', 'gpt-oss-20b-MXFP4', 'Qwen2.5-VL-7B-Instruct-Q4_K_M', 'Phi-3-mini-4k-instruct-Q4_K_M')
$list = if ($Models.Count -gt 0) { $Models } elseif ($Set -eq 'full') { $full } else { $quick }

if (-not $OutFile) { $OutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("league-recheck-{0:yyyyMMdd-HHmm}.md" -f (Get-Date)) }
$doneFile = "$OutFile.done"
Remove-Item -Path $doneFile -ErrorAction SilentlyContinue

# Never two model jobs at once: wait (up to 15 min) for any other stingray / test process to finish, else refuse.
function Get-ModelJobs { Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'stingray*' -or $_.ProcessName -like 'OpenTail.Stingray.Tests*' -or $_.ProcessName -like 'sd-cli*' -or $_.ProcessName -like 'llama-*' } }
$waited = 0
while ((Get-ModelJobs).Count -gt 0) {
    if ($waited -ge 900) { throw "Another model job is running ($((Get-ModelJobs | Select-Object -First 1).ProcessName)); not starting a second one." }
    Start-Sleep -Seconds 15; $waited += 15
}

# Prompt files (passed with -f: long prompts exceed the Windows command-line limit with -p).
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) 'league-recheck-prompts'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$pShort = Join-Path $tmp 'short.txt'
$p500 = Join-Path $repo 'docs/reference/benchmark-prompt.txt'
$p2k = Join-Path $tmp 'x4.txt'
Set-Content -Path $pShort -Value 'Explain in two sentences why the sky is blue.' -NoNewline -Encoding utf8
$base = (Get-Content -Raw -Path $p500)
Set-Content -Path $p2k -Value ($base + ' ' + $base + ' ' + $base + ' ' + $base) -NoNewline -Encoding utf8

function Find-Model([string]$stem) {
    foreach ($d in @('models', 'models/_models')) {
        $p = Join-Path $repo "$d/$stem.gguf"
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Invoke-One([string]$model, [string]$backend, [string]$promptFile, [int]$n) {
    $ngl = if ($backend -eq 'vulkan') { '-1' } else { '0' }
    $psi = [System.Diagnostics.ProcessStartInfo]::new($exe)
    foreach ($a in @('-m', $model, '-f', $promptFile, '-n', "$n", '--temp', '0', '-g', $ngl)) { [void]$psi.ArgumentList.Add($a) }
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $errTask = $proc.StandardError.ReadToEndAsync()
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutMinutes * 60 * 1000)) { try { $proc.Kill($true) } catch {}; return @{ ok = $false; why = "timeout after $TimeoutMinutes min" } }
    $text = $outTask.Result + "`n" + $errTask.Result
    if ($text -match 'Prefill:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s\s*\|\s*Decode:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s') {
        return @{ ok = $true; pt = [int]$Matches[1]; pf = [double]$Matches[2]; dt = [int]$Matches[3]; df = [double]$Matches[4] }
    }
    return @{ ok = $false; why = "no Prefill/Decode line (exit $($proc.ExitCode))" }
}

$commit = (git rev-parse --short HEAD 2>$null)
$dirty = if ((git status --porcelain 2>$null | Measure-Object).Count -gt 0) { '+dirty' } else { '' }
$started = Get-Date
@"
# League recheck, our side (C#), $($started.ToString('yyyy-MM-dd HH:mm'))

Commit ``$commit$dirty``; engine defaults (no env overrides); best of $Runs run(s); prompts: short (~15 tok), 500 (~490 tok), 2k (~2,000 tok; SnapKV auto-budget 2048 applies).
Machine: Ryzen 7 5700G + integrated Radeon (iGPU; rule 13: do not generalise to discrete GPUs).

| Model | Backend | Scenario | Prompt tok | Prefill t/s (best) | Decode t/s (best) | All runs (prefill / decode) |
|---|---|---|---:|---:|---:|---|
"@ | Set-Content -Path $OutFile -Encoding utf8

$ok = 0; $bad = 0; $skipped = 0
foreach ($stem in $list) {
    $path = Find-Model $stem
    if (-not $path) { Add-Content -Path $OutFile -Value "| $stem | - | - | - | SKIP | SKIP | checkpoint not present |"; $skipped++; continue }
    foreach ($backend in $Backends) {
        $scenarios = @(@('short', $pShort, 64), @('500', $p500, 16))
        if ($backend -eq 'vulkan' -or $CpuLong) { $scenarios += , @('2k', $p2k, 24) }
        foreach ($sc in $scenarios) {
            $name = $sc[0]; $file = $sc[1]; $n = [int]$sc[2]
            $pf = @(); $df = @(); $tok = 0; $why = ''
            for ($r = 0; $r -lt $Runs; $r++) {
                $res = Invoke-One $path $backend $file $n
                if ($res.ok) { $pf += $res.pf; $df += $res.df; $tok = $res.pt } else { $why = $res.why }
            }
            if ($pf.Count -eq 0) {
                Add-Content -Path $OutFile -Value "| $stem | $backend | $name | - | FAIL | FAIL | $why |"; $bad++
            } else {
                $bp = ($pf | Measure-Object -Maximum).Maximum; $bd = ($df | Measure-Object -Maximum).Maximum
                $runsTxt = (0..($pf.Count - 1) | ForEach-Object { '{0:F1}/{1:F1}' -f $pf[$_], $df[$_] }) -join ', '
                Add-Content -Path $OutFile -Value ("| {0} | {1} | {2} | {3} | {4:F1} | {5:F1} | {6} |" -f $stem, $backend, $name, $tok, $bp, $bd, $runsTxt); $ok++
            }
        }
    }
}

$elapsed = (Get-Date) - $started
$summary = "LEAGUE-RECHECK DONE results=$ok failed=$bad skipped=$skipped elapsed=$([int]$elapsed.TotalMinutes)min file=$OutFile"
Add-Content -Path $OutFile -Value "`n$summary"
Set-Content -Path $doneFile -Value $summary -Encoding utf8
Write-Output $summary
