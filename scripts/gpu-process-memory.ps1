<#
.SYNOPSIS
    Reads the operating system's per-process GPU memory counters for a stingray run, to compare with the planner's GPU weight figure.
.DESCRIPTION
    Starts the CLI on a model with every layer offloaded, waits for the model to load and decode, then samples the Windows
    "GPU Process Memory" counters (Shared, Dedicated, Total Committed) for that process. On an integrated GPU the memory is
    "shared" system RAM, so the total is the figure to read. The counters include driver overhead, the KV cache, scratch
    buffers and activations, so they are an upper bound on weights: compare the ratio between models, not one absolute number.
    Windows only (the counters are a Windows feature). A run that finishes before the sample prints nothing: raise -Tokens.
.EXAMPLE
    pwsh scripts/gpu-process-memory.ps1 -Model E:\_models\qwen35-0.8b\Qwen3.5-0.8B-Q8_0.gguf -Exe src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe
#>
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Exe,
    [int]$Context = 512,
    [int]$Tokens = 6000,
    [int]$WarmupSeconds = 12
)
$ErrorActionPreference = 'Stop'
$psi = [System.Diagnostics.ProcessStartInfo]::new($Exe)
foreach ($a in @('-m', $Model, '-p', 'Write a long story about a dragon.', '-n', "$Tokens", '--temp', '0', '-ngl', '-1', '-c', "$Context")) { $psi.ArgumentList.Add($a) }
$psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
$p = [System.Diagnostics.Process]::Start($psi)
# Drain the pipes so a chatty run cannot block on a full buffer.
$null = $p.StandardOutput.BaseStream.CopyToAsync([System.IO.Stream]::Null)
$null = $p.StandardError.BaseStream.CopyToAsync([System.IO.Stream]::Null)
try {
    Start-Sleep -Seconds $WarmupSeconds
    $last = $null
    for ($i = 0; $i -lt 4 -and -not $p.HasExited; $i++) {
        $s = Get-Counter -Counter '\GPU Process Memory(*)\Shared Usage', '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Total Committed' -ErrorAction SilentlyContinue
        $last = foreach ($c in $s.CounterSamples) {
            if ($c.InstanceName -like "pid_$($p.Id)_*") { [pscustomobject]@{ Counter = $c.Path.Split([char]92)[-1]; MiB = [math]::Round($c.CookedValue / 1MB, 1) } }
        }
        Start-Sleep -Seconds 1
    }
    if ($null -eq $last) { Write-Warning 'The process ended before it could be sampled; raise -Tokens.' } else { $last | Format-Table -AutoSize }
}
finally { try { if (-not $p.HasExited) { $p.Kill($true) } } catch {} }
