# End-to-end CPU prefill A/B via the real CLI (perf loop, iteration 33 follow-up).
#
# Isolated kernel benchmarks are NOT sufficient to ship here — see the log's iteration 24, where a
# reproduced 2.4-2.6x isolated win became a real ~12% end-to-end LOSS under production's
# Parallel.For contention. This script is the end-to-end gate for any prefill-attention change.
#
# Usage: pwsh scripts/bench-prefill-cli.ps1 [-Words 200,1200,2400] [-Runs 3] [-Label baseline]
param(
    [int[]]$Words = @(200, 1200, 2400),
    [int]$Runs = 3,
    [string]$Label = "run",
    [string]$Model = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    [int]$Ctx = 8192
)

$ErrorActionPreference = "Stop"
$env:DOTNET_TC_QuickJitForLoops = "0"   # tiered JIT invalidates these numbers (iteration 11)
$exe = "src/OpenTail.Stingray.Cli/bin/Release/net10.0/opentail-llm-cli.exe"
if (-not (Test-Path $exe)) { throw "Build the CLI in Release first: dotnet build src/OpenTail.Stingray.Cli -c Release" }

Write-Host "=== $Label ===" -ForegroundColor Cyan
$all = @()
foreach ($w in $Words) {
    # Pass the prompt via -f, not -p. Beyond roughly a thousand words the command line exceeds the
    # Windows limit and the CLI fails to start with "The filename or extension is too long" — which
    # surfaces as an empty result set rather than an obvious error. The CLI's own -f help says it is
    # "useful for prompts longer than the shell's command-line limit".
    $promptFile = Join-Path ([System.IO.Path]::GetTempPath()) "opentail-prefill-$w.txt"
    Set-Content -Path $promptFile -Value ("data " * $w).Trim() -NoNewline -Encoding utf8
    $samples = @()
    for ($r = 0; $r -lt $Runs; $r++) {
        $out = & $exe -m $Model -f $promptFile -n 1 --temp 0 -c $Ctx -g 0 2>&1 | Out-String
        # "Prefill: <N> tokens, <X> t/s"
        if ($out -match 'Prefill:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s') {
            $samples += [double]$Matches[2]
            $tokens = [int]$Matches[1]
        } else {
            Write-Warning "no prefill line parsed (run $r, $w words)"
        }
    }
    if ($samples.Count -gt 0) {
        # Best-of, not mean: interference on this box is one-sided (it can only slow a run down),
        # so the max observed throughput is the cleanest estimator of the code's real speed.
        $best = ($samples | Measure-Object -Maximum).Maximum
        # PowerShell's [int] cast rounds rather than truncates: [int](3 / 2) is 2, which selected
        # the maximum of three sorted samples and mislabeled it as the median. Floor explicitly.
        $medIndex = [int][Math]::Floor($samples.Count / 2.0)
        $med = ($samples | Sort-Object)[$medIndex]
        Write-Host ("{0,5} words / {1,5} tok : best {2,7:F1} t/s   median {3,7:F1}   samples {4}" -f `
            $w, $tokens, $best, $med, (($samples | ForEach-Object { "{0:F1}" -f $_ }) -join ", "))
        $all += [pscustomobject]@{ Label = $Label; Words = $w; Tokens = $tokens; Best = $best; Median = $med }
    }
}
$all
