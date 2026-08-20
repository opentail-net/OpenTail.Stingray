<#
.SYNOPSIS
    Replicates the cpu-performance-baseline.md benchmark methodology.
    Best-of-three interleaved rounds across available models, greedy, -g 0.
    Models run in the same order as the baseline doc.
#>

$ErrorActionPreference = 'Stop'

$root     = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelDir = Join-Path $root 'models'
$cliProj  = Join-Path $root 'src\OpenTail.Stingray.Cli\OpenTail.Stingray.Cli.csproj'

# Prompt designed to be ~405-465 tokens (same target as baseline doc)
$prompt = "The following is a detailed technical description for benchmarking purposes. Please summarize the key points in exactly three sentences.`n`nOpenTail.Stingray is a high-performance local inference engine built for running large language models, vision-language models, and diffusion-based generative AI on commodity hardware. The engine implements a zero-copy memory-mapped weight loading system that allows multi-gigabyte model checkpoints to be accessed in milliseconds without any managed heap allocation overhead. The CPU execution path uses hand-tuned AVX2 and FMA SIMD kernels for all quantization formats including Q4_K, Q5_K, Q6_K, Q8_0, FP16, and BF16, achieving near-theoretical memory bandwidth utilization during matrix-vector multiply operations. The GPU acceleration subsystem supports both Vulkan compute shaders and CUDA kernels, enabling cross-vendor GPU inference on AMD, Intel, Apple Silicon, and NVIDIA hardware without vendor-locked runtime dependencies. Speculative decoding is implemented natively via DSpark, EAGLE-3, and Multi-Token Prediction, delivering between 1.8 and 3.2 times inference speedup while maintaining byte-identical output with standard autoregressive decoding."

# Models matching the baseline doc (use those available locally)
$models = @(
    @{ Name = "SmolLM2-1.7B Q4_K_M";   File = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf" },
    @{ Name = "Qwen3-0.6B Q8_0";        File = "Qwen3-0.6B-Q8_0.gguf" },
    @{ Name = "Qwen3-4B Q4_K_M";        File = "Qwen3-4B-Q4_K_M.gguf" },
    @{ Name = "Qwen2.5-0.5B Q4_K_M";   File = "qwen2.5-0.5b-instruct-q4_k_m.gguf" },
    @{ Name = "Qwen2.5-1.5B Q4_K_M";   File = "qwen2.5-1.5b-instruct-q4_k_m.gguf" }
)

# Filter to only models actually on disk
$models = $models | Where-Object { Test-Path (Join-Path $modelDir $_.File) }

if ($models.Count -eq 0) { Write-Error "No benchmark models found in $modelDir"; exit 1 }

Write-Host ""
Write-Host "=============================================================================="
Write-Host " CPU PERFORMANCE BASELINE — replicating cpu-performance-baseline.md"
Write-Host " Method: 3 interleaved rounds, greedy, -g 0, -n 24"
Write-Host "=============================================================================="
Write-Host ""

# Per-model best-of-3 accumulators
$results = @{}
foreach ($m in $models) {
    $results[$m.Name] = @{ PrefillBest = 0.0; DecodeBest = 0.0 }
}

# 3 interleaved rounds
for ($round = 1; $round -le 3; $round++) {
    Write-Host "--- Round $round ---"
    foreach ($m in $models) {
        $modelPath = Join-Path $modelDir $m.File
        $output = dotnet run --project $cliProj -c Release -- `
            -m $modelPath `
            -p $prompt `
            -n 24 `
            -g 0 `
            --single-turn `
            2>&1

        # Parse "Prefill: N tokens, X t/s | Decode: N tokens, Y t/s"
        $line = $output | Where-Object { $_ -match 'Prefill:.*Decode:' } | Select-Object -Last 1
        if ($line -match 'Prefill:\s*\d+\s*tokens,\s*([\d.]+)\s*t/s.*Decode:\s*\d+\s*tokens,\s*([\d.]+)\s*t/s') {
            $prefill = [double]$Matches[1]
            $decode  = [double]$Matches[2]
            $r = $results[$m.Name]
            if ($prefill -gt $r.PrefillBest) { $r.PrefillBest = $prefill }
            if ($decode  -gt $r.DecodeBest)  { $r.DecodeBest  = $decode }
            Write-Host ("  {0,-26}  prefill {1,6:F1} t/s   decode {2,6:F1} t/s" -f $m.Name, $prefill, $decode)
        } else {
            Write-Host ("  {0,-26}  (no timing output found)" -f $m.Name)
            Write-Host $output
        }
    }
    Write-Host ""
}

Write-Host ""
Write-Host "=============================================================================="
Write-Host " RESULTS — Best-of-3 (same methodology as cpu-performance-baseline.md)"
Write-Host "=============================================================================="
Write-Host ""
Write-Host ("| {0,-26} | {1,11} | {2,10} | {3,18} |" -f "Model", "Prefill t/s", "Decode t/s", "prefill ÷ decode")
Write-Host ("| {0,-26} | {1,11} | {2,10} | {3,18} |" -f ('-' * 26), '---:', '---:', '---:')
foreach ($m in $models) {
    $r = $results[$m.Name]
    $ratio = if ($r.DecodeBest -gt 0) { "{0:F1}x" -f ($r.PrefillBest / $r.DecodeBest) } else { 'N/A' }
    Write-Host ("| {0,-26} | {1,11:F1} | {2,10:F1} | {3,18} |" -f $m.Name, $r.PrefillBest, $r.DecodeBest, $ratio)
}
Write-Host ""
Write-Host "Baseline doc (2026-08-07) for comparison:"
Write-Host "| SmolLM2-1.7B Q4_K_M       |       141.3 |       21.3 |              6.6x |"
Write-Host "| Qwen3-0.6B Q8_0           |        93.0 |       47.4 |              2.0x |"
Write-Host ""
