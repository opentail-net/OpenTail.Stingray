<#
.SYNOPSIS
    Runs C++ reference benchmarks (audio.cpp, whisper.cpp) and compares against C# Stingray figures.

.DESCRIPTION
    Automates executing C++ counterpart benchmarks on CPU (or GPU when supported), extracts wall-clock,
    audio duration, and RTF metrics, calculates C# / C++ ratios, and outputs formatted Markdown tables
    directly matching PerformanceLeague.md.

.PARAMETER Suite
    Benchmark suite: 'All', 'Tts', 'Asr', 'Whisper', 'CosyVoice', 'Chatterbox', 'QwenTts', 'FishAudio'. Default is 'All'.

.PARAMETER Threads
    Number of CPU worker threads. Defaults to 6 (dev box configuration).

.PARAMETER Prompt
    TTS text prompt for synthesis benchmarks. Defaults to "Hello, I will make some lunch, darling!".

.PARAMETER OutFile
    Optional file path to write Markdown results.

.EXAMPLE
    .\scripts\bench-cpp.ps1 -Suite All
    .\scripts\bench-cpp.ps1 -Suite Whisper -Threads 6
    .\scripts\bench-cpp.ps1 -Suite Tts -OutFile "docs/cpp-benchmark-results.md"
#>

[CmdletBinding()]
param(
    [ValidateSet("All", "Tts", "Asr", "Whisper", "CosyVoice", "Chatterbox", "QwenTts", "FishAudio")]
    [string]$Suite = "All",

    [int]$Threads = 6,

    [string]$Prompt = "Hello, I will make some lunch, darling!",

    [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"
$today = (Get-Date).ToString("yyyy-MM-dd")
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " OPEN-TAIL STINGRAY - C++ REFERENCE BENCHMARK HARNESS" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Date:    $today"
Write-Host " Suite:   $Suite"
Write-Host " Threads: $Threads"
Write-Host " Prompt:  `"$Prompt`""
Write-Host ""

# 1. Locate C++ Executables
$audioCppCli = Join-Path $repoRoot "examples\audio.cpp\build\bin\audiocpp_cli.exe"
$whisperCli  = Join-Path $repoRoot "examples\whisper.cpp\build\bin\Release\whisper-cli.exe"
$audioCppDir = Join-Path $repoRoot "examples\audio.cpp"
$whisperDir  = Join-Path $repoRoot "examples\whisper.cpp"
$s2Cli       = Join-Path $repoRoot "examples\s2.cpp\build\s2.exe"
$s2Dir       = Join-Path $repoRoot "examples\s2.cpp"
$voiceRef    = Join-Path $audioCppDir "assets\resources\b.wav"
$refText     = "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years."

$hasAudioCpp = Test-Path $audioCppCli
$hasWhisper  = Test-Path $whisperCli
$hasS2       = Test-Path $s2Cli

Write-Host "  audio.cpp CLI:   " -NoNewline
if ($hasAudioCpp) { Write-Host $audioCppCli -ForegroundColor Green } else { Write-Host "NOT FOUND" -ForegroundColor Yellow }

Write-Host "  whisper.cpp CLI: " -NoNewline
if ($hasWhisper) { Write-Host $whisperCli -ForegroundColor Green } else { Write-Host "NOT FOUND" -ForegroundColor Yellow }

Write-Host "  s2.cpp CLI:      " -NoNewline
if ($hasS2) { Write-Host $s2Cli -ForegroundColor Green } else { Write-Host "NOT FOUND" -ForegroundColor Yellow }
Write-Host ""

$results = [System.Collections.Generic.List[PSCustomObject]]::new()

# Helper function to run command and capture metrics
function Run-AudioCppTask {
    param(
        [string]$Name,
        [string]$Family,
        [string]$Task,
        [string]$ModelRelPath,
        [string]$Scenario,
        [double]$CsRtf,
        [double]$CsWallSec,
        [hashtable]$ExtraArgs = @{}
    )

    $modelPath = Join-Path $audioCppDir $ModelRelPath
    if (-not (Test-Path $modelPath)) {
        Write-Host "  [SKIP] $Name - Model not found: $modelPath" -ForegroundColor Yellow
        return
    }

    Write-Host "  [RUN] $Name ($Family) ... " -NoNewline -ForegroundColor Cyan

    $argsList = @(
        "--task", $Task,
        "--family", $Family,
        "--model", $modelPath,
        "--backend", "cpu",
        "--threads", $Threads.ToString(),
        "--text", $Prompt,
        "--metrics",
        "--out", "temp_bench_out.wav"
    )

    foreach ($k in $ExtraArgs.Keys) {
        $argsList += $k
        if ($ExtraArgs[$k] -ne $null -and $ExtraArgs[$k] -ne "") {
            $argsList += $ExtraArgs[$k]
        }
    }

    try {
        $pInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $pInfo.FileName = $audioCppCli
        $pInfo.WorkingDirectory = $audioCppDir
        $pInfo.RedirectStandardOutput = $true
        $pInfo.RedirectStandardError = $true
        $pInfo.UseShellExecute = $false
        $pInfo.Arguments = ($argsList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '

        $p = [System.Diagnostics.Process]::Start($pInfo)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()

        $combined = "$stdout`n$stderr"
        $wallMatch = [regex]::Match($combined, 'metrics\.wall_ms=([\d.]+)')
        $durMatch  = [regex]::Match($combined, 'metrics\.audio_duration_ms=([\d.]+)')
        $rtfMatch  = [regex]::Match($combined, 'metrics\.rtf=([\d.]+)')

        if ($rtfMatch.Success) {
            $wallSec = [double]$wallMatch.Groups[1].Value / 1000.0
            $durSec  = [double]$durMatch.Groups[1].Value / 1000.0
            $rtfVal  = [double]$rtfMatch.Groups[1].Value

            $ratioVal = if ($CsRtf -gt 0 -and $rtfVal -gt 0) { $rtfVal / $CsRtf } else { 0.0 }
            $ratioStr = if ($ratioVal -gt 1.0) { "<span style=`"color:#16a34a`">**{0:F2}x**</span>" -f $ratioVal } elseif ($ratioVal -gt 0) { "**{0:F2}x**" -f $ratioVal } else { "-" }

            Write-Host ("DONE ({0:F2}s wall, {1:F2}s audio, RTF={2:F2}x, Ratio={3})" -f $wallSec, $durSec, $rtfVal, $ratioStr) -ForegroundColor Green

            $results.Add([PSCustomObject]@{
                Category   = "TTS"
                Pipeline   = $Name
                Scenario   = $Scenario
                Backend    = "CPU"
                CsWall     = if ($CsWallSec -gt 0) { "{0:F2}s" -f $CsWallSec } else { "-" }
                CsRtf      = if ($CsRtf -gt 0) { "**{0:F2}x**" -f $CsRtf } else { "-" }
                CppRtf     = ("{0:F2}x" -f $rtfVal)
                CppWall    = ("{0:F2}s" -f $wallSec)
                Ratio      = $ratioStr
                Date       = $today
            })
        } else {
            Write-Host "FAILED (Could not parse metrics)" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Helper function to run Whisper.cpp benchmarks
function Run-WhisperCppTask {
    param(
        [string]$Name,
        [string]$ModelRelPath,
        [double]$CsRtf,
        [double]$CsWallSec
    )

    $modelPath = Join-Path $whisperDir $ModelRelPath
    if (-not (Test-Path $modelPath)) {
        # Check models/ or models/_models in repo root
        $fallback = Join-Path $repoRoot "models\$([System.IO.Path]::GetFileName($ModelRelPath))"
        $fallbackModels = Join-Path $repoRoot "models\_models\$([System.IO.Path]::GetFileName($ModelRelPath))"
        if (Test-Path $fallback) {
            $modelPath = $fallback
        } elseif (Test-Path $fallbackModels) {
            $modelPath = $fallbackModels
        } else {
            Write-Host "  [SKIP] Whisper $Name - Model not found ($ModelRelPath)" -ForegroundColor Yellow
            return
        }
    }

    # Generate synthetic 12s wav file if missing
    $wavFile = Join-Path $whisperDir "temp_bench_12s.wav"
    if (-not (Test-Path $wavFile)) {
        $rate = 16000
        $seconds = 12
        $samples = $rate * $seconds
        $bw = [System.IO.BinaryWriter]::new([System.IO.File]::Create($wavFile))
        # WAV Header (PCM 16-bit 16kHz mono)
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
        $bw.Write([int](36 + $samples * 2))
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
        $bw.Write([int]16)
        $bw.Write([int16]1) # PCM
        $bw.Write([int16]1) # 1 channel
        $bw.Write([int]$rate)
        $bw.Write([int]($rate * 2))
        $bw.Write([int16]2)
        $bw.Write([int16]16)
        $bw.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $bw.Write([int]($samples * 2))
        for ($i = 0; $i -lt $samples; $i++) {
            $val = [int16]([Math]::Sin(2.0 * [Math]::PI * 440.0 * $i / $rate) * 10000.0)
            $bw.Write($val)
        }
        $bw.Close()
    }

    Write-Host "  [RUN] Whisper $Name ... " -NoNewline -ForegroundColor Cyan

    $argsList = @(
        "-m", $modelPath,
        "-f", $wavFile,
        "-t", $Threads.ToString(),
        "-nt",
        "-l", "en"
    )

    try {
        $pInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $pInfo.FileName = $whisperCli
        $pInfo.WorkingDirectory = $whisperDir
        $pInfo.RedirectStandardOutput = $true
        $pInfo.RedirectStandardError = $true
        $pInfo.UseShellExecute = $false
        $pInfo.Arguments = ($argsList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '

        $p = [System.Diagnostics.Process]::Start($pInfo)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()

        $combined = "$stdout`n$stderr"
        $totMatch = [regex]::Match($combined, 'total time =\s+([\d.]+)\s+ms')

        if ($totMatch.Success) {
            $totMs = [double]$totMatch.Groups[1].Value
            $wallSec = $totMs / 1000.0
            $rtfVal  = $wallSec / 12.0

            $ratioVal = if ($CsRtf -gt 0 -and $rtfVal -gt 0) { $rtfVal / $CsRtf } else { 0.0 }
            $ratioStr = if ($ratioVal -gt 1.0) { "<span style=`"color:#16a34a`">**{0:F2}x**</span>" -f $ratioVal } elseif ($ratioVal -gt 0) { "**{0:F2}x**" -f $ratioVal } else { "-" }

            Write-Host ("DONE ({0:F2}s wall, RTF={1:F3}x, Ratio={2})" -f $wallSec, $rtfVal, $ratioStr) -ForegroundColor Green

            $results.Add([PSCustomObject]@{
                Category   = "ASR"
                Pipeline   = "Whisper $Name"
                Scenario   = "12s audio transcribe"
                Backend    = "CPU"
                CsWall     = if ($CsWallSec -gt 0) { "{0:F2}s" -f $CsWallSec } else { "-" }
                CsRtf      = if ($CsRtf -gt 0) { "**{0:F3}x**" -f $CsRtf } else { "-" }
                CppRtf     = ("{0:F3}x" -f $rtfVal)
                CppWall    = ("{0:F2}s" -f $wallSec)
                Ratio      = $ratioStr
                Date       = $today
            })
        } else {
            Write-Host "FAILED (Could not parse total time)" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Helper function to run s2.cpp benchmarks (FishSpeech S2 Pro)
function Run-S2CppTask {
    param(
        [string]$Name,
        [string]$ModelRelPath,
        [string]$Scenario,
        [double]$CsRtf,
        [double]$CsWallSec
    )

    $modelPath = Join-Path $repoRoot $ModelRelPath
    if (-not (Test-Path $modelPath)) {
        $fallback = Join-Path "F:\_models" ([System.IO.Path]::GetFileName($ModelRelPath))
        if (Test-Path $fallback) {
            $modelPath = $fallback
        } else {
            Write-Host "  [SKIP] $Name - Model not found: $modelPath" -ForegroundColor Yellow
            return
        }
    }

    $tokPath = Join-Path $s2Dir "tokenizer.json"
    if (-not (Test-Path $tokPath)) {
        Write-Host "  [SKIP] $Name - Tokenizer not found: $tokPath" -ForegroundColor Yellow
        return
    }

    Write-Host "  [RUN] $Name (s2.cpp) ... " -NoNewline -ForegroundColor Cyan

    $argsList = @(
        "-m", $modelPath,
        "-t", $tokPath,
        "--text", $Prompt,
        "--prompt-audio", $voiceRef,
        "--prompt-text", $refText,
        "-threads", $Threads.ToString(),
        "-o", "temp_s2_bench.wav"
    )

    try {
        $pInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $pInfo.FileName = $s2Cli
        $pInfo.WorkingDirectory = $s2Dir
        $pInfo.RedirectStandardOutput = $true
        $pInfo.RedirectStandardError = $true
        $pInfo.UseShellExecute = $false
        $pInfo.Arguments = ($argsList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '

        $p = [System.Diagnostics.Process]::Start($pInfo)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()

        $combined = "$stdout`n$stderr"
        $totMatch = [regex]::Match($combined, 'Synthesis:.*total=([\d.]+)\s*ms')
        if (-not $totMatch.Success) {
            $totMatch = [regex]::Match($combined, 'Generate:.*total=([\d.]+)\s*ms')
        }
        $durMatch = [regex]::Match($combined, 'audio_s=([\d.]+)')
        $rtfMatch = [regex]::Match($combined, 'total_rtf=([\d.]+)')

        if ($rtfMatch.Success) {
            $wallSec = if ($totMatch.Success) { [double]$totMatch.Groups[1].Value / 1000.0 } else { 0.0 }
            $durSec  = [double]$durMatch.Groups[1].Value
            $rtfVal  = [double]$rtfMatch.Groups[1].Value

            $ratioVal = if ($CsRtf -gt 0 -and $rtfVal -gt 0) { $rtfVal / $CsRtf } else { 0.0 }
            $ratioStr = if ($ratioVal -gt 1.0) { "<span style=`"color:#16a34a`">**{0:F2}x**</span>" -f $ratioVal } elseif ($ratioVal -gt 0) { "**{0:F2}x**" -f $ratioVal } else { "-" }

            Write-Host ("DONE ({0:F2}s wall, {1:F2}s audio, RTF={2:F2}x, Ratio={3})" -f $wallSec, $durSec, $rtfVal, $ratioStr) -ForegroundColor Green

            $results.Add([PSCustomObject]@{
                Category   = "TTS"
                Pipeline   = $Name
                Scenario   = $Scenario
                Backend    = "CPU"
                CsWall     = if ($CsWallSec -gt 0) { "{0:F2}s" -f $CsWallSec } else { "-" }
                CsRtf      = if ($CsRtf -gt 0) { "**{0:F2}x**" -f $CsRtf } else { "-" }
                CppRtf     = ("{0:F2}x" -f $rtfVal)
                CppWall    = ("{0:F2}s" -f $wallSec)
                Ratio      = $ratioStr
                Date       = $today
            })
        } else {
            Write-Host "FAILED (Could not parse s2 metrics)" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# 2. Run TTS Models (audio.cpp & s2.cpp)
if ($hasAudioCpp -and ($Suite -eq "All" -or $Suite -eq "Tts" -or $Suite -eq "CosyVoice")) {
    Write-Host "`n--- Running CosyVoice3 ---" -ForegroundColor Yellow
    Run-AudioCppTask -Name "CosyVoice3 (DiT + HiFT)" `
                     -Family "cosyvoice3" `
                     -Task "clon" `
                     -ModelRelPath "models\CosyVoice3-GGUF\cosyvoice3-q8_0.gguf" `
                     -Scenario "text -> 3.00s audio" `
                     -CsRtf 5.73 `
                     -CsWallSec 17.20 `
                     -ExtraArgs @{
                         "--voice-ref" = $voiceRef
                         "--reference-text" = $refText
                         "--request-option" = "template_name=zero_shot"
                     }
}

if ($hasAudioCpp -and ($Suite -eq "All" -or $Suite -eq "Tts" -or $Suite -eq "Chatterbox")) {
    Write-Host "`n--- Running Chatterbox Turbo ---" -ForegroundColor Yellow
    Run-AudioCppTask -Name "Chatterbox Turbo (Q4_K)" `
                     -Family "chatterbox_turbo" `
                     -Task "tts" `
                     -ModelRelPath "models\Chatterbox-Turbo-GGUF\chatterbox-turbo-q8_0.gguf" `
                     -Scenario "text -> 2.52s audio" `
                     -CsRtf 5.90 `
                     -CsWallSec 14.86
}

if ($hasAudioCpp -and ($Suite -eq "All" -or $Suite -eq "Tts" -or $Suite -eq "QwenTts")) {
    Write-Host "`n--- Running QwenTTS 0.6B ---" -ForegroundColor Yellow
    Run-AudioCppTask -Name "QwenTTS 0.6B (Q8_0 GGUF)" `
                     -Family "qwen3_tts" `
                     -Task "tts" `
                     -ModelRelPath "models\Qwen3-TTS-12Hz-0.6B-Base-GGUF\qwen3-tts-12hz-0.6b-base-q8_0.gguf" `
                     -Scenario "text -> 2.16s audio" `
                     -CsRtf 3.05 `
                     -CsWallSec 6.59 `
                     -ExtraArgs @{
                         "--voice-ref" = $voiceRef
                         "--reference-text" = $refText
                     }
}

if ($hasS2 -and ($Suite -eq "All" -or $Suite -eq "Tts" -or $Suite -eq "FishAudio")) {
    Write-Host "`n--- Running Fish Audio S2 Pro ---" -ForegroundColor Yellow
    Run-S2CppTask -Name "FishSpeech S2 Pro (Q4_K)" `
                  -ModelRelPath "models\s2-pro-q4_k_m.gguf" `
                  -Scenario "text -> 3.44s audio" `
                  -CsRtf 8.28 `
                  -CsWallSec 28.46
}

# 3. Run ASR Models (whisper.cpp)
if ($hasWhisper -and ($Suite -eq "All" -or $Suite -eq "Asr" -or $Suite -eq "Whisper")) {
    Write-Host "`n--- Running Whisper.cpp Suite ---" -ForegroundColor Yellow
    Run-WhisperCppTask -Name "Base (39M)"      -ModelRelPath "models\ggml-base.bin"     -CsRtf 0.070 -CsWallSec 0.84
    Run-WhisperCppTask -Name "Small (244M)"    -ModelRelPath "models\ggml-small.bin"    -CsRtf 0.202 -CsWallSec 2.42
    Run-WhisperCppTask -Name "Medium (769M)"   -ModelRelPath "models\ggml-medium.bin"   -CsRtf 0.560 -CsWallSec 6.71
    Run-WhisperCppTask -Name "Large-v3 (1.5B)" -ModelRelPath "models\ggml-large-v3.bin" -CsRtf 0.943 -CsWallSec 11.32
}

# 4. Generate Markdown Summary
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " C++ REFERENCE BENCHMARK SUMMARY" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

$mdReport = [System.Text.StringBuilder]::new()
$null = $mdReport.AppendLine("## C++ Reference Performance Summary (Measured: $today, Threads: $Threads)")
$null = $mdReport.AppendLine("")
$null = $mdReport.AppendLine("| Pipeline / Model | Category | Scenario | Backend | C# Wall | C# RTF | C++ Wall | C++ RTF | Ratio | Performance Check |")
$null = $mdReport.AppendLine("|---|---|---|---|---:|---:|---:|---:|---:|---|")

foreach ($r in $results) {
    $null = $mdReport.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |" -f `
        $r.Pipeline, $r.Category, $r.Scenario, $r.Backend, $r.CsWall, $r.CsRtf, $r.CppWall, $r.CppRtf, $r.Ratio, $r.Date))
}

Write-Host $mdReport.ToString()

if ($OutFile) {
    $outPath = if ([System.IO.Path]::IsPathRooted($OutFile)) { $OutFile } else { Join-Path $repoRoot $OutFile }
    $outDir  = [System.IO.Path]::GetDirectoryName($outPath)
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    Set-Content -Path $outPath -Value $mdReport.ToString()
    Write-Host "[Report] Saved benchmark results to '$outPath'" -ForegroundColor Green
}
