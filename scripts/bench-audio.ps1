<#
.SYNOPSIS
    Runs performance benchmarks for OpenTail.Stingray.Audio C# engine and formats the results.

.DESCRIPTION
    Executes the curated audio performance benchmark suite in OpenTail.Stingray.Tests.Audio,
    collects timing, wall-clock latency, Real-Time Factor (RTF), Time-To-First-Audio (TTFA),
    and formats the data into league-ready markdown tables for PerformanceLeague.md.

.PARAMETER Suite
    Benchmark suite to run:
    - All (default): Runs TTS baselines, streaming latency, and ASR benchmarks.
    - Tts: Runs all TTS full-generation baseline benchmarks.
    - Streaming / TtsStream: Runs TTS streaming benchmarks (TTFA and chunk throughput).
    - Asr: Runs ASR benchmarks (Whisper, FunASR, QwenASR, Silero VAD).
    - Components: Runs isolated component benchmarks (HiFT vocoder, DiT, CFM decoder, etc.).
    - Quick / Smoke: Runs fast TTS baselines (Kokoro, Piper, MeloTTS) and Whisper Base.

.PARAMETER Model
    Filter to a specific model or engine (e.g., 'kokoro', 'qwen', 'fishspeech', 'f5tts',
    'chatterbox', 'whisper', 'xtts', 'mms', 'piper', 'melo', 'parler', 'cosyvoice').

.PARAMETER NoBuild
    Skip the dotnet build step (runs existing Release test binary).

.PARAMETER Threads
    Optional number of CPU threads (sets STINGRAY_CPU_THREADS).

.PARAMETER ShowOutput
    Show live standard and error output from the xUnit test runner.

.PARAMETER OutFile
    Path to save the markdown benchmark report (e.g. docs/audio-benchmark-results.md).

.EXAMPLE
    .\scripts\bench-audio.ps1
    .\scripts\bench-audio.ps1 -Suite Tts
    .\scripts\bench-audio.ps1 -Model kokoro
    .\scripts\bench-audio.ps1 -Suite Asr -ShowOutput
#>

[CmdletBinding()]
param(
    [ValidateSet("All", "Tts", "Streaming", "TtsStream", "Asr", "Components", "Quick", "Smoke")]
    [string]$Suite = "All",

    [string]$Model = "",

    [switch]$NoBuild,

    [string]$Config = "Release",

    [int]$Threads = 0,

    [switch]$ShowOutput,

    [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$testProj = Join-Path $repoRoot "tests\OpenTail.Stingray.Tests.Audio\OpenTail.Stingray.Tests.Audio.csproj"
$testExe  = Join-Path $repoRoot "tests\OpenTail.Stingray.Tests.Audio\bin\$Config\net10.0\OpenTail.Stingray.Tests.Audio.exe"

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " OPENTAIL.STINGRAY AUDIO BENCHMARK RUNNER" -ForegroundColor Cyan
Write-Host " Suite: $Suite | Filter: $(if ($Model) { $Model } else { '<None>' }) | Config: $Config" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Build if needed
if (-not $NoBuild) {
    Write-Host "[Build] Building OpenTail.Stingray.Tests.Audio ($Config)..." -ForegroundColor Yellow
    $buildOutput = dotnet build $testProj -c $Config 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($buildOutput -join "`n") -ForegroundColor Red
        Write-Error "Build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[Build] Build succeeded." -ForegroundColor Green
    Write-Host ""
}

if (-not (Test-Path $testExe)) {
    Write-Error "Test executable not found at '$testExe'. Run without -NoBuild first."
    exit 1
}

# 2. Configure Environment
$env:STINGRAY_RUN_HEAVY_TESTS = "1"
if ($Threads -gt 0) {
    $env:STINGRAY_CPU_THREADS = $Threads.ToString()
    Write-Host "[Config] STINGRAY_CPU_THREADS set to $Threads" -ForegroundColor DarkGray
} else {
    Remove-Item Env:\STINGRAY_CPU_THREADS -ErrorAction SilentlyContinue
}

# 3. Define Benchmark Target Catalog
$allTargets = @(
    # --- TTS Full Pipeline Baselines ---
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "Kokoro-82M"
        Tag      = "kokoro"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_Kokoro"
        Desc     = "Kokoro 82M Q8_0 GGUF (af_heart voice)"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "Piper-lessac"
        Tag      = "piper"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_Piper"
        Desc     = "Piper lessac-medium ONNX"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "MeloTTS-zh_en"
        Tag      = "melo"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_MeloTts"
        Desc     = "MeloTTS zh_en ONNX (EN-US voice)"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "MMS-TTS-eng"
        Tag      = "mms"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_MmsTts"
        Desc     = "MMS-TTS English VITS safetensors"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "QwenTTS-0.6B"
        Tag      = "qwen"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_QwenTts"
        Desc     = "QwenTTS Talker 0.6B Q8 + Code Predictor"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "Chatterbox-Turbo"
        Tag      = "chatterbox"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_Chatterbox"
        Desc     = "Chatterbox Turbo T3 Q4_K + S3Gen vocoder"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "FishSpeech-S2Pro"
        Tag      = "fishspeech"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_FishSpeech"
        Desc     = "Fish Speech S2 Pro Fast-AR + Codec"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "F5-TTS-Base"
        Tag      = "f5tts"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_F5Tts"
        Desc     = "F5-TTS DiT flow-matching backbone"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "CosyVoice3"
        Tag      = "cosyvoice"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_CosyVoice3"
        Desc     = "CosyVoice3 LLM 0.5B + DiT + HiFT vocoder"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "XTTS-v2"
        Tag      = "xtts"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_Xtts"
        Desc     = "XTTS-v2 GPT + DVAE + HiFi-GAN"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "Parler-TTS-Mini"
        Tag      = "parler"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Baseline_Parler"
        Desc     = "Parler-TTS Mini v1 autoregressive + DAC"
    },
    @{
        Category = "TTS Baseline"
        Group    = "Tts"
        Model    = "F5-TTS-Paragraph"
        Tag      = "f5tts"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Paragraph_F5Tts"
        Desc     = "F5-TTS paragraph generation (~14s audio)"
    },

    # --- TTS Streaming Latency (TTFA) ---
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "Kokoro-82M-Stream"
        Tag      = "kokoro"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_Kokoro"
        Desc     = "Kokoro 82M Streaming TTFA"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "Piper-Stream"
        Tag      = "piper"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_Piper"
        Desc     = "Piper Streaming TTFA (16-frame chunks)"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "MeloTTS-Stream"
        Tag      = "melo"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_MeloTts"
        Desc     = "MeloTTS Streaming TTFA"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "MMS-TTS-Stream"
        Tag      = "mms"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_MmsTts"
        Desc     = "MMS-TTS Streaming TTFA (16-frame chunks)"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "QwenTTS-Stream"
        Tag      = "qwen"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_QwenTts"
        Desc     = "QwenTTS Streaming TTFA (1-frame chunks)"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "Chatterbox-Stream"
        Tag      = "chatterbox"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_Chatterbox"
        Desc     = "Chatterbox Streaming TTFA"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "FishSpeech-Stream"
        Tag      = "fishspeech"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_FishSpeech"
        Desc     = "Fish Speech Streaming TTFA (1-frame chunks)"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "F5-TTS-Stream"
        Tag      = "f5tts"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_F5Tts"
        Desc     = "F5-TTS Streaming TTFA"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "XTTS-Stream"
        Tag      = "xtts"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_Xtts"
        Desc     = "XTTS-v2 Streaming TTFA (6-token chunks)"
    },
    @{
        Category = "TTS Streaming"
        Group    = "Streaming"
        Model    = "Parler-Stream"
        Tag      = "parler"
        Class    = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest"
        Method   = "OpenTail.Stingray.Tests.Audio.TtsPerformanceBaselineDebugTest.Streaming_Parler"
        Desc     = "Parler-TTS Streaming TTFA (16-frame chunks)"
    },

    # --- ASR & Audio Understanding ---
    @{
        Category = "ASR / STT"
        Group    = "Asr"
        Model    = "Whisper-Pipelines"
        Tag      = "whisper"
        Class    = "OpenTail.Stingray.Tests.Audio.WhisperFullPipelinePerfBenchTests"
        Method   = "OpenTail.Stingray.Tests.Audio.WhisperFullPipelinePerfBenchTests.Bench_Transcribe_FixedAudio_AcrossModelSizes"
        Desc     = "Whisper 12s transcription (Base, Small, Medium, Large-v3)"
    },
    @{
        Category = "ASR / STT"
        Group    = "Asr"
        Model    = "Whisper-PhaseTiming"
        Tag      = "whisper"
        Class    = "OpenTail.Stingray.Tests.Audio.WhisperPhaseTimingBenchTests"
        Method   = "OpenTail.Stingray.Tests.Audio.WhisperPhaseTimingBenchTests.Bench_PhaseBreakdown_BaseModel"
        Desc     = "Whisper Base phase breakdown (Encoder vs Autoregressive Decoder)"
    },
    @{
        Category = "ASR / STT"
        Group    = "Asr"
        Model    = "FunASR-Nano"
        Tag      = "funasr"
        Class    = "OpenTail.Stingray.Tests.Audio.FunAsrPerfBenchTests"
        Method   = "OpenTail.Stingray.Tests.Audio.FunAsrPerfBenchTests.Bench_Transcribe_LongerAudio"
        Desc     = "FunASR SenseVoice/Nano Conformer transcription"
    },
    @{
        Category = "VAD"
        Group    = "Asr"
        Model    = "Silero-VAD"
        Tag      = "silero"
        Class    = "OpenTail.Stingray.Tests.Audio.SileroVadPerfBenchTests"
        Method   = "OpenTail.Stingray.Tests.Audio.SileroVadPerfBenchTests.Bench_DetectSegments_LongerAudio"
        Desc     = "Silero VAD 12s audio speech segment detection"
    },
    @{
        Category = "ASR / STT"
        Group    = "Asr"
        Model    = "QwenASR-Encoder"
        Tag      = "qwen"
        Class    = "OpenTail.Stingray.Tests.Audio.QwenAsrBenchmarkTests"
        Method   = "OpenTail.Stingray.Tests.Audio.QwenAsrBenchmarkTests.AudioEncoder_Benchmark_RealisticClipLength"
        Desc     = "QwenASR Audio Encoder forward pass (10s audio mel)"
    },

    # --- Component & Kernel Benchmarks ---
    @{
        Category = "Components"
        Group    = "Components"
        Model    = "CosyVoice3-Components"
        Tag      = "cosyvoice"
        Class    = "OpenTail.Stingray.Tests.Audio.CosyVoiceBenchmarkTests"
        Method   = ""
        Desc     = "CosyVoice3 HiFT vocoder, Flow encoder, DiT, CFM decoder"
    },
    @{
        Category = "Components"
        Group    = "Components"
        Model    = "Chatterbox-S3Gen"
        Tag      = "chatterbox"
        Class    = "OpenTail.Stingray.Tests.Audio.ChatterboxVocoderBenchmarkTests"
        Method   = "OpenTail.Stingray.Tests.Audio.ChatterboxVocoderBenchmarkTests.ChatterboxVocoder_Benchmark_SyntheticMelAtRealisticScale"
        Desc     = "Chatterbox S3Gen vocoder mel->waveform synthesis"
    },
    @{
        Category = "Components"
        Group    = "Components"
        Model    = "FishSpeech-FastAR"
        Tag      = "fishspeech"
        Class    = "OpenTail.Stingray.Tests.Audio.FishSpeechArStageSplitPerfBenchTests"
        Method   = ""
        Desc     = "Fish Speech Fast-AR isolated per-frame timing"
    }
)

# 4. Filter Targets
$selectedTargets = $allTargets | Where-Object {
    $target = $_
    $matchSuite = switch ($Suite) {
        "All"         { $true }
        "Tts"         { $target.Group -eq "Tts" }
        "Streaming"   { $target.Group -eq "Streaming" }
        "TtsStream"   { $target.Group -eq "Streaming" }
        "Asr"         { $target.Group -eq "Asr" }
        "Components"  { $target.Group -eq "Components" }
        "Quick"       { ($target.Model -in @("Kokoro-82M", "Piper-lessac", "MeloTTS-zh_en", "Whisper-Pipelines")) }
        "Smoke"       { ($target.Model -in @("Kokoro-82M", "Piper-lessac")) }
        Default       { $true }
    }

    $matchModel = if ([string]::IsNullOrWhiteSpace($Model)) {
        $true
    } else {
        $target.Tag -like "*$Model*" -or $target.Model -like "*$Model*"
    }

    $matchSuite -and $matchModel
}

if ($selectedTargets.Count -eq 0) {
    Write-Warning "No benchmarks match the given filter criteria (Suite: '$Suite', Model: '$Model')."
    exit 0
}

Write-Host "Discovered $($selectedTargets.Count) benchmark targets to execute." -ForegroundColor Cyan
Write-Host ""

# 5. Execution & Metric Extraction Loop
$results = [System.Collections.Generic.List[PSCustomObject]]::new()
$today = (Get-Date).ToString("yyyy-MM-dd")

foreach ($target in $selectedTargets) {
    Write-Host "-> Running [$($target.Category)] $($target.Model)... " -NoNewline -ForegroundColor White

    $argList = @("-class", $target.Class)
    if ($target.Method) {
        $argList += @("-method", $target.Method)
    }
    $argList += "-showLiveOutput"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $testExe
    $psi.WorkingDirectory = $repoRoot
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()

    # Audio model loads and multi-run benchmarks can take up to 2-3 mins per pipeline
    $timeoutSec = 240
    $timedOut = -not $proc.WaitForExit($timeoutSec * 1000)
    if ($timedOut) {
        try { $proc.Kill($true) } catch {}
    }
    $sw.Stop()

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $combinedOutput = "$stdout`n$stderr"

    if ($ShowOutput) {
        Write-Host ""
        Write-Host $combinedOutput -ForegroundColor DarkGray
    }

    # Analyze Outcome
    $isSkipped = ($combinedOutput -match 'Skipped:\s*[1-9]\d*') -or ($combinedOutput -match 'Assert\.Skip') -or ($combinedOutput -match 'model.*not found') -or ($combinedOutput -match 'checkpoint.*not found')
    $isPassed  = ($proc.ExitCode -eq 0) -and (-not $isSkipped) -and (-not $timedOut)

    if ($timedOut) {
        Write-Host "TIMEOUT (${timeoutSec}s)" -ForegroundColor Red
        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Full Generation"
            Backend       = "CPU"
            WallClock     = "TIMEOUT"
            RTF           = "-"
            TTFA          = "-"
            AudioSec      = "-"
            Status        = "TIMEOUT"
            DateVerified  = $today
            Details       = $target.Desc
        })
        continue
    }

    if ($isSkipped) {
        Write-Host "SKIPPED (checkpoint not found)" -ForegroundColor Yellow
        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Full Generation"
            Backend       = "CPU"
            WallClock     = "-"
            RTF           = "-"
            TTFA          = "-"
            AudioSec      = "-"
            Status        = "SKIPPED"
            DateVerified  = $today
            Details       = "$($target.Desc) (model file missing)"
        })
        continue
    }

    if ($proc.ExitCode -ne 0) {
        Write-Host "FAILED (exit $($proc.ExitCode))" -ForegroundColor Red
        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Full Generation"
            Backend       = "CPU"
            WallClock     = "FAIL"
            RTF           = "-"
            TTFA          = "-"
            AudioSec      = "-"
            Status        = "FAILED"
            DateVerified  = $today
            Details       = $target.Desc
        })
        continue
    }

    # Extract metrics via regex
    # Pattern 1: Baseline TTS: "[Model] prompt=... audio=2.93s ... mean=2.556s RTF=0.874"
    $parsedAny = $false

    $baselineMatch = [regex]::Match($combinedOutput, '(?s)\[(?<tag>[^\]]+)\]\s+prompt="(?<prompt>[^"]+)".*?audio=(?<audio>[\d.]+)s.*?mean=(?<mean>[\d.]+)s\s+RTF=(?<rtf>[\d.]+)')
    if ($baselineMatch.Success) {
        $parsedAny = $true
        $rtfVal = [double]$baselineMatch.Groups['rtf'].Value
        $meanSec = [double]$baselineMatch.Groups['mean'].Value
        $audioSec = [double]$baselineMatch.Groups['audio'].Value
        Write-Host ("DONE (RTF={0:F2}x, mean={1:F2}s for {2:F2}s audio)" -f $rtfVal, $meanSec, $audioSec) -ForegroundColor Green

        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = ("text -> {0:F2}s audio" -f $audioSec)
            Backend       = "CPU"
            WallClock     = ("{0:F2}s" -f $meanSec)
            RTF           = ("**{0:F2}x**" -f $rtfVal)
            TTFA          = "-"
            AudioSec      = ("{0:F2}s" -f $audioSec)
            Status        = "OK"
            DateVerified  = $today
            Details       = $target.Desc
        })
    }

    # Pattern 2: Streaming TTS: "[Model-Stream...] TTFA=0.215s TotalTime=2.560s"
    $streamMatch = [regex]::Match($combinedOutput, '(?s)\[(?<tag>[^\]]+)\].*?Time-To-First-Audio\s+\(TTFA\)=(?<ttfa>[\d.]+)s\s+TotalTime=(?<total>[\d.]+)s')
    if ($streamMatch.Success -and -not $baselineMatch.Success) {
        $parsedAny = $true
        $ttfaSec = [double]$streamMatch.Groups['ttfa'].Value
        $totSec  = [double]$streamMatch.Groups['total'].Value
        Write-Host ("DONE (TTFA={0:F3}s, Total={1:F2}s)" -f $ttfaSec, $totSec) -ForegroundColor Green

        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Streaming TTFA"
            Backend       = "CPU"
            WallClock     = ("{0:F2}s" -f $totSec)
            RTF           = "-"
            TTFA          = ("**{0:F3}s**" -f $ttfaSec)
            AudioSec      = "-"
            Status        = "OK"
            DateVerified  = $today
            Details       = $target.Desc
        })
    }

    # Pattern 3: Paragraph TTS: "[F5-TTS-Paragraph...] time=10.200s RTF=0.703"
    $paraMatch = [regex]::Match($combinedOutput, '(?s)\[(?<tag>[^\]]+)\].*?time=(?<time>[\d.]+)s\s+RTF=(?<rtf>[\d.]+)')
    if ($paraMatch.Success -and -not $baselineMatch.Success -and -not $streamMatch.Success) {
        $parsedAny = $true
        $rtfVal = [double]$paraMatch.Groups['rtf'].Value
        $timeSec = [double]$paraMatch.Groups['time'].Value
        Write-Host ("DONE (Paragraph RTF={0:F2}x, time={1:F2}s)" -f $rtfVal, $timeSec) -ForegroundColor Green

        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Paragraph synthesis"
            Backend       = "CPU"
            WallClock     = ("{0:F2}s" -f $timeSec)
            RTF           = ("**{0:F2}x**" -f $rtfVal)
            TTFA          = "-"
            AudioSec      = "-"
            Status        = "OK"
            DateVerified  = $today
            Details       = $target.Desc
        })
    }

    # Pattern 4: Whisper multi-size report lines: "Base (whisper-base.gguf): audio=12s ... mean_ms=1202.0 ... RTF=0.100"
    $whisperMatches = [regex]::Matches($combinedOutput, '(?s)(?<size>\w[\w\-]+)\s+\((?<file>[^\)]+)\):\s+audio=(?<aud>[\d.]+)s.*?mean_ms=(?<mean>[\d.]+)\s+.*?RTF=(?<rtf>[\d.]+)')
    if ($whisperMatches.Count -gt 0) {
        $parsedAny = $true
        Write-Host ("DONE ($($whisperMatches.Count) Whisper models measured)") -ForegroundColor Green
        foreach ($wm in $whisperMatches) {
            $wSize = $wm.Groups['size'].Value
            $wAud  = [double]$wm.Groups['aud'].Value
            $wMean = [double]$wm.Groups['mean'].Value / 1000.0
            $wRtf  = [double]$wm.Groups['rtf'].Value

            $results.Add([PSCustomObject]@{
                Category      = "ASR / STT"
                Model         = "Whisper-$wSize"
                Scenario      = ("Transcribe {0:F0}s audio" -f $wAud)
                Backend       = "CPU"
                WallClock     = ("{0:F2}s" -f $wMean)
                RTF           = ("**{0:F3}x**" -f $wRtf)
                TTFA          = "-"
                AudioSec      = ("{0:F0}s" -f $wAud)
                Status        = "OK"
                DateVerified  = $today
                Details       = "Whisper $wSize GGUF greedily transcribed"
            })
        }
    }

    # Pattern 5: Generic Benchmark timings (e.g. "[CosyVoiceBenchmark] FlowEncoder: 45ms" or "[VocoderBenchmark] ... in 85ms")
    $compMatches = [regex]::Matches($combinedOutput, '\[(?<label>[\w]+Benchmark)\]\s+(?<name>[^:]+):\s+(?<ms>[\d.]+)ms')
    if ($compMatches.Count -gt 0) {
        $parsedAny = $true
        Write-Host ("DONE ($($compMatches.Count) component stages measured)") -ForegroundColor Green
        foreach ($cm in $compMatches) {
            $cName = $cm.Groups['name'].Value.Trim()
            $cMs   = $cm.Groups['ms'].Value.Trim()

            $results.Add([PSCustomObject]@{
                Category      = "Components"
                Model         = $target.Model
                Scenario      = $cName
                Backend       = "CPU"
                WallClock     = "${cMs}ms"
                RTF           = "-"
                TTFA          = "-"
                AudioSec      = "-"
                Status        = "OK"
                DateVerified  = $today
                Details       = $target.Desc
            })
        }
    }

    if (-not $parsedAny) {
        Write-Host ("DONE (elapsed {0:F1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
        $results.Add([PSCustomObject]@{
            Category      = $target.Category
            Model         = $target.Model
            Scenario      = "Execution"
            Backend       = "CPU"
            WallClock     = ("{0:F2}s" -f $sw.Elapsed.TotalSeconds)
            RTF           = "-"
            TTFA          = "-"
            AudioSec      = "-"
            Status        = "OK"
            DateVerified  = $today
            Details       = $target.Desc
        })
    }
}

Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " BENCHMARK RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""

# 6. Generate Markdown League Table
$mdReport = [System.Text.StringBuilder]::new()
$null = $mdReport.AppendLine("## Audio / Speech Performance League (Measured: $today)")
$null = $mdReport.AppendLine("")
$null = $mdReport.AppendLine("> **Note:** RTF (Real-Time Factor) = wall-clock time / audio duration. RTF < 1.0x indicates faster than real-time.")
$null = $mdReport.AppendLine("")
$null = $mdReport.AppendLine("| Model / Pipeline | Category | Scenario | Backend | Wall-Clock | RTF (C#) | TTFA | Performance Check | Notes |")
$null = $mdReport.AppendLine("|---|---|---|---|---:|---:|---:|---|---|")

foreach ($r in $results) {
    $null = $mdReport.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |" -f `
        $r.Model, $r.Category, $r.Scenario, $r.Backend, $r.WallClock, $r.RTF, $r.TTFA, $r.DateVerified, $r.Details))
}

Write-Host $mdReport.ToString()

# 7. Write to output file if requested
if ($OutFile) {
    $outPath = if ([System.IO.Path]::IsPathRooted($OutFile)) { $OutFile } else { Join-Path $repoRoot $OutFile }
    $outDir  = [System.IO.Path]::GetDirectoryName($outPath)
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    Set-Content -Path $outPath -Value $mdReport.ToString()
    Write-Host "[Report] Saved benchmark results to '$outPath'" -ForegroundColor Green
}

Write-Host "Done! Collected $($results.Count) benchmark results." -ForegroundColor Cyan
