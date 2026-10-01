<#
.SYNOPSIS
  Resumable real-weight test sweep: one test class per process, state written BEFORE each class starts.

.DESCRIPTION
  Runs every class of the heavy test projects one at a time with STINGRAY_RUN_HEAVY_TESTS=1.
  If the process is killed (memory reaper, reboot, Ctrl+C), rerun the same command: finished classes are
  skipped, and the class that was in flight is recorded as interrupted and retried. A class interrupted
  -MaxInterruptions times is recorded as a suspected memory hog and skipped, so the sweep cannot loop on it.

  State (in -StateDir):
    NEXT.txt      the class about to run (written before it starts); "IDLE" when nothing is running
    state.jsonl   append-only events: start / done / interrupted / skipped-suspect
    logs\<suite>\<class>.log|.xml   console output and xUnit XML per class

  Build the test projects in Release first (the exe does not rebuild itself):
    dotnet build tests/OpenTail.Stingray.Tests.<Suite> -c Release

  Run it from a normal terminal, outside Claude Code, so a low-memory reaper cannot kill it.

.EXAMPLE
  pwsh scripts/sweep-tests.ps1                       # run / resume the whole sweep
  pwsh scripts/sweep-tests.ps1 -Suites Audio         # one suite
  pwsh scripts/sweep-tests.ps1 -ClassFilter Fish     # only classes whose name matches the regex
  pwsh scripts/sweep-tests.ps1 -Summary              # report only: failures, interrupted, quick passes
#>
param(
    [string[]]$Suites = @("Diffusion", "Audio", "Vision", "ForwardPass"),
    [string]$StateDir = (Join-Path $env:TEMP "stingray-sweep"),
    [string]$ClassFilter = "",
    [int]$MinFreeGB = 12,
    [int]$ClassTimeoutMin = 60,
    [int]$MaxInterruptions = 2,
    [switch]$Summary
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$stateFile = Join-Path $StateDir "state.jsonl"
$nextFile = Join-Path $StateDir "NEXT.txt"
New-Item -ItemType Directory -Force $StateDir | Out-Null
$commitSha = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not determine repository commit SHA for sweep identity." }

function Get-FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }

function Add-Event($e) {
    $e.ts = (Get-Date).ToString("s")
    ($e | ConvertTo-Json -Compress) | Add-Content -Path $stateFile -Encoding utf8
}

function Test-DoneIdentity($done, $suite, $class, $exeSha256, $exeLastWriteUtc,
                           $assemblyPath, $assemblySha256, $assemblyLastWriteUtc, $commitSha) {
    $storedExeLastWriteUtc = $null
    $storedAssemblyLastWriteUtc = $null
    try {
        $storedExeLastWriteUtc = ([datetime]$done.exeLastWriteUtc).ToUniversalTime().ToString("o")
        $storedAssemblyLastWriteUtc = ([datetime]$done.assemblyLastWriteUtc).ToUniversalTime().ToString("o")
    }
    catch { return $false }
    if (-not $done -or $done.commitSha -ne $commitSha -or $done.suite -ne $suite -or
        $done.class -ne $class -or $done.exeSha256 -ne $exeSha256 -or
        $storedExeLastWriteUtc -ne $exeLastWriteUtc -or $done.assemblyPath -ne $assemblyPath -or
        $done.assemblySha256 -ne $assemblySha256 -or $storedAssemblyLastWriteUtc -ne $assemblyLastWriteUtc -or
        -not ($done.PSObject.Properties.Name -contains "modelAssets")) {
        return $false
    }

    foreach ($asset in @($done.modelAssets)) {
        if (-not (Test-Path -LiteralPath $asset.path -PathType Leaf)) { return $false }
        $file = Get-Item -LiteralPath $asset.path
        if ($file.Length -ne [long]$asset.bytes) { return $false }
        if ((Get-FileHash -LiteralPath $asset.path -Algorithm SHA256).Hash -ne $asset.sha256) { return $false }
    }
    return $true
}

function Read-State {
    $keys = [ordered]@{}
    if (Test-Path $stateFile) {
        foreach ($line in Get-Content $stateFile) {
            if (-not $line.Trim()) { continue }
            $e = $line | ConvertFrom-Json
            $k = "$($e.suite)|$($e.class)"
            if (-not $keys.Contains($k)) {
                $keys[$k] = [pscustomobject]@{ suite = $e.suite; class = $e.class; done = $null; inflight = $false; interruptions = 0; skipped = $false }
            }
            $s = $keys[$k]
            switch ($e.event) {
                "start"           { if ($s.inflight) { $s.interruptions++ }; $s.inflight = $true }
                "done"            { $s.inflight = $false; $s.done = $e }
                "skipped-suspect" { $s.skipped = $true; $s.inflight = $false; $s.interruptions = [math]::Max($s.interruptions, [int]$e.interruptions) }
            }
        }
    }
    foreach ($s in $keys.Values) { if ($s.inflight) { $s.interruptions++ } }
    $keys
}

function Show-Summary {
    $keys = Read-State
    $done = @($keys.Values | Where-Object { $_.done })
    "Classes finished: $($done.Count)   in flight/interrupted: $(@($keys.Values | Where-Object { $_.inflight -and -not $_.done }).Count)   skipped as suspect: $(@($keys.Values | Where-Object { $_.skipped }).Count)"
    if (Test-Path $nextFile) { "NEXT.txt: $((Get-Content $nextFile -Raw).Trim())" }
    ""
    "== Failures (exit code != 0 or failed > 0)"
    $done | Where-Object { $_.done.exit -ne 0 -or $_.done.failed -gt 0 } |
        ForEach-Object { "  {0} {1}  exit={2} failed={3} timedOut={4}" -f $_.suite, $_.class, $_.done.exit, $_.done.failed, $_.done.timedOut }
    "== Interrupted / suspected memory hogs (killed mid-class)"
    $keys.Values | Where-Object { $_.interruptions -gt 0 -and -not $_.done } |
        ForEach-Object { "  {0} {1}  interruptions={2} skippedAsSuspect={3}" -f $_.suite, $_.class, $_.interruptions, $_.skipped }
    "== Zero tests matched (class name did not resolve)"
    $done | Where-Object { $_.done.total -eq 0 } | ForEach-Object { "  {0} {1}" -f $_.suite, $_.class }
    "== Quick passes: passed > 0 in under 0.4 s (possible silent no-op, CLAUDE.md rule 12)"
    $done | Where-Object { $_.done.passed -gt 0 -and $_.done.seconds -lt 0.4 } |
        ForEach-Object { "  {0} {1}  passed={2} {3}s" -f $_.suite, $_.class, $_.done.passed, $_.done.seconds }
    "== Top 10 by peak memory"
    $done | Sort-Object { $_.done.peakGB } -Descending | Select-Object -First 10 |
        ForEach-Object { "  {0,5} GB  {1,7}s  {2} {3}" -f $_.done.peakGB, $_.done.seconds, $_.suite, $_.class }
}

if ($Summary) { Show-Summary; return }

$env:STINGRAY_RUN_HEAVY_TESTS = "1"

foreach ($suite in $Suites) {
    $exe = Join-Path $repo "tests\OpenTail.Stingray.Tests.$suite\bin\Release\net10.0\OpenTail.Stingray.Tests.$suite.exe"
    if (-not (Test-Path $exe)) { throw "Missing $exe. Build it first: dotnet build tests/OpenTail.Stingray.Tests.$suite -c Release" }
    $exeSha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    $exeLastWriteUtc = (Get-Item -LiteralPath $exe).LastWriteTimeUtc.ToString("o")
    $assemblyPath = [IO.Path]::ChangeExtension($exe, ".dll")
    if (-not (Test-Path -LiteralPath $assemblyPath)) { throw "Missing test assembly $assemblyPath." }
    $assemblySha256 = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
    $assemblyLastWriteUtc = (Get-Item -LiteralPath $assemblyPath).LastWriteTimeUtc.ToString("o")

    $classes = @(& $exe -list classes 2>&1 | Where-Object { $_ -match '^OpenTail\.' } | Sort-Object)
    if ($ClassFilter) { $classes = @($classes | Where-Object { $_ -match $ClassFilter }) }
    $logDir = Join-Path $StateDir "logs\$suite"
    New-Item -ItemType Directory -Force $logDir | Out-Null

    $i = 0
    foreach ($class in $classes) {
        $i++
        $state = Read-State
        $k = "$suite|$class"
        $s = if ($state.Contains($k)) { $state[$k] } else { $null }
        if ($s -and $s.done) {
            if (Test-DoneIdentity $s.done $suite $class $exeSha256 $exeLastWriteUtc `
                    $assemblyPath $assemblySha256 $assemblyLastWriteUtc $commitSha) { continue }
            Write-Host "RERUN (cached result has no matching commit/binary/model identity): $suite $class"
        }
        if ($s -and $s.skipped) { continue }
        if ($s -and $s.interruptions -ge $MaxInterruptions) {
            Add-Event @{ event = "skipped-suspect"; suite = $suite; class = $class; interruptions = $s.interruptions }
            Write-Host "SKIP (killed $($s.interruptions)x, suspected memory hog): $suite $class"
            continue
        }
        if ($s -and $s.inflight) {
            Add-Event @{ event = "interrupted"; suite = $suite; class = $class; note = "in flight when a previous run died" }
        }

        $waited = 0
        while ((Get-FreeGB) -lt $MinFreeGB -and $waited -lt 30) {
            Write-Host "Waiting for free RAM >= $MinFreeGB GB (now $(Get-FreeGB) GB)..."
            Start-Sleep -Seconds 60; $waited++
        }

        # State is written BEFORE the class is touched.
        $free = Get-FreeGB
        "$suite $class ($i of $($classes.Count))  started $((Get-Date).ToString('s'))  freeGB=$free" | Set-Content $nextFile -Encoding utf8
        Add-Event @{ event = "start"; suite = $suite; class = $class; index = $i; of = $classes.Count; freeGB = $free }
        Write-Host ("[{0}/{1}] {2} {3}  (free {4} GB)" -f $i, $classes.Count, $suite, $class, $free)

        $safe = $class -replace '[^A-Za-z0-9._-]', '_'
        $log = Join-Path $logDir "$safe.log"
        $xml = Join-Path $logDir "$safe.xml"
        if (Test-Path $xml) { Remove-Item $xml -Force }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $exe -ArgumentList @("-class", $class, "-xml", $xml) -NoNewWindow -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        $null = $p.Handle
        $peak = 0L; $timedOut = $false
        while (-not $p.WaitForExit(2000)) {
            try { $p.Refresh(); if ($p.WorkingSet64 -gt $peak) { $peak = $p.WorkingSet64 } } catch { }
            if ($sw.Elapsed.TotalMinutes -gt $ClassTimeoutMin) {
                $timedOut = $true
                & taskkill /T /F /PID $p.Id | Out-Null
                break
            }
        }
        $p.WaitForExit()
        $sw.Stop()

        $total = 0; $passed = 0; $failed = 0; $skipped = 0
        if (Test-Path $xml) {
            try {
                $a = ([xml](Get-Content $xml -Raw)).assemblies.assembly
                $total = [int]$a.total; $passed = [int]$a.passed; $failed = [int]$a.failed; $skipped = [int]$a.skipped
            } catch { }
        }
        $exit = if ($timedOut) { -1 } else { $p.ExitCode }
        $modelAssets = @()
        if (Test-Path -LiteralPath $log) {
            $logText = Get-Content -LiteralPath $log -Raw
            $assetPaths = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            $assetPattern = '(?i)(?:path|modelPath|mmproj|weights|checkpoint)\s*[:=]\s*["'']?([A-Z]:\\[^\s"'';,]+\.(?:gguf|onnx|safetensors|pt|bin|wav|png|jpe?g|f32|npy))'
            foreach ($match in [regex]::Matches($logText, $assetPattern)) {
                $candidate = $match.Groups[1].Value.TrimEnd('.', ')', ']')
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { $null = $assetPaths.Add((Get-Item -LiteralPath $candidate).FullName) }
            }
            foreach ($assetPath in $assetPaths) {
                $assetFile = Get-Item -LiteralPath $assetPath
                $modelAssets += [pscustomobject]@{
                    path = $assetFile.FullName
                    bytes = [long]$assetFile.Length
                    sha256 = (Get-FileHash -LiteralPath $assetFile.FullName -Algorithm SHA256).Hash
                }
            }
        }
        Add-Event @{ event = "done"; suite = $suite; class = $class; exit = $exit; timedOut = $timedOut
                     seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2); total = $total; passed = $passed
                     failed = $failed; skipped = $skipped; peakGB = [math]::Round($peak / 1GB, 2); freeGB = (Get-FreeGB)
                     commitSha = $commitSha; exeSha256 = $exeSha256; exeLastWriteUtc = $exeLastWriteUtc
                     assemblyPath = $assemblyPath; assemblySha256 = $assemblySha256
                     assemblyLastWriteUtc = $assemblyLastWriteUtc
                     modelAssets = @($modelAssets) }
        Write-Host ("    exit={0} total={1} passed={2} failed={3} skipped={4} {5}s peak {6} GB" -f $exit, $total, $passed, $failed, $skipped, [math]::Round($sw.Elapsed.TotalSeconds, 1), [math]::Round($peak / 1GB, 2))
    }
}

"IDLE (sweep finished $((Get-Date).ToString('s')))" | Set-Content $nextFile -Encoding utf8
Show-Summary
