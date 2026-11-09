<#
.SYNOPSIS
Qwen 3.8 Flash Next (qwen4exp) developer workflow: scout, CLI smoke, the Qwen real-checkpoint tests, the Sessions knowledge
test and the Server.Fast suite. Each step writes its log and exit code to OutDir and the run carries on when a step fails.

.DESCRIPTION
Non-destructive: it reads the checkpoint, builds the projects and runs tests. It writes nothing into the repository.
The CLI smoke step is expected to be refused on a host whose RAM is below the preflight estimate (see the plan's
RUNNING item); the refusal is recorded as the step's result, not hidden. Pass -Full to add the full Core gate.

.PARAMETER Trunk
Path to the first shard of the UD-IQ1_S checkpoint.

.PARAMETER OutDir
Folder for logs and the summary. Defaults to a timestamped folder under the temp directory.

.PARAMETER Budget
Host RAM budget given to scout, e.g. 63.3G.

.PARAMETER Full
Also run the full Core gate (about 13 minutes).
#>
param(
    [string]$Trunk = "E:\_models\qwen38-flash-next\UD-IQ1_S\Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
    [string]$OutDir = (Join-Path $env:TEMP ("qwen38-workflow-" + (Get-Date -Format "yyyyMMdd-HHmmss"))),
    [string]$Budget = "63.3G",
    [switch]$Full
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$cli = Join-Path $repo "src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe"
$coreExe = Join-Path $repo "tests\OpenTail.Stingray.Tests.Core\bin\Release\net10.0\OpenTail.Stingray.Tests.Core.exe"
$sessionsExe = Join-Path $repo "tests\OpenTail.Stingray.Tests.Sessions\bin\Release\net10.0\OpenTail.Stingray.Tests.Sessions.exe"
$serverExe = Join-Path $repo "tests\OpenTail.Stingray.Tests.Server.Fast\bin\Release\net10.0\OpenTail.Stingray.Tests.Server.Fast.exe"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$results = [System.Collections.Generic.List[object]]::new()

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    $start = Get-Date
    $log = Join-Path $OutDir (($Name -replace '[^A-Za-z0-9]+', '_') + ".log")
    $global:LASTEXITCODE = 0
    try {
        & $Body *>&1 | Out-File -FilePath $log -Encoding utf8
        $code = $global:LASTEXITCODE
    }
    catch {
        $_ | Out-File -FilePath $log -Append -Encoding utf8
        $code = -1
    }
    $results.Add([pscustomobject]@{
        Step    = $Name
        Exit    = $code
        Seconds = [int]((Get-Date) - $start).TotalSeconds
        Log     = $log
    })
    Write-Host ("[{0}] exit={1}" -f $Name, $code)
}

Invoke-Step "build-projects" {
    dotnet build (Join-Path $repo "src\OpenTail.Stingray.Cli") -c Release
    dotnet build (Join-Path $repo "tests\OpenTail.Stingray.Tests.Core") -c Release
    dotnet build (Join-Path $repo "tests\OpenTail.Stingray.Tests.Sessions") -c Release
    dotnet build (Join-Path $repo "tests\OpenTail.Stingray.Tests.Server.Fast") -c Release
}

Invoke-Step "scout-checkpoint" {
    & $cli scout -m $Trunk --format json --budget $Budget -c 4096 -o (Join-Path $OutDir "scout.json")
}

Invoke-Step "cli-smoke-expected-refusal" {
    & $cli -m $Trunk -p "Write a short paragraph about the ocean." -n 128 --temp 0
}

Invoke-Step "core-qwen4exp-tests" {
    $env:STINGRAY_RUN_HEAVY_TESTS = "1"
    foreach ($cls in @("Qwen4ExpSingleTokenTests", "Qwen4ExpTurnSnapshotTests", "Qwen4ExpMtpDecoderTests",
                       "Qwen4ExpTelemetryTests", "Qwen4ExpShardedLoadingTests")) {
        Write-Host "-- $cls"
        & $coreExe -class "OpenTail.Stingray.Tests.Core.$cls"
    }
}

Invoke-Step "sessions-knowledge-collection" {
    $env:STINGRAY_RUN_HEAVY_TESTS = "1"
    & $sessionsExe -class "OpenTail.Stingray.Tests.Sessions.QwenKnowledgeCollectionRealModelTests"
}

Invoke-Step "server-fast-suite" {
    & $serverExe
}

if ($Full) {
    Invoke-Step "core-full-gate" {
        Remove-Item Env:\STINGRAY_RUN_HEAVY_TESTS -ErrorAction SilentlyContinue
        & $coreExe
    }
}

$summary = Join-Path $OutDir "summary.txt"
$results | Format-Table -AutoSize | Out-String | Set-Content -Path $summary -Encoding utf8
Write-Host ""
Write-Host "Summary written to $summary"
$results | Format-Table -AutoSize
