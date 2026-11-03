<#
.SYNOPSIS
  Runs the three task recipes (chat, speak, transcribe) end to end and prints PASS or FAIL for each.
.DESCRIPTION
  Downloads the pinned catalog files for each task into a temporary model home and uses a temporary
  config directory (STINGRAY_MODEL_HOME, STINGRAY_CONFIG_DIR), so the owner's models and settings
  (including %APPDATA%\stingray\favourites.json) are never read or written. Needs network access to
  Hugging Face and about 1 GB of disk; both temporary directories are removed afterwards.
  The speak voice is research-use only. Pass -AcceptLicence to accept that licence for the run; without it speak and
  transcribe are reported as SKIP (not as a pass).
  Checks: chat answers a fixed arithmetic prompt with non-empty text; speak writes a WAV with audio
  samples; transcribe of that WAV returns text containing the spoken words.
.PARAMETER Cli
  Command that runs the CLI. Default: the repository build (dotnet run is avoided; pass a published
  `stingray` executable path to qualify an installed package).
#>
[CmdletBinding()]
param(
    [string] $Cli,
    [string] $Configuration = 'Release',
    [string] $SpokenText = 'Hello from the qualification script.',
    [switch] $AcceptLicence,
    [switch] $KeepTemp
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($Cli)) {
    $project = Join-Path $projectRoot 'src\OpenTail.Stingray.Cli\OpenTail.Stingray.Cli.csproj'
    & dotnet build $project -c $Configuration | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'CLI build failed.' }
    $exe = Get-ChildItem (Join-Path $projectRoot "src\OpenTail.Stingray.Cli\bin\$Configuration") -Recurse -Include 'stingray.exe', 'stingray' -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $exe) { throw 'Built CLI executable not found.' }
    $Cli = $exe.FullName
}

$temp = Join-Path ([IO.Path]::GetTempPath()) ('stingray-qualify-' + [guid]::NewGuid().ToString('N'))
$modelHome = Join-Path $temp 'models'
$configDir = Join-Path $temp 'config'
New-Item -ItemType Directory -Force -Path $modelHome, $configDir | Out-Null
$saved = @{ H = $env:STINGRAY_MODEL_HOME; C = $env:STINGRAY_CONFIG_DIR }
$env:STINGRAY_MODEL_HOME = $modelHome
$env:STINGRAY_CONFIG_DIR = $configDir
$results = [ordered]@{}

function Invoke-Task([string] $Name, [scriptblock] $Body) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $detail = & $Body
        $results[$Name] = @{ Pass = $true; Detail = "$detail"; Seconds = $sw.Elapsed.TotalSeconds }
    }
    catch {
        $results[$Name] = @{ Pass = $false; Detail = $_.Exception.Message; Seconds = $sw.Elapsed.TotalSeconds }
    }
}

try {
    $wav = Join-Path $temp 'speech.wav'
    Invoke-Task 'chat' {
        & $Cli setup chat --yes 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "setup chat exited $LASTEXITCODE" }
        $out = & $Cli chat -p 'What is 2+2? Answer with the digit only.' -t 0 -n 16 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "chat exited $LASTEXITCODE" }
        if ($out -notmatch '4') { throw "reply does not contain 4: $out" }
        'reply contains 4'
    }
    if (-not $AcceptLicence) {
        $results['speak'] = @{ Skip = $true; Detail = 'the voice licence is research-use only; rerun with -AcceptLicence'; Seconds = 0 }
        $results['transcribe'] = @{ Skip = $true; Detail = 'needs the speak output'; Seconds = 0 }
    }
    else {
    Invoke-Task 'speak' {
        & $Cli setup speak --yes --accept-licence 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "setup speak exited $LASTEXITCODE" }
        & $Cli speak -t $SpokenText -o $wav 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "speak exited $LASTEXITCODE" }
        $len = (Get-Item $wav).Length
        if ($len -lt 20000) { throw "WAV is only $len bytes" }
        "$len bytes of WAV"
    }
    Invoke-Task 'transcribe' {
        if (-not (Test-Path $wav)) { throw 'no WAV from the speak step' }
        & $Cli setup transcribe --yes 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "setup transcribe exited $LASTEXITCODE" }
        $out = & $Cli transcribe -i $wav --no-timestamps 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "transcribe exited $LASTEXITCODE" }
        if ($out -notmatch '(?i)hello') { throw "transcript lacks 'hello': $out" }
        'transcript contains hello'
    }
    }
}
finally {
    $env:STINGRAY_MODEL_HOME = $saved.H
    $env:STINGRAY_CONFIG_DIR = $saved.C
    if (-not $KeepTemp) { Remove-Item -Recurse -Force $temp -ErrorAction SilentlyContinue }
}

$failed = 0
foreach ($name in $results.Keys) {
    $r = $results[$name]
    $label = if ($r.Skip) { 'SKIP' } elseif ($r.Pass) { 'PASS' } else { $failed++; 'FAIL' }
    '{0,-11} {1}  {2:N1}s  {3}' -f $name, $label, $r.Seconds, $r.Detail
}
exit $failed
