[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$')]
    [string] $PackageVersion,
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Release',
    [string] $OutputDirectory,
    [switch] $SkipTests
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packageProject = Join-Path $projectRoot 'src\OpenTail.Stingray\OpenTail.Stingray.csproj'
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot 'artifacts\nuget'
}
else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $projectRoot)
}
$packagePath = Join-Path $OutputDirectory "OpenTail.Stingray.$PackageVersion.nupkg"
$smokeRoot = Join-Path ([IO.Path]::GetTempPath()) ("opentail-nuget-smoke-" + [guid]::NewGuid().ToString('N'))

try {
    if (-not $SkipTests) {
        # Match the hosted release gate. ForwardPass includes designated-hardware/model tests;
        # those are recorded separately in the release matrix rather than silently skipped here.
        $releaseTestProjects = @(
            'tests\OpenTail.Stingray.Tests.Core\OpenTail.Stingray.Tests.Core.csproj',
            'tests\OpenTail.Stingray.Tests.Pipeline\OpenTail.Stingray.Tests.Pipeline.csproj',
            'tests\OpenTail.Stingray.Tests.Server\OpenTail.Stingray.Tests.Server.csproj',
            'tests\OpenTail.Stingray.Tests.TurboQuant\OpenTail.Stingray.Tests.TurboQuant.csproj',
            'tests\OpenTail.Stingray.Tests.Cli\OpenTail.Stingray.Tests.Cli.csproj',
            'tests\OpenTail.Stingray.Tests.Sessions\OpenTail.Stingray.Tests.Sessions.csproj',
            'tests\OpenTail.Stingray.Tests.Vision\OpenTail.Stingray.Tests.Vision.csproj'
        )
        foreach ($testProject in $releaseTestProjects) {
            & dotnet test (Join-Path $projectRoot $testProject) -c $Configuration --verbosity minimal
            if ($LASTEXITCODE -ne 0) { throw "Managed release test failed: $testProject" }
        }
    }

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    & dotnet pack $packageProject -c $Configuration --no-restore -p:MinVerVersionOverride=$PackageVersion -o $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'dotnet pack failed.' }
    if (-not (Test-Path -LiteralPath $packagePath)) { throw "Expected package was not produced: $packagePath" }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($packagePath)
    try {
        $entries = @($archive.Entries.FullName)
        $requiredEntries = @('README.md', 'THIRD_PARTY_NOTICES.md', 'lib/net10.0/OpenTail.Stingray.Core.dll', 'lib/net10.0/OpenTail.Stingray.Cpu.dll', 'lib/net10.0/OpenTail.Stingray.Engine.dll', 'lib/net10.0/OpenTail.Stingray.Vulkan.dll', 'lib/net10.0/OpenTail.Stingray.Cuda.dll')
        $missing = @($requiredEntries | Where-Object { $_ -notin $entries })
        if ($missing.Count -gt 0) { throw "Package is missing required entries: $($missing -join ', ')" }
    }
    finally { $archive.Dispose() }

    New-Item -ItemType Directory -Path $smokeRoot | Out-Null
    & dotnet new console --framework net10.0 --output $smokeRoot --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local-consumer smoke project.' }
    $smokeProject = (Get-ChildItem -LiteralPath $smokeRoot -Filter '*.csproj' -File | Select-Object -First 1).FullName
    if ([string]::IsNullOrEmpty($smokeProject)) { throw 'The local-consumer project file was not created.' }
    & dotnet add $smokeProject package OpenTail.Stingray --version $PackageVersion --source $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Local-consumer package restore failed.' }
    @'
using OpenTail.Stingray.Core;
Console.WriteLine(typeof(GgufModel).Assembly.GetName().Name);
'@ | Set-Content -NoNewline (Join-Path $smokeRoot 'Program.cs')
    & dotnet run --project $smokeProject -c $Configuration --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Local-consumer package execution failed.' }
    Write-Host "NuGet package smoke test passed: $packagePath"
}
finally {
    if (Test-Path -LiteralPath $smokeRoot) { Remove-Item -LiteralPath $smokeRoot -Recurse -Force }
}
