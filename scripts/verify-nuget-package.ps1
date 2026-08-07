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
$packageProjects = @(
    @{ Id = 'OpenTail.Stingray'; Project = 'src\OpenTail.Stingray\OpenTail.Stingray.csproj' },
    @{ Id = 'OpenTail.Stingray.Server'; Project = 'src\OpenTail.Stingray.Server\OpenTail.Stingray.Server.csproj' },
    @{ Id = 'OpenTail.Stingray.Cli'; Project = 'src\OpenTail.Stingray.Cli\OpenTail.Stingray.Cli.csproj' }
)
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot 'artifacts\nuget'
}
else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $projectRoot)
}
$packagePaths = @{}
foreach ($package in $packageProjects) {
    $packagePaths[$package.Id] = Join-Path $OutputDirectory "$($package.Id).$PackageVersion.nupkg"
}
$smokeRoot = Join-Path ([IO.Path]::GetTempPath()) ("opentail-nuget-smoke-" + [guid]::NewGuid().ToString('N'))
$serverSmokeRoot = Join-Path ([IO.Path]::GetTempPath()) ("opentail-server-smoke-" + [guid]::NewGuid().ToString('N'))
$cliSmokeRoot = Join-Path ([IO.Path]::GetTempPath()) ("opentail-cli-smoke-" + [guid]::NewGuid().ToString('N'))

try {
    $versionLine = Select-String -LiteralPath (Join-Path $projectRoot 'Directory.Build.props') -Pattern '<Version>([^<]+)</Version>' | Select-Object -First 1
    if ($null -eq $versionLine -or $versionLine.Matches.Count -eq 0) {
        throw 'Could not find the release <Version> in Directory.Build.props.'
    }
    $sourceVersion = $versionLine.Matches[0].Groups[1].Value
    if ($sourceVersion -ne $PackageVersion) {
        throw "PackageVersion '$PackageVersion' does not match Directory.Build.props version '$sourceVersion'."
    }

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
            & dotnet test (Join-Path $projectRoot $testProject) -c $Configuration --verbosity minimal -- --minimum-expected-tests 1
            if ($LASTEXITCODE -ne 0) { throw "Managed release test failed: $testProject" }
        }
    }

    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    foreach ($package in $packageProjects) {
        & dotnet pack (Join-Path $projectRoot $package.Project) -c $Configuration --no-restore -o $OutputDirectory
        if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed: $($package.Id)" }
        if (-not (Test-Path -LiteralPath $packagePaths[$package.Id])) {
            throw "Expected package was not produced: $($packagePaths[$package.Id])"
        }
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $requiredEntriesByPackage = @{
        'OpenTail.Stingray' = @('README.md', 'THIRD_PARTY_NOTICES.md', 'lib/net10.0/OpenTail.Stingray.Core.dll', 'lib/net10.0/OpenTail.Stingray.Cpu.dll', 'lib/net10.0/OpenTail.Stingray.Engine.dll', 'lib/net10.0/OpenTail.Stingray.Vulkan.dll', 'lib/net10.0/OpenTail.Stingray.Cuda.dll')
        'OpenTail.Stingray.Server' = @('README.md', 'THIRD_PARTY_NOTICES.md', 'lib/net10.0/OpenTail.Stingray.Server.dll')
        'OpenTail.Stingray.Cli' = @('README.md', 'THIRD_PARTY_NOTICES.md', 'tools/net10.0/any/stingray.dll')
    }
    foreach ($package in $packageProjects) {
        $archive = [IO.Compression.ZipFile]::OpenRead($packagePaths[$package.Id])
        try {
            $entries = @($archive.Entries.FullName)
            $missing = @($requiredEntriesByPackage[$package.Id] | Where-Object { $_ -notin $entries })
            if ($missing.Count -gt 0) { throw "$($package.Id) is missing required entries: $($missing -join ', ')" }
        }
        finally { $archive.Dispose() }
    }

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

    # Server is a library intended for an ASP.NET Core host, so compile a clean Web SDK consumer
    # rather than attempting to run it without a model. This catches a broken framework-reference
    # or transitive package contract that a contents-only ZIP inspection cannot see.
    & dotnet new web --framework net10.0 --output $serverSmokeRoot --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local server-consumer smoke project.' }
    $serverSmokeProject = (Get-ChildItem -LiteralPath $serverSmokeRoot -Filter '*.csproj' -File | Select-Object -First 1).FullName
    if ([string]::IsNullOrEmpty($serverSmokeProject)) { throw 'The local server-consumer project file was not created.' }
    & dotnet add $serverSmokeProject package OpenTail.Stingray.Server --version $PackageVersion --source $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Local server-consumer package restore failed.' }
    @'
using OpenTail.Stingray.Server;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenTailStingray(options => options.ModelPath = "model.gguf");
var app = builder.Build();
app.MapOpenTailStingray();
'@ | Set-Content -NoNewline (Join-Path $serverSmokeRoot 'Program.cs')
    & dotnet build $serverSmokeProject -c $Configuration --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Local server-consumer package compilation failed.' }

    & dotnet tool install --tool-path $cliSmokeRoot OpenTail.Stingray.Cli --version $PackageVersion --add-source $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw 'CLI package installation from the local source failed.' }
    $cliCommand = Join-Path $cliSmokeRoot $(if ($IsWindows) { 'stingray.exe' } else { 'stingray' })
    & $cliCommand --version
    if ($LASTEXITCODE -ne 0) { throw 'CLI package did not execute its version command.' }
    Write-Host "NuGet package smoke test passed: $($packagePaths.Values -join ', ')"
}
finally {
    if (Test-Path -LiteralPath $smokeRoot) { Remove-Item -LiteralPath $smokeRoot -Recurse -Force }
    if (Test-Path -LiteralPath $serverSmokeRoot) { Remove-Item -LiteralPath $serverSmokeRoot -Recurse -Force }
    if (Test-Path -LiteralPath $cliSmokeRoot) { Remove-Item -LiteralPath $cliSmokeRoot -Recurse -Force }
}
