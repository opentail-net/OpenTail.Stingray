param(
    [string]$Configuration = "Release",
    [string]$OutputDir = "./artifacts/nuget",
    [switch]$CleanOutput
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    if ($CleanOutput -and (Test-Path $OutputDir)) {
        Remove-Item -Path $OutputDir -Recurse -Force
    }

    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null

    # The packable projects (IsPackable=true). The version comes from Directory.Build.props.
    # For a release, use scripts/verify-nuget-package.ps1, which also runs the tests and smoke-tests the packages.
    $packProjects = @(
        "src/OpenTail.Stingray/OpenTail.Stingray.csproj",
        "src/OpenTail.Stingray.Server/OpenTail.Stingray.Server.csproj",
        "src/OpenTail.Stingray.Cli/OpenTail.Stingray.Cli.csproj"
    )

    foreach ($project in $packProjects) {
        Write-Host "Packing $project..." -ForegroundColor Cyan
        dotnet pack $project -c $Configuration -o $OutputDir
        if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed: $project" }
    }

    Write-Host "Done. Packages are in $OutputDir" -ForegroundColor Green
}
finally {
    Pop-Location
}

if ([Environment]::UserInteractive -and -not $env:CI) { Pause }
