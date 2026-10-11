# Prove that what we tell NuGet users compiles against the PACKED package, not against internal source projects.
#
#   1. Packs the current checkout (OpenTail.Stingray and OpenTail.Stingray.Server) into a throw-away local feed.
#   2. Finds every C# code block in the READMEs that is marked with an HTML comment line  <!-- verify:consumer -->  immediately above the
#      ```csharp fence, writes each into its own isolated console project (outside the repo, PackageReference only) and builds it.
#      A block is a complete top-level program; blocks that use WebApplication get the Web SDK and the Server package.
#   3. Builds the samples that claim to be package consumers (Chat, Extract, Speech, Diffusion, ChatServer) with -p:StingrayPackageVersion=<version>,
#      which swaps their ProjectReference items for PackageReference, and checks the restore resolved no project references.
#
# This proves the code COMPILES against the public package. It does not run inference (that needs model files); where a sample or snippet has been
# run against real weights, its own README says so.
#
# Usage: pwsh scripts/verify-package-consumers.ps1 [-Configuration Release] [-Feed <dir> -Version <v>]   (-Feed skips packing)
# Exit code 0 only if every block and sample built.
param(
    [string]$Configuration = 'Release',
    [string]$Feed = '',
    [string]$Version = '',
    [string]$Work = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) ("stingray-package-check-" + [Guid]::NewGuid().ToString('N').Substring(0, 8)) }
New-Item -ItemType Directory -Force -Path $Work | Out-Null

if (-not $Version) {
    $m = Select-String -Path (Join-Path $repo 'Directory.Build.props') -Pattern '<Version>([^<]+)</Version>' | Select-Object -First 1
    if (-not $m) { throw 'cannot read <Version> from Directory.Build.props' }
    $Version = $m.Matches[0].Groups[1].Value
}

$results = [System.Collections.Generic.List[object]]::new()
function Add-Result([string]$name, [bool]$ok, [string]$detail) {
    $results.Add([pscustomobject]@{ name = $name; ok = $ok; detail = $detail })
    Write-Host ("{0,-4} {1}  {2}" -f $(if ($ok) { 'PASS' } else { 'FAIL' }), $name, $detail)
}

# --- 1. pack
if (-not $Feed) {
    $Feed = Join-Path $Work 'feed'
    New-Item -ItemType Directory -Force -Path $Feed | Out-Null
    foreach ($p in 'src/OpenTail.Stingray/OpenTail.Stingray.csproj', 'src/OpenTail.Stingray.Server/OpenTail.Stingray.Server.csproj') {
        $out = & dotnet pack $p -c $Configuration -o $Feed 2>&1 | Out-String
        $ok = ($LASTEXITCODE -eq 0)
        Add-Result "pack $p" $ok $(if ($ok) { "version $Version" } else { ($out -split "`n" | Where-Object { $_ -match 'error' } | Select-Object -First 2) -join ' ' })
        if (-not $ok) { throw "pack failed: $p" }
    }
}
$sources = "$Feed;https://api.nuget.org/v3/index.json"

# --- 2. README code blocks
$readmes = @('README.md', 'src/OpenTail.Stingray/README.md')
$blockNo = 0
foreach ($rm in $readmes) {
    $lines = Get-Content -LiteralPath (Join-Path $repo $rm)
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].Trim() -ne '<!-- verify:consumer -->') { continue }
        # next non-blank line must be the opening fence
        $j = $i + 1; while ($j -lt $lines.Count -and $lines[$j].Trim() -eq '') { $j++ }
        if ($lines[$j].Trim() -notmatch '^```csharp') { Add-Result "$rm line $($i + 1)" $false 'marker is not followed by a ```csharp block'; continue }
        $k = $j + 1; $code = [System.Collections.Generic.List[string]]::new()
        while ($k -lt $lines.Count -and $lines[$k].Trim() -ne '```') { $code.Add($lines[$k]); $k++ }
        $blockNo++
        $dir = Join-Path $Work ("block{0:D2}" -f $blockNo); New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $text = $code -join "`n"
        $web = $text -match 'WebApplication|MapOpenTailStingray'
        $server = $text -match 'OpenTail\.Stingray\.Server'
        $sdk = if ($web) { 'Microsoft.NET.Sdk.Web' } else { 'Microsoft.NET.Sdk' }
        $refs = "    <PackageReference Include=`"OpenTail.Stingray`" Version=`"$Version`" />"
        if ($server) { $refs += "`n    <PackageReference Include=`"OpenTail.Stingray.Server`" Version=`"$Version`" />" }
        @"
<Project Sdk="$sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <RestoreSources>$sources</RestoreSources>
  </PropertyGroup>
  <ItemGroup>
$refs
  </ItemGroup>
</Project>
"@ | Set-Content -LiteralPath (Join-Path $dir 'block.csproj') -Encoding utf8
        Set-Content -LiteralPath (Join-Path $dir 'Program.cs') -Value $text -Encoding utf8
        # The repo's Directory.Build.props must not apply to the isolated consumer: the work dir is outside the repo, so it does not.
        $out = & dotnet build (Join-Path $dir 'block.csproj') -c $Configuration 2>&1 | Out-String
        $ok = ($LASTEXITCODE -eq 0)
        $first = if ($code.Count -gt 0) { ($code | Where-Object { $_.Trim() } | Select-Object -First 1).Trim() } else { '' }
        Add-Result "$rm line $($i + 1) ($first)" $ok $(if ($ok) { 'builds against the packed package' } else { (($out -split "`n" | Where-Object { $_ -match 'error' } | Select-Object -First 2) -join ' ' -replace '\[[A-Z]:\\[^\]]*\]', '') })
    }
}

# --- 3. samples in package mode
foreach ($s in 'OpenTail.Stingray.Sample.Chat', 'OpenTail.Stingray.Sample.Extract', 'OpenTail.Stingray.Sample.Speech', 'OpenTail.Stingray.Sample.Diffusion', 'ChatServer') {
    $proj = Join-Path $repo "samples/$s"
    $out = & dotnet build $proj -c $Configuration "-p:StingrayPackageVersion=$Version" "-p:RestoreSources=$($sources -replace ';', '%3B')" 2>&1 | Out-String
    $ok = ($LASTEXITCODE -eq 0)
    $detail = ''
    if ($ok) {
        $assets = Get-Content -Raw -LiteralPath (Join-Path $proj 'obj/project.assets.json')
        $projRefs = ([regex]::Matches($assets, '"type":\s*"project"')).Count
        if ($projRefs -gt 0 -or $assets -notmatch "OpenTail\.Stingray(\.Server)?/$([regex]::Escape($Version))") { $ok = $false; $detail = "restored $projRefs project reference(s) or no package: not a package build" }
        else { $detail = 'builds against the packed package, no project references' }
    } else { $detail = (($out -split "`n" | Where-Object { $_ -match 'error' } | Select-Object -First 2) -join ' ' -replace '\[[A-Z]:\\[^\]]*\]', '') }
    Add-Result "sample $s" $ok $detail
}

$failed = @($results | Where-Object { -not $_.ok })
Write-Host ""
Write-Host ("{0} checks, {1} failed. Work dir: {2}" -f $results.Count, $failed.Count, $Work)
if ($failed.Count -gt 0) { exit 1 }
exit 0
