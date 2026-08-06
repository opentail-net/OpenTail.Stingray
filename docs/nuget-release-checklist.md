# NuGet release checklist

Use this checklist for each `OpenTail.Stingray` release candidate. The version is supplied by the
release tag (`vX.Y.Z`) in CI; do not edit the repository-wide default merely to publish a release.

## Candidate gate

1. Start from the intended committed release candidate. Record its commit SHA and proposed tag.
2. Update [CHANGELOG.md](../CHANGELOG.md): move relevant Unreleased entries into the tagged release
   section and state compatibility limits, especially backend/model-format scope.
3. Run the managed release suite and package smoke test locally or on the designated release runner:

   ```powershell
   ./scripts/verify-nuget-package.ps1 -PackageVersion 1.0.0-rc.1
   ```

   The script runs the same managed projects as the hosted release gate (unless `-SkipTests` is
   explicitly chosen), packs the library, checks its README/notices and required assemblies, then
   creates a fresh .NET 10 consumer that restores and executes against that exact `.nupkg`.
4. Complete every applicable row in [release-quality-test-matrix.md](release-quality-test-matrix.md).
   A runner or hardware class that was not exercised remains **not run**, never a pass.
5. Inspect the `.nupkg` before publication. Confirm its package README states the current format and
   backend limits, and that no source-only or development-only package is being published.

## Publish and receipt

1. Create and push an annotated `vX.Y.Z` tag. The release workflow extracts `X.Y.Z`, builds, tests,
   packs, signs when configured, and publishes when the NuGet secret is present.
2. Download the release test artifact and retain the package SHA-256 plus the exact tag/commit in the
   release notes. Record the hardware/backend rows actually exercised.
3. Install the published package from NuGet.org into a clean consumer once, repeating the minimal
   compile/run smoke test. This verifies the public feed rather than only the local package folder.
4. Verify the NuGet gallery metadata: MIT licence, repository link, README rendering, tags, and
   symbol package visibility. Verify any CLI/server package separately because it is not covered by
   the `OpenTail.Stingray` library smoke test.

## Receipt template

```text
Release: OpenTail.Stingray X.Y.Z
Tag / commit: vX.Y.Z / <sha>
Package SHA-256: <hash>
Managed suite: <TRX artifact link or local transcript>
CPU / CUDA / Vulkan / hybrid rows run: <actual rows and hardware>
Package smoke: <local package path or CI log>
NuGet.org consumer smoke: <date, SDK, result>
Known skipped rows: <none or explicit list>
```
