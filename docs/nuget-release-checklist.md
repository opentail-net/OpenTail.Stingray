# NuGet release checklist

Use this checklist for each `OpenTail.Stingray` release candidate. `<Version>` in
`Directory.Build.props` is the single source of truth. The release workflow is tag-triggered and
requires an exact matching `stingray-vX.Y.Z` tag; it refuses both a bare `vX.Y.Z` tag and a tag
whose version differs from `<Version>`.

## Candidate gate

1. Set the intended plain release version in `Directory.Build.props`, then commit the release
   candidate. Record its commit SHA and proposed `stingray-vX.Y.Z` tag.
2. Update [CHANGELOG.md](../CHANGELOG.md): move relevant Unreleased entries into the tagged release
   section and state compatibility limits, especially backend/model-format scope.
3. Run the managed release suite and package smoke test locally or on the designated release runner:

   ```powershell
   ./scripts/verify-nuget-package.ps1 -PackageVersion 1.0.5
   ```

   The script runs the same managed projects as the hosted release gate (unless `-SkipTests` is
   explicitly chosen), packs **all three published packages** (library, Server, and CLI), checks
   each package's README/notices and load-bearing assembly/tool entries, restores and executes a
   fresh .NET 10 library consumer, compiles a clean ASP.NET Core Server consumer, and
   installs/runs the CLI from the locally packed feed.
4. Complete every applicable row in [reference/release-quality-test-matrix.md](reference/release-quality-test-matrix.md).
   A runner or hardware class that was not exercised remains **not run**, never a pass.
   Run `./scripts/check-test-model-coverage.ps1` alongside the managed suite and retain its
   output with the receipt. It reports the locally present/absent real-model fixtures; a normal
   green test summary may include early-returning model-gated tests and is not a coverage claim.
5. Inspect the `.nupkg` before publication. Confirm its package README states the current format and
   backend limits, and that no source-only or development-only package is being published.

## Publish and receipt

1. Create and push an annotated `stingray-vX.Y.Z` tag matching `<Version>`. The release workflow
   verifies that match, builds, tests, packs, and publishes when the NuGet secret is present.
2. Download the release test artifact and retain the package SHA-256 plus the exact tag/commit in the
   release notes. Record the hardware/backend rows actually exercised.
3. Install the published package from NuGet.org into a clean consumer once, repeating the minimal
   compile/run smoke test. This verifies the public feed rather than only the local package folder.
4. Verify the NuGet gallery metadata: MIT licence, repository link, README rendering, tags, and
   symbol package visibility. Verify any CLI/server package separately because it is not covered by
   the `OpenTail.Stingray` library smoke test.

## Receipt template

```text
Release: OpenTail.Stingray 1.0.5
Tag / commit: stingray-v1.0.5 / <sha>
Package SHA-256: <hash>
Managed suite: <TRX artifact link or local transcript>
Model-fixture coverage: <check-test-model-coverage.ps1 transcript>
CPU / CUDA / Vulkan / hybrid rows run: <actual rows and hardware>
Package smoke: <local package path or CI log>
NuGet.org consumer smoke: <date, SDK, result>
Known skipped rows: <none or explicit list>
```
