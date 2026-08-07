# Changelog

All notable user-visible changes are recorded here. The package version is the `<Version>` in
`Directory.Build.props`, matched by a `stingray-v<Version>` git tag; this file provides the
human-facing map that a raw commit graph cannot.

## Unreleased

### Added

- CPU execution for the published dense Llama/Mistral SafeTensors profile (F32/F16/BF16), with
  explicit capability boundaries; GGUF remains the recommended quantized deployment format.
- Recommended deployment profiles for CPU-only, CUDA dense, Vulkan, hybrid MoE, and local-server use.
- Guidance for the live model × backend × dtype × batching × speculation capability report.
- A release-quality matrix and retained CI test transcripts for package publication.
- Small llama-server compatibility endpoints: `POST /tokenize`, `POST /detokenize`, and
  `GET /props`, with wire-contract coverage for valid and malformed requests.

### Changed

- Capability reports explicitly distinguish experimental retained-session internals from a supported
  restart-continuation product feature.
- Hosted CPU CI now includes the Sessions and Vision managed suites, matching the release gate's
  managed coverage.
- **Breaking (library API):** `ServerRuntimeCapabilities` gained a `SessionRestartContinuation`
  positional parameter reporting that restart continuation is not a supported product feature. The
  capabilities JSON simply gains a field — additive and safe for HTTP consumers — but the record's
  primary constructor changed, so C# code constructing `ServerRuntimeCapabilities` directly must be
  updated.
- `--n-predict` now rejects negative values with an explanatory error instead of accepting them.
  llama.cpp's `-1` (until EOS) and `-2` (until context full) sentinels are not implemented; the
  default remains 512. This is deliberate: silently treating `-1` as "generate nothing" or as the
  default would be worse than saying so.

### Fixed

- Post-migration repository plumbing. The standalone repository was missing the root `global.json`
  that selects Microsoft.Testing.Platform, so `dotnet test` fell back to VSTest, found no adapter for
  xunit v3, and exited 0 having run **zero** tests — including in CI. Restored, and both workflows
  now trigger on this repository's layout (root solution, `main`) instead of the pre-migration
  subtree path and `master`. The release workflow's TRX receipts use the Microsoft.Testing.Platform
  reporter rather than the VSTest `--logger` option it cannot accept.
- Bounded-admission concurrency test raced the admission gate: it waited for only the first of five
  requests to reach the engine, so the request expected to be rejected could be admitted instead and
  block until the 100 s HTTP client timeout. The server suite now completes in ~3 s.
- `KnownEnvironmentVariables` drift guard no longer misreads the `STINGRAY_TUI` conditional-compilation
  symbol as an unregistered environment variable.
- Vulkan compile-fallback test skips, rather than fails, when the Vulkan SDK's `glslc` is absent;
  every shipped shader is served from the committed SPIR-V table, so it is a dev-tooling dependency.
- Managed CI, release validation, and the local package verifier now fail if test discovery finds
  zero tests, rather than allowing a green no-op test run.

## Release notes policy

- Add entries when a change alters user-visible behaviour, compatibility, performance claims,
  configuration defaults, or operational requirements.
- At a release tag, move relevant entries into a version/date heading and link the test receipt.
- Keep entries factual: name the backend/model scope and important limitation.
