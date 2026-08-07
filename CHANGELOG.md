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
- Opt-in named sessions for the CPU-dense GGUF server lane. With both `EnableSessions` and
  `SessionStorageDirectory` configured, cursor/KV state plus bounded completed-operation results
  survive restart; clients can inspect an operation or safely replay its idempotency key. Other
  backends and cache families remain explicitly unavailable.

### Changed

- Capability reports now distinguish hot-only sessions from the supported CPU-dense GGUF persisted
  lane, including whether bounded operation-result replay survives restart.
- Hosted CPU CI now includes the Sessions and Vision managed suites, matching the release gate's
  managed coverage.
- **Breaking (library API):** several public diagnostic records gained positional parameters, so
  their primary constructors changed. HTTP consumers are unaffected — the JSON only gains fields,
  which is additive — but C# code constructing these types directly must be updated:
  - `ServerRuntimeCapabilities` → `SessionRestartContinuation`, reporting whether restart
    continuation is available (true only for the persisted CPU-dense GGUF lane) with a reason.
  - `ServerApiSurface` → `SessionLifecycle`, separating "the `/v1/sessions` routes are served" from
    "session state survives a restart", which are different guarantees.
  - `ServerStatusSnapshot` → `Configuration`, carrying the bound server options, the int8
    activation-prefill gate, and the CPU batched-prefill receipt, so each is observable rather
    than inferred.

  New supporting records, none of which existed at 1.0.2: `ServerStatusConfiguration`,
  `ServerStatusCpuBatchedPrefill`, `ServerStatusBoundConfiguration`, `ServerRuntimeResolution`.
- CPU prefill no longer takes the int8 activation path for prompts made **entirely** of control /
  user-defined tokens, falling back to the exact F32 sequential route. Such prompts are structural
  probes rather than prose, and one two-token all-control input produced a final-logit cosine of
  −0.45 against the F32 result. Ordinary prompts — including the usual BOS + text — are unaffected
  and still measure 0.988–0.999. `STINGRAY_CPU_PREFILL_Q8=0` still disables int8 prefill entirely.
- `--n-predict` now rejects negative values with an explanatory error instead of accepting them.
  llama.cpp's `-1` (until EOS) and `-2` (until context full) sentinels are not implemented; the
  default remains 512. This is deliberate: silently treating `-1` as "generate nothing" or as the
  default would be worse than saying so.

- Recorded a measured CPU baseline against llama.cpp (`b8585`) on identical hardware, model and
  thread count — see `docs/cpu-benchmark-llamacpp-baseline.md`. On an AVX2-only machine with
  SmolLM2-1.7B Q4_K_M, prefill throughput crosses over at roughly 2500 tokens: llama.cpp leads by
  ~27% at 512 tokens and ~10% at 1024, the two are within ~2% at 2048, and Stingray is ~5% ahead at
  ~3100. Decode is at parity (26.4 vs 26.5 t/s). The short-prompt deficit has the signature of a
  fixed per-process cost rather than slower kernels — Stingray's throughput *rises* from 512 to
  1024 tokens before turning over, which is JIT of the hot SIMD kernels being amortised, and a
  long-lived server pays it once rather than per request. The note records the methodology
  asymmetry (warmed in-process AOT vs a fresh JIT-ing process per sample) because it favours
  llama.cpp at short prompts.
- Flash-64 prefill attention gains an opt-in `STINGRAY_PREFILL_ATTN_WIDE_HEADS=1` for head
  dimensions 128/256 (Llama/Qwen-class models). **Off by default, deliberately.** The
  wikitext-2 gate measures +0.52% perplexity for +14% prefill throughput on Qwen3-8B Q4_K_M —
  a real trade rather than a free win, and worse than the exact sequential path, so it is the
  model owner's call rather than a silent default. A per-prompt cosine check had suggested the
  divergence sat inside the envelope of approximations already shipped; the corpus gate showed
  it does not, which is why the corpus gate is the one that decides.
- Removed `STINGRAY_VABL`, a diagnostic ablation switch whose own comment read "Both non-zero
  modes produce WRONG output ... Revert before shipping". Its field was declared but referenced
  nowhere, and being registered meant `doctor` reported it to operators as a valid setting.
- CPU prefill attention is faster on the Flash-64 path: **+4.0%** end-to-end prefill throughput
  (SmolLM2-1.7B Q4_K_M, 1550 tokens, headDim 64, 12 logical CPUs, 6 interleaved rounds per cell,
  best-of-6; +4.3% on medians). Two changes, measured as a 2×2 because they overlap:
  the K-pack transpose is now an 8×8 AVX block rather than a scalar column walk
  (+3.0% alone, `STINGRAY_CPU_KPACK_SIMD=0` reverts), and the attention schedule now iterates KV
  tiles outside query tiles so each KV tile is packed once per group of query tiles instead of once
  per query tile (+1.6% alone, `STINGRAY_PREFILL_ATTN_KV_OUTER=0` reverts). Both are **bit-exact**
  with the previous output — a transpose only moves floats, and the reorder is a loop interchange
  that preserves each query's ascending KV order — so this is a throughput change with no numerical
  component. The reorder costs scratch: ~256 KB per thread instead of ~16 KB, tunable via
  `STINGRAY_PREFILL_ATTN_KV_OUTER_TILES`.
- CPU Q4_K integer dot products now use plain AVX-VNNI (`vpdpbusd`) where available, not only
  AVX-VNNI-INT8 (`vpdpbssd`). The previous gate covered Zen 5 / Granite Rapids-class parts only, so
  all of Zen 4 and Alder Lake through Raptor Lake fell back to the two-instruction AVX2 chain
  despite having the hardware; four of the six call sites with this shape had no VNNI path at all.
  All six now share one dispatch. The three branches are bit-identical on this data — Q4_K nibbles
  are 0-15 and Q8 activations |a| ≤ 127, so the AVX2 chain's one saturating step never saturates —
  and `STINGRAY_CPU_VNNI=0` forces the fallback so a VNNI-capable host can verify that. The
  throughput benefit is predicted from the instruction tables, **not measured**: the development
  machine is AVX2-only and cannot execute the new branch.

### Fixed

- Generation is now bounded by the active context, not just the prompt. `ForwardPass` sizes its
  attention-score and RoPE scratch from the context ceiling while its `PagedKvCache` defaults to
  131,072 positions, so decoding past the ceiling wrote past those native buffers instead of
  failing. Wiring `--ctx-size` into the CPU forward pass made this reachable in ordinary use —
  `--ctx-size 512` with the default `--n-predict 512` overruns after any prompt — and on the
  single-user server path the budget came straight from the client's `max_tokens`, which no layer
  bounded. `ForwardCore` now rejects an out-of-range position outright, so the invariant holds for
  every caller rather than relying on each one; the CLI and `InferenceEngine` stop cleanly at
  context-full (reporting length truncation) rather than reaching that check.
  `ContinuousBatchingEngine` already clamped this way and is unchanged.
- The CLI's interactive and image paths now enforce the same context bound as the single-prompt
  path. Interactive declines an oversized message and stays in the session; the image path checks
  the *expanded* length, since each placeholder becomes an open token, its soft tokens, and a close
  token — an image is easily hundreds of positions more than the token list suggests.
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
- Release validation packs and smoke-tests all three published packages: library execution, Server
  compilation in a clean ASP.NET Core host, and CLI install/version execution from the exact local
  feed. It no longer validates only the meta-package.

## Release notes policy

- Add entries when a change alters user-visible behaviour, compatibility, performance claims,
  configuration defaults, or operational requirements.
- At a release tag, move relevant entries into a version/date heading and link the test receipt.
- Keep entries factual: name the backend/model scope and important limitation.
