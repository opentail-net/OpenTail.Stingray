# Changelog

All notable user-visible changes are recorded here. The package version is the `<Version>` in
`Directory.Build.props`, matched by a `stingray-v<Version>` git tag; this file provides the
human-facing map that a raw commit graph cannot.

## Unreleased

## 1.0.3 — 2026-08-08

Tag `stingray-v1.0.3`. Scope of this release: OpenAI `tool_choice:"required"` support, a session
optimistic-concurrency wire fix, and a set of robustness/packaging corrections found in review.
Backend and model-format scope is unchanged from 1.0.2 — GGUF remains the recommended quantized
deployment format, and the CPU-dense GGUF lane is still the only one with restart-continuation
evidence.

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
  (Superseded understanding: the Fixed entry below shows the cause is exact token
  **repetition**, not token type — an all-control prompt is typically the same token twice. This
  guard is retained because it is cheap and independently correct, but it is not why the class of
  prompt failed.)
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

- OpenAI `tool_choice` is now honoured or refused, never silently ignored. A named function
  (`{"type":"function","function":{"name":"X"}}`) narrows the constrainable tool set to X, so the
  argument grammar forbids every other tool name — the filter *is* the enforcement. Naming a
  function absent from `tools` is a 400 instead of falling back to unconstrained tool use.
  `tool_choice:"required"` is now honoured: a new `ForcedToolCallConstraint` masks the vocabulary
  down to the model family's tool-call open marker until that marker is emitted, then goes inert and
  hands over to the argument grammar it is AND-composed with. This is the inverse of the existing
  constraints, which arm *on* the open marker and so can shape a call but never cause one — hence a
  new per-family `IToolCallAdapter.BuildForcedCallConstraint`, implemented for Qwen, Qwen3-Coder,
  Llama-3, DeepSeek and Gemma 4. Two consequences worth knowing: forcing puts the marker in the
  *first* generated position, which suppresses any preamble or reasoning block for that turn (the
  strict reading of "required" — the response is a call, not prose containing one); and it is
  deliberately not gated on the server-wide `ToolGrammar` option, since that option governs default
  argument-shape constraining while `required` is an explicit per-request demand to be constrained.
  Where a call cannot be forced — no tool in `tools`, no vocabulary loaded, or a family whose marker
  is not a single token (Qwen3-Coder's `<function=` is text, so it forces the `<tool_call>` envelope
  instead; DeepSeek forces its outer block marker) — the request is a 400 rather than an unforced
  generation, because prose returned to a client that asked for a guaranteed call is undetectable at
  the client. Previously `required` was accepted and ignored. `"none"` and `"auto"` were already
  correct and are unchanged.
  <br>Combining `required` with a schema-constrained `response_format` is also a 400: both claim the
  first generated token — one for the open marker, one for JSON — so their masks intersect to
  nothing, and a fully masked vocabulary does not fail loudly (`Sampler.Softmax` falls back to
  uniform, `Greedy` to the raw logits), meaning the turn would come back unconstrained and satisfy
  neither request.
  <br>Named `tool_choice` still narrows without compelling: with `ToolGrammar` on, the grammar pins
  which tool may be called, but the model is not forced to start a call. Making a named choice force
  as well would reuse the same mechanism and is a deliberate follow-up rather than part of this
  change.

### Fixed

- **Fixed (sessions, wire contract):** `committed_revision` is now a usable optimistic-concurrency
  token. The API published the cursor's accepted-position count while `RunTurnAsync` validated
  `expected_revision` against the store's turn counter, so a session advertised revision 6 and then
  answered a turn carrying 6 with `409 "Expected revision 6, but current revision is 1"` — the only
  pattern the API admits (read the value, echo it back) failed on the second turn. Four sources now
  agree by construction: the endpoint publishes the store's counter, eviction persists it, restore
  seeds from the persisted value instead of re-deriving a position count, and validation is
  unchanged. `HotSession.CommittedRevision` is renamed `AcceptedPositionCount` so it cannot be
  mistaken for a token again. Manifest format is now v3 — same layout, revised meaning of the
  revision field; v1/v2 files remain readable and are not migrated, because the contract needs the
  sources to agree rather than to hold a particular number. Full analysis in
  `docs/session-revision-contract-defect.md`.
- **Fixed (shutdown safety):** `ContinuousBatchingEngine.Dispose` no longer lets owned native, GPU
  and mmap'd resources be released while the batcher is still running. It waited five seconds — far
  less than a large chunked prefill — and then returned regardless, after which the owning engine
  immediately disposed the forward pass, backend and mapped model underneath the live loop. The wait
  is now 60s and its result is published as `DrainedOnDispose`; when the batcher has not exited, the
  prefix caches and owned resources are deliberately leaked instead, since process exit reclaims
  them and freeing live memory is an access-violation-class race.
- **Fixed (GGUF robustness):** malformed or hostile GGUF files can no longer reach unsafe pointer
  arithmetic. `GetTensorDataPtr` performed no validation at all — no shard, range, overflow or
  disposal check — making it the weaker of two doors into the same mapping; both accessors now share
  one validated descriptor. Offsets are range-checked *before* the `ulong`→`long` cast that could
  otherwise produce a negative offset which passed the old `offset + size > fileSize` test; tensor
  dimensions are validated at parse time (rank ≤ 4, positive extents) with checked element-count
  multiplication so a wrapped size cannot be bounds-checked as small; `EnsureAvailable` rejects
  negative and overflowed positions rather than silently passing them; `SkipGgufValue` advances
  through the same bounds check instead of moving the position unchecked; and `Dispose` is
  idempotent, so a second call cannot unbalance the mapped-view refcount.
- **Fixed (startup leaks):** engine construction is exception-safe on both model paths. The GGUF path
  opened the model mapping ~85 lines before its cleanup scope began, so a failure in compatibility
  validation, hyperparameter derivation or tokenizer construction leaked the mapping and its file
  handles; the SafeTensors path cleaned up only on the tokenizer check, leaking the mapping plus the
  backend and forward pass on any later failure. Both now register each resource as it is created and
  release them in reverse on any failure, with an explicit ownership transfer to the engine.
- **Fixed (packaging):** the published `OpenTail.Stingray` meta-package no longer advertises a
  memory-tier API that cannot work. `Pipeline.MemoryHierarchy` (and its `Prefetcher`, `TierConfig`,
  `PrefetchRequest`) were public and bundled, while both of `MemoryHierarchy`'s operations throw
  `NotImplementedException` — a caller following the documentation got an exception on first use.
  They are now internal. The implemented three-tier MoE offload path is unaffected: it runs through
  `ExpertSlotManager`/`CudaExpertSlotManager` + `MoEPrefetcher` over this assembly's still-public
  `SlruCache`/`ExpertCache`.
- **Fixed (test honesty):** 644 model- and GPU-gated tests across 117 files returned early when their
  fixture was absent and were therefore counted as **passes**, so suite totals overstated what had
  actually executed. They now call `Assert.SkipUnless` and report as skipped — Tests.Core alone goes
  from 0 to 21 skips on a fixture-less machine. No test's assertions changed.
- **Fixed (server admission):** `POST /v1/sessions/{id}/turns` now sits behind the same bounded
  admission gate as the chat routes. It generates exactly like `/v1/chat/completions` and
  `/v1/messages` but was mapped without `WithConcurrencyLimit()`, so `MaxQueuedRequests`
  (`STINGRAY_MAX_QUEUE`) bounded only the three stateless routes while any number of named sessions
  could enqueue prompts alongside them — the limit was route-shaped rather than engine-shaped.
- **Fixed (CI):** `OpenTail.Stingray.Tests.ForwardPass` now runs in both the PR gate and the release
  workflow. It is the largest suite and carries most of the CPU inference, batching, KV-cache and
  quantisation-parity coverage, and it was in neither — so an engine regression could merge and
  publish with green checks.
- **Fixed (release safety):** the tag/`<Version>` guard no longer exempts `workflow_dispatch`. The
  documented "publication is tag-triggered only" policy was not enforced: a manual run on any ref
  could publish whatever `<Version>` was in `Directory.Build.props`, or report success having
  published nothing because `--skip-duplicate` discarded an already-released version. Manual runs
  must now be dispatched against a `stingray-v*` tag ref.

- **Fixed (int8 CPU prefill):** prompts consisting of a single repeated token collapsed the int8
  activation path — final-logit cosine 0.40-0.48 against the exact F32 route at every length from 2
  to 64 tokens, and -0.124 for a repeated space. It affected ordinary words too (`the` x8 measured
  0.470), not just whitespace or control tokens, so the pre-existing all-control-token guard did not
  cover it. One differing token restores full accuracy, so the fix is a distinct-token test:
  `ForwardPass` now routes single-distinct-token prompts to the exact F32 path at the same three
  sites as the control-token guard. With one repeated token the rows entering each matmul differ
  only by positional effects, so the signal rides as small differences on a large common component
  and per-row int8 scaling quantises it away. Two numerical hypotheses (embedding and
  activation-point dynamic range) were measured and disproved before landing this; see
  `docs/cpu-prefill-quality-gate.md`.

- `STINGRAY_PER_LAYER_HD_PREFILL=1` now fails fast with an explanation on per-layer-head-dim models
  (gemma4) instead of entering an unsafe path. It was documented as forcing batched prefill and
  producing "wrong output"; it actually produced an `AccessViolationException`, because the batched
  route indexes KV with the model-wide head dim (512) on layers carrying 256 and walks off the
  buffers. The switch existed to make the outstanding per-layer-head-dim prefill work measurable,
  which it could never do — a path that corrupts memory cannot be timed. The sequential route is
  unchanged and correct.


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
