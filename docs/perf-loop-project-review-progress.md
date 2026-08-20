# Project-by-project performance & quality review — progress log

Started 2026-08-20, self-paced /loop firing every ~30 min (cron `7,37 * * * *`, job `1f1cb024`).
One project examined directly (no subagents) per firing. Read this file first on every firing —
it is the source of truth across firings, not conversation memory.

**Scope as of firing 8 (2026-08-20): performance AND DRY/code-quality**, folded into the same
per-project read rather than run as a separate pass — reading the file once for both lenses.

**Method — performance:** read the project for genuine opportunities (not micro-nitpicks), and
only ship a change that's benchmarked before/after with the existing infra (interleaved runs,
best-of-N, skeptical of single-run deltas — see docs/cpu-performance-baseline.md methodology).
Never leave the build broken between firings.

**Method — DRY/quality:** note duplicated logic, dead code, and structurally confusing patterns
worth a follow-up ticket. Bar for *shipping* a refactor in-loop is high (behavior-preserving,
covered by existing tests, small — e.g. deleting confirmed-dead code); anything bigger gets
recorded here as a finding, not applied blind. Examples already surfaced before this scope
widened (see conversation, not yet logged as findings below): the "try fast kernel, BLAS as
last resort" dispatch pattern is copy-pasted across four call sites (`ForwardPass.
MatMulBatchedCached`, `SimdKernels.MatMulBatched` x2, `CpuBackend.Sgemm`) instead of going through
one shared helper — worth a ticket, not urgent; `Pipeline/Prefetcher.cs` is confirmed dead code
(own doc comment says it can't function) — safe deletion candidate; `Sessions/InferenceSession.cs`
(~1300 lines) is known-legacy pending deletion per `docs/030-delete-inferencesession-todo.md`.

**Context already covered today, don't re-open unless something new surfaces:** the OpenBLAS
ordering bugs across `SimdKernels.MatMulBatched`/`MatMulBatchedF32`/`ForwardPass.
MatMulBatchedCached` — see `docs/done/openblas-elimination-findings-2026-08-20.md`. That
investigation already picked over `OpenTail.Stingray.Cpu`'s prefill matmul path and
`OpenTail.Stingray.Engine`'s `ForwardPass.PrefillCore`/`ForwardPass.Attention`/`ForwardPass.
Helpers` in detail (found byte-identical to a known-good baseline, no further easy wins there).
Other corners of Cpu/Engine not touched by that investigation are still fair game.

## Projects (14 total)

- [ ] OpenTail.Stingray (root/meta) — likely low-yield, no hot-path code of its own; do near the end
- [x] OpenTail.Stingray.Core — tokenizer char-cache fix shipped (~4-8% faster Encode); PreTokenizerPatterns
      already uses build-time [GeneratedRegex]; GgufModel/ModelGraph are one-time load cost, not hot path;
      JinjaChatTemplate not yet reviewed (once per turn, not per-token — low urgency)
- [ ] OpenTail.Stingray.Cpu (partially covered — see note above; re-scan corners not covered)
- [x] OpenTail.Stingray.Cuda — reviewed independently (not just citing docs/done/gpu-review-log.md,
      though that doc's context — this machine has literally zero NVIDIA hardware, confirmed there —
      still applies). Checked hardware-independent anti-patterns specifically, since kernel-level
      tuning claims can't be judged without a real GPU: kernel compile-once-with-disk-cubin-cache
      (CompileAndLoadKernels), GpuBufferPool reuse to avoid cudaMalloc/cudaFree churn, pinned-host
      fast paths for transfers — all confirmed genuinely implemented, not just commented. No
      hardware-independent bug found. Declined to propose kernel-level tuning changes (SM
      occupancy/tile sizes/MMQ coverage) since they're unverifiable here and the repo's own history
      (gpu-review-log.md) records 5 wrong performance predictions from code-only CUDA reasoning —
      not repeating that. Finding 3 in that doc (Q6_K/Q5_K prefill MMQ) remains the one credible,
      unmeasured lead for whoever has real NVIDIA hardware.
- [x] OpenTail.Stingray.Vulkan — reviewed. Most already covered by docs/done/gpu-review-log.md
      (capability discovery, cooperative matrix, subgroup pinning — all verified against this actual
      AMD device). Checked what wasn't: VulkanPath2Dispatcher/VulkanMatMulPathConfig (experimental
      tiled GEMM) — correctly gated behind explicit opt-in (STINGRAY_VULKAN_PATH2=1), not
      auto-detected from hardware presence, so no BLAS-style silent-regression risk here. gpu-review-
      log.md already measured this integrated APU's Vulkan ceiling at 74.1 t/s prefill vs 150+ t/s
      CPU for the same model — a hardware bandwidth limit, not a software bug. No change made.
- [ ] OpenTail.Stingray.Engine (partially covered — see note above; re-scan corners not covered)
- [x] OpenTail.Stingray.Pipeline — reviewed. Prefetcher.cs is dead code (own doc comment: "cannot
      function", throws NotImplementedException via MemoryHierarchy.PromoteToGpuAsync; real impl is
      Engine's MoEPrefetcher — flagged for when the loop revisits Engine, not yet reviewed). SlruCache
      is a standard O(1) LinkedList+Dictionary SLRU; its O(n) eviction-candidate scan is fine at the
      small resident-expert-cache scale it runs at. No change made.
- [x] OpenTail.Stingray.Sessions — reviewed (HotSession.cs, KvMemoryGovernor.cs, InferenceSession.cs
      skipped — legacy/superseded path, docs/030-delete-inferencesession-todo.md). One minor finding:
      HotSession.WaitForStateReleaseAsync busy-polls with Task.Delay(1), but only on the cancel/
      exception recovery path in RunTurnAsync, not steady-state — fixing needs restructuring the
      in-use flag into a real signal on concurrency-sensitive commit/rollback code, not worth the
      risk for a cold-path win. KvMemoryGovernor's poll loop uses a real configurable interval for
      background housekeeping (intentional). No change made.
- [x] OpenTail.Stingray.Server.Host — reviewed. 29-line Program.cs, pure one-time startup wiring
      (env-var validation, DI registration, app.Run()). Nothing runs more than once per process.
      Nothing to optimize.
- [x] OpenTail.Stingray.TurboQuant — reviewed (KVarNCompressor, WalshHadamard, FastScan, KvCacheCompressor,
      TurboQuantOps). Already mature: pre-allocated scratch in ctors, stackalloc-only hot paths, SIMD
      butterfly network with sensible scalar fallback below vector width, fused dequant-dot avoids
      materializing decompressed vectors. No safe win found without deeper profiling; no change made.
- [x] OpenTail.Stingray.Diffusion — reviewed (DiffusionOps, ZImageDiT, FluxDiT, RRDBNet, TeaCache).
      Perf: DiT Forward() allocates several float[] buffers per call, runs once per denoising step
      (4-30x/image) — likely noise-level against real matmul/attention compute per this codebase's
      own precedent (perf-loop-progress.md's SiLU-fusion rejection, <1% measured). Declined to build
      a benchmark to confirm a low-probability win. TeaCache is already a sophisticated step-skipping
      accelerator, legitimate prior work. DRY: FluxDiT/ZImageDiT share shape and both delegate to
      DiffusionOps, but are genuinely different architectures (multi-stream MMDiT vs single-stream
      S3-DiT) — low-confidence, not close-diffed, not logged as a confirmed finding. No change made.
- [x] OpenTail.Stingray.Vision — reviewed. Preprocessor layer (22 architectures) correctly factored:
      thin per-arch wrappers (10-25 lines, verified by reading Gemma3/NemotronImagePreprocessor)
      delegating to shared BaseVisionPreprocessor/Gemma4VImagePreprocessor primitives — not a DRY
      issue despite the file count. DRY finding (confirmed, not speculative): encoder layer is
      inconsistent — PixtralVisionEncoder delegates its transformer block to shared
      VisionOps.Attention/AttentionGqa (34 call sites), but Gemma3VisionEncoder hand-rolls the same
      block inline against raw SimdKernels calls (LayerNorm/MatVec/Softmax/Gelu), zero VisionOps use.
      Not all 22 encoders checked individually. Not fixed: real multi-file surgery, most model
      fixtures not present locally to verify correctness post-change, and it's not costing measured
      throughput (same SimdKernels layer either way) — a maintainability finding, not urgent. No

      **Follow-up shipped same day, user-directed**: rather than converge Gemma3 onto the shared
      helper (would have been a regression — Gemma3's hand-rolled attention uses TensorPrimitives.Dot/
      Multiply/Add specifically because VisionOps.Attention's inner loops were raw scalar, "far too
      slow" at Gemma3's 4096-patch scale per its own doc comment), vectorized VisionOps.Attention AND
      AttentionGqa the same way instead. Benefits all 14 real encoders that call these (not just
      Pixtral) for free, with no regression risk to Gemma3 (untouched). Verified: existing
      VisionOpsTests (6/6 pass, includes hand-computed reference values and the attention-sink
      attenuation test). New permanent regression benchmarks added to VisionOpsBenchmarkTests.cs
      (Benchmark_Attention_ScalarVsVectorized, Benchmark_AttentionGqa_ScalarVsVectorized, old scalar
      kept verbatim as comparison baseline, matching the file's existing MatVec benchmark pattern):
      both show >1.2x speedup at 1024-token/16-head ViT-L scale and <1e-3 max numerical divergence
      (float reassociation, not a correctness bug) vs the pre-change scalar implementation.
      **Closed.** A full Tests.Vision suite run was attempted as an extra regression check but got
      through only 2 tests in 30 minutes before being killed (unrelated to this change — traced the
      hang: none of the three encoders in this repo with a real Forward()-level test and local model
      fixture (Gemma3, Gemma4V, Llama4 Scout) call VisionOps.Attention/AttentionGqa at all, so
      whatever the suite hung on, it wasn't exercising the changed code). None of the 14 real callers
      of the changed methods (CogVlm, DeepSeekOcr, DotsOcr, Exaone4, Granite4, HunyuanVl, InternVl,
      Llava, MimoVl, Nemotron, PaddleOcr, Pixtral, Step3Vl, YoutuVl) have a local Forward()-level test
      fixture, so the direct math-level verification above (hand-computed reference tests + numerical
      agreement against the retained pre-change scalar implementation) is the correctness evidence
      that actually covers this change, and it's solid. Treating this as done.
- [x] OpenTail.Stingray.Vision (VL end-to-end follow-up) — user-directed: downloaded a correctly-
      paired real VL model (InternVL3-2B, matched to the local mmproj via its source repo) to
      actually run the vectorized encoders end-to-end for the first time ever. Found and fixed two
      real bugs unrelated to the vectorization itself (both pre-existing, both in code that had
      literally never been exercised before today): (1) RunCommand.cs hardcoded --image to gemma4
      only, rejecting all 21 other working encoders — removed, UnifiedVisionPipeline.Open already
      dispatches everything generically. (2) VisionOps.GetTensorPtr<T> had no dtype verification,
      so a Q8_0-quantized mmproj got silently reinterpreted as raw F16, causing a genuine
      AccessViolationException inside MatVecF16 — added a dtype check that now throws a clear,
      catchable error instead. Neither bug is in Attention/AttentionGqa; the vectorization's own
      correctness evidence (math-level tests) stands unchanged. Full writeup:
      docs/done/vl-untested-code-findings-2026-08-20.md. Also added [Fact(Timeout=...)] to the 3
      real-model Forward()-level tests (Gemma3/Gemma4V/Llama4) so a genuinely stuck run fails fast
      instead of needing a manual kill after 30+ min, per user request. Quantized mmproj weights
      (the majority of what's locally available) still don't fully work — needs a dequant-aware
      MatVec path, real follow-up work, not attempted. Open design question raised but not acted
      on: GetTensorPtr<T>'s raw-pointer API is more primitive than this codebase's own TensorRef-
      based pattern used elsewhere (Cpu/Engine) — a Span<T> swap or full TensorRef unification is
      proposed follow-up, awaiting direction before touching it.
- [x] OpenTail.Stingray.Audio — reviewed. Real finding, NOT fixed this firing:
      Primitives/SpectralKernels.ComputePowerSpectrum (shared by F5TTS/Parakeet/QwenASR/Whisper mel
      extractors) is an O(N²) DFT with cached cos/sin tables, not a real FFT despite the "fast
      twiddle-accelerated" naming. Whisper's n_fft=400 is not a power of 2, so a correct fix needs
      mixed-radix or Bluestein's algorithm, not textbook radix-2 — real complexity. No existing test
      for SpectralKernels to safety-net a rewrite, and a subtle bug here would silently degrade
      transcription/synthesis quality in a way hard to eyeball-verify. Needs a dedicated pass (build
      a reference-signal correctness test first, then the algorithm), not a squeeze into one firing.
      No change made.
- [x] OpenTail.Stingray.Server — reviewed. HTTP/JSON API layer, not itself a compute hot path.
      JSON serialization is source-generated (OpenTailStingrayJsonContext), no reflection.
      ChatTemplateRenderer parses its Jinja template once in the constructor, held as an instance
      field, not re-parsed per request; regexes use [GeneratedRegex]. ModelRuntimeManager's locking
      is coarse but scoped to admin bookkeeping (model load/evict/route decisions), not held during
      per-token inference. No change made.
- [ ] OpenTail.Stingray.Cli / OpenTail.Stingray.Server.Host (thin frontends, likely low-yield — do last)

## Firing log

### Firing 1 (2026-08-20)
Reviewed `OpenTail.Stingray.Core`, focused on `GgufTokenizer.cs` (hot path: runs once per request).

- **`SpmMergePieces`** (BPE merge) is already a proper O(n log n) priority-queue implementation
  (adjacency-list + `PriorityQueue`, stale-candidate skip on dequeue), not the naive O(n²) scan.
  No action.
- **Candidate found, not shipped yet**: `EncodeByteLevelBpe`'s `EmitPiece` does
  `foreach (char ch in encoded) pieces.Add(ch.ToString());` — a fresh single-char string
  allocation per character, every call. `EncodeByteToGpt2`'s doc comment fixes the output range
  to a small, known set (printable ASCII, extended printable, U+0100–U+0142 — ~322 possible
  chars), so this is cacheable with a static `string[]` built once and indexed by char code.
  **Applied and kept.** Added a reusable opt-in benchmark instead of a one-off harness: `GgufTokenizer.Encode`
  now times itself behind `STINGRAY_PROFILE_TOKENIZE=1` and prints `[TokenizeProfile] {chars} ->
  {tokens} in {ms} ({chars/ms}, {tok/ms})` to stderr (mirrors the existing `STINGRAY_PROFILE_PREFILL`/
  `STINGRAY_PROFILE_DECODE` convention; registered in `KnownEnvironmentVariables.cs`). Runnable from
  PowerShell/bash directly against the CLI, no new project needed:
  `STINGRAY_PROFILE_TOKENIZE=1 stingray.exe -m <model> -f <bigfile.txt> -n 1 --temp 0 -g 0 --single-turn`.

  Measured on a 164,148-char / 39,149-token synthetic prompt (80x the standard benchmark prompt
  concatenated), best-of-N, current box:
  - Before: 93.7 – 94.2ms steady state (n=3, one 105.6ms cold run excluded)
  - After: 86.5 – 90.7ms steady state (n=6)
  - ~4-8% faster, consistent direction across repeated runs, not just a single sample.

  **Fix**: `EncodeByteLevelBpe`'s `EmitPiece` allocated a fresh single-char string via
  `ch.ToString()` per character per call. `EncodeByteToGpt2`'s output range is fixed and small
  (~0x143 possible chars, per its own doc comment), so replaced with a static `string[]` built
  once and indexed by char code. `Tests.Core` tokenizer suite still green (52 passed, 0 failed,
  12 skipped for missing model fixtures, unrelated).

  Modest win, not the main cost center — `SpmMergePieces`'s per-merge `ls + rs` string
  concatenation is the more expensive allocation and would need a token-id-based merge
  representation (not string-keyed) to remove, a much larger change out of scope for this pass.
