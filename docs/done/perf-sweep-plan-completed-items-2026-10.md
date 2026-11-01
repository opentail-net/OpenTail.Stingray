# PerformanceLeague sweep: completed items split from perf-sweep-plan.md (2026-10-30)

Finished items moved verbatim out of [../4-performance/perf-sweep-plan.md](../4-performance/perf-sweep-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Phase 11 — Citrinet-ASR (~0.20-0.25x, worse than Whisper Tiny's Phase 5.1 target)
- [x] 11.1 Checked: `CitrinetAsr.cs` uses `JasperKernels.Conv1d` exclusively (depthwise-separable
      conv, not simple Linear/matvec layers — the Voxtral/Pass-A pattern doesn't directly apply
      here at all, this is architecturally a different shape of kernel). Inspected
      `JasperKernels.Conv1d` itself: already reasonably optimized — `Parallel.For` across output
      channels in every branch (depthwise, pointwise-1x1, general), and the pointwise 1x1 fast
      path already uses `TensorPrimitives.MultiplyAdd` (real SIMD). **No naive-unwired-SIMD bug
      found** — this is NOT a repeat of Voxtral's bug class.

> from: ## Phase 11 — Citrinet-ASR (~0.20-0.25x, worse than Whisper Tiny's Phase 5.1 target)
- [x] 11.2 (done 2026-10-01) **The 0.20x was apples-to-oranges.** It set our single cold run (0.872 s, mostly .NET
      JIT) against audio.cpp's whole-process time (171 ms); audio.cpp's own `--metrics` inference wall is
      31-39 ms (8 thr; 39-41 at 16). Warm, ours is ~60-70 ms (mel 10.8 ms + encoder 49-59 ms, 20 reps,
      `CitrinetAsrRealWeightsTests` pipeline), i.e. **~0.6x warm**; cold first call 570-850 ms is JIT (NativeAOT
      territory, not kernels). Encoder split by conv kind (instrumented, excl. first run): pointwise 1x1 ~2/3,
      depthwise ~1/4, general (3 calls) the rest; ~235 conv calls of ~0.2 ms each, each with its own
      `Parallel.For` and fresh `float[][]` allocations, so it is dispatch/allocation-bound, not FLOP-bound.
      Original text: given 11.1 found no quick pattern-match win, real per-stage profiling (mirroring Phase
      3.1/9.1b's `Stopwatch`-around-stages methodology) is the correct next step to find where the
      ~0.20x gap actually comes from (mel extraction? which Jasper block? the CTC decode head?) —
      NOT attempted this pass given the small absolute times involved (872ms total) make coarse
      instrumentation noisy; would need enough repetitions to trust a split, as originally noted.

> from: ## Phase 10 — Vision-Language Model decode weakness (~0.48x pattern, worst point 0.41x prefill)
        - [x] Our int8 is not ggml's: SmolLM2 F32 is 0.03% from llama.cpp PPL, our int8 0.69%.
              **Cross-checked 2026-10-02** (ADR-0003 "Known gap"): kernels exact on both sides;
              decode Q8_KS about 2.5x more precise than ggml's Q8_K; batched Q4_K prefill uses
              ggml's Q8_K. Not a defect. Open: a Q8_KS repacked GEMM, and the 0.69%'s actual source.

> from: ## Phase 10 — Vision-Language Model decode weakness (~0.48x pattern, worst point 0.41x prefill)
- [x] 10.3 **Resolved on re-measure 2026-10-02:** `-g 99 --backend vulkan`, 620-token prompt,
      iGPU (Radeon 5700G): Granite-Vision-3.2-2B decode **12.1 t/s** (was 4.1), prefill 12.8 t/s;
      Granite-4.0-3B-Vision on the same run 10.1 t/s decode, 42.9 prefill. The 3.2-specific
      anomaly is gone (now in line with the 3B checkpoint); some change since 2026-09-11 fixed it,
      not bisected. Vulkan still trails CPU here (12.1 vs 25.9 t/s), which on this iGPU-only
      machine is not evidence about discrete GPUs (CLAUDE.md rule 13). Q3_K Vulkan prefill (12.8)
      vs Q4_K (42.9) may deserve its own look on real GPU hardware. Original item:
      Separately investigate the Granite-Vision-3.2-2B Vulkan decode regression (4.1 vs
      17.1 t/s CPU) — real, flagged, unexplained gap distinct from the usual "Vulkan trails CPU on
      prefill only" pattern seen everywhere else in this doc.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
- [x] 1.1 Root-cause: `VoxtralTextDecoder`/`VoxtralAudioEncoder`'s `LinearNoBias`/`LinearBias`
      used naive scalar triple-loop matvecs, never wired to this codebase's existing AVX2/FMA
      `SimdKernels.MatVecF32`. Fixed. **Result: 1203.1s → 214.8s mean (5.6x), ratio 0.02x → 0.11x.**
      Correctness re-verified (5/5 tests, exact reference transcript). Recorded in
      `PerformanceLeague.md` 2026-09-12.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
      - [x] 1.2a `FastVectorTypeConverter.ConvertF32ToQ8_0` was flagged in its own doc comment as
            "never verified... a quantizer that runs and is subtly wrong is worse than either" —
            wrote `ConvertF32ToQ8_0VerificationTests` (aggregate RMS relative error vs a real F32
            matvec, not a single-row bound which is too noisy for random quantization error).
            **Passes: RMS relative error < 5%, correct behavior for a real 8-bit quantizer.**
            Cleared to wire into a real pipeline. Committed as a real, durable test (not scratch).

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
      - [x] 1.2b `VoxtralTextDecoderWeights` (NOT yet `VoxtralAudioEncoderWeights` — text decoder
            only this pass) linear-layer fields (`QWeight`/`KWeight`/`VWeight`/`OWeight`/
            `GateWeight`/`UpWeight`/`DownWeight`/`Ada1Weight`/`Ada2Weight`, plus a Q8_0 copy of the
            tied lm_head `EmbedTokensWeight`) now quantized to Q8_0 at load time via 1.2a's
            verified converter. `EmbedTokensWeight` itself stays F32 too (needed for per-token
            embedding-row lookup, a direct index, not a matvec). `LinearNoBias` switched to
            `SimdKernels.MatVecQ8_0`. Builds clean; one test fixup needed
            (`VoxtralTextDecoderLoadTests` asserted `float.IsFinite` on now-`byte[]` weight
            buffers — changed to a non-emptiness check, correctness is covered by 1.2a's seam
            test + this file's own finite-logits assertion).

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
      - [x] 1.2c Re-verified with REAL audio: 5/5 tests pass, transcript still exact match
            ("This little work was finished in the year 1803, and intended for immediate
            publication."), logits shifted only slightly (mean -31.182 vs -31.115 pre-quant,
            std 1.330 vs 1.327 — expected quantization noise, no quality loss).

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
      - [x] 1.2d Re-benchmarked: 213.458s → 164.163s (1.3x), ratio 0.11x → 0.147x. Cumulative
            from original baseline: 1203.1s → 164.163s = **7.3x**, ratio 0.02x → 0.147x. Recorded
            in `PerformanceLeague.md` 2026-09-12. Still 6.8x off the C++ reference — not
            plateaued yet.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
- [x] 1.3 Cache `TimeEmbedding`/`AdaGate` across decode steps. Implemented (`AdaScalePerLayer` on
      `KvCache`), correctness re-verified byte-identical (2/2 tests). **Measured: 213.458s vs
      214.804s baseline — no real win (<1% noise).** Kept anyway (free, correct), but confirms
      this was never the real lever — the 32-dim ops were negligible next to the 3072-dim
      matvecs. Recorded in `PerformanceLeague.md` 2026-09-12.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
      - [x] Implemented `VoxtralPipeline : ISpeechToTextPipeline`
            (`src/OpenTail.Stingray.Audio/VoxtralRealtime/VoxtralPipeline.cs`) wrapping the exact
            mel→audio-tower→prefill→incremental-decode→tekken-decode sequence
            `VoxtralGenerationLoopTests` already golden-verified by hand. Extracted the test's
            inline tekken-vocab decode into a shared `TekkenVocab` class (DRY, CLAUDE.md rule 7)
            instead of duplicating it. Builds clean. Wrote a new real end-to-end test
            (`VoxtralPipelineEndToEndTests`, task id bh8cfwxzw as of this writing) exercising the
            actual pipeline class against real weights/audio. **PASSED**: exact transcript match
            through the real public `ISpeechToTextPipeline` interface, not just test-harness code.
            Voxtral now has a real usable pipeline. Phase 1.4 done.

> from: ## Phase 9 — Worst-RTF TTS/audio-generation pipelines (114.14x — worse than Voxtral's original 85.5x)
- [x] 9.1 Checked the same class of question already fruitful for Voxtral: is ACE-Step using
      `SimdKernels`'s SIMD path, or a naive scalar loop? **Already SIMD — this is NOT a repeat of
      Voxtral's bug.** `AceStepDiT.cs` (`src/OpenTail.Stingray.Diffusion/AceStep/Transformer/`)
      imports and uses `OpenTail.Stingray.Audio.Primitives.DenseKernels.Linear`/`LinearNoBias`,
      the shared dense-math helper used by every Transformer/Conformer pipeline in this codebase
      — and that helper already calls `SimdKernels.MatVecF32` (`DenseKernels.cs:16-37`), same as
      the fix that gave Voxtral its 5.6x win. So ACE-Step's 114.14x RTF is NOT explained by an
      unwired-SIMD bug; whatever is slow here is a different, real bottleneck (attention cost,
      VAE decode, text encoder, CFG step count, or genuine compute floor for this DiT's size) that
      needs actual profiling, not another quick kernel-wiring fix. "Turbo" implies an 8-step fast
      schedule was intended and the harness does run only 8 steps — so the slowness isn't an
      inflated step count either; the per-step cost itself is the problem.
      - [x] 9.1b Added real `Stopwatch`-based per-stage timing to `AceStepPipeline.Generate`
        (`src/OpenTail.Stingray.Diffusion/AceStep/AceStepPipeline.cs`), gated behind the same
        `STINGRAY_PROFILE_DECODE=1` env var (reused, not a new one, per DRY) mirroring
        `HybridGdnForwardPass`'s pattern: text encoder / silence-VAE-encode / timbre+condition /
        DiT flow scheduler / VAE decode, each timed and reported as % of total. Builds clean. A
        profiled run of `AceStepPerfBaselineDebugTest` (task id b3vdj0a71 as of this writing) is
        running in the background — this model is slow (~228s baseline mean), expect a long wait.
        READ ITS OUTPUT before proposing any fix — do not guess which stage dominates.
      - [x] **Real profiled result (3 runs, mean 213.894s total)**: Silence VAE encode
        **82.93-84.99% of total time (175-186s per run)** — vastly dominates. DiT flow scheduler
        only 9.41-10.68%, VAE decode 3.62-3.88%, timbre+condition 1.55-2.32%, text encoder
        0.36-0.42%. **Overturns the natural assumption that the DiT would dominate.**
      - [x] **Root cause found and fixed, not a kernel bug**: `ComputeSilenceLatent(frames)`
        encodes a fixed all-zero silence PCM buffer through the VAE encoder — a PURE function of
        `frames` alone (`= max(750, latentFrames)`, derived only from `DurationSeconds`), with
        zero dependence on prompt/lyrics/seed/timbre. It was being fully recomputed from scratch
        on every single `Generate()` call despite being 100% deterministic. Added a
        `Dictionary<int, float[][]>` memoization cache keyed by `frames` on `AceStepPipeline`
        (`_silenceLatentCache`). Verified safe to cache (not just "looks safe"): checked both
        downstream consumers of the cached rows —
        `AceStepFlowScheduler.Generate`'s `srcLatents` usage only ever reads via `Array.Copy(
        srcLatents[t], 0, row, ...)` into a new local row, and `AceStepTimbreEncoder.Forward` only
        reads its input rows into a freshly-allocated `embeds` array — neither mutates the cached
        arrays in place. Builds clean.
      - [x] Correctness re-check run: `AceStepPipelineEndToEndTests` FAILED ("left channel
        contains NaN/Inf") — but confirmed via `git stash`/re-run on the UNMODIFIED code that this
        is a real, PRE-EXISTING bug unrelated to the caching change (identical failure, same
        message, on code before this fix existed). Not something this phase caused or is
        responsible for fixing. `AceStepOobleckEncoderGoldenParityTests` skipped (no golden dump
        present on this machine, unrelated). Caching fix confirmed safe to keep.
      - [x] Perf benchmark ran: **213.894s → 201.607s mean — only ~6% faster, NOT the expected
        ~80%+ drop**. Re-profiled with `STINGRAY_PROFILE_DECODE=1` and found silence-encode was
        STILL 82-85% on ALL 4 calls (warmup + 3 timed) — the cache was never actually being hit.
      - [x] **Real bug found in the fix itself, not the model**: `ComputeSilenceLatent` checked
        the cache (`_silenceLatentCache.TryGetValue`) but never WROTE to it after computing —
        `_silenceLatentCache[frames] = rows;` was simply missing before `return rows;`. A real,
        now-fixed bug in this fix's own first version (embarrassing but honest — recorded here
        rather than glossed over). Fixed, rebuilt clean.
      - [x] **First re-benchmark attempt was INVALID** — result showed no improvement
        (229.684s mean, worse than baseline), which was suspicious given the cache-store fix was
        real and verified in source. Checked DLL timestamps: the test project's own copy of
        `OpenTail.Stingray.Diffusion.dll` was STALE (18:20, built before the fix) versus the
        source project's fresh build (18:50) — building only `src/OpenTail.Stingray.Diffusion`
        does not automatically refresh a referencing test project's bin-folder copy; the TEST
        project itself must be rebuilt too. A real, generalizable process lesson for this whole
        sweep: **after any source-project fix, rebuild the TEST project before trusting a
        benchmark run, not just the source project** — a stale referenced DLL silently produces a
        "no improvement" result that looks like a real negative finding but is actually a build
        artifact. Rebuilt `tests/OpenTail.Stingray.Tests.Diffusion` properly (confirmed DLL
        timestamp now matches). **Real number, confirmed: 228.29s → 31.210s mean
        (30.434/31.789/31.406), a 7.3x speedup, RTF 114.14x → 15.605x.** ACE-Step is no longer the
        worst RTF in `PerformanceLeague.md`. Recorded. **Phase 9's core fix is done and verified.**
      - **Honest scope note for the writeup**: this is a repeated-calls-on-one-instance win (the
        realistic server-serving pattern, and exactly what the existing perf harness already
        measures), not a cold-single-call win — the very first `Generate()` call at a new
        duration still pays the full silence-encode cost once. That's expected and correct for a
        pure-function cache, not a limitation to hide.

> from: ## Phase 3 — Small-model decode weakness (SmolLM2/Qwen2.5, 0.125x-0.31x at 135M-500M)
- [x] 3.1 Real `STINGRAY_PROFILE_DECODE=1` decode profile on `SmolLM2-135M-Instruct-Q4_K_M.gguf`
      (real weights, 289 real generated tokens, 45.6 t/s — consistent with the 0.125x-band pattern
      already in `PerformanceLeague.md`): **FFN 59.56% (3741.82ms), QKV projection 19.88%
      (1248.55ms), Output projection 16.34% (1026.69ms), Attention only 3.63%.** Crucially,
      **non-trunk per-token overhead (sampling/stream-decode/dispatch) is only 0.54%** — this
      rules out the simplest hypothesis (fixed per-token overhead swamping tiny real work). The
      cost really is inside the trunk matvecs themselves.
      - **Next concrete hypothesis** (not yet tested): `SimdKernels`'s `MinRowsForParallel = 64`
        gate means FFN's gate/up/down matvecs (rows ~1536/576 at this hidden size) DO parallelize
        across `Parallel.For`, but at hidden=576 the actual per-thread compute (~576×1536÷6≈147K
        MACs) may be small enough that thread-wake/join overhead is a real fraction of the total —
        this is the exact failure mode `docs/done/perf-loop-progress.md` already documents for
        small GEMM shapes on the batched-prefill path, just not yet checked for this
        single-token-decode shape. **Test, don't assume**: compare single-threaded vs current
        parallel FFN matvec timing at this exact (576×1536-class) shape before touching any code.

> from: ## Phase 3 — Small-model decode weakness (SmolLM2/Qwen2.5, 0.125x-0.31x at 135M-500M)
- [x] 3.2 Measured directly using the existing `STINGRAY_CPU_THREADS=1` env var (no code change
      needed to test): 4 total runs (2 parallel default, 2 forced single-threaded), same prompt,
      same 289-token real generation. **Single-threaded was faster in every comparison**:
      49.9 vs 45.6 t/s, then 60.1 vs 58.1 t/s, then 62.0 vs 57.1 t/s (~3-9% faster
      single-threaded, direction consistent across all 4 runs). Confirms the hypothesis: at
      SmolLM2-135M's tiny hidden size (576), `Parallel.For`'s thread-wake/join overhead is a real,
      measurable net loss versus just running the matvec on one thread.
      - **Attempted a code fix 2026-09-12, REVERTED — real, severe negative result.** Added a
        `ShouldParallelize(rows, cols)` helper (`rows >= 64 && rows*cols >= 2_000_000`) and
        sed-replaced all 41 `if (rows >= MinRowsForParallel)` call sites in `SimdKernels.cs` with
        it (verified beforehand that all 41 sites share the identical `rows`/`cols` local-variable
        shape). Solution-wide build succeeded; the one pre-existing `Tests.ForwardPass.Fast`
        failure (`SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec`) was confirmed via
        `git stash` to already fail on unmodified code, unrelated to this change. **But the real
        benchmark was a severe regression, not the intended improvement**: SmolLM2-135M decode
        dropped from the ~45-60 t/s baseline to **15.5-16.0 t/s** (3-4x worse) — far worse than
        either the original parallel code OR the `STINGRAY_CPU_THREADS=1` comparison that
        motivated this fix in the first place. **Reverted immediately** (`git checkout --`) rather
        than debug a shared-kernel change under uncertainty; confirmed the revert restored
        baseline (61.0 t/s). Root cause not fully diagnosed, but the leading hypothesis: a blind
        textual find-replace across 41 call sites assumed uniform `rows`/`cols` *semantics*
        (verified only that the variable *names* matched, not that every function's `rows`
        parameter means "independent output rows" the same way everywhere — some of the 41 sites
        are in `MatVecDual`/`MatVec2In`/`MatVec4In`-style fused/batched functions where `rows`
        may already represent something coarser, and selectively disabling parallelism there while
        leaving other calls (e.g. the ~28M-work vocab/lm_head projection) still parallel every
        decode step may have caused pathological thread-pool interaction, not a simple overhead
        tradeoff. **This needs per-function understanding before any retry, not another blanket
        threshold change** — a real, scoped follow-up, not something to attempt again casually.
      - **2026-10-02 re-measure** (128 decode tokens, single runs; llama-bench tg128 at 8 threads):
        SmolLM2-135M 125 t/s at our default 16 threads / 149 at 8 / 161 at 4 / 72 at 1 vs llama
        281 (0.45x default); SmolLM2-360M 95 / 97 / 81 / 30 vs 127 (0.75x); Qwen2.5-0.5B 75 / 76 /
        66 / 22 vs 97 (0.77x). Single-threaded is now far slower everywhere (3.2's finding is
        stale); **8 threads were never worse than 16** on these three. Decode profile, 135M:
        QKV ~49 us and output projection ~49 us per layer for 576x576-class Q5_0 matrices (most
        of this model is Q5_0, `ffn_down` Q6_K), i.e. per-call dispatch overhead.
      - **Tried and reverted 2026-10-02**: `MatVecQ5_0` rows in dynamically balanced chunks of 8
        (bit-identical). Interleaved x2: 135M mixed (145.7/127.7 -> 142.3/154.9), 360M worse
        (94.1/88.5 -> 74.2/73.6), Qwen2.5-0.5B worse (75.9/74.9 -> 65.2/64.6).
      - **Done 2026-10-02 (fcfb5c57): default kernel threads = physical cores** (new
        `CpuTopology`; was `Environment.ProcessorCount`). Interleaved 16 vs 8 x2, ~600-token
        prompt + 48 gen: SmolLM2-135M decode 151/145 -> 162/154; Qwen2.5-0.5B prefill 145/148 ->
        154/158, decode 71/71 -> 73/74; Granite-Vision-3.2 prefill 42/42 -> 45/46; SmolLM2-1.7B
        prefill 81/82 -> 86/84; Mistral-7B tie. Never slower; outputs unchanged (no arithmetic
        change). Remaining small-model gap (135M ~0.55x, 0.5B ~0.75x vs llama-bench) is per-call
        overhead on tiny matvecs, which per-kernel chunking did not fix (see above).
      - **2026-10-02: spin-then-park worker pool, shipped opt-in (`STINGRAY_CPU_POOL=spin`)**,
        ported from TensorSharp. Decode, interleaved, same session as llama-bench: SmolLM2-135M
        0.52x -> 0.66x, 360M 0.68x -> 0.81x, Qwen2.5-0.5B 0.73x -> 0.87x; Mistral-7B tie. Not the
        default because MoE prefill loses 18-19%. Full table and the variants that lost:
        [2026-10-02-tensorsharp-takeaways-plan.md](../2-coverage/2026-10-02-tensorsharp-takeaways-plan.md) §1.
      - **Follow-up 9e911dba**: the routed-MoE decode and batched-MoE prefill sweeps had their own
        `ParallelOptions` pinned to `Environment.ProcessorCount`, bypassing the new default. Now
        `SimdKernels.CpuThreads`. Decode, interleaved x2: OLMoE-1B-7B 36.4 -> 37.1, LFM2-8B-A1B
        31.8 -> 33.5, Qwen3-Coder-30B-A3B 14.6 -> 15.2 t/s; prefill equal or better.
      - **Lesson for this sweep**: the direct `STINGRAY_CPU_THREADS=1` A/B (3.2's original
        measurement) is still a valid, true finding — thread-wake overhead really is a net loss at
        this specific tiny shape when EVERYTHING is single-threaded together. It does NOT license
        assuming a naive per-call threshold change generalizes safely across a shared kernel file
        with 41+ heterogeneous call sites — verify per-function, not by pattern-matching text.

> from: ## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)
- [x] 2.1 Got a real `STINGRAY_PROFILE_DECODE=1` GDN/Attention/MoE time-split on
      `Qwen3.6-35B-A3B-UD-Q6_K.gguf` (real weights). **Result: MoE/FFN 76.06%, GDN recurrence
      21.11%, Attention 2.82% — the GDN scan is NOT the bottleneck, MoE/FFN is.** Recorded in
      `PerformanceLeague.md` 2026-09-12.

> from: ## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)
- [x] 2.2 Inspected `MoeFfnCore`: already batches 24 per-expert `Parallel.For` sweeps into 2,
      already uses prequantized Q8_K-input int-domain dot kernels. No naive-loop bug found —
      this looks like near-floor cost for a 256-expert top-8 MoE run one token at a time across
      40 layers, not an unexploited kernel gap. **No fix proposed yet; genuinely inconclusive**,
      not force-closed. Two concrete angles left unexplored, for the next iteration:
      (a) per-token TPL `Parallel.For` dispatch overhead at this expert count/shape — measure it
      directly (e.g. wrap with a no-op kernel and diff timings) before assuming it's negligible;
      (b) whether any small-batch (>1 token) decode path could amortize router/dot-kernel setup
      cost, mirroring what `CudaHybridGdnBatchedPrefillTests` already does on the CUDA side.

> from: ## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)
- [x] 2.3a Closed analytically rather than re-running the model: 40 layers × 512 decode tokens =
      20480 `MoeFfnCore` calls total, and the profiler measured 152347.39ms total MoE/FFN time —
      **≈7.44ms per call**. Two `Parallel.For` dispatches per call at a typical 10-100μs TPL
      dispatch overhead each is ≤0.2ms, under 3% of the per-call time even at the pessimistic end,
      and each dispatch covers `numActive(8) × expertDim` work items (thousands), which is exactly
      the regime TPL amortizes well. **TPL dispatch overhead is not the bottleneck** — the 7.44ms
      is real compute (quantized dot products over large expert weight matrices). No live rerun
      needed to reach this; the existing profiler output already contained the answer.

> from: ## Phase 6 — Gemma family (0.12x-0.13x prefill, real "batched-prefill-missing" architectural gap)
- [x] 6.1 Confirmed directly with a fresh real run: `Gemma-4-12B-it-Q4_K_M.gguf`, prefill 3.8 t/s
      vs decode 3.5 t/s — essentially identical (0.9-1.1x depending on run), matching the doc's
      existing 0.12-0.13x-ratio rows exactly. Confirmed in CODE, not just timing: `ForwardPass.cs`
      computes `perLayerHdUnsupported = _layerHeadDim is not null` and when true, "prefill" is
      literally `for (i in N) Forward(tokens[i], startPos+i)` — the decode path called in a loop.

> from: ## Phase 6 — Gemma family (0.12x-0.13x prefill, real "batched-prefill-missing" architectural gap)
- [x] 6.2 **This is already fully investigated ground from a prior session** —
      `docs/done/gemma4-12b-evidence.md` (2026-08-07) measured the identical 0.9x prefill:decode
      ratio, confirmed the same root cause in code, and went further: `PrefillCore` (the batched
      path) already has SOME per-layer-head-dim plumbing (per-layer qDim/kvDim, RoPE/Q-K-norm via
      `ApplyRopeLayer`), but Gemma4 needs real features `PrefillCore` doesn't implement AT ALL —
      per-layer KV head count (MQA/GQA mix), KV-layer sharing (`_layerKvSrc`), `attention_k_eq_v`,
      a per-head V norm before the cache write, and critically **sliding-window attention**
      (`PrefillCoreAttention` has no `windowSize` parameter at all). A prior attempt to force the
      batched path anyway (`STINGRAY_PER_LAYER_HD_PREFILL=1`) didn't just produce wrong output —
      it produced a real `AccessViolationException` (KV indexed at the model-wide head dim on
      layers that actually carry a smaller one, walking off the buffer). That flag now fails fast
      with an explanation instead of corrupting memory. **"A path that corrupts memory cannot be
      timed" — the upside of fixing this was never even quantifiable by the cheap route.**

---

## More completed items moved from perf-sweep-plan.md on 2026-11-01

Finished items moved verbatim out of [../4-performance/perf-sweep-plan.md](../4-performance/perf-sweep-plan.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Phase 10 — Vision-Language Model decode weakness (~0.48x pattern, worst point 0.41x prefill)
- [x] 10.1 *(closed 2026-11-01: answered by the 2026-10-01/02 re-measure below and by 10.2 and 10.3)* These are all **text-only** measurements (no `--image`/`--mmproj` — see the section's
      own caveat) — i.e. this is the LLM backbone underneath a VLM, not vision-encoding cost. So
      the ~0.48x decode pattern converging across 3 different backbones is suspicious: check
      whether something in how these checkpoints are loaded/dispatched (e.g. a shared
      VLM-specific code path even in text-only mode) differs from a "pure" LLM of the same
      backbone size, rather than assuming it's per-backbone-architecture coincidence.
      **2026-10-01/02 re-measure:** the shared ~0.48x was mostly the Q4_K decode scheduling issue
      fixed in Phase 7; InternVL3-2B and Granite-4.0-3B-Vision now decode at ~0.75-0.79x.
      Granite-Vision-3.2-2B stayed at ~0.47x because every projection is Q3_K (see 10.2).

> from: ## Phase 10 — Vision-Language Model decode weakness (~0.48x pattern, worst point 0.41x prefill)
- [x] 10.4 *(closed 2026-11-01: done through 10.2 and 10.3; the benchmarks are in the League, the Q3_K follow-ups are tracked under 10.2)* Implement + verify (real image-understanding sanity check if touching anything
      vision-adjacent, not just text-only re-runs) + re-benchmark + record.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
- [x] 1.2 *(closed 2026-11-01: done and recorded in the League 2026-09-12, 1.51x and 0.221x; later work moved this pipeline to about 0.81x on 2026-10-02)* Quantize Voxtral weights to Q8_0 (currently raw F32 — 4x the memory traffic of every
      other model in this engine per matvec). Likely the single biggest remaining lever. Needs a
      perplexity/transcript-quality check before shipping (CLAUDE.md rule 7).
      - [x] 1.2e *(closed 2026-11-01: benchmarked 164.2 s to 108.9 s and recorded 2026-09-12)* Repeat for `VoxtralAudioEncoderWeights`: Q/K/V/O/gate/up/down/Projector1/
            Projector2 quantized to Q8_0 (biases and conv weights stay F32 — conv path uses a
            different mechanism, out of scope this pass); `VoxtralAudioEncoder.LinearNoBias`/
            `LinearBias` switched to `SimdKernels.MatVecQ8_0` (bias now added as a separate pass
            after the quantized dot, since `MatVecQ8_0` has no fused-bias overload). Also fixed
            `VoxtralAudioEncoderWeightsLoadTests`' now-wrong `float.IsFinite`/exact-length
            assertions on the newly-`byte[]` fields. Builds clean. A real-audio correctness
            re-check PASSED: 5/5 tests, transcript still exact match. **Benchmarked: 164.163s →
            108.861s (1.51x), ratio 0.147x → 0.221x. Cumulative from original baseline: 1203.1s →
            108.861s = 11.05x, ratio 0.02x → 0.221x.** Recorded in `PerformanceLeague.md`
            2026-09-12. Diminishing returns visible (5.6x → 1.3x → 1.51x per Q8_0 pass) but still
            3.5x off the C++ reference — not fully plateaued.

> from: ## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)
- [x] 1.4 Wire a real CLI/pipeline path using `PrefillWithCache`/`Step` — currently only test
      harnesses reach the KV-cache incremental path at all; without this, kernel speed is moot for
      real usage. **Concrete lead found**: this codebase has an established `ISpeechToTextPipeline`
      interface (`src/OpenTail.Stingray.Audio/ISpeechToTextPipeline.cs`) with working
      implementations for every other ASR model (`ParakeetPipeline`, `WhisperPipeline`,
      `QwenAsrPipeline`, `FunAsrPipeline`, `ParaformerOnnxPipeline`) — a `VoxtralPipeline` class
      following the same shape (mel-extract → prefill → greedy `Step` loop, mirroring what
      `VoxtralGenerationLoopTests`/`VoxtralPerfBaselineDebugTest` already do by hand) is the
      concrete next step, not a research question. None of the existing pipelines appear wired
      into `src/OpenTail.Stingray.Cli` either (grepped, no hits) — check `Server.Host` or a
      not-yet-found CLI ASR command before assuming a new dispatch point is needed from scratch.
      - [x] *(closed 2026-11-01: `stingray transcribe` exists, `TranscribeCommand` and `SttCommand` wire the `ISpeechToTextPipeline` implementations, and `VoxtralPipeline` is referenced from `SttCommand`)* Was: no CLI/Server.Host dispatch point found yet wiring ANY of this project's
            `ISpeechToTextPipeline` implementations (not just Voxtral) into a runnable command —
            worth a real look at `src/OpenTail.Stingray.Server*` before assuming one needs to be
            built from scratch, but out of scope for finishing 1.4's Voxtral-specific goal.

> from: ## Phase 8 — DeepSeek-V2-Lite-Chat (`deepseek2`, prefill 0.49x)
- [x] 8.2 *(closed 2026-11-01: moot, `deepseek2` was admitted 2026-09-26 and no `--allow-unverified-arch` is needed)* This architecture requires `--allow-unverified-arch` — confirm the perf harness/CLI
      invocation still works the same way for a profiled run (`STINGRAY_PROFILE_DECODE=1` plus
      the flag) before assuming standard tooling applies unmodified.

> from: ## Phase 9 — Worst-RTF TTS/audio-generation pipelines (114.14x — worse than Voxtral's original 85.5x)
- [x] 9.2 *(closed 2026-11-01: golden-parity tests now exist, `AceStepDiTGoldenParityTests`, `AceStepLyricEncoderGoldenParityTests` and `AceStepOobleckDecoderGoldenParityTests`)* No numeric golden reference exists yet for this port (per the doc's own caveat,
      "non-degeneracy checked only") — any perf change here needs at minimum a real audio-energy/
      non-degeneracy re-check before/after, since byte-identical or transcript-based verification
      (Voxtral's approach) isn't available for this model class.

> from: ## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)
- [x] 2.4 *(closed 2026-11-01: the 9B and 27B size points were measured and fixed 2026-10-02, see the Phase 2 notes above and ADR-0002)* Repeat 2.1-2.3 for the 9B (Ornith-1.0-9B) and 27B (Qwen3.6-27B Q3_K_XL) size points if
      a 35B fix is found and generalizes; record each in `PerformanceLeague.md`.

> from: ## Phase 6 — Gemma family (0.12x-0.13x prefill, real "batched-prefill-missing" architectural gap)
- [x] 6.3 *(closed 2026-11-01: obsolete. `ForwardPass.PrefillCore` handles per-layer KV heads and a per-layer window, checked in code; the section's own 2026-10-02 note says the rest landed 2026-09-16. `PrefillWithCache` still refuses per-layer head dimensions, which is a separate limit.)* **Real feature work, not attempted this pass** (same category as Phase 4/13's GPU
      dispatch fusion and Phase 14's EXAONE post-norm gap) — implementing per-layer KV heads,
      KV-layer sharing, `attention_k_eq_v`, per-head V norm, and a real `windowSize` parameter for
      `PrefillCoreAttention` is a multi-feature engineering project, not a quick fix or a flag
      flip. `docs/done/gemma4-12b-evidence.md`'s own conclusion stands: scope this properly (check
      `docs/done/cpu-prefill-repack-gemm-plan.md` for reusable batched-prefill design) before attempting
      it, and do NOT try to force the existing path again — that route is confirmed memory-unsafe,
      not just slow-to-verify. **Phase 6 is correctly closed at "well-understood, real,
      substantial follow-on work" for this pass — matches the category, doesn't need re-deriving.**
