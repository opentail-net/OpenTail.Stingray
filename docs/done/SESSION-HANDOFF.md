# Session handoff — CPU perf optimization loop

**If you're a fresh session reading this: welcome. This file tells you how to resume. The actual
technical investigation log — findings, numbers, what's been tried — lives in
`docs/perf-loop-progress.md`. Read that file FIRST for content; read this file for mechanics
(what's running, what's queued, how the loop is wired).**

## The standing mandate (verbatim, unchanged since it was set)

The user asked for autonomous CPU-inference performance optimization work on
`C:\Git\OpenTail\extensions\OpenTail.Stingray` to continue every hour until end of day 2026-07-25 (or
until every reasonable optimization avenue is genuinely exhausted). A recurring cron job
(`CronCreate`, cron expression `7 * * * *`, fires hourly at :07) re-injects this exact prompt each
time:

> 1h Continue autonomous CPU-inference performance optimization work on
> C:\Git\OpenTail\extensions\OpenTail.Stingray (from-scratch managed CPU LLM inference engine,
> benchmarked against llama.cpp) until end of day 2026-07-25. FIRST ACTION EACH FIRING: read
> C:\Git\OpenTail\extensions\OpenTail.Stingray\docs\perf-loop-progress.md (create it if missing) to see
> what's already been tried and what's next -- this is the source of truth across firings, not
> conversation memory. [... full text is the cron job's stored prompt; if the cron job itself is
> gone, re-read `docs/perf-loop-progress.md`'s own header, which restates the mandate and
> discipline in full, and re-create the cron with `CronCreate` if the user wants it resumed.]

**Discipline (non-negotiable, carried through every iteration so far):**
- Never extend/modify a kernel's arithmetic without a seam test verified against a
  HAND-COMPUTED reference (not against other kernel code in this repo).
- Real benchmarks only: ≥10 warmup calls (this session's own experience says often more — see
  iteration 9 below), run at least twice.
- **n=6 minimum samples per side for any A/B verdict** — established in iteration 5 after an
  n=3 sample gave a false-positive "12.7% win" that didn't replicate at n=6. Do not regress to
  fewer samples for speed.
- Keep `Tests.ForwardPass` green. Known pre-existing baseline: 16 `VulkanShaderTests`/
  `VulkanInitTests` failures (no Vulkan device on this box) + occasionally
  `ContinuousBatchingConstraintTests.ConstrainedAndUnconstrained_Coexist_PerSequenceMasking`
  (documented flaky, confirmed unrelated to CPU/GEMM work, passes in isolation). Any OTHER
  failure is a real regression — investigate by name, don't wave it through.
- No Codex review available this session — self-review against this discipline instead.
- **Do not commit anything to git.** Record work in `docs/perf-loop-progress.md` only. The user
  will review and commit when they're back.
- Don't ask the user questions — they're AFK. Make the reasonable call and proceed.
- Stop condition: call `ScheduleWakeup` with `stop:true` only when every reasonable avenue is
  exhausted, logging the final honest verdict first.

## How to resume RIGHT NOW (as of the point this handoff was written)

1. Read `docs/perf-loop-progress.md` in full — it's long but it's the whole story, in order,
   with real numbers for every iteration (1 through 9 so far).
2. **Check for background processes that may still be running or may have completed.** At the
   moment this handoff was written, two were in flight:
   - A `PrefillWarmupBenchmark` run (`benchmarks/OpenTail.Stingray.Bench/PrefillWarmupBenchmark.cs`,
     new this session) — `dotnet run --project benchmarks/OpenTail.Stingray.Bench -c Release --no-build
     -- --filter "*PrefillWarmupBenchmark*"`, testing prefill at TokenCount 64/256/903 with 20
     warmup iterations and 6 measured iterations each (a genuinely long run — expect many
     minutes, especially at 903 tokens). This is answering perf-loop-progress.md's own
     highest-priority open question: what is the REAL (properly warmed-up) prefill gap to
     llama.cpp, since the naive cold-start number (~7.7x) was shown to be unreliable.
   - An `llama-bench.exe -ngl 0 -dev none` confirmation run (answering the user's own question:
     "is llama.cpp secretly using the Vega iGPU?" — already answered definitively NO via
     `--list-devices` showing "(none)" and the DLL set containing no Vulkan/CUDA backend, this
     was just a belt-and-suspenders explicit-flag re-confirmation).
   If either process's background-task notification hasn't arrived yet in a fresh session (it
   won't carry over sessions), just re-run the commands above directly — they're idempotent,
   nothing destructive, safe to re-run.
3. Once you have the prefill benchmark's real numbers, **write them into
   `docs/perf-loop-progress.md` as "Iteration 9"** (or the next unused iteration number — check
   the file's own most recent iteration number, don't renumber existing ones), following the
   exact same format every prior iteration used: what was tried, verified how, honest result
   (win/loss/inconclusive), what's next.
4. Then continue down `docs/perf-loop-progress.md`'s own "Next up" list, one item per hourly
   firing, exactly as every iteration so far has done.

## Key facts already established (don't re-derive these — read `perf-loop-progress.md` for the
full evidence, this is just an index so you know they're already answered)

- This box: AMD Ryzen 7 5700G, but only **12 logical / 6 physical cores** available right now (a
  stock 5700G has 16/8 — confirmed via BenchmarkDotNet's own system banner, not a guess). Core
  count has been observed to fluctuate 12↔16 across different sessions on this shared/virtualized
  box.
- Raw sequential memory-read bandwidth, measured directly: **36.77 GB/s** (stdev 0.80 — the
  tightest, most reproducible measurement of the whole investigation). ~72% of DDR4-3200's
  theoretical 51.2 GB/s peak — normal, not evidence of severe VM throttling.
- `tools/llama.cpp` is confirmed genuinely CPU-only (`VERSION: b8585-cpu`, no GPU DLL present,
  `--list-devices` reports zero devices).
- **Decode is near-parity with llama.cpp**: fresh, verified, matched-thread-count benchmark —
  llama.cpp 29.71 t/s vs ours 26.48 t/s, a ~1.12x gap. NOT the 4x this investigation originally
  assumed. Every decode-focused lever tried (dispatch mechanism, thread count, prefetch, FMA
  accumulator widening) came back negative/no-effect — consistent with there not being much
  headroom left there.
- **Prefill's real gap is still being established** (the in-flight background benchmark above is
  answering this). The naive cold-start CLI number suggested ~7.7x, but that's known to be
  contaminated by JIT warmup effects this codebase's own docs already documented
  (`cpu-prefill-repack-gemm-plan.md` §29: `MatMulBatched` needs ~9 calls to reach steady state).
  **Do not treat 7.7x as real until the warmed-up number lands.**
- Three independent sources (this session's own bandwidth math, an external ChatGPT review, an
  external Gemini review — both transcribed in this session, Gemini's in `_gemeni.txt`) all
  independently predicted the "4x" premise was overstated, before the real llama-bench number
  confirmed it for decode. Worth trusting convergent external review input in this investigation.

## Files changed this session (uncommitted — for the user to review, not to commit yourself)

- `src/OpenTail.Stingray.Engine/DecodeProfileTimers.cs` (new) — opt-in decode trunk profiler,
  `STINGRAY_PROFILE_DECODE=1`.
- `src/OpenTail.Stingray.Engine/PrefillProfileTimers.cs` (new) — opt-in prefill trunk profiler,
  `STINGRAY_PROFILE_PREFILL=1`.
- `src/OpenTail.Stingray.Engine/ForwardPass.cs` (modified) — wired both profilers into `RunTrunk`
  (decode) and `PrefillCore` (prefill).
- `src/OpenTail.Stingray.Cli/RunCommand.cs` (modified) — prints profiler reports; added
  non-trunk-overhead timing to `DecodeLoop`.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs` (modified) — added `DotQ4K_Wide8` (8-accumulator variant,
  tested, genuine loss/no-effect, kept for the record) and the `STINGRAY_MATVEC_WIDE8=1`
  runtime toggle (default off, zero cost when unset).
- `tests/OpenTail.Stingray.Tests.ForwardPass/DecodeMatVecDispatchPerfTests.cs` (new) — Parallel.For vs
  PersistentThreadPool dispatch comparison at decode shapes.
- `tests/OpenTail.Stingray.Tests.ForwardPass/DotQ4KWide8SeamTests.cs` (new) — correctness + perf tests
  for the widened kernel.
- `tests/OpenTail.Stingray.Tests.ForwardPass/RawMemoryBandwidthPerfTests.cs` (new) — the 36.77 GB/s
  measurement.
- `benchmarks/OpenTail.Stingray.Bench/PrefillWarmupBenchmark.cs` (new) — the properly-warmed prefill
  benchmark currently running in the background.
- `docs/perf-loop-progress.md` (new) — the full investigation log. **This is the important one.**
- `_gemeni.txt` (new, untracked) — external review transcript, already triaged into the progress
  log.

## If the cron job is gone and the user wants the loop resumed

Use `CronCreate` with `cron: "7 * * * *"`, `recurring: true`, and the exact prompt text quoted at
the top of this file (or ask the user if they want it re-scoped now that decode is known to be
near-parity — that's a judgment call for whoever resumes this, not something to assume).
