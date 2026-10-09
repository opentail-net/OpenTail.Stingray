# Scratch arena: a measure-first experiment (plan)

**Status:** Proposed (2026-10-09). Origin: a suggestion to add "zero-allocation tensor arenas" (a pooled native slab with scoped borrowing) to cut GC pressure and memory.
**Rule that governs it:** CLAUDE.md rule 7: a performance change is kept only if it is measurably better on real weights with enough samples; otherwise it is reverted.
**Related:** `PerformanceLeague.md` (~line 1306, the earlier zero-allocation measurement), `docs/done/perf-investigation-brief.md` item 4, `todo.md`.

## 1. What is already known (so we do not re-litigate it)

| Fact | Source |
|---|---|
| GC is 0.4-0.9% of decode time (SmolLM2-135M at 169 t/s: 0.9%; Qwen3.8-27B with MTP: 0.4%). Managed allocation is about 1-2.7 MiB per token. | `PerformanceLeague.md`, measured with `STINGRAY_GC_STATS=1` |
| GC-committed memory is 2% of peak on a 27B (294 of 13,503 MiB), 4% on SmolLM2-1.7B, 28% on SmolLM2-135M (95 of 345 MiB). **That is the most an arena could ever save.** | same |
| The forward passes already keep scratch in pre-allocated `NativeMemory` (284 `NativeMemory.Alloc*` sites in 51 files, 225 in the Engine), nearly all allocated once at construction. KV pages are native and lazy; layers without attention allocate none (`c4cea419`). | code, this session |
| What still reaches the managed heap per token: the logits `.ToArray()` copies (`ContinuousBatchingEngine`, `CudaForwardPass`) and `new float[VocabSize]` in the CUDA paths, about 1 MiB each. | code, this session |
| **Two kinds of per-call native allocation are NOT pooled and are unmeasured:** (a) activation scratch inside the CPU matmul kernels, "roughly 120 alloc/free pairs of 1-10 MB per prefill chunk"; (b) `ForwardPass.PrefillCore` allocates 33 large batch buffers per prefill call (`AllocZeroed` of `N x dim`) and `HybridGdnForwardPass` does the same per chunk. The brief said: "Measure before pooling." | `perf-investigation-brief.md` item 4, `ForwardPass.PrefillCore.cs` |

The pasted premise ("temporary tensors, KV caches and logits hit the managed heap") is therefore mostly false for this engine. The plausible payoff is narrower: **fresh large native allocations each prefill call cost page faults and allocator work that a reused slab would not.** That is the only thing this plan tests.

## 2. Goal and non-goals

**Goal:** find out, with numbers, whether reusing scratch memory across prefill calls makes prefill measurably faster, and adopt it only where it does.

**Non-goals:** retrofitting the 284 construction-time allocation sites (no measurable gain, high risk); any claim of a memory saving beyond the GC-committed bound above; changing numerics (results must be bit-identical); touching GPU backends (their pools exist and this machine cannot measure them: rule 13).

## 3. Phases, with time

Times are working time for one person at the pace of this project's recent work, and include verification. Elapsed time is longer because heavy timing runs need an idle machine and one process at a time.

| Phase | Work | Time | Decision at the end |
|---|---|---|---|
| **P0 baseline** | Count and size the per-call native allocations during a real prefill (instrument with counters behind an env switch, no behaviour change). Time prefill on SmolLM2-1.7B (707 tokens, the existing methodology: interleaved pairs, idle machine) and Mistral-7B. Separately measure page-fault and allocator cost, for example by timing the same prefill with a hack that reuses one set of buffers. | **2-4 hours** | **Go / no-go.** If the reuse hack is within noise (under about 2%), stop here: record the result in `PerformanceLeague.md`, close the item. Cost of stopping: half a day. |
| **P1 `NativeArena`** | One new file in Core or Cpu: aligned slab (`NativeMemory.AlignedAlloc`), bump allocation with scoped mark/release, typed pointer and `Span` views, `Dispose`, double-release and use-after-release checks in debug, thread-local variant for kernel scratch. About 15 unit tests (alignment, reset, overflow growth policy, scope nesting, thread isolation). Touches nothing existing. | **2-3 hours** | none (it is inert until adopted) |
| **P2 adopt in the kernels** | Replace the per-matmul activation scratch `Alloc`/`Free` in the CPU matmul kernels with a thread-local arena scope. Re-run `BatchInvariantGemmTests`, the kernel tests and `KernelBench` to confirm bit-identical outputs. | **3-5 hours** | Keep only if the A/B shows a gain. |
| **P3 adopt in prefill** | Move `ForwardPass.PrefillCore` and `HybridGdnForwardPass` per-call batch buffers onto a per-context arena that persists across calls (grows to the largest chunk seen, never shrinks mid-run). Re-run the golden/parity tests (`ForwardPass.Fast`, the heavy ones with real weights once). | **3-5 hours** | Keep only if the A/B shows a gain. |
| **P4 A/B and decision** | Interleaved A/B, at least 6 pairs per model, idle machine, recording tokens/s, peak working set and GC stats for baseline vs arena, on SmolLM2-1.7B, Mistral-7B and a hybrid (Qwen3.5-4B). Rule 7 decides: keep what is measurably better, revert the rest, write the numbers in `PerformanceLeague.md`. | **2-3 hours** (mostly waiting for runs) | Keep / revert per adoption site. |
| **P5 optional logits copies** | Replace the ~1 MiB/token `.ToArray()` logits copies with pooled buffers. Only if P0 shows managed allocation matters, which the earlier measurement says it does not. | **2-3 hours** | likely skipped |

**Totals**
- **If P0 says no:** about **half a day**, and the question is answered with a recorded number.
- **If it pays off:** about **1.5-2 working days** end to end (P0 to P4), spread over roughly **3 calendar days** because the timing runs must be done one at a time on an idle machine.
- P5 adds about half a day and is expected to be dropped.

What dominates the time is **measurement, not code**: the arena type is a few hours; proving a 2-5% effect over machine noise is the long part.

## 4. Acceptance criteria

1. **Numerics unchanged:** outputs bit-identical to the baseline on the kernel tests, `ForwardPass.Fast`, and the golden parity tests on real weights (timing checked per CLAUDE.md rule 12, not just pass/fail).
2. **Measured, not argued:** each adopted site has an interleaved A/B (at least 6 pairs) showing a gain larger than the run-to-run spread, recorded with its date and machine.
3. **No memory regression:** peak working set is not higher with the arena than without; the memory estimator's calibration table (`docs/reference/061-coverage-tooling.md`) is re-run if it moves, since the estimator is an upper bound calibrated on current behaviour.
4. **Safe by construction:** the arena detects use-after-release and double-release in debug builds, and a scope can never outlive its owner (ref-struct views).
5. **Honest about scope:** the result states that this is a CPU-prefill finding from an iGPU-only machine; nothing is claimed about GPU backends (rule 13).

## 5. Risks

- **Lifetime bugs do not fail to compile.** A buffer reused while still read yields wrong numbers intermittently. Mitigation: debug-mode poisoning on release, scoped ref-struct views, bit-identical tests, and adopting site by site.
- **Thread-local arenas and parallel kernels.** Scratch used by worker threads must not alias; each worker gets its own scope. Mitigation: a dedicated test running the kernels at several thread counts.
- **A larger resident footprint.** A slab that grows to the largest chunk and never shrinks keeps that memory for the process lifetime. Mitigation: cap growth to the chunk size actually used, report it in the memory estimator, and compare peak working set in P4.
- **Noise.** A 2% effect is invisible without interleaved pairs on an idle machine (memory note: run heavy jobs one at a time; timing under contention is not evidence).
- **It may simply not help.** That is an acceptable, cheap outcome (about half a day), and the point of putting P0 first.

## 6. Decision needed from the user

None to start. P0 changes no behaviour and is what decides whether the rest happens. Suggested slot: after the first-run work (`2026-10-09-known-good-checkpoints-and-first-run-plan.md`, P0), unless you want it first.
