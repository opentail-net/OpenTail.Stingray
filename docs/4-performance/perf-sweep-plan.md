# PerformanceLeague sweep — phased plan

> Completed items moved to [done/perf-sweep-plan-completed-items-2026-10.md](../done/perf-sweep-plan-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

> **STATUS 2026-10-02.** Horizontal Pass A, Phase 5, and Phase 16 are also closed and moved verbatim to
> [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md).
> **STATUS 2026-10-01.** Horizontal Passes B and C and Phases 7 and 15 are also closed and moved to the same file. **STATUS 2026-09-27.** Phases 12 and 14 are closed and moved verbatim to
> [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md). Also since the phases below were written: Phase 8's item 8.2 is
> moot (`deepseek2` admitted 2026-09-26, no `--allow-unverified-arch` needed; its perf items are open);
> Phase 2's 35B point moved from 0.05x to 0.63x of llama.cpp prefill (2026-09-25,
> [done/2026-09-25-hf-top-downloads-coverage-plan.md](../done/2026-09-25-hf-top-downloads-coverage-plan.md)
> Phase 8); Phase 3's SmolLM2 prefill is now ~0.71-0.81x (f801243, see
> [101-work-queue-after-coverage-plan.md](../done/101-work-queue-after-coverage-plan.md)). The other phases were
> not re-verified item by item in this pass.

**Read this file first on every loop firing.** Source of truth across firings (a `/loop`
session may not carry full context forward). Check a box when a numbered item is measured,
committed to the checkbox list here, and recorded back into `PerformanceLeague.md` — not before.
Never mark a box done from a plausible-sounding fix alone; only from a real re-benchmark with
real weights (CLAUDE.md rule 7: measure, don't assume).

Ranking source: `PerformanceLeague.md`, picking the worst-ratio row not yet perf-swept.

**Strategy note (2026-09-12, per explicit user direction):** once a bug CLASS is confirmed real in
one model (not a one-off), scan the whole codebase for the same class and fix it everywhere at
once, verify each affected pipeline mechanically, THEN move to the next class — horizontal passes
across models, not vertical model-by-model sweeps repeating the same lesson N times. Phases below
remain the model-specific record, but see "Horizontal Passes" for the cross-cutting work this
produced.

---

## Order of the phases below (re-sorted 2026-10-01: quickest to resolve first, longest tail last)

Phase numbers are historical and are not renumbered, so the sections are no longer in numeric order.
Quick and well-understood work comes first; uncertain, architectural or hardware-dependent work comes
last. Closed sections are one-line pointers into [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md).

1. Phase 11: Citrinet-ASR, per-stage profile of a small, fast pipeline (~0.6x warm; pointwise attempt reverted).
2. Phase 10: VLM text decode, profile first (10.2), then the Vulkan regression (10.3).
3. Phase 1: Voxtral Q8_0 weights (a known, large lever with a plateau declared).
4. Phase 8: DeepSeek-V2-Lite prefill (0.49x), profile first.
5. Phase 9: worst-RTF TTS/audio-generation pipelines (large, no numeric golden for some).
6. Phase 3: small-model decode (a blanket threading fix already failed and was reverted; needs per-function work).
7. Phase 2: Qwen3.6 hybrid-GDN decode (plateau declared; remaining angle is architectural).
8. Phase 6: Gemma batched prefill (a multi-feature project).
9. Phases 4 and 13: iGPU dispatch overhead (uncertain; the premise needs a discrete GPU to re-test).

---

## CROSS-CUTTING ANALYSIS (read this first, every firing) — the 7 root causes behind almost every weak number in this doc

Synthesized 2026-09-12 across everything surveyed this sweep (Voxtral, Qwen3.6 hybrid-GDN,
SmolLM2/Qwen2.5 small models, ACE-Step, Gemma, Llama, DeepSeek, EXAONE, VLMs, SDXL/FLUX, speculative
decoding). Almost every weak number in `PerformanceLeague.md` traces back to one of these 7 causes,
not to N unrelated one-off bugs — **this is the priority order for horizontal passes**, ranked by
how mechanically batch-fixable each one is (most batch-fixable first):

1. **Naive scalar matvec never wired to the engine's own SIMD kernels.** Voxtral's original bug,
   confirmed as a real, repeated pattern via **Horizontal Pass A** — 15 more files found and fixed
   in one mechanical sweep (see below). Most batch-fixable of all 7: pattern-matchable, one-line
   delegation per file, zero call-site risk.
2. **Weights left at full F32 precision when the engine already has fast quantized (Q8_0/Q4_K)
   kernels.** Voxtral's second win (two passes, 1.3x + 1.51x). Batch-fixable the same way as #1,
   just not yet swept: check every "new coverage"/recently-ported pipeline still loading weights
   via `ReadF32` with no quantization step, before assuming its slowness is fundamental to the
   model rather than an unwired kernel gap. **Not yet swept horizontally — do this next.**
3. **Deterministic, cacheable work recomputed on every call.** ACE-Step's silence-latent (83-85%
   of total time, a pure function of duration alone) is the clean example — and its own fix had a
   real bug (checked the cache, never wrote to it) that would have shipped a false "no measurable
   improvement" conclusion had it not been caught by re-profiling instead of trusting the first
   overall-time number. Batch-fixable in principle: sweep every pipeline with a "fixed
   conditioning"/"reference" input path (TTS voice-cloning reference encoders, VAE silence/zero
   priors, any text encoder called with a static system prompt) for the same
   recompute-what-never-changes pattern. **Not yet swept horizontally — nobody has checked this
   elsewhere yet.**
4. **Per-dispatch GPU overhead dominating at small problem sizes.** Nearly every sub-1B model
   loses on Vulkan prefill (dispatch-bound, not compute-bound), and both SDXL-Turbo
   attention-residency attempts failed for the identical reason (many small per-call round-trips
   beat a few big ones on this specific shared-memory iGPU). **NOT mechanically batch-fixable
   the same way as #1-3** — this needs a real, reusable "batch N calls into 1 dispatch" primitive
   that doesn't exist yet in this codebase, not a pattern-match-and-replace. The lesson has been
   learned twice independently (naive shader, then tiled shader, both reverted) instead of being
   captured once as reusable infrastructure — that gap, not either individual shader, is the real
   finding.
5. **Threading gated on a heuristic too coarse for the shape space.** `MinRowsForParallel=64` is
   row-count-only. **Already tried a batch fix here and it backfired badly** (rows×cols threshold
   across all 41 `MatVec*` call sites caused a 3-4x regression, reverted) — proof that this
   specific category is NOT safe to fix with a single global formula the way #1-3 are. The real
   shape-to-thread-benefit relationship is more complex than either heuristic captures and needs
   per-function analysis, not another blanket change.
6. **Whole architecture families blocked from GPU offload by narrow tensor-name assumptions.**
   EXAONE-4.5's post-norm layout crashes `HybridForwardPass` outright (hardcoded pre-norm tensor
   names, no fallback), forcing CPU-only for an entire model family regardless of how fast GPU
   offload might otherwise be. A coverage gap, not a speed gap (Phase 14) — real, scoped, one-time
   engineering work, not a repeatable pattern to scan for elsewhere.
7. **Hardware ceiling, already correctly diagnosed, not fixable in software here.** Speculative
   decoding (-37% to -75%) needs VNNI (Zen 4+); this CPU is Zen 3. The one category where "make it
   great" isn't a code question — Phase 15 exists only to disambiguate which sub-cause explains
   the two data points, not to find a fix.

**What this means for sequencing**: #1 is done (Horizontal Pass A). #2 and #3 are the next
horizontal passes to run — both are proven-safe, mechanical, and unswept elsewhere. #4 and #6 need
real (larger, scoped) engineering, not a pattern-match sweep. #5 is a closed cautionary tale, not
a re-attempt candidate without per-function analysis. #7 is a disambiguation task, not a fix.

---

## Horizontal Passes A, B and C — CLOSED; moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

## Phase 15 — DSpark speculative decoding — CLOSED; moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

## Phase 5 — Sweep remainder — CLOSED 2026-10-01; moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

---

## Phase 11 — Citrinet-ASR (~0.20-0.25x, worse than Whisper Tiny's Phase 5.1 target)

`Citrinet-ASR (Jasper-style conv encoder)`: `0.872s (single run) | 0.171s (mean of 3) | ~0.20x`
against a real, byte-identical-transcript-verified C++ reference (`examples/audio.cpp`'s
`citrinet_asr` family) — i.e. correctness is already solid here, this is a pure throughput gap on
a real, working ASR pipeline, worse than the Whisper Tiny row already queued in Phase 5.1.

- [ ] 11.3 (2026-10-01 attempt reverted) A register-tiled pointwise kernel (4 out-ch x 16 frames, bit-identical:
      same logits hash) was NOT faster warm (52-56 vs 48.6-49.2 ms encoder), only cold (232-271 vs 572-670 ms
      first call, i.e. JIT). Next angle if revisited: remove per-conv allocation and per-call `Parallel.For` (one
      parallel region per block, preallocated ping-pong buffers), then re-measure. Original text: implement +
      re-verify the exact-transcript-match correctness this pipeline already has
      (do not regress it) + re-benchmark (3+ runs, this pipeline is fast enough to afford more
      samples than the slow TTS/diffusion pipelines elsewhere in this doc) + record.

## Phase 16 — RWKV6/RWKV7 prefill — CLOSED 2026-10-02 (0.21x -> ~0.45x prefill, decode 0.94-0.95x; rest is Q8_0 GEMM); moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

## Phase 10 — Vision-Language Model decode weakness (~0.48x pattern, worst point 0.41x prefill)

Real multimodal capability, not a niche architecture: `InternVL3-2B` (0.78x prefill / 0.48x
decode), `Granite-4.0-3B-Vision` (0.84x prefill / 0.48x decode), `dots.ocr` (0.71x prefill /
0.79x decode), and worst of all `Granite-Vision-3.2-2B Q3_K_S` (**0.41x prefill** / 0.48x decode)
— three different backbones (Qwen2-1.5B, Granite, Granite) converging on roughly the same ~0.48x
decode ratio is a real pattern, not noise on one checkpoint. Separately, `Granite-Vision-3.2-2B`'s
Vulkan decode (4.1 t/s) is dramatically worse than its own CPU decode (17.1 t/s) — flagged in the
doc as "worth another look" and not yet investigated.

- [ ] 10.1 These are all **text-only** measurements (no `--image`/`--mmproj` — see the section's
      own caveat) — i.e. this is the LLM backbone underneath a VLM, not vision-encoding cost. So
      the ~0.48x decode pattern converging across 3 different backbones is suspicious: check
      whether something in how these checkpoints are loaded/dispatched (e.g. a shared
      VLM-specific code path even in text-only mode) differs from a "pure" LLM of the same
      backbone size, rather than assuming it's per-backbone-architecture coincidence.
      **2026-10-01/02 re-measure:** the shared ~0.48x was mostly the Q4_K decode scheduling issue
      fixed in Phase 7; InternVL3-2B and Granite-4.0-3B-Vision now decode at ~0.75-0.79x.
      Granite-Vision-3.2-2B stayed at ~0.47x because every projection is Q3_K (see 10.2).
- [x] 10.2 `Granite-Vision-3.2-2B Q3_K_S`: **DONE 2026-10-02.** Q3_K matvec measured at only
      7-21 GB/s on the real weights (scratch harness, all 40 layers' tensors per kind), against
      30+ GB/s for Q4_K. Not scheduling (Q3_K already balances per row); it was the kernel:
      `DotQ3K_Q8K_Avx2` unpacked scales scalar-wise into bounds-checked spans and picked its
      shifts through `switch`es inside loops the JIT does not unroll. Rewritten in the shape of
      ggml's `ggml_vec_dot_q3_K_q8_K` AVX2 path (SIMD scale unpack, pshufb scale broadcast, high
      bit as `ql·q8 - 4·(1-h)·q8`, unrolled constant shifts).
      - Matvec (tiered JIT off, 3 interleaved rounds): attn 6.6-7.1 -> 4.4-4.7 ms, ffn_up 22-27
        -> 15 ms, ffn_down 24-26 -> 18 ms, gate+up dual 43-44 -> 29 ms. Caveat found later the
        same day: tiering-off inflates every `HalfToFloat` call (see Phase 1), so these absolute
        numbers are pessimistic; the end-to-end CLI A/B below ran with the normal JIT.
      - CLI, interleaved old/new builds, 3 runs each, 620-token prompt: prefill 20.3 -> 42.2 t/s
        (0.25x -> **0.51x** of llama-bench pp512 81.95), decode 16.9 -> 25.9 t/s (0.48x ->
        **0.74x** of tg64 34.95; ~0.82x on a 26-token prompt). PerformanceLeague rows added.
      - Numerics: not bit-identical to the old kernel. The block integer is the same but spread
        over the i32 lanes as ggml spreads it, so the per-lane rounding changes. Against the
        integer-exact scalar reference on 27,686 real rows: mean rel error 3.2e-7 (old 8.3e-7),
        max 1.1e-4 (old 3.9e-4, both on near-zero rows). wiki.test.raw `[1024,+)` PPL moved
        602.9 -> 544.4 vs llama-perplexity 617.1 ± 75.5: this checkpoint is ~PPL 600 as a text LM
        in llama.cpp too, so float-level changes swing it; the kernel-level check is the evidence.
        ForwardPass.Fast 749/749 pass.
      - Remaining gap: decode ~0.74-0.82x, prefill 0.51x (Q3_K has an int8 batched path, off by
        default with the rest of int8 prefill, a policy decision).
      - **Evidence for the int8-prefill policy decision (user's call, default unchanged),
        2026-10-02.** wiki.test.raw, `stingray perplexity --batched -g 0 -c 2048`, `[1024,+)`
        bucket vs `llama-perplexity -c 2048 --chunks 1`; CLI prefill 2 runs each, ~600-token
        prompt:

        | Model | llama.cpp PPL | F32 prefill | int8 prefill | Prefill t/s F32 -> int8 |
        |---|---|---|---|---|
        | Qwen2.5-0.5B Q4_K_M | 12.0055 | 11.9702 | 11.9693 | 137 -> 309 (2.2x) |
        | SmolLM2-1.7B Q4_K_M | 6.9414 | 6.9437 (+0.03%) | 6.9891 (+0.69%) | 77.5 -> 227 (2.9x) |
        | Granite-Vision-3.2-2B Q3_K_S | 617 (degenerate LM) | 546.8 | 605.7 | 40.7 -> 43.8 |

        Speed: 2.2-2.9x prefill on Q4_K. Quality: neutral on Qwen2.5-0.5B, 0.69% further from
        llama.cpp on SmolLM2-1.7B (F32 matches llama.cpp to 0.03%). Q3_K gains nothing because
        the `DotQ3K_Q8KS_*` kernels still had the pre-fix switch-in-loop shape (next item).
      - **Decided 2026-10-02 (user): int8 prefill default ON for dense models**, routed MoE experts
        still F32 (`STINGRAY_MOE_PREFILL_Q8`). Record, evidence and rollback (`=0`):
        [ADR-0003](../reference/adr-0003-cpu-int8-prefill-default.md). Open follow-ups toward
        llama.cpp numerics and MoE int8, each to be verified before acting:
        - [ ] Q3_K, MoE activation and small-prompt follow-ups, then the Granite re-run: see [prefill-numerics open list](../1-correctness/2026-10-02-prefill-numerics-investigation.md) (section "Follow-ups from the MoE audit").
      - **Done 2026-10-02 (0298f88e): the `DotQ3K_Q8KS_*` (1/2/4/8-input) int8-prefill kernels**
        got the same treatment, kept bit-identical: SIMD scale unpack, running `qs >> 2` and a
        doubling hmask bit in place of the switches, `Q3KAccumInput` unchanged. Real-weight
        64-token batched matmul hash matches HEAD; ffn_up 433-490 -> 381-429 ms, ffn_down 439-459
        -> 394-411 ms (~10%). The per-input, per-group lane-0 correction + cvt + FMA chain now
        dominates; going further means changing that chain's lane layout (not bit-identical).
      - **Tried and reverted 2026-10-02: row pairs for Q3_K.** (a) `DotQ3K_Q8K_2Row`, two rows
        interleaved per block sharing the Q8_K loads, bit-identical to single dots: attn 4.7-5.7 ->
        6.0-6.3 ms, ffn_down 19.3-20.3 -> 20.9-21.2 ms (slower). (b) Pair scheduling only (two
        sequential single dots per `Parallel.For` iteration): attn 4.3-5.0 -> 5.1-5.4 ms, ffn_down
        18.2-19.0 -> 19.6 ms (slower). Tiered JIT off, 3 interleaved rounds each. Unlike Q4_K, the
        Q3_K dot is compute-bound (heavier unpack); two rows' worth of live vectors likely spills
        the 16 YMM registers, and per-row dynamic balancing is already the right granularity.
- [ ] 10.4 Implement + verify (real image-understanding sanity check if touching anything
      vision-adjacent, not just text-only re-runs) + re-benchmark + record.

---

## Phase 1 — Voxtral-Mini-4B-Realtime (worst ratio in the doc, 0.02x → 0.221x, PLATEAU DECLARED 2026-09-12)

**Plateau rationale**: 4 real, correctness-verified perf sweeps landed (11.05x cumulative,
0.02x → 0.221x) plus real CLI/pipeline wiring (1.4, usability). Per-sweep gains are shrinking
(5.6x → 1.3x → 1.51x on wall-clock) — the remaining gap to the C++ reference (3.5x) is no longer
a single obvious lever the way the original naive-scalar-matvec bug was. 1.5 (allocation
reduction) is left open below as a real, scoped follow-up, not abandoned — but per the
mandate ("once you reach a plateau... record results and pick the next worst performing
inference"), moving to Phase 2 now rather than continuing to grind marginal gains here.

- [ ] 1.2 Quantize Voxtral weights to Q8_0 (currently raw F32 — 4x the memory traffic of every
      other model in this engine per matvec). Likely the single biggest remaining lever. Needs a
      perplexity/transcript-quality check before shipping (CLAUDE.md rule 7).
      - [ ] 1.2e Repeat for `VoxtralAudioEncoderWeights`: Q/K/V/O/gate/up/down/Projector1/
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
      - [ ] Still open: no CLI/Server.Host dispatch point found yet wiring ANY of this project's
            `ISpeechToTextPipeline` implementations (not just Voxtral) into a runnable command —
            worth a real look at `src/OpenTail.Stingray.Server*` before assuming one needs to be
            built from scratch, but out of scope for finishing 1.4's Voxtral-specific goal.
- **2026-10-02 re-measure + three bit-identical wins** (stale header: the League already had
  0.46x from 2026-09-17). Stage profile on a.wav (5.95s) showed the audio encoder at 53% of
  inference: every encoder linear was one full Q8_0 matvec per frame (weights re-streamed per
  frame). Fixed with `Q8_0BatchedLinear` (Cpu; weight-stationary, fused 4-input dot) for the
  encoder (21a2afdb) and the text-decoder prefill incl. LM head (ece9cb49); encoder attention
  vectorized in scalar order + RoPE table (34b304c4). Every step verified bit-identical (encoder
  and prefill-logits hashes unchanged). a.wav: encoder 12.8 -> 5.4-5.9s, prefill 3.3 -> 1.1s,
  inference 24.3 -> 14.7-15.3s. **14.07s `sample_16k.wav`, side by side with `audiocpp_cli
  --threads 8` the same day: 30.1s vs 24.1-25.0s, ~0.81x** (identical transcripts). Remaining:
  encoder 10.0-10.7s and 185 decode steps at ~100 ms each (18.3-18.9s). Peak RAM 13.6 GB vs
  audio.cpp 9.7 GB: the GGUF path copies every Q8_0 weight out of the mmap (`GetRawBytes`) and
  keeps an F32 embedding table plus a re-quantized copy.
  - Stage comparison vs audio.cpp's own `--log` timings (same clip): encoder 6.07s vs ours
    10.0-10.7s; prefill 0.83 vs 1.1s; 186 decode steps 17.95 vs 18.3-18.9s (decode at parity).
    NativeAOT-published build: 29.9s, same as the JIT build (encoder 9.8s), same output hashes.
  - Remaining encoder gap is GEMM throughput: `Q8_0BatchedLinear` measures 160-206 G MAC/s
    (5120x1280 / 1280x5120 / 2048x1280 at 899 rows), so the 32 layers' linears alone are ~5.1s.
    Tried and reverted: frames-outer / 32-row tiles (no gain, prefill slower). GC is not it (one
    gen2, ~250 ms pause total). Next lever: a register-tiled Q8_0 GEMM micro-kernel.
  - Benchmark trap found on the way: scratch harnesses run with `DOTNET_TieredCompilation=0`
    make CoreLib's `(float)Half` cast (every quantized kernel's block-scale conversion) hit an
    AVX/SSE transition penalty, ~62 ns per Q8_0 block vs 2.3 ns with tiering on. Production is
    unaffected (`IlcInstructionSet=native` NativeAOT). Benchmark with the default JIT.
- [ ] 1.5 Reduce per-call allocations in `LinearNoBias`/`RmsNormRow` (fresh `float[]` every call)
      by reusing preallocated scratch buffers, matching `HybridGdnForwardPass`'s pointer-scratch
      pattern. Only worth it once 1.2-1.4 land and allocation becomes the visible bottleneck.
- [ ] 1.6 Re-benchmark end-to-end (3 runs, real 14.1s clip) after 1.2-1.5, update
      `PerformanceLeague.md`, and only then mark Phase 1 fully closed.

## Phase 8 — DeepSeek-V2-Lite-Chat (`deepseek2`, prefill 0.49x)

Major, widely-used open MoE family. Note: this checkpoint has an already-accepted, separately
investigated correctness caveat (numerically wrong greedy output, root-caused to MoE
routing-landscape flatness in `docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md`,
NOT a perf question) — this phase is purely about the throughput number, which the doc's own
methodology treats as real and meaningful independent of that caveat (same approach used for
Ornith-1.0-9B/Qwen3-ASR elsewhere in the doc).

- [ ] 8.1 `DeepSeek-V2-Lite-Chat Q8_0`: prefill 28.0 t/s vs llama.cpp 56.90 t/s (**0.49x**), decode
      13.6 vs 14.83 t/s (0.92x, near-parity). The prefill-specific gap (decode is fine) suggests
      something batched-prefill-shaped again, similar in symptom to Phase 6 but a different
      architecture (MLA attention + MoE, not Gemma's dense path) — do not assume the same root
      cause without checking; profile prefill specifically.
- [ ] 8.2 This architecture requires `--allow-unverified-arch` — confirm the perf harness/CLI
      invocation still works the same way for a profiled run (`STINGRAY_PROFILE_DECODE=1` plus
      the flag) before assuming standard tooling applies unmodified.
- [ ] 8.3 Implement + verify (numeric-output-quality note: this checkpoint's greedy output is
      already known-wrong regardless of any perf fix — verify by token-for-token match against a
      pre-fix run instead of by transcript sensibility) + re-benchmark + record.

## Phase 9 — Worst-RTF TTS/audio-generation pipelines (114.14x — worse than Voxtral's original 85.5x)

`PerformanceLeague.md`'s own TTS section explicitly flags **ACE-Step Turbo (Qwen3 text encoder +
DiT + Oobleck VAE) at 114.14x RTF as "Worst RTF in this entire doc"** — worse than Voxtral's
pre-sweep 85.5x, the model this whole plan started with. Several other rows in the same section
are in a similar range: Stable Audio 3 Medium (96.75x), AudioGen-medium (31.16x), Stable Audio 3
Small Music (25.36x). These are important production TTS/music-generation models, not obscure
ones.

- [ ] 9.2 No numeric golden reference exists yet for this port (per the doc's own caveat,
      "non-degeneracy checked only") — any perf change here needs at minimum a real audio-energy/
      non-degeneracy re-check before/after, since byte-identical or transcript-based verification
      (Voxtral's approach) isn't available for this model class.
- [ ] 9.3 Implement + verify + re-benchmark + record. If ACE-Step Turbo plateaus or has no quick
      win, move to Stable Audio 3 Medium (96.75x) as the next-worst using the same approach.
      - **2026-10-02, Stable Audio 3 Medium.** Re-measured (4s, 12 steps, CFG 6, seed 1234; scratch
        harness hashing the PCM): CPU 66.3s, Vulkan 48.7s (League had 263s / 60.65s from 09-17/18).
        Stage timers: conditioning 1.6s, DiT steps 51.4s CPU / 33.0s Vulkan, **VAE decode 13.0-13.5s
        on both backends** (it always runs on CPU). Inside the VAE (seq 1853): attention 7.9s, FF
        3.8s, mapping conv 1.2s, norms 0.5s: everything but the GEMM linears was single-threaded.
        Fixed in 94288c40 (parallel over rows/heads/positions, arithmetic unchanged): VAE 6.0s,
        totals **CPU 59.0s (14.76x RTF), Vulkan 40.5s (10.12x)**, PCM hashes unchanged on both.
        Next: the CPU DiT steps (51.4s, 24 passes); the VAE's remaining time is its F32 linears.
      - **CPU DiT, same day:** per-block timers over the 24 passes: self-attention 14.4s,
        **cross-attention 19.6s**, FF 14.5s (seq 173, nCond 257). Cross-attention recomputed each
        layer's `to_kv` on the constant conditioning every pass; a834180d caches it per embedding
        (bounded to the 2 per generation): interleaved A/B 59.2/59.3s -> 50.3/54.7s, PCM hash
        unchanged. Then aa9419c5 vectorized the scalar P.V loop in
        `StableAudioAttentionKernels.DotProductAttention` in scalar order: 56.8/49.7s -> 44.9/44.2s,
        hash unchanged. **Stable Audio 3 Medium CPU today: 66.3s -> ~44.5s (11.1x RTF); Vulkan 48.7s
        -> 40.5s.** What remains on both backends is F32 GEMM time in the DiT/VAE linears (a shared
        GEMM-kernel project, not a Stable Audio item); moving on to the next phase.

## Phase 3 — Small-model decode weakness (SmolLM2/Qwen2.5, 0.125x-0.31x at 135M-500M)


## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)

- [ ] 2.3b (angle b, small-batch decode amortization) Not attempted — real architectural work
      (batching >1 token through one token-at-a-time-only decode path), out of scope for a quick
      pass. **Phase 2 plateau declared 2026-09-12**: this is a real, already-well-optimized
      256-expert-top-8 MoE running one token at a time; no unexploited kernel-level gap found by
      inspection or profiling. Moving to the next-worst unswept row per the sweep's own rule.
- **2026-10-02, 9B size point (Ornith-1.0-9B Q4_K_M) re-measured:** decode 6.8 vs llama-bench
  7.24 t/s (**0.94x**, no longer a decode problem), prefill 11.2 vs 48.1 t/s (0.23x). Cause:
  `PrefillChunked` batched the GDN/attention projections but ran the **dense FFN per token**.
  a61f53ab batches it (`DenseFfnChunked` via `BatchedProjection`): prefill **33.0 t/s (0.69x)**,
  greedy continuation identical, last-token logits vs per-token argmax-equal, cosine 0.999272
  (pre-change chunked 0.999314; note the existing 0.9995 threshold in
  `HybridGdnChunkedPrefill_MatchesSequentialPrefill` was set on Qwen3.6-35B, and the unchanged
  chunked path already measures 0.999314 on Ornith). MoE hybrid models (35B-A3B) already had
  `MoeFfnBatchedPrefill` and are unaffected. The dense 27B checkpoint on hand (Qwen3.8-27B
  UD-Q3_K_XL) shows no change (2.1 t/s prefill before and after, identical text): its prefill
  equals its decode rate because `Prefill` skips the chunked path for models with an MTP head
  (`_hasMtp`; chunked GDN is not bit-exact and flipped the Qwen3.6-MTP llama.cpp parity
  knife-edge). Enabling chunked prefill for MTP models is a numerics decision for the user, not
  a perf item; it is the remaining 27B prefill lever (0.07x row).
  **Decided 2026-10-02 (user): optimized prefill for MTP models too**, after a 41-prompt
  investigation found no systematic accuracy difference
  ([ADR-0002](../reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md)); the gate is removed.
  The 27B speed is recorded in ADR-0002 and the League.
- [ ] 2.4 Repeat 2.1-2.3 for the 9B (Ornith-1.0-9B) and 27B (Qwen3.6-27B Q3_K_XL) size points if
      a 35B fix is found and generalizes; record each in `PerformanceLeague.md`.

## Phase 6 — Gemma family (0.12x-0.13x prefill, real "batched-prefill-missing" architectural gap)

**2026-10-02: stale, effectively closed.** `ForwardPass.PrefillDispatch` has routed per-layer-
head-dim models (Gemma 4) through the batched `PrefillCore` since 2026-09-16 (per-layer dims, KV
heads, KV sharing, k_eq_v, V norm, sliding window). Re-measured vs llama-bench today: Gemma-4-12B
prefill 0.39x / decode 0.84x, Gemma-4-E4B 0.33x / 0.85x, Gemma-3-4B 0.39x / 0.85x. The remaining
prefill gap is the general F32-prefill default: with `STINGRAY_CPU_PREFILL_Q8=1` Gemma-3-4B is
107.9 t/s (0.95x) and Gemma-4-12B 36.4 t/s (0.97x). That default is the user's decision (10.2's
evidence table); no Gemma-specific work remains here. **Decided 2026-10-02: int8 prefill is the
default again** ([ADR-0003](../reference/adr-0003-cpu-int8-prefill-default.md)); uncontended
re-measure of these rows pending.

Flagship Google model family, real and severe: `PerformanceLeague.md`'s Gemma section notes
`prefill:decode still ~1.0x` for these rows — i.e. prefill and decode take about the same t/s,
the signature of a model with NO real batched-prefill path (every prefill token processed as if
it were a lone decode step, one at a time, instead of one batched matmul over all prompt tokens
at once). This is a different, more fundamental gap than the small-model-threading issue in
Phase 3.

- [ ] 6.3 **Real feature work, not attempted this pass** (same category as Phase 4/13's GPU
      dispatch fusion and Phase 14's EXAONE post-norm gap) — implementing per-layer KV heads,
      KV-layer sharing, `attention_k_eq_v`, per-head V norm, and a real `windowSize` parameter for
      `PrefillCoreAttention` is a multi-feature engineering project, not a quick fix or a flag
      flip. `docs/done/gemma4-12b-evidence.md`'s own conclusion stands: scope this properly (check
      `docs/done/cpu-prefill-repack-gemm-plan.md` for reusable batched-prefill design) before attempting
      it, and do NOT try to force the existing path again — that route is confirmed memory-unsafe,
      not just slow-to-verify. **Phase 6 is correctly closed at "well-understood, real,
      substantial follow-on work" for this pass — matches the category, doesn't need re-deriving.**

## Phase 7 — Dense 7-8B decode gap — CLOSED 2026-10-01 (Mistral-7B ~0.80x -> 0.88x); moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

## Phase 4 — Vulkan iGPU dispatch overhead (near-universal small-model prefill regression)

- [ ] 4.1 Every small model (<1B) measured on Vulkan iGPU loses badly to CPU on prefill — confirmed
      pattern, not yet attacked. Investigate batching/fusing dispatches to amortize per-call
      overhead (per `CLAUDE.md` rule 13 — do not conclude "GPU path is bad," this iGPU shares
      system RAM and has no dedicated bandwidth advantage; the actual lever is fewer, larger calls).
- [ ] 4.2 Implement + verify + re-benchmark.

## Phase 13 — SDXL-Turbo/shared-VaeDecoder UNet-attention gap via batched dispatch fusion (not the already-2x-failed per-call residency approach)

The doc's own session-arc for SDXL-Turbo already closed 8 real wins and correctly identified +
rejected 2 real regressions attempting GPU-attention residency (naive shader, then tiled/
flash-attention-style shader) — both failed specifically because "per-dispatch fixed overhead on
this iGPU... dominates at these problem sizes" with small per-call payloads. The doc's own
conclusion: this needs "either much larger batched dispatches (fusing many attention calls into
one) or access to a discrete GPU" — batched fusion is the one specific angle NOT yet tried.

- [ ] 13.1 Read the two reverted attempts' full writeups in `PerformanceLeague.md` (rows
      "~~Naive GPU attention shader~~" and "~~Tiled (flash-attention-style) GPU attention
      shader~~") before touching anything — both are real, measured, root-caused failures; do not
      re-attempt either as-is.
- [ ] 13.2 Investigate whether `SpatialTransformer`'s per-block Q/K/V/attention calls across
      MULTIPLE denoising steps (not just multiple heads within one step) could be batched into
      fewer, larger GPU dispatches — this is architecturally different from what was tried (which
      batched within one step's attention call, not across the call-count dimension). Only pursue
      if real analysis shows the multi-step batching is actually implementable given the
      sequential (each step depends on the previous step's output) nature of denoising — flag as
      a dead end honestly if the sequential dependency makes this impossible, rather than forcing
      a third attention-residency attempt.
- [ ] 13.3 If no viable batching angle exists, this phase should be marked genuinely blocked
      (needs a discrete GPU to re-test the premise, per the doc's own conclusion) rather than
      re-attempting either failed approach a third time.

## Discipline (carried over from `perf-loop-progress.md`)

- Real weights, ≥3 samples, write measured numbers not assumptions.
- No subagents in this project (CLAUDE.md rule 6) — do all work directly.
- A correctness re-check accompanies every perf change — same transcript/logits/tokens, just
  faster. A change that's faster but wrong is not a win.
- Never symlink model checkpoints on this machine (no admin/dev-mode) — pass the real
  `models/_models/...` path directly instead; `ln -s`/`mklink` silently copies tens of GB.
- **After fixing a `src/` project, rebuild the TEST project too before trusting any benchmark or
  correctness run against it** — building only the source project does not refresh a referencing
  test project's own bin-folder copy of that DLL. A stale test-side DLL silently reproduces the
  OLD (pre-fix) behavior and looks exactly like a real "no improvement"/negative result. Real
  incident: ACE-Step's cache-store fix was correct and verified in source, but the first
  re-benchmark showed zero improvement because `tests/OpenTail.Stingray.Tests.Diffusion`'s own DLL
  copy was stale. If a benchmark result looks suspiciously unchanged after a real code fix, check
  `ls -la` timestamps on the test project's referenced DLLs before concluding the fix didn't work.
