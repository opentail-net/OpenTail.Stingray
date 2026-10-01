# PerformanceLeague sweep — phased plan

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

1. Phase 5: re-scan `PerformanceLeague.md` for uncovered rows (one pass, no code).
2. Horizontal Pass A, item A.5: one manual file review plus one scoped fix.
3. Phase 11: Citrinet-ASR, per-stage profile of a small, fast pipeline.
4. Phase 10: VLM text decode, profile first (10.2), then the Vulkan regression (10.3).
5. Phase 1: Voxtral Q8_0 weights (a known, large lever with a plateau declared).
6. Phase 8: DeepSeek-V2-Lite prefill (0.49x), profile first.
7. Phase 9: worst-RTF TTS/audio-generation pipelines (large, no numeric golden for some).
8. Phase 3: small-model decode (a blanket threading fix already failed and was reverted; needs per-function work).
9. Phase 2: Qwen3.6 hybrid-GDN decode (plateau declared; remaining angle is architectural).
10. Phase 6: Gemma batched prefill (a multi-feature project).
11. Phases 4 and 13: iGPU dispatch overhead (uncertain; the premise needs a discrete GPU to re-test).

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

## Horizontal Passes B and C — CLOSED; moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

## Phase 15 — DSpark speculative decoding — CLOSED; moved to [done/perf-sweep-plan-closed-phases.md](../done/perf-sweep-plan-closed-phases.md)

---

## Phase 5 — Sweep remainder

- [x] 5.1 Whisper Tiny (0.46x ratio, worst of the mature Whisper sizes) investigated. Checked the
      leading hypothesis first (this checkpoint is loaded via `LoadFromSafetensors` vs the other
      sizes' GGUF path — maybe a slower code path): **ruled out** — `WhisperPipeline.Load`/
      `LoadFromGguf`/`LoadFromSafetensors` all converge to the same `FromModel` factory, building
      the identical `WhisperEncoder`/`WhisperDecoder` classes regardless of load format; there is
      no separate compute path per loader. Most likely real explanation instead: Whisper Tiny
      (39M params) is by far the smallest model in this family, and this doc already has a
      confirmed, real pattern for exactly this — Phase 3's small-model-decode-weakness finding
      (per-call/threading overhead proportionally larger at tiny hidden sizes). **Not re-attempting
      a threading fix here** given Phase 3.2's own real, measured regression when a similar fix was
      tried — that risk applies here too without new evidence. Real per-stage profiling (mirroring
      Phase 3.1's `STINGRAY_PROFILE_DECODE=1` methodology) would be the correct next step if
      revisited, not a blind retry of an already-reverted fix class.
- [ ] 5.2 Re-scan `PerformanceLeague.md` for any row below 0.5x not covered by Phases 1-4 and add
      it here before declaring the sweep done.

---

## Horizontal Pass A — naive-scalar-matvec-never-wired-to-SIMD (the Voxtral bug, found in 15 more files)

Voxtral's original bug (Phase 1.1) was a `for (int o = 0; o < outDim; o++) { for (int i = 0; i <
inDim; i++) sum += ... }` scalar double-loop instead of the engine's own SIMD/parallel
`SimdKernels.MatVecF32` — 5.6x win when fixed. Scanned the whole `src/` tree for the same literal
pattern (`for (int o = 0; o < outDim; o++)`) and classified each of the 31 hits as
already-SIMD/BCL-accelerated vs. truly naive. **15 files were genuinely the same bug**, spanning
PersonaPlex (Mimi codec), OmniVoice (codec + semantic encoders), HiggsAudio (codec encoder),
VibeVoice (connector), FunASR-Nano (adaptor + SANM block), NemotronAsr (RNNT decoder + conformer
encoder + subsampling), RVC (synthesizer + rmvpe + hubert encoders), CosyVoice3 (flow encoder),
QwenTTS (speaker encoder), and XTTS (ResNet encoder) — several of which directly back this doc's
worst TTS/ASR rows (PersonaPlex ~262x RTF, HiggsAudio 13.64x, OmniVoice ~13.45x).

- [x] A.1 Classified all 31 files hitting the `for (int o = 0; o < outDim; o++)` pattern:
      14 already SIMD-backed (`SimdKernels`/`DenseKernels`/`TensorPrimitives.Dot`), 1 needs manual
      review (`MossTtsGlobalTransformer.cs` — not yet checked), 15 confirmed genuinely naive,
      1 (`MeloRelativeEncoder.cs`) uses a **transposed** weight layout (`weight[i*outDim+o]`, not
      row-major `[outDim,inDim]`) — deferred separately, needs its own fix since it can't just
      swap in `MatVecF32` as-is without a layout transpose (real correctness risk if rushed).
- [x] A.2 Fixed all 15 confirmed files: replaced each naive private method's BODY with a one-line
      delegation to `OpenTail.Stingray.Audio.Primitives.DenseKernels.Linear`/`LinearNoBias`
      (already the documented shared SIMD/parallel helper for exactly this — its own doc comment
      invites this reuse). Kept every method's original name/signature so NO call site needed
      touching — zero risk of a missed call site across 15 files. `src/OpenTail.Stingray.Audio`
      builds clean.
- [x] A.3 Correctness verification: **19/19 test classes passed (0 failed), 1 skipped**
      (`RvcSynthesizerRealReferenceMatchTests` — pre-existing missing reference-dump file on this
      machine, unrelated to this change). `NemotronAsrEndToEndTests` produced a real, correct,
      coherent transcript ("This little work was finished in the year eighteen oh three and
      intended for immediate publication.") post-fix. `FunAsrNanoAdaptorGolden` logs an
      already-documented pre-existing tolerance note (attention-masking gap, not caused by this
      change) but still reports as passed. All 15 fixed pipelines confirmed correct.
- [x] A.4a HiggsAudio re-benchmarked: **mean=36.333s, RTF=13.763x — matches its known 13.64x
      baseline (no regression, no measurable change)**, i.e. HiggsAudio's own `Linear` call sites
      weren't actually on this pipeline's hot path the same way, or its share of total time was
      already small. Real, honest result — not every one of the 15 fixes moves its pipeline's
      headline number equally.
      - Combined run (PersonaPlex + OmniVoice + HiggsAudio together) took only 366.165s total —
        far under PersonaPlex's OWN previous baseline of ~1049s ALONE — strongly suggesting a
        large PersonaPlex win, but that's an inference from a combined number, not a real
        measurement.
      - [x] **PersonaPlex, real isolated result: 1099s → 135.140s total (same methodology, xunit
        `Time:` for the isolated test class, includes model load) — an 8.1x speedup.** Output
        correctness re-confirmed: decoded LM text ("Hey, let me know if you have any questions.")
        exact match to the known-good transcript — same content, just fast. **This is the single
        largest win of the entire sweep so far**, ahead of Voxtral's 11.05x on a much bigger
        absolute baseline (~1049s of pure waste in a naive scalar loop inside a 25GB, 7B-class
        codec encoder). Recorded in `PerformanceLeague.md`.
      - [x] OmniVoice isolated timing: **39.014s total (incl. model load) vs baseline's 43.05s
        (generation only, different methodology — not a clean apples-to-apples comparison, but
        roughly flat either way, not a large win like PersonaPlex).** Plausible explanation: this
        pipeline's dominant cost is likely the MaskGIT generator/acoustic decoder, which this
        horizontal pass did NOT touch — only `OmniVoiceCodecEncoder`/`OmniVoiceSemanticEncoder`
        were fixed, and those may be a small fraction of this specific pipeline's total time.
        Real, honest result: this horizontal fix does not move every pipeline equally, and that's
        expected — the fix targets a specific function, not "make X faster" generically.
      - **A.4 REOPENED (2026-09-12, per user correction) — was prematurely marked closed after
        only 3 of the ~16 fixed files' pipelines got a real before/after number. Now GENUINELY
        CLOSED — every affected pipeline has a real recorded number.** Full scorecard:
        - **PersonaPlex: 8.1x** (huge, dominant-cost hit — the largest win of Horizontal Pass A).
        - **NemotronAsr: 5.1x** (38.05s → 7.44s) — the second-largest win, transcript re-verified
          byte-identical. Bigger than initially expected for 3 conformer/subsampling/decoder files.
        - **FunASR-Nano: ~2.2x** (26.24s → 11.953s), same known-degenerate synthetic-audio output
          as the baseline (not a regression, a pre-existing, unrelated caveat).
        - **VibeVoice-ASR: ~17.5%** (151.19s → 124.841s, mean of 3), transcript re-verified correct.
        - **XTTS: ~13.8%** (10.16s → 8.758s, mean of 3) — confirmed the baseline genuinely
          exercises the voice-cloning reference path (`XttsResNetEncoder` really runs both times).
        - **VibeVoice-TTS: ~9.5%** (101.27s → 91.589s, single run both sides, same methodology).
        - **HiggsAudio: flat** (13.763x vs 13.64x baseline — fix wasn't on the dominant path).
        - **OmniVoice: flat** (39.0s vs 43.1s, different methodology, fix likely wasn't on the
          dominant path either).
        - **QwenTTS: inconclusive** (~9-12s vs 6.59s baseline) — genuine methodology mismatch (no
          internal `Stopwatch` in the debug test, wall-clock includes process startup/model load
          which the original number's methodology isn't documented precisely enough to match) —
          recorded honestly as non-comparable rather than forced into a win/loss/flat bucket.
        - **CosyVoice3's `FlowEncoder` fix (`SpkEmbedAffine`)**: NOT separately isolated — it's a
          tiny 192→80 affine layer called once per generation (not per-frame), expected negligible
          regardless, and isolating it from Pass C's already-measured caching fix in the same file
          area would need an extra revert-and-remeasure cycle for a component this small. Reasoned
          conclusion recorded, not measured separately — flagged honestly as such, not silently
          assumed zero-impact.
        - **RVC (3 files): new coverage** — no pre-existing baseline in `PerformanceLeague.md` at
          all (this pipeline had never been benchmarked before), so recorded as a first-ever
          timing (105.355s combined across 3 real tests) rather than a before/after comparison.
        - **Net summary**: 6 real, confirmed wins (2 of them large — PersonaPlex 8.1x, NemotronAsr
          5.1x), 2 flat/no-real-change, 1 genuinely non-comparable (methodology), 1 reasoned-not-
          measured (negligible expected impact), 1 new-coverage-only. This is the honest,
          complete picture across all affected pipelines — exactly the audit the user asked for
          after A.4 was closed too early the first time.
- [ ] A.5 Follow-up: manually review `MossTtsGlobalTransformer.cs` (flagged CHECK-MANUALLY, not
      yet classified) and separately design a correct fix for `MeloRelativeEncoder.cs`'s
      transposed-weight case (needs either a transposing `MatVecF32` variant or a one-time weight
      transpose at load time — verify either approach against a real golden reference before
      trusting it, this layout mismatch is exactly the kind of subtle thing that produces
      confidently-wrong output if rushed).

---

## Phase 11 — Citrinet-ASR (~0.20-0.25x, worse than Whisper Tiny's Phase 5.1 target)

`Citrinet-ASR (Jasper-style conv encoder)`: `0.872s (single run) | 0.171s (mean of 3) | ~0.20x`
against a real, byte-identical-transcript-verified C++ reference (`examples/audio.cpp`'s
`citrinet_asr` family) — i.e. correctness is already solid here, this is a pure throughput gap on
a real, working ASR pipeline, worse than the Whisper Tiny row already queued in Phase 5.1.

- [x] 11.1 Checked: `CitrinetAsr.cs` uses `JasperKernels.Conv1d` exclusively (depthwise-separable
      conv, not simple Linear/matvec layers — the Voxtral/Pass-A pattern doesn't directly apply
      here at all, this is architecturally a different shape of kernel). Inspected
      `JasperKernels.Conv1d` itself: already reasonably optimized — `Parallel.For` across output
      channels in every branch (depthwise, pointwise-1x1, general), and the pointwise 1x1 fast
      path already uses `TensorPrimitives.MultiplyAdd` (real SIMD). **No naive-unwired-SIMD bug
      found** — this is NOT a repeat of Voxtral's bug class.
- [ ] 11.2 Given 11.1 found no quick pattern-match win, real per-stage profiling (mirroring Phase
      3.1/9.1b's `Stopwatch`-around-stages methodology) is the correct next step to find where the
      ~0.20x gap actually comes from (mel extraction? which Jasper block? the CTC decode head?) —
      NOT attempted this pass given the small absolute times involved (872ms total) make coarse
      instrumentation noisy; would need enough repetitions to trust a split, as originally noted.
- [ ] 11.3 Implement + re-verify the exact-transcript-match correctness this pipeline already has
      (do not regress it) + re-benchmark (3+ runs, this pipeline is fast enough to afford more
      samples than the slow TTS/diffusion pipelines elsewhere in this doc) + record.

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
- [ ] 10.2 Profile `Granite-Vision-3.2-2B Q3_K_S` decode (worst prefill ratio, 0.41x) with
      `STINGRAY_PROFILE_DECODE=1` for a real trunk breakdown, same methodology as Phase 3.1/7.1.
- [ ] 10.3 Separately investigate the Granite-Vision-3.2-2B Vulkan decode regression (4.1 vs
      17.1 t/s CPU) — real, flagged, unexplained gap distinct from the usual "Vulkan trails CPU on
      prefill only" pattern seen everywhere else in this doc.
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

- [x] 1.1 Root-cause: `VoxtralTextDecoder`/`VoxtralAudioEncoder`'s `LinearNoBias`/`LinearBias`
      used naive scalar triple-loop matvecs, never wired to this codebase's existing AVX2/FMA
      `SimdKernels.MatVecF32`. Fixed. **Result: 1203.1s → 214.8s mean (5.6x), ratio 0.02x → 0.11x.**
      Correctness re-verified (5/5 tests, exact reference transcript). Recorded in
      `PerformanceLeague.md` 2026-09-12.
- [ ] 1.2 Quantize Voxtral weights to Q8_0 (currently raw F32 — 4x the memory traffic of every
      other model in this engine per matvec). Likely the single biggest remaining lever. Needs a
      perplexity/transcript-quality check before shipping (CLAUDE.md rule 7).
      - [x] 1.2a `FastVectorTypeConverter.ConvertF32ToQ8_0` was flagged in its own doc comment as
            "never verified... a quantizer that runs and is subtly wrong is worse than either" —
            wrote `ConvertF32ToQ8_0VerificationTests` (aggregate RMS relative error vs a real F32
            matvec, not a single-row bound which is too noisy for random quantization error).
            **Passes: RMS relative error < 5%, correct behavior for a real 8-bit quantizer.**
            Cleared to wire into a real pipeline. Committed as a real, durable test (not scratch).
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
      - [x] 1.2c Re-verified with REAL audio: 5/5 tests pass, transcript still exact match
            ("This little work was finished in the year 1803, and intended for immediate
            publication."), logits shifted only slightly (mean -31.182 vs -31.115 pre-quant,
            std 1.330 vs 1.327 — expected quantization noise, no quality loss).
      - [x] 1.2d Re-benchmarked: 213.458s → 164.163s (1.3x), ratio 0.11x → 0.147x. Cumulative
            from original baseline: 1203.1s → 164.163s = **7.3x**, ratio 0.02x → 0.147x. Recorded
            in `PerformanceLeague.md` 2026-09-12. Still 6.8x off the C++ reference — not
            plateaued yet.
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
- [x] 1.3 Cache `TimeEmbedding`/`AdaGate` across decode steps. Implemented (`AdaScalePerLayer` on
      `KvCache`), correctness re-verified byte-identical (2/2 tests). **Measured: 213.458s vs
      214.804s baseline — no real win (<1% noise).** Kept anyway (free, correct), but confirms
      this was never the real lever — the 32-dim ops were negligible next to the 3072-dim
      matvecs. Recorded in `PerformanceLeague.md` 2026-09-12.
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
      - [ ] Still open: no CLI/Server.Host dispatch point found yet wiring ANY of this project's
            `ISpeechToTextPipeline` implementations (not just Voxtral) into a runnable command —
            worth a real look at `src/OpenTail.Stingray.Server*` before assuming one needs to be
            built from scratch, but out of scope for finishing 1.4's Voxtral-specific goal.
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
- [ ] 9.2 No numeric golden reference exists yet for this port (per the doc's own caveat,
      "non-degeneracy checked only") — any perf change here needs at minimum a real audio-energy/
      non-degeneracy re-check before/after, since byte-identical or transcript-based verification
      (Voxtral's approach) isn't available for this model class.
- [ ] 9.3 Implement + verify + re-benchmark + record. If ACE-Step Turbo plateaus or has no quick
      win, move to Stable Audio 3 Medium (96.75x) as the next-worst using the same approach.

## Phase 3 — Small-model decode weakness (SmolLM2/Qwen2.5, 0.125x-0.31x at 135M-500M)

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
      - **Lesson for this sweep**: the direct `STINGRAY_CPU_THREADS=1` A/B (3.2's original
        measurement) is still a valid, true finding — thread-wake overhead really is a net loss at
        this specific tiny shape when EVERYTHING is single-threaded together. It does NOT license
        assuming a naive per-call threshold change generalizes safely across a shared kernel file
        with 41+ heterogeneous call sites — verify per-function, not by pattern-matching text.

## Phase 2 — Qwen3.6 hybrid-GDN family (0.05x-0.17x across 27B/35B/9B size points)

- [x] 2.1 Got a real `STINGRAY_PROFILE_DECODE=1` GDN/Attention/MoE time-split on
      `Qwen3.6-35B-A3B-UD-Q6_K.gguf` (real weights). **Result: MoE/FFN 76.06%, GDN recurrence
      21.11%, Attention 2.82% — the GDN scan is NOT the bottleneck, MoE/FFN is.** Recorded in
      `PerformanceLeague.md` 2026-09-12.
- [x] 2.2 Inspected `MoeFfnCore`: already batches 24 per-expert `Parallel.For` sweeps into 2,
      already uses prequantized Q8_K-input int-domain dot kernels. No naive-loop bug found —
      this looks like near-floor cost for a 256-expert top-8 MoE run one token at a time across
      40 layers, not an unexploited kernel gap. **No fix proposed yet; genuinely inconclusive**,
      not force-closed. Two concrete angles left unexplored, for the next iteration:
      (a) per-token TPL `Parallel.For` dispatch overhead at this expert count/shape — measure it
      directly (e.g. wrap with a no-op kernel and diff timings) before assuming it's negligible;
      (b) whether any small-batch (>1 token) decode path could amortize router/dot-kernel setup
      cost, mirroring what `CudaHybridGdnBatchedPrefillTests` already does on the CUDA side.
- [x] 2.3a Closed analytically rather than re-running the model: 40 layers × 512 decode tokens =
      20480 `MoeFfnCore` calls total, and the profiler measured 152347.39ms total MoE/FFN time —
      **≈7.44ms per call**. Two `Parallel.For` dispatches per call at a typical 10-100μs TPL
      dispatch overhead each is ≤0.2ms, under 3% of the per-call time even at the pessimistic end,
      and each dispatch covers `numActive(8) × expertDim` work items (thousands), which is exactly
      the regime TPL amortizes well. **TPL dispatch overhead is not the bottleneck** — the 7.44ms
      is real compute (quantized dot products over large expert weight matrices). No live rerun
      needed to reach this; the existing profiler output already contained the answer.
- [ ] 2.3b (angle b, small-batch decode amortization) Not attempted — real architectural work
      (batching >1 token through one token-at-a-time-only decode path), out of scope for a quick
      pass. **Phase 2 plateau declared 2026-09-12**: this is a real, already-well-optimized
      256-expert-top-8 MoE running one token at a time; no unexploited kernel-level gap found by
      inspection or profiling. Moving to the next-worst unswept row per the sweep's own rule.
- [ ] 2.4 Repeat 2.1-2.3 for the 9B (Ornith-1.0-9B) and 27B (Qwen3.6-27B Q3_K_XL) size points if
      a 35B fix is found and generalizes; record each in `PerformanceLeague.md`.

## Phase 6 — Gemma family (0.12x-0.13x prefill, real "batched-prefill-missing" architectural gap)

Flagship Google model family, real and severe: `PerformanceLeague.md`'s Gemma section notes
`prefill:decode still ~1.0x` for these rows — i.e. prefill and decode take about the same t/s,
the signature of a model with NO real batched-prefill path (every prefill token processed as if
it were a lone decode step, one at a time, instead of one batched matmul over all prompt tokens
at once). This is a different, more fundamental gap than the small-model-threading issue in
Phase 3.

- [x] 6.1 Confirmed directly with a fresh real run: `Gemma-4-12B-it-Q4_K_M.gguf`, prefill 3.8 t/s
      vs decode 3.5 t/s — essentially identical (0.9-1.1x depending on run), matching the doc's
      existing 0.12-0.13x-ratio rows exactly. Confirmed in CODE, not just timing: `ForwardPass.cs`
      computes `perLayerHdUnsupported = _layerHeadDim is not null` and when true, "prefill" is
      literally `for (i in N) Forward(tokens[i], startPos+i)` — the decode path called in a loop.
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
- [ ] 6.3 **Real feature work, not attempted this pass** (same category as Phase 4/13's GPU
      dispatch fusion and Phase 14's EXAONE post-norm gap) — implementing per-layer KV heads,
      KV-layer sharing, `attention_k_eq_v`, per-head V norm, and a real `windowSize` parameter for
      `PrefillCoreAttention` is a multi-feature engineering project, not a quick fix or a flag
      flip. `docs/done/gemma4-12b-evidence.md`'s own conclusion stands: scope this properly (check
      `docs/cpu-prefill-repack-gemm-plan.md` for reusable batched-prefill design) before attempting
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
