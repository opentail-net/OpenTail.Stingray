# PerformanceLeague sweep: closed phases

> **ARCHIVED 2026-10-02.** Moved verbatim out of [../perf-sweep-plan.md](../4-performance/perf-sweep-plan.md).
> - **Phase 5** (sweep remainder): Whisper Tiny investigated (architecture/threading overhead, not loader);
>   `PerformanceLeague.md` rows < 0.5x re-scanned: Qwen3-Embedding-0.6B Q8_0 prompt fixed with fused
>   `DotQ8_0_Q8_0_4In_Avx2` (66daf1c2, 113-124 -> 158-184 t/s), small K-quant prefill gap identified
>   (CPU int8 prefill policy decision).
> - **Phase 16** (RWKV6/RWKV7 prefill): parallel heads with AVX2 head steps and TensorPrimitives
>   (ad0c3851). RWKV7 prefill 56.6-57.1 -> 69.9-80.3 t/s, decode 18.7-19.2 -> 22.1-22.3 (0.95x); RWKV6
>   prefill 63.6-64.5 -> 78.3-79.1, decode 19.4-19.8 -> 21.5-21.9 (0.94x). Residual gap is ~89% Q8_0 GEMM.
> - **Horizontal Pass A** (naive scalar matvec): 15 files identified and converted to `DenseKernels.Linear`/
>   `LinearNoBias`, 19/19 test classes passing. Real confirmed wins across PersonaPlex (8.1x), NemotronAsr
>   (5.1x), FunASR-Nano (2.2x), VibeVoice-ASR (17.5%), XTTS (13.8%), VibeVoice-TTS (9.5%). A.5 manual
>   reviews completed (neither is the Voxtral bug).
> - **Phase 12** (FLUX.1 Vulkan speedup): answered. Per-stage profiling exists (2026-09-25 row in
>   `PerformanceLeague.md`: CLIP 0.6s, T5 15.8s on CPU, DiT 23.5s per step, VAE 7.3s), the GPU gap was
>   measured kernel by kernel against ggml's `test-backend-ops` (017765c), and FLUX.1 Vulkan went
>   173.6s -> 132.0s with register-tiled flash attention (a5daf08). The "near-zero speedup" no longer holds.
> - **Phase 14** (post-norm GPU layer split): `HybridForwardPass` gained optional pre-norm and
>   post-norms for EXAONE 4.5 (93332cf, 69141da), matching llama-server exactly at `-g 16`, with hybrid
>   decode 1.8-1.9 t/s and prefill 1.8 t/s measured on the iGPU (log in
>   [../101-work-queue-after-coverage-plan.md](101-work-queue-after-coverage-plan.md)).

## Phase 12 — FLUX.1-schnell's near-zero Vulkan speedup (~3%, unexplained, flagged in the doc itself)

`PerformanceLeague.md`'s own FLUX.1-schnell row explicitly flags this as unresolved: "not yet
known whether the bottleneck is the T5-XXL encoder... or the DiT body itself" — a real, named
open question on a flagship diffusion model, not something this plan is inventing.

- [x] 12.1 Added real per-stage `Stopwatch` timing to `ImagePipeline.Generate`
      (`src/OpenTail.Stingray.Diffusion/ImagePipeline.cs`) — CLIP-L encode / T5-XXL encode /
      noise+pos-id setup / DiT denoise loop / VAE decode, gated behind the same
      `STINGRAY_PROFILE_DECODE=1` env var used for ACE-Step's identical Phase 9 instrumentation
      (reused, not duplicated). Builds clean.
- [ ] 12.2 Only `models/flux1-schnell/tokenizer_t5/tokenizer.json` was present locally (no
      DiT/CLIP-L/T5-XXL/VAE weights) — downloading the full ~13GB checkpoint now (68GB free,
      user-confirmed OK to use) to actually run the profiling.
- [ ] 12.3 Run the profiled CPU and Vulkan generations, answer whether T5-XXL or the DiT
      dominates (and specifically whether T5-XXL ignores the backend flag the way several
      ONNX-based TTS rows elsewhere in this doc do), implement + verify (real on-prompt image
      content check, not just non-degeneracy — this pipeline has a known separate
      background-tiling artifact already, re-verify any fix doesn't touch or worsen that) +
      re-benchmark + record.

## Phase 14 — Post-norm architecture GPU-layer-split gap (`HybridForwardPass`, blocks EXAONE-4.5/OLMo2-style models from GPU offload entirely)

Real, infra-level perf gap, not a one-model bug: `HybridForwardPass.cs` (the CPU+GPU layer-split
path) hardcodes pre-norm tensor names (`attn_norm.weight`/`ffn_norm.weight`) with no fallback to
post-norm-only architectures' real tensor names (`post_attention_norm.weight`/`post_ffw_norm.weight`),
unlike plain `ForwardPass.cs` which already has this fallback. This forces EXAONE-4.5-33B (and any
other post-norm-only architecture) onto CPU-only (`-g 0`), leaving real GPU-offload throughput
entirely unmeasured and unavailable for this whole architecture family.

- [ ] 14.1 Mirror `ForwardPass.cs`'s existing `FindTensor(...) is not null` fallback + post-norm
      forward math (norm applied after attn/ffn, before the residual add) into
      `HybridForwardPass.cs` — real architectural work per the doc's own assessment, not a
      one-line tensor-name swap.
- [ ] 14.2 Verify correctness first on a model this codebase already handles correctly via plain
      `ForwardPass` (confirm `HybridForwardPass`'s new post-norm path produces identical logits to
      the working CPU-only path) before trusting any new GPU-offload throughput number.
- [ ] 14.3 Benchmark EXAONE-4.5-33B with real GPU layer-split enabled (currently impossible) vs
      the existing CPU-only 1.6/1.7 t/s baseline — record the real speedup this unlocks.


---

<!-- Appended 2026-10-01: further fully-closed sections split verbatim from 4-performance/perf-sweep-plan.md: Horizontal Pass B, Horizontal Pass C, Phase 7, Phase 15. -->

## Horizontal Pass B — F32-weights-never-quantized (cross-cutting analysis item #2)

Per user direction: try ONE candidate first, verify it actually works end-to-end, before batch-
applying to the rest — unlike Pass A's uniform delegation, Q8_0 quantization is more invasive
(byte-layout conversion, per-file `Linear` call-site changes) so each candidate needs its own
correctness verification, not a blind repeat.

- [x] B.1 Scoped candidates: grepped all 34 `ReadF32` call sites across
      `src/OpenTail.Stingray.Audio`. Four "`*LlmTensorSource`" files (PersonaPlex, CosyVoice,
      OmniVoice, FunASR-Nano, QwenASR) all reuse the shared, already-quantized `ForwardPass`
      engine via `IModelTensorSource` — NOT naive, not candidates. `XttsGptWeights`
      (`src/OpenTail.Stingray.Audio/Xtts/XttsGptWeights.cs`) is a real, hand-rolled 30-layer/
      1024-hidden/4096-FFN GPT2 decoder already using `SimdKernels.MatVecF32` (SIMD, but full F32)
      — architecturally the same shape as Voxtral's pre-quantization state. Picked as the first
      (only) candidate to try.
- [x] B.2 Quantized `XttsGptLayerWeights`' 4 matvec weight matrices (`AttnCAttnWeight`/
      `AttnCProjWeight`/`MlpCFcWeight`/`MlpCProjWeight`) to Q8_0 at load time, reusing
      `VoxtralTextDecoderWeights.QuantizeQ8_0` (already-verified converter, cross-project call
      within the same `OpenTail.Stingray.Audio` assembly). `XttsGptTrunk.LinearWithBias` (used by
      `Step`, the incremental decode path) switched to `SimdKernels.MatVecQ8_0` + separate bias
      add. A SECOND forward-pass variant in the same file (`CausalSelfAttention`/`Mlp`, used by a
      batched/prefill path) called a SHARED kernel (`VitsAttentionKernels.Conv1x1`, used by 13
      other pipelines — MeloTTS/Piper/MmsTts/VITS-family) — did NOT touch that shared function;
      added a new, additive-only `Conv1x1Q8_0` overload instead and pointed only XTTS's 4 call
      sites at it, so no other pipeline's behavior changes. `src/OpenTail.Stingray.Audio` and its
      test project both build clean.
- [x] B.3 Correctness verification: **3/4 passed, 1 skipped (missing reference audio clip on this
      machine, pre-existing environment gap, unrelated), 0 failed.** XTTS's Q8_0 GPT decoder
      confirmed correct.
- [x] B.4 **XTTS result: real regression, not a win: 3.22x RTF baseline → 5.832x RTF
      post-quantization** (mean 18.912s vs baseline's 10.16s, and noisy across runs:
      16.2/23.2/17.4s). **Reverted immediately** (`git checkout --` on all three touched files,
      confirmed clean rebuild). Leading hypothesis: `MatVecQ8_0` re-quantizes the input activation
      row on EVERY call (`QuantizeRowToQ8_0` inside it) — at XTTS's smaller hidden size
      (1024/4096, vs Voxtral's 3072/9216) that per-call quantization overhead may be
      proportionally much larger relative to the bandwidth saved, especially across ~30 layers × 4
      quantized calls × many autoregressive mel-token steps.

  **Per explicit user direction: Pass B is NOT closed — this is a serial, one-candidate-at-a-time
  trial across models, not a one-size-fits-all verdict.** XTTS's regression is XTTS's own result
  (plausibly explained by its unusually small 1024-dim hidden size for an autoregressive decoder),
  not proof the technique fails elsewhere — Voxtral's own two Q8_0 passes were real wins at a
  larger 3072-dim hidden size. Each remaining candidate gets its own real try, its own real
  benchmark, and its own honest mark (win / flat / regression-reverted) below, exactly like B.4's
  XTTS entry — never batch-applied.

  **Candidates surveyed, marked one at a time:**
  - [x] **XttsGptWeights (hidden=1024, autoregressive decode)** — TRIED, **REGRESSION**, reverted
        (see B.4 above).
  - [x] **F5TtsWeights** (hidden=1024, DiT flow model) — **ALREADY DONE by a prior session** —
        `F5Kernels.LinearQ8_0`/`LinearGpuQ8_0` are real, wired, in active use (confirmed via
        `F5DiTModel.cs`/`F5DiTBlock.cs`), and `F5TtsWeights.cs` already quantizes via a shared
        `Q8_0WeightQuantizer.Quantize` helper. Not a fresh candidate — nothing to try here.
  - [x] **ParlerDecoderWeights** (hidden=1024, autoregressive, has its own KV cache — same size AND
        same call-pattern risk profile as XTTS) — **ALREADY DONE by a prior session** too, per its
        own doc comment ("same real technique and rationale as Fish Speech's fast-AR"), using
        `Q8_0WeightQuantizer.QuantizeRef` + an `IQuantWeightRef.MatVec` abstraction. **Important
        finding: this DIRECTLY CONTRADICTS the working hypothesis** that small hidden size +
        autoregressive decode causes a regression — Parler is autoregressive at the same
        hidden=1024 as XTTS and (per its own comment) was found to be a real win, not a
        regression. Could not find an isolated before/after number in `PerformanceLeague.md` to
        independently re-confirm that claim (the doc's own Parler-TTS row predates any Q8_0
        mention) — the claim rests on the code comment alone, not a re-verified measurement here.
        **The real differentiator between Parler's apparent success and XTTS's confirmed
        regression is still unexplained** — worth a real side-by-side kernel-dispatch comparison
        as a future investigation, not resolved this pass.
  - [x] **MusicGen/AudioGen transformer weights** — use `CfmLinearWeight.FromF32WithF16Conversion`
        (F16, a different existing half-measure optimization, not F32 and not Q8_0). Not a clean
        naive-F32 candidate — going further to Q8_0 would be a distinct, bigger follow-up (F16→Q8_0
        is a different comparison than F32→Q8_0), not attempted this pass.
  - [x] **XTTS's OWN remaining weight classes** (`XttsVocoderWeights` checked specifically) — this
        is a HiFi-GAN-style convolutional vocoder (`conv_pre`/transposed-upsample convs/`cond`
        convs), NOT simple Linear/matvec layers — doesn't fit this codebase's existing Q8_0
        infrastructure (matvec-specific) without a genuinely new conv-quantization kernel family.
        Out of scope for a quick serial try; a real, separate, bigger undertaking if pursued.
  - [x] **MmsTtsWeights** — checked, NOT a viable candidate: VITS-family, mostly convolutional
        (duration predictor, HiFi-GAN decoder), and its one attention-shaped part (text encoder)
        has a small hidden size typical of VITS (much smaller than even XTTS's 1024). MMS-TTS is
        ALSO already one of the fastest pipelines in the whole doc (0.38x RTF, faster than
        real-time) — not a valuable target regardless of quantization feasibility.
  - [x] **CosyVoiceWeights (v1)** — its LLM (`CosyVoiceLlmTensorSource.cs`) also reuses the shared
        `ForwardPass` engine (same pattern as CosyVoice3's LLM, already quantized) — not a fresh
        candidate. `CosyVoiceWeights.cs` itself (896-dim) is a smaller supporting piece, not
        separately investigated further.
  - [x] **CosyVoice3's OWN DiT (`CosyVoice3DiTWeights`/`CosyVoice3DiTModel.cs`) — genuine fresh
        candidate found and TRIED.** Confirmed via its own doc comment to be, tensor-for-tensor,
        the IDENTICAL architecture to F5-TTS's DiT (hidden=1024, heads=16, ffn=2048) — and its
        `Lin` dispatch was calling `F5Kernels.Linear`/`LinearGpu` (F32), NOT the
        `LinearQ8_0`/`LinearGpuQ8_0` variants F5-TTS itself uses for the exact same shape. Very
        low-risk: reused already-proven kernels (F5's own Q8_0 path), not new kernel-writing.
        Quantized the same 9 fields F5 quantizes (`NormOutLinear`/`ProjOut` + 7 per-block
        matrices), left `InputProj`/`TimeMlp0`/`TimeMlp2` as F32 (matching F5's own scope exactly,
        added a small `LinF32` overload for those 3 call sites). Builds clean (both src and test
        projects rebuilt).
      - **Correctness: PASS after one real test fixup.** One test failure
        (`Weights_LoadRealTensors_ExpectedRealShapes`, expected 81920/actual 87040) was a stale
        assertion checking `float[]`-element-count on a field now correctly `byte[]` Q8_0
        (87040 = 80 rows × (1024/32)×34 bytes/row — the math checks out, not a real bug). Fixed
        the test assertion (same pattern as Voxtral's earlier test fixups). Re-ran: 5/5 pass.
      - **Benchmark: REAL REGRESSION, reverted.** Used `CosyVoice3ReferenceCacheTests` (already-
        timed from the Pass C work) for a clean before/after on identical config: **before (Pass C
        only) call1=34.36s/call2=29.41s → after (+ this DiT quantization) call1=39.01s/
        call2=35.26s — BOTH calls got slower, ~13-20% worse.** Reverted immediately (`git checkout
        --` on all three touched files including the test fixup), confirmed clean rebuild.
      - **This is now a THIRD real data point on the Q8_0 puzzle, and it deepens the mystery
        rather than resolving it**: CosyVoice3's DiT is tensor-for-tensor identical to F5-TTS's
        DiT, uses the exact same proven kernels, yet regressed — while F5-TTS itself and Parler
        (a completely different architecture) both reportedly succeed at similar or smaller hidden
        sizes. Architecture-identity reasoning ("F5 succeeds at this exact shape, so this should
        too") was NOT sufficient to predict the real result — a real, humbling finding about the
        limits of pattern-matching for this technique. **No revised hypothesis yet**; what
        specifically differs between F5-TTS's own use of this kernel and CosyVoice3's use of the
        identical kernel (batch size per ODE step? number of frames per call? something in how
        the DiT is invoked, not the architecture itself) remains a real open question for whoever
        picks this up next — worth a genuine side-by-side comparison of call-site shapes (numFrames
        per call, ODE step count) between the two pipelines before attempting a 4th candidate.

  **Status after this round**: MmsTts and CosyVoice v1's LLM ruled out for good reasons (conv-heavy/
  already-fast, and already-quantized-via-ForwardPass respectively). Found ONE genuine, well-
  reasoned fresh candidate (CosyVoice3's DiT) via architecture-identity reasoning with F5-TTS
  (proven-successful) rather than guessing — TRIED, real regression (39.01s/35.26s vs pre-DiT-
  quant 34.36s/29.41s, both calls ~13-20% slower), reverted. The Parler-vs-XTTS contradiction on
  the "small hidden size" hypothesis remains unresolved.

  **Follow-up investigation, per user's own hypothesis ("this does not push the processor to its
  limits, yet"): a real, genuine NESTED-PARALLELISM structural issue exists in the shared
  `F5Kernels.LinearQ8_0` kernel** (used by both F5-TTS and the now-reverted CosyVoice3 DiT
  attempt) — its multi-frame branch does `Parallel.For(0, t, ti => { ... MatVecQ8_0(...) ... })`,
  but `MatVecQ8_0` ALREADY parallelizes internally (row-based `Parallel.For`) whenever `outDim` is
  large enough, which every real caller here satisfies (1024-6144). That is genuine nested
  parallelism — t frames each spawning an independent internal parallel region, all competing for
  the same thread pool at once. **Tested three real, timed variants on F5-TTS (already-shipped,
  already-correct, safe to validate against) to see if this structural issue is a real practical
  cost:**
  1. Original (nested): **34.649s**.
  2. Sequential-outer-loop "fix" (frames looped one at a time, each still calling the fully
     internally-parallel `MatVecQ8_0`): **46.744s — WORSE.** Serializing t frames' worth of
     `Parallel.For` setup/teardown cost more than removing the nesting saved.
  3. True single-level flat parallelism, per the user's own suggested design (ONE
     `Parallel.For(0, t*outDim, ...)` over the whole frame×row space, calling the low-level
     `DotQ8_0_Q8_0` SIMD dot directly, never touching `MatVecQ8_0`'s own internal threading):
     **36.337s — statistically indistinguishable from the original**, not a real win.
  - Correctness held in all three variants (0 test failures throughout). **Reverted to the
    original nested version** — measured fastest or tied-fastest of the three, and it's the
    existing, already-shipped, already-correct code, so there's no reason to carry a more complex
    implementation for a non-improvement.
  - **Conclusion: the plausible-sounding "nested parallelism must be wasteful" theory, and the
    equally plausible "fuse into one flat parallel region" fix, BOTH failed to beat the original
    when actually measured.** Most likely explanation: .NET's `ThreadPool`/`Parallel.For`
    scheduling already handles this pattern reasonably gracefully via work-stealing across a
    shared queue (rather than literally spawning t independent OS-level thread pools), so the
    theorized oversubscription cost was smaller in practice than the theory predicted. **This is
    now a second closed cautionary tale** (alongside Phase 3's threading regression) that
    plausible-sounding parallelism restructuring needs real measurement, not just structural
    reasoning — even when the reasoning sounds airtight and comes with a clear mechanism.
  - **This does NOT explain the CosyVoice3 DiT regression** (which used the ORIGINAL nested
    `LinearQ8_0`, completely unchanged, and still regressed vs its own F32 baseline) — the
    CosyVoice3 mystery from the paragraph above remains genuinely open; this investigation only
    closes off nested parallelism as a candidate explanation, it doesn't resolve the real puzzle.

  **Pass B PAUSED here (2026-09-12), per explicit user direction — this line of investigation was
  a real but unproductive aside.** Scorecard for this pass: 1 genuine win (XTTS's own case doesn't
  count as a win — see below), 3 real regressions (XTTS GPT decoder, CosyVoice3 DiT, and the
  nested-parallelism "fix" attempts averaged worse than baseline), 2 already-done-elsewhere
  (F5-TTS, Parler), 2 ruled-out-architecturally (MusicGen/AudioGen's F16, XTTS's vocoder/MmsTts's
  convolutional shape), and 1 genuinely unresolved mystery (why Parler/F5 succeed at Q8_0 while
  XTTS/CosyVoice3's DiT regress, despite similar or identical shapes). **Net effect of this whole
  pass on real throughput: zero** — every actual code change from Pass B was reverted. The one
  real win recorded this session at a similar layer (CosyVoice3's reference-conditioning cache,
  ~14.4%) was Pass C, a different technique, not this one. **Moving on to the next unresolved
  item in the sweep** rather than continuing to chase this specific mystery further.
      direction ("if it works, do a pass for the rest") — repeat B.1-B.4's pattern across the
      remaining candidates from the original 34-file scan (F5TTS, MusicGen/AudioGen transformer
      weights, Parler decoder, CosyVoice weights, XTTS's own vocoder/conditioning/dvae weights,
      MmsTts, Whisper GGML path if applicable) — one at a time, each independently verified with
      real generation output, not a blind batch repeat of B.2's diff.
      If XTTS's fix fails correctness: revert (`git diff` shows exactly what changed across
      `XttsGptWeights.cs`/`XttsGptTrunk.cs`/`VitsAttentionKernels.cs`), do NOT proceed to the rest
      until the single-candidate approach is proven to actually work.

---

## Horizontal Pass C — cacheable-deterministic-recompute (cross-cutting analysis item #3)

Searched for the same bug class as ACE-Step's silence-latent (a fixed/deterministic input
recomputed via a real forward pass on every single call). Ruled out several plausible spots first:
PersonaPlex's `SilenceTokens` are compile-time int constants (nothing to cache); Chatterbox's
default speaker embedding is a loaded weight, and its placeholder fallback is a cheap closed-form
formula (not a forward pass); CosyVoice3's own no-reference fallback is already a plain zero
vector. `XttsPipeline` already has this exact optimization done correctly
(`_refCache.GetOrAdd(referenceAudioPath, ...)` — a real, atomic, already-correct cache). Found one
genuine, unfixed match:

- [x] C.1 `CosyVoice3Pipeline.Generate(text, ..., referenceAudioPath, ...)` calls
      `ExtractSpeakerEmbedding`/`ExtractReferenceMel`/`ExtractPromptTokens` — all three PURE
      functions of `referenceAudioPath` alone (two of them real ONNX graphs: CamPlus x-vector,
      CosyVoice speech tokenizer) — fresh on every call, with NO cache at all (confirmed by
      grepping for `_refCache`/`ConcurrentDictionary` in the file — zero hits, unlike `XttsPipeline`).
      A caller synthesizing multiple sentences with the same voice reference (a very normal usage
      pattern) pays this cost every single time.
- [x] C.2 Verified safe to cache before implementing: `promptTokens`'s later `[..alignedTokens]`
      slice and `refMel`'s trim both allocate NEW arrays and only rebind the local variable —
      neither mutates the cached tuple's arrays. `speakerEmbedding` is only read downstream.
      Implemented `_refCache` (a `ConcurrentDictionary<string, (SpeakerEmbedding, RefMel,
      PromptTokens)>`) mirroring `XttsPipeline`'s own already-correct pattern exactly — used
      `GetOrAdd` (atomic, can't repeat the check-but-never-write bug ACE-Step's manual
      `Dictionary` version had). `explicitSpeakerEmbedding` (an existing override parameter) still
      bypasses the cache correctly. Null/missing `referenceAudioPath` falls back to the original
      uncached path (can't cache a null dictionary key, and there's nothing expensive to skip in
      that branch anyway). Builds clean — **rebuilt BOTH `src/OpenTail.Stingray.Audio` and
      `tests/OpenTail.Stingray.Tests.Audio`**, confirmed DLL timestamps match (the ACE-Step lesson
      applied immediately, not learned twice).
- [x] C.3 Ran `CosyVoice3ReferenceCacheTests` + `CosyVoice3RealReferenceMatchTest` (4 total, 2
      passed, 2 skipped for unrelated missing reference files — `cosyvoice3-REFERENCE-gen2.wav`
      and a WAV-comparison pair, nothing to do with this change, 0 failed). Real, non-silent audio
      produced both calls; existing golden-parity test still passes.
- [x] C.4 **Real result: call1=34.36s, call2=29.41s — ~14.4% faster on the cache hit.** A genuine,
      modest, honest win, not an ACE-Step-scale dramatic one — the two ONNX graphs
      (speaker/prompt-token extraction) are real cost but not the dominant share of this
      pipeline's total generation time (the LLM speech-token generation + DiT flow-matching ODE
      solve + HiFT vocoder likely dominate). Recorded as new coverage in `PerformanceLeague.md`
      (this pipeline had no prior standalone RTF baseline to compare against). **Pass C's single
      found-and-fixed candidate is closed.** No further Pass C candidates identified this pass
      after checking several plausible spots (PersonaPlex, Chatterbox, CosyVoice3's own
      no-reference fallback) — this bug class appears rarer than #1 (naive-matvec) or as common
      as initially guessed; not forcing further candidates without a real lead.

---


## Phase 7 — Dense 7-8B decode gap (Mistral-7B/Ministral-8B, ~0.69-0.71x decode despite ~1.0x prefill)

Real, moderate, affects some of the most mainstream dense chat models this engine runs.
`Mistral-7B-Instruct-v0.3 Q4_K_M`: prefill 1.00x parity, decode **0.69x**. `Ministral-8B-Instruct-2410
Q4_K_M`: prefill 0.99x parity, decode **0.71x**. Unlike Phase 6, prefill is already at genuine
parity here — this is a decode-specific gap on otherwise-excellent dense models.

- [x] 7.1 (done 2026-10-01, see result below) Real `STINGRAY_PROFILE_DECODE=1` profile on `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
      decode (real weights) — get the QKV/Attention/OutProj/FFN/RmsNorm/RoPE split, same
      methodology as Phase 3.1, at this dense 7B size (not MoE, not tiny) to see whether the same
      shape/threading question applies or whether it's a different bottleneck at this size
      (e.g. KV-cache read bandwidth, since decode at 571-token context reads the full KV cache
      every step).
- [x] 7.2 (done 2026-10-01) Compare against Qwen3-8B's decode ratio (0.82x, close to parity) and the same model's
      DRAM-bandwidth note in `PerformanceLeague.md` ("6.8 t/s = 34.2 GB/s = 93% of the measured
      36.8 GB/s DRAM ceiling") — if Mistral-7B/Ministral-8B are ALSO near the DRAM ceiling, the
      0.69-0.71x gap vs llama.cpp may be a real algorithmic/dequant-efficiency difference at
      matched bandwidth utilization, not a wasted-cycles bug. Measure before concluding either way.
- [x] 7.3 (done 2026-10-01, see "Phase 7.3 result" below) Implement + verify + re-benchmark if a real, actionable gap is found.


**Phase 7 measured result (2026-10-01, Mistral-7B Q4_K_M, CPU, real weights, 3 runs):** decode 7.4-7.5 t/s vs llama-bench 9.34 t/s (0.80x; 8.58 t/s and 0.87x if llama.cpp is also given 16 threads). Profile split: FFN 72.7%, QKV 14.9%, output projection 10.1%, attention 1.7%, RmsNorm+RoPE+misc 0.6%, so it is not KV bandwidth or any non-matvec overhead. Thread sweep: Stingray 8 thr 6.0 t/s, 12 thr 7.6, 16 thr 7.5; llama.cpp peaks at 8 threads. Per parameter, QKV and out-proj cost ~3.2 ms/Gparam against ~2.2 for the FFN, so the smaller matrices stream about 1.45x less efficiently; bringing them to FFN efficiency would be worth ~8% decode (~0.87x), and the rest of the gap is lower per-thread bandwidth in the Q4_K matvec generally. Real lead, not yet implemented (needs its own measured pass). Recorded in PerformanceLeague.md.

**Phase 7.3 result (2026-10-01): fixed, ~0.80x -> 0.88x on Mistral-7B.** The per-tensor lead above was mis-specified: the profile's "output projection" bucket also holds the LM head, and the per-parameter comparison ignored that `ffn_down`/`attn_v`/`output` are Q6_K. Per-byte, measured on the real weights (scratch benchmark cycling all 32 layers so nothing sits in L3): Q6_K matvec streamed 35-40 GB/s, Q4_K only ~31 GB/s even on the big FFN matrices. The difference was scheduling, not the dot kernel: `MatVecQ6K` uses a dynamic per-row `Parallel.For`, while `MatVecQ4K` and the Q4_K branch of `MatVecDual` (gate+up) split rows into exactly one static chunk per thread, so the slowest thread set every matvec's pace. Changed both to one row pair per dynamically balanced iteration. Chunk-size sweep, end to end, Mistral-7B: static 7.5 t/s, 8 rows 8.0, 4 rows 8.3, 2 rows 8.5. Gate+up isolated (5 alternated runs): 57.5-62.9 ms -> 53.3-53.7 ms per token. Same-day A/B: SmolLM2-1.7B 21.8-26.8 -> 28.3-30.2 t/s, Qwen3-8B 6.5-7.1 -> 7.4-7.8, OLMoE 35.0-36.0 -> 35.9-38.4, SmolLM2-135M unchanged within noise. Ratios against llama-bench (8 threads) are in PerformanceLeague.md: Mistral 0.88x, Qwen3-8B 0.87x, Ministral-8B 0.84x, SmolLM2-1.7B 0.79x. Pairs still start on even rows, so the 2-row/1-row kernel split, and therefore every result, is unchanged: Mistral's 128-token greedy text is byte-identical, ForwardPass.Fast 749/749, heavy ForwardPass (STINGRAY_RUN_HEAVY_TESTS=1) 156 passed, 58 skipped, 1 failed: the failure (`DeepSeek2GreedyParityTests`, diverges at token 9) reproduced identically on the pre-change code. Not yet done: other kernels still use static per-thread chunks (`MatVecF32`, several IQ/Q8K paths in `SimdKernels.cs`); each needs its own measured A/B.

## Phase 15 — DSpark speculative decoding: disambiguate weak-draft-head vs. per-block overhead (currently an unresolved "confirmed loss")

`PerformanceLeague.md` already root-causes both speculative-decoding configs as hardware-limited
losses (VNNI/`vpdpbusd` needed, absent on this Zen 3 CPU) — but explicitly leaves one sub-question
open: "the draft head itself may be weaker, or block-7 confidence-gated speculation costs more per
rejected block than simple n-token drafting does; not yet disambiguated." This is a real,
previously-flagged loose end, not a fresh guess.

- [x] 15.1/15.2 **Disambiguated using data already in `PerformanceLeague.md` — no new inference run
      needed.** DSpark's own per-step breakdown: draft 3438ms, verify 5879ms, commit 68ms (total
      ≈9385ms/step for a block-7 draft). Verify costing ~1.7x draft is expected, unremarkable
      behavior for batch-verifying 7 tokens against the full model (matches this doc's own
      already-established "Q4_K dot is ~87% compute-bound... verifying k tokens costs ~k× compute
      regardless of dispatch" finding) — nothing here suggests an anomalous "block-7
      confidence-gating tax" on the verify/commit side specifically. The much larger, more direct
      differentiator between the two configs is **acceptance rate: 23% (DSpark) vs 62% (n-gram/
      draft-model)** — a >2.5x gap. Since draft+verify compute is paid in full regardless of
      whether the drafted tokens are accepted, a much lower acceptance rate directly means most of
      that real, fixed-per-step cost produces nothing usable far more often. **Conclusion: the
      dominant explanation is a WEAKER DRAFT HEAD for the DSpark pairing, not a structurally more
      expensive per-rejected-block verify/commit cost.** The per-step cost structure itself looks
      normal; the acceptance-rate gap is where the real difference lives.
- [x] 15.3 **Phase 15 CLOSED.** Both speculative-decoding configs remain confirmed hardware-limited
      losses (no VNNI on this Zen 3 CPU) — that conclusion stands unchanged. The doc's own open
      sub-question (weak draft head vs. per-block overhead) is now disambiguated in favor of "weak
      draft head" based on the acceptance-rate gap being the dominant differentiator, reasoned from
      existing data rather than requiring a fresh benchmark run. Nothing actionable on this
      hardware either way — a valid, complete closure per this sweep's own plateau discipline.

---

<!-- Appended 2026-10-02: further fully-closed sections split verbatim from 4-performance/perf-sweep-plan.md: Phase 5, Phase 16, Horizontal Pass A. -->

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
- [x] 5.2 (done 2026-10-01) Re-scanned `PerformanceLeague.md` for rows below 0.5x. Covered already: small-model
      decode (Phase 3), hybrid GDN (2), Gemma prefill (6), RWKV prefill (16), DeepSeek2 prefill (8), Voxtral (1),
      VLMs (10), Whisper Tiny (5.1). Bold sub-0.5 values on Whisper Base/Small, Parakeet and Qwen3-ASR are RTFs,
      not ratios (Base/Small 0.83x, Parakeet 1.03-1.16x; Qwen3-ASR has no C++ reference). Two uncovered rows:
      - **Qwen3-Embedding-0.6B Q8_0 prompt, 0.14-0.16x (2026-09-17)**: fixed. Cause: Q8_0 weights get no help from
        the int8 prefill tier, and `MatVec4In` ran four separate serial-FMA dots per weight row. New fused
        `DotQ8_0_Q8_0_4In_Avx2` (bit-identical per token, so on by default): Qwen3-0.6B Q8_0 prefill 113-124 ->
        158-184 t/s, embedding 605 tokens 4.65-4.89 s -> 3.39-3.57 s (~0.50x of llama-bench `-embd 1` pp512 at 16
        thr, 350 t/s), RWKV7 Q8_0 prefill 42-43 -> 60-61 t/s. 66daf1c2.
      - **SmolLM2-1.7B prefill, 0.24-0.33x (2026-08/09)** and small K-quant prefill generally: the CPU int8 prefill
        tier (`STINGRAY_CPU_PREFILL_Q8`) was turned OFF by default on 2026-10-01 (7791e3c9, exact-numerics policy).
        Measured today on Qwen2.5-0.5B Q4_K_M (633 tok): 141-143 t/s off vs 260-308 t/s on, i.e. the default
        halves K-quant prefill. Whether to restore it is a policy decision, not a perf bug; note it also brings
        DeepSeek-V2-Lite back to token-exact llama.cpp parity (bugstofix #23). Remaining Q8_0 gap vs llama.cpp
        (~0.4x on Qwen3-0.6B: 158-184 vs 443 t/s) is GEMM-kernel throughput (llama.cpp uses repacked tiles).

---

## Phase 16 — RWKV6/RWKV7 prefill — DONE 2026-10-02 (0.21x -> ~0.45x prefill, decode 0.94-0.95x; rest is Q8_0 GEMM)

Added 2026-10-01 with the `rwkv6`/`rwkv7` admissions (code: `RwkvForwardPassBase`, `Rwkv6ForwardPass`,
`Rwkv7ForwardPass`). Prefill already runs batched: one matmul per projection over chunks of up to 256
tokens, with only the WKV recurrence stepping token by token. That took RWKV7 Q8_0 from 17.5-19.8 to
38.7-39.7 t/s over 605 tokens (same-day A/B, idle machine), and the fused Q8_0 4-input dot (66daf1c2)
to 60-61 t/s over 580 tokens. llama.cpp pp512 is 185 t/s (16 thr). Measured
before profiling: the int8 prefill tier (`STINGRAY_CPU_PREFILL_Q8=1`) and BLAS on/off did not help
(29-32 t/s, those two runs under background load, so only indicative).

- [x] 16.1 (done 2026-10-02) 587-token RWKV7 prefill: matmuls 60%, the other 40% (4.0 s) a single-threaded scalar
      per-token block, dominated by the WKV7 state update (~12M MACs/token). Original text: per-stage profile of a ~600-token prefill: batched matmuls vs the per-token WKV step, group
      norm, lerps and LayerNorms (all scalar loops today). An env-gated stage timer is enough; the
      2026-10-01 attempt was abandoned because another process was loading the CPU.
- [x] 16.2 (done 2026-10-02, ad0c3851) Heads now run in parallel (each walks the chunk's tokens through its own
      state slice) with AVX2 head steps, and the element-wise ops use TensorPrimitives: non-matmul time 4.0 -> 0.7 s.
      CLI A/B: RWKV7 prefill 56.6-57.1 -> 69.9-80.3 t/s, decode 18.7-19.2 -> 22.1-22.3 (0.95x); RWKV6 prefill
      63.6-64.5 -> 78.3-79.1, decode 19.4-19.8 -> 21.5-21.9 (0.94x). Parity unchanged or better. What is left
      (~0.45x prefill) is ~89% Q8_0 GEMM, i.e. the same kernel-throughput gap as small Q8_0 transformers (5.2).
      Original text: fix the biggest stage, re-verify `Rwkv6/Rwkv7GreedyParityTests` and `RwkvRecurrentStateTests`,
      re-benchmark (3 alternated runs vs the current build), update the League rows.

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
- [x] A.5 (done 2026-10-01: both reviewed, neither is the Voxtral bug, no change made) `MossTtsGlobalTransformer.cs`:
      every linear already goes through `SimdKernels.MatVecF32` and LayerNorm through `SimdKernels.LayerNorm`;
      leftover inefficiencies are not scalar matvecs (`LinearBatched` nests `Parallel.For` over tokens around an
      already-parallel matvec instead of the batched F32 GEMM; the attention context is accumulated with a scalar
      headDim loop), worth doing only if MOSS-TTS gets a League row. `MeloRelativeEncoder.LinearVec` is the
      speaker-embedding projection, run once per encoder/flow layer per utterance (~50k MACs each), already
      parallelized; a transposed-weight kernel would not move any measured number. Original item text:
      manually review `MossTtsGlobalTransformer.cs` (flagged CHECK-MANUALLY, not
      yet classified) and separately design a correct fix for `MeloRelativeEncoder.cs`'s
      transposed-weight case (needs either a transposing `MatVecF32` variant or a one-time weight
      transpose at load time — verify either approach against a real golden reference before
      trusting it, this layout mismatch is exactly the kind of subtle thing that produces
      confidently-wrong output if rushed).


