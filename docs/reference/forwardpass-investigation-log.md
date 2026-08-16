# ForwardPass CPU attention — investigation log

`src/OpenTail.Stingray.Engine/ForwardPass.*.cs` (the CPU dense forward pass, originally one file,
split into topical partials) keeps its inline comments focused on the *current* decision: what the
code does today, why, and what breaks if you change it. Some of its comments used to also carry
the full chronological investigation that led there — dated measurements, ruled-out hypotheses,
sections explicitly labeled superseded. That narrative is relocated here verbatim (nothing
summarized or dropped) so a future investigation into the same question doesn't have to redo the
work, without every reader of the source file having to wade through it first.

Each section below notes where its comment used to live inline.

## Flash-128/256 wide attention heads — perplexity investigation

Originally the tail of `PrefillCoreAttention`'s header comment in
`src/OpenTail.Stingray.Engine/ForwardPass.Attention.cs` (was `ForwardPass.cs` before the file
split). Current, active summary is inline at that method; this is the full investigation that
reached it.

---

Why: Flash128_MatchesMaterialisedAttention (Qwen3-8B, 32 heads / 8 KV, headDim 128,
36 layers, Q4_K_M, 256 tokens) fails its own gate — final-logit maxAbs 0.310 against a
0.01 tolerance. Flash-vs-materialised should differ only by FP reassociation, so 0.310 is
not obviously explainable as drift. Flash-64 is default-ON, so shipping the wider widths
would change prefill numerics for the most common head dim on evidence that is currently
ambiguous. Held back rather than reverted: the generalisation is very likely correct and
the open question is about the measurement, not the arithmetic.

What is already ruled out (do not redo this work):
  * The GEMM kernel. GemmF32StridedParityTests covers the exact shapes this path uses —
    (64,128,64), (64,64,128), (64,128,128), the ragged query-tile tails, and the 256
    variants — and passes. Those tests are retained precisely to pin the kernel for this.
  * A generic headDim-128 defect. Qwen3-0.6B (16 heads / 8 KV, headDim 128) diverges by
    8.3e-6 with an identical greedy token, and the gate above genuinely activates there,
    so that is a real measurement rather than a skipped test.
  * An interaction with int8 activation prefill. The divergence survives with
    SimdKernels.Q8PrefillEnabled=false (0.258).
  * A scratch-sharing race. Ownership is per-iteration `using var` in the default
    schedule and ThreadLocal in the tile-jobs schedule; all nine buffers are sized and
    freed correctly.
  * The BF16 KV branch. Both BF16 store flags are env-driven, not size-driven, so neither
    model above took it.

MEASURED 2026-08-07 — parity question RESOLVED; a perplexity gate is what remains.
On realistic prose (320 tokens, Qwen3-8B), flash-vs-materialised and the ACCEPTED
int8-activation-prefill baseline, captured in one process:
    flash-vs-materialised : cos 0.999345, maxAbs 0.762, greedy 63762
    q8-vs-f32 (ships ON)  : cos 0.999504, maxAbs 0.807, greedy 63762
Flash-128's divergence is the same order as an approximation this project already ships
enabled by default, and its maxAbs is SMALLER. Greedy token identical. The original 0.01
maxAbs bound was the wrong instrument, as suspected below. Note the OOD token sequence in
the old test read 0.310 while realistic prose reads 0.762 — a reminder that absolute
logit deltas are input-dependent and only meaningful against a calibrated baseline.

PERPLEXITY GATE RUN 2026-08-07 — DECISION: keep off by default, ship as an opt-in trade.
wikitext-2 subset (120 KB), Qwen3-8B Q4_K_M, `perplexity --batched --batch-chunk-size 512`:
    flash-128 OFF : ppl 6.0579 [256,1024)   7.5318 [1024,+)   22.35 tok/s
    flash-128 ON  : ppl 6.0896 (+0.52%)     7.5672 (+0.47%)   25.52 tok/s (+14%)
Perplexity is deterministic for a fixed path, so +0.5% is reproducible, not noise. It also
lands WORSE than the exact sequential path (6.0789), not merely worse than the batched one,
so it is not inside the envelope of the approximations already shipped. This project's
precedent for a default-on numerics change is the Q4Kx8 repack at 16.0488 -> 16.0484,
i.e. ~0%; half a percent is two orders of magnitude larger. The +14% prefill is real but
does not buy that quality on someone's behalf — it is the model owner's call, so the
widths stay behind STINGRAY_PREFILL_ATTN_WIDE_HEADS=1.

Note the earlier cosine/greedy check (cos 0.999345 vs the Q8 baseline's 0.999504, identical
greedy token) said "same envelope" and was NOT sufficient: a per-prompt cosine on final
logits missed a 0.5% corpus perplexity shift. Corpus gates outrank single-prompt similarity.

Timing caveat: one sample per arm on a contended machine, so +14% is soft; the perplexity
figures are exact. Full wikitext-2 test split has not been run, only a 120 KB subset.
The superseded reasoning is kept below for context.

SUPERSEDED: the unresolved question, and the next step to return this to active use: decide whether
0.310 is a defect or an unrealistic tolerance. On the SAME model and input, toggling int8
activation prefill — which ships ENABLED BY DEFAULT — moves the same logits by 0.352,
i.e. more than the delta this test rejects. So measure the right quantity: cosine
similarity and greedy-token agreement for flash-vs-materialised, compared against the
accepted Q8-vs-F32 baseline on the same run (maxAbs on raw logits of a 36-layer model is
the wrong instrument, and cosine is what this project already uses for such judgements).
If flash-vs-materialised is at least as tight as the shipped Q8 baseline, retune the test
to cosine + greedy and re-enable. If it is materially worse, the defect is real and is
specific to the 32-head / headsPerKv=4 geometry, since 16/8 at the same width is clean —
bisect by feeding realistic tokens instead of BuildTokens' out-of-distribution
`1 + i*17 % 997` sequence, which can make attention near-degenerate and amplify.

## Gemma 4 per-layer head-dim batched prefill — superseded framings

Originally two paragraphs inside `PrefillDispatch`'s routing comment in
`src/OpenTail.Stingray.Engine/ForwardPass.cs` (was also `ForwardPass.cs` before the file split;
this method stayed in the core file). Both describe an intermediate state of issue #351's
per-layer-head-dim plumbing that the comment immediately below them (kept inline, since it states
the actual current reason: gemma4-specific features PrefillCore never implemented, sliding-window
attention being the real blocker) explicitly supersedes ("STILL take the sequential trunk, and the
reason is no longer ... those are now separated — see below"). Kept here in case the per-layer
batched-prefill work is picked back up and this partial history is useful.

---

MoE models: batched prefill runs the CSR-bucketed per-expert FFN (MoeFfnBatched) when
MoeBatchedPrefillSupported admits the model; the configurations it excludes (TurboQuant
cache, router/norm traces, Gemma-family post-layer transforms) still prefill per token.
Per-layer head-dim models (Gemma 4): the batched PrefillCore path assumes a
single qDim/kvDim across layers, so fall back to sequential Forward until
Phase 8 plumbs per-layer head_dim through the batched paths.

Note for onAllPositionLogits callers: this fallback never calls MatMulBatched,
so it cannot exercise Q8PrefillEnabled — a caller diagnosing that path specifically
should confirm the model isn't MoE / doesn't have per-layer head dims first.
Per-layer head dims (issue #351) are now plumbed through the batched blocks: buffers are
sized from _maxHeadDim with per-layer qDim/kvDim strides, Q/K norms and RoPE take the
layer's dim via ApplyRopeLayer (which also picks the SWA rope table), and
PrefillCoreAttention derives headDim per layer. SnapKV is NOT covered — SnapKvSelector is
still constructed from a single model-wide head dim — so per-layer models with SnapKV
eviction active keep taking the sequential path.
