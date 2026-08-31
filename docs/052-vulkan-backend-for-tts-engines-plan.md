# 052 — Vulkan GPU backend as an explicit `--backend` option for the slow-tail TTS engines

## User instruction (verbatim intent, recorded so this isn't second-guessed later)

The user explicitly asked for GPU (Vulkan) support to be built and shipped as a real,
selectable `--backend` parameter option for the slower TTS engines — **not** conditioned on
this development machine's own (weak integrated) GPU actually beating CPU here. Quote (paraphrased
from the live session, all-caps emphasis preserved because it signals how firm this is):

> "the default path will be CPU, but on OTHER HARDWARE GPU MAY BE FASTER / BETTER. THIS PC CAN
> ONLY PROVE CORRECTNESS NOT SPEED FOR GPU, BECAUSE THE GPU HERE IS REALLY REALLY WEAK. It does
> not mean we should not build an OPTION for GPU if GPU is slower - just don't take that path
> readily - leave it as a PARAMETER CHOICE."
>
> "YOU KNOW I AM ASKING YOU TO DO THIS. BECAUSE IT'S NOT JUST FOR THIS PC. ... I TOLD YOU TO MAKE
> GPU PRESENT, AND AVAILABLE AS A PARAMETER OPTION. DO NOT STOP. ... YOU NEED TO COMPLETE THE
> WHOLE JOB."

**Implication for every engine below**: ship it as `--backend vulkan`, verify *correctness*
(golden/existing tests, or at minimum finite/bounded/non-degenerate real-weight output) on this
machine, measure and honestly report real A/B numbers on this machine (even when the number is a
wash or a regression here) — but do **not** gate whether the option ships on this machine's
number being a win. This machine is a correctness oracle, not a performance oracle, for this
feature.

## Why this doc exists

`--backend`/`-g` already existed on `TtsCommand.cs` before this plan (used by Orpheus). This plan
is about actually wiring real Vulkan compute into the engines that were previously CPU-only
despite that flag existing, starting with the slowest ones (README's TTS benchmark table, "Batch
RTF" column, worst-to-best): FishSpeech (16.3x), F5-TTS (13.0x), CosyVoice 3 (7.5x), Parler-TTS
(5.7x), Chatterbox (4.7x).

## Hard constraint discovered this session (applies to every engine below)

No Vulkan SDK is installed on this machine (`glslc`/`VULKAN_SDK` not found). This repo's Vulkan
compute shaders are precompiled SPIR-V (`src/OpenTail.Stingray.Vulkan/Shaders.Precompiled.g.cs`)
and regenerating them requires the SDK (`scripts/gen-spirv.ps1`, CLAUDE.md rule 5). **New GPU
kernels cannot be added in this session.** Every engine's GPU path is therefore necessarily a
**hybrid**: only ops already covered by `IComputeBackend`'s existing surface (`Sgemm`, `MatMul`,
`FullSeqAttention`, `RmsNorm`, `Softmax`, `SiLU`, `AddInPlace`, `ElementwiseMul`, `RoPE`) can move
to GPU. Convolutions, Mish activation, channel-first LayerNorm, and anything else needing a new
shader stay CPU-only regardless of `--backend`. This is a real, load-bearing scope limit, not
laziness — note it in each engine's section below rather than silently reproducing it each time.

## Status

### 1. Chatterbox-Turbo — DONE (commit `f7d70b0`)

Shared `CfmUNetKernels.cs` (also used by CosyVoice2, not separately benchmarked) CFM UNet decoder.
Added `IComputeBackend? backend` param threaded `RunEstimator` → `TransformerBlock` →
`SelfAttention`; QKV/Out/FFN-up/FFN-down projections route through `CfmLinearWeight.GpuMatMul`
(new method, weight uploaded once and cached on the instance, activations uploaded/downloaded per
call) when backend is non-null. Attention itself (the per-head softmax(QK^T)V loop) stays on CPU
— it's not Sgemm-shaped and the existing AVX2/FMA kernel there is already excellent; also the
`Parallel.Invoke` CFG cond/uncond branches were serialized when GPU-backed (VulkanBackend is not
verified thread-safe for concurrent dispatch from two threads onto one instance — real
correctness risk, not paranoia).

**Wired via**: `TtsCommand.cs`'s `--backend vulkan` (explicit opt-in only, not `"auto"`) →
`ChatterboxPipeline.Load(..., backend:)` → `ChatterboxDecoder` → `ChatterboxCfmDecoder.Generate`.

**Verified**: `ChatterboxCfmDecoderTests` (STINGRAY_RUN_HEAVY_TESTS=1) still passes on the CPU path
(no regression). Real A/B on this machine, same reference sentence ("Hello, I will make some
lunch, darling!"): CPU 2.78x RTF vs GPU 2.81x RTF — a wash here (isolated matmul probe showed a
real, size-dependent split, GPU 3.4x slower at dim=256 attention width / 4.1x faster at dim=1024
FFN width, but the CFM decoder — 2 Euler steps — is a small slice of Chatterbox's total time next
to the T3 acoustic LM's autoregressive decode, so the two effects cancel out end-to-end here).

### 2. CosyVoice 3 — TODO, next up

Does **not** share `CfmUNetKernels` — has its own DiT (`CosyVoice3DiTModel.cs`,
`CosyVoice3DiTWeights.cs`). Needs its own, separate `IComputeBackend?` threading through its
transformer-block linear projections + FFN, same hybrid approach as Chatterbox (Sgemm-shaped ops
to GPU, conv/norm/activation stay CPU). Check `CosyVoice3DiTModel.cs`'s block structure for the
real per-op layout before assuming it matches Chatterbox's shape 1:1 (per CLAUDE.md rule 8, verify
against the real reference math, don't assume). Wire via `CosyVoice3Pipeline.Load(..., backend:)`
in `TtsCommand.cs`. Verify against whatever golden/structural test already covers CosyVoice3 (grep
`tests/OpenTail.Stingray.Tests.Audio` for `CosyVoice3`). A/B against the CPU RTF already on record
(7.476x, README).

### 3. F5-TTS — TODO

DiT flow-matching, own kernel implementation under `src/OpenTail.Stingray.Audio/F5TTS/`. Same
hybrid-port approach. Wire via `F5TtsPipeline.Load(..., backend:)`. A/B against README's recorded
12.965x CPU RTF.

### 4. FishSpeech S2 Pro — TODO

Dual-AR + Firefly codec, `src/OpenTail.Stingray.Audio/FishSpeech/`. Likely the most structurally
different of the four (dual autoregressive transformers, not a flow-matching UNet/DiT) — audit
its real hot path before assuming the same QKV/FFN-Sgemm pattern applies cleanly. Wire via
`FishSpeechFullPipeline.Load(..., backend:)`. A/B against README's recorded 16.285x CPU RTF.

### 5. Parler-TTS — stretch goal, only if time remains

Ranked 8/11 (5.659x CPU RTF), already uses `CfmLinearWeight` for its T5 encoder
(`T5EncoderWeights.cs`) — the `GpuMatMul` plumbing from step 1 may be directly reusable there with
much less new code than 2-4. Lower priority than 2-4 since it's less slow and the win, if any, is
smaller in absolute terms.

## Process for each remaining engine (repeat of what worked for Chatterbox)

1. Read the real kernel file(s) for the engine — confirm the actual per-block op sequence (which
   ops are Sgemm-shaped vs conv/norm/activation) rather than assuming it mirrors Chatterbox.
2. Add `IComputeBackend? backend = null` params threaded from the pipeline's `Load()` down to
   wherever the per-step/per-block linear projections happen; dispatch through
   `CfmLinearWeight.GpuMatMul` (or a same-pattern new helper if the engine's weight type isn't
   `CfmLinearWeight`) when non-null, existing CPU path unchanged when null.
3. Watch for concurrency: if the engine runs any GPU-backed calls from more than one thread
   concurrently (CFG cond/uncond branches, batched anything), serialize those specific calls when
   `backend` is non-null — same reasoning as Chatterbox's `Parallel.Invoke` fix.
4. Wire the CLI: `TtsCommand.cs`'s existing `--backend vulkan` opt-in (already generic across
   engines) → construct/pass the already-built `gpuBackend` into the engine's `Load(...)`.
5. Build (`dotnet build src/OpenTail.Stingray.Audio -c Release` then
   `src/OpenTail.Stingray.Cli -c Release`).
6. Verify correctness: run the engine's existing golden/structural test with
   `STINGRAY_RUN_HEAVY_TESTS=1` if one exists; otherwise run a real generation and confirm
   finite/bounded, non-degenerate output (and, if practical, listen for a "did this get worse"
   sanity check the way earlier debugging in this repo's history did).
7. A/B: run the real CLI generation once each with `--backend cpu` and `--backend vulkan`, same
   reference sentence, report the real numbers (don't round toward "it worked" if it's a wash or a
   regression on this machine — that's expected and fine per the user's instruction, just report
   it honestly).
8. Commit each engine's port as its own commit (mirroring Chatterbox's `f7d70b0`) so partial
   progress is never lost mid-session — do not batch multiple engines into one commit.
9. Update this doc's status section for that engine before moving to the next one.

## Do not

- Do not gate shipping the option on this machine's GPU winning — that directly contradicts the
  user's explicit instruction and is the whole reason this doc exists.
- Do not silently default any engine's GPU path to "auto" — `--backend vulkan` stays an explicit
  opt-in per engine, matching Chatterbox's precedent, until/unless the user says otherwise.
- Do not skip the correctness check to save time — a fast wrong answer is worse than a slower
  right one, and this codebase already treats "measure, don't assume" as load-bearing (CLAUDE.md
  rule 7).
