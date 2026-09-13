# Wan 2.1/2.2 UMT5-XXL Text Encoder GPU Residency Plan (2026-09-13)

## Context — a proven template already exists, copy it

`src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs` (FLUX's T5-XXL encoder) just went
through this exact transformation — see `docs/071-flux-t5xxl-gpu-residency-plan.md` for the
original plan and `git log` commit `53839a3` for what actually landed:
`T5GpuWeights.cs`, `T5GpuWorkspace.cs`, and `T5Encoder.InitGpu`/`EncodeGpu` (all real, committed,
verified via `T5GpuParityTests`). **Wan's own text encoder, `UMT5Encoder.cs`, is architecturally
almost identical** — same layer count, same dims, same attention/FFN math — so this doc is mostly
"do the same thing again," with the real differences called out explicitly below. Read `docs/071`
first for the full original reasoning; this doc only covers what's different for UMT5.

## Why this is worth doing

`src/OpenTail.Stingray.Diffusion/TextEncoders/UMT5Encoder.cs` (202 lines) is, like `T5Encoder` was
before `docs/071`, **fully CPU-only** — confirmed by grep, zero references to `IComputeBackend`,
`Gpu`, or `Vulkan` anywhere in the file. This is the text encoder Wan 2.1/2.2 uses for its own
cross-attention conditioning (`WanModel.PrecomputeCrossKvCache`/`PrecomputeCrossKvCacheGpu` both
take a pre-encoded `textContext` array as input — encoding it is a separate step this class owns).
Given how much of Wan's DiT is now real, correct, GPU-resident work (`docs/072`, `docs/073`,
commits `0132aa7`/`d877353` — GPU now *faster* than CPU per-block, 759.7ms vs 1021.1ms), the text
encoder is now the most obviously CPU-bound piece left in the Wan pipeline, the same position
T5-XXL was in for FLUX before `docs/071`.

## Real architecture — confirmed identical to T5-XXL except three things

Per `UMT5Encoder.cs`'s own class doc comment (already fact-checked against the real
`google/umt5-xxl` config and the real downloaded checkpoint, not assumed):

- **Same core math as `T5Encoder`**: 24 layers, `d_model=4096`, 64 heads, `head_dim=64`,
  `d_ff=10240`, RMSNorm (no bias), additive relative-position-bias attention (no `1/sqrt(head_dim)`
  scaling — a real, confirmed T5-family quirk, not a bug), gated-GELU FFN. **This means
  `T5MultiHeadAttentionRelBias` (the fused GPU attention kernel built for T5-XXL,
  `IVisionOpsBackend`/`Shaders.cs`) is directly reusable as-is** — headDim and the bias-addition
  math are identical, nothing new to write there.
- **Difference 1 — per-layer relative position bias**, not a single shared one. T5-XXL computes
  `_relPosBias` ONCE (from block 0's bias weight) and reuses it for every layer; real UMT5 has its
  own `blocks.{i}.pos_embedding.embedding.weight` on **every** block (confirmed: "a real, genuine
  PER-LAYER relative position bias... unlike plain T5 which only has this on block 0"). Practical
  effect: `T5GpuWorkspace`'s single `RelPosBias` tensor becomes an array of 24 (one per layer,
  mirroring `WanGpuWeights.Blocks[]`'s per-layer pattern already established for the DiT side),
  each precomputed once per sequence length via the existing `ComputeRelPosBias` (CPU, cheap,
  `O(seq²×heads)`, unchanged) and uploaded once — not recomputed per encode call.
- **Difference 2 — different tensor names.** Confirmed directly against the real checkpoint (not
  assumed from HF conventions): `token_embedding.weight` (not `shared.weight`),
  `blocks.{i}.norm1`/`norm2.weight` (not `layer.0/1.layer_norm.weight`),
  `blocks.{i}.attn.{q,k,v,o}.weight` (not `layer.0.SelfAttention.{q,k,v,o}.weight`),
  `blocks.{i}.ffn.gate.0.weight`/`fc1.weight`/`fc2.weight` (not `layer.1.DenseReluDense.wi_0/wi_1/
  wo.weight`), `blocks.{i}.pos_embedding.embedding.weight` (per-layer, see above), final
  `norm.weight` (not `encoder.final_layer_norm.weight`). Use these real names directly — don't
  guess from `T5GpuWeights.cs`'s own naming.
- **Difference 3 — much larger vocab** (256384 vs 32128, UMT5 is multilingual) — irrelevant to the
  GPU work, only affects the one-time CPU token-embedding lookup (`token_embedding.weight`), which
  stays on CPU exactly like `T5Encoder.EncodeGpu`'s own step 1 already does (a single host-side
  lookup + one upload, not worth GPU residency).

## Recommended approach — copy `T5GpuWeights`/`T5GpuWorkspace`/`EncodeGpu` directly

1. **`UMT5GpuWeights` class** (new file, e.g. `src/OpenTail.Stingray.Diffusion/UMT5GpuWeights.cs`):
   copy `T5GpuWeights.cs` verbatim, then: rename tensor-name strings per Difference 2 above, and
   add a per-layer `PosEmbeddingWeight` (`blocks.{i}.pos_embedding.embedding.weight`, shape
   `[RelPosBuckets, Heads]` = `[32, 64]`, same as T5's own bias weight shape) to
   `T5LayerGpuWeights`'s per-layer analog.
2. **`UMT5GpuWorkspace` class** (new file): copy `T5GpuWorkspace.cs`, but make `RelPosBias` an
   array of 24 device tensors (one per layer) instead of one shared tensor — precompute each via
   the existing CPU `ComputeRelPosBias(rpW_layer_i, seqLen, Heads)` at `InitGpu` time (same timing
   as T5's own single precompute, just done 24× instead of once — cheap, `O(seq²×heads)` each,
   done once per sequence length, not per token).
3. **`InitGpu`/`EncodeGpu` methods on `UMT5Encoder`**: copy `T5Encoder.InitGpu`/`EncodeGpu`
   directly, adjusting: the token-embedding weight name (`token_embedding.weight`), per-layer
   tensor names (Difference 2), and passing `ws.RelPosBias[i]` (not a single shared tensor) into
   `T5MultiHeadAttentionRelBias` inside the per-layer loop.
4. **Verify correctness with a real, cheap parity test first** — mirror `T5GpuParityTests.cs`
   exactly: real UMT5-XXL weights (Wan's checkpoint is already used elsewhere in this session's
   Wan work, so it's on this machine), a small-but-real token count, compare `EncodeGpu`'s output
   against the existing CPU `Encode`'s output for the identical input. Use the same tolerance
   reasoning `T5GpuParityTests`/`FluxGpuVsCpuForwardBisectDebugTest` established (small, expected
   FP16-GPU-vs-FP32-CPU gap, not zero — measure the actual gap, don't assume the same `5e-2`/`1e-2`
   numbers transfer without checking).
5. **Wire it in wherever Wan's real end-to-end pipeline calls `UMT5Encoder.Encode`** — there isn't
   a full `WanImagePipeline`/`WanVideoPipeline` equivalent to FLUX's `ImagePipeline.cs` yet (per
   `docs/072`'s own note: "a full real end-to-end Wan2.1 20-step/2-frame generation has never been
   run on Vulkan"), so check whether this wiring happens in a test harness, a future pipeline
   class, or wherever the real UMT5 encode call site ends up living by the time this is picked up.
6. **Real benchmark once wired**: compare `EncodeGpu` against `Encode` at Wan's real prompt-length
   scale (check what sequence length Wan's own `WanPipeline._get_t5_prompt_embeds` uses — likely a
   fixed padded length, similar to FLUX's real T5-256 discovery in `docs/056`'s Round 9 — **do not
   assume Wan doesn't also need fixed-length padding; check the real diffusers `pipeline_wan.py`
   source directly, the same way the FLUX T5-padding bug was found**, rather than assuming
   token-count-only encoding is correct here too).
7. **Update `PerformanceLeague.md`** honestly with the real numbers once measured — following this
   session's established bar: report GPU-vs-CPU per-encode timing for real, don't assume a win
   just because T5-XXL and Wan's DiT both got real wins from similar work; measure UMT5
   specifically (its headDim=64 and 24-per-layer-bias-tensor pattern differ enough from both prior
   wins that GFLOP/s-per-shape could genuinely differ).

## Practical constraints (same as every prior handoff this session, repeated because they matter)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code** — `UMT5Encoder.Encode` (CPU) stays; this is a
  pure addition, exactly like `T5Encoder.Encode`/`EncodeGpu` coexist today.
- **Ask before committing** — recent work in this exact area (T5-XXL GPU residency, Wan2.1's
  GPU-path fix and kernel tuning) was committed directly with explicit user authorization each
  time; confirm current expectations before committing rather than assuming the same standing
  permission applies indefinitely.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) is healthy before a benchmark; a real incident earlier this
  session came from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Reuse `T5GpuParityTests.cs`'s exact test shape/pattern** for the new UMT5 parity test — it's a
  proven, fast (~16s), real-weights template; don't design a new one from scratch.

## Success criterion

Real, measured GPU-vs-CPU encode time for UMT5-XXL, following the same honesty bar every prior doc
this session established: if the GPU path is slower, say so and treat it as a Wan-DiT-shaped
"correct but not yet tuned" starting point (per `docs/072`'s own precedent), not a completed win.
Update `PerformanceLeague.md` only with what was actually measured.
