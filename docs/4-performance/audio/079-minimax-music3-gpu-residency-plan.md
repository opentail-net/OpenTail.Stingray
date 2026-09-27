# MiniMax-Music3 Transformer GPU Residency Plan (2026-09-13)

## Context — read `docs/069` (FLUX) and `docs/072`/`docs/073` (Wan) first

Same transformation FLUX and Wan already went through: per-matmul CPU/GPU ping-pong → real GPU
residency. `src/OpenTail.Stingray.Diffusion/MiniMaxMusic3/MiniMaxMusic3Transformer.cs` already has
**some** GPU scaffolding — `MiniMaxMusic3GpuTransformerWeights.cs` exists and
`GetOrCreateGpuWeights` is called from the real `Forward` method (confirmed at lines ~150, ~342) —
but this is the same `MatQ`-style per-matmul dispatch pattern FLUX/SD3.5/Z-Image all had before
their own residency work, **not** a real fused per-block GPU forward pass. There is no
`ForwardGpu`/`TransformerBlockGpu` method, and no persistent GPU workspace — every matmul still
individually uploads/dispatches/downloads/frees against the resident weights. This scope is
**narrower** than earlier docs in this series: MiniMax-Music3 is a large, multi-component pipeline
(condition encoder, this flow-matching transformer, an autoregressive RVQ depth decoder with its
own KV cache, a vocoder) — this doc covers **only the flow-matching `Transformer` component**, the
one already closest to residency-ready.

## Real architecture — confirmed from the file and `MiniMaxMusic3Config.cs` directly

- **`headDim=64`** (`TransformerAttentionHeadDim=64`, `TransformerNumAttentionHeads=32`,
  `MiniMaxMusic3Config.cs`) — same as SD3.5-medium (`docs/076`) and LTX-Video (`docs/077`): use
  `MultiHeadAttentionTiled` (headDim=64), NOT `MultiHeadAttentionTiled128`.
- **In-context conditioning, NOT cross-attention.** This is architecturally different from every
  other model in this doc series. `Forward`'s own code builds one flat per-token concatenated
  buffer — `[latent(128) | zeros(128) | condition(2048)]` = 2304 channels per token — fed through
  the transformer as a single sequence, rather than a separate cross-attention sub-layer attending
  to a text/condition K/V (FLUX/Wan/HunyuanVideo's approach). **There is no separate cross-attn
  block to port here** — the "condition" is baked directly into the input sequence, so a GPU port
  is architecturally simpler in one sense (no separate KV-cache-against-condition step needed
  inside the transformer block itself; `condition` must already be precomputed and resampled onto
  the latent timeline before this call, per the method's own doc comment) — self-attention +
  FFN only, applied to the concatenated sequence.
- **`TransformerConditionDim=2048`, `TransformerFfInnerDim=8192`** — real dims to use when sizing
  a GPU workspace; confirm the real block count and any AdaLN/modulation convention by reading the
  rest of `Forward`'s body directly (this doc's own research did not trace every block-internal
  op) before assuming it matches FLUX's shift+scale+gate pattern or any other model's convention —
  check, don't assume, per this project's own established discipline.
- **This transformer is one piece of a larger pipeline**: `MiniMaxMusic3ConditionEncoder.cs` (the
  real source of the `condition` array), `MiniMaxMusic3RvqDepthDecoder.cs` +
  `MiniMaxMusic3RvqDepthKvCache.cs`/`MiniMaxMusic3RvqDepthWorkspace.cs` (a separate autoregressive
  decoder, already has its own KV-cache/workspace infrastructure — worth checking whether it's
  further along toward residency than the flow transformer itself, since those class names suggest
  real workspace/cache design already exists there too), `MiniMaxMusic3GlobalModel.cs` +
  `MiniMaxMusic3GlobalKvCache.cs`, and `MiniMaxMusic3Vocoder.cs`. **This doc only covers the flow
  transformer** (`MiniMaxMusic3Transformer.Forward`) — the RVQ depth decoder and global model may
  warrant their own separate follow-up docs once their own current GPU-residency status is
  checked (not done as part of this doc's research — start there if picking up more than just the
  flow transformer).

## Known status — real, cited baseline exists, but no Vulkan number yet

Per `PerformanceLeague.md`'s MiniMax-Music3 row: real end-to-end CPU run, `3352.9s (~56 min)`,
200-frame/~8s audio generation, post a real frame-index off-by-one fix (documented, no known open
correctness bug — unlike LTX-Video, `docs/077`, this one has a citation trail suggesting the CPU
path is trustworthy). **A real C++ (`stable-diffusion.cpp`-family `minimax_music3` /
`examples/audio.cpp`) comparison was attempted and explicitly deprioritized** (per the same
`PerformanceLeague.md` row's own note): the upstream `audio-cpp/MiniMax-Music3-GGUF` repo needed a
real additional multi-GB download of separately-split GGUF weight files, on top of the run itself
costing ~56 minutes, and this was judged not worth the cost that session. **No Vulkan GPU timing
exists for this model at all yet** — this doc's job is to get a real, honest first GPU number, the
same "before" state FLUX/Wan/Z-Image/SD3.5/LTX-Video/HunyuanVideo all started from in their own
docs.

## Recommended approach

1. **Read the rest of `Forward`'s body** (`MiniMaxMusic3Transformer.cs`) to get the real per-block
   structure (norm/modulation convention, residual/gate wiring, FFN activation) before writing any
   GPU code — this doc's own research stopped at the input-concatenation step and headDim; don't
   assume the rest matches another model without checking.
2. **`MiniMaxMusic3GpuTransformerWorkspace` class** (new file, since only the weights side
   currently has GPU scaffolding): preallocated activation/Q/K/V/attention-output/FFN buffers
   sized for the real `seqLen = length + 1` (per the existing code's own `seqLen` variable) and
   `concatChannels=2304`, following `FluxGpuWorkspace.cs`'s structure.
3. **`ForwardGpu`/`TransformerBlockGpu`** (new methods): self-attention via `Sgemm` +
   `MultiHeadAttentionTiled` (headDim=64) + O-projection + residual, FFN analogously — reusing
   `MiniMaxMusic3GpuTransformerWeights`'s already-uploaded weights (confirm its own field names/
   structure directly before assuming they match `FluxGpuWeights`'s naming).
4. **Batch per-block**, not per-op (`BeginBatch()`/`EndBatch()` per block call).
5. **Verify correctness with a real, cheap parity test first** — mirror
   `FluxGpuVsCpuForwardBisectDebugTest.cs`'s pattern: real weights, a small-but-real
   `length`/token count, compare the GPU forward pass against the existing CPU `Forward` (backend:
   null) for identical `latent`/`condition`/`timestep` input, real numeric assertion.
6. **Real end-to-end re-verification**: use the existing known-good reference —
   `docs/diffusion-samples/minimax_music3_v1_folk_verse_200frames.wav` (per `PerformanceLeague.md`,
   generated post the frame-index fix) — confirm the GPU path's output is audibly consistent with
   the CPU path's own known output (this doc's own note: the existing sample "not yet judged by ear
   by any session" — if this is still true when picked up, that's worth doing regardless of the
   GPU work, as a real independent correctness check).
7. **Update `PerformanceLeague.md`** with real, measured before/after numbers, following the same
   honesty bar every prior doc this session established — this would be the **first real Vulkan
   timing for this model**, so there's no prior GPU number to compare against, only the 3352.9s CPU
   baseline.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code.**
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration** — a full real run currently costs ~56 minutes
  (CPU), far too slow to iterate against directly.

## Success criterion

A real, measured first Vulkan GPU timing for this transformer (whatever it turns out to be,
faster or slower than CPU — report honestly either way, per this session's established bar), plus
a real correctness check (numeric parity test, and ideally the by-ear confirmation of the existing
audio sample this doc flagged as still outstanding). Update `PerformanceLeague.md` only with what
was actually measured.
