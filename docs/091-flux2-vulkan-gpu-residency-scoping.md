# 091 — FLUX.2 Vulkan GPU-residency scoping (NOT started, planning only)

**Status: scoping document, no implementation done.** This is a plan for a future task, written
after CPU performance work landed (`docs/090-flux2-cpu-perf-handoff.md`) and before deciding
whether/when to actually build it. Do not start implementation from this doc alone without
re-checking the gating conditions below first — they may have changed.

## Gating conditions — check these before starting

1. **CPU correctness must be settled first.** As of this doc's writing, FLUX.2's real output is a
   structured periodic grid/tiling pattern, not yet coherent (a timestep-blindness bug was found
   and fixed this session, `docs/087`/`docs/088`; a second, narrower structural bug — likely
   patchify/token-ordering — is suspected but not yet found). **Porting broken math to GPU produces
   a faster wrong answer and doubles the eventual debugging cost** (CLAUDE.md rule 7, and this
   project's own established discipline — see the rule's own citation of past mistakes). Do not
   start this task until a real end-to-end FLUX.2 run produces genuinely coherent output on CPU.

2. **CPU performance must be re-measured after `docs/090`'s fix lands and is verified.** If the
   `QuantizedWeightCache` integration (already landed as of this doc's writing, not yet fully
   verified at production resolution) gets CPU generation into a reasonable time budget (see
   `docs/090` for context), GPU work may turn out to be lower priority than expected — measure
   first, don't assume GPU is needed just because it wasn't fast before the CPU fix.

3. **This machine's GPU is an integrated Radeon (Ryzen 5700G), not a discrete GPU.** CLAUDE.md rule
   13 documents a real, measured case (MiniMax-Music3's DiT) where this exact iGPU was **2.5x
   SLOWER than CPU** for a real workload, due to per-call dispatch overhead and shared-memory
   bandwidth with no VRAM advantage over the CPU it shares memory with. **Do not assume GPU
   residency will help FLUX.2 without a real, measured comparison on THIS hardware** — a large
   model with large per-call payloads (FLUX.2's biggest linear is ~340M elements) is a more
   favorable case for GPU dispatch-overhead amortization than MiniMax-Music3's smaller ops was, but
   this must be verified, not assumed. If it turns out slower, that is a real, useful finding about
   THIS machine, not evidence the GPU code path itself is wrong (CLAUDE.md rule 13's own framing).

## Why this task, if/when it happens, should look like FLUX.1's GPU port, not a fresh design

FLUX.2 is BFL's newer release in the exact same architectural lineage as FLUX.1 (confirmed in-repo
via `examples/flux`/`examples/flux2`): both use double-stream blocks + single-stream blocks with
shared (not per-block) modulation, RoPE-after-norm joint attention, and a fused single-block
`linear1`/`linear2`. FLUX.1 already has a complete, working, measured Vulkan GPU-residency port:

- `src/OpenTail.Stingray.Diffusion/FluxGpuWeights.cs` — one-time FP16 weight upload/residency,
  structured as `DoubleBlockGpuWeights[]`/`SingleBlockGpuWeights[]` arrays, mirroring the model's
  own block structure (see its `DoubleBlockGpuWeights` nested class for the exact per-block tensor
  set: QKV, QK-norm scales, attn-proj, MLP weights, modulation weights — FLUX.2 needs the analogous
  set, adjusted for FLUX.2's different dimensions (`HiddenSize=6144` vs FLUX.1's, `MlpRatio=3.0`
  SiLU-gated vs FLUX.1's GEGLU, 4-axis RoPE vs FLUX.1's 3-axis, shared single-vs-double modulation
  already documented in `docs/087`).
- `src/OpenTail.Stingray.Diffusion/FluxGpuWorkspace.cs` — the per-step GPU scratch-buffer
  workspace (intermediate activations), reused across steps to avoid repeated allocation.
- `PerformanceLeague.md`'s FLUX.1-schnell row documents the real, measured progression this port
  went through (298.5s → 198.1s warm, `M=1` matrix-vector fast-path dispatch, 2-block chunked
  command-buffer batching, pipelined concat/slice/final-layer, chunked T5-XXL text-encoder
  batching) — read this for the actual sequence of real, measured wins, not just the end state, so
  the FLUX.2 port can go through the same stages instead of guessing which optimization matters
  most.

**Port `FluxGpuWeights.cs`/`FluxGpuWorkspace.cs`'s pattern directly, adjusted for FLUX.2's real
dimensions and gated-FFN/4-axis-RoPE differences (already fully documented in `docs/087`'s
architecture derivation) — do not design a new GPU-residency pattern from scratch.**

## What's specifically different for FLUX.2 vs. the FLUX.1 GPU port

- **SiLU-gated FFN, not GEGLU**: FLUX.2's MLP is `SiLU(gate) * value` on a `2*mlpHidden`-wide
  up-projection split in half (confirmed against `examples/flux2/src/flux2/model.py`'s
  `SiLUActivation`, see `docs/087`) — FLUX.1's GPU MLP kernel almost certainly assumes GEGLU or a
  different gate convention; do not reuse it unmodified, verify the activation math specifically.
- **4-axis RoPE (t,h,w,l), not 3-axis**: `Flux2RoPE.cs`'s CPU implementation already generalizes
  `InterleavedRoPE` to N axes; the GPU RoPE kernel (if FLUX.1's is hardcoded to 3 axes) needs the
  same generalization, or a FLUX.2-specific variant.
- **A much larger text encoder (Mistral-24B, not FLUX.1's CLIP-L+T5-XXL combo)**: text conditioning
  for FLUX.2 (`Flux2TextConditioning.Encode`) runs through the shared `Engine.ForwardPass`/
  `EnableHiddenTaps` mechanism, not a diffusion-specific text encoder GPU path — whether this needs
  its own GPU work is a separate question from the DiT's own GPU residency, and should be profiled
  independently (it may already be the dominant cost given its size — measure before assuming the
  DiT is the bottleneck).
- **No reference-image conditioning support yet on CPU** (`Flux2DiT.Forward` throws
  `NotSupportedException` for reference images, a documented gap in `docs/087`) — the GPU port
  should match whatever the CPU path supports at the time it's built, not attempt to add ref-image
  support as part of a GPU-residency task (that's a separate, unrelated correctness task).

## Suggested implementation order, if/when this is picked up

1. Re-verify the CPU path is both correct (coherent output) and reasonably fast (per the gating
   conditions above) — get fresh real numbers, don't reuse pre-`docs/090`-fix numbers.
2. Profile where CPU time actually goes at that point (DiT forward vs. Mistral text encoding vs.
   VAE decode) — this determines whether DiT GPU residency is even the right next lever, or whether
   the 24B text encoder dominates and deserves attention first.
3. If DiT GPU residency is warranted: build `Flux2GpuWeights.cs`/`Flux2GpuWorkspace.cs` following
   `FluxGpuWeights.cs`/`FluxGpuWorkspace.cs`'s structure, adjusted per the differences above.
4. Get a real, measured single-step GPU-vs-CPU comparison on THIS machine's iGPU before committing
   further — per the gating conditions, this could go either way on this hardware.
5. If GPU wins: follow FLUX.1's own measured optimization sequence
   (`PerformanceLeague.md`'s FLUX.1-schnell rows) — matrix-vector fast paths, command-buffer
   batching, pipelining — rather than re-deriving which optimizations matter from scratch.
6. Record every real measurement in `PerformanceLeague.md` and `docs/088`, following this project's
   existing documentation conventions (see any other model's entries for the expected format).
