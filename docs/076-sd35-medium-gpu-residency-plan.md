# SD3.5-medium MMDiT GPU Residency Plan (2026-09-13)

## Context — read `docs/069` (FLUX) and `docs/072`/`docs/073` (Wan) first

Same transformation both `FluxDiT` and `WanModel` already went through (real, proven, committed:
`53839a3`, `0132aa7`, `d877353`): per-matmul CPU/GPU ping-pong → real GPU residency (persistent
VRAM-resident weights + workspace, a real per-block GPU forward pass). `src/OpenTail.Stingray.
Diffusion/SD3/MMDiTModel.cs` is confirmed (by grep) to still be in the "before" state — its
`_backend` field is only ever used via the old upload/`Sgemm`/download/free-per-call pattern (see
`MMDiTModel.cs` lines ~250-261), no `ForwardGpu`/`GpuWeights`/`GpuWorkspace` exist yet.

## Real architecture — confirmed from the file directly, two things genuinely differ from FLUX/Wan

- **`HeadDim = HiddenSize / NumHeads = 1536 / 24 = 64`, NOT 128.** This is the one thing to get
  right before reusing anything: FLUX, Wan, and Z-Image (per `docs/075`) all have `headDim=128`
  and can reuse `MultiHeadAttentionTiled128` (this session's fused, vec4-optimized attention
  kernel) directly. **SD3.5-medium cannot** — it needs the *other* existing tiled attention kernel,
  `MultiHeadAttentionTiled` (headDim=64), which is the SAME kernel SDXL/SD1.5 already use
  (confirmed pre-existing, NOT rewritten by this session's optimization pass — only the headDim=128
  variant got the 32×16 vec4 rewrite this session). Do not assume the headDim=128 kernel's recent
  speedups transfer here; if `MultiHeadAttentionTiled` (headDim=64) turns out to be a bottleneck,
  that's a separate, real tuning target, not something already solved by this session's FLUX/Wan
  work.
- **Absolute 2D sincos positional embedding, NOT RoPE.** `MMDiTModel.cs`'s `AddCroppedPosEmbed`
  loads a real, checkpoint-stored `pos_embed` tensor (`[1, PosEmbedMaxSize², HiddenSize]`,
  `PosEmbedMaxSize=384` derived from the tensor's own length) and adds a cropped slice of it
  directly to the image-token embeddings — a plain elementwise add, not a rotation. This is
  **simpler and lower-risk than FLUX/Wan's RoPE port** (no pairing-convention question to verify at
  all — see `docs/075`'s Z-Image RoPE caveat for why that question mattered elsewhere). Port this
  as a straightforward `AddInPlace`/`AddRowBroadcastInPlace`-style GPU op with the same cropped
  slice logic, not a new kernel.
- **AdaLN modulation uses the SAME shift+scale+gate convention as FLUX** (confirmed: `ModulateNorm`
  is called with explicit `modIdxOffset` chunk indices — `0`, `3`, `6` — into one modulation
  vector, the same 6-chunk-per-block-phase layout `AdaLNModulate`'s existing `shiftOffset`/
  `scaleOffset` overload and `ScaleGateAdd`'s `gateOffset` overload already support). **Unlike
  Z-Image (`docs/075`), SD3.5 needs NO new gate/scale pre-transform — the existing
  `AdaLNModulate`/`ScaleGateAdd` GPU kernels should be usable as-is.** Confirm the exact chunk
  count/order against `ModulateNorm`'s call sites (`MMDiTModel.cs` lines ~407-503) before wiring,
  but this is the most direct reuse of any of the four models this session has looked at.
- **One block type, not two.** Unlike FLUX (separate `DoubleBlockGpu`/`SingleBlockGpu` methods) or
  Z-Image (single-stream layers + separate refiner blocks), real SD3.5 MMDiT is **uniformly
  dual-stream across all `Depth=24` blocks**, with only the **last** block (`contextPreOnly = b ==
  Depth - 1`, confirmed at `MMDiTModel.cs` line ~395) skipping the text stream's own MLP/output
  (its output is discarded — a real, checkpoint-accurate detail, not a shortcut). A GPU port needs
  only **one** `JointBlockGpu`-style method with a `contextPreOnly` boolean, simpler than FLUX's
  two-method split.

  **2026-09-14 correction — real complexity is higher than this section implies.** Directly read
  the full per-block loop (`MMDiTModel.cs` lines ~380-498, not just the summary) before starting
  any implementation: this is NOT a simple `contextPreOnly`-branch-only structure.
  - **`dualAttn` is a SECOND, separate, per-model config bit** (detected via whether
    `{blk}.x_block.attn2.qkv.weight` exists), independent of `contextPreOnly`, changing the image
    modulation chunk count from 6 to 9 and adding a whole SECOND image-only (non-joint)
    self-attention pass (`attn2`, own qkv/proj/ln_q/ln_k weights, own gate at chunk offset 8) that
    runs AFTER the joint attention's residual and BEFORE the MLP. A GPU port needs to handle FOUR
    real combinations (`contextPreOnly` × `dualAttn`), not just two.
  - **Joint attention runs over ONE combined `[totalTokens, HiddenSize]` Q/K/V spanning BOTH
    streams concatenated** (`UnpackQkv` writes image tokens at offset 0 and text tokens at offset
    `numImgTokens` into the SAME `ws.Q`/`ws.K`/`ws.V` buffers, then one
    `JointMultiHeadAttention` call covers all `totalTokens`) — NOT two separate per-stream
    attention calls joined afterward. A GPU port's workspace needs one shared, `totalTokens`-sized
    Q/K/V buffer (not separate per-stream ones), and the existing `MultiHeadAttentionTiled`
    (headDim=64) kernel needs to run over this combined layout correctly.
  - **The dual-attention block's SECOND norm (`NormedImg2`, chunks 6/7) must be computed from the
    block's ORIGINAL pre-attention `x`, cached BEFORE the first attention's residual mutates `x`**
    (see the file's own extensive comment at lines 410-419 — this was a real, previously-fixed bug
    in the CPU port itself; a GPU port must preserve this exact ordering, not recompute it lazily
    after the joint-attention residual like it would be tempting to structure it).
  - **Per-head QK-RMSNorm is real and required** (`ln_q`/`ln_k`, `[headDim]`, applied per-head,
    no bias) — for BOTH `attn` and `attn2` (when `dualAttn`), on both img and txt streams for the
    joint `attn`. Four separate RMSNorm applications per block minimum, more with `dualAttn`.
  - **Net assessment**: this port is genuinely MORE complex than F5TTS's own GPU-residency work
    (docs/080, landed 2026-09-14) — more real branches, more weight tensors per block, a
    combined-buffer attention layout rather than two independent streams, and a real
    already-fixed-once CPU bug (the NormedImg2-timing one) whose fix must be preserved exactly
    under a differently-structured GPU dispatch. Budget for this accordingly — do not assume it is
    the "simple, most direct FLUX reuse" this doc originally characterized it as; scope a real
    GpuWeights class covering `attn`+`attn2`+both streams' weights, a workspace with a genuinely
    shared (not per-stream) Q/K/V buffer, and expect the four-combination branch logic to need its
    own careful parity testing per combination, not just one generic test.

  **2026-09-14, same check — also currently BLOCKED by disk space**: no SD3.5-medium checkpoint
  (`.safetensors`/`.gguf`) exists anywhere under `models/` on this machine right now (checked via a
  broad filename search). The real 656.9s CPU baseline cited below must have been measured when
  the checkpoint was previously present and since rotated out (per this project's own disk-space-
  constraint convention, `models/` holds a curated working set, not everything). The C: drive is
  at ~453MB free (see `docs/081` update #8's same finding for the Wan/sd-cli investigation) — not
  enough to download this checkpoint back right now. **This work is genuinely blocked until either
  disk space is freed or the checkpoint becomes available again** — not attempted further this
  session for that reason, not lack of scoping.

## Recommended approach

1. **`MMDiTGpuWeights` class** (new file): mirror `FluxGpuWeights.cs`/`WanGpuWeights.cs` — upload
   all 24 blocks' Q/K/V/O (both img and txt streams), norms, adaLN-modulation weights, and FFN
   weights once, resident in VRAM, same FP16-if-available `UploadWeight` pattern both prior
   classes use. Also upload the real `pos_embed` tensor once (it's invariant across every
   denoising step and every image at a given resolution — same invariant-caching logic already
   used for FLUX's RoPE tables and text context).
2. **`MMDiTGpuWorkspace` class** (new file): preallocated activation/normed/Q/K/V/attention-output/
   FFN buffers sized for the real image + text token counts, following `FluxGpuWorkspace`'s
   structure directly (SD3.5's shapes are closer to FLUX's own dual-stream-only structure than to
   Z-Image's refiner-block split).
3. **`JointBlockGpu`** (new method, mirroring `ApplyBlock`'s/`DoubleBlockGpu`'s existing
   structure): AdaLN modulate (reuse `AdaLNModulate` as-is, per the finding above) → Q/K/V via
   `Sgemm` for both streams → joint attention via `MultiHeadAttentionTiled` (headDim=64, NOT
   Tiled128) → O projection → gated residual (`ScaleGateAdd`, reuse as-is) → FFN sub-block
   analogously, with a `contextPreOnly` branch skipping the text stream's FFN/output on the last
   block.
4. **Batch per-block**, not per-op or per-step (`BeginBatch()`/`EndBatch()` per `JointBlockGpu`
   call) — the same granularity choice FLUX settled on and Wan's own kernel-tuning work confirmed
   matters (or doesn't, depending on the model — measure for SD3.5 specifically, don't assume
   either FLUX's or Wan's finding transfers without checking).
5. **Verify correctness with a real, cheap parity test first** — mirror
   `FluxGpuVsCpuForwardBisectDebugTest.cs`/`WanGpuParityTests.cs` exactly: real weights, a
   small-but-real token count, compare the GPU forward pass against the existing CPU forward pass
   for identical input, with a real numeric assertion (measure the actual FP16-GPU-vs-FP32-CPU
   gap, don't assume a specific tolerance number transfers from the other models without checking).
6. **Real end-to-end re-verification**: SD3.5-medium already has a real, working CPU reference —
   see `PerformanceLeague.md`'s SD3.5-medium row (656.9s, 256×256, 20 steps, seed 42, post the
   2026-09-05 correctness fixes — dual-attention norm input, VAE scale/shift, unpatchify channel
   order, missing positional embedding). Use the same prompt/resolution/seed/steps for the
   GPU-path re-verification and confirm the output still looks correct, not just that the run
   completes.
7. **Update `PerformanceLeague.md`** with real, measured before/after numbers — following the same
   honesty bar every prior doc this session established: report whatever the first real
   GPU-resident version actually measures, win or not, rather than assume a win because FLUX/Wan
   eventually got one.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code** — the CPU forward path stays; this is a pure
  addition, exactly like FLUX's/Wan's CPU and GPU paths coexist today.
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration**, not full end-to-end runs, until confident —
  a full real SD3.5-medium run currently costs ~657s (CPU), too slow to iterate against directly.

## Success criterion

Real, measured GPU-vs-CPU per-block (or per-forward) time for SD3.5-medium's MMDiT, reported
honestly regardless of which way it comes out — correctness (a real parity assertion, plus a real
visual re-check against the known-good 2026-09-05-fixed CPU output) matters more than a specific
speed number on the first pass. Update `PerformanceLeague.md` only with what was actually measured.
