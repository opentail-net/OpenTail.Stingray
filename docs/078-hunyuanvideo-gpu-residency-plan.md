# HunyuanVideo GPU Residency Plan (2026-09-13)

## Context — read `docs/069` (FLUX) and `docs/072`/`docs/073` (Wan) first

Same transformation FLUX and Wan already went through: per-matmul CPU/GPU ping-pong → real GPU
residency. `src/OpenTail.Stingray.Diffusion/HunyuanVideo/HunyuanVideoModel.cs` is architecturally
the closest of any model looked at so far to **FLUX itself** — same "Dual-Stream and Single-Stream
DiT" structure (confirmed by the class doc comment), same `headDim=128` (`dim=3072, numHeads=24`,
confirmed via `_headDim = _dim / _numHeads`).

## Important finding: the `_backend` field is dead code — do not assume any GPU work exists

`HunyuanVideoModel.cs` has an `IComputeBackend? _backend` field and a `_gpuWeights` dictionary
allocated in the constructor when a backend is passed — but **grep confirms zero other references
to `_backend` anywhere in the file.** The constructor accepts and stores it, then it is never read.
This is pure vestigial scaffolding, not a partial GPU implementation — **do not mistake this for
"GPU work already started here."** This is the same class of trap `docs/072` found in Wan's own
`ForwardGpu` (which looked GPU-resident but silently ran CPU compute) — this one is actually
simpler/safer (it's just unused, not silently wrong), but still worth being explicit about so
nobody wastes time assuming partial credit exists. HunyuanVideo is, in practice, **100% CPU**,
starting from the same position FLUX was in before any of its GPU work (no `MatQ`-style dispatch
even, unlike SD3.5/Z-Image which at least had that).

## Real architecture — confirmed from the file directly, one genuinely new kernel needed

- **`headDim=128`**, same as FLUX/Wan/Z-Image — `MultiHeadAttentionTiled128` (this session's fused,
  vec4-optimized attention kernel) is directly reusable for the attention math itself.
- **3D RoPE, confirmed SPLIT-HALF (NEOX-style) pairing, with FLUX's exact 16/56/56 axis split** —
  `HunyuanVideoRoPE.Compute3DRoPE` (`src/OpenTail.Stingray.Diffusion/HunyuanVideo/
  HunyuanVideoRoPE.cs`) calls `SplitHalfRoPE.FillFrequencies`/`SplitHalfRoPE.ApplyRoPE` explicitly
  — the OTHER pairing convention from FLUX's own `Flux2DRoPE`/`Compute3DRoPECompact` (which is
  interleaved). **Neither of the two existing GPU RoPE kernels is a correct drop-in reuse here**:
  - `Flux2DRoPE` (GPU): correct axis split (16/56/56) but WRONG pairing (interleaved, not
    split-half).
  - `RoPE3D` (GPU, the one already found to be wrong for Wan in `docs/072`): correct pairing
    (split-half) but WRONG axis split (equal `headDim/6` per axis-half, not 16/56/56).
  - **This means a genuinely new GPU kernel is needed** — a split-half rotation with the real
    16/56/56 (unequal) axis split, i.e. take `RoPE3D`'s pairing math but replace its
    `bandSize = headDim/6` equal-split assumption with FLUX's real 16/56/56 axis boundaries (own
    band per axis: `[0,16)`, `[16,72)`, `[72,128)`, split-half within each band rather than
    interleaved-pair). Also note: `theta=256.0` here (`HunyuanVideoRoPE.Compute3DRoPE`'s own
    default), NOT FLUX's `10000.0` — confirm this is really used at inference time (check
    `Forward`'s actual call site) before assuming the default value in the method signature is
    what's live.
- **Text conditioning dimension `TextDim=4096`** ("LLaMA-3 / Qwen2.5-VL text dimension", per the
  class doc comment) — a real LLM-based text encoder, not T5. Check whether that encoder itself is
  already wired/real or still deferred (mirroring LTX-Video's own T5 deferral, `docs/077`) before
  assuming full end-to-end GPU residency is meaningful yet — a DiT-only GPU port is still real,
  useful work even if the text encoder side is CPU-only or stubbed, same as this session's other
  docs treat DiT and text-encoder GPU work as separate, sequential efforts (`docs/071`/`docs/074`
  did T5-XXL/UMT5 as follow-ups to their respective DiTs, not prerequisites).
- **Dual-stream (`_depthDouble`) + single-stream (`_depthSingle`) blocks**, same two-method shape
  as `FluxDiT.DoubleBlockGpu`/`SingleBlockGpu` — confirm `_depthSingle`'s real detected value (the
  constructor's own default is `depthSingle=0`, but `DetectConfig` reads the real value from the
  checkpoint — don't assume single-stream blocks are actually present without checking the real
  detected count).

## Recommended approach

1. **Write the new split-half + unequal-axis-split RoPE GPU kernel first** (see above) — this is
   the one piece of real, new kernel work here, everything else reuses proven infrastructure.
   Verify it against `SplitHalfRoPE.ApplyRoPE`'s CPU output with a small, cheap unit test before
   wiring it into the block loop (mirror how `docs/072` verified `Flux2DRoPE`'s reuse for Wan with
   a real numeric comparison, not just code-reading confidence).
2. **`HunyuanVideoGpuWeights`/`HunyuanVideoGpuWorkspace` classes** (new files): mirror
   `FluxGpuWeights.cs`/`FluxGpuWorkspace.cs` directly — same dual/single-stream block shape, same
   FP16-if-available upload pattern.
3. **`DoubleBlockGpu`/`SingleBlockGpu`** (new methods, mirroring `FluxDiT`'s existing methods of
   the same name almost exactly, given the architectural similarity): Q/K/V via `Sgemm`, the new
   RoPE kernel from step 1, attention via `MultiHeadAttentionTiled128`, O projection, gated
   residual (`ScaleGateAdd`), FFN. Confirm AdaLN modulation convention (shift+scale+gate, likely
   the same as FLUX given the shared architecture family, but verify against the real
   `ApplyBlock`/block-forward method rather than assuming).
4. **Batch per-block**, not per-op or per-step (`BeginBatch()`/`EndBatch()` per block call).
5. **Verify correctness with a real, cheap parity test first** — mirror
   `FluxGpuVsCpuForwardBisectDebugTest.cs`'s pattern: real weights, small-but-real token count,
   compare GPU forward pass against CPU `Forward` for identical input, real numeric assertion.
6. **Real end-to-end re-verification** — check whether HunyuanVideo has an existing known-good
   reference output anywhere (no `PerformanceLeague.md` row currently exists for it, unlike every
   other model in this doc series — confirm current correctness/coverage status directly, e.g. via
   `docs/00-current-work.md` or `docs/audio-review-progress.md`-style tracking docs, before
   assuming a CPU baseline to compare against is even established).
7. **Update `PerformanceLeague.md`** with real, measured before/after numbers, following the same
   honesty bar every prior doc this session established.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code.**
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration**, not full end-to-end runs, until confident.

## Success criterion

Real, measured GPU-vs-CPU per-block (or per-forward) time for HunyuanVideo's DiT, reported
honestly regardless of which way it comes out. Given no existing `PerformanceLeague.md` entry or
confirmed-correct reference output was found for this model during this doc's own research,
**establishing a real, correct CPU baseline and reference output may itself be a necessary first
step** before GPU-vs-CPU parity is even meaningful — check this explicitly before assuming one
already exists.
