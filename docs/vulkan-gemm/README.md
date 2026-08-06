# Vulkan quantized matmul — Path 1 / Path 2 scaffold

The same shape as `docs/repack-gemm/` on the CPU side: an incumbent (Path 1), a reserved slot for a
faster future kernel (Path 2), a runtime switch between them, and counters so the choice is decided
by measurement rather than argument.

**Path 2 does not exist.** This document exists so the decision to write it — or to delete the
scaffold and walk away — can be made from numbers that are available today.

## Why

Every quantized matmul shader in `Shaders.cs` is a `MatVec*`. The five `Sgemm*` shaders are dense
float, serving the diffusion path. So the quantized matmul is matrix-**vector** with register
blocking over tokens: one workgroup per 8 output rows, `acc[MAX_NTOK]` accumulators in VGPRs, a
weight word reused across up to `MaxBatchVerifyK` = 16 tokens. Above 16 tokens `GpuForwardPass`
chunks, and every chunk re-streams the whole model.

This is structurally what the CPU prefill was before Path 2 landed there. The CPU fix was not a
wider register tile — it was another level of the memory hierarchy, a repacked tile that stayed
resident while activation columns streamed past. The GPU equivalent is a weight tile in shared
memory: what llama.cpp calls `mul_mm` as distinct from `mul_mv`.

## Using it

| variable | effect |
|---|---|
| `STINGRAY_VULKAN_MM_PATH` | `1` (default) or `2`. Path 2 declines every shape today and falls through to Path 1. |
| `STINGRAY_VULKAN_MM_STATS` | `1` to count dispatches and weight bytes; off by default so an unmeasured run is byte-identical to before. |

```
STINGRAY_VULKAN_MM_STATS=1 opentail-llm-cli -m model.gguf --backend vulkan -g -1 -f prompt.txt
```

Code: `src/OpenTail.Stingray.Vulkan/VulkanMatMulPath.cs` (switch + counters), the seam and
`TryMatMulBatchedPath2` stub in `VulkanBackend.MatMulBatched`, tests in
`tests/.../VulkanMatMulPathTests.cs` (27, no GPU required).

## Baseline, measured

AMD Radeon integrated, Vulkan 1.3, SmolLM2-1.7B-Instruct-Q4_K_M (1006.7 MiB), 931-token prompt:

```
[GpuForwardPass] prefill N=931 chunk=16
[VulkanMatMul] path=Path1
  dispatches : path1=9914 path2=0 declines=0 fallback=0
  amortization: 15.78 tokens/dispatch (156416 token-dispatches over 9914 dispatches)
  weight I/O : 54626.6 MiB total, 58.7 MiB/token over 931 tokens
Prefill: 931 tokens, 74.5 t/s
```

Three cross-checks that the counters measure what they claim:

- 9914 dispatches / 59 chunks ≈ 168 = 24 layers × 7 trunk matmuls. Structure matches the model.
- 54626.6 MiB / 59 chunks = **925.9 MiB per chunk** against a **1006.7 MiB** model file. The
  difference is embeddings and output, which are not trunk matmuls. Weight I/O is real.
- `fallback=0` — no dtype fell to the per-token loop on this model.

Setting `STINGRAY_VULKAN_MM_PATH=2` reports `declines=9914`, identical weight I/O, and 74.7 t/s
against 74.5. The seam costs nothing and cannot change an answer.

## The prize is 1.54x, not 59x — measured

53.3 GiB streamed to prefill 931 tokens of a 1 GiB model looks like a 59x redundancy, and that was
the number this scaffold was built to chase. It is the wrong number, and the chunk-size split makes
the right one measurable without writing a single line of GLSL.

Chunk size is inversely proportional to how many times prefill streams the weights, so sweeping it
is a direct experiment on how much of prefill time is weight bandwidth. Same prompt, same model,
`STINGRAY_VULKAN_PREFILL_CHUNK`:

| chunk | prefill t/s | weight GiB | time (s) |
|---:|---:|---:|---:|
| 1 | 12.2 | 841.8 | 76.31 |
| 2 | 22.7 | 421.3 | 41.01 |
| 4 | 37.9 | 210.7 | 24.56 |
| 8 | 58.4 | 105.8 | 15.94 |
| 16 | 75.3 | 53.3 | 12.36 |

Two-point fit on the extremes, checked against the three interior points (all within 4.3%):

```
time(s) ≈ weight_GiB / 12.33 + 8.04
```

Weight streaming costs **12.33 GiB/s marginal**, and there is a **fixed 8.04 s** that no amount of
weight amortization touches. At the default chunk of 16 that splits as **4.3 s of weights against
8.0 s of everything else** — so weights are 35% of prefill, not the whole of it.

Therefore, with weight traffic driven to zero:

| chunk | projected t/s | vs chunk 16 |
|---:|---:|---:|
| 32 | 91 | 1.21x |
| 64 | 102 | 1.36x |
| 128 | 108 | 1.44x |
| ∞ (perfect GEMM) | **116** | **1.54x** |

**1.54x is the ceiling**, assuming a tiled kernel adds no cost of its own and leaves the fixed 8.04 s
untouched. Most of it arrives by chunk 32–64; the tail is worth little. A real kernel lands below
this.

### What the fixed 8.04 s is, and why it matters more

Activation traffic is `(rows / N_ROWS) × nTok × cols × 4` bytes per dispatch — each workgroup covers
8 rows and re-reads the whole activation block. The dispatch count falls as `1/chunk` while per-
dispatch activation traffic rises linearly with it, so **total activation traffic is chunk-
independent**, which is exactly why it sits in the fixed term and not the slope. That it fits the
constant so cleanly is corroboration, not coincidence.

So a Path 2 that only widens the tile chases the smaller half. A Path 2 that stages activations in
shared memory attacks the larger one. Any tiled kernel written here should be designed against the
8.04 s, not the 53.3 GiB.

## Path 2 exists: tiled Q4_K GEMM, +4.6%

`Shaders.MatMulTiledQ4K` — BM=64 rows x BN=16 tokens, BK=64, TM=TN=2, both operands staged in LDS
with llama.cpp's `+1` bank-conflict padding. BK=64 is one Q4_K "c-chunk" (8 dwords whose low nibbles
are sub-block 2c and high nibbles 2c+1), so the dequant is structurally identical to the incumbent's
and the two differ in one variable only.

| | Path 1 | Path 2 |
|---|---|---|
| prefill, 931 tok | 74.5 t/s | **77.9 t/s** (1.046x) |
| dispatches served | 9914 | 8496 (+1418 Q6_K declines → Path 1) |
| weight I/O | 54626.6 MiB | 54626.6 MiB — *unchanged* |

Weight traffic is identical, so the gain is purely structural: no per-token shared-memory tree
reduction (Path 1 runs 16 of them, 7 barriers each, per workgroup), and each staged value is reused
across the tile instead of re-read per lane. Full suite passes on **both** paths, 1261/1261.

### Why it is only 4.6% of a possible 54%

BN is still 16, so the tile amortizes exactly as many tokens as Path 1 did and streams the weights
the same 59 times. **The prize needs BN > 16, and that is now a `#define` rather than a register
array** — the `acc[MAX_NTOK]`-per-VGPR ceiling that capped Path 1 does not exist here.

The blocker is Q6_K. `ffn_down` is Q6_K (1418 of 9914 dispatches, 1 per layer), it declines to Path 1,
and Path 1 throws above nTok=16. So raising the chunk requires a **tiled Q6_K shader first** — that
is the next commit, and it unlocks chunk 64 for a projected 1.36x.

### Numerics: Path 2 is not a drop-in

Path 2 carries one running sum per output across K; Path 1 forms per-lane partials then a tree
reduction. Same additions, different association. Measured **9.4e-3 max abs logit delta against the
per-token path, versus 5e-3 for Path 1** — it broke `BatchedPrefill_ShortPromptBelowOneChunk_
MatchesPerToken` on first run.

The tolerance is now path-aware rather than loosened, so Path 1's tighter guarantee is preserved and
Path 2's looser one is stated with its reason. Greedy argmax matches on both.

**Path 2 must not become the default on a 4.6% speed win alone.** A delta at this scale is inside the
noise that flipped 2 of 6 short-prompt argmaxes when int8 activations were tried. It needs the same
perplexity gate the CPU's Path 2 cleared (wikitext-2, 8191 scored tokens), which has not been run for
this kernel.

## Toolchain: unblocked

`glslc` built from source at `examples/glslc/shaderc` — **5,568,512 bytes**, shaderc v2026.4-dev /
glslang 11.1.0-1516 / spirv-tools v2026.3.

**Regenerating the whole SPIR-V table drifts ~33 of 89 shaders** against the committed bytes, because
that toolchain is much newer than whatever produced them. The regenerated table passes all 1261 tests
and is performance-neutral (74.9/75.5 vs 74.5/75.3 t/s) — but that is one AMD iGPU, so it is not
evidence for other drivers, and it was restored rather than adopted.

New shaders therefore splice in instead: bump `Count`, add one hash-keyed `case` line, append one
`_sNN` array. The switch does not care about ordering. `VulkanPrecompiledShaderTests` confirms the
other 89 entries stay byte-identical.

## Previously: blocked on a toolchain, not on effort

**This machine has no Vulkan SDK and no `glslc`.** `ShaderCompiler` serves a committed SPIR-V table
and shells out to `glslc` only on a table miss, and `tools/SpirvGen` needs the SDK to regenerate it —
so a new or edited shader cannot be compiled here at all, and `VulkanPrecompiledShaderTests` fails on
drift by design. No Path 2 kernel was written, because an unverifiable one is worse than none.

This is the same shape as the CUDA finding in `gpu-review-log.md`: a real defect, a cheap fix,
blocked on a toolchain rather than hardware. **Installing the Vulkan SDK unblocks it** — no different
GPU required.

The cheapest first increment, when unblocked: `MatVecBatchedQ4K` and `MatVecBatchedQ6K` with
`MAX_NTOK` at 32, gated behind Path 2, `Path2MaxTokensPerDispatch` raised to match, and the `nTok`
range check in `MatMulBatched` relaxed. Projected 1.21x, and it tests the register-pressure question
the existing shader comment leaves open ("32 was not tried") for a few lines of change. Note
`Path2Cap_CannotExceed_WhatPath1CanFallBackTo` will fail until the Path 1 shaders are widened too —
that is deliberate: Path 2 may decline any shape, and a decline falls through to a Path 1 that would
otherwise throw on the wider chunk.

**To drop the whole thing:** delete `VulkanMatMulPath.cs`, the seam and stub in `VulkanBackend.cs`,
the report block in `GpuForwardPass.Prefill`, the two test files, and three entries in
`KnownEnvironmentVariables`. The `MaxBatchVerifyK` / `MaxPrefillChunk` split is worth keeping
regardless — it is what made the sweep above possible.

## Negative result: the metric was wrong first

The first version divided weight bytes by the summed `nTok` and reported **357.6 KiB/token**. The
true figure is **58.7 MiB/token** — off by 168x, the number of matmuls in the model, because summed
`nTok` counts token-*dispatches* and not tokens.

It was caught by running the counters on hardware and checking the total against an independently
computed model size, before any conclusion was drawn from them. `WeightBytesPerToken` now requires
the caller to pass the token count, `Report(0)` omits the line rather than fabricating it, and
`BytesPerToken_IsNotDerivableFromTokenDispatches` pins the distinction.

The headline metric was changed to **tokens/dispatch** for the same reason: it needs no external
input, so it cannot be mis-attributed this way at all.
