# CPU architecture and kernel opportunities

## Purpose

This note records the concrete CPU fast-path gaps that remain after the Q4_K repacked-GEMM and
Flash64 performance wave. It deliberately distinguishes:

- **Confirmed implementation gaps**: visible in the current dispatch code or directly comparable
  with an implementation in the local llama.cpp source.
- **Measured gaps**: backed by an existing isolated or end-to-end measurement.
- **Expected gains**: plausible consequences of filling a confirmed gap, but not performance facts
  until measured on representative hardware and models.

The completed Q4_K/head-dimension-64 work should remain closed. These opportunities are better
treated as a separate **CPU architecture coverage** programme: they extend fast paths to other
hardware, quantization formats and model architectures rather than continuing to tune the already
optimized reference case.

## Summary and suggested priority

| Priority | Confirmed missing path | Main beneficiaries | Evidence strength |
|---:|---|---|---|
| 1 | Flash attention for `headDim=128` (and possibly 256) | Common dense Llama/Qwen/Gemma-family shapes | Dispatch gap confirmed; gain unmeasured |
| 2 | Faster Q6_K AVX2 dot path | Mixed Q4_K_M models | Dispatch/coverage gap plus measured 1.68x isolated gap |
| 3 | Native x86 kernels for Q4_0, IQ4_NL, MXFP4 and related formats | Models stored in those GGUF formats | Scalar fallback confirmed; llama.cpp kernels exist |
| 4 | Batched prefill with per-layer head dimensions | Gemma 4 and similar architectures | Whole batched path explicitly disabled |
| 5 | ARM64 NEON/dot-product/i8mm kernel suite | Apple Silicon, Snapdragon, Graviton and ARM servers | No OpenTail ARM intrinsics; llama.cpp implementations exist |
| 6 | Batched CPU MoE prefill | CPU-only MoE models | Generic CPU path explicitly sequentializes prefill |
| 7 | Exact Q3_K/Q2_K multi-input kernels | Low-bit MTP/speculative verification | Multi-input dispatch cases are absent |

No one item benefits literally every CPU, model and prompt. The broadest opportunities each cover a
large class: `headDim=128` models, ARM64 systems, or models using a currently-unspecialized GGUF
format. Work should be selected using an available representative model and hardware, not merely by
the size of the theoretical instruction-count reduction.

## 1. ARM64 has no optimized OpenTail CPU kernel family

### Confirmed state

`src/OpenTail.Stingray.Cpu/SimdKernels.cs` is x86-oriented and imports
`System.Runtime.Intrinsics.X86`. There are no uses of `AdvSimd`, the ARM64 intrinsics namespace,
NEON, ARM dot-product instructions or i8mm anywhere in `src/OpenTail.Stingray.Cpu`.

The optimized Q4_K repack and Path 2 implementation is likewise explicitly AVX2/FMA-gated.
Consequently an ARM64 process cannot use the same fused quantized dot/GEMM implementation that made
the current x86 CPU path competitive. Large-batch F32 BLAS may still be available through a native
BLAS installation, but decode and small-batch fused quantized operations do not have an equivalent
OpenTail ARM fast path.

The local llama.cpp source provides a concrete implementation reference. Its CPU repack selector
contains ARM NEON/dot-product/i8mm paths for multiple formats, including Q4_0, Q4_K, Q5_K, Q6_K,
IQ4_NL, MXFP4 and Q8_0 (availability varies with the ARM feature set).

### Why it matters

This is not a niche tensor-shape optimization. It is a missing hardware backend within the CPU
backend, affecting both prompt processing and token generation across an entire hardware class.
Potential beneficiaries include Apple Silicon, Windows on ARM, Snapdragon systems and ARM cloud
servers.

### Safe programme shape

1. Add a pure CPU-capability record covering `AdvSimd`, dot-product and i8mm support.
2. Start with one common format, preferably Q4_K, and a byte-identical scalar oracle.
3. Build isolated dot, GEMV and GEMM harnesses before integrating dispatch.
4. Validate on at least one dot-product-capable ARM64 machine and one baseline NEON-only target.
5. Retain scalar/x86 dispatch unchanged when ARM eligibility fails.

This work should not ship based only on cross-compilation. It needs ARM execution and performance
evidence.

## 2. Several supported GGUF formats have no fused native matvec

### Confirmed state

`SimdKernels.MatVec` directly specializes only:

- Float32
- Q4_K
- Q5_K
- Q6_K
- Q3_K
- Q2_K
- Q8_0

`Dequantize.ToFloat32` supports a much wider set, including Float16, BFloat16, Q8_1, Q4_0, Q4_1,
Q5_0, Q5_1, MXFP4, NVFP4, Q1_0, Q2_0 and IQ4_NL. Those unhandled by `MatVec` fall through to
`MatVecDequantFallback`.

That fallback is structurally expensive:

1. Allocate an F32 row buffer for each worker.
2. Scalar-dequantize one complete weight row into that buffer.
3. Run an F32 dot product over the expanded row.

This is qualitatively different from the fused register-dequantizing kernels used for the K-quants.
It expands every row before consuming it and gives up most of the quantized format's bandwidth
advantage during computation.

### Concrete llama.cpp comparison

The local llama.cpp `ggml-cpu/repack.cpp` and `arch/x86/repack.cpp` contain x86 repacked GEMV/GEMM
implementations for:

- Q4_0 on AVX2
- IQ4_NL on AVX2
- MXFP4 on AVX2
- Q2_K on AVX-512

The existence of these kernels makes the coverage gap factual rather than a proposed new algorithm.
Their OpenTail ports would still require isolated and end-to-end validation; a successful native
implementation elsewhere does not establish its .NET code-generation quality or crossover points.

### Suggested order

Choose according to model availability, not enum order:

1. Q4_0, because it is simple and widely understood.
2. IQ4_NL or MXFP4 when a real target model is available.
3. Q2_K AVX-512 only when an AVX-512 machine and representative Q2_K model are available.
4. Treat the other fallback formats as coverage work, not presumed performance priorities.

For each format, measure both decode/GEMV and prefill/GEMM. A direct matvec can materially help
decode even if large prefill already routes through dequantized OpenBLAS SGEMM.

## 3. CPU Flash attention covers only `headDim=64`

### Confirmed state

The production Flash64 gate in `ForwardPass.PrefillCoreAttention` requires all of:

- AVX2 and FMA;
- `headDim == 64`;
- a single common head dimension (`_layerHeadDim is null`);
- total context length of at least 256 tokens.

`headDim=128`, `headDim=256`, and per-layer head-dimension models therefore use the older general
attention implementation. The isolated `tools/attn-bench` harness is also compiled around a
constant `HeadDim = 64`, so it currently cannot establish whether a widened implementation wins.

### Concrete opportunity

A Flash128 path can reuse the existing online-softmax structure, causal masking, GQA mapping,
tiling discipline and parity strategy. It is not a new algorithm, but it is a new kernel shape:
twice the Q/K work and twice the output accumulator footprint may change the best query/KV tiles
and scheduling choice.

`headDim=128` should be the first extension because it covers a large family of contemporary dense
models. `headDim=256` should follow only when a representative model shows that attention is a
meaningful fraction of CPU prefill.

### Required gates

1. Parameterize `attn-bench` for 64 and 128 rather than duplicating it.
2. Compare the current general path with Flash128 in an isolated interleaved benchmark.
3. Preserve the same head-job schedule initially; scheduling and arithmetic should remain separate
   experiments.
4. Add chunked-vs-unchunked, Flash-vs-reference and greedy-token tests above and below the 256-token
   threshold.
5. Test at least one MHA and one GQA model end to end.

## 4. Q6_K retains a measured x86 kernel gap

### Confirmed and measured state

The reference Q4_K_M model is a mixed quantization. Its measured tensor census was:

- 144 Q4_K tensors;
- 25 Q6_K tensors;
- 49 F32 tensors.

Only Q4_K is eligible for OpenTail's repacked Path 2. The repacked tensors represented 729 MiB of a
1005 MiB model, leaving approximately 27.5% of weight bytes on the ordinary paths.

An isolated, byte-equivalent Q6_K x Q8_K comparison has already been run:

| Implementation | Best time, 8192 columns x 512 rows |
|---|---:|
| llama.cpp `ggml_vec_dot_q6_K_q8_K` | 0.1481 ms |
| OpenTail `SimdKernels.DotQ6K_Q8K` | 0.2487 ms |

OpenTail was **1.68x slower** by best time. The checksum matched exactly. This proves a bounded
kernel/code-generation gap, but not a 1.68x application gap: Q6_K is only part of the total phase,
and end-to-end execution includes scheduling, activation quantization, attention and other tensors.

### Best next experiment

Compare the current C# AVX2 instruction sequence directly with llama.cpp's x86 Q6_K dot and port
one bounded structural difference into the existing isolated harness. Do not begin with a Q6_K
repacked-GEMM design: llama.cpp itself does not select a Q6_K repack on AVX2, and the existing
measured single-dot difference is the cheaper evidence-backed target.

## 5. Per-layer head dimensions disable the entire batched CPU prefill path

### Confirmed state

`ForwardPass.PrefillDispatch` explicitly sends any model with `_layerHeadDim is not null` through a
token-by-token loop because `PrefillCore` assumes one Q/KV dimension across all layers. The comment
identifies Gemma 4 as the motivating architecture and leaves per-layer plumbing for a later phase.

This fallback misses substantially more than Flash attention. It also loses:

- batched matrix multiplication;
- repacked Q4_K GEMM;
- batched normalization/projection orchestration;
- batched attention;
- the corresponding weight reuse across prompt tokens.

### Concrete opportunity

Plumb `qDim`, `kvDim` and `headDim` per layer through `PrefillCore`, sizing shared buffers by the
existing maximum dimensions but slicing each operation by the active layer's dimensions. Once the
batched path is correct, layer shapes can independently select Flash64, a future Flash128/256 path,
or the general attention implementation.

This is architecture enablement rather than a single microkernel. It may be one of the largest
remaining CPU prefill improvements for an affected model because the present fallback is completely
sequential.

## 6. Generic CPU MoE prefill is explicitly sequential

### Confirmed state

The same `PrefillDispatch` condition sends `_hp.IsMoE` through repeated single-token `Forward`
calls. Its source comment states that batched FFN is not yet supported for MoE.

The repository contains specialized hybrid/GDN and GPU MoE machinery, so this statement should not
be generalized to every backend or every MoE architecture. It is specifically a gap in the generic
CPU `ForwardPass` route.

### Concrete opportunity

A batched CPU MoE path needs more than a wider dot product:

1. Compute routing for all prompt tokens.
2. Bucket token/expert pairs by expert.
3. Run each selected expert over its token panel using the existing batched quantized kernels where
   eligible.
4. Scatter weighted expert results back to prompt order.
5. Preserve router/top-k and accumulation numerics.

This is a substantial architectural project, but the current sequential fallback and the required
data flow are both explicit. It should be pursued only with a representative CPU-only MoE model and
a phase profile showing routed experts dominate prompt processing.

## 7. Q3_K and Q2_K lack exact multi-input verification kernels

### Confirmed state

`MatVec2In` and `MatVec4In` have explicit multi-input cases for Q4_K, Q5_K and Q6_K. Q3_K and Q2_K
are absent, so the default path decomposes the operation into repeated smaller/single matvec calls.

Q3_K already has Q8_KS `_2In`, `_4In` and `_8In` kernels for prefill, but those change activation
quantization and therefore cannot simply replace the exact F32-activation path used by speculative
verification. An exact Q3_K multi-input kernel must widen the existing F32 dot while preserving each
token's accumulation order. Q2_K requires the same treatment from its single-input F32 kernel.

### Scope

This primarily benefits:

- MTP verification;
- speculative decoding;
- other small multi-token calls that require results independent of batch companions.

It is a real coverage gap but a lower priority than Flash128, Q6_K and whole-architecture prefill
fallbacks because it requires the combination of a matching quantization format and multi-token
verification.

## Additional lower-priority seams

### Q8_0 batched Q8 activation path

OpenTail has single-input `Q8_0 x Q8_K` and `Q8_0 x Q8_KS` dots, but
`TryResolveQ8Dispatch` rejects Q8_0 because it has no `_4In`/`_8In` family. `MatVec2In` and
`MatVec4In` deliberately perform several single dots while the weight row is hot in L1, noting that
Q8_0 has no expensive nibble unpack to amortize. A fused kernel is therefore a confirmed missing
specialization but not a high-confidence performance win. Benchmark it only when a Q8_0-heavy model
and a relevant small-batch workload justify the work.

### Additional VNNI use

The Q4_K/Q8_KS accumulator uses `AvxVnniInt8` when available, but much of the remaining quantized
kernel family is written around AVX2 multiply-add sequences. Extending VNNI mechanically is not a
recommendation: it is useful only where the signedness and scale layout map cleanly and where an
isolated benchmark shows the inner product, rather than surrounding decode/scale work, is limiting.

### AVX-512 widening

Some exact Q4_K/Q5_K/Q6_K dots already contain AVX-512 paths, while repacked Path 2 is AVX2. A
512-bit Path 2 is not currently an evidence-backed target: wider vectors can increase register
pressure or reduce clock speed. Do not pursue it without an isolated kernel result on the actual
microarchitecture.

## Recommended programme

The previous performance wave should remain closed and banked. If architecture coverage work is
approved, use a fresh measurement log and treat every line below as a separate gated project:

1. **Flash128**, if a representative model is available locally.
2. **Q6_K AVX2 parity**, starting from the existing 1.68x isolated comparison.
3. **Q4_0/IQ4_NL/MXFP4 x86 kernels**, selected by real model demand.
4. **Per-layer-head-dimension batched prefill**.
5. **ARM64 kernel suite**, once ARM execution hardware or CI is available.
6. **Generic CPU MoE batched prefill**.
7. **Q3_K/Q2_K exact multi-input coverage**.

For every project:

- inventory the target model's actual tensor dtypes and shapes before implementation;
- compare against the exact current fallback in an isolated harness;
- interleave benchmark arms and take enough samples to clear the observed noise floor;
- measure end to end after the isolated gate passes;
- test chunked prefill, continuous batching and numerical/greedy parity as applicable;
- keep the old path as a capability- and shape-gated fallback;
- do not promote an isolated improvement to a default without an end-to-end result.

## Source pointers

- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:87` — batched matmul and Q8 dispatch.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:234` — dtype-to-Q8-kernel mapping and explicit exclusions.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:582` — direct `MatVec` dtype switch.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:822` — `MatVec2In` coverage.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:989` — `MatVec4In` coverage.
- `src/OpenTail.Stingray.Cpu/SimdKernels.cs:1154` — generic row-dequantization fallback.
- `src/OpenTail.Stingray.Engine/ForwardPass.cs:765` — sequential MoE/per-layer-head-dimension prefill fallback.
- `src/OpenTail.Stingray.Engine/ForwardPass.cs:840` — Q4_K-only repacked prefill route.
- `src/OpenTail.Stingray.Engine/ForwardPass.cs:940` — Q4_K dtype eligibility.
- `src/OpenTail.Stingray.Engine/ForwardPass.cs:2323` — Flash64 eligibility.
- `tools/attn-bench/Program.cs:29` — head-dimension-64-only isolated harness.
- `docs/repack-gemm/README.md:67` — llama.cpp repack coverage analysis.
- `docs/repack-gemm/port-log.md:1567` — carried-forward CPU leads.
- `docs/perf-loop-progress.md:3336` — measured Q6_K C++/C# isolated comparison.
- `../LLamaSharp/llama.cpp/ggml/src/ggml-cpu/repack.cpp:4550` — local llama.cpp repack selection.
- `../LLamaSharp/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:1448` — local x86 repacked kernels.
- `../LLamaSharp/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` — local ARM repacked kernels.


---

# Work log (loop, from 2026-08-03)

## Item 2 — Q6_K AVX2 scale broadcast: rewritten, not yet measured

`SimdKernels.DotQ6K_Q8K_Avx2` rebuilt each of the 8 per-super-block scale groups from scalar
memory — two sign-extending byte loads, two broadcasts and a `vinserti128` per group, ~7 ops x 8
groups, inside the block loop. llama.cpp (`ggml-cpu/arch/x86/quants.c:2350`) keeps the 16 scale
bytes resident in a register and issues one `pshufb` per group, ~2 ops, touching no memory.

Replaced with a `Q6KScaleShuffle` mask table (port of llama.cpp's `get_scale_shuffle`) plus
`Ssse3.Shuffle` + `ConvertToVector256Int16`. Mask `i` selects `scales[2i]` into lanes 0-7 and
`scales[2i+1]` into lanes 8-15 — byte-identical to what the scalar form produced.

**Status: builds clean, full suite green apart from three unrelated pre-existing failures (below).
NOT yet benchmarked.** The isolated baseline to beat is 1.68x slower than llama.cpp (0.2487 vs
0.1481 ms, checksums equal). Amdahl caps any end-to-end gain well below that: Q6_K is ~27.5% of
weight bytes in the reference model.

## Newly exposed: Gemma 4 E4B Q4_0 on Vulkan is broken

Downloading `gemma-4-E4B_q4_0-it.gguf` made three tests reachable that had been passing only by
early-returning on a missing model file:

- `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_MatchesCpuArgmax` — 262144 non-finite logits
- `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent` — `CPU=506 Vulkan=0`
- `Gemma4VulkanNarrowedKvE2ETests.Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax` — degenerate

**Attribution:** these are Vulkan-side, on a Q4_0 model; the change above is CPU-only and touches
only the Q6_K dot. The CPU reference in the same tests produces sensible tokens while Vulkan emits
zeros. Pre-existing defect, newly visible — not a regression. To be confirmed by stashing the CPU
change and re-running, before any further work builds on the assumption.

**A test that passes because its input is missing is not a passing test.** Three real defects sat
green behind an early return. Worth auditing the other model-gated tests for the same pattern.

## Item 2 RESULT: the pshufb scale-broadcast port is 6.3x SLOWER — reverted

Measured immediately after the rewrite, both arms built and run back-to-back on an idle box,
`tools/kernel-bench-cs`, k=8192 rows=512 reps=12, `DOTNET_TC_QuickJitForLoops=0`:

| arm | checksum | best | mean | sd |
|---|---|---:|---:|---:|
| incumbent (scalar `Vector128.Create((short)sc[..])`) | 2363.599609 | **0.2063 ms** | 0.2167 | 0.0073 |
| `Q6KScaleShuffle` + `Ssse3.Shuffle` (llama.cpp port) | 2363.599609 | **1.2999 ms** | 1.3135 | 0.0227 |

**Checksums identical, so the port is numerically correct — and 6.3x slower.** Reverted.

### Why the op-count reasoning was wrong

The analysis counted x86 instructions and concluded ~7 ops became ~2. It ignored what
`Q6KScaleShuffle[ishuf + n]` actually costs in .NET: `static readonly Vector128<byte>[]` is a
**managed array**, so each of the four accesses per inner iteration is a static-field load plus an
array bounds check plus a 16-byte load — in the innermost loop of the kernel. llama.cpp's
`get_scale_shuffle` reads a C `static const` array with no bounds check and frequently folds away.

This is precisely the caveat this document already carried: *"a successful native implementation
elsewhere does not establish its .NET code-generation quality or crossover points."* The port was
made anyway on instruction-count reasoning alone. Recorded so the next person does not re-derive it.

### Baseline correction

The incumbent measures **0.2063 ms** here, not the 0.2487 ms recorded earlier — so the gap to
llama.cpp's 0.1481 ms is **1.39x, not 1.68x**. The earlier figure came from a different session and
machine state. Use 0.2063 ms as the number to beat, and re-measure both arms together rather than
comparing against a stored constant.

### Refined hypothesis, NOT yet tested

The mask table itself may still be right; the *access* is what costs. Untested alternatives:
1. Hoist all 8 masks into locals before the block loop so they are loaded once, not per iteration.
2. `ReadOnlySpan<byte>` over a static data literal (data-section reference, no bounds check).
3. Fully unroll the `j` loop so every `Vector128.Create` mask becomes a JIT constant.

Do not attempt these without re-running the A/B; the incumbent is fast and the burden of proof is
on the replacement.

## Item 2 (attribution) CONFIRMED: Gemma4 Vulkan failures are pre-existing

With the Q6_K rewrite reverted out of the tree, the full suite still reports **1261 total, 3 failed**
— the same three Gemma4 E4B Q4_0 Vulkan tests. The CPU change was not the cause; the earlier
attribution-by-reasoning is now attribution-by-experiment.

They are genuine defects in the Vulkan Gemma 4 path, invisible until `gemma-4-E4B_q4_0-it.gguf` was
downloaded because the tests early-return when the model file is absent. Next: fix them, and audit
the other model-gated tests for the same silently-passing pattern.

## Gemma4 E4B Q4_0 Vulkan defect — diagnostic starting point

Established so far (no fix yet):

- **Not a dtype-coverage gap.** `Q4_0` is in `GpuForwardPass.IsRawGpuQuant`, so Q4_0 weights upload
  as raw blocks and dequantize in-shader via `Shaders.MatVecQ4_0`
  (`VulkanBackend.cs:2411`). The dtype is wired up.
- **Not the batched trunk.** `_canBatchedTrunk` requires every trunk weight to be Q4_K/Q6_K, and
  this model is 342 Q4_0 + 2 Q6_K, so the trunk is excluded already; Gemma 4 also routes through
  `ForwardGemma4`. Neither the new Path 2 nor the batched prefill can be implicated.
- **Corruption is upstream of the output projection.** All 262144 logits are non-finite — the whole
  vocab. The two Q6_K tensors are almost certainly `token_embd`/`output`, and a Q6_K projection that
  produced garbage for every row would more likely give finite nonsense than uniform non-finites.
  A hidden state that is already NaN/Inf on entry explains it better.
- **CPU is healthy on the same model** (`CPU=506 Vulkan=0`), so this is Vulkan-side only.

Next step: bisect where the state first goes non-finite — embedding output, PLE injection, then
per-layer — rather than inspecting shaders by eye. The test name
(`..._VulkanForward_LongDecodeIsCoherent`) already suggests PLE-injection / shared-KV /
attention-scale as the suspected class; confirm before trusting it.

**Do not assume the three failures share one root cause** — two are plain forward/decode, one is the
narrowed-KV path. They may be one bug or three.

## Test-suite audit: the green suite substantially overstates coverage

Prompted by the Gemma4 defects, which sat green for as long as the model file was absent.

**106 test files** in `OpenTail.Stingray.Tests.ForwardPass` use the early-return-on-missing-model
pattern, gating on **22 distinct GGUF filenames**. Of those, **3 are present on this machine and 19
are absent**:

```
Llama-4-Scout-17B-16E-...        Qwen3-0.6B-Instruct-Q4_K_M       Qwen3-1.7B-Instruct-Q4_K_M
Qwen3-4B-Q4_K_M                  Qwen3-8B-Q4_K_M                  Qwen3-Coder-30B-A3B-Instruct
Qwen3.6-27B-MTP-Q4_K_M           Qwen3.6-35B-A3B-UD-Q4_K_M        Qwen_Qwen3-4B-Q4_K_M
gemma-3-1b-it-Q8_0               gemma-4-12B-it-qat-Q4_K_M        gemma-4-12b-it-Q4_K_M
gemma-4-12b-it-qat-q4_0          gemma-4-E2B-it-Q8_0              gemma-4-E4B-it-Q8_0
gemma4-12b-q4km                  gemma4-v2-Q4_K_M                 mmproj-gemma-4-12b-it-qat-q4_0
qwen3-4b-q4_k_m
```

**"1261 passed, 0 failed" is not 1261 tests' worth of evidence.** Downloading exactly one of these
19 files immediately produced three genuine Vulkan defects. There is no reason to think that model
was uniquely broken — it was uniquely *present*.

Note also `Qwen3-0.6B-Instruct-Q4_K_M.gguf` is gated but what was downloaded is
`Qwen3-0.6B-Q8_0.gguf`; different filename, so those tests still skip. Filename-exact gating means
"having the model" is not the same as "satisfying the gate".

### Recommended, in order of value per byte

1. Make skipping **visible**. An early `return` is indistinguishable from a pass in the summary.
   Report these as skipped (xUnit `Assert.Skip`/`SkipWhen`) so the count states what actually ran.
2. Add a CI/local check that prints the present/absent gated-model table, so coverage loss is
   legible without reading 106 files.
3. Prioritise downloads by how many tests each unblocks, not by model size — several names above
   are near-duplicates of each other and of files already held.

This is cheap relative to any kernel work in this document, and it changes what every other result
in it means.

## Tool: scripts/check-test-model-coverage.ps1

Makes the invisible skipping legible. Scans test sources for `.gguf` literals, classifies them as
real gated models vs throwaway fixtures, and prints a present/absent table sorted by **test files
unblocked** (the right download priority — not model size).

Current state: **3 of 21 gated models present (14.3%)**, 18 absent, affecting up to 60 test-file
references. Highest-value downloads:

| model | test files |
|---|---:|
| `Qwen3-8B-Q4_K_M.gguf` | 16 |
| `gemma-4-E4B-it-Q8_0.gguf` | 11 |
| `Qwen3.6-27B-MTP-Q4_K_M.gguf` | 8 |
| `gemma-4-12b-it-qat-q4_0.gguf` | 5 |

`-FailOnMissing` exits non-zero for CI; off by default because on a dev box most models are
legitimately absent and a script that always fails gets ignored.

### Classification, and a mistake worth recording

First cut used only `download-model.ps1`'s literal filenames as the authority for "real model".
That **hid the second-largest gap**: `gemma-4-E4B-it-Q8_0.gguf` backs 11 test files but is never
named literally in that script, so it was silently reclassified as a fixture. Trading cry-wolf for
silent under-reporting is the worse failure — an over-reporting checker gets ignored, but an
under-reporting one gets believed.

Fixed by adding a second signal: a quant marker in the filename (`[Qq]\d[_KkMm0-9]` — matches
Q4_K_M, Q8_0, q4_0, q4km). Every real GGUF here carries one; the fixtures (`a`, `b`, `c`, `x`,
`smol`, `model`, `broken`, `from-*`, `qwen35moe`) carry none. A name counts as real if it appears
in download-model.ps1, exists in models/, **or** matches the quant marker.

## Item 3 (Flash128) scoping: "parameterize attn-bench" has a measurement trap

`tools/attn-bench/Program.cs` hardcodes the shape as compile-time constants:

```csharp
private const int NumHeads = 32;
private const int HeadDim  = 64;
private const int QDim     = NumHeads * HeadDim;
private const int KvDim    = NumHeads * HeadDim;
```

**103 references** to these four names across the file.

Good news: the inner loops look shape-agnostic — the `d` loops step through `HeadDim` in
`Vector256` strides (`v + i*KvDim + h*HeadDim + d`, lines 603/618/633), and the `64`s at lines
239/291/554/771 are **token** tiles, not head-dimension. So the arithmetic should widen to 128
without restructuring.

**The trap:** turning `HeadDim` into a runtime parameter deletes a compile-time constant the JIT
currently folds — loop bounds, strides and unroll factors all derive from it. A naive
parameterization would then compare *const-64* against *runtime-128* and attribute the difference to
head dimension, when part of it is just lost constant folding. That is the same class of error as
this session's Q6_K result, where a change that looked like fewer instructions cost 6.3x because of
what the runtime actually emitted.

**Do it shape-specialized, not shape-parameterized.** Either a generic struct type-parameter
supplying `HeadDim` (the JIT specializes per instantiation and keeps the constants) or two
explicitly compiled paths. Then 64-vs-64 must reproduce the existing numbers before any 128 result
is believed — that is the control arm, and without it the comparison is unanchored.

Not started beyond this scoping; recorded so the next attempt does not walk into it.

## Qwen3-8B downloaded: same test count, 4.4x the work

Downloaded `Qwen3-8B-Q4_K_M.gguf` (4.68 GB) — the top entry in the coverage table, backing 16 test
files. Full suite before and after:

| | total | failed | time |
|---|---:|---:|---:|
| without Qwen3-8B | 1261 | 3 | 140 s |
| with Qwen3-8B | 1261 | 3 | **621 s** |

**The test count is identical. The runtime is 4.4x.** The same 1261 tests were "passing" before;
roughly 480 seconds of real verification simply was not happening. Nothing demonstrates more
directly that a pass count is not a coverage measure — the number moved not at all while the work
done quadrupled.

No new failures: the 16 Qwen3-8B test files all pass. That is a genuine (and welcome) result rather
than a null one, but note it is the second model tried — the first (`gemma-4-E4B_q4_0-it.gguf`)
produced three real defects. Two samples, one clean and one not; no basis yet for predicting the
remaining 17.

Coverage now **4/21 gated models (19%)**. The still-absent high-value entries:
`gemma-4-E4B-it-Q8_0.gguf` (11 test files), `Qwen3.6-27B-MTP-Q4_K_M.gguf` (8),
`gemma-4-12b-it-qat-q4_0.gguf` (5).

The 3 remaining failures are unchanged and still the Gemma4 E4B Q4_0 Vulkan defects.

---

# Loop stopped 2026-08-03 — handover

The 10-minute loop was stopped early, before the end-of-day deadline and before items 3-7 were
attempted. Reason: the agent's context was exhausted and successive fires were producing handover
notes rather than work. That is a worse use of budget than stopping, and the remaining items each
need room to iterate (build -> measure -> adjust), not a fresh cold start every ten minutes.

## Done

| item | outcome |
|---|---|
| Q6_K AVX2 scale-broadcast | **Negative.** Numerically correct (checksum identical) but **6.3x slower** (1.2999 ms vs 0.2063 ms). Reverted in the working tree. |
| Gemma4 attribution | **Confirmed pre-existing** by revert-and-rerun, not by argument. |
| Test-coverage audit | **106 test files** gate on a model file and return early; **4 of 21** gated models present. Tooling shipped. |
| `scripts/check-test-model-coverage.ps1` | New. Present/absent table sorted by test files unblocked. |
| Models downloaded | Qwen3-0.6B-Q8_0, OLMoE-1B-7B, gemma-4-E4B q4_0 + mmproj, Qwen3-8B-Q4_K_M |

## Not done

Items 3 (tiled Q6_K shader), 5 (Q4_0 x86 kernel), 6 (per-layer head dims), 7 (CPU MoE prefill) were
not started. Item 2b (fix the Gemma4 Vulkan defects) and item 4 (Flash128) are scoped only — see
their sections above for the diagnostic starting point and the measurement trap respectively.

## The one thing to land first

`src/OpenTail.Stingray.Cpu/SimdKernels.cs` is **modified and uncommitted**: the working tree holds the
fast incumbent Q6_K kernel, while HEAD (8b5d469) still carries the 6.3x-slower rewrite. If a commit
sweep runs without this, the regression ships.

## Two corrections, same root cause

Both this session's errors were the same mistake — reasoning about instruction counts instead of
measuring what the runtime emits:

1. **Q6_K**: "~7 ops become ~2" was 6.3x slower, because `static readonly Vector128<byte>[]` is a
   managed array and every access carries a bounds check, in the innermost loop.
2. **Flash128** (caught before it cost anything): making `HeadDim` a runtime parameter would delete
   constants the JIT folds, so a naive 64-vs-128 comparison would credit lost constant folding to
   head dimension.

The general form: **a native implementation's instruction count does not predict .NET codegen.**
This document already said so before either mistake was made.

## The most transferable finding

`1261 passed, 0 failed` was never 1261 tests' worth of evidence. Adding one model (Qwen3-8B)
changed the count by **zero** and the runtime by **4.4x** (140 s -> 621 s). Adding a different one
(gemma-4-E4B q4_0) surfaced **three real defects** that had been green for as long as the file was
absent. Fix the visibility (report skips as skips) before trusting any coverage claim in this repo.

## Gemma4 Vulkan defect — STRONG LEAD: headDim=512, and no head-dim gate exists

Reproduced outside the test harness with the CLI (so it is not a test-fixture artefact):

```
opentail-llm-cli -m models/gemma-4-E4B_q4_0-it.gguf --backend vulkan -g -1 -p "The capital of France is" -n 6 --temp 0
  Model loaded — 42L, 2560d, headDim=512, 262144 vocab, ctx=131072
  The capital of France is<pad><pad><pad><pad><pad><pad>
```

**`headDim=512`.** Attention kernels here are written around 64/128/256; 512 is outside that range.
Two facts make this the leading hypothesis rather than a guess:

1. **There is no head-dimension gate anywhere in `GpuForwardPass`** — grepping for `headDim >`,
   `_headDim >`, `headDim <=` returns nothing. Vulkan accepts any head dimension and dispatches
   regardless. A shader that assigns one invocation per head-dim element under `local_size_x = 256`
   silently covers only half the head at 512, which is exactly the shape of "finite-but-wrong state
   that degenerates into NaN/pad".
2. **The CPU path is healthy on the same weights** (`CPU=506 Vulkan=0`), and the CPU attention is
   the general shape-agnostic implementation.

This also reframes the earlier note that corruption is "upstream of the output projection" — that
remains true, and attention is upstream.

### Next steps, in order

1. Find which Vulkan attention shader is selected for this model and check its head-dim assumption
   (thread-per-element mapping, any `[256]` sized per-head buffer, tile constants).
2. **Add the missing capability gate first, before fixing any shader.** A model whose head dimension
   the shaders cannot serve should fall back to CPU attention (or refuse with a clear message), not
   emit `<pad>`. That is a correctness fix independent of whether a 512-wide kernel is ever written,
   and it is what turned this from a silent wrong answer into a diagnosable one.
3. Only then decide whether a headDim=512 kernel is worth writing.

Note the interaction with item (C): Flash64 is gated on `headDim == 64`, so it correctly declines
here. The defect is in the GENERAL Vulkan attention path, which has no equivalent guard.

## Added: non-finite logits guard (GpuForwardPass.GuardFiniteLogits)

Inserted after all **4** `ReadFromStaging(_logitsBuf)` sites. Throws with the count of bad logits,
the first bad index, and the head dimension, pointing at `--backend cpu` and this document.

Verified both directions on hardware:

| model | before | after |
|---|---|---|
| gemma-4-E4B q4_0 (Vulkan) | `The capital of France is<pad><pad>...` | throws, from `Prefill` -> `Forward` |
| SmolLM2-1.7B Q4_K_M (Vulkan) | "…is Paris" | "…is Paris", 78.9 t/s — no false positive |
| Qwen3-0.6B Q8_0 (Vulkan) | thinking output | unchanged, 25.1 t/s — no false positive |

**Bonus datum from the stack trace:** it fires from `Prefill` -> `Forward` (the per-token loop), so
the state is already non-finite during PREFILL, not at decode. That narrows the bisect considerably
— the corruption does not require multi-token history or KV reuse.

### What this is and is not

**Is:** a fix for the silent-wrong-answer class. Argmax over NaNs returns index 0, index 0 is a valid
token id, so generation proceeded looking merely stupid rather than broken. It will catch the next
bug of this class too.

**Is not:** a fix for the Gemma4 defect itself, and it does not claim to be. Root cause is still
unidentified.

### Retraction: the headDim=512 hypothesis is NOT supported

The previous section proposed headDim=512 as the leading cause on the grounds that the shaders are
written for 64/128/256. Inspecting them does not support that:

- the per-head RmsNorm shader uses a grid-stride loop, `for (i = tid; i < head_dim; i += 256u)`
- the main `Attention` shader uses a runtime bound, `for (d = 0; d < head_dim; d++)`

Both are head-dimension agnostic. A blanket head-dim gate was therefore **not** added — it would
have encoded a guess, risked blocking head dimensions that work, and missed the real constraint.
headDim=512 remains unusual and worth revisiting, but it is not currently evidence-backed.

## Gemma4 bisect, step 1: embedding + PLE upload are CLEAN

Added an env-gated probe (`STINGRAY_GEMMA4_PROBE=embed`, `GpuForwardPass.ForwardGemma4`) that
submits the command buffer immediately after the embedding lookup + `EmbeddingScale` and downloads
`_hidden` before any layer runs:

```
[gemma4-probe embed] token=2 embDim=2560 nonFinite=0 maxAbsFinite=5.065E+000
  first8=[-1.175E+000, 2.023E+000, 1.175E+000, 1.175E+000, -1.762E+000, -1.175E+000, 8.483E-001, 8.483E-001]
```

**Zero non-finite values, and the magnitudes are sane** (max |v| = 5.07, no denormals, no zeros-only).
So `EmbedTokenGemma4`, the embedding scale, and the PLE row gather/upload that precedes them are all
healthy. **The corruption is inside the 42-layer trunk (`RunGemma4Layers`).**

Combined with the earlier finding that the guard fires from `Prefill` -> `Forward` (single-token
path), the defect needs neither multi-token history nor KV reuse — one token through the trunk is
enough to reproduce.

### Next step

Extend the probe to `STINGRAY_GEMMA4_PROBE=layer:<n>`: submit and dump `_hidden` after layer n,
then walk n upward to find the first layer that produces non-finite state. Note `RunGemma4Layers`
uses **per-layer head dimensions** (`_hp.LayerHeadDim![layer]`) and shared-KV tail layers, so the
first bad n should be cross-referenced against that layer's `layerHd`/`layerKv` and whether it is a
shared-KV layer — those are the structural features that differ between layers and are the most
likely place for a Vulkan-only assumption to break.

## Gemma4 bisect, steps 2-3: found it — GLSL `tanh` overflows to NaN

### Step 2: not a gradual divergence, dead at layer 0

`STINGRAY_GEMMA4_PROBE=layers` dumps `_hidden` after **every** layer in ONE model load (split the
command buffer per layer, fence-wait, download, re-`BeginRecord` — device buffers persist across
submissions, so the trunk resumes exactly where it left off). That is 42 data points for the price of
one 4 GB load instead of 42 separate runs.

```
layer   hd   kv  shared  swa   kEqV  nonFinite      maxAbs         rms
    0  256    2      no  yes     no       2560  0.000E+000  0.000E+000
    1  256    2      no  yes     no       2560  0.000E+000  0.000E+000
   ...  (all 42 identical)
```

**All 2560 values non-finite after layer 0.** The per-layer-head-dim / shared-KV hypothesis from
step 1 was wrong — those features vary across layers, and the failure does not. It is layer 0.

### Step 3: `STINGRAY_GEMMA4_PROBE=stage0`, one level finer

Same trick inside layer 0, dumping all 24 intermediates:

```
stage                           n nonFinite      maxAbs         rms
00 hidden@entry              2560         0  5.065E+000  9.836E-001
01 attn_norm                 2560         0  3.630E+002  1.344E+001
02 q_proj                    2048         0  8.364E+001  1.644E+001
...
16 ffn_gate                 10240         0  2.031E+001  1.022E+000
17 ffn_up                   10240         0  2.633E+001  9.606E-001
18 gelu_mul                 10240         1  9.925E+001  1.587E+000   firstBad=[2049]=NaN
    gelu input at [2049]: g=20.309921 up=-26.330753 tanhArg=315.09953614443674
19 ffn_down                  2560      2560  0.000E+000  0.000E+000   firstBad=[0]=NaN
```

**ONE NaN in 10240**, at the GELU. The next op is `ffn_down`, a matmul — every output row dots the
whole 10240-wide vector, so one NaN input contaminates all 2560 outputs. From there it rides the
residual through 42 layers to the logits.

### Root cause

GLSL spec-defines `tanh(x)` as `(e^x - e^-x)/(e^x + e^-x)` and drivers implement it literally.
float32 `exp` overflows past ~88, so `|inner| > ~44` gives `inf/inf = NaN`. Gemma 4's wide FFN
produced a single gate value of **g = 20.31**, hence `inner = 0.7978*(g + 0.044715*g^3) = 315.1`.

Only one element in 10240 crossed the threshold, which is exactly why this presented as an
intermittent, model-specific, silently-wrong bug rather than an obvious one.

**The CPU backend already had the fix.** `SimdKernels.GeluTanhMul` clamps `2*inner` to +/-20 with a
comment naming *"~ +/- 20 gate inputs from a wide-dim trunk like Gemma 4"* — this exact bug was found
and fixed on CPU, and the Vulkan shader was never updated to match. CUDA is safe for a different
reason: `tanhf` is a proper libdevice implementation that saturates.

### Fix

`Shaders.GeluTanhMul` and `Shaders.Softcap` now clamp the tanh argument to +/-10. `|x| > 10` already
saturates tanh to +/-1 within float32 precision, so the clamp cannot change a representable result.
`Softcap` was fixed at the same time on inspection, not measurement: it has the identical unguarded
`tanh(x/cap)` and Gemma is the only user of both.

| | before | after |
|---|---|---|
| gemma-4-E4B q4_0, Vulkan, `-g -1` | `<pad><pad>...` | `The capital of France is **Paris**.` |
| Gemma4 Vulkan E2E tests | 3 failed | 1 failed |

### SPIR-V: spliced, not regenerated

Added `SpirvGen --only Name1,Name2`, which rewrites just the named entries in place (same `_sN` slot,
new hash key + blob) and leaves the other 88 blobs byte-identical. A full regeneration drifts ~33 of
89 shaders because the local glslc is far newer than whatever produced the committed table. Verified:
the diff is exactly 6 lines — 2 case lines, 2 byte-count comments, 2 arrays.

The first implementation used a multiline regex with `$`, which silently matched **nothing** because
the committed file is CRLF and .NET's `$` anchors before `\n` only. It now uses plain string matching
anchored on the trailing `// Name` comment (which also disambiguates `Attention` from
`AttentionBf16`). A splicer that silently no-ops is worse than one that fails loudly.

### Not fixed: a second, independent Gemma4 Vulkan defect

The NaN was one bug, not the only one. With it fixed the remaining failure changed character
completely — from non-finite output to a **finite numeric divergence**:

```
CPU/Vulkan diverge at decode step 0: CPU=506 Vulkan=9079.
Full Vulkan decode: [9079,236761,106,236761,106,1,106,106,...]
```

The CLI produces correct text on the same model, so this is narrower than "Gemma4 Vulkan is broken" —
the failing tests are the narrowed-KV (BFloat16) and long-decode paths. The same probe infrastructure
(now committed and env-gated) applies directly: dump per-layer state for CPU and Vulkan on the same
token and diff, rather than only checking finiteness.

#### Narrowing the second defect (read-only, while the suite ran)

Of the three Gemma4 Vulkan E2E tests, the one that now **passes** is
`Gemma4_E4B_Q4_0_VulkanForward_MatchesCpuArgmax` — CPU/Vulkan first-argmax parity on the 9-token
prompt `[2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901]`. The two that still fail both use the
6-token prompt `[2, 818, 5279, 529, 7001, 563]` and both produce the identical Vulkan sequence
`[9079, 236761, 106, 236761, 106, 1, 106, ...]`.

So first-argmax parity holds on one prompt and not another, on the same model and code path. That
rules out a whole-path error (a missing V-norm, a wrong attention scale) — those would fail both.
It points at prompt-dependent numeric divergence.

Worth checking before assuming a kernel bug: the failing prompt's ids do not look like the Gemma
tokenization of the string the comment claims ("The capital of France is" tokenizes to the ids the
*passing* test uses). If `[818, 5279, 529, 7001, 563]` is a mistokenized prompt carried over from a
different tokenizer, the test is feeding near-random tokens, where CPU and Vulkan sitting on
opposite sides of a near-tie is expected rather than a defect — and the assertion should be
top-5-tolerant like the sibling suite's, instead of exact-argmax. The CLI producing coherent text
on this model supports that reading but does not prove it.

**Do not fix by loosening the tolerance until the tie hypothesis is checked**: dump both backends'
top-5 logits for that prompt. If they are a genuine near-tie the test is wrong; if they are far
apart it is a real kernel divergence and the per-layer probe applies.

### Verification: the fix, and the tests that guard it

Full `Tests.ForwardPass` suite before adding new tests: **1261 total, 2 failed** — down from 3, with
no regressions anywhere else. The 616 s runtime and the total count are both unchanged, so nothing
was traded away for the fix.

New `VulkanTanhOverflowTests` (3 tests, kernel-level and model-free so they run on any Vulkan
device rather than gating on a 4 GB GGUF):

- `GeluTanhMul_LargeGate_StaysFinite` — includes the literal `g = 20.309921` from the probe.
- `GeluTanhMul_MatchesCpuKernel` — cross-backend against `SimdKernels.GeluTanhMul`, the kernel that
  already had the clamp. This is the guard that would have caught the drift in the first place.
- `SoftcapInPlace_ExtremeLogits_StayFiniteAndBounded` — finite, within +/-cap, sign-preserving.

**The tests were proved to have teeth, not merely to pass.** Reverting both shaders to the unclamped
form and re-splicing makes all three fail with the real values:

```
GeluTanhMul produced NaN for gate=20.309921 (tanh arg 315.1)
gate=20.309921 up=-26.330753: Vulkan=NaN CPU=-534.7755
SoftcapInPlace produced NaN for logit 1000000 (cap 30)
```

Restoring gives back a byte-identical `Shaders.Precompiled.g.cs` and 3/3 green. This matters here
more than usual: this session has repeatedly found suites whose green came from silent early
returns, so "the new test passes" is not evidence until the test has been seen to fail.

For the same reason the new tests use **`Assert.Skip`** rather than `return` when no Vulkan device
is present — a skip is reported as `Skipped: N`, an early return is indistinguishable from a pass.
Confirmed `Skipped: 0` on this box, i.e. they really did execute on the GPU (0.3 s is bare backend
creation with no model upload, not a silent no-op). This is item (G) from the plan, applied to the
tests added here.

Final suite with the new tests included: **1264 total, 2 failed, 0 skipped, 603 s** — 1261 + 3, and
the only failures are the two pre-existing Gemma4 numeric-divergence tests described above. The
count moving by exactly 3 also confirms the new tests were discovered and executed rather than
filtered out.

### Top-5 dump: the "bad test fixture" hypothesis is REFUTED, and inverted

Dumped both backends' top-5 for the failing prompt. The hypothesis was wrong in both directions.

```
failing-prompt ids  2,818,5279,529,7001,563
  decodes to        "<bos>The capital of France is"
  per-token         [<bos> | The |  capital |  of |  France |  is]
"The capital of France is" encodes to  818,5279,529,7001,563   <- exact match

passing-prompt ids  2,651,6037,576,6081,603,1234,4567,8901
  decodes to        "<bos>ath হইqu carry Btern)} plastic"
```

The **failing** test's prompt is the correct tokenization of the real sentence. The **passing**
test's prompt is the gibberish one. I had them backwards — so the test that passes is the weaker
test, and its green says little about correctness.

```
rank | CPU id (logit)          text     | Vulkan id (logit)        text
  0  |    506 (  27.7069)  " the"       |   9079 (  26.5121)  " Paris"
  1  |   9079 (  26.0906)  " Paris"     |   5213 (  22.3774)  " **"
  2  |    496 (  25.2127)  " a"         |    506 (  21.6716)  " the"

CPU top1-top2 gap    : 1.6162      Vulkan top1-top2 gap : 4.1347
whole-vocab |CPU-Vulkan|: max 13.4404, mean 2.6318
```

**This is not a near-tie.** A mean absolute difference of 2.63 logits across the whole 262144-entry
vocabulary (max 13.44), against logits of magnitude ~27, is a real numeric divergence — an order
above Q4_0 cross-backend rounding. The tolerance must NOT be loosened.

Greedy decode from each backend on that prompt:

```
CPU    greedy: " the city of France. The capital of France is the capital of France"
Vulkan greedy: " Paris.<turn|>.<turn|><eos><turn|><turn|><turn|>..."
```

Both are bad, in different ways — and note **Vulkan's first token is the correct answer while the
CPU's is not**. The CPU degenerates into a repetition loop; Vulkan answers correctly and then emits
control-token spam. The prompt is raw (no chat template) and the model is instruction-tuned, which
explains some degeneration on both sides, but not the `<turn|>` run.

Consequence for the test: it asserts CPU == Vulkan and treats CPU as the oracle, but on this prompt
the CPU's own output is the worse of the two. "Make Vulkan match CPU" is therefore not obviously the
right goal, and this needs a third reference (llama.cpp on the same GGUF and prompt) to say which
backend is actually wrong before either is changed.

## Item 2 (Vulkan tiled Q6_K GEMM): LANDED — declines 1418 -> 2

`Shaders.MatMulTiledQ6K`, the Q6_K twin of `MatMulTiledQ4K`: same BM=64 / BN=16 / BK=64 / TM=TN=2,
same LDS layout and `+1` bank-conflict padding, differing only in the dequant that fills `buf_a`.

**Why BK=64 is the right tile for Q6_K too.** Q6_K stores element `lane + 32*j` (lane 0..31,
j 0..7), so scale group j owns the contiguous run `[32j, 32j+32)`. A 64-element k-step therefore
covers exactly groups `2c` and `2c+1` — no group is ever split across a k-step, and each thread
needs one ql byte pair, one qh byte and two scales. Q4_K gets the same property from its c-chunks.
That the two very different layouts agree at BK=64 is why one tile shape serves both, and it is the
reason this shader is a dequant swap rather than a redesign.

Q6_K rows are 210 bytes and thus NOT dword-aligned, so every weight access goes through the same
`gByte`/`gInt8` byte gather as `MatVecBatchedQ6K` — shared verbatim so the two cannot drift.

### Measured, interleaved arms, SmolLM2-1.7B-Q4_K_M, 990-token prompt

| arm | run 1 | run 2 | mean | dispatches served by Path 2 |
|---|---:|---:|---:|---|
| Path 1 | 73.9 | 73.3 | **73.6 t/s** | 0 / 10418 |
| Path 2 | 77.3 | 80.3 | **78.8 t/s** | 10416 / 10418 |

**1.07x**, and more importantly `declines` fell from **1418 to 2**. The 1418 was one Q6_K `ffn_down`
per layer; Path 2 now serves essentially the whole trunk.

### This unblocks the actual prize, and is not itself the prize

1.07x is roughly the same structural gain Q4_K alone gave (1.046x), for the same reason: **BN is
still 16**, so the tile amortizes exactly as many tokens as Path 1 did and streams the weights the
same number of times. The measured ceiling for driving weight traffic to zero is 1.54x, most of it
arriving by chunk 32-64.

What changed is that the blocker is gone. Raising BN now requires:

- BN 16 -> 32 in both tiled shaders, `TiledQ4KBn`, and `Path2MaxTokensPerDispatch`;
- stage B reworked — it currently maps 256 threads as `k = tid >> 4, v = tid & 15` (16 tokens x 16
  vec4), which is exactly one pass only while BN=16; BN=32 needs two;
- TN 2 -> 4, since 32 threads span M (BM/TM) and the remaining 8 must span N (BN/TN);
- LDS check: `buf_a` is 64*65*4 = 16.6 KB and `buf_b` becomes 32*65*4 = 8.3 KB, so 24.9 KB total —
  fits a 32 KB budget, but BN=64 would be 33.2 KB and would not.

**Blocking issue to resolve first: the 2 remaining declines.** Any shape Path 2 declines falls
through to Path 1, and Path 1 *throws* above nTok=16. So raising the chunk before identifying those
2 dispatches would turn a decline into a crash mid-prefill. They must be either served by Path 2 or
proven never to occur at chunk > 16. `Path2Cap_CannotExceed_WhatPath1CanFallBackTo` already pins
this invariant and is the test that will catch it.

## NEGATIVE RESULT: BN=32 halves weight traffic exactly as predicted, and is 34% SLOWER

Raised BN 16 -> 32 in both tiled shaders (TN 2 -> 4, stage B to two passes),
`TiledQ4KBn` and `Path2MaxTokensPerDispatch` to 32. Also had to relax an up-front
`nTok is < 1 or > 16` guard in `MatMulBatched` that rejected a legal Path 2 chunk before the path
seam ever saw it — that guard is now bound to `VulkanMatMulPathConfig.MaxTokensPerDispatch`, which
is the correct expression of it and is kept.

The mechanism worked **perfectly**:

| | chunk 16 | chunk 32 |
|---|---:|---:|
| tokens/dispatch | 15.97 | **31.92** |
| weight I/O | 57404.3 MiB | **28702.1 MiB** (exactly half) |
| dispatches | 10418 | 5210 |
| **prefill** | **75.0 t/s** | **49.8 t/s** |

Weight traffic halved to the megabyte. Throughput fell to **0.66x**. The chunk-sweep model
(`time ≈ weight_GiB/12.33 + 8.04`) predicted **91 t/s**; the measurement is 49.8.

### What this falsifies

The 1.54x ceiling was derived with an assumption stated explicitly at the time — *"assuming a tiled
kernel adds no cost of its own and leaves the fixed 8.04 s untouched."* **That assumption is now
falsified.** The chunk sweep was performed by varying the chunk of the *Path 1 matvec*, so its slope
and intercept describe that kernel's cost structure, not a tiled GEMM's. Extrapolating one kernel's
bandwidth curve onto a different kernel was the error, and it is worth recording because the
extrapolation looked well-founded: five points, a two-point fit, all interior points within 4.3%.

A model can fit its own data perfectly and still not transfer to a different implementation.

### Most likely cause, untested

Per-thread state doubled: `acc[TM][TN]` went 2x2 -> 2x4, plus `bv[TN]` 2 -> 4, so ~14 live floats
per thread against ~8. On a GPU already at low occupancy this plausibly crossed a VGPR threshold and
halved the waves in flight, cancelling the bandwidth win and more. LDS is *not* the explanation:
`buf_a` 16.6 KB + `buf_b` 4.2 -> 8.3 KB, and both 20.8 KB and 24.9 KB permit exactly one workgroup
per 32 KB, so occupancy from LDS is unchanged at 1 either way.

The way to test it is to widen the tile **without** widening per-thread state — e.g. BM 64 -> 128
with TM 2 / TN 2 held, or BN=32 with TM 1. Neither was tried.

### Reverted

BN is back to 16, both shaders re-spliced (hashes back to `0xBEF709BBFECD0ED1` /
`0x26CE8AE6ACA43BDF`, blob sizes back to 43084 / 57484 bytes), and the A/B reconfirms
**Path 1 75.4 vs Path 2 79.6 t/s**. The `MatMulBatched` guard fix is kept — it was a real latent
bug that would have blocked any future tile widening.

**Standing conclusion: the Vulkan prefill prize is NOT simply "raise the chunk".** Weight
amortization is achievable and measurable, and on this hardware it is not the binding constraint.
Anyone resuming this should re-derive the ceiling from a tiled-kernel sweep, not the matvec one.

## Item 3 (Flash128) gate 1 RETRACTED: static-abstract generics do NOT preserve the folding

The earlier scoping section proposed two ways to widen `attn-bench` without destroying the JIT's
constant folding: *"either a generic struct type-parameter supplying HeadDim (the JIT specializes
per instantiation and keeps the constants) or two explicitly compiled paths."*

**The first option is now empirically refuted.** I built it — `AttnBench<TShape> where TShape :
struct, IAttnShape`, with `NumHeads`/`HeadDim` as C# 11 `static abstract` members returning
literals, `QDim`/`KvDim` derived from them — and ran the 64-vs-64 control arm against a
reconstructed pre-refactor build. Same args (`3218 8192 9`), same machine, interleaved:

| arm | best (3 runs) | median (3 runs) |
|---|---|---|
| control, `const int HeadDim = 64` | 391.9, 389.7, 399.5 | 474.9, 476.0, 487.7 |
| specialized, generic `Shape64` | 500.6, 523.7, 612.8 | 616.1, 591.8, 622.7 |

**~25-30% slower on identical arithmetic.** Value-type generic instantiation gives each shape its
own code, but a `static abstract` property is not the same thing as a `const`: the 103 references
become property accesses that the JIT may inline yet evidently does not promote to compile-time
constants for the strength reduction, bounds elimination and unrolling the literals were driving.

**This is the trap firing on the mitigation designed to avoid it.** Had I skipped the control arm
and gone straight to a 128 measurement, I would have compared const-folded-64 against
generic-128, seen roughly the expected slowdown for double the head dimension, and reported a
completely fabricated Flash128 baseline. The control is the only reason this is visible — and it is
the second time in this programme that reasoning about emitted code was wrong in the same direction
(the Q6_K pshufb rewrite was the first).

**Harness reverted** to the `const int` version, which reproduces 389.7/476.0. No 128 capability is
in the tree.

### Next attempt must use two explicitly compiled paths

The remaining option from the scoping note, now the only one. Constraints it has to satisfy:

- the dimensions must remain `const int` in the code the kernels compile against — nothing weaker
  has survived measurement;
- both shapes must be runnable from **one process**, or the comparison picks up process-level
  variance on top of a harness whose sd is already 20-45 on a ~400-600 ms measurement;
- one source of truth for the kernel bodies, since hand-duplicating 1182 lines guarantees drift.

That points at generating two `.g.cs` copies of the kernel class from a single template at build
time (different namespace + different `const` block per copy), not at `#if` with two exes. Note
`tools/*` is **gitignored**, so this harness has no committed baseline at all — the control above
had to be reconstructed by inverting the refactor, and any future baseline claim has the same
problem until that changes.

### What was built (superseded by the retraction above)

`tools/attn-bench/Program.cs` is now `AttnBench<TShape> where TShape : struct, IAttnShape`, with
`NumHeads`/`HeadDim` supplied as **static abstract** interface members. Each instantiation is
JIT-specialized, so the dimensions inline back to literals and the 103 references keep the constant
folding the old `const int`s gave. A non-generic `AttnBench.Run` dispatches on argv[4]
(`64` default, `128`), and an unknown head dim fails with a message that explicitly says to add
another `IAttnShape` struct rather than make the dimensions runtime values.

This is the trap identified in the earlier scoping section, avoided rather than walked into: a
naive parameterization would have compared const-folded 64 against runtime 128 and charged the lost
folding to head dimension.

**QDim is held equal across shapes** — 32x64 and 16x128 both give 2048 — so total FLOPs, buffer
sizes and row strides are identical and head dimension is the only variable. `Shape128` keeps
KvDim == QDim (MHA) rather than modelling Qwen3-0.6B's GQA, because gate 3 requires scheduling and
arithmetic to stay separate experiments; GQA is a later arm.

It was subsequently built and measured, and the control arm rejected it — see the retraction
above. Kept here only as the record of what was tried.

## Item 3 gate 1 DONE (second attempt): two compiled paths, control arm PASSES

After the static-abstract generic was rejected, built the remaining option from the scoping note:
one kernel source compiled twice.

```
tools/attn-shared/AttnKernels.cs        1197 lines, ONE source of truth
tools/attn-shape64/AttnShape64.csproj   compiles it as-is                -> namespace AttnShapes.H64
tools/attn-shape128/AttnShape128.csproj compiles it with ATTN_HEADDIM_128 -> namespace AttnShapes.H128
tools/attn-bench/                       driver only; references both, dispatches on argv[4]
```

The two projects differ **only** in `DefineConstants`. The dimensions stay `const int` behind an
`#if`, so every one of the 103 references is const-folded exactly as before, and a per-shape
namespace lets both live in one process — which matters because this harness's own sd is 20-45 on a
~400-600 ms measurement, so splitting arms across processes would add variance to a comparison that
is already noisy.

### Control arm passes

| arm | best | median |
|---|---|---|
| pre-refactor `const`, 3 runs | 391.9 / 389.7 / 399.5 | 474.9 / 476.0 / 487.7 |
| **new hd64 path, 2 runs** | **396.5 / 388.9** | **463.8 / 473.5** |

Within noise of the incumbent. The refactor costs nothing, unlike the generic attempt's 25-30%.
Only now is a 128 number worth reading.

### First headDim=128 measurement

| shape | best | median | sd |
|---|---:|---:|---:|
| hd64 = 32 heads x 64 | 396.5 / 388.9 | 463.8 / 473.5 | 32.7 / 38.0 |
| **hd128 = 16 heads x 128** | **361.9 / 328.6** | **379.8 / 367.9** | 10.6 / 23.4 |

At equal QDim (2048 either way), **headDim=128 is ~20% FASTER than headDim=64** on the shipped
general-attention variant, and markedly more stable (sd 10.6-23.4 vs 32.7-38.0). Fewer, longer
per-head inner loops amortize setup better than more, shorter ones.

### What this does and does not say

This is variant **A, "shipped: flat tile=64"** — the *general* attention path, on both arms. It is
the correct **baseline** for Flash128, not Flash128 itself. What it establishes is that the widened
shape is not intrinsically disadvantaged in this harness, which is what had to be true before
writing a Flash128 kernel was worth doing.

It also reframes the item's premise. The doc's original argument was that `headDim=128` models are
stuck on a slower general path. That remains true, but the general path is *not* slower at 128 than
at 64 for equal work — so the Flash128 prize is the Flash-vs-general delta at 128 (variant B already
shows 1.15-1.18x for the register-blocked variant at 64), not a head-dim penalty being recovered.

Next: run the Flash variants at 128 in this harness, then gate 2 proper — Flash128 vs the general
path, interleaved, before touching `ForwardPass.PrefillCoreAttention`.

## Item 3 gate 2: BLOCKED, and the blocker is a shape-hardcoded microkernel

Ran the Flash variants at both shapes. hd64 works; **hd128 crashed the process outright** —
`Fatal error.`, exit 127, no managed exception.

### Root cause

`SimdKernels.GemmF32_64x64_6x2` computes `C[M,64] = A[M,64] * B[64,64]`. **`m` is the only
parameter**; K, N and both row strides are compile-time 64 inside the microkernel. The Flash
variants use it for both GEMM phases, passing head dimension as one of those hardcoded 64s:

- QK: `Scores[tn,kv] = Q[tn,hd] . K[kv,hd]` — K-dim is **headDim**
- PV: `Out[tn,hd] = P[tn,kv] * V[kv,hd]` — N-dim is **headDim**

Four of the seven `FlashScratch` buffers (`Accumulator`, `QPack`, `KPack`, `VPack`) also have a
headDim axis and were allocated `64 * 64` regardless. `Scores` and `RunningMax`/`RunningSum` are
genuinely token-tile sized and were fine — which is why the earlier scoping pass, which checked the
literal 64s at lines 239/291/554/771 and correctly found them to be *token* tiles, still missed
this. The fixed head-dim 64s are in the **allocations**, not the loops.

### Fixed

Scratch is now sized by `HeadDim`, and both flash branches **refuse explicitly** at
`HeadDim != 64` with a message naming the microkernel. Sizing the buffers alone would have been
worse than the crash: it converts a hard failure into silently wrong numbers, because the GEMM would
still contract over 64 of the 128 dimensions and every Flash figure would be plausible and false.

Verified: hd128 flash now exits 0 with the explanation; hd64 flash is unchanged
(A 333.7, B 501.9, C 205.6 — C still 1.62x, matching the pre-change 1.60x/1.61x).

A side effect worth noting: the guards produce CS0162 *unreachable code* in both builds, because
`HeadDim` is a `const` and one side always folds away. Under `TreatWarningsAsErrors` that broke the
build, and it is **direct evidence the constant folding is real** — the thing the whole two-project
construction exists to preserve. `NoWarn CS0162` is set in the two shape projects with that rationale.

### What item 3 actually requires now

Not "reuse the existing online-softmax structure with a wider tile". The prerequisite is a **new
FP32 microkernel** — either `GemmF32_64x128_6x2` (and a K=128 QK variant), or a strided
generalization of the existing one that takes K, N and row strides as arguments. The doc's original
framing ("not a new algorithm, but a new kernel shape") was right; this pins down that the shape
lives in `SimdKernels`, not in the attention code.

Also note the standing measurement: at equal QDim the **general** path is already ~20% faster at
headDim=128 than at 64, and at 64 the best Flash variant is 1.62x the general path. So the Flash128
prize is that 1.62x-ish multiple applied to an already-faster baseline — worth having, but it is not
recovering a deficit, and a new microkernel is the price of entry.

## Units, and a result that reframes the whole Vulkan programme

### attn-bench numbers are MILLISECONDS, not t/s

`AttnKernels.cs:184` records `Stopwatch.GetElapsedTime(t0).TotalMilliseconds`. The `best`/`median`/`sd`
columns are **milliseconds for the isolated attention kernel over N=3218 tokens — lower is better**.
They are not throughput and must not be read as t/s or compared against a prefill t/s figure. The
harness header says why: it uses a flat `[pos][kvDim]` K/V buffer rather than `PagedKvCache`, so
every variant pays the same access cost and the **deltas** are meaningful while the absolute numbers
are not production-faithful.

Restating the headDim comparison precisely, since "~20% faster" quoted only the median:

| statistic | hd64 (32x64) | hd128 (16x128) | hd128 advantage |
|---|---:|---:|---:|
| best, run 1 | 396.5 ms | 361.9 ms | 8.7% |
| best, run 2 | 388.9 ms | 328.6 ms | 15.5% |
| median, run 1 | 463.8 ms | 379.8 ms | 18.1% |
| median, run 2 | 473.5 ms | 367.9 ms | 22.3% |

**9-22% depending on the statistic**, not a flat 20%. hd128's sd is also roughly half hd64's
(10.6-23.4 vs 32.7-38.0), which is why its median gains more than its best.

### CPU prefill is 2.03x FASTER than Vulkan on this machine

Never measured in this programme until now. Same model, same 990-token prompt, interleaved:

| backend | run 1 | run 2 | mean |
|---|---:|---:|---:|
| CPU (`--backend cpu -g 0`) | 164.6 t/s | 163.7 t/s | **164.2 t/s** |
| Vulkan, Path 2, all 24 layers | 80.6 t/s | 80.5 t/s | **80.6 t/s** |

**The Vulkan backend is less than half the speed of the CPU backend for prefill on this box.**

This does not invalidate any Vulkan measurement above — the Path 1 vs Path 2 deltas, the tiled Q6_K
result and the BN=32 negative are all still correct *relative to each other*. But it reframes what
they are worth: the entire Vulkan effort in this session has been optimizing the **slower** backend
by a factor of two, and the 1.07x won there is worth ~3 t/s against an 84 t/s gap.

The hardware explains it: an AMD Radeon **integrated** GPU sharing system RAM, against 12 AVX2
cores. The iGPU has no bandwidth advantage and loses on compute. The 12.33 GiB/s marginal weight
bandwidth measured in the chunk sweep is an iGPU number and was always flagged as generalising to
nothing; this is the same caveat arriving at the top level.

**Consequence for prioritisation.** On this machine the CPU items (1, 3, 4, 5, 6) target the backend
that is actually fast, and item 2's Vulkan work only pays on hardware where Vulkan wins in the first
place — a discrete GPU, which is not available here. Anyone continuing should either measure on a
discrete card before investing further in the Vulkan path, or treat items 4/5/6 as the higher-value
queue. This is the single most important number in this document and it was missing from it.

## Item 4 (Q4_0 native x86) inventory: the gap is confirmed and total

| | references in `SimdKernels.cs` |
|---|---:|
| `Q4_K` | 37 |
| **`Q4_0`** | **0** |

Q4_0 has **no fused SIMD kernel of any kind** on the CPU — no matvec, no batched matvec, no int8
path. The only Q4_0 code is `Dequantize.DequantQ4_0` (Dequantize.cs:519, block size 32, 18 bytes per
block), so every Q4_0 matmul dequantizes a whole tensor to fp32 and then runs the generic float
path. That is the widest per-format gap in this document: Q6_K at least has a fused kernel that is
merely 1.39x off llama.cpp.

`gemma-4-E4B_q4_0-it.gguf` carries 342 Q4_0 tensors — 218 of MiB scale plus 124 of KiB scale.

### Status of the whole list at this point

| # | item | status |
|---|---|---|
| 1 | Q6_K AVX2 scale broadcast | closed, NEGATIVE (6.3x slower, reverted) |
| 2 | Vulkan tiled Q6_K GEMM | landed, 1.07x; BN=32 reverted as a 34% regression |
| 3 | Flash128 | harness done + control passes; kernel BLOCKED on a new SimdKernels microkernel |
| 4 | Q4_0 native x86 | inventory done (above), kernel not started |
| 5 | per-layer head-dim batched prefill | not started |
| 6 | CPU MoE batched prefill | not started |
| - | ARM64 | skipped, no hardware |

Two closed, one partial, three untouched. Read alongside the CPU-vs-Vulkan result above: items 1 and
3-6 all target the CPU backend, which measures **2.03x faster than Vulkan** on this machine, so the
remaining CPU items are the higher-value queue and item 2's further Vulkan work is the lower.

## Item 4: `SimdKernels.DotQ4_0` written and correctness-gated (NOT yet wired, NOT yet timed)

Added the first fused CPU kernel Q4_0 has ever had — `DotQ4_0(byte* row, float* input, int cols)`,
the direct sibling of `DotQ8_0`: Q4_0 weight row against fp32 activations, no dequantize pass.

AVX2 shape per 18-byte block: one 16-byte load of `qs`, split into low/high nibbles (no byte-wise
shift exists, so shift as u16 then mask), bias by -8 using **byte** arithmetic — 0..15 minus 8 wraps
to exactly the right two's-complement sbyte for -8..7, so no widening is needed before the
sign-extending `ConvertToVector256Int32`. Four 8-wide FMAs per block.

**The element order is the trap.** Q4_0 is not interleaved: `qs[j]`'s low nibble is element `j` and
its high nibble is element `j + 16` (`Dequantize.DequantQ4_0` is the authority). A kernel that
interleaved them would produce entirely plausible magnitudes.

### Correctness gate, run BEFORE any timing

`Q40DotParityTests`, 6 tests, all passing:

- `DotQ4_0_MatchesDequantizeThenDot` at cols 32/64/256/2048/2560, 8 random seeds each, against an
  **independent oracle**: `Dequantize.ToFloat32` followed by a plain scalar dot. That oracle shares
  no code with the new kernel and is exactly what production did before it, so agreement means the
  kernel can be substituted without changing any output.
- `DotQ4_0_NibbleToElementMapping_IsLowFirstThenHigh` drives a one-hot input across all 32 elements
  to pin the mapping specifically, since a swap would survive an aggregate check.

**Mutation-verified.** Swapping the two `AccumulateHalf` activation offsets (`inp` <-> `inp + 16`)
makes all 5 parity cases fail; restoring gives back 6/6. The gate detects the exact error it exists
to catch, rather than merely passing.

### Explicitly NOT done

- **Not wired into any dispatch.** `TryResolveQ8Dispatch` and the `DotQ8_0` call sites are untouched,
  so production still dequantizes Q4_0. The kernel is dead code until that is changed, and no
  end-to-end number can be claimed.
- **Not timed, isolated or end to end.** No performance claim is made here at all. The expected win
  is structural (removing a full fp32 materialization of every Q4_0 tensor plus 8x the bytes read),
  but "expected" is not "measured", and this programme has already produced two changes that looked
  like strictly less work and ran slower.

Next: wire it at the `DotQ8_0` dispatch sites behind a dtype check, measure isolated against
dequantize-then-dot, then end to end on `gemma-4-E4B_q4_0-it.gguf` with interleaved arms.

## Item 4 RESULT: Q4_0 fused kernel wired — **6.4x end-to-end prefill**

`MatVecQ4_0` added and `case DType.Q4_0:` wired into `SimdKernels.MatVec`. Before this, Q4_0 fell
to `default:` -> `MatVecDequantFallback`, which for **every row of every matmul** dequantized the
row into a scratch fp32 buffer and then called `DotF32`.

### Measured, gemma-4-E4B_q4_0-it.gguf, CPU backend (`-g 0`), 809-token prompt

| arm | run 1 | run 2 |
|---|---:|---:|
| OLD — `MatVecDequantFallback` | 1.4 t/s | (timed out: 809 tok at 1.4 t/s is ~578 s of prefill alone) |
| **NEW — `MatVecQ4_0` / `DotQ4_0`** | **9.1 t/s** | **9.0 t/s** |

**6.4x.** By a wide margin the largest win in this programme, and it needed no clever arithmetic —
just not throwing away the quantization before doing the work. The old path read 4 bytes per element
where the new one reads 4 bits, and paid a full extra pass over the weights to do it.

The A/B is a real interleave in the sense that matters: the only difference between the two binaries
is the presence of the `case DType.Q4_0:` line, added and removed by script, with a rebuild between.
The control had to be cut short because the old path is slow enough to exceed a 10-minute command
budget on a 809-token prompt — which is itself the result.

### Why this was the biggest item in the list and was ranked third

The doc ranked Q4_0 at priority 3, behind Flash128 and Q6_K, on the reasoning that Q6_K had a
*measured* 1.68x isolated gap while Q4_0 merely had "scalar fallback confirmed". That
under-weighted it badly: a 1.68x gap on a kernel that exists is worth less than the difference
between having a kernel and not having one. Q6_K's item closed at 6.3x SLOWER after a rewrite;
Q4_0's closed at 6.4x faster by writing the missing kernel.

The transferable lesson: **rank by whether the fast path exists at all, not by the size of the
measured gap on paths that do.** A missing kernel is not a small version of a slow kernel.

### Caveats, honestly stated

- Correctness was gated first (`Q40DotParityTests`, 6 tests, mutation-verified) — see the previous
  section. The numerics are the same as dequantize-then-dot to within fp32 reassociation.
- Only `MatVec` is wired. `MatVecDual`, `MatVec2In` and `MatVec4In` still send Q4_0 to their own
  fallbacks, so batched prefill and the multi-input speculative-verify paths have NOT had this
  applied and will show a smaller or no gain.
- Full suite not yet re-run at the time of writing.

### Item 4 full-suite verification, and an OUTSTANDING perplexity gate

```
Tests.ForwardPass  Total: 1270, Errors: 0, Failed: 2, Skipped: 0, Time: 737.358s
```

1270 = the previous 1264 plus the 6 new `Q40DotParityTests`. The 2 failures are the same pre-existing
Gemma4 **Vulkan** pair (`LongDecodeIsCoherent`, `NarrowedKv_MatchesFp32Argmax`) that were failing
before this work and are unrelated to a CPU-side change. No regression.

**The perplexity gate has since been RUN and PASSED — see the section below.** (Left as written
at the time to preserve the record that the win was defaulted-on before its gate was run.)

~~The perplexity gate has NOT been run, and this programme's rules require it.~~ The rule is "never
promote an isolated win to a default without an end-to-end result plus a perplexity gate where
numerics change", and the numerics **do** change: `DotQ4_0` accumulates in a different order than
dequantize-to-fp32-then-`DotF32`, so outputs shift at fp32 reassociation level. `Q40DotParityTests`
pins agreement to 1e-4 relative against the old path's own values, which is a tight bound and much
tighter than e.g. an int8-activation change — but it is not a perplexity result and must not be
reported as one.

Two things make this awkward to close and they should be stated rather than glossed:

1. The control arm is the **6.4x slower** path. A perplexity run over any real corpus on the old
   route costs roughly six times the new one, and the new one is already only ~9 t/s on this model.
2. `case DType.Q4_0:` is already wired in `SimdKernels.MatVec`, i.e. the win is **already the
   default** while its gate is outstanding. That ordering is backwards relative to the rule.

Either run the gate on a short corpus and record it, or revert the wiring behind an opt-in flag
until it is run. Do not leave it defaulted-on and ungated on the strength of the unit tests alone —
the whole point of the rule is that unit-level agreement and corpus-level quality are different
claims.

**Also still on the fallback:** `MatVecDual`, `MatVec2In` and `MatVec4In` route Q4_0 to their own
dequantize fallbacks, so batched prefill and speculative-verify have received none of the 6.4x.
That is the largest remaining piece of item 4 and it mirrors an already-proven kernel.


## Item 4 perplexity gate: PASSED (delta 1.0e-5 NLL)

Corpus: first 3200 bytes of `scripts/kvarn-gate/wiki.test.raw` (wikitext-2 test), `-c 512`,
`gemma-4-E4B_q4_0-it.gguf`, CPU backend, **511 tokens scored**. Same corpus, same context, same
model; the only difference between the two binaries is the presence of `case DType.Q4_0:` in
`SimdKernels.MatVec`, removed and restored by script with a rebuild between.

| arm | mean NLL | perplexity | bucket [1,256) ppl | bucket [256,1024) ppl | speed |
|---|---:|---:|---:|---:|---:|
| OLD - dequantize fallback | 7.033338 | 1133.8086 | 1370.2747 | 938.8435 | 2.18 tok/s |
| **NEW - `MatVecQ4_0`** | **7.033328** | **1133.7973** | 1370.2424 | 938.8469 | **9.44 tok/s** |
| delta | **-1.0e-5** | **-0.011 (-0.001%)** | -0.032 | +0.003 | **4.3x** |

**The quality difference is 1.0e-5 in mean NLL — pure fp32 reassociation noise**, and it happens to
fall very slightly in the new kernel's favour, which is meaningless at this magnitude. Both
positional buckets agree to four significant figures, one moving each way. The change is
quality-neutral and the gate is cleared: `case DType.Q4_0:` is restored and is legitimately the
default.

Note the speed column measures a different workload from the earlier prefill A/B: perplexity scoring
is **4.3x** faster here versus **6.4x** for prefill, because scoring does more non-matmul work per
token. Both are real; they are not the same number and should not be quoted interchangeably.

### On the ordering

This gate should have run **before** the wiring was defaulted on, not after. It passed, so nothing
was shipped wrong — but "it turned out fine" is not the same as "the process was followed", and the
rule exists precisely because the outcome is unknown at the point the decision is made. The
preceding section is left intact rather than edited to look clean.

## CORRECTION: Q4_0 already reaches MatVecDual / MatVec2In / MatVec4In

Two earlier sections of this document state that `MatVecDual`, `MatVec2In` and `MatVec4In` "still
route Q4_0 to their own dequantize fallbacks, so batched prefill and speculative-verify have
received none of the 6.4x", and call fixing that the highest-value remaining work. **That is wrong.**

Reading the actual default branches (SimdKernels.cs:809, :971, :1148):

```csharp
default:
    // Fallback: two sequential MatVec calls. Loses the weight-bandwidth
    // benefit but stays correct for dtypes we haven't specialised yet.
    MatVec(output1, weights, input1, rows, cols, dtype);
    MatVec(output2, weights, input2, rows, cols, dtype);
```

All three delegate to **`MatVec`** — directly for Dual/2In, transitively for 4In (which calls
`MatVec2In` twice). `MatVec` is exactly where `case DType.Q4_0:` was wired. So Q4_0 already gets the
fused `MatVecQ4_0` through every one of those paths, and the dequantize cost is already gone from
all of them.

What those paths lose is **only the weight-bandwidth amortization** — reading each weight row once
and dotting it against 2 or 4 activation columns, instead of re-reading it per input. That is a real
optimization, but it is a much smaller and different prize than "none of the 6.4x", and it is not
obviously the highest-value remaining item.

**How the error happened, since it is a repeatable trap:** I inferred from "Q4_0 is not a `case` in
those switches" that Q4_0 was therefore unhandled there, without reading what their `default:`
actually did. The dispatch is layered — a missing case in an outer switch does not mean a missing
kernel, it means delegation. The comment in the code says so in plain English two lines below the
`default:` label.

The correct statement of the remaining Q4_0 work: add `_2In`/`_4In` variants of `DotQ4_0` so the
multi-input paths amortize the weight row read, worth roughly what the equivalent Q4_K variants are
worth, not 6.4x.

## Item 5 prerequisite: the designated test model CANNOT show the win

Item 5 is "per-layer head dimensions disable the entire batched CPU prefill path", and the plan
names `gemma-4-E4B_q4_0-it.gguf` as the model to prove it on. The gate is real and confirmed at
`ForwardPass.cs:785`:

```csharp
if (_hp.IsMoE || _layerHeadDim is not null)
{
    for (int i = 0; i < N; i++) logits = Forward(tokens[i], startPos + i);   // sequential
    return logits;
}
```

with a matching refusal in `SupportsBatchVerify` (:1573) and `BatchVerify` (:1587). The in-code
comment already says "Phase 8 plumbs per-layer head_dim through the batched paths", so this is known
future work rather than an oversight.

**But plumbing it would deliver nothing on that model.** The batched prefill path's entire value is
weight-bandwidth amortization via `TryMatMulBatchedQ8`, and `TryResolveQ8Dispatch`
(SimdKernels.cs:237-271) handles exactly three dtypes:

| dtype | batched Q8 dispatch |
|---|---|
| Q4_K, Q3_K, Q6_K | yes |
| **Q4_0** | **no — falls to `default:` and returns false** |

Outside the `case DType.Q4_0:` just added to `MatVec`, Q4_0 appears nowhere else in SimdKernels. So
for a Q4_0 model the batched path would be entered and then immediately fall back to a per-token
loop — the same work the sequential path already does. **The two blockers compose: removing the
per-layer-head-dim gate alone changes nothing for `gemma-4-E4B_q4_0-it.gguf`.**

### What this means for the plan

Three options, in increasing cost:

1. **Prove item 5 on a per-layer-head-dim model that is NOT Q4_0** — gemma-4-12B QAT is Q4_K and has
   per-layer head dims, so it isolates the gate cleanly. `download-model.ps1 -Model gemma4-12b-qat`.
   This is the correct way to measure item 5 as specified.
2. Add Q4_0 to `TryResolveQ8Dispatch` (a `DotQ4_0_Q8_0` family plus `_4In`/`_8In` variants) — this is
   really the batched half of item 4, and it is a prerequisite for item 5 paying off on *this* model.
3. Both, if the E4B Q4_0 file specifically must show a gain.

**Do not start the per-layer-head-dim plumbing against the Q4_0 file.** It is a multi-site change
through attention, RoPE and the KV append, and measured on that model it would correctly report ~no
change — which is very easy to misread as "the plumbing didn't work" rather than "the model has no
batched kernel to plumb to".

Recorded before writing any of it, because the ~9 t/s baseline on that model is already known and
the temptation is to treat the gate as the only thing standing between it and a batched speedup.

## Option 2 done: Q4_0 now HAS a batched dispatch — and gemma4 still cannot use it

Added the Q4_0 batched-prefill family and wired it into `TryResolveQ8Dispatch`:

- `QuantizeRowToQ8_0` / `Q8_0ScratchBytes` — 32-element activation blocks, `[d:float32][qs:32 x int8]`
  (36 B). fp32 scale rather than the on-disk fp16: this scratch is never serialized, so there is no
  format to match and fp32 avoids a half round trip on top of already-lossy quantization.
- `DotQ4_0_Q8_0` — integer dot. Weights stay UNSIGNED (0..15) so `maddubs` applies directly, and the
  -8 every nibble carries is applied once per block as `-8 * sum(q8)` via
  `sum((q4-8)*q8) = sum(q4*q8) - 8*sum(q8)`. That identity is what makes it cheaper than dequantizing.
- `DotQ4_0_Q8_0_4In` — unpacks each weight row once and reuses it across four activation rows.

Correctness gated first: 12 tests in `Q40DotParityTests` (6 new), covering SIMD-vs-scalar parity,
lossy-but-bounded agreement with the fp32 path, and `_4In` == four single calls. **Mutation-verified**
— dropping the `-8 * HSumInt16To32(s)` bias correction fails **6 of 12**; restoring gives 12/12.

### End-to-end on gemma-4-E4B_q4_0-it.gguf: 9.3 t/s vs 9.1/9.0 before — NO CHANGE

Exactly as the prerequisite section predicted, now confirmed by measurement rather than inference.
`ForwardPass.cs:785` bails to sequential `Forward` when `_layerHeadDim is not null`, so gemma4 never
reaches `MatMulBatched` at all — the new dispatch is correct and completely dormant on this model.

**The two blockers compose in both directions.** Earlier: "removing the per-layer-head-dim gate alone
changes nothing, because Q4_0 has no batched kernel." Now the converse is measured: adding the Q4_0
batched kernel alone changes nothing, because the gate stops the model reaching it. Neither is
sufficient; item 5's plumbing is now the single remaining blocker for this model.

### What this bought

Not a speedup on gemma4 — a *removed prerequisite*. Any Q4_0 model **without** per-layer head dims
now gets batched prefill for the first time, and item 5's plumbing is now worth doing against this
model because there is finally a batched kernel behind the gate for it to reach. Before this change,
doing item 5 first would have measured ~no gain and looked like a failed plumbing job.

The 6.4x single-token win from `MatVecQ4_0` is unaffected and still active — this is strictly
additive, on a path gemma4 does not currently take.

### Full-suite verification for the Q4_0 batched half

```
Tests.ForwardPass  Total: 1276, Errors: 0, Failed: 2, Skipped: 0, Time: 626.383s
```

1276 = the previous 1270 plus the 6 new `Q40DotParityTests` cases for the Q8_0 path. The 2 failures
are the same pre-existing Gemma4 **Vulkan** pair. **No regression** from adding
`case DType.Q4_0:` to `TryResolveQ8Dispatch`.

Note what this does and does not prove. The suite passing here is weaker evidence than usual for
this particular change, because **the batched Q4_0 path is dormant on every model the suite
exercises** — gemma4 is gated out by `_layerHeadDim`, and no other Q4_0 model is present. So this
confirms the wiring broke nothing; it does **not** confirm the batched kernel is correct in
production use. That confidence comes from the 12 unit tests (mutation-verified), not from the suite.

The perplexity gate for the batched half is still outstanding and is correctly deferred: the path
cannot execute on any available model, so it is not "defaulted on and ungated" — but it becomes live
the moment item 5's gate is removed, and the gate must be run as part of that work, not after it.

## Item 5 scope inventory (measured, not started)

The gate itself is one `if` at `ForwardPass.cs:785`, plus refusals at `SupportsBatchVerify:1573` and
`BatchVerify:1587`. **Removing them is not the work.** Behind the gate, `_headDim` is used **55
times** in `ForwardPass.cs`, and the batched region appears **twice** — two near-parallel blocks
(~1018-1122 and ~1327-1403, the prefill and the verify/snapshot variants) that each independently do:

```csharp
_snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
int qDim  = _numHeads   * _headDim;
int kvDim = _numKvHeads * _headDim;
...
PerHeadPureRmsNorm(qn, _numHeads,   _headDim, _hp.RmsNormEps);
PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
```

Every one of those becomes per-layer: `qDim`/`kvDim` change per layer, the Q/K norms take the
layer's head dim, RoPE takes it, the KV append stride takes it, and `SnapKvSelector` is constructed
from a single head dim it can no longer assume. The scratch buffers sized from `qDim`/`kvDim` must
be sized from `_maxHeadDim` (which already exists, computed at :280-282) with per-layer *views* into
them — exactly the pattern `GpuForwardPass.RunGemma4Layers` already uses on the Vulkan side, where
per-layer views (`qView`, `kView`, ...) are cut from max-sized allocations.

**Duplication is the main risk.** Two blocks doing the same thing with slightly different
surroundings is how a per-layer bug lands in one and not the other, and the failure mode would be
wrong logits only on some layers — the same silent-wrong class as the Gemma4 Vulkan NaN. The Vulkan
implementation is the reference to mirror op-for-op, and it is worth extracting the shared shape
computation rather than editing both blocks by hand.

**Not started.** This was scoped rather than begun because it is a multi-site change that must not
be left half-applied across two near-identical blocks, and because the perplexity gate for the Q4_0
batched path has to run as part of it (removing the gate is what makes that path live on gemma4).
Baseline to beat: **9.3 t/s** on `gemma-4-E4B_q4_0-it.gguf`, 809-token prompt, CPU.

## Item 5 plumbing: shape work DONE and inert; gate NOT yet removed

Applied to **both** batched blocks (prefill ~1032 and verify/snapshot ~1338):

- `int qDim = _numHeads * _headDim` -> `int qDimMax = _numHeads * _maxHeadDim` (same for kvDim), used
  only for the four `NativeMemory.AllocZeroed` calls. `_maxHeadDim` already existed and already sizes
  the PagedKvCache (see the comment at :121), so the cache was never the blocker it looked like.
- Per-layer `layerHd` / `qDim` / `kvDim` declared **inside** the layer loop. Because the buffers are
  allocated once outside the loop and sized for the max, a narrower layer simply packs its rows
  tighter — `qDim`/`kvDim` are the per-token stride for that layer only.
- `PerHeadPureRmsNorm(qn, _numHeads, _headDim, ...)` -> `layerHd`, in both blocks (4 sites).

**The rename was deliberately chosen to make the compiler do the auditing.** Renaming the outer
symbols rather than shadowing them means every remaining use of `qDim`/`kvDim` outside the loop fails
to compile. That found exactly 8 stragglers — the allocation lines, 4 per block — and proved there
were no others hiding in 300 lines of duplicated code. That is a far stronger guarantee than reading
both blocks and hoping they were checked symmetrically, which is the specific risk the scope
inventory called out.

Clean build after each step.

### Deliberately NOT done in this pass

**The gate at `ForwardPass.cs:785` is still in place**, so none of the above executes yet — gemma4
still takes the sequential path and prefill is unchanged at ~9.3 t/s. This is intentional: the shape
plumbing is inert and independently verifiable, whereas removing the gate simultaneously activates
the batched path *and* the dormant Q4_0 batched kernel, which is two live changes at once with a
required perplexity gate between them.

Still outstanding before the gate can come out:

1. `SnapKvSelector` at :1335 is constructed from the model-wide `_headDim` and cannot yet express
   per-layer dims. It is only reached when SnapKV eviction is active, so the gate could be relaxed
   for the non-SnapKV case first and left refusing when `_snapKv` is in play.
2. RoPE / attention / KV-append calls inside the loops still need auditing against `layerHd` — the
   compiler cannot catch these because they take `_headDim` as an ordinary argument that is still a
   valid symbol. This is the one part the rename trick does NOT cover, and it must be read manually.
3. `SupportsBatchVerify` (:1573) and `BatchVerify` (:1587) refusals.
4. The Q4_0 batched perplexity gate, which becomes live the moment the gate is removed.

**No performance claim is made here.** Nothing has changed behaviourally; this is groundwork.

## Item 5: RoPE audit — the batched path was calling the WRONG variant entirely

The manual audit flagged in the previous section (the part the rename trick cannot catch, because
these calls take `_headDim` indirectly) found a bigger problem than a head-dim mismatch.

Both batched blocks called:

```csharp
ApplyRope(qn, startPos + n, _numHeads);      // heads, but NOT layer
```

`ApplyRope` (:2157) reads the model-wide `_headDim` **and** the model-wide rope tables. A per-layer
variant already existed for the sequential path:

```csharp
ApplyRopeLayer(float* x, int pos, int heads, int layer, int layerHd)   // :2171
```

which selects the layer's head dim *and* the SWA rope table (`_ropeCosTableSwa` / `_ropeSinTableSwa`,
`_ropeHalfDimSwa`) when `_isSwaLayer[layer]`.

**So the batched path would have been wrong twice over on gemma4**: the wrong head dimension *and*
the wrong RoPE theta on every SWA layer. gemma4 alternates SWA and global layers, so this would have
corrupted the majority of them. Fixed at all 4 call sites (2 per block) to `ApplyRopeLayer(...,
layer, layerHd)`. Clean build.

This is the concrete justification for not removing the gate in the same pass as the shape plumbing.
Had the gate come out together with the buffer resizing, this would have shipped as "batched prefill
now works for gemma4" and produced plausible-looking but wrong logits on the SWA layers — the same
silent-wrong class as the Gemma4 Vulkan `tanh` NaN, and harder to spot because nothing would be
non-finite.

### Remaining before the gate can come out

1. `SnapKvSelector` (:1335) still built from model-wide `_headDim`.
2. Attention and KV-append call sites inside the loops: still to audit the same way as RoPE was —
   look for a per-layer variant first, since one existed for RoPE and may exist for these too.
3. `SupportsBatchVerify` (:1573) / `BatchVerify` (:1587) refusals.
4. The Q4_0 batched perplexity gate, live the moment the gate is removed.

Still inert: the gate remains, nothing executes, prefill unchanged at ~9.3 t/s. No performance claim.

## Item 5: gate removal ATTEMPTED and REVERTED — batched path still throws

Completed the remaining audit, then removed the gate. It **crashes**:

```
System.ArgumentOutOfRangeException: Specified argument was out of the range of valid values.
  at ForwardPass.Prefill(...) ForwardPass.cs:727
```

Gate restored (`_layerHeadDim is not null` unconditionally again). Verified working immediately
after: `"The capital of France is **Paris."`, prefill 7.4 t/s. **No regression from any of this
work** — the audit fixes below are all in place and all inert while the gate stands.

### Audit fixes that ARE landed (correct, inert, keep)

- `PrefillCoreAttention` (:2304): `int headDim = _headDim` -> `_layerHeadDim?[layer] ?? _headDim`.
  Note its attn-scale line was *already* gemma4-aware (`_layerHeadDim is not null ? 1.0f : ...`) —
  someone had prepared half of this function and not the other half.
- `ApplyRope` -> `ApplyRopeLayer` at 4 sites (previous section): head dim **and** SWA rope table.
- Per-layer `qDim`/`kvDim` strides + `_maxHeadDim`-sized buffers in both blocks.
- `PerHeadPureRmsNorm` -> `layerHd`, 4 sites.
- `cache.Append(layer, kn[kvDim], vn[kvDim])` already uses the per-layer `kvDim`.

### What is still wrong — the honest state

The exception is unlocated. Candidates, in order of suspicion:

1. **`PagedKvCache` per-layer stride.** The cache is sized from `_maxHeadDim`, but `Append` receives
   a span of the *layer's* `kvDim` while reads may assume a fixed per-layer slot stride. A shorter
   append into a max-strided slot is the most likely `ArgumentOutOfRange` source.
2. `SnapKvSelector` (:1018, :1335) — still constructed from model-wide `_headDim`. The gate attempt
   deliberately kept refusing when SnapKV is enabled, so this should NOT be the cause on a default
   run, but it is unverified.
3. `batchAttnOut` consumption after attention — the output projection reads `qDim` per token and may
   still assume the model-wide stride somewhere outside the audited loops.

**Next step is to get the actual stack frame**, not to guess: re-remove the gate and run with the
exception's inner frames printed (the CLI truncated to the `Prefill` frame). One stack trace will
name the site; three hypotheses will not.

### The judgement being recorded

The gate was removed only after the audit *appeared* complete — compiler-verified renames, a found
RoPE bug, and a per-layer attention fix. It still crashed immediately. That is the third time in this
programme that "the audit looks complete" has been wrong about this code, and it is the argument for
the gate existing at all: a crash is the *good* outcome here, because the alternative failure mode
for a half-plumbed per-layer path is plausible-looking wrong logits.

## Item 5 ROOT CAUSE: PagedKvCache is single-kvDim, and my stride design was wrong

Got the real stack frame instead of guessing:

```
System.ArgumentOutOfRangeException
  at PagedKvCache.WriteKv(...)   PagedKvCache.cs:455
  at PagedKvCache.Append(...)    PagedKvCache.cs:334
  at ForwardPass.PrefillCore(...) ForwardPass.cs:1141      <- cache.Append(layer, kn[kvDim], vn[kvDim])
```

`PagedKvCache` holds ONE model-wide `_kvDim`. `Append` computes `keyDst = page + offset * _kvDim`
and `WriteKv` copies exactly `_kvDim` floats out of the supplied span. Passing a span of the
**layer's** `kvDim` (512 on a 256-head-dim layer vs 1024 on a 512 one) is shorter than `_kvDim`, so
it throws.

### The design correction

My plumbing made the per-token **stride** into `batchK`/`batchV` per-layer. That is wrong, and the
cache is what proves it: the cache slot is uniformly `_kvDim` wide, so the batch buffers must be too.

The right split is:

| use | correct value |
|---|---|
| per-token **stride** into batchQ/K/V/AttnOut | `qDimMax` / `kvDimMax` (uniform, matches the cache) |
| head dim for **norms, RoPE, attention math** | `layerHd` (per layer) |
| span length passed to `cache.Append` | `_kvDim` (= `kvDimMax`), padded tail |

So `layerHd` belongs in the arithmetic, NOT in the addressing. This is exactly what the sequential
`Forward` path does — it uses `layerHd` for the math while the cache stays uniformly strided — and
it is what `GpuForwardPass.RunGemma4Layers` does too, where the per-layer *views* are cut from
max-sized allocations but the allocations never shrink.

Concretely, the next attempt should: keep `qDimMax`/`kvDimMax` as the strides everywhere (revert the
per-layer `qDim`/`kvDim` locals to max), keep `layerHd` only where a head dimension is a *quantity*
(`PerHeadPureRmsNorm`, `ApplyRopeLayer`, `PrefillCoreAttention`), and zero the padded tail of each
K/V row so the cache's uniform copy does not read stale data from a previous layer.

### Status

**Gate restored and verified working** — `"The capital of France is **Paris."`. All audit fixes stay
(they are correct and independently right): per-layer attention head dim, `ApplyRopeLayer` with its
SWA table, `layerHd` norms. Only the *stride* decision was wrong.

That the per-layer stride idea survived a compiler-verified rename, a full manual audit and two
review passes — and was only caught by running it — is the fourth time this code has punished
reading over executing. The cache was the component nobody had looked at, because it is not in
`ForwardPass.cs` at all.

## Item 5 review: the gate CANNOT be safely removed yet — three distinct quantities, not two

Applied the stride correction (uniform `qDimMax`/`kvDimMax` strides, `layerHd` for arithmetic) and
re-tested gate removal. It now fails **differently**:

```
System.AccessViolationException: Attempted to read or write protected memory.
   at SimdKernels.DotF32(Single*, Single*, Int32)
   at SimdKernels.MatVecF32 ... (Parallel.For)
```

The KV-cache throw is gone — that fix was right. The new failure is in the **projection matmuls**,
and it exposes that my "two quantities" model was still too coarse.

### There are THREE quantities, and each wants a different value

| quantity | correct value | why |
|---|---|---|
| per-token **stride** into batchQ/K/V/AttnOut | `qDimMax` / `kvDimMax` | `PagedKvCache` has one model-wide `_kvDim` |
| **matmul shape** (rows/cols of `_wq`/`_wk`/`_wv`/`_wo`) | `_numHeads * layerHd` etc. | the weight tensors are only that big — using max reads past the allocation, which is the AV above |
| **head dim** for norms / RoPE / attention | `layerHd` | arithmetic |

My first attempt conflated stride with matmul shape (per-layer both) and hit the cache. The second
conflated them the other way (max both) and hit the weights. Both failed on the first run, which is
the argument for running rather than reviewing — the fix that "obviously follows" from one crash
walked straight into the other.

### Review verdict: NO

The gate stays. Removing it needs the three quantities separated at every site in **both** blocks —
notably the matmul calls must take the layer's row/col counts while addressing max-strided buffers,
which means the calls need explicit `rows`/`cols` arguments rather than deriving them from the
stride. That is a larger change than either attempt so far, and it must be verified by running,
not by reading.

**Current state is safe and verified**: gate in place, `"The capital of France is **Paris."`,
prefill 6.7 t/s. All the genuinely-correct audit fixes remain landed and inert — per-layer attention
head dim, `ApplyRopeLayer` with SWA table selection, `layerHd` norms, max-sized buffers. The stride
locals are now `qDimMax`/`kvDimMax`, which is correct for addressing and wrong only in that the
matmuls read the same locals.

### Recommendation

Item 5 is worth **less** than its position implies and should drop below item 6. It has now consumed
three attempts, is blocked behind a multi-site refactor of two duplicated blocks, and its measured
prize is bounded by whatever batching buys on a model already at ~9 t/s. Item 6 (CPU MoE batched
prefill) is untouched and independent. Flash128's microkernel is a cleaner piece of work than this.

### Full suite after the item 5 audit changes: 1276 / 2 failed, no regression

```
Tests.ForwardPass  Total: 1276, Errors: 0, Failed: 2, Skipped: 0, Time: 619.092s
```

Same count and same two failures (the pre-existing Gemma4 **Vulkan** pair) as before the item 5 work.
Nothing the audit landed has broken anything.

**But the ApplyRopeLayer swap remains effectively unverified, and the green does not change that.**
The swap only behaves differently from `ApplyRope` when `_isSwaLayer[layer]` is true *and* the model
takes the batched prefill path. Models on disk:

| model | SWA layers | reaches batched prefill |
|---|---|---|
| SmolLM2-1.7B, Qwen3-0.6B, Qwen3-8B | no | yes |
| OLMoE | no | no (MoE gate) |
| gemma-4-E4B q4_0 | **yes** | **no** (per-layer head-dim gate) |

**No available model has SWA layers AND reaches the batched path**, so the changed branch never
executes in the suite. This is the same shape of caveat as the Q4_0 batched kernel: the suite proves
the change is inert where it runs, not that it is correct where it matters.

The swap is still almost certainly a latent bug fix — `ApplyRope` unconditionally used the global
rope tables, so a SWA model on the batched path was getting the wrong theta on its SWA layers — but
"almost certainly correct by inspection" is precisely the standard that has been wrong three times
in this file. It needs a gemma2/gemma3-class model (SWA, uniform head dim) to be exercised at all.
`download-model.ps1` has `gemma4-12b-qat`, which is per-layer and therefore ALSO gated out; a
plain gemma2/gemma3 GGUF is what would actually cover this.

Recorded as a known coverage hole rather than left implied by a passing suite.

## Item 6 (CPU MoE batched prefill): baseline measured, and it is NOT a gate removal

**Baseline: 33.2 t/s** prefill, OLMoE-1B-7B-0924-Instruct-Q4_K_M, 812-token prompt, CPU `-g 0`
(decode 26.0 t/s). Note this is 3.6x faster than the gemma4 Q4_0 baseline (9.3 t/s) — OLMoE is a 1B
active / 7B total MoE, so far less work per token.

MoE shares the same `if (_hp.IsMoE || ...)` bail at `ForwardPass.cs:785` as item 5, which invites the
assumption that it is the same kind of fix. **It is not.** Grepping the batched prefill block
(~1049-1345) for expert/MoE handling:

| symbol | occurrences in the batched block |
|---|---:|
| `batchFfnGate` / `batchFfnUp` | 4 / 3 |
| `expert`, `Expert`, `Moe`, `MoE` | **0** |

The batched block implements a **dense FFN only**. There is nothing behind the MoE gate to enable —
the code does not exist. Item 5 was "plumb an existing batched path for a new shape"; item 6 is
"write a batched MoE FFN that does not exist yet".

### What it actually requires

Batching MoE is structurally harder than batching dense, because routing is per token: token 0 may
select experts {3, 17} while token 1 selects {3, 42}. A batched MoE FFN has to

1. route all N tokens first (the router matmul batches trivially — it is dense),
2. build a per-expert token list (the scatter/gather that dense batching never needs),
3. for each expert, run one batched matmul over only its assigned tokens,
4. scatter the weighted results back into per-token outputs.

Step 2 is the whole difficulty and has no analogue anywhere in the current batched path. The payoff
is real — an expert's weights get read once per group instead of once per token that selected it,
which is exactly the weight-amortization argument that made batching worth it for dense — but the
work is a new kernel plus a permutation layer, not a shape fix.

### Honest ranking after this measurement

Of the three open items, **Flash128's microkernel is the cleanest**: it is one self-contained
function (`GemmF32_64x128_6x2`, or a strided generalization of the existing 64x64 taking K, N and row
strides as arguments), it has a working harness with a passing control arm already built, and its
blocker is fully understood. Item 6 needs a new permutation layer; item 5 needs a three-quantity
refactor across two duplicated blocks that has already failed three times.

Recommended order from here: **Flash128 microkernel, then item 6, then item 5.**

## Item 6 LANDED: batched MoE prefill, 3.4x end-to-end, bit-exact under the same kernels

`ForwardPass.MoeFfnBatched` is written and default-on (`STINGRAY_MOE_BATCHED_PREFILL=0` to
disable). MoE prompts no longer bail to the per-token trunk; they route through `PrefillCore` like
dense models do.

### The measurement

Interleaved arms, one flag apart, same binary, `DOTNET_TC_QuickJitForLoops=0`, CPU `-g 0`,
OLMoE-1B-7B-0924-Instruct-Q4_K_M, best of 3:

| prompt | sequential (old) | batched (new) | speedup |
|---|---:|---:|---:|
| 446 tokens | 35.3 t/s | **121.6 t/s** | **3.44x** |
| 1613 tokens | 31.1 t/s | **110.5 t/s** | **3.55x** |
| 2047-token perplexity scoring | 28.4 tok/s | **85.5 tok/s** | **3.01x** |

Decode is untouched (~26-30 t/s in every arm), as it must be — nothing on the decode path changed.
The perplexity number is lower than the prefill number for the same reason it was in item 4: scoring
does more non-matmul work per token, so Amdahl bites harder. All three are real; they are different
workloads and should not be quoted interchangeably.

### The correctness result, which is stronger than expected

With `STINGRAY_CPU_PREFILL_Q8=0` the batched path is **bit-identical** to the sequential trunk —
not close, identical. Two independent confirmations:

- `MoeBatchedPrefillParityTests.BatchedMoePrefill_MatchesSequential_F32` compares the full logit
  vector bit for bit (`SingleToInt32Bits`) and passes.
- The perplexity harness, over 2047 scored positions, reproduces the sequential arm to every printed
  digit: mean NLL 5.247082, ppl 190.0111, and all three positional buckets (83.6768 / 98.6503 /
  381.0527) match exactly.

That is the ideal outcome for a scheduling change: bucketing tokens by expert reorders *which*
dots run together, and changes nothing about the arithmetic.

### The bug that got in the way, because it is the interesting part

The first working version reduced each token's expert contributions **in expert order** — the order
the CSR bucket loop naturally produces — rather than in top-k slot order. Every kernel was
identical to sequential. The result moved the final logits by **up to 0.20**, enough to change the
sampled token on a real prompt.

That number deserves attention. Reassociating 8 fp32 adds should be a last-bit effect; 0.20 on
logits is five orders of magnitude larger. The residual stream amplifies it: a ~1e-7 perturbation
enters at layer 0's FFN output and is carried and re-amplified through 16 layers. **Accumulation
order is not a rounding detail in a deep residual network** — it is a semantic choice, and the only
safe one is to match whatever the reference path does.

Fixing it meant storing *unweighted* down partials per `(token, slot)` and reducing them in slot
order afterwards, which costs one extra pass over `N x k x embDim` floats and a buffer of the same
size (53 MB at 812 tokens, `numActive=8`, `embDim=2048`). That is a real memory cost, paid
deliberately for exactness. The CUDA hybrid's `BatchedRoutedExpertsCpu` makes the same trade for the
same reason, which is a useful corroboration — that path was written independently and arrived at
the identical constraint.

**Diagnostic note:** the divergence was localised by a DENSE control arm — the same
batched-vs-sequential comparison on SmolLM2, where no MoE code runs. It returned exactly 0, which
proved the batched core itself was innocent and the 0.20 was entirely mine. Without the control the
obvious (and wrong) conclusion was "batched prefill just drifts a bit". The control has since been
removed from the test file: once the MoE arm is bit-exact, exactness is a strictly tighter guard
than any control-scaled tolerance, and keeping it would only double the test's runtime.

### Perplexity gate: PASSED (delta -0.0065 NLL, -0.12%)

Corpus: first 12000 bytes of `scripts/kvarn-gate/wiki.test.raw`, `-c 2048`, `--batched`, CPU,
**2047 tokens scored**. Same corpus, same context, same binary — the arms differ only by
`STINGRAY_MOE_BATCHED_PREFILL`.

| arm | mean NLL | perplexity | [1,256) | [256,1024) | [1024,+) | speed |
|---|---:|---:|---:|---:|---:|---:|
| sequential (control) | 5.247082 | 190.0111 | 83.6768 | 98.6503 | 381.0527 | 28.43 tok/s |
| batched, int8 experts | **5.240590** | **188.7814** | 83.6207 | 99.7941 | 372.9631 | **85.47 tok/s** |
| batched, int8 OFF | 5.247082 | 190.0111 | 83.6768 | 98.6503 | 381.0527 | 38.68 tok/s |
| delta (default arm) | **-0.0065** | **-1.23 (-0.65%)** | -0.06 | +1.14 | -8.09 | **3.01x** |

The third arm is what makes this readable: it isolates the ENTIRE quality delta to the int8
activation path, since the batching alone reproduces the control exactly. The int8 expert GEMMs are
the same `MatMulBatched(allowQ8: true)` the dense batched prefill has shipped by default since the
Q8-prefill ship — this extends an already-defaulted numerics decision to expert weights, it does not
make a new one. Buckets move both ways (middle worse, far context better), net slightly in the new
path's favour, which at 0.12% is noise. Gate cleared, and it was run **before** defaulting this time.

Worth noting for anyone tuning: the int8-off arm still runs at **38.68 tok/s vs 28.43**, a **1.36x**
win with *zero* numerics change at all. If a deployment wants MoE prefill faster at literally no
quality cost, `STINGRAY_CPU_PREFILL_Q8=0` gives 1.36x; the remaining 2.2x is what int8 buys.

### What is admitted, and what still bails

`MoeBatchedPrefillSupported` gates the path. Excluded, and each for a reason that predates this work:

| excluded | why |
|---|---|
| TurboQuant KV cache | `PrefillCoreTq` is a separate batched core; not extended |
| `_traceRouters` / `_traceNorms` | both print `_currentPos`, which the batched path does not advance per token — a trace that lied about position is worse than no batching |
| post-attn / post-FFW norm, per-layer output scale, PLE | the batched cores never modelled these, for dense models either; admitting a MoE model that has them would silently diverge from `RunTrunk` |
| per-layer head dims | item 5's blocker, unchanged |

`BatchForwardMulti` and `PrefillPackedMulti` still reject MoE outright. Those are multi-*sequence*
paths, where int8 quantising rows together would make one user's logits depend on who else is in the
batch — a different question from prefill, and deliberately out of scope here.

Callers updated: `Prefill` / `PrefillWithPerPositionLogits` (`ForwardPass.cs:817`) and
`PrefillWithCache` (continuous-batching admission). `PerplexityCommand --batched` no longer rejects
MoE models, since it is now the only way to measure this path's quality at all.

### Remaining open items after item 6

Item 3 (Flash128 microkernel) and item 5 (per-layer head dims, three-quantity refactor). The ranking
from the previous section stands: **Flash128 next**, item 5 last.

### Item 6 full-suite verification

`dotnet build -c Release` clean (0 warnings, 0 errors — `TreatWarningsAsErrors` is on globally, so
that is a real gate). Full suite: **2252 total, 4 failed**. All four accounted for:

| failure | verdict |
|---|---|
| `KnownEnvironmentVariablesTests.ListMatchesSource` | **real, fixed.** Not the new MoE switch (that was registered when it was written) — the drift test caught `STINGRAY_GEMMA4_PROBE`, added earlier in this programme and never listed, plus a pre-existing `STINGRAY_MAX_QUEUED_REQUESTS`. Both registered; the test now passes 8/8. A good demonstration of why that test exists: an unregistered variable makes the CLI warn about the user's own correct spelling. |
| `ConcurrencyLimitTests.BoundedQueue_...` | **flake under load.** 1m40s in the full run (a timeout), passes 4/4 in isolation. Touches nothing on the changed path — server request admission, not the forward pass. |
| `Gemma4VulkanNarrowedKvE2ETests`, `Gemma4VulkanPleE2ETests` | **pre-existing**, the known Gemma4 Vulkan divergence pair. Unrelated to CPU MoE. |

Net: no regression introduced, one latent registry gap closed.

## Item 3 LANDED: Flash-128 unblocked by a strided microkernel — and the flash win holds at hd128

`SimdKernels.GemmF32_6x2` is written: `C[m,n] = A[m,k] * B[k,n]` with independent row strides,
same 6-row by 2-YMM microkernel, same k-loop, same FMA order as the hardcoded
`GemmF32_64x64_6x2`. Flash mode no longer refuses at `headDim = 128`.

### The headline: flash still beats the register kernel at hd128

`attn-bench 512 512 5 flash <hd>`, N=512, ctxLen=512, same box, same process:

| head dim | A: register 8x8 | C: paired 6x2 GEMMs | C vs A |
|---|---:|---:|---:|
| 64 (control) | 8.7 ms | 6.9 ms | **1.26x** |
| **128 (new)** | 7.2 ms | **5.6 ms** | **1.28x** |

This is the question item 3 existed to answer, and the answer is yes: the paired-GEMM flash
formulation's advantage is not an artefact of the 64-dim shape it happened to be built for. The
hd64 row is the control arm and it reproduces the previously-known ratio, so the harness is
measuring the same thing in both columns.

Note `maxrel` reads 4.4E-1 at hd128 — alarming until you look at the control, where hd64 reads
**1.1E+0**, worse. Both are near-zero reference values inflating a relative error whose absolute
magnitude is 6E-6 / 1.2E-5. Pre-existing property of the online-softmax comparison, not a hd128
defect. Recording it because the number looks like a finding and is not one.

### The microkernel A/B, which came out backwards from the hypothesis

The stated worry was that runtime k/n/strides would lose constant folding and be slower — the same
trap the `static abstract` attn-bench arm fell into at 25-30% slower. **Measured at the identical
64x64x64 shape, over 3 independent runs, the strided kernel is FASTER:**

| arm | best ms | GFLOP/s | vs control |
|---|---:|---:|---:|
| hardcoded 64x64x64 | 0.00510 | 102.7 | — |
| **strided 64x64x64** | **0.00473** | **110.9** | **1.077x** |
| strided 64x128x64 (Q·Kᵀ hd128) | 0.01019 | 102.9 | 0.501x |
| strided 64x64x128 (P·V hd128) | 0.00961 | 109.1 | 0.531x |

Reproduced at 1.073x / 1.079x / 1.077x. The mechanism is not subtle: the strided version hoists the
six A-row and six C-row base pointers out of the j and k loops and walks B with a pointer
increment, where the hardcoded one recomputes `(i+n)*64 + j` inside the loop and relies on the JIT
to strength-reduce it. **Better address arithmetic beat folded constants.** The hd128 rows run at
the same GFLOP/s doing exactly 2x the work, so they cost 0.50x per call — clean scaling, no penalty
for the larger shape.

### End-to-end: NO measurable win. Do not quote 1.08x as a prefill number.

Production flash-64 (`ForwardPass.ComputePrefillFlashAttention64Tile`) now uses the strided kernel,
gated by `STINGRAY_FLASH64_STRIDED_GEMM` so both arms live in one binary. Interleaved, 8 samples
each, SmolLM2-1.7B-Q4_K_M, 2431-token prompt:

| arm | mean t/s | median | best | samples |
|---|---:|---:|---:|---|
| strided | 142.7 | 142.7 | 146.2 | 141.9 143.4 136.6 140.3 146.2 141.3 146.2 145.3 |
| hardcoded | 141.1 | 141.9 | 144.2 | 141.6 136.7 137.7 144.2 142.2 142.8 142.8 141.1 |

A 1.1% difference in means against a ±5% within-arm spread is **not a result**. Amdahl explains it
without any appeal to noise: attention is ~27-33% of prefill and this GEMM is only part of that, so
an 8% microkernel win is ~2% end-to-end at best — under this box's floor. The swap is kept because
it is bit-identical (see below), unblocks hd128, and means production exercises one kernel instead
of two; **not** because it made prefill faster. It did not, measurably.

### Correctness

`GemmF32StridedParityTests`, 13 tests. The contract at the overlapping shape is **bit equality**,
not a tolerance — the two kernels share the k-loop and the FMA order, so a single differing bit
means a real indexing or stride error. Checked at m ∈ {0,1,5,6,7,64} (covering the ragged 1-5 row
tail and the empty case) with `accumulate` both ways. The flash-128 shapes and wider-than-extent
strides are checked against an independent triple-loop reference.

**Both mutation-tested**, per the standing rule that a test is not trusted until it has been seen
to fail:

| mutation | caught by | failures |
|---|---|---|
| B row advance uses `n` instead of `ldb` | `Strided_HonoursStridesWiderThanExtents` only | 1 / 13 |
| ragged-row tail ignores `accumulate` | the bit-exact parity theory | 8 / 13 |

The first is the instructive one: **only one test caught it**, because every other shape in the file
is packed (`ldb == n`) and a stride bug is literally invisible on packed data. That test exists for
exactly this and would have been easy to skip as redundant.

### A second blocker that the microkernel work would have hidden

Removing the refusal was not sufficient — the first hd128 run died with a bare `Fatal error.`
(exit 127). `Flash`'s `qPack`/`kPack`/`vPack` were allocated `64 * 64 * sizeof(float)`, hardcoded,
while `FlashScratch` (used by `FlashTileJobs`) was already `64 * HeadDim`. At hd128 that is a 2x
overrun. The hardcoding had been invisible precisely because the refusal meant the path never ran
at any other head dim.

This is the "reading is not running" lesson again, in its exact original form: the previous
analysis identified the microkernel as *the* blocker by inspection, and it was *a* blocker, but the
buffer sizing was a second one that only execution revealed. Both are now fixed.

### Remaining open item

Item 5 (per-layer head dims, three-quantity refactor) is the last one. After that the kernel
programme is complete and the next work is `docs/session-native-inference-runtime-plan.md`.

## Item 5: the three quantities, finally pinned to specific lines

Not implemented yet — but the previous three attempts all failed because the quantities were named
abstractly ("stride vs shape vs head dim") rather than located. They are now located, and the two
observed crash modes map one-to-one onto two specific conflations.

| # | quantity | value | fixed by | used at |
|---|---|---|---|---|
| 1 | **cache head stride** | `_maxHeadDim` | `PagedKvCache` construction (`ForwardPass.cs:319` passes `_maxHeadDim`), so `_kvDim = _numKvHeads * _maxHeadDim` and `ValueAtHead` strides V planes by it | `PagedKvCache.KeyAt` / `ValueAtHead`, every `Append` |
| 2 | **buffer / matmul shape** | `_numHeads * layerHd`, `_numKvHeads * layerHd` | the WEIGHT's actual row count — `_wq[layer]` simply has no more rows than this | `MatMulBatchedCached` row/col args, `PrefillCoreAttention`'s own `qDim` |
| 3 | **arithmetic head dim** | `layerHd` | the layer | RoPE, QK-norm, attention dot length, softmax width |

**Conflating 2 with 1** (using `qDimMax`/`kvDimMax` as the matmul shape, which is what
`ForwardPass.cs:1129-1131` does today) asks a narrow layer's weight for rows it does not have →
`AccessViolationException` in `SimdKernels.DotF32`. That is attempt 2's crash.

**Conflating 1 with 2** (using the per-layer compact width when appending) hands `WriteKv` a span
shorter than the model-wide `_kvDim` it unconditionally copies → `ArgumentOutOfRangeException` at
`PagedKvCache.cs:455`. That is attempt 1's crash.

The current code sits deliberately at the first of those, with a comment at `ForwardPass.cs:1101`
saying strides stay uniform — correct for quantity 1, wrong for quantity 2, which is why the gate
is still closed.

### The design that separates all three

Q and attention-output buffers take the **compact per-layer stride** `_numHeads * layerHd` — the
value both `MatMulBatchedCached` and `PrefillCoreAttention` already want (the latter computes
exactly this as its own local `qDim` at `ForwardPass.cs:2377`, which today disagrees with what the
caller writes). The buffers stay allocated at `_maxHeadDim` width, so a narrow layer just packs
tighter and leaves a tail unused. Nothing needs an output-stride parameter on `MatMulBatched`.

K/V stay compact through the projection, RoPE and QK-norm, and are widened only at the single point
that requires quantity 1: `cache.Append`. Because the cache strides heads by `_maxHeadDim`, that
widening is a **per-head scatter** (head `h` lands at `h * _maxHeadDim`), not a flat copy into a
longer span — the detail most likely to be got wrong, since a flat copy has the right LENGTH and
therefore throws nothing.

### Two findings from the read-through, worth recording separately

1. `PrefillCoreAttention` indexes KV heads as `kvHead * headDim` with `headDim = layerHd`, against a
   cache whose head stride is `_maxHeadDim`. For a narrow layer that reads the wrong bytes. It is
   inert today only because the per-layer gate keeps this function off that path — so opening the
   gate without fixing this is a silent-wrong-answer, not a crash.
2. `ForwardPass.cs:2873` computes `slotStride` — correctly, `_numKvHeads * _maxHeadDim` for
   per-layer models — and the very next line is `_ = slotStride;`. The right value is computed and
   discarded. Whatever that line was meant to feed, it does not.

Both are noted rather than fixed here: they belong to item 5's implementation, and changing them
now would be an unmeasured edit to a path no test currently exercises.

### Item 3 full-suite verification

`dotnet build -c Release` clean. Full suite: **2265 total, 3 failed** (up from 2252 — the +13 are
`GemmF32StridedParityTests`). The three are the same known set as before, minus the env-var drift
test which this session fixed:

| failure | verdict |
|---|---|
| `Gemma4VulkanNarrowedKvE2ETests`, `Gemma4VulkanPleE2ETests` | pre-existing Gemma4 Vulkan divergence pair |
| `ConcurrencyLimitTests.BoundedQueue_...` | the same load flake as the item-6 run; passes in isolation |

No new failures. **Coverage caveat, stated rather than implied:** the strided kernel's production
use is flash-64, where it is bit-identical to the kernel it replaced — so a green suite proves the
swap is inert, not that hd128 is right. The hd128 shapes are covered by
`GemmF32StridedParityTests` against an independent reference and by attn-bench's own maxabs check,
not by any model on disk. No available GGUF drives production flash at headDim 128.

## Item 5 RETRACTED as characterised — it is not a three-quantity refactor

The three quantities were separated as designed, the gate was opened, and gemma-4-E4B was run
through it. **It crashed on something else entirely**, and the crash is the finding.

### What was implemented and KEPT (all of it inert on today's default paths)

| change | why it is right regardless |
|---|---|
| `PrefillCoreAttention` takes the cache's head stride (`_maxHeadDim`) for its `KeyAt` head offset instead of the layer's `headDim` | quantity 1. `ValueAtHead` already strided by `_maxHeadDim` internally; `KeyAt` returns the row base so the caller owned the offset, and it was wrong. Identical for non-per-layer models (`_maxHeadDim == headDim`). |
| `PrefillCore`'s `qDim`/`kvDim` are now `_numHeads * layerHd` / `_numKvHeads * layerHd`, not the padded max | quantity 2. Identical for non-per-layer models. |
| `ScatterToCacheStride` + zeroed K/V staging rows, used only when `_layerHeadDim is not null` | quantity 1 at the one point that needs it. A per-head scatter, not a pad — a flat copy has the right LENGTH and would throw nothing while misplacing every head after the first. |

Full suite green with these in (see below): they change nothing for any model on disk today.

### What execution revealed that inspection had not

Opening the gate produced an `AccessViolationException` in `ForwardPass.PerHeadRmsNorm`, via
`ApplyQkNorm` — a call the batched block makes with the model-wide `_headDim`. A per-layer sibling,
`ApplyQkNormLayer(q, k, layer, layerHd, kvHeads)`, already existed and the batched block simply did
not use it. Fixing that one call is trivial; tracing *why* it takes a `kvHeads` argument is what
opened the floor.

The sequential trunk's per-layer block (`ForwardPass.cs:1984-2100`) carries **five** more per-layer
concepts the batched core models nowhere:

| # | concept | source | batched core has |
|---|---|---|---|
| 4 | per-layer KV HEAD COUNT | `_hp.LayerKvHeads[layer]` — gemma4 mixes MQA global layers with GQA SWA layers | `_numKvHeads`, model-wide |
| 5 | KV-layer SHARING | `_layerKvSrc[layer]` — shared layers skip K/V projection AND attention reads `effLayer`'s cache | nothing |
| 6 | `attention_k_eq_v` | global layers carry no `attn_v` tensor at all | nothing |
| 7 | gemma4 V per-head `PureRmsNorm` before the cache write | `ForwardPass.cs:2085-2088` | nothing |
| 8 | **SLIDING-WINDOW attention** | `windowSize` from `_isSwaLayer[layer]` | **`PrefillCoreAttention` has no `windowSize` parameter of any kind** |

Number 8 is the actual blocker and it is a different KIND of problem from the rest. Four through
seven are plumbing — tedious, mechanical, low-risk. Sliding-window attention in the batched
prefill core is a **missing feature**: the tiled score/softmax/value loop would need a per-layer
start bound, and flash-64's online-softmax path would need one too.

### Why the previous characterisation was wrong, and how

The "three quantities" framing came from reading the two crash sites of attempts 1 and 2 and
generalising from them. Both crashes were real and both diagnoses were correct — but they were the
first two things to fail, not the complete set, and each fix merely advanced the crash to the next
unmodelled concept. **Fixing what a stack trace points at is not the same as knowing what else is
missing**, and no amount of further reading was going to surface items 4-8; only running it did.

This is the fourth time in this programme that an audit-by-inspection has been wrong about a path
no test exercises. The pattern is now unambiguous enough to state as a rule: for a gated path, the
only trustworthy characterisation is one obtained by opening the gate and running it.

`STINGRAY_PER_LAYER_HD_PREFILL=1` forces the batched path for anyone continuing this work. It is
default-OFF and NOT a supported configuration — with the gate forced, gemma4 gets model-wide KV
heads and no sliding window, i.e. wrong output. It exists so the remaining work is measurable.

### Honest status

Item 5 is **not done and not blocked** — it is mis-sized. Realistically it is "port the sequential
trunk's gemma4 feature set into the batched core, starting with sliding-window attention", which is
its own piece of work with its own design, not the tail of this kernel programme.

## Kernel programme: CLOSED

| item | outcome |
|---|---|
| 1 — Q6_K AVX2 scale broadcast | closed NEGATIVE (no win) |
| 2 — Vulkan tiled Q6_K GEMM | landed, 1.07x, declines 1418 → 2 |
| 3 — Flash128 microkernel | **landed**; hd128 unblocked, flash wins 1.28x there; strided kernel 1.077x isolated, no measurable end-to-end effect |
| 4 — Q4_0 x86 kernel | landed, **6.4x** prefill, perplexity-gated |
| 5 — per-layer head dims | **retracted as characterised**; re-scoped above, needs SWA in the batched core |
| 6 — CPU MoE batched prefill | **landed**, **3.4x**, bit-exact with int8 off, perplexity-gated |

Standing reframing that outlived the programme: **CPU prefill is ~2x faster than Vulkan on this
box**, so CPU work is worth about twice what Vulkan work is.

Next work is `docs/session-native-inference-runtime-plan.md`. Starting it now.

---

# Q6_K RE-MEASURED 2026-08-04 — 1.71x, and the "1.39x correction" was the bad number

Re-run because a stale figure was about to be quoted a third time. Both arms measured **together**,
interleaved, on a quiet box (CPU 6%), `reps=50`, `k=8192 rows=512`.

| arm | build | median | spread |
|---|---|---|---|
| llama.cpp `ggml_vec_dot_q6_K_q8_K` | ggml `c0bc8591e`, MSVC `/arch:AVX2`, `GGML_CPU_ALL_VARIANTS=OFF` | **0.1200 ms** | ±0.6% |
| OpenTail `SimdKernels.DotQ6K_Q8K` | .NET 10 Release, `DOTNET_TC_QuickJitForLoops=0` | **0.2050 ms** | ±2.7% |

**Ratio 1.71x**, identical whether taken best-of-best or median-of-median. Checksums agree exactly
(`2363.599609` both sides) — the harness's mandatory gate, checked before any timing was read.

Raw pairs (llama.cpp / OpenTail, ms): 0.1199/0.2050, 0.1210/0.2161, 0.1209/0.2054, 0.1200/0.2044,
0.1196/0.2046.

## The §"Baseline correction" figure of 1.39x is withdrawn

That correction moved the *OpenTail* arm (0.2487 → 0.2063) and left llama.cpp's at a stored 0.1481.
Our arm reproduces almost exactly (0.2050 vs 0.2063). **llama.cpp's does not**: it measures 0.1200,
i.e. the stored 0.1481 was ~23% slow. So the correction fixed one arm against a stale constant for
the other — precisely what that same section warned against two paragraphs earlier ("re-measure both
arms together rather than comparing against a stored constant"). The original 1.68x was close to
right; 1.39x never was.

## The harness has a warmup fairness bug — `reps=8` inflates the C# arm ~6x

At the README's own default `reps=8`, the C# arm reports **1.2720 ms**. At `reps=30` it reports
**0.1988 ms**. Same code, same box, same shape — the difference is entirely JIT tier-up.

This is not symmetric between the arms and therefore not self-cancelling:

- The **C++ arm has no warmup at all**. It is fully optimised from the first instruction and pays
  only cache/page-in cost.
- The **C# arm ramps** through tier-0 → tier-1. `DOTNET_TC_QuickJitForLoops=0` (which the harness
  already forces) removes the worst case, where a loop method stays stuck in unoptimised tier-0, but
  it does not make the first calls free.

So sharing one `reps` between the arms systematically penalises C#. At `reps=8` the harness would
have reported **9.4x** (1.2720 / 0.1354) instead of 1.71x — a wrong answer, in the direction that
flatters the conclusion the harness exists to test.

**Best-of-N does not save you here**, which is the counter-intuitive part: with N too small, every
sample including the best one is still a warmup sample.

Fix the harness rather than relying on humans remembering the README note: run an explicit discarded
warmup phase, then measure, as BenchmarkDotNet does (own process, pilot phase, warm up until the
measurement statistically converges, then measure). Until that lands, **do not run this harness below
`reps=30`.**

Note NativeAOT — which this project already targets — removes JIT warmup entirely but does **not**
improve steady-state codegen (it is RyuJIT in AOT mode). It would fix the benchmark-hygiene problem
and change nothing about the 1.71x.

## What 1.71x implies end-to-end, and the evidence gap under it

Q6_K is ~27.5% of weight bytes on SmolLM2 (144 q4_K / 25 q6_K / 49 f32), and on this model Q6_K sits
in `ffn_down` and `attn_v` — inside the FFN and QKV stages, which the per-stage profile puts at
**88% of prefill at 1K**, and which are exactly the stages carrying the deficit.

Taking Q4_K as at parity (it is a literal port — see below for why that assumption is NOT yet
evidence):

```
GEMM ratio = 0.725·1.00 + 0.275·1.71 = 1.195
prefill    = 0.88·1.195 + 0.12·1.00  = 1.172
```

against a measured end-to-end gap of **1.16x at 1K**. The agreement is good enough to take
seriously and too good to trust on its own — weight-byte share is not time share, and the non-GEMM
12% is assumed at parity rather than measured.

**The load-bearing assumption is unverified: Path 2 has never been measured against llama.cpp's
Q4_K GEMM in isolation.** Path 2's 1.83x is against *Path 1*, not against `ggml_gemm_q4_K_8x8_q8_K`.
Q4_K is 72.5% of weight bytes, so if Path 2 is even 1.1x behind, it contributes more absolute time
than all of Q6_K.

**That comparison is now possible and was not before.** `tools/kernel-bench/README.md` withholds
Q4_K on the grounds that "OpenTail uses Q8_KS activations for Q4_K while llama.cpp uses Q8_K, so
their algorithms and numerics differ". **Path 2 uses Q8_K.** The stated reason for the exclusion no
longer applies to the path prefill actually takes, and the checksum gate that makes this harness
trustworthy should now work for Q4_K too.

Next experiment, in this order:
1. **Add Q4_K/Path 2 to `tools/kernel-bench`** (both arms, checksum-gated, `reps>=30`). This either
   confirms Q4_K parity — making Q6_K the whole story — or relocates the gap entirely.
2. Only then decide whether Q6_K is worth attacking, and re-read the "Refined hypothesis" section
   above first: the one algorithmic candidate (pshufb scale broadcast) was already tried and was
   **6.3x slower**, so the residual is code generation, not a missing technique.

Caveat on the llama.cpp arm: this build is MSVC. Official llama.cpp Windows binaries are not, and
clang typically generates better AVX2 than MSVC — so 1.71x is more likely an under-estimate of the
application-level Q6_K gap than an over-estimate.

## Where the 1.71x actually is: scale-vector construction, not the dot

Both kernels are otherwise instruction-for-instruction the same — same `m3`/`m15` masks, same 6-bit
reconstruction, same `maddubs` → `madd_epi16(scale, p16)` → int32 `sumi`, same single
`cvtepi32_ps` + FMA per super-block, same `q8sclsub` bsums trick for the -32 offset. The integer-
domain scale application is already present on our side. The divergence is **how the four scale
vectors per 128-element group are built.**

llama.cpp (`arch/x86/quants.c`, in-loop), with `scales` held in a register across the whole block:

```c
const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
```

**2 instructions per scale** — one `pshufb`, one `vpmovsxbw`. No memory traffic; `scales` never
leaves the register file.

Ours (`SimdKernels.cs:3458`):

```csharp
var sc16_0 = Vector256.Create(
    Vector128.Create((short)sc[isc + 0]),
    Vector128.Create((short)sc[isc + 1]));
```

`Vector128.Create` from a **runtime scalar** is a byte load + sign-extend + `vpbroadcastw`, done
twice, then a `vinsertf128` to fuse the halves — **~5-7 instructions per scale**, and it re-reads
`sc[]` from memory on every iteration instead of keeping the scales resident.

Rough count per 256-element super-block: 8 scale vectors x ~5 extra ops = **~24-40 extra
instructions** on a core loop of roughly 58. That bracket contains the measured 1.71x.

### This is the same idea as the reverted item 2 — but item 2 failed for an unrelated reason

Item 2 above tried exactly this (pshufb scale broadcast) and measured **6.3x slower**. The recorded
post-mortem is explicit that the defeat came from the *mask table*, not the technique:
`Q6KScaleShuffle` was a `static readonly Vector128<byte>[]` — a **managed array**, so every access
in the innermost loop paid a static-field load plus a bounds check, against llama.cpp's
`static const` table that folds away.

Checked today: **the current kernel has no managed-array access in its inner loop** — every constant
is a `Vector256.Create` hoisted above the block loop, everything else is raw pointer arithmetic. So
the bounds-check mechanism is absent from the incumbent, and it was purely an artefact of the failed
replacement.

The doc's own "Refined hypothesis, NOT yet tested" lists the fixes. The most promising is not in that
list: since the `j` loop runs exactly twice and `isc = j*8`, **fully unrolling it makes every shuffle
mask a compile-time constant**, so `Vector128.Shuffle(scales128, Vector128.Create(<literals>))`
materialises as a data-section load with no table, no array, and no bounds check.

**Caveat, stated because this document has been wrong here before:** the argument above is an
instruction count, which is precisely the reasoning that produced the 6.3x regression. It is a
hypothesis, not a result. Burden of proof is on the replacement, checksum first, then an interleaved
A/B at `reps>=30`.

## RESULT: the pshufb scale build LANDS — 1.27x isolated, 1.71x -> 1.41x vs llama.cpp. End-to-end null.

Implemented as `SimdKernels.Q6KBuildScaleVectors` and applied to all four Q6_K AVX2 kernels
(single-input plus `_2In`/`_4In`/`_8In`). Eight scale vectors are built once per super-block from
the already-register-resident `scales128` with `pshufb` + `cvtepi8_epi16` and compile-time masks,
then read back inside the `j` loop as a plain stack load.

**Correctness: bit-identical, not approximate.** Checksum `2363.599609` before and after — the same
value the C++ arm produces. `pshufb` yields `[sc[2k] x8, sc[2k+1] x8]` and `cvtepi8_epi16`
sign-extends lane-for-lane, which is exactly what the pair-of-broadcasts produced. **No perplexity
gate is required**, because the output bytes do not move.

### Isolated (5 interleaved pairs each, reps=50, quiet box)

| | llama.cpp | OpenTail | gap |
|---|---|---|---|
| before | 0.1200 ms | 0.2050 ms | 1.71x |
| after  | 0.1151 ms | **0.1619 ms** | **1.41x** |

Kernel **1.27x faster**. Samples after: 0.1631 / 0.1613 / 0.1664 / 0.1619 / 0.1605.

### End-to-end: NULL at both prompt sizes

Two separately-built CLI binaries (md5-verified distinct, to avoid the stale-binary trap recorded
elsewhere in this doc), interleaved OLD/NEW, SmolLM2-1.7B-Q4_K_M, `-g 0 -c 8192`:

| prompt | OLD best | NEW best | OLD median | NEW median |
|---|---|---|---|---|
| 731 tok | 146.4 t/s | 147.8 t/s | 145.9 | 146.9 |
| 3031 tok | 142.7 t/s | 139.2 t/s | 139.4 | 138.6 |

+1.0% and -2.5% by best — both inside this box's ~±5% floor. **No measurable end-to-end change.**

### Why the null was predicted, and what it means for the Amdahl argument above

Prefill uses `_8In`, which builds the scale vectors **once per 8 tokens**. The cost removed here was
therefore already amortised 8x on the path that matters, so at most ~1/8 of the isolated gain was
ever available end-to-end — below the noise floor by construction.

**This also weakens the Amdahl reasoning recorded earlier in this section.** That argument used the
1.71x measured on `DotQ6K_Q8K` — the *single-input* kernel — as though it were the prefill gap. It is
not: prefill never calls it. llama.cpp's `ggml_vec_dot_q6_K_q8_K` *is* called per (row, token) and
does not amortise the scale build, so our `_8In` may already be closer to it than the isolated
harness suggests, in the one direction the harness cannot see.

**The harness measures a kernel prefill does not use.** That is a structural limitation, not a
tuning issue, and it applies to every past and future number produced by `tools/kernel-bench`.

### Decision

**Keep.** Strictly better: 1.27x on the isolated kernel, bit-identical output, no numerics gate
needed, and it removes a real inefficiency on the 1-input and `_2In` paths (which decode and small
batches do use). It is not justified by end-to-end prefill, and should not be cited as such.

### Still open, and now the top lead

Q4_K Path 2 has still never been measured against `ggml_gemm_q4_K_8x8_q8_K`. It is **72.5% of weight
bytes** against Q6_K's 27.5%, and the harness's stated reason for excluding Q4_K (Q8_KS vs Q8_K)
does not apply to Path 2, which uses Q8_K. Add a Q4_K arm to `tools/kernel-bench` at a batched shape
— NOT a 1-input shape, per the limitation above — and gate it on checksum as usual.

## NEXT: Q4_K Path 2 vs `ggml_gemm_q4_K_8x8_q8_K` — design, not yet built

The top remaining lead (72.5% of weight bytes, never compared to llama.cpp). Design worked out
2026-08-04; recorded because three separate traps were found while working it out and re-deriving
them is expensive.

### Trap 1 — there is no public weight-repack in llama.cpp

`ggml/src/ggml-cpu/repack.h` exports under `extern "C"`: `ggml_quantize_mat_q8_K_4x8`,
`ggml_gemm_q4_K_8x8_q8_K`, and the matching `_generic` variants. It does **not** export a weight
repack — only `ggml_backend_cpu_repack_buffer_type()`; `make_block_q4_Kx8` is static in
`repack.cpp`.

So the C++ arm cannot easily produce `block_q4_Kx8` weights. Reimplementing the repack in the
harness would let the two arms consume **differently-laid-out weights**, each internally consistent,
each producing a plausible timing for different work — the exact failure this harness exists to
prevent.

**Fix: make the C# side the producer of the shared bytes.**
1. C#: deterministic Q4_K weight bytes (as `RepackedGemmPath2Tests.Measure` does — synthetic bytes
   with d/dmin constrained to a sane range; the GEMM does not care whether they came from a real
   quantisation) -> `SimdKernels.RepackQ4KMatrix` -> **dump repacked weights + raw f32 activations
   to disk**; run Path 2; print checksum.
2. C++: **load those exact bytes**, run `ggml_gemm_q4_K_8x8_q8_K`, print checksum.

Both arms then consume byte-identical weights by construction. If our repack layout ever diverges
from llama.cpp's, the C++ arm computes garbage and the checksum gate catches it — the right failure
mode.

### Trap 2 — where activation quantisation is timed

`SimdKernels.TryMatMulBatchedQ4Kx8` quantises activations **inside** the call. llama.cpp's
`ggml_gemm_q4_K_8x8_q8_K` expects the caller to have run `ggml_quantize_mat_q8_K_4x8` already, and
`ggml_compute_forward_mul_mat` does it once per matmul. **Time quantise+gemm on the C++ side too**,
or the C# arm is charged for work the C++ arm is not. Both arms should also feed the quantiser the
same raw f32 activations rather than pre-quantised bytes, since the two quantisers are only claimed
bit-identical, not proven so against llama.cpp.

### Trap 3 — a batched shape, and enough warmup

Use a batched shape (M >= 16), NOT the 1-input shape the Q6_K arm uses: see the section above on
`tools/kernel-bench` benchmarking a kernel prefill never calls. And raise warmup — `Bench(..., warmup: 3)`
in `Program.cs:55` and `bench(..., reps, 3)` in `main.cpp:135` are both too low; the C# arm needs
~10+ before it is at steady state, and the arms do not need the SAME warmup, only enough each.

### Also carry over

- Add the `_generic` self-check the Q6_K arm already has: if `ggml_gemm_q4_K_8x8_q8_K` is not
  meaningfully faster than `ggml_gemm_q4_K_8x8_q8_K_generic`, the harness linked the scalar variant
  and must fail loudly rather than report an inverted result.
- Note in the harness that a plain sum checksum is **layout-invariant** and so cannot catch a
  transposition. Layout correctness is pinned by `RepackedGemmPath2Tests` against an exact float
  reference at trunk shape; this harness only proves both arms did the same work on the same data.
- Delete the stale `Program.cs:75` message ("Q4_K is intentionally withheld until its distinct
  activation format has a byte-identical weight-input path") once the arm lands — Path 2 uses Q8_K,
  so the stated reason no longer applies to the path prefill actually takes.

---

# Lead A step 1 (2026-08-04): BF16 KV is 1.44-1.50x faster ABOVE L3, 15% SLOWER below it

Isolated probe before touching any storage code, as Lead A requires. Not a full attention kernel:
the memory-traffic-dominated part, `dot(q[headDim], k[pos])` over every cached position — what
attention phase 1 does and where the cache bytes are actually read.

**Probe validity.** The F32 arm is derived from the BF16 arm by widening, so both hold identical
values and the checksums must match exactly — they do (`151.856476` / `224.089971` / `103.190835`,
per shape). The probe refuses to print timings if they diverge.

| headDim | working set (F32) | F32 best | BF16 best | BF16 vs F32 |
|---|---|---|---|---|
| 64 | 64 MiB (> L3) | 2.350 ms / 28.6 GB/s | 1.569 ms / 21.4 GB/s | **1.498x faster** |
| 128 | 64 MiB (> L3) | 2.083 ms / 32.2 GB/s | 1.443 ms / 23.3 GB/s | **1.444x faster** |
| 64 | 8 MiB (fits L3) | 0.124 ms / 67.4 GB/s | 0.146 ms / 28.7 GB/s | **0.851x — SLOWER** |

## The crossover is the finding, not the speedup

Above L3 the kernel is DRAM-bound and halving the bytes buys ~1.45-1.50x. Inside L3 bandwidth is
not the constraint (67.4 GB/s, roughly 2x the DRAM ceiling), so the widen ops are pure added cost
and BF16 **loses 15%**.

This is precisely the hypothesis shape the doc warned about — *"memory-bound so narrowing helps"* is
a hypothesis, and two of that shape were refuted on 2026-08-02. It is true here, but only in one
regime, and asserting it unconditionally would have been wrong.

**Where the crossover sits.** Attention reads one layer's K at a time: SmolLM2 kvDim 2048 x 4 B =
8 KiB per token per layer, so a 16 MB L3 holds roughly 2048 tokens of K. Above ~2K context BF16
wins; below it, it costs. In real inference the KV cache also competes with weights for L3, so the
effective crossover is **lower** than 2K — i.e. more favourable to BF16 than this probe alone shows.

## Verdict: GO to step 2, with a gate

Proceed to the narrowed store in `PagedKvCache`, but the arithmetic stays fp32 and the narrowing
should be **gated on context length** (or accepted as a small short-context cost in exchange for the
1.5 GiB, which is the actual prize). Do not default it on the strength of the long-context number
alone.

Reminder of the standing order for step 3: **the perplexity gate runs BEFORE any default, never
after.**

Probe source: scratchpad `kvprobe` (standalone, no tree changes). Worth promoting into `tools/`
if step 2 proceeds, since the crossover will need re-measuring per model shape.

## Lead A step 2 — SCOPING: the `float*` accessor is the real obstacle, not the page allocation

Before writing any storage change, the read surface was inventoried (reading is not running, but it
is enough to size a change):

- `PagedKvCache.KeyAt(layer, position)` and `ValueAtHead(layer, position, kvHead)` both return a raw
  **`float*` pointing directly into the page**. That contract only holds if the page IS float.
- **68 call sites** across 8+ production files: `ForwardPass`, `HybridForwardPass`,
  `HybridGdnForwardPass`, `CudaHybridForwardPass`, `KvCache`, `TurboQuantKvCache`, `SnapKvSelector`,
  plus tests.
- `_pageBytes = PageSize * _kvDim * 2 * sizeof(float)` and the pool is typed `float*[][]`.

So "narrow the store" is not a localised change to page allocation. Three shapes, none cheap:

| option | memory win | speed win | cost |
|---|---|---|---|
| A. BF16 pages + BF16-native attention kernels (what the probe measured) | yes | yes, 1.44-1.50x above L3 | new kernel variants + rewrite of the read path |
| B. BF16 pages, widen into scratch inside `KeyAt` | yes | **unmeasured — probably lost**, it adds a pass the probe did not model | moderate |
| C. Keep F32 pages | none | none | zero |

**The probe measured option A specifically** — widening fused into the dot. It says nothing about
option B, which is the cheap one. Quoting the 1.5x in support of B would be exactly the
isolated-number-does-not-transfer error this document already records twice.

### Verdict for this step

Step 2 as originally scoped ("narrowed store behind a flag") **understated the work**. It is a
programme, not an edit: the `float*`-returning accessor is an architectural commitment to fp32
storage, and 68 sites depend on it. This is not a negative result — the 1.5 GiB and the
long-context speed are both still real — but it is materially larger than the mini-plan assumed,
and it should be started with that understood rather than discovered halfway through.

**Recommended next move before any implementation:** measure option B in the probe (widen-to-scratch
then dot, versus fused widen-in-dot). If B retains most of the win it is dramatically cheaper than A
and the whole lead becomes tractable. If B loses the win, the choice is an honest one between
1.5 GiB of memory (worth having on its own, given plan §0's bounded-residency property has no other
lever) and a large kernel programme for the speed.

## Lead A step 2 RESULT: option B (the cheap one) does NOT deliver. Option A does, but it is a programme.

Third arm added to the probe (`RunBf16Scratch`): widen the BF16 row into scratch, then dot from
scratch — what `KeyAt` would do if it kept returning `float*` while pages went BF16. All three arms
checksum identically (`151.856476`), so the comparison is sound.

| shape | working set | A: fused widen-in-dot | B: widen to scratch |
|---|---|---|---|
| headDim 64 | 64 MiB (> L3) | **1.498x** | 1.160x |
| headDim 128 | 64 MiB (> L3) | **1.465x** | **1.042x** |
| headDim 64 | 8 MiB (fits L3) | 1.067x | **0.745x** |

**Option B is not a win.** At headDim 128 — Qwen3, i.e. exactly the models with the long contexts
this is supposed to help — it is +4%, indistinguishable from noise. In-cache it is **25% slower**.
The extra store/reload pass costs almost exactly what halving the DRAM traffic saves, so B only
looks good in the one shape it was first measured in.

**This is the isolated-number trap, caught in advance.** Had step 2 been implemented as "narrow the
store, widen in `KeyAt`" on the strength of step 1's 1.5x, the result would have been a large change
delivering ~4% on the models that matter and a regression at short context.

### Correction to step 1's L3-resident figure

Option A at 8 MiB measured **1.067x** here against **0.851x** in the first run. Run-to-run spread in
the L3-resident regime is large (both arms fit in cache, so the measurement is compute-bound and
sensitive to machine state). Treat the earlier "15% slower" as unreliable; across both runs option A
is between **1.07x and 1.50x — never negative**. The crossover claim in step 1 is therefore weaker
than stated: option A appears to win in every regime measured, just by much less when cached.

### Where this leaves the lead

- **Speed requires option A**: BF16-native attention kernels reading the cache directly. That is a
  kernel programme (new variants alongside Flash64 and the general path), not an edit.
- **Memory (1.5 GiB) is available from either option**, and is worth having on its own — plan §0's
  bounded-residency property has no other lever today, and compressed KV cannot back a session.
- `KeyAt`/`ValueAtHead` returning `float*` into the page is the architectural commitment to fp32.
  Any real narrowing has to break that, at 68 call sites, whichever option is chosen.
- Additional design constraint found while sizing B: attention is parallel over heads, so a
  scratch-widening `KeyAt` needs **per-thread** scratch, and returning a pointer into shared scratch
  is unsafe if any caller holds it across calls. That has to be audited, not assumed.

**Loop stopped here.** Not because the lead is dead — the memory win is real and option A is a
genuine speed win — but because the cheap path is now closed by measurement, and what remains is a
multi-session kernel programme whose scope is a call for the user, not something to start silently.

## Lead A step 0 (2026-08-04): decode re-measured — the lead is REOPENED, and one earlier baseline is retracted

The step-2 verdict above left the lead parked on the grounds that the speed half needed a kernel
programme. Before committing to that programme the mini-plan required decode itself to be
re-measured against llama.cpp, since every decode-vs-llama.cpp figure in this document predated the
stale-figure purge. That measurement is below, and it changes the picture.

### RETRACTION: `llama-bench -p 3072 -n 128` does not measure decode at long context

The first baseline taken this session read:

```
tg128 (short ctx)     : 31.15 t/s
pp3072                : 141.37 t/s
tg128 (after pp3072)  : 28.90 t/s      <-- WRONG, this is not "after"
```

and was read as "llama.cpp's decode degrades only 7% across 3K of context". **It does not measure
that.** In `llama-bench`, `-p` and `-n` declare two *independent* tests, each run from a fresh
context; the combined prompt-then-generate test is the separate `-pg <pp,tg>` flag. The third row is
therefore another empty-context `tg128`, and the "7% degradation" is nothing but run-to-run spread
between two identical benchmarks. Any comparison built on it is void.

This is the same class of error as the stale-`llamacpp` figure retracted earlier in this document:
a number that looked like the quantity of interest, was not, and was cheap to check.

### Our decode, measured (3 interleaved runs, `--temp 0 -c 8192 -g 0`, SmolLM2-1.7B-Instruct-Q4_K_M)

```
short ctx  : 26.0, 26.9, 25.8 t/s   (128 tokens generated)
after 3031 : 13.8, 12.8, 12.9 t/s   (20 tokens — EOS stopped generation early)
```

Decode roughly **halves** across 3K of context. The long-context arm generated only 20 tokens
because the model hit EOS; the CLI has no `--ignore-eos`, so that arm has a smaller sample and any
fixed per-generation cost is amortised over 20 rather than 128 tokens. Treat the long figure as
indicative until re-run.

### Why this is bandwidth, not a bug — and why it points straight at KV bytes

SmolLM2-1.7B has `head_count_kv = 32` against `head_count = 32`: **no GQA**. Every head keeps its
own K and V, so the cache is as large as it can be for the parameter count:

```
KV per token = 2 (K,V) x 24 layers x 32 heads x 64 dim x 4 B = 384 KiB/token
at 3031 tokens                                              = 1.11 GiB read per decoded token
model weights (Q4_K_M)                                      = 0.98 GiB read per decoded token
```

Checking that against what was measured:

| regime | bytes/decoded token | measured | implied bandwidth |
|---|---|---|---|
| short ctx | 0.98 GiB | 26.2 t/s | 25.7 GB/s |
| 3031-token ctx | 2.09 GiB | 13.2 t/s | 27.6 GB/s |

**The same effective bandwidth explains both regimes to within 7%.** Decode is memory-bound, the
model is consistent across a 2.1x change in bytes moved, and at 3K context **53% of all bytes read
are KV cache**. This is the strongest evidence yet gathered that option A's target is the real
constraint — it is no longer an isolated-probe extrapolation but a fit to end-to-end throughput.

Predicted effect of halving KV (fp32 -> BF16 store) at 3031 tokens, holding bandwidth constant:

```
2.09 GiB -> 1.54 GiB  =>  1.36x decode
```

That prediction is falsifiable and should be checked against the end-to-end A/B if option A is
built, not assumed.

### A stale negative is sitting in the source and must be corrected

`PagedKvCache.cs` (the remarks on `s_kvBf16`) currently states:

> **Measured cost of the eventual storage change** (scratchpad benchmark, one head, 2431 positions,
> headDim 64): attention is ~8% SLOWER at BF16, not faster — it is ALU/load-port bound, not
> bandwidth bound.

That benchmark's working set is `2431 x 64 x 4 B = 622 KB` for **one head** — it fits in L2. It
measured the in-cache regime, and in that regime the finding is broadly right (the probe puts option
A at 1.07x there, i.e. near-neutral). But decode does not run in that regime: it streams **every**
head of **every** layer, 1.11 GiB at 3K context, which is ~35x the L3 on this machine.

So the comment generalises an in-cache measurement to a streaming workload and lands on the opposite
of the truth for the case that matters. It is exactly the shape of error this document keeps
recording: a real measurement, taken in the wrong regime, left somewhere it will be read as a
verdict. It should be rewritten to say "in-cache: neutral; streaming: 1.44-1.50x faster", with the
working-set caveat attached, before anyone reads it as a reason not to proceed.

### Correction: the accessor surface is 25 production call sites, not 68

The step-2 scoping above put the cost of breaking the `float*`-returning accessor at "68 call
sites". That figure counted production and test files together. Measured properly:

```
KeyAt( | ValueAtHead(   in src/    :  25
                        in tests/  :  45
```

| file | sites |
|---|---|
| `ForwardPass.cs` | 10 |
| `HybridGdnForwardPass.cs` | 4 |
| `CudaHybridForwardPass.cs` | 3 |
| `SnapKvSelector.cs`, `PagedKvCache.cs`, `HybridForwardPass.cs` | 2 each |
| `TurboQuantKvCache.cs`, `KvCache.cs` | 1 each |

Only four declarations exist (`KvCache.KeyAt`, `PagedKvCache.KeyAt`, `PagedKvCache.ValueAtHead`,
`TurboQuantKvCache.Fp32KeyAt`). 45 test call sites still have to compile, but they are mechanical and
several are seam tests that would legitimately keep using an F32 accessor. **The programme is
materially smaller than step 2 concluded** — that conclusion was a factor of ~2.7 pessimistic on its
central cost driver, and the pessimism came from an unchecked grep count rather than a measurement.

## Lead A BUILT (2026-08-04): `STINGRAY_KV_STORE=bf16` — decode +38%, matching the byte model's prediction

The lead was un-parked and implemented. This section records what was built and every measurement,
including the one that was a self-inflicted regression.

### llama.cpp's long-context decode, correctly measured

`-pg 3072,128` reports COMBINED throughput `(3072+128)/total_time`, not decode. Decomposed against
`pp3072` and `tg128` from the **same** invocation (mixing runs makes the derived decode swing between
15.5 and 20.5 t/s, so this must be one run):

```
pp3072        = 136.95 t/s  ->  prefill 22.43 s
pp3072+tg128  = 110.80 t/s  ->  total   28.88 s
                                 decode  6.45 s  ->  128/6.45 = 19.8 t/s
tg128         =  29.60 t/s  (empty context)
```

### The bytes model, fitted across two engines and two regimes

| engine | KV dtype | regime | bytes/token | t/s | implied BW |
|---|---|---|---|---|---|
| llama.cpp | f16 | short | 0.98 GiB | 29.60 | 29.0 GB/s |
| llama.cpp | f16 | 3072 ctx | 1.54 GiB | 19.8 | 30.5 GB/s |
| OpenTail | f32 | short | 0.98 GiB | 26.2 | 25.7 GB/s |
| OpenTail | f32 | 3031 ctx | 2.09 GiB | 13.2 | 27.6 GB/s |

Four points, two independent engines, all within 26-31 GB/s. The long-context gap decomposes as
**1.36x** (bytes) x **1.08x** (llama.cpp's better bandwidth utilisation) = **1.47x** predicted,
against **1.50x** observed. Decode is memory-bound and KV bytes are the lever.

### What was built

- `PagedKvCache`: `STINGRAY_KV_STORE=bf16` halves `_pageBytes`; `WriteKvBf16` writes
  round-to-nearest-even BF16 (NaN-preserving) into the identical page geometry;
  `Bf16KeyAt`/`Bf16ValueAtHead` return `ushort*`. The F32 accessors **throw** under BF16 rather than
  reinterpret 2-byte data as 4-byte — a wrong-format read produces plausible garbage, which is the
  expensive kind of bug. `Compact` (SnapKV) refuses BF16 for the same reason.
- `SimdKernels`: `DotF32Bf16` (accumulator tree copied from `DotF32` so the two stores differ only
  by stored precision, never by reduction order), `AccumulateScaledBf16`, `WidenBf16ToF32`.
- `ForwardPass`: decode's score pass and weighted-V sum branch on a hoisted `bf16` flag; Flash-64
  widens each 64x64 K/V tile once during packing; `PrefillCoreAttentionBf16` covers what Flash-64
  declines.

### Measurement 1 — decode confirmed, prefill self-sabotaged

First cut routed BF16 *around* Flash-64. 5314-token prompt, 3 interleaved runs:

| | F32 | BF16 | ratio |
|---|---|---|---|
| Prefill | 122.2 t/s | 72.5 t/s | **0.59x** |
| Decode | 9.27 t/s | 12.77 t/s | **1.38x** |

**Decode landed on the prediction** (1.38x measured vs 1.36x predicted from bytes alone).
**The prefill loss was not BF16** — it was the missing Flash-64, replaced by the naive mirror.

### The fix, and why the cheap trick that was DEAD for decode is RIGHT here

Flash-64 already packs K and V into F32 tile scratch, and each packed tile feeds a 64x64 GEMM. So
widening once per tile amortises **64-fold**, and every kernel downstream stays bit-for-bit the F32
one. This is option B — widen-to-scratch — which step 2 measured as dead. It is dead for *decode*,
where each byte is consumed once and the extra store/reload pass costs exactly what halving DRAM
traffic saves. It is right for *prefill*, where the tile is reused 64 times. **The two regimes want
opposite designs, and the earlier "option B is dead" verdict was correct only for the streaming one.**

### Measurement 2 — with Flash-64 BF16 packing: prefill neutral, decode 1.47x

Same prompt (5314 tokens), 3 interleaved runs, same binary, flag-selected:

| | F32 | BF16 | ratio |
|---|---|---|---|
| Prefill | 117.2 / 122.4 / 126.4 -> **122.0** | 116.5 / 122.8 / 123.4 -> **120.9** | **0.99x** |
| Decode | 9.5 / 9.4 / 9.4 -> **9.43** | 13.6 / 14.1 / 14.0 -> **13.90** | **1.47x** |

Prefill is neutral (0.99x, inside the run-to-run spread of either arm). Decode is **1.47x**, above
the 1.36x the pure-bytes model predicted — the extra comes from the F32 arm's KV having grown past
what the pure ratio assumed at 5314 tokens rather than 3031.

For scale: this takes decode at long context from **9.43 to 13.90 t/s**, against llama.cpp's derived
**19.8 t/s** at 3072. Not parity — but the remaining distance is now the ~1.08x bandwidth-utilisation
gap plus context-length differences, not a 2x structural deficit.

Fast parity gate after the change: **66 passed, 0 failed** (2.3 s). The F32 path is untouched by
construction — Flash-64's F32 branch is the original code, and the BF16 branch is separate.

### Measurement 3 — memory, measured not asserted

Peak working set, same 5314-token prompt, `-c 8192`:

| store | peak WS |
|---|---|
| F32 | 4.42 GiB |
| BF16 | **3.45 GiB** |

**0.97 GiB saved**, against 0.97 GiB predicted (5314 tokens x 384 KiB / 2). At a full 8192-token
context the saving is 1.5 GiB. This is the half of the lead Dmitri asked to keep alive separately,
and it is now real rather than projected.

## Lead A: STATUS AT HANDOFF

**Built, measured, opt-in, NOT defaulted, NOT committed.** `STINGRAY_KV_STORE=bf16`.

| | result |
|---|---|
| Decode @ 5314 tok | **1.47x** (9.43 -> 13.90 t/s) |
| Prefill @ 5314 tok | 0.99x (neutral) |
| Peak memory | **-0.97 GiB** (4.42 -> 3.45) |
| Fast parity gate | 66 passed, 0 failed |
| Output quality | coherent; diverges from F32 around token ~45 under greedy sampling, as reduced mantissa predicts |

### REMAINING BEFORE THIS COULD BE DEFAULTED ON — do not skip

1. **The perplexity gate has NOT been run.** This is the standing rule ("run the perplexity gate
   BEFORE defaulting on any numerics-changing path, never after") and it is the one gate still open.
   BF16 KV is a real precision reduction; the coherence check above is not a substitute.
   `perplexity -m <model> -f <corpus> -c 2048`, both arms.
2. **RESOLVED. Full ForwardPass suite: 1311 tests, 1309 passed, 2 failed, 11m05s.** Both failures
   named:
   - `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent`
   - `Gemma4VulkanNarrowedKvE2ETests.Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax`

   Both are Gemma4 E4B Q4_0 **Vulkan** tests — exactly the documented pre-existing set. They cannot
   be caused by this change: it is CPU-only, and after the constructor-parameter fix below the
   Vulkan pass provably never receives a BF16 cache.
3. Short-context decode is **slower** (21.7 vs 24.6 t/s at 44 tokens) — the in-cache regime. If this
   is ever defaulted, it should be gated on context length, not switched on globally.
4. Paths that refuse BF16 and throw: `KeyAt`/`ValueAtHead` (all non-dense-CPU readers — hybrid,
   CUDA-hybrid, SnapKV, TurboQuant) and `Compact`. Deliberate: a wrong-format read yields plausible
   garbage. Any of those reaching a BF16 cache is a crash, not corruption — but it IS a crash.
5. `STINGRAY_KV_STORE` added to `KnownEnvironmentVariables` (it warned "not read by this build"
   during measurement); that edit is **not yet compiled**.

### GATE 1 PASSED — perplexity, F32 vs BF16 KV (SmolLM2-1.7B-Q4_K_M, 2047 tokens, ctx 2048)

| | F32 | BF16 | delta |
|---|---|---|---|
| **perplexity** | **22.9159** | **22.8835** | **-0.14%** |
| mean NLL | 3.131833 | 3.130418 | -0.05% |
| bucket [1,256) | 57.2515 | 57.3603 | +0.19% |
| bucket [256,1024) | 19.2830 | 19.2277 | -0.29% |
| bucket [1024,+) | 20.7652 | 20.7414 | -0.11% |

All deltas are under 0.3% and split in BOTH directions, including the position buckets — that is
noise, not degradation. **BF16 KV costs no measurable quality on this model.** The scoring run was
itself 1.17x faster (24.53 vs 20.93 tok/s), an independent end-to-end confirmation of the speed-up
on a workload that is neither the prefill nor the decode benchmark.

Caveat worth keeping: one model, one corpus, 2047 tokens. A GQA model (fewer KV heads) has
proportionally less KV traffic and a different quality profile; this result should not be
generalised to the whole model matrix without re-running.

### Robustness fix found while closing the gates

`STINGRAY_KV_STORE` was read by a global static inside `PagedKvCache`, but `HybridGdnForwardPass`,
`CudaHybridGdnForwardPass` and `VulkanHybridGdnForwardPass` also construct one. Setting the variable
would therefore hand those passes a BF16 cache and make them throw on their first `KeyAt` — an env
var breaking model families that have no BF16 reader.

Now a constructor parameter (`bf16Store`), defaulting false; only the dense CPU `ForwardPass` passes
`PagedKvCache.Bf16StoreRequested`. `ForkSharedPrefix` propagates it explicitly, because the fork
shares pages by reference and copy-on-writes them with a raw byte copy sized by `_pageBytes` — a
mismatch there would be silent corruption rather than a throw.

### GATE 3 — the crossover, measured. It is ~1024 tokens.

Decode, one run per point (decode was stable to +-1% across the 3-run arms above), `-n 96`:

| prompt tokens | F32 decode | BF16 decode | ratio | prefill F32 / BF16 |
|---|---|---|---|---|
| 286 | 26.6 | 22.1 | **0.83x** | 139.7 / 134.8 |
| 1058 | 20.7 | 21.1 | **1.02x** | 143.5 / 147.6 |
| 1989 | 16.7 | 20.6 | **1.23x** | 136.9 / 141.0 |
| 4002 | 11.7 | 15.7 | **1.34x** | 132.2 / 133.6 |
| 5314 | 9.43 | 13.90 | **1.47x** | 122.0 / 120.9 |

**Crossover ~1024 tokens**, monotonic above it. Prefill is neutral at every size (BF16 is marginally
ahead from 1024 up — the Flash-64 tile widening pays for itself once tiles are being reused).

The short-context penalty is REAL and larger than the first 44-token spot check suggested: **-17% at
286 tokens**. Below the crossover the KV fits in L3, the halved DRAM traffic buys nothing, and the
widen ops are pure added cost.

### Why this is NOT being auto-defaulted, despite passing every gate

The obvious gate is "switch on when the context is long". The store format is fixed at cache
construction, so the only quantity available at that moment is the CONFIGURED context (`-c`) — and
that is a bad proxy for the actual sequence length. `-c` defaults to the model maximum, so a user
running 50-token chats at `-c 8192` would be auto-switched into the regime where BF16 loses 17%.
Shipping that proxy would trade a measured 1.47x on long contexts for a measured -17% on the common
short ones, decided by a variable that reflects allocation intent rather than use.

The correct gate is on ACTUAL length, which means converting the cache in place when it crosses the
threshold: re-allocate each page at half width and narrow its contents in one pass. That is cheap in
principle (a single sweep of an already-resident buffer, once per session) but it is a real feature —
it has to interact with page sharing, copy-on-write forks and the warm pool.

**Therefore: BF16 stays explicit opt-in.** It is fully characterised, it is correct, it costs no
quality, and anyone whose workload is long-context can set one variable and take the 1.47x today.
Auto-switching is left as designed-but-unbuilt rather than shipped on a proxy known to be wrong.

## Conversion-at-threshold BUILT: `STINGRAY_KV_STORE=auto`

Supersedes the "not auto-defaulted" verdict above. The objection there was to gating on `-c` (an
allocation hint); gating on ACTUAL length was always the right design and is now implemented.

`ConvertPagesToBf16()` narrows every page in the pool in one pass and flips the cache's format.
Triggered from `IncrementPosition` — a token boundary with every layer's K/V for the current token
already written, and no attention call in flight, which is what makes it safe (readers latch the
format once per attention call). Threshold `STINGRAY_KV_BF16_MIN_TOKENS`, default **1024**, the
measured crossover.

**It refuses when pages are shared.** A shared-prefix fork reads these pages as F32 and copy-on-writes
them with a raw byte copy sized by `_pageBytes`; narrowing underneath it is silent corruption, not a
throw. A cache that is sharing or being shared stays F32 for life. That costs one missed optimisation
on prefix-reuse sessions — the right side to err on.

All pages in the pool are converted, not just currently-allocated blocks: warm-pool pages keep their
allocation, and leaving those wide would hand a full-size F32 page to a BF16 cache after `Reset`.

### Measured — auto picks the winning arm at both ends

| prompt | F32 | BF16 | **auto** | picked |
|---|---|---|---|---|
| 286 tok | 26.6 | 22.1 | **26.0** | F32 |
| 1989 tok | 16.7 | 20.6 | **20.3** | BF16 |
| 4002 tok | 11.7 | 15.7 | **15.3** | BF16 |

Auto tracks the better arm within ~2% at every point, and output stays coherent across the
conversion. `PagedKvCache.Bf16Conversions` counts narrowings so a null A/B can be distinguished from
"the conversion never fired".

### `Compact` (SnapKV) is now BF16-aware rather than throwing

Auto mode made the previous `NotSupportedException` reachable for any SnapKV session crossing 1024
tokens — acceptable when BF16 was explicit opt-in, a trap once it can engage by itself. `Compact` now
widens into its F32 stage buffer and narrows on write-back, so survivor selection and ordering are
identical for both formats.

### Expected position against llama.cpp — NOT a win, and this is the ceiling

llama.cpp's KV is already f16, i.e. the same 2 bytes. Narrowing equalises KV traffic; what remains is
the effective-bandwidth gap (ours 25.7-27.6 GB/s, theirs 29-30.5).

| context | ours (auto) | llama.cpp | ratio |
|---|---|---|---|
| ~286 | 26.0 | 29.60 (measured) | 0.88x |
| ~1989 | 20.3 | ~23.9 (est) | 0.85x |
| ~4002 | 15.3 | ~17.5 (est) | 0.87x |

From ~48% of llama.cpp at long context to a uniform ~85-88%. The residual is a bandwidth-utilisation
problem, not a KV-format one — narrowing cannot fix it, since it is the same constant on both sides
of the ratio. That is the next investigation, and its cause is not yet known.

## Bandwidth utilisation investigated (2026-08-04): the gap was largely an ARTIFACT, and the one concrete mechanism is FALSIFIED

### RETRACTION: there is no uniform ~10% bandwidth deficit vs llama.cpp

Earlier sections put us "uniformly ~10% behind llama.cpp on effective bandwidth". That came from
comparing points at different context lengths without accounting for KV growth DURING generation.
Redone with the average context over the generated tokens:

| point | bytes/token | t/s | achieved BW |
|---|---|---|---|
| llama.cpp, ~64 avg ctx | 0.992 GiB | 29.60 | **29.4 GB/s** |
| ours F32, ~334 avg ctx | 1.102 GiB | 26.6 | **29.3 GB/s** |
| ours F32, ~4050 avg ctx | 2.499 GiB | 11.7 | **29.2 GB/s** |
| ours BF16, ~4050 avg ctx | 1.740 GiB | 15.7 | **27.3 GB/s** |
| llama.cpp, ~3136 avg ctx | 1.554 GiB | 19.8 | **30.8 GB/s** |

**Our F32 path already matches llama.cpp** (29.2-29.3 vs 29.4). The general deficit does not exist.
What remains is narrower: our BF16 path sustains ~27.3 where our own F32 sustains ~29.2, about 6%.

### Hypothesis tested and REJECTED: widen instruction count

llama.cpp's f16 widens with `vcvtph2ps` (F16C) — one uop per 8 values. .NET exposes no F16C, so our
BF16 widen is `vpmovzxwd` + `vpslld`, two. But BF16 is the top half of an F32, so interleaving with
zero yields the widened bits directly: `vpunpcklwd(0, x) == x << 16`, one uop per 8 outputs. The
outputs come out lane-permuted, which is free here because the query is reused across every cached
position and can be pre-permuted once outside the streaming loop.

Probe (`scratchpad/bfwiden`, both arms checksum-identical):

| shape | current (`vpmovzxwd`+`vpslld`) | unpack-with-zero | ratio |
|---|---|---|---|
| headDim 64, 32 MiB stream | 22.5 GB/s | 21.7 GB/s | **0.963x** |
| headDim 128, 64 MiB stream | 23.1 GB/s | 23.0 GB/s | **0.995x** |

**Halving the widen uop count buys nothing.** The loop is DRAM-bound, so ALU work is already hidden;
removing it changes nothing. The engine runs this 12-way threaded — even more bandwidth-saturated
than this single-threaded probe — so the result only holds harder in production.

### Where that leaves it

The residual 6% is **derived, not measured**: "achieved bandwidth" is computed from t/s and an
assumed byte count that ignores activations, output projection, sampling, and the score/softmax
arrays, none of which scale with KV. A 6% signal is inside that model's error bar, and the one
concrete mechanism proposed for it is now measured not to matter.

**Conclusion: there is no demonstrated bandwidth-utilisation deficit to fix.** Do not re-open this on
the strength of the derived table above; anyone resuming it needs a DIRECT measurement (hardware
counters, or an isolated harness around the real decode kernels) rather than throughput arithmetic.
