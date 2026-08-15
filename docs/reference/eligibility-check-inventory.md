# Feature-eligibility check inventory

**Generated:** 2026-07-26 by scanning `src/**/*.cs`. **14 checks.**

Phase 0 deliverable 2 of `04-quality-of-life-improvements-plan.md`: "inventory all
model/backend/feature eligibility checks and their current owners".

**Why this is the deliverable that matters most.** Six times in the performance log a large win
turned out to be a fast path that existed but was not being taken — a gate quietly answered "no"
and nothing said so (per-token weight streaming; per-query K/V; SnapKV disabling the batched
prefill trunk; per-token K/V on CPU; flash-decoding gated at 4096; flash attention gated to fp32
KV). Each was found by measuring which path executed, never by reading the kernel. Every row below
is a place that can silently disable a feature, which is exactly what §5.2 means by "explain every
consequential decision".

**Ownership is the open question.** These predicates live in constructors, static initialisers and
kernel dispatch, so today the answer to "why is speculation off?" is distributed across the call
sites rather than reported anywhere. §7.2 is right that the first move is to EXPOSE the resolved
decision, not to move the logic — relocating it risks changing the shipped default path, which the
plan's own §12 lists as a top risk.

**Scope note:** name-matched on predicate prefixes (`Can*`, `Supports*`, `IsSupported*`,
`TryResolve*`, and the quantization/backend families), then filtered. Deliberately conservative —
enum checks, parser predicates and general helpers are excluded, so treat this as a floor rather
than a complete census.


## OpenTail.Stingray.Cpu

| Check | Location | What it gates |
|---|---|---|
| `TryResolveQ8Dispatch` | `OpenTail.Stingray.Cpu/SimdKernels.cs:234` | Resolve the exact quantizer, scratch size, and dot family for one dtype. Q8_K and Q8_KS are different scratch layouts and are NOT interchangeable (... |
| `CanRepackQ4Kx8` | `OpenTail.Stingray.Cpu/SimdKernels.cs:6377` |  |

## OpenTail.Stingray.Engine

| Check | Location | What it gates |
|---|---|---|
| `IsBatchedPrefillSupported` | `OpenTail.Stingray.Engine/CudaForwardPass.cs:3009` |  |
| `IsKvarnBatchedPrefillSupported` | `OpenTail.Stingray.Engine/CudaForwardPass.cs:3062` |  |
| `IsDensePackablePrefill` | `OpenTail.Stingray.Engine/CudaForwardPass.cs:3919` | Whether the dense packed multi-prompt prefill trunk (issue #193) can run this model: the single-sequence batched-trunk prefill is supported AND non... |
| `IsRawOffloadQuant` | `OpenTail.Stingray.Engine/CudaHybridGdnForwardPass.cs:4152` | (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0) — eligible for the whole-layer batched upload + per-expert view. Q3_K has an in-kernel-dequant GEMM-N (#100) so it too u... |
| `CanNarrowKv` | `OpenTail.Stingray.Engine/GpuForwardPass.cs:390` | Whether a narrowed KV store is actually usable for this model/config — the same conditions the constructor throws on, expressed as a predicate so t... |
| `IsRawGpuQuant` | `OpenTail.Stingray.Engine/GpuForwardPass.cs:3020` | Weight quantizations uploaded to the GPU as raw blocks (dequantized in-shader by the matching VulkanBackend.MatMul matvec dispatch) rather than exp... |
| `IsTextGenerationArchitectureSupported` | `OpenTail.Stingray.Engine/ModelCompatibility.cs:23` |  |
| `IsSupportedWeightDType` | `OpenTail.Stingray.Engine/ModelCompatibility.cs:31` | Matrix weight formats implemented by the portable CPU path. CUDA/Vulkan routes share this conservative baseline at model-load time, so a model cann... |
| `IsKVarNHeadDim` | `OpenTail.Stingray.Engine/TqSupport.cs:41` |  |
| `IsKVarNCudaHeadDim` | `OpenTail.Stingray.Engine/TqSupport.cs:45` |  |
| `IsLloydMaxHeadDim` | `OpenTail.Stingray.Engine/TqSupport.cs:49` |  |

## OpenTail.Stingray.Vulkan

| Check | Location | What it gates |
|---|---|---|
| `SupportsFlashAttention` | `OpenTail.Stingray.Vulkan/VulkanBackend.cs:2769` | Whether  can serve this shape. The kernel's shared buffers are sized at compile time, so head dim and query count are hard limits rather than a slo... |

---

## Opt-in performance flags: fast paths that are OFF by default

A second class of gate, found by auditing which flags are `== "1"` (opt-in, default OFF) versus
`!= "0"` (opt-out, default ON). An opt-in PERFORMANCE flag is a path someone built, and then left
disabled.

**Nine of these ten appear ZERO times in the 59-iteration `perf-loop-progress.md`.** Only
`MATVEC_WIDE8` was ever measured there.

| Flag | Why it is a candidate |
|---|---|
| `STINGRAY_TRUNK_MATVEC_FAST` | named "fast", never measured |
| `STINGRAY_GDN_DECODE_FAST` | named "fast", never measured |
| `STINGRAY_DECODE_CUDA_GRAPH`, `STINGRAY_CUDA_GRAPH` | CUDA graph replay is normally a large decode win |
| `STINGRAY_BATCH_DECODE_GEMM` | batched-decode kernel, off |
| `STINGRAY_BATCH_DECODE_MMQ` | batched-decode kernel, off |
| `STINGRAY_ACT_SOA`, `STINGRAY_ACT_SOA_CPA` | activation LAYOUT change — layout has paid twice (iterations 35, 52) |
| `STINGRAY_PREFILL_FLASH_TC1` | tensor-core prefill flash variant |
| `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL` | chunked prefill, off |
| `STINGRAY_CPU_GDN` | CPU Gated-DeltaNet path |

Diagnostics (`TRACE_*`, `PROFILE_*`, `PROBE_*`, `BYPASS_*`) are excluded — those are correctly
opt-in.

**Off-by-default is NOT evidence of a wrong default.** Three innocent explanations exist and this
audit cannot distinguish them:

1. **Measured and lost.** Iteration 24 is the precedent: a reproduced 2.4-2.6x isolated win that
   was a real 11.9% end-to-end LOSS under 12-way contention. Correctly off, merely unrecorded here.
2. **Correct but slower on this hardware.** No VNNI on this box and an ~8-CU integrated GPU;
   `PREFILL_FLASH_TC1` needs tensor cores that do not exist here.
3. **Incomplete or unsafe.** Parked mid-development.

Also note the perf log covers CPU and Vulkan on one machine. Most of these are CUDA or GDN paths
that the campaign never touched, so "never measured" partly reflects that campaign's scope rather
than neglect.

**Cheapest way to triage, roughly an hour, no benchmarking:** for each flag check (a) whether the
gated path is still reachable and (b) whether a git-log message records a decision. That separates
parked debris from a live untested win. Measure only the survivors, starting with `ACT_SOA` and the
two `*_FAST` flags — layout wins have paid twice, and naming a flag "fast" then leaving it off is
the strongest available signal that someone believed it was faster.
