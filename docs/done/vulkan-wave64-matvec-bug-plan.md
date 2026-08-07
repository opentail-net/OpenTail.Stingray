# Vulkan matvec correctness bug on fixed-Wave64 GPUs (AMD Vega/GCN) — plan doc

> **UPDATE — a second, unrelated Vulkan correctness bug was found after this one was fixed.**
> `MatVecBatchedQ4KInt8` used the `GL_EXT_integer_dot_product` `dotPacked4x8AccSatEXT` intrinsic,
> which is broken on this same AMD GCN/Vega driver — exactly as its `MatVecBatchedQ6KInt8` sibling
> was. It measured **4-8% relative error** against the FP `MatVecQ4K` path at real trunk shapes
> (vs the ~0.4% int8 activation quantization should cost). Compounded over 24 layers this produced
> completely wrong logits and **silently broke `GpuForwardPass.BatchVerifyBatched`**, the Vulkan
> speculative-decode verify trunk, which had no parity test that actually executed on this box
> (the Vulkan BatchVerify tests target `VulkanHybridGdnForwardPass` and skip without their models).
> The pre-existing `MatVecBatchedQ4KMatchesSingleRow` test missed it because a `maxAbs < 1.0`
> tolerance on small well-conditioned synthetic weights cannot detect a *relative* error.
> Fixed the same way (manual scalar dot); see `docs/done/perf-loop-progress.md` iteration 26.
> **Lesson matching this doc's own:** an untested Vulkan path should be assumed wrong, and a kernel
> whose contract is "lossy but argmax-stable" needs a RELATIVE error bound measured at production
> shapes, not an absolute tolerance on a toy matrix.

**Status as of 2026-08-07: the subgroup-width correctness fix is implemented.** The vulnerable
matvec and batched-matvec reductions now use explicit workgroup shared-memory trees rather than
subgroup operations; `Shaders.cs` has no executable `subgroupAdd`/`subgroupElect` use remaining.
The hardware-backed `VulkanWave64SeamTests.MatVecF32Wave64Seam_EveryRowWrittenCorrectly` passed
on 2026-08-07. The real-model CLI smoke is also complete; the remaining work is formal
production-shape, all-format relative-error coverage, not the original implementation.

**Additional hardware evidence, 2026-08-07:** the existing `VulkanShaderTests` execution on the
AMD fixed-Wave64 device reported zero mismatches for the Q4_K, Q5_K, Q6_K, Q8_0, and Q4_0 matvec
checks. Its full-GPU forward-path check also reported the same top token (`Hello`, 19556) and the
same greedy text prefix as CPU. This is encouraging production-kernel evidence, not the distinct
CLI acceptance receipt in Task 4: retain that final `-g 24 --backend vulkan` command and its
token-for-token transcript before declaring the backend release-ready.

## Why this doc exists

The user asked, mid-session, whether work could be delegated to "Vega" (the AMD Radeon iGPU on
this box), remembering it as faster than CPU before. Investigating that surfaced a real,
previously-mislabeled bug: OpenTail.Stingray's Vulkan GPU inference path produces **wrong output**
(not just slow output) on this hardware. This is unrelated to the CPU perf-optimization loop
(`docs/done/perf-loop-progress.md`) and is written up separately because it's a correctness bug, not
a performance question, and needs its own seam-test-driven fix process.

The user also flagged a broader concern after seeing this: that the investigation "seems shoddy
from every angle." That concern is addressed directly in its own section at the end, not
dodged — this bug is itself evidence for part of it, and there's a concrete, honest reason why
(see "What this says about the rest of the investigation" below).

## Symptom

Running the CLI with an explicit full GPU offload (`-g 24 --backend vulkan`, bypassing the
separate VRAM mis-detection issue described below) against
`models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` produces garbage output — blank lines instead of the
expected text — and the existing `VulkanShaderTests` test suite (already in the repo, already
run regularly) fails 15/89 tests with large numeric mismatches:

```
MatVecQ4K:  2019/2048 mismatches (>5% rel error)
MatVecQ6K:  501/512 mismatches
MatVecQ5_K: 129/131 mismatches
MatVecQ4_0: 116/131 mismatches
MatVecQ8_0: 130/131 mismatches
CPU top token: "Hello" (19556)   GPU top token: "\n" (198)
```

The mismatch pattern is distinctive: many GPU output elements are **exactly 0.0000** (never
written at all), and the rest are numerically wrong by large margins — not floating-point
rounding noise.

**This has been happening on every run of `VulkanShaderTests`/`VulkanInitTests` in this
environment.** The CPU perf-loop investigation's progress log (`docs/done/perf-loop-progress.md`) had
been treating the resulting 16 pre-existing failures as *"no Vulkan device on this box"* —
**that explanation is wrong.** A real, working, Vulkan 1.3-conformant device is present
(`vulkaninfo` confirms `AMD Radeon(TM) Graphics`, integrated GPU, `DRIVER_ID_AMD_PROPRIETARY`,
`conformanceVersion 1.3.3.1`). The failures are a real correctness bug that was misdiagnosed
as an environment-availability non-issue, apparently without ever reading the actual assertion
output. See "What this says about the rest of the investigation" for how that happened and what
it means for trusting the rest of the log.

## Root cause

Confirmed at the hardware level, not inferred:

```
$ vulkaninfo | grep -A3 SubgroupSizeControlProperties
VkPhysicalDeviceSubgroupSizeControlProperties:
    minSubgroupSize = 64
    maxSubgroupSize = 64
```

This AMD Vega/GCN iGPU has a **fixed wavefront (subgroup) size of 64** — not configurable, not a
range, exactly 64 always.

`src/OpenTail.Stingray.Vulkan/Shaders.cs` has (at minimum — grep count, not necessarily exhaustive)
**9 separate compute shaders** — `MatVecQ4K`, `MatVecQ6K`, `MatVecQ4_0`, `MatVecQ5_K`,
`MatVecQ8_0`, `MatVecF32`, and others sharing the same pattern — that all:

1. Launch `local_size_x = 256` (8 logical "rows" per workgroup × `THREADS_PER_ROW = 32` lanes
   each, hardcoded via `#define THREADS_PER_ROW 32`).
2. Have each of the 32 lanes in a row-group compute a partial dot-product sum.
3. Call `subgroupAdd(acc)` to reduce those 32 partial sums into one value.
4. Call `subgroupElect()` to pick exactly one lane to write the row's output.

This is correct **only if the hardware subgroup is exactly 32 wide** — the GLSL subgroup
built-ins operate on whatever the *hardware* subgroup actually is, not on the programmer's
intended 32-lane grouping. On this GPU, the real hardware subgroup is 64 lanes, which spans
**two** of the shader's intended 32-lane row-groups. Concretely, for workgroup rows R0 and R1
(lanes 0-31 and 32-63):

- `subgroupAdd` sums all 64 lanes together — R0's and R1's partial sums get added into one
  number, corrupting both.
- `subgroupElect()` elects exactly one lane out of the 64 — so only one of {R0, R1} ever writes
  its output element; the other's `output_data[row]` is left at whatever the buffer already
  contained (zero, in a fresh allocation) — this is the exact "half the outputs are 0.0000"
  pattern seen above.

**This bug was already known and partially addressed — issue #318.** `ComputePipeline.cs` has
an existing mitigation: it queries the device's subgroup-size range via
`VK_EXT_subgroup_size_control` and, when possible, pins the pipeline's `requiredSubgroupSize` to
32 at creation time (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`), forcing the driver
to actually schedule 32-wide subgroups regardless of native wavefront size. The gating check:

```csharp
// ComputePipeline.cs:68-72
internal static bool ShouldPinSubgroupSize32(VulkanBackend backend, int localSizeX) =>
    backend.HasSubgroupSizeControl
    && backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
    && !(backend.MinSubgroupSize == 32 && backend.MaxSubgroupSize == 32)
    && localSizeX > 0 && localSizeX % 32 == 0;
```

**This is the actual gap.** The check requires `32` to fall within `[MinSubgroupSize,
MaxSubgroupSize]`. Issue #318's design assumed AMD Wave64 hardware exposes a *flexible* range
(e.g., RDNA supports both 32 and 64, selectable) — pinning down to 32 was meant to force
subgroup-32 behavior on exactly that class of hardware. But this GPU (older AMD GCN/Vega
architecture) reports `minSubgroupSize = maxSubgroupSize = 64` — a **fixed** 64, not a
selectable range that happens to default elsewhere. `32` is never in `[64, 64]`, so
`ShouldPinSubgroupSize32` returns `false`, no pinning happens, and the shader runs with its
native (wrong-for-this-shader) 64-wide subgroup — **silently**. There is no runtime check that
says "pinning isn't possible and the shader can't tolerate the native width — refuse to use this
kernel / fall back to CPU." It just runs and produces wrong numbers.

**Confirmed this is the same root cause across the whole family**, not just `MatVecQ4K`: grepped
every `THREADS_PER_ROW`/`subgroupAdd`/`subgroupElect` occurrence in `Shaders.cs` — the same
"`#define THREADS_PER_ROW 32`, `subgroupAdd`, `subgroupElect`" pattern repeats across at least 9
shader definitions (`MatVecQ6K` line ~3390, `MatVecF32` line ~3476, two more block-quant
variants around 3527/3632, a batched variant around 3843, another around 4000, another around
4144, plus `MatVecQ4_0`/`MatVecQ5_K`/`MatVecQ8_0`-style ones around 4292/4396/4455 — exact list
needs a final pass before implementation, see Task 1 below). All of them inherit the same
Wave64-vulnerable assumption.

## Two separate, independently-confirmed problems (don't conflate them)

1. **This bug (correctness)**: GPU matvec kernels produce wrong output on fixed-Wave64 hardware.
   Root-caused above. Fixing this is necessary before Vulkan/Vega offload can be used AT ALL on
   this box, for correctness, independent of speed.
2. **A separate VRAM mis-detection bug (already found, not yet fixed, lower priority than #1)**:
   `VulkanBackend.VramBytes` (`VulkanBackend.cs:616-628`) only sums `DEVICE_LOCAL`-flagged
   heaps. On this iGPU that's 512 MB (two 256 MB heaps) — but the real usable GPU memory is a
   third, ~31.4 GiB `HOST_VISIBLE`/`HOST_COHERENT` heap that isn't flagged `DEVICE_LOCAL`
   (classic APU/shared-memory layout). `TierPlanner` sees "0.5 GB VRAM," decides a 1.06 GB model
   doesn't fit, and silently places 0 GPU layers — falling back to CPU without telling the user
   why. Forcing an explicit `-g 24` bypasses this and proves the memory IS usable (the model
   uploads fine) — which is how problem #1 above was even reachable to test.
   **Do not fix problem #2 without fixing problem #1 first** — making GPU offload easier to
   reach by default would just make more users hit the silent-wrong-output bug instead of the
   silent-CPU-fallback. Sequencing matters here.

## Why the existing test suite didn't block this from being unnoticed

`VulkanShaderTests` is not a weak or missing test — it already correctly catches this bug in
detail (the mismatch counts/values above came straight from its own assertions). The gap was
process, not coverage: **whoever wrote the "16 pre-existing Vulkan failures, no device on this
box" note in `docs/done/perf-loop-progress.md` did not read what the failures actually said** — a
"no device" failure and a "computed wrong values" failure look identical at the
pass/fail-count level (`dotnet test` summary) but are completely different at the assertion-log
level, which was sitting right there. That's a real process failure worth naming plainly: a
red/green test count was trusted without reading the red tests' own output.

## Residual validation plan

### Task 1 — Retain and re-run the inventory
The original inventory is now closed: the vulnerable 32-lane reductions have been replaced by
shared-memory trees, and a source scan on 2026-08-07 found no executable
`subgroupAdd`/`subgroupElect` calls in `Shaders.cs`. Keep that scan in future shader reviews;
new subgroup code must either be subgroup-width agnostic or explicitly prove its size contract.

### Task 2 — Fix strategy (decided and implemented)
Two real options, different risk/effort tradeoffs:

- **(A) Rework each kernel's lane grouping to match the hardware's actual subgroup size at
  dispatch time** — e.g. 4 rows × 64 lanes/row instead of 8 rows × 32 lanes/row (still 256
  threads/workgroup, so no dispatch-side changes needed), with each lane's per-block element
  ownership doubled to cover 64 strided elements instead of 32. This preserves the
  subgroup-reduction performance approach but requires re-deriving the element-index arithmetic
  for **every** affected quant format's block layout (Q4_K, Q6_K, Q5_K, Q4_0, Q8_0, F32 all have
  different byte layouts — this is not a mechanical find-replace of the constant). Each reworked
  kernel needs the full hand-computed-reference seam-test treatment this codebase's own
  discipline requires for kernel changes (see `docs/done/perf-loop-progress.md`'s non-negotiable
  discipline section) — no exceptions for "it's just changing a constant," the indexing math
  changes meaningfully.
- **(B) Replace `subgroupAdd`/`subgroupElect` with an explicit shared-memory (`shared float[]`)
  reduction** that doesn't depend on hardware subgroup width at all — classic
  parallel-reduction-in-shared-memory pattern, portable to any subgroup size including future
  unknown hardware. Simpler to reason about correctness-wise (no hardware-width assumption to
  get wrong again), likely a small performance cost vs. native subgroup ops on hardware where
  the 32-pin *does* work (NVIDIA, RDNA) — needs an A/B on non-broken hardware to quantify, but
  this box can't run that A/B (its subgroup is stuck at 64, can't test the pinned-32 path here).

**Implemented choice: (B).** Shared-memory trees are now used for the affected reductions. This
keeps the kernels portable across Wave16/32/64/128 and avoids a device-specific lane-indexing
fork. The subgroup-size pin remains compatibility infrastructure for future shaders, but is no
longer relied upon for these correctness-critical reductions.

### Task 3 — Extend seam and production-shape validation
The F32 Wave64 seam is now a hardware-backed proof that all eight logical rows write correctly.
Q4_K and Q6_K already exercise real SmolLM2 tensors with a relative-error check; Q4_0, Q5_K,
and Q8_0 currently exercise exact synthetic parity at small shapes. Add production-shape cases
for those latter formats when suitable fixtures are available, and retain relative-error plus
top-token/argmax checks where a lossy int8 activation path is involved. Do not treat a synthetic
absolute-only tolerance as a substitute for that measurement.

### Task 4 — Re-verify end-to-end
**Complete 2026-08-07.** The real CLI smoke that surfaced this was re-run with full offload:

    stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -g 24 --backend vulkan -p "The capital of France is" -n 8 --temp 0

It produced `The capital of France is Paris.` after the echoed prompt, exactly matching the
separate CPU (`-g 0`) greedy run. Vulkan reported all 24 layers offloaded to AMD Radeon(TM)
Graphics; the reported decode rates (21.8 Vulkan, 12.8 CPU t/s) are a one-run smoke receipt, not
a performance claim. This closes composition correctness for that reference model; it does not
replace Task 3's per-format production-shape relative-error coverage.

### Task 5 — Only then, revisit performance
Once correctness is fixed, re-measure GPU vs CPU throughput honestly (the earlier smoke test's
6.7-6.8 t/s GPU vs ~20-27 t/s CPU comparison was measured on the *currently-broken* kernels —
wrong output can also mean accidentally-fast-because-it's-doing-less-work, or slow for unrelated
reasons; that number should not be trusted as "GPU is slower" until correctness is fixed and the
comparison is re-run). This is when the original question — "is Vega actually useful here" —
gets a real answer. Right now it's simply unanswered, not "no."

## What this says about the rest of the investigation

The user's concern ("this seems shoddy from every angle") deserves a direct answer, not a
deflection into "here's a new finding instead."

**Concretely wrong, found and corrected this session:**
- The "16 pre-existing Vulkan failures = no device" explanation, repeated across multiple
  iterations of `docs/done/perf-loop-progress.md` without being re-checked, was false — a real device
  is present and the failures are a real correctness bug. This should have been caught the first
  time those 16 failures were seen, by reading the assertion output instead of just the
  pass/fail count.
- Iteration 9's prefill numbers (~34 t/s at TokenCount=64, ~32 t/s at 256) did not reproduce on
  a clean re-run this session (got ~21.5 t/s and ~20.5 t/s instead — see
  `docs/done/perf-loop-progress.md` iteration 10) — likely, per that iteration's own analysis, a
  tiered-JIT confound that was never controlled for in **any** of iterations 1-9's short
  BenchmarkDotNet/CLI runs, not just iteration 9's. That's a real methodological gap sitting
  under most of the numbers in that log, not yet resolved.
- Iteration 4's "12.7% prefetch win" was a false positive from n=3 sampling, caught and reverted
  in iteration 5 — the log documents this honestly, which is good, but it happened at all
  because the box's noise level was underestimated early on.

**What's actually solid:** the CPU-side raw memory bandwidth measurement (36.77 GB/s, tight
stdev, iteration 7), the decode near-parity finding against a freshly-run llama-bench comparison
(iteration 8, verified against a real external binary, not an assumption), and this Vulkan bug's
root cause (traced to specific code, specific hardware properties, reproduced twice). Those share
a common trait the shakier results don't: they were checked against an independent, external
source of truth (a second measurement method, a real competing binary, actual hardware queried
directly) rather than trusted from a single internal benchmark run.

**The honest pattern:** results that were cross-checked against something outside this
codebase's own benchmarks hold up; results that relied on a single internal benchmark or a
skimmed test-failure count did not, twice now. That's a specific, actionable standard to hold
the rest of this work to, not just a vague "be more careful" — before trusting a number or an
explanation, ask what it was checked against, and if the answer is "another run of the same
harness," treat it as provisional.
