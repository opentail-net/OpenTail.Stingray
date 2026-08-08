# GPU review — verification log

Working through `GR_performance.md`. That review states plainly that it performed no build,
execution or benchmark; its findings are code-level conclusions. This log records what happened
when each was checked against the code and the hardware actually present.

**Hardware on this machine (checked, not assumed):**

| | |
|---|---|
| NVIDIA GPU | **none** — no `nvidia-smi`, no CUDA device |
| GPU | AMD Radeon(TM) Graphics (integrated) |
| Vulkan | 1.3.260, AMD proprietary driver |
| Subgroup size | **min = 64, max = 64** (Wave64-locked) |

Consequence: **every CUDA performance finding is permanently unmeasurable here.** Vulkan findings
are testable only where this specific device can exhibit the behaviour — which, as finding 6 shows,
is not a given just because the vendor is AMD.

---

## Finding 6 — subgroup-size pinning is overbroad → **NOT ACTIONABLE (and partly incorrect)**

### The review's claim

> `ComputePipeline.cs` pins subgroup size 32 for essentially every shader whose `local_size_x` is
> divisible by 32 … No active subgroup intrinsic was found in the Vulkan shader sources.
> Consequently, wave32 can be forced on AMD even for kernels that do not need it, potentially
> sacrificing native wave64 behavior.

### What the code actually does

`ComputePipeline.ShouldPinSubgroupSize32` (line 68) requires **all** of:

```csharp
backend.HasSubgroupSizeControl
&& backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
&& !(backend.MinSubgroupSize == 32 && backend.MaxSubgroupSize == 32)
&& localSizeX > 0 && localSizeX % 32 == 0;
```

Which device families does that actually select?

| device | min/max subgroup | pin fires? |
|---|---|---|
| NVIDIA | 32 / 32 | **no** — excluded by the `!(32,32)` clause |
| AMD GCN / this APU | 64 / 64 | **no** — `64 <= 32` is false |
| AMD RDNA | 32 / 64 | yes |
| Intel | varies (8–32) | sometimes |

### Two corrections to the review

**1. "No active subgroup intrinsic" — correct, but I nearly recorded the opposite.** `grep -c
subgroupAdd Shaders.cs` returns 1, which looks like a live intrinsic. It is inside a *comment*
(line 3892). There are genuinely zero active subgroup ops. The shaders were rewritten to
shared-memory tree reductions, which is what fixed issue #318 — so the pin is indeed vestigial for
correctness.

**2. "Sacrificing native wave64 behavior" — this is backwards.** The pin only fires where
`min < 32 <= max`, i.e. **RDNA**, and RDNA's *native* mode is wave32; wave64 there is the legacy
mode executed as two wave32 passes. Pinning 32 on RDNA is aligned with the hardware, not against it.
The family where wave64 genuinely is native is GCN — and on GCN the pin cannot fire, because
`min == 64`.

So the guard already excludes both cases where pinning would plausibly hurt. What remains is a
narrow hygiene point: requiring a specific subgroup size constrains the driver's pipeline selection
for shaders that no longer need any such guarantee.

### Why it cannot be measured here

This device reports **min = max = 64**, so the pin never executes. There is no before/after to
measure. A prior negative result in the code makes the case weaker still — `Shaders.cs:3890-3897`
records that a `subgroupClusteredAdd(acc, 32)` variant was built and measured as the obvious fix for
barrier cost, was bit-identical, and gave **no speedup** (6.82/8.38/8.62 vs 7.07/8.61/8.60 GB/s):

> "these barriers are not what limits this kernel. Reverted rather than carry an extra
> subgroup-extension requirement for nothing. See docs/done/perf-loop-progress.md iteration 27."

### Verdict

**Code hygiene, not a performance fix.** Removing the pin would be safe (no shader depends on it)
but its benefit is unmeasurable on any hardware present, and the mechanism the review proposes for
the benefit is wrong. Not worth changing blind.

### My own error, recorded

I recommended starting here *because it was measurable on this box*. It is not. I confirmed AMD +
Vulkan were present and inferred that an AMD-specific finding was therefore testable, without
checking the device's subgroup range — which is the one property that decides it. Checking took one
`vulkaninfo` call.

This is the same failure that produced the "isolated speedups predict end-to-end" errors during the
CPU work: **confirming a precondition is not the same as confirming the thing.**

---

## Finding 1 — CUDA monolithic module cannot compile below Ampere → **CONFIRMED, partially fixed**

### Verified

- `CudaBackend.cs:7166` compiles `CombinedKernelSource` — one module, every kernel together.
- `CudaBackend.cs:7182` targets the exact detected device: `--gpu-architecture=sm_{_smVersion}`.
- Compile failure **throws** (`nvrtcCompileProgram failed`), so no custom kernel loads at all.
- `CudaTextKernels.cs` contains 16 unguarded `mma.sync` sites: eleven
  `m16n8k32.row.col.s32.s8.s8.s32` and five `m16n8k16.row.col.f32.f16.f16.f32`. The only
  `__CUDA_ARCH__` guards are one isolated block at lines 2912-2936.

### Correction to the review: Turing fails too, not just Pascal

Both shapes in use require **sm_80 (Ampere)**. Turing's widest are `m16n8k8` (fp16) and `m8n8k16`
(int8). So Pascal, Volta **and Turing** all fail to compile. The review names Pascal as "the obvious
failure case" — anyone validating a fix on an RTX 20-series card would wrongly conclude it was
cleared. `CudaDeviceCapsTests.Turing_HasMma_ButNotTheWideShapesTheModuleUses` pins this.

### Done: pure capability layer + synthetic tests

`src/OpenTail.Stingray.Cuda/CudaDeviceCaps.cs` — a pure record over compute capability, no P/Invoke, no
device handle, with each threshold citing the PTX ISA requirement. `CanCompileMonolithicModule`
encodes the defect directly. **24 synthetic tests pass** across sm 53/61/70/75/80/86/89/90,
including a monotonicity check that no capability regresses as SM rises.

Capability handling before this was a bare `_smVersion` int with a single inline `>= 80` comparison
(`CudaBackend.cs:2096`), so the review's "gate dispatch through a CudaDeviceCaps structure" was
genuinely absent rather than merely untidy.

### NOT done, deliberately: the CUDA source split

This machine has **no `nvcc`, no `ptxas`, no NVRTC, no CUDA toolkit and no NVIDIA GPU.** The kernel
source cannot be compiled for any architecture here, so `#if __CUDA_ARCH__` guards would be
unverified edits to the one path that cannot be tested. A misplaced `#endif`, or a kernel compiled
away while the host still looks it up by name, would break every Ampere-and-later user in order to
fix Pascal-through-Turing. The current state at least works on sm_80+; an unverified fix could take
that away.

**Unblocking this needs a CUDA toolkit, NOT a GPU** — `nvcc` compiles for virtual architectures
offline. Compiling `CombinedKernelSource` for `compute_61/70/75/80` would verify a split in minutes
on any box with the toolkit installed.

---

## Finding 2 — Vulkan capability discovery → **CONFIRMED as written, NOT reproducible here**

### Verified

- `VulkanBackend.cs:368-369` sets `HasCooperativeMatrix` / `HasSubgroupSizeControl` purely from
  `extNames.Contains(...)` — extension name treated as capability.
- `VulkanBackend.cs:443-446` hardcodes `shaderFloat16 = True`, `shaderInt8 = True` in the device
  creation chain. The chain is gated on *extension presence*, never on the feature bit.
- **No `vkGetPhysicalDeviceFeatures2` call exists anywhere in the file.**

The failure mode is real: an extension may be advertised with an individual feature bit false, and
`vkCreateDevice` then fails with `VK_ERROR_FEATURE_NOT_PRESENT`.

### But it does not manifest on this device

| | this device |
|---|---|
| `VK_KHR_shader_float16_int8` | present, `shaderFloat16 = true`, `shaderInt8 = true` |
| `VK_KHR_16bit_storage` / `8bit_storage` | present, bits true |
| `VK_KHR_shader_integer_dot_product` | present |

Every extension we probe is advertised **and** every bit we request is genuinely true, so the
unconditional request happens to succeed. Vulkan inference runs correctly (verified end to end).
The core-promotion half also cannot bite here, because the extensions are advertised despite the
driver being Vulkan 1.3 where these are core.

**So this is a latent robustness defect, demonstrable only on hardware not present.** Recorded
rather than fixed blind — though note the correct fix (request only bits the device reports) is
*monotonically safer* than the status quo, since asking for strictly fewer features cannot fail
where asking for more succeeded.

---

## Finding 5 — Vulkan cooperative matrix → **MOOT ON THIS HARDWARE**

`VK_KHR_cooperative_matrix` is **ABSENT** from this device's extension list. There is nothing to
enable, select, or measure. The review's design advice (keep scalar as fallback, query shapes,
gate on the full feature set) is sound but unimplementable without hardware that exposes it.

---

## Naming — Vulkan "BF16 KV" is IEEE FP16 → **FIXED and verified**

The shaders pack and unpack with `unpackHalf2x16` (`Shaders.cs:2374, 2674, 2725`) — IEEE half by
definition. `Shaders.cs:2320` already noted the discrepancy in passing.

This is worth more than tidiness: **FP16 saturates at 65504 and overflows to infinity; BF16 has full
F32 exponent range and cannot.** That is precisely the property a KV cache is judged on. Anyone
debugging an inf/NaN in attention, or porting the scheme to another backend assuming range-safety,
is misled by the label — a live risk given the CPU-side work this week concluded BF16 is the right
choice *there* for unrelated reasons (shift-based widening; .NET exposes no F16C).

Diagnostics now report the physical format:

```
[GpuForwardPass] Context size: 2048 (model max: 8192) [KV fp16 (selector: bf16; Vulkan stores IEEE half)]
```

Verified on the AMD device. The env var and `DType.BFloat16` spelling are deliberately unchanged:
renaming them is a public API change, and CUDA's path genuinely *is* bfloat16, so the selector is
overloaded across backends rather than simply wrong.

---

## Finding 3 — Q6_K/Q5_K prefill MMQ → **NOT STARTED, by instruction and by evidence**

Unmeasurable without an NVIDIA GPU. The mechanism (eliminating a full-weight HBM round-trip) is
credible, but this week produced five wrong performance predictions from code-level reasoning
alone — including two of mine that survived scrutiny right up until measurement. Writing
tensor-core prefill kernels that cannot be benchmarked would repeat that at much greater cost.

## Vulkan performance context (relevant to findings 5 and 7)

This is an **integrated** Radeon sharing system RAM. Measured: Vulkan prefill **74.1 t/s** against
**150+ t/s on the CPU path** for the same model. It is not representative hardware for Vulkan
throughput work, so even the findings that could technically be exercised here would produce
numbers that do not generalise to a discrete card.
