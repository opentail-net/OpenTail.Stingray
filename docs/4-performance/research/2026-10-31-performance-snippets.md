# Code and reference excerpts behind the 4-performance research (2026-10-31)

Companion to the research sections appended to the performance docs in [docs/4-performance](..). Excerpts are from the working tree on 2026-10-31 unless marked as an external reference. Nothing here has been changed or built.

## 1. CUDA: MMA instructions, guarded only for async copy

`src/OpenTail.Stingray.Cuda/CudaTextKernels.cs`, around line 2623: the int8 MMA instruction is emitted with no architecture guard:

```cuda
""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
```

The only `__CUDA_ARCH__` guard in that file (line 2912) covers `cp.async`, not the MMA:

```cuda
#if __CUDA_ARCH__ >= 800
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    asm volatile(""cp.async.cg.shared.global [%0], [%1], 16;"" :: ""r""(s), ""l""(gmem));
#else
```

## 2. External reference: how llama.cpp gates the same instructions

`ggml/src/ggml-cuda/mma.cuh` (llama.cpp commit eb4e244b, the last change to this file, 2026-10-29; fetched 2026-10-31): [raw file](https://raw.githubusercontent.com/ggml-org/llama.cpp/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-cuda/mma.cuh)

| Instruction | Native path | Turing fallback in llama.cpp |
|---|---|---|
| int8 `m16n8k16` | Ampere and later (`__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE`) | 2x `m8n8k16` |
| int8 `m16n8k32` | Ampere and later | 4x `m8n8k16` |
| fp16 `m16n8k16` | Ampere and later | 2x or 4x `m16n8k8` |

The whole block sits under `TURING_MMA_AVAILABLE`. The shape of the fix is: guard the native path, and keep a Turing fallback. OpenTail has neither.

## 3. External reference: llama.cpp MMQ type coverage

`ggml/src/ggml-cuda/mmq.cu` (llama.cpp commit eb4e244b, the last change to this file, 2026-10-29; fetched 2026-10-31): [raw file](https://raw.githubusercontent.com/ggml-org/llama.cpp/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-cuda/mmq.cu). `ggml_cuda_mul_mat_q_switch_type` dispatches 22 types to `mul_mat_q_case`, including Q4_K, **Q5_K and Q6_K**. Q5_K and Q6_K use the generic template with no type-specific kernel. Q6_K has an extra limit on RDNA3/RDNA4 WMMA batch size.

## 4. Vulkan: subgroup size bounds and the mixed-dot extension

Khronos refpage, `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (fetched 2026-10-31): [docs.vulkan.org](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html). `requiredSubgroupSize` must be a power of two, no less than `minSubgroupSize` and no more than `maxSubgroupSize`.

`src/OpenTail.Stingray.Vulkan/ComputePipeline.cs`, lines 67-70, applies that bound before the pin:

```csharp
internal static bool ShouldPinSubgroupSize32(VulkanBackend backend, int localSizeX) =>
    backend.HasSubgroupSizeControl
    && backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
```

Mixed-precision dot extension, `VK_VALVE_shader_mixed_float_dot_product`: registered, revision 1, **not ratified** per the registry, spec dated 2026-02-04, requires `VK_KHR_shader_float16_int8` or Vulkan 1.2 ([docs.vulkan.org refpage](https://docs.vulkan.org/refpages/latest/refpages/source/VK_VALVE_shader_mixed_float_dot_product.html)). Not referenced anywhere in `src/` (grep, 2026-10-31).

## 5. Int8 pair sums: the saturation bound

Intel's pseudocode for `vpmaddubsw` saturates each adjacent pair sum to int16, not each product ([Intel developer guide](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/multiplication-intrinsics.html), [felixcloutier PMADDUBSW](https://felixcloutier.com/x86/pmaddubsw)).

Worst-case pair sum in the sign trick:

- with `|a|` at most 127 and `|w|` at most 128 (`w` = -128 gives `|w|` = 128): `2 * 128 * 127 = 32512`, which is below 32767;
- with `|a|` at most 127 and `|w|` at most 127: `2 * 127 * 127 = 32258`.

So the pair sum never saturates. The risk is in the sign step, not the pair sum: negating an activation of -128 wraps in int8. The unit test should target `a = -128`.

## 6. Threshold for CPU row parallelism

`src/OpenTail.Stingray.Cpu/SimdKernels.cs`, line 10:

```csharp
private const int MinRowsForParallel = 64;
```

## 7. Opt-in CPU micro-GEMM

`src/OpenTail.Stingray.Cpu/MicroGemmKernel.cs`, lines 6 and 33: gated behind `STINGRAY_CPU_MICRO_GEMM`, read once from the environment. Off by default, as the perf docs say.

## 8. External reference: ik_llama.cpp pp512 table (AVX2 label, Ryzen 7950X, 16 threads)

From [ik_llama.cpp discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164), LLaMA-3.1-8B, fetched 2026-10-31:

| Type | llama.cpp t/s | ik_llama.cpp t/s | Speedup |
|---|---|---|---|
| Q8_0 | 147.92 | 268.19 | 1.81x |
| Q4_0 | 153.52 | 273.54 | 1.78x |
| Q4_K_S | 108.40 | 269.60 | 2.49x |
| Q5_K_S | 75.52 | 254.68 | 3.37x |

The table is labelled AVX2, but the author says the Zen 4 AVX-512 path is used on Linux when available. The ISA for these numbers is therefore unclear.

## 9. External reference: T-MAC headline figures

From [arXiv 2407.00088v2](https://arxiv.org/html/2407.00088v2): kernel speedup up to 6.6x and average 3.6x over llama.cpp (introduction); single-thread GEMV maxima of 11.2x, 5.8x, 4.7x and 3.1x for 1 to 4 bits (section 5.2); multi-thread 2-bit mpGEMM up to 5.3x on a Surface Book 3 (Intel i5-1035G7, AVX2). The paper gives end-to-end 2.8x for Llama-2-7B 2-bit in the introduction, but section 5.3 assigns 2.8x to the 4-bit model. The 8-bit weight width is not evaluated.

---

## 10. Capability gating: GR_performance findings 1, 2, 5 and 6

Companion to [GR_performance.md](../GR_performance.md). Excerpts were copied from the working tree on 2026-10-30 (line numbers are from that tree). Nothing here has been changed or built. The sketch in 10.6 is a proposal, not tested code. This section was merged from the former `2026-10-30-capability-gating-snippets.md`.

### 10.1 CUDA: one NVRTC module, compiled for the exact device arch

`src/OpenTail.Stingray.Cuda/CudaBackend.cs` (around line 7204):

```csharp
string archFlag = _smVersion > 0 ? $"--gpu-architecture=sm_{_smVersion}" : "--gpu-architecture=compute_52";
```

All sources are concatenated into one module (`CombinedKernelSource`, `CudaBackend.cs` around line 7150):

```csharp
private static string CombinedKernelSource =>
    CudaKernels.Source + "\n" + CudaTextKernels.Source + "\n" + CudaWsKernels.Source +
    "\n" + CudaRaggedKernels.Source;
```

`CudaTextKernels.cs` contains `mma.sync` instructions that need sm_80 (lines 2623 onward, `m16n8k32 ... s8`, and
7586 onward, `m16n8k16 ... f16`). The only arch guard in that file is around `cp.async` (line 2912,
`#if __CUDA_ARCH__ >= 800`), which is a runtime fallback, not a compile-time exclusion of the MMA kernels.

### 10.2 CUDA: the capability record already exists, but nothing calls it

`src/OpenTail.Stingray.Cuda/CudaDeviceCaps.cs`:

```csharp
public bool CanCompileMonolithicModule => SupportsMmaF16M16N8K16 && SupportsMmaS8M16N8K32;
```

A search for `CanCompileMonolithicModule` and `CudaDeviceCaps` across `src/` (2026-10-30) finds no use outside
the file that defines them. The gate exists in code and is not wired into `CompileAndLoadKernels`.

Note for the GR finding: the record's own comment says Turing (sm_75) also fails, not only Pascal. Its MMA
claims are cited to the PTX ISA, which I could not read here (see the GR file for the attempt).

### 10.3 Vulkan: capability is the extension name, features are set to true

`src/OpenTail.Stingray.Vulkan/VulkanBackend.cs`, device enumeration (around lines 363-386):

```csharp
_vki.vkEnumerateDeviceExtensionProperties(_physicalDevice, null, &extCount, p);
...
bool hasFloat16Int8   = extNames.Contains("VK_KHR_shader_float16_int8");
bool has16BitStorage  = extNames.Contains("VK_KHR_16bit_storage");
...
HasCooperativeMatrix    = extNames.Contains("VK_KHR_cooperative_matrix");
```

Device creation (around lines 457-463) sets the feature bits without querying them:

```csharp
VkPhysicalDeviceShaderFloat16Int8Features f16Features = new()
{
    shaderFloat16 = VkBool32.True,
    shaderInt8    = VkBool32.True,
    ...
};
```

`VK_KHR_cooperative_matrix` is never added to the list of enabled extensions (`extPtrs`, around line 490). The
enabled list covers float16/int8, 16/8-bit storage, integer dot product, bf16, fp8 and subgroup size control only.
No `vkGetPhysicalDeviceFeatures2` call appears in `src/OpenTail.Stingray.Vulkan` (grep, 2026-10-30).

Why this matters, from vendor facts (WebSearch, 2026-10-30):
- `VK_KHR_16bit_storage` was promoted to core in Vulkan 1.1; `VK_KHR_shader_float16_int8` in 1.2. A 1.2+ device
  that does not list the extension name would have the feature available but not detected by this code.
- RADV exposes `VK_KHR_cooperative_matrix` only on RDNA3 and newer. A Vega-class iGPU (this machine, gfx90c)
  is not in that set, so cooperative-matrix work cannot be measured on this box.
- llama.cpp's Vulkan backend guards against the AMD proprietary driver reporting the extension on every GPU, so
  the extension list alone is not trusted there. That is the opposite failure to the one GR describes; both
  argue for querying features.

### 10.4 Vulkan: subgroup size pinning is conditional, and it is device-dependent

`src/OpenTail.Stingray.Vulkan/ComputePipeline.cs` (lines 67-70):

```csharp
internal static bool ShouldPinSubgroupSize32(VulkanBackend backend, int localSizeX) =>
    backend.HasSubgroupSizeControl
    && backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
    && !(backend.MinSubgroupSize == 32 && backend.MaxSubgroupSize == 32)
    && localSizeX > 0 && localSizeX % 32 == 0;
```

On a device whose range is exactly 64 (the comment at `src/OpenTail.Stingray.Vulkan/Shaders.cs` around line 4622
describes this device that way), the condition is false, so no pin is applied. On a device whose range includes 32
(RDNA-style 32-64), the pin applies to every shader with `local_size_x` a multiple of 32. The GR claim is therefore
true for that class of device, but it is not "essentially every shader on every device."

### 10.5 Vulkan: subgroup intrinsics are in use (contrary to the GR text)

`src/OpenTail.Stingray.Vulkan/Shaders.cs` (around line 10580 onward, `FlashAttention128`):

```glsl
#extension GL_KHR_shader_subgroup_basic   : require
#extension GL_KHR_shader_subgroup_shuffle : require
...
x = max(x, subgroupShuffleXor(x, 1u));
x += subgroupShuffleXor(x, 8u);
```

The same pattern appears in at least two other attention shaders (around lines 10732 and 10883). Their own comment
says they need a subgroup of at least 16. So pinning 32 is compatible with them, and the GR statement "no active
subgroup intrinsic was found" needs correcting.

### 10.6 Sketch (proposal, untested): query features before enabling them

This is the shape GR finding 2 asks for. It uses the Vulkan 1.2 feature struct for core-promoted features and only
sets bits the device reports. It is not wired into the backend.

```csharp
// 1. Ask what the device can do (not what the extension list names).
var f12 = new VkPhysicalDeviceVulkan12Features { sType = ... };
var f2  = new VkPhysicalDeviceFeatures2 { pNext = &f12 };
vki.vkGetPhysicalDeviceFeatures2(physicalDevice, &f2);

// 2. Enable only what the device reports true.
bool hasFloat16 = f12.shaderFloat16 == VkBool32.True;
bool hasInt8    = f12.shaderInt8    == VkBool32.True;
bool hasStorage16 = f12.storageBuffer16BitAccess == VkBool32.True;

// 3. Keep acceleration properties separate from feature bits.
// e.g. integer dot product: query VkPhysicalDeviceShaderIntegerDotProductFeatures and the
// matching properties, then decide the path. Do not infer it from the extension name alone.
```

Before any of this is used, check the struct and field names against the Vulkan headers that `Vortice.Vulkan` ships.
