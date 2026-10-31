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

`ggml/src/ggml-cuda/mma.cuh` (llama.cpp master, fetched 2026-10-31): [raw file](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mma.cuh)

| Instruction | Native path | Turing fallback in llama.cpp |
|---|---|---|
| int8 `m16n8k16` | Ampere and later (`__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE`) | 2x `m8n8k16` |
| int8 `m16n8k32` | Ampere and later | 4x `m8n8k16` |
| fp16 `m16n8k16` | Ampere and later | 2x or 4x `m16n8k8` |

The whole block sits under `TURING_MMA_AVAILABLE`. The shape of the fix is: guard the native path, and keep a Turing fallback. OpenTail has neither.

## 3. External reference: llama.cpp MMQ type coverage

`ggml/src/ggml-cuda/mmq.cu` (fetched 2026-10-31): [raw file](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmq.cu). `ggml_cuda_mul_mat_q_switch_type` dispatches 22 types to `mul_mat_q_case`, including Q4_K, **Q5_K and Q6_K**. Q5_K and Q6_K use the generic template with no type-specific kernel. Q6_K has an extra limit on RDNA3/RDNA4 WMMA batch size.

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
