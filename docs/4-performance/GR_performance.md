# GPU Performance Review

The remaining GPU gap is not that OpenTail lacks modern kernels. It already has tensor-core flash attention, split-KV decode, grouped GQA reuse, CUDA graphs, and int8 MMA matmul. The larger difference is that llama.cpp has a mature hardware-and-shape-specific kernel planner, while OpenTail generally uses a small number of fixed CUDA/Vulkan implementations.

This review found two concrete compatibility problems and several strong performance candidates.

> No build, compilation, execution, or benchmark was performed during this review because the machine was occupied. Compatibility findings below are code-level conclusions. Performance candidates are ranked by eliminated memory traffic and the maturity of analogous llama.cpp paths; they still require later measurement.

## Highest-confidence findings

### 1. CUDA's advertised Pascal support is probably broken

OpenTail advertises FP16 operation on `sm_53+` in `src/OpenTail.Stingray.Cuda/CudaBackend.cs`, but compiles all CUDA sources into one monolithic NVRTC module.

That source contains unconditional `mma.sync` instructions, including:

- int8 `m16n8k32` instructions in `CudaTextKernels.cs`
- fp16 `m16n8k16` instructions in `CudaTextKernels.cs`

The module is compiled for the exact GPU architecture. These MMA variants are not valid across the entire `sm_53+` range; Pascal is the obvious failure case. Because compilation is monolithic, even an unused tensor-core test kernel can prevent all custom kernels from loading.

Recommended correction:

- Split baseline CUDA kernels from Turing/Ampere/Hopper-specialized modules, or conditionally compile MMA implementations by `__CUDA_ARCH__`.
- Always retain a scalar/DP4A-compatible module for older GPUs.
- Gate actual dispatch through a `CudaDeviceCaps` structure, not merely head-dimension alignment.
- Add synthetic dispatch tests for SM 53, 61, 70, 75, 80, 86, 89, 90 and future architectures.

This should be addressed before performance work because it is a logical compatibility defect, not a benchmark-dependent theory.

### 2. Vulkan capability discovery is not spec-robust

OpenTail currently treats extension-name presence as capability presence in `VulkanBackend.cs`, then requests feature bits as unconditionally true during device creation, including `shaderFloat16` and `shaderInt8`.

This has two failure modes:

- An extension can be advertised while an individual feature bit is false, causing device creation or pipeline validation failures.
- Features promoted into a newer Vulkan core version can be available without the corresponding extension name, causing OpenTail to miss valid fast paths.

llama.cpp builds a feature chain, calls `vkGetPhysicalDeviceFeatures2`, and intersects extension, feature, and property support.

Recommended correction:

- Query feature structures before device creation.
- Distinguish extension availability, feature availability, acceleration properties, and enabled capability.
- Handle Vulkan core promotion using the physical device's API version.
- Only place true, queried feature bits into the device creation chain.

## Best likely CUDA performance opportunity

### 3. Add direct Q6_K and Q5_K prefill MMQ

OpenTail's large-batch direct MMA path currently accepts only Q8_0, Q4_K and Q4_0.

Q5_K and Q6_K prefill instead use `MatMulBatchedGemm`, which:

1. Dequantizes the complete static weight matrix to FP16 scratch.
2. Writes that scratch to VRAM.
3. Reads it again through cuBLAS GEMM.

That is a full-weight HBM round-trip on every invocation.

OpenTail already has a direct Q6_K MMA implementation for small-N decode, so the decoding arithmetic and packing work are largely understood. Extending that implementation to a large-N prefill tile is the most credible next throughput improvement.

llama.cpp directly routes Q3_K, Q4_K, Q5_K and Q6_K through MMQ, with architecture-specific configuration.

Suggested order:

1. Q6_K direct prefill MMQ.
2. Q5_K direct prefill MMQ.
3. Q3_K if it is common enough in supported models.
4. Only then revisit cuBLAS-based alternatives.

This should remove real memory traffic. The exact speedup still needs later measurement, but the mechanism does not depend on timing noise.

## The main architectural difference from llama.cpp

### 4. OpenTail needs a GPU kernel-plan layer

OpenTail's tensor-core attention and MMQ paths generally use fixed configurations. For example, the second tensor-core flash-attention implementation uses a fixed four-warp arrangement.

llama.cpp chooses among vector, tiled, WMMA and MMA attention based on:

- Compute capability
- Query count
- Head dimensions
- GQA ratio
- KV length and dtype
- Shared-memory constraints

Its MMQ configuration also varies by Pascal, Ampere, Blackwell, RDNA and CDNA.

A sensible OpenTail structure would be:

```text
CudaDeviceCaps + operation shape
              |
              v
       CudaKernelPlan
              |
              v
baseline | Turing | Ampere/Ada | Hopper/Blackwell
```

The planner should be a pure, testable C# component. Kernels remain separate implementations, and unsupported hardware always receives a valid baseline plan.

Initially, differentiate only:

- Pre-Turing baseline
- Turing
- Ampere/Ada
- Hopper/Blackwell
- Decode/vector attention versus multi-query prefill MMA

That captures most of the value without copying llama.cpp's full complexity immediately.

## Vulkan performance findings

### 5. Cooperative matrix support is detected but effectively unused

`HasCooperativeMatrix` is set from the extension name in `VulkanBackend.cs`, but no OpenTail cooperative-matrix shader path was found, and the extension/feature is not enabled in the device chain.

Current Vulkan flash attention is scalar/shared-memory based. llama.cpp chooses scalar, KHR cooperative-matrix, or NV cooperative-matrix-2 implementations.

This is probably the largest long-term Vulkan opportunity, particularly on recent NVIDIA, AMD and Intel hardware. It is not a safe single-kernel replacement. The correct design is:

- Preserve the existing scalar implementation as the universal fallback.
- Query cooperative-matrix shapes and supported operand/accumulator types.
- Generate or select kernels for the shapes actually exposed by the driver.
- Gate every dispatch on the complete queried feature set.
- Keep vendor-specific tuning in data tables rather than scattered conditionals.

### 6. Subgroup-size pinning is overbroad

`ComputePipeline.cs` pins subgroup size 32 for essentially every shader whose `local_size_x` is divisible by 32.

However, the current shader source contains shared-memory reductions designed to work across Wave16/32/64/128 and explicitly describes subgroup pinning as irrelevant for scalar kernels. No active subgroup intrinsic was found in the Vulkan shader sources.

Consequently, wave32 can be forced on AMD even for kernels that do not need it, potentially sacrificing native wave64 behavior.

The pipeline descriptor should explicitly state whether a shader requires subgroup 32. Local workgroup size alone is not enough to infer that requirement.

### 7. Vulkan batched matmul coverage remains narrow

The Vulkan batched path has specialized weight-stationary kernels primarily for Q4_K and Q6_K. Other formats fall back toward temporary allocations and token-by-token dispatch. The engine also presently keeps Vulkan prefill batches small.

After capability discovery is corrected, a reasonable progression is:

1. Direct Q8_0/Q4_0/Q5_K batched quantized matmul.
2. Larger shape-dependent batch tiles.
3. Integer-dot-product variants where the driver reports accelerated dot products.
4. Cooperative-matrix variants where supported.

## Lower-priority opportunities

These are valid directions, but should not be the starting point:

- **cuBLASLt:** OpenTail uses `cublasGemmEx`; Lt could provide algorithm selection, persistent workspace and newer epilogues. Direct quantized MMQ is likely more valuable first.
- **Native FP4:** MXFP4 and NVFP4 exist in the core dtype list, but no corresponding CUDA or Vulkan kernel routing was found. This is mainly a Blackwell/future-GPU opportunity.
- **Future CUDA toolkit compatibility:** compiling only for exact `sm_NN` can fail when a new GPU is paired with an older NVRTC. Query supported NVRTC architectures and fall back to the highest compatible compute target.
- **Module splitting:** separating attention, matmul, baseline and architecture-specific CUDA modules would improve compatibility and reduce invalidation/startup compilation.
- **Multi-GPU:** OpenTail currently assumes device zero in important CUDA setup paths. llama.cpp has broader device-aware execution, but this is a product-scope feature rather than a near-term single-GPU speed fix.
- **Vulkan KV naming:** the path described as BF16 KV storage actually packs IEEE FP16 values. It appears intentional, but the internal name and diagnostics should say FP16 to avoid future precision mistakes.

## Recommended implementation order

1. Fix CUDA architecture compatibility and Vulkan feature discovery.
2. Introduce pure `CudaKernelPlan` and `VulkanDeviceCaps` objects with synthetic hardware tests.
3. Implement Q6_K direct prefill MMQ, then Q5_K.
4. Add a small set of CUDA attention/MMQ configurations for Turing, Ampere/Ada and Hopper/Blackwell.
5. Expand Vulkan direct batched quant kernels.
6. Add optional Vulkan cooperative-matrix attention and matmul.
7. Consider cuBLASLt and native FP4 afterward.

## Validation without access to every GPU

Much of the architecture can be validated without owning the target cards:

- Make device-capability discovery and kernel selection pure, table-driven code.
- Unit-test synthetic CUDA capability records for each supported SM family.
- Unit-test Vulkan records covering NVIDIA, AMD Wave32/Wave64, and Intel feature combinations.
- Assert that no plan selects an instruction, dtype, subgroup size, shared-memory allocation, or cooperative-matrix shape unsupported by its capability record.
- Compile CUDA kernel modules for a matrix of virtual architectures when the machine is available, ensuring the baseline module compiles for Pascal and MMA modules are only compiled for valid targets.
- Compile and validate all generated SPIR-V variants with strict validation layers.
- Keep correctness fixtures separate from performance benchmarks, especially for chunked versus unchunked attention and quantized matmul tolerances.

None of these conclusions establishes whether OpenTail is presently faster than llama.cpp. That comparison requires controlled, interleaved end-to-end measurements once the environment is quiet.
