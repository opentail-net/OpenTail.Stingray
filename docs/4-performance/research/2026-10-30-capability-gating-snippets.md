# Capability gating: code excerpts behind the GPU findings (2026-10-30)

Companion to [GR_performance.md](../GR_performance.md) (findings 1, 2, 5, 6). Every excerpt is copied from the
working tree on 2026-10-30; line numbers are from that tree. Nothing here has been changed or built. The sketch
at the end is a proposal, not tested code.

## 1. CUDA: one NVRTC module, compiled for the exact device arch

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

## 2. CUDA: the capability record already exists, but nothing calls it

`src/OpenTail.Stingray.Cuda/CudaDeviceCaps.cs`:

```csharp
public bool CanCompileMonolithicModule => SupportsMmaF16M16N8K16 && SupportsMmaS8M16N8K32;
```

A search for `CanCompileMonolithicModule` and `CudaDeviceCaps` across `src/` (2026-10-30) finds no use outside
the file that defines them. The gate exists in code and is not wired into `CompileAndLoadKernels`.

Note for the GR finding: the record's own comment says Turing (sm_75) also fails, not only Pascal. Its MMA
claims are cited to the PTX ISA, which I could not read here (see the GR file for the attempt).

## 3. Vulkan: capability is the extension name, features are set to true

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

## 4. Vulkan: subgroup size pinning is conditional, and it is device-dependent

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

## 5. Vulkan: subgroup intrinsics are in use (contrary to the GR text)

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

## Sketch (proposal, untested): query features before enabling them

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
