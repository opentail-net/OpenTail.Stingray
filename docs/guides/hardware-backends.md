# Technical Guide: Hardware Backends & Compute Dispatch

> **Topic:** Configuring and optimizing execution across CPU, Vulkan, and CUDA backends  
> **Audience:** Operators and developers configuring local GPU acceleration  
> **Source References:** `OpenTail.Stingray.Cpu/`, `OpenTail.Stingray.Vulkan/`, `OpenTail.Stingray.Cuda/`

---

## 1. Supported Compute Backends

Stingray provides three hardware execution backends:

| Backend | Flag | Hardware Supported | Key Mechanics | Current Status |
|---|---|---|---|---|
| **CPU** | `--backend cpu` | x64 (AVX-512, AVX2) & ARM64 | Pure managed SIMD, fused quantized GEMM | 🔬 Primary verified baseline |
| **Vulkan** | `--backend vulkan` | AMD, Intel, NVIDIA, Apple (MoltenVK) | SPIR-V compute shaders, shared-memory FlashAttention | 🔬 Verified on integrated/discrete GPUs |
| **CUDA** | `--backend cuda` | NVIDIA (Compute Capability 7.0+) | Managed CUDA driver interop, cuBLAS GEMM | ⚪ Implemented; hardware validation in `docs/9/` |

---

## 2. Vulkan Compute Architecture

The Vulkan backend (`OpenTail.Stingray.Vulkan`) compiles SPIR-V compute shaders to execute matrix operations directly on GPU execution units:
- **Weight-Stationary GEMM:** High-performance compute shaders are specialized for `Q4_K` and `Q6_K` quantization formats. Weights are retained in device memory while activation vectors stream through local workgroups.
- **Shared-Memory FlashAttention:** FlashAttention kernels compute scaled dot-product attention in tiles using workgroup shared memory (`local_size_x = 32` or `64`), eliminating high-bandwidth VRAM roundtrips.
- **No Cooperative Matrix Dependency:** Attention and GEMM use standard, cross-vendor compute shaders. (Extension detection exists for `VK_KHR_cooperative_matrix`, but public shaders do not require or assume cooperative-matrix hardware).

### Configuring Vulkan Execution
```bash
# Offload all layers to Vulkan
stingray -m models/qwen2.5-7b-instruct-q4_k_m.gguf -p "Hello" --backend vulkan -g -1

# Partial layer offload (e.g. 16 layers to GPU, remainder to CPU)
stingray -m models/qwen2.5-7b-instruct-q4_k_m.gguf -p "Hello" --backend vulkan --gpu-layers 16
```

---

## 3. CUDA Backend Prerequisites

The CUDA backend (`OpenTail.Stingray.Cuda`) provides high-throughput matrix multiplication for NVIDIA hardware:
- **Dependencies:** Requires NVIDIA drivers with CUDA 12.x runtime libraries (`cudart64_12.dll`, `cublas64_12.dll` on Windows; `libcudart.so.12` on Linux).
- **Environment:** If CUDA is installed in a non-standard location, set `CUDA_PATH` or ensure runtime DLLs are present on the system `PATH`.
- **Status:** Contributor testing on physical NVIDIA hardware is organized in [docs/9-external-hardware/](../9-external-hardware).

---

## 4. Hardware Diagnostic Checks

To verify your system's detected compute topology, run `stingray doctor`:

```bash
stingray doctor
```

This inspects available SIMD instructions, Vulkan physical devices, VRAM limits, and CUDA driver availability.
