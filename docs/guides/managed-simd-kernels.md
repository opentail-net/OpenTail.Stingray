# Technical Guide: Managed SIMD Kernels & CPU Optimization

> **Topic:** High-performance vector math, SIMD intrinsics, zero-allocation memory design, and multi-threaded CPU execution in pure C#  
> **Audience:** Performance engineers and engine developers  
> **Source References:** `OpenTail.Stingray.Cpu/`, `SimdOps.cs`, `GemmKernels.cs`

---

## 1. Zero-Allocation Philosophy

Stingray proves that managed .NET code can rival native C++ execution speeds when designed with zero-allocation constraints:
- **No Per-Token Allocations:** The inner forward pass and token generation loop allocate zero heap objects. All working tensors, activations, and KV states reside in pre-allocated buffers.
- **Span-Based Slicing:** Multi-dimensional tensor views are represented using `Span<float>` and `ReadOnlySpan<byte>` with unsafe memory pinning (`fixed` or pinned handle wrappers).
- **Struct Aggregates:** Parameters and intermediate results are passed as `readonly ref struct` types to eliminate boxing and GC pressure.

---

## 2. Hardware Vectorization with .NET Intrinsics

Stingray uses `System.Runtime.Intrinsics.X86` and `System.Runtime.Intrinsics.Arm` to target hardware vector units directly without external assembly or DLL imports:

| Vector ISA | Vector Width | Target Instructions | Primary Operations |
|---|---|---|---|
| **AVX-512** | 512 bits (16 floats) | `Avx512F`, `Avx512DQ`, `Avx512BW` | Fused multiply-add (FMA), wide RMSNorm, vectorized Softmax |
| **AVX2 & FMA** | 256 bits (8 floats) | `Avx2`, `Fma`, `Avx` | Quantized block dequantization, GEMM dot products |
| **ARM Neon** | 128 bits (4 floats) | `AdvSimd` | Mobile / Apple Silicon vectorization |

### Fused Dequantize-and-Multiply (GEMM)
Rather than dequantizing entire weight matrices into intermediate 32-bit float buffers (which would flood the L3 CPU cache), Stingray executes fused dequantization:
- Weights are stored in block-quantized formats (e.g. `Q4_K` with 256 weights per block, grouped scales and mins).
- The SIMD kernel loads 4-bit nibbles, expands them into 8-element vector registers using shuffle masks, applies the block scale, and accumulates directly into output registers using FMA instructions.

---

## 3. Multi-Threading & Thread Governors

CPU inference scalability depends on keeping CPU caches hot and avoiding thread migration across NUMA nodes:

### Thread Configuration
- Controlled via the `-t, --threads` CLI argument or `ContextParams.ThreadCount`.
- By default, Stingray detects physical performance cores and allocates one worker thread per physical core, avoiding SMT/hyperthreading contention.

```bash
# Pin execution to 8 worker threads
stingray -m model.gguf -p "Prompt" -t 8
```

### Environment Variable Tuning
- `STINGRAY_CPU_THREADS`: Sets global default worker thread pool size.
- `STINGRAY_CPU_VNNI=0`: Forces the AVX2 dot-product chain even where VNNI exists. This is a test seam for the bit-identical-branch checks, not a tuning knob.

See [`docs/reference/env-var-inventory.md`](../reference/env-var-inventory.md) for full thread and SIMD configuration switches.
