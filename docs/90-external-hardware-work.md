# External-hardware work queue

This is deliberately **not** a model-acquisition queue. GGUFs, SafeTensors packages, and mmproj
fixtures can be copied or downloaded onto this machine; work that can run on CPU or the local AMD
Vulkan adapter remains in [00-current-work.md](00-current-work.md). This queue contains only work whose
acceptance evidence requires hardware not present locally.

## Available locally

- CPU AVX2/FMA inference and session receipts.
- AMD integrated-GPU Vulkan correctness and smoke receipts.
- Any model fixture once obtained locally, including Gemma 4 12B text and E4B mmproj files.

## NVIDIA / CUDA runner

The development machine has no NVIDIA device. These items need a designated NVIDIA runner with
the exact model/configuration recorded alongside each result:

1. **Release evidence.** CUDA dense load/decode/sample, long-context KV dtype, placement/capability
   output, CUDA-hybrid MoE, and MTP/speculation receipts. These are release gates, not claims that
   can be inferred from CPU or Vulkan tests.
2. **Gemma 4 12B.** Re-run the dense no-PLE CPU/CUDA/hybrid acceptance sequence after the
   per-layer V-cache stride correction. The 12B GGUF may be acquired locally, but CUDA and hybrid
   portions require this runner.
3. **CUDA dense gate/up fusion.** The design review is archived in
   [done/cuda-fused-gate-up-plan.md](done/cuda-fused-gate-up-plan.md). Do not implement it blind:
   establish an interleaved baseline first, test CUDA graphs both enabled and disabled, then retain
   the change only if it improves real prefill or decode at a representative model shape.
4. **CUDA graph default discrepancy.** Dense and hybrid paths historically interpret
   `STINGRAY_CUDA_GRAPH` differently. Reproduce both paths on actual hardware before changing the
   default or documentation; a source-only harmonisation would be an unmeasured performance change.

## ARM64 runner

ARM64 NEON, dot-product, and i8mm kernels need an ARM64 device. Keep the work out of the local
CPU optimisation queue until a runner is available; its first receipt should report ISA features,
model/dtype, correctness parity, and a measured baseline before any architecture-specific kernel
is introduced.

## Receipt discipline

For every row: record commit SHA, exact model hash/quantization, backend/device/driver, command,
warm-up protocol, repeated samples, numerical or token-parity result, and whether concurrent
workloads were present. A result from another machine is evidence only for that recorded setup.
