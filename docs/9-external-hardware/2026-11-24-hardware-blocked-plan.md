# HARDWARE BLOCKED plan

Written 2026-11-24. Open work whose acceptance evidence needs hardware this machine does not have. Items move here from their source plans, which keep a pointer. Work that can run on the CPU or the integrated Radeon stays in its own plan (CLAUDE.md rule 13).

**This machine:** Ryzen 7 5700G (16 logical CPUs), integrated AMD Radeon, about 63 GiB RAM, one machine. No discrete GPU, no NVIDIA device, no ARM64 device.

## Rules for items here

- Each item says what hardware it needs, what it blocks, what can still run here, and the evidence to collect.
- No default changes on an item here until that evidence is recorded (CLAUDE.md rule 13). An iGPU-only result does not settle a discrete-GPU question either way.
- The external-hardware queue (CUDA, ARM64, discrete-GPU speed, Phase 6 decode) is merged into this plan below, under its own section. The owner decided the merge on 2026-11-24.

## Items

### H1. Batched MoE prefill, Phase 3: GPU grouped-expert prefill on Vulkan

- **Needs:** a discrete GPU. The plan says Phase 3 is "tested on a discrete GPU before any default change".
- **Blocks:** the default setting and any speed claim.
- **Runs here:** correctness only (cosine against the CPU path, and the argmax near-tie rule), on the integrated Radeon.
- **Source:** [2-coverage/2026-10-03-batched-moe-prefill-plan.md](../2-coverage/2026-10-03-batched-moe-prefill-plan.md), Phase 3 item.
- [ ] Phase 3 (3a, and 3b only if profiling justifies it), tested on a discrete GPU before any default change

### H2. Batched MoE prefill, Phase 4: streamed experts for oversized models

- **Needs:** a discrete GPU. The plan keeps Phases 3 and 4 off by default until measured on one.
- **Blocks:** the default setting and any speed claim.
- **Runs here:** correctness, with the CPU fallback below the gate.
- **Source:** [2-coverage/2026-10-03-batched-moe-prefill-plan.md](../2-coverage/2026-10-03-batched-moe-prefill-plan.md), Phase 4 item.
- [ ] Phase 4 (streamed experts), after the Phase 3 kernels exist, or with the CPU fallback below the gate

### H3. Many-core host measurement for the spin-then-park worker pool

- **Needs:** a host with far more cores than this one. The source note says a measurement on 16 logical CPUs "says nothing about 100+ cores".
- **Blocks:** the claimed gains from the spin-then-park pool. TensorSharp measured them on a many-core host.
- **Runs here:** the opt-in pool is already measured on this host (see the source plan).
- **Source:** [2-coverage/2026-10-02-tensorsharp-takeaways-plan.md](../2-coverage/2026-10-02-tensorsharp-takeaways-plan.md), section 1, "Next, if pursued".
- [ ] Measure on a many-core host, where TensorSharp measured its gains.

## From the external-hardware queue (merged 2026-11-24)

The queue was [90-external-hardware-work.md](../done/90-external-hardware-work.md) before this merge. Its text and open items are kept below unchanged apart from heading levels.

This is deliberately **not** a model-acquisition queue. GGUFs, SafeTensors packages, and mmproj
fixtures can be copied or downloaded onto this machine; work that can run on CPU or the local AMD
Vulkan adapter remains in [00-current-work.md](../00-current-work.md). This queue contains only work whose
acceptance evidence requires hardware not present locally.

### Available locally

- CPU AVX2/FMA inference and session receipts.
- AMD integrated-GPU Vulkan correctness and smoke receipts.
- Any model fixture once obtained locally, including Gemma 4 12B text and E4B mmproj files.

### NVIDIA / CUDA runner

The development machine has no NVIDIA device. These items need a designated NVIDIA runner with
the exact model/configuration recorded alongside each result:

- [ ] 1. **Release evidence.** CUDA dense load/decode/sample, long-context KV dtype, placement/capability
   output, CUDA-hybrid MoE, and MTP/speculation receipts. These are release gates, not claims that
   can be inferred from CPU or Vulkan tests.
- [ ] 2. **Gemma 4 12B.** Re-run the dense no-PLE CPU/CUDA/hybrid acceptance sequence after the
   per-layer V-cache stride correction. The 12B GGUF may be acquired locally, but CUDA and hybrid
   portions require this runner.
- [ ] 3. **CUDA dense gate/up fusion.** The design review is archived in
   [done/cuda-fused-gate-up-plan.md](../done/cuda-fused-gate-up-plan.md). Do not implement it blind:
   establish an interleaved baseline first, test CUDA graphs both enabled and disabled, then retain
   the change only if it improves real prefill or decode at a representative model shape.
- [ ] 4. **CUDA graph default discrepancy.** Dense and hybrid paths historically interpret
   `STINGRAY_CUDA_GRAPH` differently. Reproduce both paths on actual hardware before changing the
   default or documentation; a source-only harmonisation would be an unmeasured performance change.

### Found 2026-09-26, needs CUDA or a discrete GPU

Carried from [../done/101-work-queue-after-coverage-plan.md](../done/101-work-queue-after-coverage-plan.md):

- [ ] 5. **CUDA `rope_freqs`.** `CudaForwardPass`/`CudaHybridForwardPass` apply RoPE frequency factors for
   Gemma 4 only, so Llama-3.1-style models run unscaled RoPE on CUDA (wrong only at long context).
   Fixed on Vulkan (full and `-g N`) on 2026-09-26. Not gated, since that would push every Llama 3.1
   CUDA user to CPU.
- [ ] 6. **CUDA partial offload for the newer architectures** (fused QKV, LayerNorm, parallel residual,
   partial RoPE, non-gated FFN). Vulkan `-g N` covers them via `VulkanLayerSplitForwardPass`; CUDA
   falls back to CPU with a note.
- [ ] 7. **Discrete-GPU measurements** before changing any default (CLAUDE.md rule 13): speculative
   decoding with a draft model on Vulkan (lossless, slower on the iGPU), the UMT5 GPU path vs CPU,
   gpt-oss on Vulkan (matvec-bandwidth-bound on the iGPU).

### ARM64 runner

- [ ] **ARM64 NEON, dot-product, and i8mm kernels baseline.** ARM64 NEON, dot-product, and i8mm kernels need an ARM64 device. Keep the work out of the local
   CPU optimisation queue until a runner is available; its first receipt should report ISA features,
   model/dtype, correctness parity, and a measured baseline before any architecture-specific kernel
   is introduced.

### Receipt discipline

For every row: record commit SHA, exact model hash/quantization, backend/device/driver, command,
warm-up protocol, repeated samples, numerical or token-parity result, and whether concurrent
workloads were present. A result from another machine is evidence only for that recorded setup.

### Moved from plan docs (2026-11-01)

Hardware-only verification items taken out of the plans so that the plans carry only work that can run here. Each entry names its source; the source keeps a pointer.

> from: `docs/00-current-work.md#open-work-as-of-2026-11-03`
  - [ ] P2d (blocked: no NVIDIA hardware) The CUDA full and hybrid passes have no reference to the residual or logit scale either, and the new guard does not cover them; confirm on a machine with a CUDA GPU before relying on Granite or Gemma 3 there

> from: `docs/00-current-work.md#open-work-as-of-2026-11-03`
  - [ ] `startPos > 0` on the CUDA hybrid (CUDA cannot be tested here; the code change would mirror the Vulkan hybrid one)

> from: `docs/00-current-work.md#open-work-as-of-2026-11-03`
- [ ] **P2d** CUDA passes have no scale references (blocked: no NVIDIA hardware)

> from: `docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md`
- [ ] Phase 1 CUDA variant: written, compiles, never run (no NVIDIA GPU here; code: `CudaHybridPrefillHandoff.cs`)

> from: `docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md`
- [ ] Phase 6: discrete-GPU decode miss split and device-side cache decisions

> from: `docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md`
- [ ] Decision for the user: a discrete-GPU machine for Phase 3/4 speed measurement

> from: `docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md`
- [ ] CUDA variant: run on an NVIDIA machine

> from: `docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md`
- [ ] Phase 6 (discrete GPU only)

> from: `docs/00-current-work.md`
  - [ ] **CUDA items:** (`rope_freqs` for Llama-3.1-style models; M-RoPE image positions + deepstack in `CudaForwardPass`, done for Vulkan 2026-09-28) blocked on CUDA GPU.

> from: `docs/00-current-work.md`
  - [ ] **Discrete-GPU performance items** (from 15, 2026-09-28): Vulkan batched prefill, FLUX.1/FLUX.2 double-block GEMM and fusion (attention shader and audits: measurable on the iGPU but only pays off on a discrete GPU, see item 15), TTS/ASR GPU residency. This machine's iGPU shares DRAM with the CPU and trails it on prefill-shaped work, so a result here would not say whether the GPU code is good (CLAUDE.md rule 13). Blocked on a machine with a discrete GPU.

> from: `docs/2-coverage/2026-10-03-streamed-residency-plan.md`
- [ ] CUDA streamed path (not done)

> from: `docs/4-performance/2026-11-08-performance-plan.md`
- [ ] **A10** CUDA decode attention (`llm_attention`, `_bf16`, `_q8_0`) has the identical one-block-per-head, one-thread-per-key structure; needs an NVIDIA machine to measure before porting.

## Not moved here (owner call, or the source file is in use)

- **OpenTelemetry plan, clean-machine run** (`docs/guides/observability.md`): needs a second machine. Not moved because that plan changed today in another session.
- **Streamed residency, larger-than-RAM run** (GLM-5.3 Q2_K_XL, 236 GiB): needs more RAM than this host has. Not moved for the same reason.
- **glm-dsa real-weight parity** and **coverage programme Phase 2A**: blocked by disk budget. Owner decision 2026-11-24: disk-limited items do not belong in this plan. They stay in their own plans.
