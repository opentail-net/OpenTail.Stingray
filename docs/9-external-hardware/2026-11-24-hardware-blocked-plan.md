# HARDWARE BLOCKED plan

Written 2026-11-24. Open work whose acceptance evidence needs hardware this machine does not have. Items move here from their source plans, which keep a pointer. Work that can run on the CPU or the integrated Radeon stays in its own plan (CLAUDE.md rule 13).

**This machine:** Ryzen 7 5700G (16 logical CPUs), integrated AMD Radeon, about 63 GiB RAM, one machine. No discrete GPU, no NVIDIA device, no ARM64 device.

## Rules for items here

- Each item says what hardware it needs, what it blocks, what can still run here, and the evidence to collect.
- No default changes on an item here until that evidence is recorded (CLAUDE.md rule 13). An iGPU-only result does not settle a discrete-GPU question either way.
- [90-external-hardware-work.md](90-external-hardware-work.md) keeps its own earlier list (CUDA, ARM64, discrete-GPU speed, Phase 6 decode). Those items are not duplicated here. Whether the two lists should be merged is an owner decision.

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

## Not moved here (owner call, or the source file is in use)

- **OpenTelemetry plan, clean-machine run** (`docs/guides/observability.md`): needs a second machine. Not moved because that plan changed today in another session.
- **Streamed residency, larger-than-RAM run** (GLM-5.3 Q2_K_XL, 236 GiB): needs more RAM than this host has. Not moved for the same reason.
- **glm-dsa real-weight parity** and **coverage programme Phase 2A**: both are blocked by disk budget on this host. Disk is not a discrete piece of hardware, so these need an owner decision on whether they belong here.
