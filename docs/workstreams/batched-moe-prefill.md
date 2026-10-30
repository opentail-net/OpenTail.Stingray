# Workstream: batched MoE prefill

**Owner:** this file. **Last verified:** 2026-10-30.
**Plan:** [../2-coverage/2026-10-03-batched-moe-prefill-plan.md](../2-coverage/2026-10-03-batched-moe-prefill-plan.md). **Related:** [../2-coverage/2026-10-04-moe-handoff-closure-plan.md](../2-coverage/2026-10-04-moe-handoff-closure-plan.md) (handoff receipts), [../2-coverage/2026-10-03-streamed-residency-plan.md](../2-coverage/2026-10-03-streamed-residency-plan.md).

## Correct state

- **Done (code checked 2026-10-30):** Phase 0 (baseline), Phase 1 (CPU-prefill fast path with exact KV handoff: `HybridPrefillHandoff.cs`, `GpuPrefillHandoff.cs`), Phase 1b (full-offload `GpuForwardPass` handoff), and the Phase 5 measured chooser (`Engine/Calibration/ExpertCalibration.cs`).
- **Not built:** Phase 2 (batched hybrid trunk with host MoE, gated on the Phase 0/1 measurements), Phase 3 (GPU grouped-expert prefill), Phase 4 (streamed experts for oversized models), Phase 6 (discrete-GPU decode miss split).
- **Phase 1 CUDA variant:** written and compiles, never run (no NVIDIA GPU here).
- **Chooser not wired:** the engine does not read the calibration profile yet, because the Phase 3/4 paths it would choose between do not exist.
- **Correction to the backlog:** [../00-current-work.md](../00-current-work.md) item 5 says "`VulkanLayerSplitForwardPass` prefills per token". That is accurate for the layer-split and MoE paths only. `GpuForwardPass` already batches (see [vulkan-gemm-llm-prefill.md](vulkan-gemm-llm-prefill.md)).

## Open items

- [ ] Phase 2 decision: proceed only if the Phase 0/1 measurements show GPU attention beating CPU batched attention on the target hardware (hard gate)
- [ ] Phase 3 (3a, and 3b only if profiling justifies it), tested on a discrete GPU before any default change
- [ ] Phase 4 (streamed experts), after the Phase 3 kernels exist, or with the CPU fallback below the gate
- [ ] Phase 5 wiring: the engine reads the profile, once there is more than one path to choose between
- [ ] CUDA variant: run on an NVIDIA machine
- [ ] Phase 6 (discrete GPU only)

## Decisions for the owner

- Sign-off on Phase 0 and Phase 1 as the first implementation. Phase 1 is built; this is recorded for the owner's approval.
- A machine with a discrete GPU to measure Phase 3/4 speed. Without one they stay correctness-only and off by default.

## Cadence

- Review when a discrete-GPU machine is available, or when a Phase 2 gate measurement is taken.
- Re-read at least every two weeks while any phase is open.
- Close when each phase is built and gated, or written off with the reason in the plan.
