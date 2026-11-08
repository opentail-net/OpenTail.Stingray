# Plan: close the CPU-prefill handoff, one model family at a time (2026-10-04) - completed items

Ticked items moved from [docs/2-coverage/2026-10-04-moe-handoff-closure-plan.md](2026-10-04-moe-handoff-closure-plan.md) on 2026-11-03. Each keeps its section heading and nested lines.

## Open (checked 2026-10-30 against the status line)

- [x] cohere2moe: classified, no Stage A (not worth it on this hardware, per the step A outcome). Decided 2026-10-04; the record is the `cohere2moe` line above. Closed 2026-11-01.

- [x] llama4: handoff receipt recorded in `PrefillHandoffFamilies.Receipts` (`llama4`, `VulkanHybrid`: `LlamaFourGreedyParityTests` and `HybridCpuPrefillHandoffTests`, incl. `Handoff_Llama4_PastTheChunkBoundary_DecodeAgreesWithCpu`; results in the Llama 4 entries above). Closed 2026-11-01: the box was stale; checked the receipt and both test names in the code, did not re-run the 4-layer Scout checkpoint.

## Moved 2026-11-08 from 2026-10-04-moe-handoff-closure-plan.md

- [x] cohere2moe: classified, no Stage A (not worth it on this hardware, per the step A outcome). Decided 2026-10-04; the record is the `cohere2moe` line above. Closed 2026-11-01.
- [x] llama4: handoff receipt recorded in `PrefillHandoffFamilies.Receipts` (`llama4`, `VulkanHybrid`: `LlamaFourGreedyParityTests` and `HybridCpuPrefillHandoffTests`, incl. `Handoff_Llama4_PastTheChunkBoundary_DecodeAgreesWithCpu`; results in the Llama 4 entries above). Closed 2026-11-01: the box was stale; checked the receipt and both test names in the code, did not re-run the 4-layer Scout checkpoint.
