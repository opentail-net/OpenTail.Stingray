# Completed items split from 2026-10-03-streamed-residency-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-03-streamed-residency-plan.md](../2-coverage/2026-10-03-streamed-residency-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] `STINGRAY_MOE_TIMING=1` diagnostic (where a layer-step's time goes)

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] `STINGRAY_MOE_SLOTS=N` to force a small cache for testing

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] Fallback input copy (defect 1); same change applied to `VulkanHybridGdnForwardPass`, which has no local checkpoint
  to run (build-verified only)

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] Upload outside the lock (defect 2); `ExpertSlotManagerConcurrencyTests`, shown to fail on the old code

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] Slot leases (defect 3); `VulkanHybridOlmoeParityTests` at 8 and 16 slots

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] Memory-bounded test (`StreamedExpertMemoryBoundTests`) and a `VulkanBackend.LiveBufferCount` diagnostic

> from: ## 3. Done (all on the Vulkan hybrid path, `-g N` on a MoE model)
- [x] Eviction correctness: logits within cosine 0.996-0.999 of CPU with no argmax flips on OLMoE (64 experts) at 8 and 16
  slots, and Qwen3-Coder-30B-A3B (128 experts x 4 GPU layers through 16 slots): cosine 0.9982, one argmax flip on a
  near-tie (allowed by the test contract: only a gap above 2 % of the logit range fails)

