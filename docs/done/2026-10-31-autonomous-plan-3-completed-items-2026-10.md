# Completed items split from 2026-10-31-autonomous-plan-3.md (2026-11-01)

Finished items moved verbatim out of [../2026-10-31-autonomous-plan-3.md](../2026-10-31-autonomous-plan-3.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist
- [x] **P1c** *(owner decided 2026-11-01 to stop the hunt; reopen only if the one-line canary in the Vulkan hybrid MoE test fires)* The NaN in hybrid MoE decode (Qwen3-Coder `-g 4`): 1 sighting in 18 control runs, then 0 in 12, then 0 in a 40-run hunt with the finiteness assertion (0 non-finite, 0 failed, min step cosine 0.9838 to 0.9979): 0 in 52 runs since the sighting. Not reproduced and not explained; open until reproduced or until the owner decides to stop hunting

> from: ## Checklist
- [x] **Q1** Full-GPU against CPU perplexity sweep over the dense checkpoints on disk. Done 2026-10-31 (`stingray perplexity -c 512` on the first 512 tokens of `wiki.test.raw`, `-g 0` against `-g -1`; the CPU run uses an fp32 KV cache and the GPU run bf16, so a fraction of a percent is not kernel error). No model shows a gap like gemma-2 did (+22 percent). Qwen3-0.6B Q8_0 19.07 / 19.12 (+0.3%); Falcon3-3B 10.16 / 10.09 (-0.6%); SmolLM3 8.893 / 8.901 (+0.1%); Qwen3-4B 14.20 / 14.12 (-0.5%); gemma-3-4b 16.43 / 16.66 (+1.4%, the largest; consistent with its step-level cosine of 0.992 / 0.9993); Granite 3.3-2B 7.460 / 7.404 (-0.8%); ERNIE-4.5-0.3B Q8_0 16.30 / 16.37 (+0.4%); Maincoder-1B 15.80 / 15.85 (+0.4%). One 512-token window per model, not a full-corpus figure

> from: ## Checklist
- [x] **Q2** Apply embedding, residual, logit and attention-scale fields in the Vulkan hybrid so Granite and Gemma 3 can partial-offload. Done 2026-10-31 (the owner asked for it, with "refuse if not useful"). Gemma 3 also needed the sliding-window RoPE base (10K on local layers; the hybrid used 1M everywhere), applied to both the GPU and CPU layers. Against the CPU pass on real weights, `-g 8` and `-g 20`: Granite 3.3-2B minimum per-step cosine 0.996, median 0.9995 (was -0.36); gemma-3-4b minimum 0.990, median 0.9992 (was 0.47), the same as the full Vulkan pass (0.992 / 0.9993). The refusal is now a soft-cap guard (Gemma 2 only). Ten mutations of the production change each fail a test (the logit-scale one only after a norm-ratio check was added, because cosine ignores a uniform scale). **Not faster on this machine**: decode `-g 8` 18.1 tok/s against 21.3 for `-g 0` (Granite) and 14.1 against 14.5 (gemma-3-4b), short-prompt prefill 6 against 45 tok/s, three interleaved runs each, machine not idle. This is the integrated GPU (rule 13), so it does not say a discrete GPU would lose; what it enables is a correct split, not a proven speed-up. No handoff receipt was added for either family

> from: ## Checklist
- [x] **Q3** *(decided 2026-11-01 by the owner: no; gemma2 stays CPU only, the limitation is in STATUS)* GPU soft-capping for gemma2 (attention and final logits), so the dense GPU path matches the CPU (perplexity 15.38 against 12.58 today). Needs an owner decision

