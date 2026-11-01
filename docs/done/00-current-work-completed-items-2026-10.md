# Completed items split from 00-current-work.md (2026-11-01)

Finished items moved verbatim out of [../00-current-work.md](../00-current-work.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## 2. Model coverage
- [x] 9. **Gemma 1 (`gemma`)**: fixed 2026-11-01 on the CPU. It predicted `<bos>` at every position because the descriptor had no model semantics; `FamilyModelSemantics.Gemma1` added (sqrt(n_embd) embedding scale, GELU-tanh gated FFN). Golden `gemma.golden.json` 22/24 (near-tie) on gemma-1.1-2b-it Q8_0; STATUS row added. Only the 2B file, one case and the CPU were run; the 7B and Vulkan are unverified.

> from: ## 2. Model coverage
- [x] 11. **Missing GGML op kernels**: none left; RWKV6 / RWKV7 closed 2026-10-01 (see
    [done/bugstofix-resolved-2026-10.md](bugstofix-resolved-2026-10.md); the rest of the old op gap: [done/17](17-ggml-op-coverage-verification-plan.md)).

