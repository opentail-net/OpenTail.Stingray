# Completed items split from 2026-10-02-prefill-numerics-investigation.md (2026-10-31)

Finished items moved verbatim out of [../1-correctness/2026-10-02-prefill-numerics-investigation.md](../1-correctness/2026-10-02-prefill-numerics-investigation.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist
- [x] 1. Evidence gathering (code and history)

> from: ## Checklist
- [x] 2. llama.cpp's own recurrence (reviewer)

> from: ## Checklist
- [x] 3. Attribution matrix design

> from: ## Checklist
- [x] 4. First result: short prompts

> from: ## Checklist
- [x] 5. Higher-precision recurrence reference D

> from: ## Checklist
- [x] 6. Full matrix, prompt as the statistical unit

> from: ## Checklist
- [x] 7. GDN probe: born in the recurrence or amplified downstream

> from: ## Checklist
- [x] 8. Prefill -> decode continuity

> from: ## Checklist
- [x] 9. The two KL outliers and the row check

> from: ## Checklist
- [x] 10. int8 default interaction

> from: ## Checklist
- [x] 11. ggml cross-check on identical rows (done 2026-10-02)


---

## More completed items moved from 2026-10-02-prefill-numerics-investigation.md on 2026-11-01

Finished items moved verbatim out of [../1-correctness/2026-10-02-prefill-numerics-investigation.md](../1-correctness/2026-10-02-prefill-numerics-investigation.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ### Follow-ups from the MoE audit (moved from docs/todo.md, 2026-10-30)
- [x] *(closed 2026-11-01: confirmed in `TryResolveQ8Dispatch` and measured harmless, ADR-0003 addendum; both formats kept.)* Q3_K batched path uses Q8_KS while decode uses Q8_K (claimed; verify in
      `TryResolveQ8Dispatch`). Unifying on Q8_K would make Q3_K prefill match decode and ggml

