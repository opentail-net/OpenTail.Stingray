# Correctness plan

Status 2026-11-08. The one plan for correctness work outside the Qwen port and the bug log. It replaces the open boxes of the prefill-numerics investigation and the documentation reconciliation plan. The bug log ([bugstofix.md](bugstofix.md)) keeps its one open entry, listed here. Unverified port claims are in [2026-10-03-unverified-port-claims.md](2026-10-03-unverified-port-claims.md), with the owner's approvals in the merged plan ([../2026-11-08-open-work-merged.md](../2026-11-08-open-work-merged.md), B5).

**Rules:** a correctness claim names its reference and the revision used. Numerics changes are gated by perplexity against llama.cpp and recorded in an ADR (see [numerics-investigation-method](../reference/numerics-investigation-method.md)). "Reproducible is not accurate": a stable result is not the same as a correct one.

---

## Part A: Runnable now, in order

### A1. Q8_KS against ggml Q8_K: close the activation question

- **Why:** the item was narrowed on 2026-11-03. Q8_KS is finer than ggml's Q8_K by design, so aligning them would make Stingray's activations coarser. The 0.69% perplexity gap to llama.cpp is not explained by the activation quantiser alone. The per-weight-type comparison is open (merged plan A9b).
- **Next step:** finish the per-weight-type comparison from merged plan A9b. Then write the conclusion in the investigation doc: either the gap is named, or it is recorded as unexplained by the quantiser.
- **Done when:** the investigation doc has a single conclusion line with the evidence, and the open box is closed.

### A2. MoE small prompts through the folded decode path

- **Why:** prompts of 32 tokens or fewer could go through the folded decode-style dispatch. The result would be bit-identical, and the change would be a speed question.
- **Next step:** this is the same item as merged plan A9a. Measure it there; do not measure it twice.
- **Done when:** the merged plan item is closed.

### A3. LLaVA-NeXT AnyRes residual (bug log entry 15)

- **Why:** the main defect (view order) is fixed. Geometry, token counts and the first token agree with llama.cpp. Still unverified: pixel-level parity and per-view embedding parity.
- **Next step:** compare the per-view embeddings against the pinned llama.cpp `mtmd` debug output, the same route as the Qwen vision check. Record the comparison.
- **Done when:** the pixel-level and per-view comparisons are recorded, or the entry in the bug log names the remaining difference.
- **Doc:** [bugstofix.md](bugstofix.md), entry 15.

---

## Part B: Decisions

1. **Documentation claim reconciliation, R2 (HunyuanVideo and Wan 2.2).** The owner has decided "duplicate first". The item is blocked by the checkpoints and, for HunyuanVideo v1, the missing independent reference (see [../2026-11-08-open-work-merged.md](../2026-11-08-open-work-merged.md), Part C).

---

## Part C: Blocked

- **MTP re-baseline and the 27B MTP validation** (`MtpDecoder_GreedyParity_LlamaCpp` and ADR-0002 follow-ups). The re-baseline needs `Qwen3.6-27B-MTP-Q4_K_M.gguf`, which is not on this machine. Owner doc: [../reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md](../reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md).
- **HunyuanVideo and Wan 2.2 findings (R2).** See Part B.
- **Ported-but-unverified families' correctness claims** (rule 14). See [2026-10-03-unverified-port-claims.md](2026-10-03-unverified-port-claims.md).

---

## Owner docs kept as detail

- [2026-10-02-prefill-numerics-investigation.md](2026-10-02-prefill-numerics-investigation.md): the evidence behind A1 and A2.
- [2026-10-03-unverified-port-claims.md](2026-10-03-unverified-port-claims.md): the claims and their blockers.
- [2026-10-09-documentation-claim-reconciliation-plan.md](2026-10-09-documentation-claim-reconciliation-plan.md): R2.
- [bugstofix.md](bugstofix.md): the bug log.
- [2026-11-07-gemma4-tensorsharp-reference-comparison.md](2026-11-07-gemma4-tensorsharp-reference-comparison.md): the Gemma 4 comparison record.
- [2026-11-01-documentation-claim-reconciliation-report.md](2026-11-01-documentation-claim-reconciliation-report.md): the report.
