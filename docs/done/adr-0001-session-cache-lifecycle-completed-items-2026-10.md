# Completed items split from adr-0001-session-cache-lifecycle.md (2026-11-01)

Finished items moved verbatim out of [../reference/adr-0001-session-cache-lifecycle.md](../reference/adr-0001-session-cache-lifecycle.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Verification required before broadening scope
- [x] A fake-backend unit test covers ownership transfer, append re-admission and single-writer exclusion.

> from: ## Verification required before broadening scope
- [x] Mid-generation cancellation proves turn-start rollback before retention with a deterministic fake backend.

> from: ## Verification required before broadening scope
- [x] Existing continuous-batching and prefix-cache tests still pass.

> from: ## Verification required before broadening scope
- [x] The initial API, ownership boundary and patch surface are recorded here; persistence remains
  blocked on the real CPU-model equivalence gate.

