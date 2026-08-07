# Quality-of-life improvements — current work

**Status:** planning/diagnostics baseline shipped; deterministic static-plan decision fixtures
and typed host-key classification are now in place. Environment ownership and execution parity remain.

## Remaining work

1. Classify every remaining environment variable; remove obsolete bench switches. The 41 typed
   server keys have a conservative ownership register in `host-config-inventory.md`.
2. Extend effective configuration from static plan inputs to real server/loader startup values.
   Server environment precedence is centralized in `ServerEnvironmentOverrides`; `/status`
   publishes its non-sensitive applied-variable receipt. Remaining work is a source-tracked
   snapshot of the bound configuration and loader-resolved decisions, without exposing values
   such as filesystem paths.
3. Extend the golden capability fixtures only where a loader actually executes the route. Static
   CPU/backend/dtype/batching/speculation decisions are covered; hardware rows remain release gates.
4. Finish server observability: VRAM/RAM breakdown, streaming timing, and per-request cache signal.
   The opt-in non-streaming timing extension now covers OpenAI, Anthropic, and Responses;
   it deliberately remains absent from streaming event contracts.
5. Add model/API compatibility corpus and protocol smoke coverage.
6. Keep sessions out of the product configuration surface until restart continuation is proven.

Inputs: [env-var-inventory.md](env-var-inventory.md), [host-config-inventory.md](host-config-inventory.md),
and [eligibility-check-inventory.md](eligibility-check-inventory.md).

Historical implementation record: [done/quality-of-life-improvements-plan-2026-07.md](done/quality-of-life-improvements-plan-2026-07.md).
