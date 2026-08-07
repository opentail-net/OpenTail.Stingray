# Quality-of-life improvements — current work

**Status:** planning/diagnostics baseline shipped; deterministic static-plan decision fixtures
and typed host-key classification are now in place. Environment ownership and execution parity remain.

## Remaining work

1. Regenerate the drifted CLI/environment inventories from source, then classify every remaining
   setting and remove obsolete bench switches. The 41 typed server keys have a conservative
   ownership register in `host-config-inventory.md`.
2. Extend effective configuration from static plan inputs to real server/loader startup values.
   Server environment precedence is centralized in `ServerEnvironmentOverrides`; `/status`
   publishes its non-sensitive applied-variable receipt and a `configuration.bound` snapshot
   after that precedence has been applied (admission, CPU/prefill/cache budgets, TurboQuant/KV,
   tool grammar, and session persistence as a boolean — never filesystem paths). Remaining work
   `configuration.resolved` now also records the concrete built-in loader route (actual backend,
   forward-pass family, model format, and resolved context), so a requested `auto`/hybrid setup is
   no longer mistaken for the route that actually loaded. Remaining work is actual device/cache
   allocation telemetry, without exposing values such as filesystem paths. `/capabilities` now
   also states the restart-session verdict explicitly rather than leaving session support ambiguous.
3. Extend the golden capability fixtures only where a loader actually executes the route. Static
   CPU/backend/dtype/batching/speculation decisions are covered; hardware rows remain release gates.
4. Finish server observability: VRAM/RAM breakdown, streaming timing, and per-request cache signal.
   The opt-in non-streaming timing extension now covers OpenAI, Anthropic, and Responses;
   it deliberately remains absent from streaming event contracts.
5. Add model/API compatibility corpus and protocol smoke coverage. The small llama-server
   compatibility surface (`/tokenize`, `/detokenize`, `/props`) now has fake-engine wire-contract
   coverage, including malformed-payload errors; expand only with a named consumer contract.
6. Keep sessions out of the product configuration surface until the proven CPU-dense restart lane
   has a named lifecycle and capability gate. The proof itself is now covered by the real-GGUF,
   cross-process replay acceptance test; it is not yet a product contract.

Inputs: [env-var-inventory.md](env-var-inventory.md), [host-config-inventory.md](host-config-inventory.md),
and [eligibility-check-inventory.md](eligibility-check-inventory.md).

Historical implementation record: [done/quality-of-life-improvements-plan-2026-07.md](done/quality-of-life-improvements-plan-2026-07.md).
