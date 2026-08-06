# Quality-of-life improvements — current work

**Status:** planning/diagnostics baseline shipped; configuration ownership and execution parity remain.

## Remaining work

1. Classify every environment variable and host key; remove obsolete bench switches.
2. Extend effective configuration from static plan inputs to real server/loader startup values.
3. Add golden capability fixtures for model/backend/dtype/batching/speculation decisions.
4. Finish server observability: VRAM/RAM breakdown, streaming timing, and per-request cache signal.
5. Add model/API compatibility corpus and protocol smoke coverage.
6. Keep sessions out of the product configuration surface until restart continuation is proven.

Inputs: [env-var-inventory.md](env-var-inventory.md), [host-config-inventory.md](host-config-inventory.md),
and [eligibility-check-inventory.md](eligibility-check-inventory.md).

Historical implementation record: [done/quality-of-life-improvements-plan-2026-07.md](done/quality-of-life-improvements-plan-2026-07.md).
