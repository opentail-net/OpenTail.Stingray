# Quality-of-life improvements — Item 1 (env-var/CLI inventory reconciliation), DONE

> **CLOSED 2026-09-02.** Split out of [04-quality-of-life-improvements-plan.md](../3-product-and-runtime/04-quality-of-life-improvements-plan.md)
> on 2026-09-27 — this was the only fully-closed item in that doc; items 2-6 remain open there.

**DONE, 2026-08-15.** Regenerated and fully classified both inventories.
**Alias dedup, 2026-09-02:** two true duplicate-alias pairs (same flag, multiple names) found
and collapsed to their canonical name — `STINGRAY_MICRO_GEMM`/`STINGRAY_Q4K_MICRO_GEMM` retired
in favor of `STINGRAY_CPU_MICRO_GEMM`; `STINGRAY_VULKAN_PATH2_EXPERIMENTAL` retired in favor of
`STINGRAY_VULKAN_PATH2`. Registry 176 → 173. `MicroGemmKernel.ReadFromEnvironment`/
`VulkanPath2Dispatcher.ReadFromEnvironment` no longer read the retired names; see
`env-var-inventory.md`'s 2026-09-02 reconciliation entry for detail. Full solution rebuilds
clean; `KnownEnvironmentVariablesTests` (`ListMatchesSource`,
`Inventory_DeclaredCurrentRegistryCountMatchesSource`) pass.
**Dead-switch sampling check, 2026-09-02:** randomly sampled 20 of the 101 `bench`/`experimental`
rows and checked each one's actual call site in `src/` by hand (not just grep-presence — read
whether the gated code path is real and reachable). **Result: 0 of 20 were dead.** Every one is
read at exactly one live call site gating real, currently-reachable dispatch code (kernel-tier
selection, CUDA/Vulkan feature toggles, MoE/GDN/decode paths); 7 of the 20 gate CUDA-specific
code that's merely unreachable on this specific machine (no NVIDIA GPU, integrated AMD Radeon
only) — real code for anyone on NVIDIA hardware, not dead code here. This is evidence against
this doc's own earlier prediction ("expect a meaningful fraction... to be dead ablation/bench
leftovers") — at least for this sample, the `experimental` surface is load-bearing, not debt. A
full 101-row manual audit is very unlikely to yield much deletion based on this sample; not
pursued further this session on that basis. If revisited, a larger or stratified sample (e.g.
deliberately including older/rarely-touched names) would be needed before concluding the whole
surface is clean, since 20/101 is suggestive, not exhaustive.
`env-var-inventory.md`: the registry had drifted independently in both directions since the
2026-08-08 reconciliation (grown to 162 names; the table was separately missing 5 rows and still
carrying 3 confirmed-ghost rows that don't exist in source) — re-diffed to zero drift either way,
then every row given an explicit Class (previously only ~half were covered, via a separate
"ownership register" summary that had never been propagated down into the per-row table; the
remainder default to `experimental` per the doc's own stated fallback rule, applied mechanically
by name pattern — `TRACE_*`/`PROBE_*`/`PROFILE_*`/`*_STATS`/`*_VALIDATION` → `diagnostic`,
`*_BENCH` → `bench`, sibling groups like the three `SNAPKV_*` knobs classified together).
`cli-option-inventory.md`: found and fixed a real generator bug along the way — the description
parser only matched `[Description("...")]` on a single line, so multi-line concatenated
attributes (`"..." + "..."`) silently produced blank descriptions (3 rows affected, all in
`RunCommand`); fixed to accumulate across lines, re-ran clean at the same 153-row count, then
classified every row. The 41 typed server keys have their own ownership register in
`reference/host-config-inventory.md`, already complete since 2026-08-08. **Not done as part of this pass:**
removing obsolete bench switches — classification surfaced which rows are `bench`/dead-looking
`experimental` candidates, but retiring any of them needs a per-variable owner call, not a
drive-by deletion.
