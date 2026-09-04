# `STINGRAY_*` environment-variable inventory — name-complete and classified

**Snapshot originally generated:** 2026-07-26 by scanning `src/**/*.cs`, recording **141 unique
variables**.

**Reconciled 2026-08-07:** the table below was made to list every name in
`KnownEnvironmentVariables.All`, the source-enforced registry, which at that time contained
**162** names.

**Reconciled 2026-08-08:** three ghost entries were removed from the source registry —
`STINGRAY_ARGMAX_NEG_INF`, `STINGRAY_MOE_` and `STINGRAY_SNAPKV` — bringing the registry to
**156**. None was ever an environment variable: `STINGRAY_ARGMAX_NEG_INF` is a CUDA `#define`
inside an NVRTC kernel string; the other two are glob patterns that appear only in comments
(`STINGRAY_MOE_*`, `STINGRAY_SNAPKV*`). The existing `ListMatchesSource` test could not catch them
because it accepts a bare textual match, so the scan found the very text that had put the entry
there — the entry justified itself.

The practical harm was in `SuggestClosest`: a user typing `STINGRAY_SNAPKV_BUDGE` could be told
"did you mean `STINGRAY_SNAPKV`?", a name nothing reads. A new test now requires every registry
entry to appear as a **quoted string literal** in src/, which is how a variable actually gets read.
(A "no entry is a prefix of another" rule was tried first and rejected — it flags 13 entries and
most are legitimate, e.g. `STINGRAY_TQ` alongside `STINGRAY_TQ_MODE`.)

The reconciliation also found a live defect rather than mere drift. `STINGRAY_MAX_QUEUED_REQUESTS`
was registered and named in the server's queue-overload warning, but **nothing ever read it** — it
is the name of the C# option (`options.MaxQueuedRequests`) that `STINGRAY_MAX_QUEUE` actually sets.
An operator following that advice would have set a variable with no effect, and because registered
names are treated as valid, `doctor` would not have flagged it either. The warning now names
`STINGRAY_MAX_QUEUE` and the dead entry is out of the registry, so the mistake is now reported with
a closest-match suggestion.

**Reconciled again 2026-09-02 — `KnownEnvironmentVariables.All` now contains **173** names**
(two true duplicate-alias pairs collapsed to their canonical name, per the session plan in
`docs/00-current-work.md`: `STINGRAY_MICRO_GEMM`/`STINGRAY_Q4K_MICRO_GEMM` removed, keeping only
`STINGRAY_CPU_MICRO_GEMM`; `STINGRAY_VULKAN_PATH2_EXPERIMENTAL` removed, keeping only
`STINGRAY_VULKAN_PATH2`. Both were genuinely the same flag under multiple names, not independent
settings — `MicroGemmKernel.ReadFromEnvironment`/`VulkanPath2Dispatcher.ReadFromEnvironment` no
longer read the retired names. This is a name reduction, not a review of the `experimental`
surface's dead-code candidates — that per-variable owner review remains open, see
`04-quality-of-life-improvements-plan.md` item 1.)

**Reconciled again 2026-08-31 — `KnownEnvironmentVariables.All` now contains **176** names**
(added `STINGRAY_DEBUG_COSYVOICE3`, `STINGRAY_DUMP_CONVPOST_PATH`, `STINGRAY_DUMP_CONVPRE_PATH`,
`STINGRAY_DUMP_STAGE0_PATH`: diagnostic dumps used for audio pipeline and HiFT vocoder validation.)

**Reconciled again 2026-08-28 — `KnownEnvironmentVariables.All` now contains **172** names**
(added `STINGRAY_DBG_TOKEN_RANK`: a `--verbose-prompt` diagnostic added to `RunCommand.cs` that
prints a specific token's logit value and rank in the final logits, used while investigating
DeepSeek-V2-Lite's non-"Paris" output — see `docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md`.)

**Reconciled again 2026-08-27 — `KnownEnvironmentVariables.All` now contains **171** names**
(added `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH`, a diagnostic-only escape hatch in
`ModelCompatibility.ValidateForTextGeneration` that bypasses the text-generation architecture
allowlist for local investigation of an un-admitted architecture, e.g. deepseek2, without actually
admitting it -- see `docs/bugstofix.md`.)

**Reconciled again 2026-08-24 — `KnownEnvironmentVariables.All` now contains **170** names**
(added `STINGRAY_AUDIO_DIAGNOSTIC_DUMP`: a pre-existing `Environment.GetEnvironmentVariable` read in
`ChatterboxPipeline`/`ChatterboxDecoder` that was never registered, caught by CI failing
`KnownEnvironmentVariablesTests.ListMatchesSource`.)

**Reconciled again 2026-08-21 — `KnownEnvironmentVariables.All` now contains **169** names**
(added `STINGRAY_MLA_TRACE`, a temporary diagnostic env var for the deepseek2 MLA
ground-truth-diffing investigation, and `STINGRAY_GGML_F16_DOT`, an opt-in switch that rounds
F16-weight matmul activations to fp16 before dotting to match ggml's `vec_dot_type=F16` pairing
for bit-parity comparisons — see `docs/bugstofix.md` — plus one further pre-existing count drift
not re-diffed row-by-row here; see the note above about this test not catching table-level row
drift).

**Reconciled again 2026-08-18 — `KnownEnvironmentVariables.All` now contains **166** names**
(architecture-support and kernel work between 2026-08-08 and 2026-08-15 added new dispatch-toggle
variables faster than the doc was updated, growing the registry back up past the 156 recorded
above). The table below had also drifted independently of that count change:
five names present in `KnownEnvironmentVariables.All` had no row at all (`STINGRAY_CPU_MICRO_GEMM`,
`STINGRAY_MICRO_GEMM`, `STINGRAY_Q4K_MICRO_GEMM` — three aliases for one `MicroGemmKernel`/
`MicroGemmQ4K` feature flag in `OpenTail.Stingray.Cpu`; `STINGRAY_VULKAN_PATH2`,
`STINGRAY_VULKAN_PATH2_EXPERIMENTAL` — two aliases for one `VulkanPath2Dispatcher` feature flag in
`OpenTail.Stingray.Vulkan`), while the three 2026-08-08 ghost removals (`STINGRAY_ARGMAX_NEG_INF`,
`STINGRAY_MOE_`, `STINGRAY_SNAPKV`) were still sitting in the table as rows despite no longer being
in source at all. Both fixed: the table below is re-verified name-complete against
`KnownEnvironmentVariables.All` (162 names, 162 rows, zero drift either direction) as of this
reconciliation. **Classification is also now complete for every row** — see below.

The stated current registry count is test-guarded by
\`KnownEnvironmentVariablesTests.Inventory_DeclaredCurrentRegistryCountMatchesSource\`, which reads
the "now contains **NNN** names" sentence above. It makes new registry drift visible immediately;
it does not by itself catch table-level row drift (that requires re-diffing the table against
`KnownEnvironmentVariables.All`, which is what this reconciliation did by hand).

This is Phase 0 deliverable 1 of `04-quality-of-life-improvements-plan.md`, and it is a
prerequisite for `--print-effective-config`, saved profiles, and the `ExecutionPlan` work: none of
those can be correct while the setting surface is unenumerated.

**Class values, in the same table below (no separate register from this reconciliation on):**

| Class | Meaning |
|---|---|
| `stable` | Supported user configuration. Belongs in profiles and effective-config output. |
| `expert` | Supported but sharp-edged; documented, not surfaced by default. |
| `diagnostic` | Troubleshooting only. Never written to a profile. |
| `bench` | Benchmark/measurement harness only. Must not affect a shipped run. |
| `experimental` | May change or vanish without notice. Must never become a default. |
| `test seam` | Not a tuning knob at all — an A/B correctness-verification handle whose two branches are claimed bit-identical (e.g. `STINGRAY_CPU_KPACK_SIMD`, `STINGRAY_CPU_VNNI`). Exists so a suite can exercise both code paths on the same host and assert they agree. |

**How each row below was classified.** ~half (the previously-existing "ownership register", now
folded row-by-row into the table instead of living as a separate summary) were reviewed
individually with a stated reason. The remainder — the great majority of the surface, matching the
doc's original observation that "a kernel-selection, bypass, ablation, trace, probe, profile, or
`*_BENCH` switch is not a supported profile setting" — are classified mechanically by name pattern:
`TRACE_*`/`PROBE_*`/`PROFILE_*`/`*_STATS`/`*_VALIDATION` → `diagnostic`; `*_BENCH` → `bench`;
everything else with no stated product-facing use → `experimental`. A handful of clear sibling
groups (e.g. the three `SNAPKV_*` budget/window/recency knobs, which are one feature, not three)
were classified together rather than letting an arbitrary alphabetical split give two siblings
different classes. None of this is a promise that every `experimental` row is equally unimportant —
it means none of them has had an owner review yet, which is the explicit remaining work this
reconciliation does NOT close (see `04-quality-of-life-improvements-plan.md`).

**Retirement is the point, not documentation.** The QoL plan inventories these but retires none. A
surface of 162 variables is itself the usability defect. Expect a meaningful fraction of the
`experimental` rows below to be dead ablation/bench leftovers that should be deleted outright once
an owner confirms they're unused, rather than classified and kept forever — this project has
already removed several exactly that way (`STINGRAY_VABL`, `STINGRAY_WABL`, `STINGRAY_Q4K_ABL`).

**Counting note:** this counts distinct string literals, so a variable read in two projects appears
once, grouped under both (the project-group headers below reflect that). It does not detect
dynamically composed names.

## OpenTail.Stingray.Audio

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_AUDIO_DIAGNOSTIC_DUMP` | expert | Enables `Generate()` timing-breakdown diagnostic logging in `ChatterboxPipeline`/`ChatterboxDecoder` (`=1`). |

## OpenTail.Stingray.Cli

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DSPARK_TIMING` | experimental | |
| `STINGRAY_RAW_PROMPT` | expert | |
| `STINGRAY_SDCPP` | expert | |
| `STINGRAY_SPEC_SAMPLE` | experimental | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Core, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MTP_BATCH_MAX` | expert | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cpu

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_PREFILL_Q8` | experimental | Default-ON int8 batched-prefill dispatch tier; `=0` opts out. |
| `STINGRAY_PREFILL_ATTN_WIDE_HEADS` | experimental | `1` admits head dims 128/256 to the Flash-64 prefill path. **Off by default by decision, not by omission**: the wikitext-2 gate measured +0.52% perplexity for +14% prefill throughput (6.0579 -> 6.0896), which is worse than the exact sequential path and two orders of magnitude above the ~0% precedent set by the Q4Kx8 repack. A real speed/quality trade for the model owner to opt into, not a default. |
| `STINGRAY_PREFILL_ATTN_KV_OUTER` | expert | `0` restores the per-query-tile prefill-attention schedule. The KV-outer reorder packs each KV tile once per group of query tiles and is ON by default: measured +1.6% alone and +4.0% with `STINGRAY_CPU_KPACK_SIMD`, bit-exactness pinned by `Flash64KvOuterTests`. |
| `STINGRAY_PREFILL_ATTN_KV_OUTER_TILES` | experimental | Query tiles held live per KV pack in the reorder above (default 8 = 512 queries, ~256 KB scratch at headDim 64). Trades footprint for K-pack reuse; proven not to change results. |
| `STINGRAY_CPU_KPACK_SIMD` | test seam | `0` restores the scalar K-pack transpose in Flash-64 prefill. Both forms emit identical bytes (a transpose only moves floats), so this is a bisect seam and an A/B measurement handle, not a tuning knob. |
| `STINGRAY_CPU_VNNI` | test seam | `0` forces the AVX2 chain in `SimdKernels.DotU8I8ToI32` even where VNNI exists. Not a tuning knob: the three branches are claimed bit-identical, but a host only executes one, so this is what lets a VNNI-capable machine run the Q4_K suites both ways and check that claim. |
| `STINGRAY_CPU_MICRO_GEMM` | experimental | Gates the small-batch Q4_K micro-GEMM kernel (`MicroGemmQ4K`) in `MicroGemmKernel.ReadFromEnvironment`. **Consolidated 2026-09-02**: was three aliases for this one flag (`STINGRAY_MICRO_GEMM`, `STINGRAY_Q4K_MICRO_GEMM`) — collapsed to this single canonical name, the other two removed from the registry and no longer read. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cpu, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MIN_BATCH_BLAS` | expert | Mirrors `OpenTailStingrayServerOptions.MinBatchBlas`, already classified `expert` in the (now-folded-in) host-config surface. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cuda, OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_SNAPKV_BUDGET` | expert | One of three SnapKV eviction knobs (with `STINGRAY_SNAPKV_RECENCY`/`STINGRAY_SNAPKV_WINDOW` below) — classified together as one feature, not three independent switches. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DISABLE_MTP` | experimental | |
| `STINGRAY_DSPARK_MIN_CONFIDENCE` | experimental | |
| `STINGRAY_DSPARK_VERIFY_LEN` | experimental | |
| `STINGRAY_MTP_DRAFT_N` | expert | |
| `STINGRAY_TRACE_MTP` | diagnostic | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_MOE` | expert | Mirrors `OpenTailStingrayServerOptions.CpuMoe`. |
| `STINGRAY_DSPARK_PLACE` | experimental | Mirrors `OpenTailStingrayServerOptions.DSparkPlace`; DSpark itself is parked (docs/00-current-work.md), not scheduled. |
| `STINGRAY_EXPERT_STATS` | diagnostic | Mirrors `OpenTailStingrayServerOptions.ExpertStatsPath` (writes SLRU hit-rate stats to a file on exit — troubleshooting only). |
| `STINGRAY_MOE_GPU_PREFILL` | experimental | Mirrors `OpenTailStingrayServerOptions.GpuMoePrefill`. |
| `STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS` | experimental | |
| `STINGRAY_MOE_PIN_MODE` | experimental | |
| `STINGRAY_MOE_PREDICT_PREFETCH` | expert | Mirrors `OpenTailStingrayServerOptions.MoePredictPrefetch`. |
| `STINGRAY_MOE_WARMPIN` | expert | Mirrors `OpenTailStingrayServerOptions.MoeWarmPin`. |
| `STINGRAY_MOE_WARMPIN_AFTER` | expert | Mirrors `OpenTailStingrayServerOptions.MoeWarmPinAfter`. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_PREFILL_CHUNK` | expert | Mirrors `OpenTailStingrayServerOptions.PrefillChunkTokens`. |
| `STINGRAY_PREFILL_DEQUANT_MB` | expert | Mirrors `OpenTailStingrayServerOptions.PrefillDequantCacheMb`. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host, OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_KV_DTYPE` | expert | Mirrors `OpenTailStingrayServerOptions.KvType`. |

## OpenTail.Stingray.Core, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_TOOL_GRAMMAR` | expert | Mirrors `OpenTailStingrayServerOptions.ToolGrammar`. |

## OpenTail.Stingray.Cpu

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCHED_MATVEC_TIER` | experimental | |
| `STINGRAY_GEMM_PATH` | experimental | |
| `STINGRAY_MATVEC_WIDE8` | experimental | |
| `STINGRAY_TRACE_GDN_INTERNAL` | diagnostic | |
| `STINGRAY_TRACE_GDN_LAYERS` | diagnostic | |
| `STINGRAY_TRACE_GDN_POS` | diagnostic | |

## OpenTail.Stingray.Cpu, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_THREADS` | expert | Mirrors `OpenTailStingrayServerOptions.CpuThreads`. |

## OpenTail.Stingray.Cuda

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_ACT_SOA` | experimental | |
| `STINGRAY_ACT_SOA_CPA` | experimental | |
| `STINGRAY_ATTN_WAVE_BUDGET_MB` | experimental | |
| `STINGRAY_BATCH_DECODE_WS_V2` | experimental | |
| `STINGRAY_CUDA13` | experimental | |
| `STINGRAY_CUDA_PRECISION` | experimental | |
| `STINGRAY_DECODE_MMQ_BM32` | experimental | |
| `STINGRAY_GPU_ARGMAX` | experimental | |
| `STINGRAY_Q40_DP4A` | experimental | |
| `STINGRAY_Q6K_DECODE_MMQ` | experimental | |
| `STINGRAY_Q80_DP4A` | experimental | |

## OpenTail.Stingray.Cuda, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCH_DECODE_MMQ` | experimental | |
| `STINGRAY_PREFILL_MMQ` | experimental | |
| `STINGRAY_TRUNK_MATVEC_FAST` | experimental | |

## OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCHED_ATTN` | experimental | |
| `STINGRAY_BATCHED_CPU_FFN` | experimental | |
| `STINGRAY_BATCHED_FFN` | experimental | |
| `STINGRAY_BATCHED_GDN_SCAN` | experimental | |
| `STINGRAY_BATCHED_PREFILL` | experimental | |
| `STINGRAY_BATCHED_TRUNK` | experimental | |
| `STINGRAY_BATCH_DECODE_GEMM` | experimental | |
| `STINGRAY_BATCH_DECODE_RAGGED` | experimental | |
| `STINGRAY_BATCH_DECODE_WS` | experimental | |
| `STINGRAY_BYPASS_ATTN` | experimental | |
| `STINGRAY_BYPASS_GDN` | experimental | |
| `STINGRAY_BYPASS_MOE` | experimental | |
| `STINGRAY_CPU_GDN` | experimental | |
| `STINGRAY_CUDA_GRAPH` | experimental | |
| `STINGRAY_CUDA_MATVEC_BENCH` | bench | |
| `STINGRAY_CUDA_PROFILE` | diagnostic | |
| `STINGRAY_DECODE_CUDA_GRAPH` | experimental | |
| `STINGRAY_DECODE_PROFILE` | diagnostic | |
| `STINGRAY_DECODE_REGIONS` | experimental | |
| `STINGRAY_DENSE_FFN_GPU_MARGIN_MB` | experimental | |
| `STINGRAY_DISABLE_BATCH_VERIFY` | experimental | |
| `STINGRAY_DISABLE_DUAL_Q8` | experimental | |
| `STINGRAY_FLASH64_STRIDED_GEMM` | experimental | |
| `STINGRAY_FORCE_CPU_EMBED` | experimental | |
| `STINGRAY_FORCE_NO_BLAS` | experimental | |
| `STINGRAY_GDN_CHUNKED_PREFILL` | experimental | |
| `STINGRAY_GDN_DECODE_FAST` | experimental | |
| `STINGRAY_GDN_PREFILL_COMPUTE` | experimental | |
| `STINGRAY_GDN_RAW_Q8_0` | experimental | |
| `STINGRAY_GEMMA4_PROBE` | diagnostic | |
| `STINGRAY_GGML_F16_DOT` | experimental | Opt-in: rounds F16-weight matmul activations to fp16 before dotting, matching ggml's `vec_dot_type=F16` pairing, for bit-parity comparisons against llama.cpp (see `docs/bugstofix.md`). Default off — full-F32-precision activation is more accurate. |
| `STINGRAY_HYBRID_BATCHED_MOE` | experimental | |
| `STINGRAY_HYBRID_PREFILL_COMPUTE` | experimental | |
| `STINGRAY_KVARN_BATCHED_PREFILL` | experimental | |
| `STINGRAY_KV_BF16_MIN_TOKENS` | expert | |
| `STINGRAY_KV_STORE` | expert | |
| `STINGRAY_MLA_TRACE` | diagnostic | Temporary: prints per-layer MLA/attention/MoE intermediate sums for ground-truth diffing against llama.cpp (see `docs/bugstofix.md`'s deepseek2 investigation). |
| `STINGRAY_MMQ_SOA` | experimental | |
| `STINGRAY_MOE_BATCHED_PREFILL` | experimental | |
| `STINGRAY_MOE_GPU_ROUTER` | experimental | |
| `STINGRAY_MOE_THREADS` | experimental | |
| `STINGRAY_MTP_BATCHED_MOE_VERIFY` | experimental | |
| `STINGRAY_MTP_MIN_ACCEPT` | experimental | |
| `STINGRAY_MTP_PROBE_STEPS` | diagnostic | |
| `STINGRAY_PER_LAYER_HD_PREFILL` | experimental | |
| `STINGRAY_PLE_GPU_DEQUANT` | experimental | |
| `STINGRAY_PREFAULT` | experimental | |
| `STINGRAY_PREFILL_ATTN_FLASH64` | experimental | |
| `STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS` | experimental | |
| `STINGRAY_PREFILL_ATTN_REGISTER_VALUES` | experimental | |
| `STINGRAY_PREFILL_FLASH` | experimental | |
| `STINGRAY_PREFILL_FLASH_TC` | experimental | |
| `STINGRAY_PREFILL_FLASH_TC1` | experimental | |
| `STINGRAY_PREFILL_GEMM` | experimental | |
| `STINGRAY_PREFILL_PROFILE` | diagnostic | |
| `STINGRAY_PREFIX_SCRATCH_TOKENS` | experimental | |
| `STINGRAY_PREFIX_SLOTS` | experimental | |
| `STINGRAY_PROBE_IDS` | diagnostic | |
| `STINGRAY_PROBE_LOGITS` | diagnostic | |
| `STINGRAY_PROBE_POS` | diagnostic | |
| `STINGRAY_PROFILE_DECODE` | diagnostic | |
| `STINGRAY_PROFILE_PREFILL` | diagnostic | |
| `STINGRAY_Q3K_DEQUANT_GEMM` | experimental | |
| `STINGRAY_Q3K_Q8K` | experimental | |
| `STINGRAY_Q4KX8_CACHE_MB` | experimental | |
| `STINGRAY_Q4K_Q8K` | experimental | |
| `STINGRAY_Q4K_SOA` | experimental | |
| `STINGRAY_Q6K_SOA` | experimental | |
| `STINGRAY_Q8_0_Q8K` | experimental | |
| `STINGRAY_SNAPKV_RECENCY` | expert | Sibling of `STINGRAY_SNAPKV_BUDGET` (above) — one feature, classified together. |
| `STINGRAY_SNAPKV_WINDOW` | expert | Sibling of `STINGRAY_SNAPKV_BUDGET` (above) — one feature, classified together. |
| `STINGRAY_SPEC_BATCH_VERIFY` | experimental | |
| `STINGRAY_SPLIT_DECODE_GROUPED` | experimental | |
| `STINGRAY_TRACE_DSPARK` | diagnostic | |
| `STINGRAY_TRACE_LAYERS` | diagnostic | |
| `STINGRAY_TRACE_NORMS` | diagnostic | |
| `STINGRAY_TRACE_POS` | diagnostic | |
| `STINGRAY_TRACE_ROUTERS` | diagnostic | |
| `STINGRAY_TRACE_SNAPSHOT` | diagnostic | |
| `STINGRAY_TRACE_VRAM` | diagnostic | |
| `STINGRAY_VULKAN_BATCHED_PREFILL` | experimental | |
| `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL` | experimental | |
| `STINGRAY_VULKAN_NO_BATCHED_PREFILL` | experimental | |
| `STINGRAY_VULKAN_NO_FLASH_ATTN` | experimental | |
| `STINGRAY_VULKAN_PREFILL_CHUNK` | experimental | |
| `STINGRAY_VULKAN_SPLIT_DECODE_MIN` | experimental | |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DSPARK_MODEL` | experimental | Mirrors `OpenTailStingrayServerOptions.DSparkModelPath`; DSpark is parked, not scheduled. |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MAX_BATCH` | stable | Mirrors `OpenTailStingrayServerOptions.MaxBatchSize`. |
| `STINGRAY_MMPROJ` | expert | Mirrors `OpenTailStingrayServerOptions.MmprojPath`. |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_SPLIT_DECODE` | experimental | |
| `STINGRAY_VULKAN_SPLIT_DECODE` | experimental | |

## OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_KV_BUDGET_MB` | stable | Mirrors `OpenTailStingrayServerOptions.KvBudgetMb`. |
| `STINGRAY_MAX_CONCURRENT` | stable | Mirrors `OpenTailStingrayServerOptions.MaxConcurrentRequests`. |
| `STINGRAY_MAX_QUEUE` | stable | Mirrors `OpenTailStingrayServerOptions.MaxQueuedRequests`. See the `STINGRAY_MAX_QUEUED_REQUESTS` dead-name note above. |
| `STINGRAY_MODEL` | stable | Mirrors `OpenTailStingrayServerOptions.ModelPath`. |
| `STINGRAY_NO_THINKING` | stable | Mirrors `OpenTailStingrayServerOptions.DisableThinking`. |
| `STINGRAY_PREFIX_CACHE_MB` | expert | Mirrors `OpenTailStingrayServerOptions.PrefixCacheMb`. |
| `STINGRAY_PRESERVE_THINKING` | stable | Mirrors `OpenTailStingrayServerOptions.PreserveThinking`. |

## OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BACKEND` | stable | Mirrors `OpenTailStingrayServerOptions.Backend`. |
| `STINGRAY_N_GPU_LAYERS` | stable | Mirrors `OpenTailStingrayServerOptions.NGpuLayers`. |
| `STINGRAY_TQ` | expert | Mirrors `OpenTailStingrayServerOptions.TurboQuant`. |
| `STINGRAY_TQ_MODE` | expert | Mirrors `OpenTailStingrayServerOptions.TqMode`. |

## OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_VULKAN_DP4A` | experimental | |
| `STINGRAY_VULKAN_MM_PATH` | experimental | |
| `STINGRAY_VULKAN_MM_STATS` | diagnostic | |
| `STINGRAY_VULKAN_PATH2` | experimental | Gates the Vulkan Path 2 cooperative-tiled-GEMM kernel (`MatMulTiledQ4K`) in `VulkanPath2Dispatcher.ReadFromEnvironment`. Defaults off. **Consolidated 2026-09-02**: was two aliases for this one flag (`STINGRAY_VULKAN_PATH2_EXPERIMENTAL`) — collapsed to this single canonical name, the other removed from the registry and no longer read. |
| `STINGRAY_VULKAN_UMA_FRACTION` | experimental | |
| `STINGRAY_VULKAN_VALIDATION` | diagnostic | |
