# `STINGRAY_*` environment-variable inventory — historical snapshot, regeneration required

**Snapshot generated:** 2026-07-26 by scanning `src/**/*.cs`. It records **141 unique
variables**.

**Drift found 2026-08-07:** `KnownEnvironmentVariables.All`, the source-enforced registry,
now contains **156** names. The 141 rows below are consequently historical classification input,
not a complete current environment surface. Regenerate this document from the registry/source
before using it to define supported configuration, profiles, or precedence.

The stated current registry count is test-guarded by
\`KnownEnvironmentVariablesTests.Inventory_DeclaredCurrentRegistryCountMatchesSource\`. It makes
new registry drift visible immediately; it does not replace the pending owner/use-row refresh.

This is Phase 0 deliverable 1 of `02-quality-of-life-improvements-plan.md`, and it is a
prerequisite for `--print-effective-config`, saved profiles, and the `ExecutionPlan` work:
none of those can be correct while the setting surface is unenumerated.

**The Class column is deliberately empty.** Classification is a judgement call per variable and
must be made by whoever owns the behaviour — it cannot be inferred from the name. Fill each with
one of:

| Class | Meaning |
|---|---|
| `stable` | Supported user configuration. Belongs in profiles and effective-config output. |
| `expert` | Supported but sharp-edged; documented, not surfaced by default. |
| `diagnostic` | Troubleshooting only. Never written to a profile. |
| `bench` | Benchmark/measurement harness only. Must not affect a shipped run. |
| `experimental` | May change or vanish without notice. Must never become a default. |

Until a variable is classified, treat it as `experimental` — the safe assumption.

## Ownership register — product-facing variables (reviewed)

The generated inventory below remains the complete source list. This register records the
high-confidence deployment surface that may be documented or represented in effective
configuration; all names not listed here remain `experimental` pending an owner review. In
particular, a kernel-selection, bypass, ablation, trace, probe, profile, or `*_BENCH` switch is
not a supported profile setting.

| Class | Variables |
|---|---|
| stable | `STINGRAY_BACKEND`, `STINGRAY_MODEL`, `STINGRAY_N_GPU_LAYERS`, `STINGRAY_MAX_BATCH`, `STINGRAY_MAX_CONCURRENT`, `STINGRAY_MAX_QUEUE`, `STINGRAY_MAX_QUEUED_REQUESTS`, `STINGRAY_KV_BUDGET_MB`, `STINGRAY_NO_THINKING`, `STINGRAY_PRESERVE_THINKING` |
| expert | `STINGRAY_CPU_THREADS`, `STINGRAY_KV_DTYPE`, `STINGRAY_KV_STORE`, `STINGRAY_KV_BF16_MIN_TOKENS`, `STINGRAY_MMPROJ`, `STINGRAY_MIN_BATCH_BLAS`, `STINGRAY_PREFILL_CHUNK`, `STINGRAY_PREFILL_DEQUANT_MB`, `STINGRAY_PREFIX_CACHE_MB`, `STINGRAY_TQ`, `STINGRAY_TQ_MODE`, `STINGRAY_SNAPKV_BUDGET`, `STINGRAY_TOOL_GRAMMAR`, `STINGRAY_CPU_MOE`, `STINGRAY_MOE_PREDICT_PREFETCH`, `STINGRAY_MOE_WARMPIN`, `STINGRAY_MOE_WARMPIN_AFTER`, `STINGRAY_MTP_DRAFT_N`, `STINGRAY_MTP_BATCH_MAX`, `STINGRAY_RAW_PROMPT`, `STINGRAY_SDCPP` |
| diagnostic | `STINGRAY_EXPERT_STATS`, `STINGRAY_GEMMA4_PROBE`, `STINGRAY_TRACE_DSPARK`, `STINGRAY_TRACE_GDN_INTERNAL`, `STINGRAY_TRACE_GDN_LAYERS`, `STINGRAY_TRACE_GDN_POS`, `STINGRAY_TRACE_LAYERS`, `STINGRAY_TRACE_MTP`, `STINGRAY_TRACE_NORMS`, `STINGRAY_TRACE_POS`, `STINGRAY_TRACE_ROUTERS`, `STINGRAY_TRACE_SNAPSHOT`, `STINGRAY_TRACE_VRAM`, `STINGRAY_CUDA_PROFILE`, `STINGRAY_DECODE_PROFILE`, `STINGRAY_PREFILL_PROFILE`, `STINGRAY_PROFILE_DECODE`, `STINGRAY_PROFILE_PREFILL`, `STINGRAY_PROBE_IDS`, `STINGRAY_PROBE_LOGITS`, `STINGRAY_PROBE_POS`, `STINGRAY_VULKAN_MM_STATS`, `STINGRAY_VULKAN_VALIDATION` |
| experimental | `STINGRAY_DSPARK_MODEL`, `STINGRAY_DSPARK_PLACE`, `STINGRAY_DSPARK_MIN_CONFIDENCE`, `STINGRAY_DSPARK_VERIFY_LEN`, `STINGRAY_DSPARK_TIMING`, `STINGRAY_MOE_GPU_PREFILL`, `STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS`, `STINGRAY_MOE_PIN_MODE`, `STINGRAY_DISABLE_MTP`, `STINGRAY_SPEC_SAMPLE`, `STINGRAY_BATCHED_MATVEC_TIER`, `STINGRAY_FLASH64_STRIDED_GEMM`, `STINGRAY_GEMM_PATH`, `STINGRAY_MOE_BATCHED_PREFILL`, `STINGRAY_PER_LAYER_HD_PREFILL`, `STINGRAY_PREFILL_ATTN_FLASH64`, `STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS`, `STINGRAY_PREFILL_ATTN_REGISTER_VALUES`, `STINGRAY_VULKAN_MM_PATH`, `STINGRAY_VULKAN_PREFILL_CHUNK` |

**Retirement is the point, not documentation.** The QoL plan inventories these but retires none.
A surface of 141 variables is itself the usability defect. Expect a meaningful fraction to be
`bench` or `diagnostic` leftovers that should be deleted outright rather than documented forever;
this session alone added and removed several (`STINGRAY_VABL`, `STINGRAY_WABL`,
`STINGRAY_Q4K_ABL`) as throwaway ablation switches.

**Counting note:** this counts distinct string literals, so a variable read in two projects
appears once, grouped under both. It does not detect dynamically composed names.

## 2026-08-07 reconciliation ledger

Compared with `KnownEnvironmentVariables.All`, the historical table is missing the 14 names
below. Their presence means the registry's source-drift test accepts them; it does **not** imply
support. Their source-traced class is recorded in the ownership register above; add the generated
owner/use rows during the refresh.

| Current name absent from the 2026-07 snapshot |
|---|
| `STINGRAY_BATCHED_MATVEC_TIER` |
| `STINGRAY_FLASH64_STRIDED_GEMM` |
| `STINGRAY_GEMM_PATH` |
| `STINGRAY_GEMMA4_PROBE` |
| `STINGRAY_KV_BF16_MIN_TOKENS` |
| `STINGRAY_KV_STORE` |
| `STINGRAY_MOE_BATCHED_PREFILL` |
| `STINGRAY_PER_LAYER_HD_PREFILL` |
| `STINGRAY_PREFILL_ATTN_FLASH64` |
| `STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS` |
| `STINGRAY_PREFILL_ATTN_REGISTER_VALUES` |
| `STINGRAY_VULKAN_MM_PATH` |
| `STINGRAY_VULKAN_MM_STATS` |
| `STINGRAY_VULKAN_PREFILL_CHUNK` |

Conversely, `STINGRAY_Q4K_ABL` and `STINGRAY_WABL` remain in the historical table but are no
longer in the registry, confirming they were retired experimental/ablation switches. Retain their
historical rows only until the full generated refresh replaces this snapshot.


## OpenTail.Stingray.Cli

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DSPARK_TIMING` | | |
| `STINGRAY_RAW_PROMPT` | | |
| `STINGRAY_SDCPP` | | |
| `STINGRAY_SPEC_SAMPLE` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Core, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MTP_BATCH_MAX` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cpu

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_PREFILL_Q8` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cpu, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MIN_BATCH_BLAS` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cuda, OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_SNAPKV_BUDGET` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DISABLE_MTP` | | |
| `STINGRAY_DSPARK_MIN_CONFIDENCE` | | |
| `STINGRAY_DSPARK_VERIFY_LEN` | | |
| `STINGRAY_MTP_DRAFT_N` | | |
| `STINGRAY_TRACE_MTP` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_MOE` | | |
| `STINGRAY_DSPARK_PLACE` | | |
| `STINGRAY_EXPERT_STATS` | | |
| `STINGRAY_MOE_GPU_PREFILL` | | |
| `STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS` | | |
| `STINGRAY_MOE_PIN_MODE` | | |
| `STINGRAY_MOE_PREDICT_PREFETCH` | | |
| `STINGRAY_MOE_WARMPIN` | | |
| `STINGRAY_MOE_WARMPIN_AFTER` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_PREFILL_CHUNK` | | |
| `STINGRAY_PREFILL_DEQUANT_MB` | | |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host, OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_KV_DTYPE` | | |

## OpenTail.Stingray.Core, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_TOOL_GRAMMAR` | | |

## OpenTail.Stingray.Cpu

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MATVEC_WIDE8` | | |
| `STINGRAY_TRACE_GDN_INTERNAL` | | |
| `STINGRAY_TRACE_GDN_LAYERS` | | |
| `STINGRAY_TRACE_GDN_POS` | | |

## OpenTail.Stingray.Cpu, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_THREADS` | | |

## OpenTail.Stingray.Cuda

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_ACT_SOA` | | |
| `STINGRAY_ACT_SOA_CPA` | | |
| `STINGRAY_ARGMAX_NEG_INF` | | |
| `STINGRAY_ATTN_WAVE_BUDGET_MB` | | |
| `STINGRAY_BATCH_DECODE_WS_V2` | | |
| `STINGRAY_CUDA13` | | |
| `STINGRAY_CUDA_PRECISION` | | |
| `STINGRAY_DECODE_MMQ_BM32` | | |
| `STINGRAY_GPU_ARGMAX` | | |
| `STINGRAY_Q40_DP4A` | | |
| `STINGRAY_Q6K_DECODE_MMQ` | | |
| `STINGRAY_Q80_DP4A` | | |

## OpenTail.Stingray.Cuda, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCH_DECODE_MMQ` | | |
| `STINGRAY_PREFILL_MMQ` | | |
| `STINGRAY_TRUNK_MATVEC_FAST` | | |

## OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCHED_ATTN` | | |
| `STINGRAY_BATCHED_CPU_FFN` | | |
| `STINGRAY_BATCHED_FFN` | | |
| `STINGRAY_BATCHED_GDN_SCAN` | | |
| `STINGRAY_BATCHED_PREFILL` | | |
| `STINGRAY_BATCHED_TRUNK` | | |
| `STINGRAY_BATCH_DECODE_GEMM` | | |
| `STINGRAY_BATCH_DECODE_RAGGED` | | |
| `STINGRAY_BATCH_DECODE_WS` | | |
| `STINGRAY_BYPASS_ATTN` | | |
| `STINGRAY_BYPASS_GDN` | | |
| `STINGRAY_BYPASS_MOE` | | |
| `STINGRAY_CPU_GDN` | | |
| `STINGRAY_CUDA_GRAPH` | | |
| `STINGRAY_CUDA_MATVEC_BENCH` | | |
| `STINGRAY_CUDA_PROFILE` | | |
| `STINGRAY_DECODE_CUDA_GRAPH` | | |
| `STINGRAY_DECODE_PROFILE` | | |
| `STINGRAY_DECODE_REGIONS` | | |
| `STINGRAY_DENSE_FFN_GPU_MARGIN_MB` | | |
| `STINGRAY_DISABLE_BATCH_VERIFY` | | |
| `STINGRAY_DISABLE_DUAL_Q8` | | |
| `STINGRAY_FORCE_CPU_EMBED` | | |
| `STINGRAY_GDN_CHUNKED_PREFILL` | | |
| `STINGRAY_GDN_DECODE_FAST` | | |
| `STINGRAY_GDN_PREFILL_COMPUTE` | | |
| `STINGRAY_GDN_RAW_Q8_0` | | |
| `STINGRAY_HYBRID_BATCHED_MOE` | | |
| `STINGRAY_HYBRID_PREFILL_COMPUTE` | | |
| `STINGRAY_KVARN_BATCHED_PREFILL` | | |
| `STINGRAY_MMQ_SOA` | | |
| `STINGRAY_MOE_GPU_ROUTER` | | |
| `STINGRAY_MOE_THREADS` | | |
| `STINGRAY_MTP_BATCHED_MOE_VERIFY` | | |
| `STINGRAY_MTP_MIN_ACCEPT` | | |
| `STINGRAY_MTP_PROBE_STEPS` | | |
| `STINGRAY_PLE_GPU_DEQUANT` | | |
| `STINGRAY_PREFAULT` | | |
| `STINGRAY_PREFILL_FLASH` | | |
| `STINGRAY_PREFILL_FLASH_TC` | | |
| `STINGRAY_PREFILL_FLASH_TC1` | | |
| `STINGRAY_PREFILL_GEMM` | | |
| `STINGRAY_PREFILL_PROFILE` | | |
| `STINGRAY_PREFIX_SCRATCH_TOKENS` | | |
| `STINGRAY_PREFIX_SLOTS` | | |
| `STINGRAY_PROBE_IDS` | | |
| `STINGRAY_PROBE_LOGITS` | | |
| `STINGRAY_PROBE_POS` | | |
| `STINGRAY_PROFILE_DECODE` | | |
| `STINGRAY_PROFILE_PREFILL` | | |
| `STINGRAY_Q3K_DEQUANT_GEMM` | | |
| `STINGRAY_Q3K_Q8K` | | |
| `STINGRAY_Q4KX8_CACHE_MB` | | |
| `STINGRAY_Q4K_Q8K` | | |
| `STINGRAY_Q4K_SOA` | | |
| `STINGRAY_Q6K_SOA` | | |
| `STINGRAY_Q8_0_Q8K` | | |
| `STINGRAY_SNAPKV` | | |
| `STINGRAY_SNAPKV_RECENCY` | | |
| `STINGRAY_SNAPKV_WINDOW` | | |
| `STINGRAY_SPEC_BATCH_VERIFY` | | |
| `STINGRAY_SPLIT_DECODE_GROUPED` | | |
| `STINGRAY_TRACE_DSPARK` | | |
| `STINGRAY_TRACE_LAYERS` | | |
| `STINGRAY_TRACE_NORMS` | | |
| `STINGRAY_TRACE_POS` | | |
| `STINGRAY_TRACE_ROUTERS` | | |
| `STINGRAY_TRACE_SNAPSHOT` | | |
| `STINGRAY_TRACE_VRAM` | | |
| `STINGRAY_VABL` | | |
| `STINGRAY_VULKAN_BATCHED_PREFILL` | | |
| `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL` | | |
| `STINGRAY_VULKAN_NO_BATCHED_PREFILL` | | |
| `STINGRAY_VULKAN_NO_FLASH_ATTN` | | |
| `STINGRAY_VULKAN_SPLIT_DECODE_MIN` | | |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_DSPARK_MODEL` | | |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MAX_BATCH` | | |
| `STINGRAY_MMPROJ` | | |

## OpenTail.Stingray.Engine, OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_SPLIT_DECODE` | | |
| `STINGRAY_VULKAN_SPLIT_DECODE` | | |

## OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MOE_` | | |

## OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_KV_BUDGET_MB` | | |
| `STINGRAY_MAX_CONCURRENT` | | |
| `STINGRAY_MAX_QUEUE` | | |
| `STINGRAY_MODEL` | | |
| `STINGRAY_NO_THINKING` | | |
| `STINGRAY_PREFIX_CACHE_MB` | | |
| `STINGRAY_PRESERVE_THINKING` | | |

## OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BACKEND` | | |
| `STINGRAY_N_GPU_LAYERS` | | |
| `STINGRAY_TQ` | | |
| `STINGRAY_TQ_MODE` | | |

## OpenTail.Stingray.Vulkan

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_VULKAN_DP4A` | | |
| `STINGRAY_VULKAN_UMA_FRACTION` | | |
| `STINGRAY_VULKAN_VALIDATION` | | |
