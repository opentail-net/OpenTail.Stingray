# Host configuration inventory

**Generated:** 2026-07-26 from `OpenTailStingrayServerOptions`. **41 bindable keys.**
The options type also has two deliberately non-bindable programmatic hooks
(`EngineFactory` and `OutputConstraintFactory`); they are excluded from this inventory because
JSON configuration cannot supply delegates.

Third and final part of Phase 0 deliverable 1 of `02-quality-of-life-improvements-plan.md`,
alongside `env-var-inventory.md` (a 141-row historical snapshot; the source registry now has 156 names) and
`cli-option-inventory.md` (a 94-row historical snapshot; source now has 149 declarations).

These bind from the **`OpenTail.Stingray`** configuration section — `appsettings.json`,
`appsettings.{Environment}.json`, and the git-ignored `appsettings.Local.json` — via
`services.Configure<OpenTailStingrayServerOptions>(...)`. A handful are then overwritten by
`STINGRAY_*` reads in the host's `Program.cs` for backward compatibility, which is precisely
the ad-hoc precedence §7.3 needs to replace: the override order currently lives in the order of
statements in one file rather than in a stated rule.

**Classification register (completed for this typed host surface).** The generated table below
remains useful as the source-level inventory; this register supplies its currently blank Class
column without rewriting generated descriptions. `stable` keys are supported deployment
configuration, `expert` keys materially affect performance/placement/resources, `diagnostic`
keys produce evidence, and `experimental` keys are not a release promise.

| Class | Keys |
|---|---|
| stable | `DisableThinking`, `PreserveThinking`, `ModelPath`, `Backend`, `ContextSize`, `MaxBatchSize`, `MaxQueuedRequests`, `MaxConcurrentRequests`, `KvBudgetMb`, `Sampling`, `Temperature`, `TopK`, `TopP`, `MinP`, `RepetitionPenalty`, `MaxNewTokens`, `MaxThinkingTokens` |
| expert | `ToolGrammar`, `MmprojPath`, `Architecture`, `NGpuLayers`, `TurboQuant`, `TqMode`, `KvType`, `MinBatchBlas`, `CpuThreads`, `PrefillChunkTokens`, `PrefillDequantCacheMb`, `PrefixCacheMb`, `MoeWarmPin`, `MoeWarmPinAfter`, `MoePredictPrefetch`, `CpuMoe`, `SpecType`, `SpecDraftNMax`, `SpecDraftNMin`, `SpecDraftPMin` |
| diagnostic | `ExpertStatsPath` |
| experimental | `DSparkModelPath`, `DSparkPlace`, `GpuMoePrefill` |

**This surface is small and typed, unlike the other two.** It is already close to what §7.3 wants a
saved profile to look like, so it is the natural seed for the profile schema rather than a fourth
parallel mechanism.

| Key | Type | Class | Description |
|---|---|---|---|
| `DisableThinking` | `bool` | | Globally disable reasoning for every request (server-side --no-thinking / STINGRAY_NO_THINKING). For agentic clients that never send the per-request opt-out. |
| `PreserveThinking` | `bool` | | Globally keep prior assistant turns' reasoning in the chat-template history instead of stripping it (server-side default for the per-request preserve_thinking flag / STINGRAY... |
| `ToolGrammar` | `bool` | | Enable schema/grammar-constrained decoding for tool-call arguments (issue #374). When on and a tool-active request is served by a family with constraint support (Gemma 4, Qwen a... |
| `ModelPath` | `string?` | | Path to the GGUF model file. Required unless  is supplied. Relative paths resolve against the current directory, the entry-assembly directory, and a handful of parent directories. |
| `MmprojPath` | `string?` | | Optional path to a multimodal projector GGUF (mmproj-*.gguf) enabling image input (issue #253). Only Gemma 4 gemma4uv projectors are supported today, and only on a backend whose... |
| `DSparkModelPath` | `string?` | | Optional path to a DSpark draft head — model.safetensors, or its directory with config.json alongside (docs/07-dspark-plan.md, PR #413 Phase 6). Set via this property or the OPENTA... |
| `DSparkPlace` | `string?` | | Where the DSpark draft head runs: auto (default; the placement planner decides from VRAM/RAM headroom), gpu, cpu, or off. Set via this property or the STINGRAY_DSPARK_PLACE e... |
| `Architecture` | `string` | | Architecture hint used by  as a fallback when the model's GGUF metadata is missing general.architecture and no Jinja template is bundled. Defaults to "qwen2" (ChatML). |
| `Backend` | `ServerBackend` | | GPU backend selection. Mirrors the CLI's --backend. Auto picks CUDA when available, falls through to Vulkan, then CPU. Only consulted when is non-zero. |
| `NGpuLayers` | `int` | | Number of model layers to offload to the GPU. Mirrors the CLI's --n-gpu-layers (-g): 0 = CPU only, -1 = let TierPlanner size the split from available VRAM, N = explicit. Default 0. |
| `ContextSize` | `int` | | Context size / max sequence length. 0 = use the model's GGUF default. Mirrors --ctx-size. |
| `TurboQuant` | `bool` | | Enable TurboQuant KV-cache compression. Mirrors --tq. Requires head dimension ∈ {128, 256}; the loader falls back to non-TQ otherwise. The quantizer is selected by . |
| `TqMode` | `string` | | TurboQuant quantizer for . Mirrors the CLI's --tq-mode: "auto" (default) picks KVarN (4-bit K / 2-bit V, issue #180) wherever the resolved forward pass supports it and falls bac... |
| `KvType` | `string?` | | KV-cache element type for the CUDA dense path. Mirrors the CLI's --kv-type / STINGRAY_KV_DTYPE: "fp32" (default), "bf16" (half the KV VRAM → ~2× context), or "q8_0" (block-qu... |
| `MinBatchBlas` | `int` | | Minimum batch size before promotes the inner loop to OpenBLAS SGEMM. Mirrors --min-batch-blas / STINGRAY_MIN_BATCH_BLAS. 0 = leave the engine default. |
| `CpuThreads` | `int` | | Worker threads for CPU SIMD matrix-vector kernels. 0 = use (or STINGRAY_CPU_THREADS when set). Set this below the logical processor count when the server shares CPU or memory... |
| `MaxBatchSize` | `int` | | Maximum concurrent decode sequences. Values &gt; 1 select ; ≤ 1 selects . |
| `MaxQueuedRequests` | `int` | | Maximum number of requests allowed to wait behind the active inference batch. Together with , this bounds the server's generation work-in-flight at MaxBatchSize + MaxQueuedReque... |
| `MaxConcurrentRequests` | `int?` | | Maximum number of generation requests allowed in flight at once before the server fast-rejects with HTTP 429 (issue #109). null (default) keeps the legacy behaviour: the single-... |
| `PrefillChunkTokens` | `int` | | Prompt tokens prefilled per batcher iteration under continuous batching (issue #183 Gap 1). Active sequences advance one decode step between chunks, so a long inbound prompt no ... |
| `PrefillDequantCacheMb` | `long?` | | Dequant-once BLAS weight-cache budget in MiB (issue #189). The CPU batched-prefill path re-dequantizes each projection weight to F32 on every call, so small prefill chunks re-pa... |
| `KvBudgetMb` | `long` | | KV-cache memory budget in MiB gating request admission under continuous batching (issue #183 Gap 3). Each admitted sequence reserves promptTokens + max_tokens worth of KV; when ... |
| `PrefixCacheMb` | `long` | | Retained KV-prefix cache budget in MiB for CPU continuous batching. Prefixes are exact canonical token prefixes rounded down to 16-token pages, shared copy-on-write with new req... |
| `MoeWarmPin` | `int?` | | Pin the top-N hottest experts per layer after warmup. null = disabled (frequency-aware SLRU eviction is sufficient on its own). Mirrors --moe-warmpin / STINGRAY_MOE_WARMPIN. |
| `MoeWarmPinAfter` | `long` | | Number of expert accesses to observe before warm-pin selects the hot set. Only meaningful when  is set. Mirrors --moe-warmpin-after. |
| `MoePredictPrefetch` | `bool` | | Next-layer predictive expert prefetch on the Vulkan path. Mirrors --no-moe-predict-prefetch (defaulting to true here — set false to disable, equivalent to STINGRAY_MOE_PREDIC... |
| `ExpertStatsPath` | `string?` | | Path to write GPU expert-cache (SLRU) hit-rate stats to on process exit. Mirrors --expert-stats / STINGRAY_EXPERT_STATS. |
| `CpuMoe` | `bool?` | | Force all routed MoE experts onto the CPU side — the engine's all-or-nothing STINGRAY_CPU_MOE override, read by  and at construction. Server analogue of the CLI's --cpu-moe (... |
| `GpuMoePrefill` | `bool?` | | GPU op-offload of the CPU-MoE routed prefill — the engine's STINGRAY_MOE_GPU_PREFILL gate, read by  at construction. Uploads each used expert's host-resident weights to the G... |
| `SpecType` | `ServerSpecType` | | Speculative decoding mode. Mirrors --spec-type: Auto enables MTP when supported; None forces single-token; Mtp requires an MTP head. Applied as a per-request default when the re... |
| `SpecDraftNMax` | `int` | | Max draft tokens per speculative step. Mirrors --spec-draft-n-max. |
| `SpecDraftNMin` | `int` | | Min draft tokens per speculative step. Mirrors --spec-draft-n-min. |
| `SpecDraftPMin` | `float` | | Minimum draft probability for probabilistic accept under MTP verification. Mirrors --spec-draft-p-min. 1.0 = strict argmax-match (byte-identical to no-MTP). |
| `Sampling` | `SamplingDefaults` | | Sampling parameters applied when the inbound request omits them. The HTTP request fields (e.g. OpenAI temperature, Anthropic top_p) still take precedence — these are only the fa... |
| `Temperature` | `float` | | Temperature. 0 = greedy. Mirrors --temp. |
| `TopK` | `int` | | Top-k truncation. 0 = disabled. Mirrors --top-k. |
| `TopP` | `float` | | Top-p (nucleus) cutoff. 1.0 = disabled. Mirrors --top-p. |
| `MinP` | `float` | | Min-p cutoff. 0 = disabled. Mirrors --min-p. |
| `RepetitionPenalty` | `float` | | Repetition penalty. 1.0 = disabled. Mirrors --rep-penalty. |
| `MaxNewTokens` | `int` | | Cap on generated tokens when the request doesn't specify max_tokens. Mirrors --n-predict. |
| `MaxThinkingTokens` | `int` | | Maximum reasoning tokens before the engine forces &lt;/think&gt;. 0 = unlimited. Mirrors --max-thinking-tokens. |
