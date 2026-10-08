# Forward-pass selection and refusal matrix

> **Historical snapshot (pre-`5c591e6`).** Selection is now plan-driven (`ExecutionPlanner` -> `ExecutionPlan` -> `ArchitectureDescriptor.ConstructForwardPass`). Current plan: [2026-10-08-architecture-semantics-admission-plan.md](2026-10-08-architecture-semantics-admission-plan.md).

Characterization of the current CLI and server loader selection paths. This document records existing behavior; it does not propose or make runtime changes. Line numbers are source lines re-read on 2026-10-05 and should be refreshed after source edits.

**Classification:** **architecture-driven** means architecture ID/family decides; **shape-driven** means hyperparameters or tensor presence decides; **hardware/flag-driven** means requested backend, layer count, device availability, planner result, TurboQuant, or draft options decide. Many rows combine categories; each lists all relevant categories. The matrix covers model forward-pass class selection and explicit compatibility refusals/fallbacks, including speculative decoders that select a target pass. It omits unrelated validation (prompt, tokenizer, model path, image/mmproj) except package capability gates that directly refuse forward-pass modes.

## CLI: `src/OpenTail.Stingray.Cli/RunCommand.cs`

| Lines | Decision point | Category | Result / refusal as implemented |
|---|---|---|---|
| 795–812 | SafeTensors package capability report is unsupported | shape-driven | Refuses package; prints each rejection and recommends GGUF. |
| 815–822 | `effNGpuLayers != 0` for a package | hardware/flag-driven | Refuses GPU offload: “GPU offload … is not yet supported for SafeTensors packages”; advises CPU or GGUF. |
| 823–828 | `--tq` with package | hardware/flag-driven | Refuses TurboQuant for SafeTensors. |
| 829–834 | `--draft-model` or `--draft-lookup` with package | hardware/flag-driven | Refuses speculative decoding for SafeTensors. |
| 835–839 | `--dspark-model` with package | hardware/flag-driven | Refuses DSpark for SafeTensors. |
| 868–878 | Accepted SafeTensors package | hardware/flag-driven (package format) | Constructs CPU `ForwardPass`; package path never reaches GGUF backend selection. |
| 927–935 | Speculation requested while a token constraint is active | hardware/flag-driven | Warns and disables speculation, then normal constrained generation. |
| 939–946 | CUDA/Vulkan target pass supports batch verification | shape-driven + hardware/flag-driven | Enables the corresponding GPU speculative target, but draft-mode restrictions described by comments and warning differ from the actual draft-model branch at 1043–1054. |
| 955–959 | Both draft-model and lookup requested | hardware/flag-driven | Hard refusal: mutually exclusive. |
| 960–963 | GPU requested but target is not a supported spec pass | hardware/flag-driven + shape-driven | Warns and falls back to normal decoding. The warning text names Vulkan `--draft-lookup`, but the actual Vulkan branch below also constructs a Vulkan draft-model pass when `vulkanSpecTarget` is true. |
| 964–967 | Sampled decoding with draft lookup | hardware/flag-driven | Lookup is greedy-only; warns and falls back. |
| 968–975 | Sampled spec disabled by env or sampling has penalties/bias | hardware/flag-driven | Warns/notes and falls back to normal sampled generation. |
| 976–1004 | Eligible prompt-lookup draft | hardware/flag-driven | Runs target `gpuFwd` when GPU speculative target is eligible, otherwise CPU `fwd`. |
| 1005–1009 | Draft model file missing | hardware/flag-driven | Hard refusal: `Draft model not found: {path}`. |
| 1017–1064 | Draft-model target path | hardware/flag-driven | Creates CUDA draft `CudaForwardPass`, Vulkan draft `GpuForwardPass`, or CPU draft `ForwardPass` according to the selected target. The Vulkan `else if (vulkanSpecTarget)` at 1043–1054 means model-draft is actually accepted for supported Vulkan targets too, despite the comments at 943–944 and warning text at 962 characterizing it as lookup-only. |
| 1101–1105 | DSpark selected without a DSpark model path | hardware/flag-driven | Hard refusal: `--spec-type dspark requires --dspark-model`. |
| 1106–1110 | DSpark plus draft-model/lookup | hardware/flag-driven | Hard refusal: mutually exclusive. |
| 1111–1117 | DSpark plus explicit MTP spec type | hardware/flag-driven | Hard refusal: conflicting spec types. |
| 1118–1125 | DSpark minimum confidence > 1 | hardware/flag-driven | Hard refusal; threshold must be in [0, 1]. |
| 1132–1140 | DSpark target discovery | hardware/flag-driven + shape-driven | Eligible only CPU `fwd` with `nGpuLayers == 0`, or dense full-CUDA `CudaForwardPass`; otherwise target remains null. |
| 1142–1158 | DSpark target/mode compatibility | hardware/flag-driven + shape-driven | Disables DSpark with warning and falls back to normal generation for explicit `--spec-type none`, constraints, tools, sampled temperature, unsupported target, or target lacking hidden taps (SnapKV, TurboQuant, MoE, Gemma 4). |
| 1160–1163 | DSpark requested without a single prompt | hardware/flag-driven | Interactive mode warns and falls back. |
| 1281–1293 | Chat-template override contains no Jinja syntax | hardware/flag-driven | Refuses non-Jinja named shortcut instead of approximating. |
| 291–315 | CLI option validation affecting generation limits, KV layout, or device placement | hardware/flag-driven | Refuses negative `-n`/EOS sentinel, mismatched K/V dtypes, tensor split, split mode, main-GPU, mlock, no-mmap, NUMA, batch-size, or ubatch-size. |
| 1368–1401 | Resolve requested `--backend` when GPU layers requested; CUDA availability and TurboQuant codec/head-dim constrain auto or CUDA | hardware/flag-driven + shape-driven | Selects CUDA/Vulkan; unknown backend is a hard refusal. Lloyd-Max CUDA with head dim outside {128,256} falls back to Vulkan. |
| 1404–1412 | CUDA partial path lacks a required operation (`PartialOffloadUnsupportedReason`) | shape-driven + hardware/flag-driven | Warns and sets GPU layer count to zero, selecting CPU. |
| 1414–1454 | Zero GPU layers | hardware/flag-driven + architecture-driven + shape-driven | Existing standalone architecture pass wins (e.g. gpt-oss/RWKV/DeepSeek2 GPU object); else hybrid-GDN CPU bridge; else generic CPU `ForwardPass` (enables TurboQuant when requested). |
| 1456–1489 | CUDA selected and `hp.IsHybridSsm` | shape-driven + hardware/flag-driven | Selects `CudaHybridGdnForwardPass`; layer types determine its internal placement. `hp.IsMoE` affects MoE routing/placement. |
| 1493–1555 | CUDA dense path; auto layer planner, Gemma4 KV-share boundary, and MoE expert-cache budget | shape-driven + hardware/flag-driven | Planner/explicit count chooses CPU fallback (`cudaGpuLayers == 0`), `CudaHybridForwardPass` (partial or auto-MoE-cache overflow), or full `CudaForwardPass`. |
| 1555–1592 | Partial CUDA placement with KVarN TurboQuant | hardware/flag-driven + shape-driven | Auto mode downgrades to Lloyd-Max with warning for supported head dims; if Lloyd-Max has no codebook for the head dim, errors. Explicit KVarN requires full CUDA offload; otherwise errors. |
| 1612–1628 | CUDA planner places zero layers | hardware/flag-driven | Disposes CUDA backend and uses CPU `ForwardPass` (with TurboQuant if enabled). |
| 1632–1643 | CUDA dense path with one or more GPU layers, not hybrid GDN | hardware/flag-driven + shape-driven | Selects `CudaForwardPass`. |
| 1655–1692 | Vulkan selected and `hp.IsHybridSsm` | shape-driven + hardware/flag-driven | Selects `VulkanHybridGdnForwardPass`; layer types and MoE shape affect internal placement. |
| 1695–1717 | Vulkan `-g -1` planner returns zero layers | hardware/flag-driven | CPU `ForwardPass`, with supported Lloyd-Max TurboQuant if requested. |
| 1719–1732 | Auto Vulkan split-only large MoE exceeds size/split cap conditions | shape-driven + hardware/flag-driven | Caps automatic split to at most four layers when model is at least 30 GiB, MoE, and partial-offload gap exists; explicit `-g N` is not clamped by this special guard. |
| 1735–1749 | Vulkan layer count reaches all model layers | hardware/flag-driven | Selects `GpuForwardPass` (full Vulkan). |
| 1751–1765 | Partial Vulkan and either per-layer head dimension or unsupported hybrid partial op | shape-driven + hardware/flag-driven | Selects `VulkanLayerSplitForwardPass`, capped by `MaxGpuLayers`. |
| 1767–1779 | Other partial Vulkan placement | hardware/flag-driven | TierPlanner pins requested/auto count and selects `HybridForwardPass`. |
| 1878–1891 | Architecture profile unsupported and `--allow-unverified-arch` absent | architecture-driven + hardware/flag-driven | `ModelCompatibility.ValidateForTextGeneration` refuses; explicit flag instead warns and proceeds. |
| 1974–1978 | Hybrid GDN plus TurboQuant | shape-driven + hardware/flag-driven | Hard refusal: TurboQuant unsupported for hybrid GDN (no KV cache on GDN layers). |
| 1979–1983 | Hybrid GDN plus draft-model/lookup | shape-driven + hardware/flag-driven | Hard refusal: speculative decoding cannot rewind destructive GDN state. |
| 1992–2005 | DeepSeek2 MLA: `KvLoraRank`, KV-B tensor present, Q-A tensor absent, GPU request, not explicitly CUDA-only/partial/TurboQuant/speculative | shape-driven + hardware/flag-driven | Selects full-offload `DeepSeek2GpuForwardPass` on Vulkan and sets effective GPU count to zero to bypass generic selection. |
| 2008–2014 | Generic GPU unsupported feature (`UnsupportedReason`) | shape-driven + hardware/flag-driven | Warns and falls back to CPU. |
| 2016–2027 | RWKV architecture ID | architecture-driven + hardware/flag-driven | Rejects TurboQuant or standard speculation; warns for GPU request, forces CPU, creates `RwkvForwardPassBase.Create`. |
| 2028–2055 | `gpt-oss` architecture ID | architecture-driven + hardware/flag-driven | Rejects TurboQuant/speculation. CUDA-only or partial offload is redirected to CPU; remaining nonzero full Vulkan request selects `GptOssGpuForwardPass`; otherwise CPU `GptOssForwardPass`. |
| 2057–2061 | `hp.IsHybridSsm` after architecture-specific branches, effective GPU layers zero | shape-driven + hardware/flag-driven | Selects CPU `HybridGdnForwardPass`. |
| 2062–2073 | Non-hybrid and no architecture GPU pass | shape-driven + hardware/flag-driven | Selects generic CPU `ForwardPass`; dequant cache is disabled when GPU layers remain. |
| 2087–2101 | Parse TurboQuant mode | hardware/flag-driven | Unknown mode is a hard refusal. |
| 2103–2124 | Auto TurboQuant codec support matrix | shape-driven + hardware/flag-driven | Selects KVarN where supported, otherwise warns and resolves Lloyd-Max. |
| 2126–2158 | Explicit/selected KVarN compatibility | shape-driven + hardware/flag-driven | Rejects missing `--tq`, SnapKV composition, Vulkan, unavailable CUDA, MoE on CUDA, or GPU KVarN without CUDA. |
| 2160–2185 | TurboQuant head dimension | shape-driven + hardware/flag-driven | Rejects KVarN outside pow2 [8,1024], CUDA KVarN above 256, or Lloyd-Max outside {128,256}. |

## Server: `src/OpenTail.Stingray.Server/InferenceEngineLoader.cs`

| Lines | Decision point | Category | Result / refusal as implemented |
|---|---|---|---|
| 13–32 | `LoadFromPlan` backend name | hardware/flag-driven | Maps `vulkan` and `cuda`; any other value maps to CPU. |
| 73–90 | SafeTensors sessions enabled / package capability | hardware/flag-driven + shape-driven | Sessions for packages hard-fail as CPU-dense GGUF-only; unsupported package profile hard-fails with inspector rejections. |
| 111–135 | Accepted SafeTensors package | hardware/flag-driven (package format) | Always builds CPU `ForwardPass` and `ContinuousBatchingEngine`; no GPU/TQ/draft selection here. |
| 161–167 | GGUF compatibility validation and architecture ID | architecture-driven | Unsupported architecture/tensor-format profile throws before selecting backend. |
| 189–194 | TurboQuant with hybrid GDN | shape-driven + hardware/flag-driven | Hard refusal with same hybrid-GDN incompatibility text as CLI. |
| 195–235 | TurboQuant mode/head-dimension parsing and validation | hardware/flag-driven + shape-driven | Unknown mode, inactive explicit KVarN, invalid KVarN dimensions or invalid explicit Lloyd-Max dimensions hard-fail. |
| 272–305 | Configured image input | architecture-driven + shape-driven + hardware/flag-driven | Requires Gemma4 architecture and embedding-capable pass (CPU or full CUDA Gemma4); refuses batching with images. |
| 308–364 | Continuous batching / sessions / DSpark combination | shape-driven + hardware/flag-driven | Sessions require supported CPU-dense GGUF pass; batching is only created for `IBatchedForwardPass`; DSpark with active supported continuous batching is refused. Otherwise constructs single-user `InferenceEngine` and optionally attaches DSpark. |
| 519–523, 525–553 | Per-path TurboQuant codec resolution | shape-driven + hardware/flag-driven | Explicit KVarN on blocked path throws; auto warns/falls back to Lloyd-Max; unsupported Lloyd-Max head dimension throws. |
| 559–569 | DeepSeek2 MLA: positive KV LoRA rank, KV-B present, Q-A absent, non-CPU, no TurboQuant, backend Auto/Vulkan, full layer request | shape-driven + hardware/flag-driven | Selects full Vulkan `DeepSeek2GpuForwardPass`. |
| 571–575 | Generic GPU unsupported feature | shape-driven + hardware/flag-driven | Logs and forces CPU (`nGpuLayers=0`). |
| 577–591 | Resolve auto backend / explicit CPU / no GPU layers | hardware/flag-driven | Auto GPU chooses CUDA when eligible and available, otherwise Vulkan; zero layers or CPU selects CPU. |
| 594–601 | CUDA partial-offload feature gap | shape-driven + hardware/flag-driven | Logs reason and forces CPU. (Vulkan partial placement is handled by split selection below.) |
| 603–613 | RWKV architecture ID | architecture-driven + hardware/flag-driven | Rejects TurboQuant; logs GPU fallback; creates `RwkvForwardPassBase.Create` on CPU. |
| 615–635 | `gpt-oss` architecture ID | architecture-driven + hardware/flag-driven | Rejects TurboQuant; full Vulkan only selects `GptOssGpuForwardPass`; CUDA/partial requests log and use CPU `GptOssForwardPass`. |
| 637–664 | CPU backend | shape-driven + hardware/flag-driven | `hp.IsHybridSsm` selects CPU `HybridGdnForwardPass`, else generic `ForwardPass`; batch eligibility additionally requires no MoE, TurboQuant, layer head dims, attention output gate, or input embedding RMS norm. |
| 666–672 | GPU shared CPU trunk setup | shape-driven + hardware/flag-driven | Builds CPU dense baseline unless hybrid GDN; it is an auxiliary pass for GPU selection, not necessarily the returned pass. |
| 674–692 | CUDA + hybrid GDN | shape-driven + hardware/flag-driven | Selects `CudaHybridGdnForwardPass`. |
| 695–710 | CUDA dense, TierPlanner yields zero layers | hardware/flag-driven | Returns CPU dense pass, TurboQuant enabled if requested. |
| 712–742 | CUDA dense, all layers fit | hardware/flag-driven + shape-driven | Selects `CudaForwardPass`; `SupportsContinuousBatching` and MoE/TQ-related runtime capability govern returned batching flag. |
| 744–757 | CUDA dense, partial planner result | hardware/flag-driven + shape-driven | Resolves TurboQuant KVarN partial-path warning/error, then selects `CudaHybridForwardPass`. |
| 760–780 | Vulkan + hybrid GDN | shape-driven + hardware/flag-driven | Selects `VulkanHybridGdnForwardPass`. |
| 782–791 | Vulkan dense, planner yields zero layers | hardware/flag-driven | Returns CPU dense pass, TurboQuant enabled if requested. |
| 793–801 | Vulkan dense, all layers fit | hardware/flag-driven | Resolves Vulkan TurboQuant codec then selects full `GpuForwardPass`. |
| 803–812 | Vulkan partial, per-layer head dimensions or unsupported hybrid operation, and TurboQuant off | shape-driven + hardware/flag-driven | Selects `VulkanLayerSplitForwardPass` with GPU-layer cap. If TurboQuant is on this branch does not apply. |
| 814–820 | Other Vulkan partial path | hardware/flag-driven + shape-driven | Resolves Vulkan TurboQuant codec and selects `HybridForwardPass`. |
| 392–426 | Configured DSpark model/config, dimensions, and target tap support | shape-driven + hardware/flag-driven | Missing files, mismatch, or unsupported target throws. Tap-target refusal text names CPU/NGpuLayers=0 or full CUDA/-1 and excludes MoE/Gemma-4/TurboQuant/SnapKV. |
| 435–461 | DSpark placement | hardware/flag-driven | GPU placement without target CUDA is replanned without VRAM; Off placement throws rather than silently disabling configured DSpark. |
| 463–467 | DSpark placement resolved | hardware/flag-driven | Creates CUDA draft model for GPU placement, otherwise CPU draft model. |

## CLI vs. server divergences

These are observable selection/refusal differences for equivalent model/request conditions, not proposed corrections. Following the Step 6 S5 migration (commit `88bbc653`), forward-pass selection and compatibility refusals are unified through `ForwardPassSelection.Select`. Note that **rwkv** and **hybrid-GDN** have no local GGUF models available and are covered only by selector tests (`ForwardPassSelectionTests`).

| Point | CLI | Server | Status after Step 6 migration |
|---|---|---|---|
| DeepSeek2 MLA with speculative decoding requested | At 1994–2005, draft-model or lookup prevents the special Vulkan pass; generic path then processes it. | At 559–569, there is no draft flag and MLA branch can select Vulkan pass. | **Preserved asymmetry**: CLI-only draft flags (`HasDraftModel`, `DraftLookup`) prevent MLA Vulkan pass in the selector when `Frontend == Cli`; server has no standard draft options. |
| DeepSeek2 MLA with explicit CUDA backend | At 1997–2005, explicit CUDA-only blocks Vulkan MLA selection; the generic unsupported-feature logic can fall back to CPU. | At 559–563 backend Auto/Vulkan is required, so CUDA bypasses MLA; generic unsupported-feature logic can fall back to CPU. | **Unified**: Selector requires `mlaRequestedBackend` (`Backend != Cuda` on CLI, `Auto` or `Vulkan` on server), falling back to generic CPU/GPU path. |
| DeepSeek2 MLA with partial layer count | CLI `partial` blocks special MLA and falls through generic GPU checks. | Full layer count required, so MLA special case is skipped. | **Unified**: Selector requires full offload (`!partialRequest`), falling back to generic path. |
| RWKV with draft-model or prompt lookup | 2016–2022 hard-refuses TurboQuant and either draft mode. | 603–613 only rejects TurboQuant; loader has no standard draft-decoding option and returns RWKV CPU pass. | **Unified**: Selector handles `ForwardPassFamily.Rwkv`, refusing TurboQuant for both frontends and speculative options when `Frontend == Cli`. Covered only by selector tests (no local GGUF). |
| Hybrid GDN with standard draft options | 1979–1983 hard-refuses. | `BuildForwardPass` has no standard draft option; it may load hybrid GDN. | **Unified**: Selector refuses speculative decoding for hybrid GDN when `Frontend == Cli`. Covered only by selector tests (no local GGUF). |
| gpt-oss with standard speculative options | 2030–2034 hard-refuses drafts. | 615–635 has no draft gate and can load CPU or Vulkan gpt-oss. | **Unified**: Selector refuses drafts on gpt-oss when `Frontend == Cli`. |
| gpt-oss CUDA-only/full-layer request | 2037–2040 treats CUDA-only (`wantsCudaOnly`) as unsupported for gpt-oss and forces CPU, including an explicit CUDA full-layer request. | 622 requires Vulkan for GPU; CUDA returns CPU. | **Unified**: Selector returns `GptOssCpu` for CUDA or partial offload. |
| gpt-oss Auto GPU, all layers | CLI auto backend can choose CUDA when available; then `wantsCudaOnly` is false and nonzero full offload reaches `GptOssGpuForwardPass` on Vulkan construction path; server Auto resolves to CUDA when available, and CUDA is not eligible in gpt-oss branch, so returns CPU. | Auto resolution selects CUDA when available; gpt-oss full-GPU gate requires Vulkan, hence CPU. | **Unified**: Selector returns `GptOssVulkan` for full GPU offload on Vulkan/Auto (with CLI branch explicitly preserving Vulkan selection). |
| Hybrid GDN with TurboQuant | CLI refuses before backend selection at 1974–1978. | Server refuses before `BuildForwardPass` at 192–194. | **Unified**: Selector returns identical refusal `"TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers)."`. Covered only by selector tests (no local GGUF). |
| SafeTensors backend features | CLI explicitly refuses GPU, TurboQuant, standard drafts, DSpark, and images at 815–844; then CPU `ForwardPass`. | Server accepts packages into CPU `ForwardPass` at 111–135 but does not expose the same per-feature CLI flags here; session mode is explicitly refused at 77–80. | **Unified**: SafeTensors capability and feature gates are verified via `ForwardPassSelection.Select(IsSafeTensors: true)`. |
| Package inference details | CLI accepts package features only after explicit capability gates, then uses CPU `ForwardPass` at 795–878. | Server additionally rejects SafeTensors sessions at 77–80 and starts `ContinuousBatchingEngine` for supported packages at 111–135. | **Preserved frontend distinction**: Server wraps packages in `ContinuousBatchingEngine`; CLI uses single-session execution. |
| Image input | CLI validates mmproj and prompt before the forward path at 1931–1967, then image execution uses active pass at 2232–2239. | Server image path at 272–305 restricts text architecture to Gemma4 and requires embedding-input support; refuses images with batching. | **Preserved frontend distinction**: Server enforces Gemma4 embedding guard in loader; CLI routes through `UnifiedVisionPipeline`. |
| DSpark Off or unsupported configuration | CLI warns and continues ordinary generation for Off/unsupported target or interactive use at 1142–1189. | Server throws for unsupported hidden taps or placement Off at 422–426, 456–461. | **Preserved frontend distinction**: Server treats missing/Off DSpark as deployment error; CLI treats DSpark as optional feature and falls back. |
| DSpark context-window overflow | CLI checks prompt + block fit at 2547–2555 and returns error code 1 for that run. | Server attachment validates head/target/tap support and placement at 415–461; no corresponding prompt exists at startup because the request prompt is not yet known. | **Preserved frontend distinction**: Request-time prompt size check vs startup model-load check. |
| Auto CUDA layer planning / Gemma4 boundary / expert cache | CLI has explicit Gemma4 KV-source boundary clamp and special MoE expert-cache auto-hybrid path (1493–1548). | CUDA loader uses TierPlanner and Gemma4 clamp (702–704), but no equivalent `MoeRoutedExpertBytes > ExpertCacheBudgetBytes` full-offload-to-hybrid override in this selection block (712–757). | **Preserved frontend distinction**: Hardware placement calculation (`TierPlanner`, layer counts) remains in the frontend. |
| Auto Vulkan large split-only MoE safeguard | CLI caps certain >=30 GiB models to 4 layers under auto placement (1719–1732). | Server selects split pass based on shape and requested planner count only (803–812); no model-size auto cap here. | **Unified in selector / Preserved frontend placement**: Selector models `LayerSplitOnlyLargeMoe`; hardware layer clamping remains frontend placement. |
| Vulkan partial split with TurboQuant | CLI split decision at 1751–1760 has no `!TurboQuant` guard; a selected split can reach a constructor incompatible with requested TurboQuant. | Server split branch explicitly requires `!turboQuant` at 803–805; otherwise continues to generic hybrid path/codec resolution. | **Preserved divergence**: Selector checks `Frontend == Cli || !TurboQuant` for layer split. |
| TurboQuant explicit/auto validation | CLI checks many KVarN gates after model/GPU setup at 2087–2185, including SnapKV, CUDA availability, MoE, and GPU head dim. | Server validates mode/head dim before pass construction at 189–235 and resolves per path in 525–553; explicit KVarN blocked-path throws at 538–539. | **Unified**: Selector provides sliced validation via request flags (`ValidateTurboQuantModeOnly`, `ValidateTurboQuantHeadDimOnly`, `ValidateKVarNOnly`), keeping exact CLI order and server semantics. |
| Generic unsupported GPU feature | CLI at 2008–2014 and CUDA partial gate at 1404–1412 force CPU; Vulkan can use layer split for `PartialOffloadUnsupportedReason` at 1751–1765. | Server at 571–575 uses `UnsupportedReason`; CUDA partial gate forces CPU at 594–601, while Vulkan split is selected at 803–812. | **Unified**: Handled by selector via `UnsupportedGpuPath`, `UnsupportedPartialCudaPath`, `UnsupportedPartialVulkanPath`. |
| Vulkan speculative decoding diagnostics | CLI at 939–946 and 1043–1054 actually accepts model-draft verification on a supported full-offload `GpuForwardPass`; warning/comment at 943–944 and warning at 962 describe Vulkan as prompt-lookup-only. | No standard server draft decoder option in `BuildForwardPass`. | **Preserved CLI diagnostic nuance**: Speculative decoder setup remains in CLI frontend. |

### Selector Request Flags

The selector (`ForwardPassSelection.Select`) exposes specific request flags that allow frontends (notably the CLI) to evaluate slices of rules at precise points in execution. This design preserves the exact ordering of checks and error messages without duplicating validation logic:

- `ValidateTurboQuantModeOnly`: Asks the selector to validate solely the syntax and string name of `--tq-mode` (accepting `auto`, `lloydmax`/`lloyd-max`, `kvarn`, or empty), returning `CpuDense` on success or refusing with an unknown mode error (`Unknown --tq-mode value '...'`), without performing model-shape, architecture, or hardware checks.
- `ValidateTurboQuantHeadDimOnly`: Isolates head dimension validation for TurboQuant (requiring LloydMax head dimensions in {128, 256}, or KVarN power of 2 in [8, 1024] / CUDA cap ≤ 256). Used by the CLI after quantizer resolution.
- `SkipFamilyRefusals`: Bypasses recurrent/family refusals (e.g. RWKV or gpt-oss rejecting TurboQuant or speculative decoding) when the CLI evaluates a targeted slice of rules (such as validating TurboQuant head dimensions or KVarN preconditions) without failing prematurely on the family type.
- `SkipTurboQuantShapeValidation`: Skips general TurboQuant shape and device validation during the earlier architecture selection phase (e.g. when checking DeepSeek2, RWKV, or gpt-oss compatibility), allowing the CLI to defer TurboQuant shape errors until its dedicated TurboQuant validation stage.
- `UnsupportedBackendName`: Allows the CLI to pass an invalid `--backend` string to the selector so the selector produces the canonical error message (`Unknown --backend value '...'. Expected one of: auto, vulkan, cuda.`) at backend validation time.

## Baselines

Recorded after `dotnet build src/OpenTail.Stingray.Cli -c Release` succeeded on the working tree (0 warnings; build completed in 35.2 s). Baselines are greedy and output identity only; generation stops at EOS if reached before `-n`. `--verbose-prompt` prints the rendered prompt token IDs and each selected token ID as `[DBG] tok=… next=…`. The fixed user prompt is `Write a short story: Once upon a time`; the GGUF chat template adds its stock system/user/assistant wrapper. The input file is `models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf` (105,454,432 bytes).

### SmolLM2-135M-Instruct Q4_K_M — CPU

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

Exact command:

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Write a short story: Once upon a time" -n 24 --temp 0 -g 0 --seed 1 --verbose-prompt --no-display-prompt`

Rendered prompt IDs (39): `1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 19161, 253, 1890, 1977, 42, 4027, 1980, 253, 655, 2, 198, 1, 520, 9531, 198`.

Generated text (24 tokens): `Once upon a time, there lived a young girl named Lily. She was always curious and loved to explore her surroundings.`

Generated token IDs: `6403, 1980, 253, 655, 28, 665, 4161, 253, 1805, 8180, 3365, 14176, 30, 2306, 436, 1811, 7436, 284, 5732, 288, 2217, 874, 10882, 30`.

### SmolLM2-135M-Instruct Q4_K_M — full Vulkan

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

Exact command:

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Write a short story: Once upon a time" -n 24 --temp 0 -g -1 --backend vulkan --seed 1 --verbose-prompt --no-display-prompt`

The selected device was `Vulkan GPU (AMD Radeon(TM) Graphics)`; backend reported all 30 layers on GPU. Prompt IDs matched the CPU run (same 39 IDs listed above).

Generated text (24 tokens): `Once upon a time, there was a young girl named Lily who lived in a small village surrounded by rolling hills and lush`

Generated token IDs: `6403, 1980, 253, 655, 28, 665, 436, 253, 1805, 8180, 3365, 14176, 617, 4161, 281, 253, 1165, 6560, 10000, 411, 13549, 12610, 284, 19533`.

The CPU and Vulkan runs are **not token-identical** (first difference at generated token index 6: CPU 4161 `lived`, Vulkan 436 `was`); each vector is a frozen baseline for its execution path, not a parity claim.

### Optional architecture models

A local `models/_models/gpt-oss-20b-MXFP4.gguf` is present (12,109,566,624 bytes). A local `models/_models/DeepSeek-V2-Lite-Chat.Q2_K.gguf` is also present (6,430,464,768 bytes). Both were run before the CLI migration, greedy, `-n 16`, with the architecture-specific CPU pass and full Vulkan path respectively. Both Vulkan runs used `Vulkan GPU (AMD Radeon(TM) Graphics)` and all layers on GPU. There is no local RWKV GGUF.

#### gpt-oss-20b MXFP4 — CPU

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/gpt-oss-20b-MXFP4.gguf -p "The capital of France is" -n 24 --temp 0 -g 0 --seed 1 --verbose-prompt --no-display-prompt`

Re-run for the step-6 CLI migration at `-n 16`:

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/gpt-oss-20b-MXFP4.gguf -p "The capital of France is" -n 16 --temp 0 -g 0 --seed 1 --verbose-prompt --no-display-prompt`

Rendered prompt IDs (65): `200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788, 17527, 558, 87447, 100594, 25, 220, 1323, 19, 12, 3218, 198, 6576, 3521, 25, 1202, 30377, 289, 25, 14093, 279, 2, 13888, 18403, 25, 8450, 11, 49159, 11, 1721, 13, 21030, 2804, 413, 7360, 395, 1753, 3176, 13, 200007, 200006, 1428, 200008, 976, 9029, 328, 10128, 382, 200007, 200006, 173781`.

Generated text (24 tokens; text inside quotes is rendered as `"The capital of France is"`): `<|channel|>analysis<|message|>The user says: "The capital of France is". They likely want the answer. The correct answer:`

Generated token IDs: `200005, 35644, 200008, 976, 1825, 5003, 25, 392, 976, 9029, 328, 10128, 382, 4050, 3164, 6960, 1682, 290, 6052, 13, 623, 6145, 6052, 25`.

Additional `-n 16` run used for the Section 5 CPU/Vulkan comparison, with the same CPU command except `-n 16`: prompt IDs matched the 65 IDs above. Generated text: `<|channel|>analysis<|message|>The user says: "The capital of France is". They likely`; token IDs: `200005, 35644, 200008, 976, 1825, 5003, 25, 392, 976, 9029, 328, 10128, 382, 4050, 3164, 6960`.

#### gpt-oss-20b MXFP4 — full Vulkan

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

Exact command (same prompt and generation settings as CPU, using `-g -1 --backend vulkan`):

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/gpt-oss-20b-MXFP4.gguf -p "The capital of France is" -n 16 --temp 0 -g -1 --backend vulkan --seed 1 --verbose-prompt --no-display-prompt`

Prompt IDs (65) are identical to the gpt-oss CPU baseline above. Generated text and IDs matched the CPU baseline exactly: `<|channel|>analysis<|message|>The user says: "The capital of France is". They likely`; `200005, 35644, 200008, 976, 1825, 5003, 25, 392, 976, 9029, 328, 10128, 382, 4050, 3164, 6960`.

#### DeepSeek-V2-Lite-Chat Q2_K — CPU

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

Exact command:

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/DeepSeek-V2-Lite-Chat.Q2_K.gguf -p "The capital of France is" -n 16 --temp 0 -g 0 --seed 1 --verbose-prompt --no-display-prompt`

Prompt tokens (12): `100000, 5726, 25, 429, 6077, 280, 7239, 317, 185, 185, 77398, 25`.

Generated text (16 tokens): ` Paris is the capital of France.\n\n*** Please note that this answer was`.

Generated token IDs: `8913, 317, 254, 6077, 280, 7239, 13, 185, 185, 16656, 6456, 4347, 344, 437, 3510, 438`.

#### DeepSeek-V2-Lite-Chat Q2_K — full Vulkan / DeepSeek2 MLA

Verification: identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653).

Exact command:

`src/OpenTail.Stingray.Cli/bin/Release/net10.0/stingray.exe -m models/_models/DeepSeek-V2-Lite-Chat.Q2_K.gguf -p "The capital of France is" -n 16 --temp 0 -g -1 --backend vulkan --seed 1 --verbose-prompt --no-display-prompt`

Prompt IDs (12) are identical to the DeepSeek2 CPU baseline above. Generated text (16 tokens): ` Paris is the capital of France.\n\nWould you like to know more about`.

Generated token IDs: `8913, 317, 254, 6077, 280, 7239, 13, 185, 185, 18684, 340, 837, 276, 1006, 691, 786`.

The DeepSeek2 CPU and full-Vulkan generated vectors differ beginning at generated token index 9 (`***` CPU vs `Would` Vulkan); each is a frozen execution-path baseline, not a parity claim. gpt-oss CPU and Vulkan vectors match for these 16 tokens.

### CLI refusal baselines (pre-migration)

All commands below used the unchanged CLI built in Release. Exact stderr lines:

| Model / options | Exit | Error line | Status |
|---|---:|---|---|
| SmolLM2-135M `--tq-mode bogus` | 1 | `Error: Unknown --tq-mode value 'bogus'. Expected one of: auto, lloydmax, kvarn.` | identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653) |
| SmolLM2-135M `--tq-mode kvarn` without `--tq` | 1 | `Error: --tq-mode kvarn requires --tq.` | identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653) |
| gpt-oss-20b `--tq --draft-lookup` | 1 | `Error: gpt-oss runs on its own CPU forward pass, which supports neither TurboQuant nor speculative decoding.` | identical to pre-CLI migration, re-run 2026-10-05 (commit 88bbc653) |

No hybrid-GDN GGUF or RWKV GGUF was found locally; the hybrid-GDN `--tq` refusal is therefore not captured against a real model, and RWKV remains selector-test-only.

### Server CPU baseline for Section 4

Started the Release server host on `http://127.0.0.1:5261` with `STINGRAY_MODEL=models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf`, `STINGRAY_N_GPU_LAYERS=0`, configuration `OpenTail:Stingray:Sampling:RepetitionPenalty=1.1`, and `MaxNewTokens=24`. Sent a `/v1/chat/completions` request with user content `Write a short story: Once upon a time`, `temperature=0`, `max_tokens=24`. Response: 39 prompt tokens, 24 completion tokens, exact text `Once upon a time, there lived a young girl named Lily. She was always curious and loved to explore her surroundings.` This matches the frozen S1 CLI CPU text exactly.

No local RWKV GGUF was found under `models/_models`. Timing is excluded; runs were not repeated for statistical performance measurement.
