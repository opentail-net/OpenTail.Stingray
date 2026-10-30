# Completed items split from 2026-10-08-architecture-semantics-admission-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-08-architecture-semantics-admission-plan.md](../2-coverage/2026-10-08-architecture-semantics-admission-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **0. Audit**
  - [x] 0.1 Classify every `FromGgufMetadata` caller (A generic / B semantics / C raw-parser test); record in Appendix A
  - [x] 0.2 Classify every named-arch branch in `ModelGraph.cs` (A generic / B semantics / C execution / D frontend / E handoff); record in Appendix B
  - [x] 0.3 Confirm `DefaultForwardPassFactories.cs` is unused; confirm whether Audio/Diffusion reference Engine; locate Cohere2 descriptor

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **1. Raw probe + load-context hyperparams**
  - [x] 1.1 Add `required ModelHyperparams Hyperparams` to `ArchitectureLoadContext`
  - [x] 1.2 Switch `CommonForwardPassFactory` (2 sites) and `DeepSeek2Architectures` to `ctx.Hyperparams`
  - [x] 1.3 Remove `Hyperparams` from `ArchitectureProbe`; fix all construction sites and tests

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **2. Core baseline + semantics context**
  - [x] 2.1 Add `ModelArchitectureSemanticsContext` (Core)
  - [x] 2.2 Add `CreateBaseline(metadata, tensorSource, metadataArchitecture)`; `FromGgufMetadata` becomes baseline + legacy branches (shim)
  - [x] 2.3 Baseline tests (no ArchitectureRegistry dependency)

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **3. Descriptor hook + resolver**
  - [x] 3.1 `ArchitectureDescriptor.ApplyModelSemantics`
  - [x] 3.2 `ArchitectureModelResolver` / `ResolvedModelArchitecture` (declared / canonical / metadata names; unknown stays fail-closed)
  - [x] 3.3 Resolution-order tests: declared, alias, relabelled, metadata-free, unknown
  - [x] 3.4 Descriptor contract tests (hook optional, Admitted keeps evidence + factory)

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **4. Route loaders through the resolver**
  - [x] 4.1 `Model.Load`
  - [x] 4.2 `ModelDescription.FromPackage` / `FromModel`
  - [x] 4.3 `RuntimeInstance` consumes `model.Hyperparams` only
  - [x] 4.4 Convergence test + `static-plan` output diff over admitted set

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **5. Pilots** (each: move rule, delete legacy branch, equivalence test, existing receipt/regression green)
  - [x] 5.1 SmolLM3 (NoPE step 4)
  - [x] 5.2 Cohere2 (LayerNorm, parallel residual, SWA, logit scale)
  - [x] 5.3 Gemma4 (per-layer dims/RoPE/KV heads/KV source, embed scale, softcap)
  - [x] 5.4 Gate: pilots needed only the one hook; else stop and re-assess

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **6. Remaining branches** (same recipe as pilots; one checkbox per group)
  - [x] 6.1 Llama4 / NoPE / sigmoid routing
  - [x] 6.2 OLMo / OLMo2 / OLMoE
  - [x] 6.3 JAIS / JAIS2
  - [x] 6.4 Granite / MiniCPM
  - [x] 6.5 Gemma / Gemma3
  - [x] 6.6 Qwen-VL / Qwen3-VL / Qwen3.5 GDN
  - [x] 6.7 LFM2 / Nemotron-H / AFMoE
  - [x] 6.8 Hunyuan / Maincoder / GPT-NeoX / Falcon / StableLM
  - [x] 6.9 DeepSeek2 (MLA semantics only; Vulkan execution untouched)

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **7. Cleanup**
  - [x] 7.1 Non-LLM callers (Audio/Diffusion/Cli/Embedding) moved per Appendix A
  - [x] 7.2 Remove emptied legacy branches from Core
  - [x] 7.3 Delete `DefaultForwardPassFactories.cs`
  - [x] 7.4 Update docs describing pre-`5c591e6` switches

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
- [x] **8. New-architecture exercises**
  - [x] 8.1 (A) plain dense: zero central edits
  - [x] 8.2 (B) semantic variant: zero central edits
  - [x] 8.3 (C) new compute family: execution edits only

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
  - [x] 9.1 Full solution `dotnet build` clean (warnings are errors)

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
  - [x] 9.2 `.Fast` test projects green (2026-10-08: Audio.Fast 94, Core 1019/49 skipped, ForwardPass.Fast 1041, Server.Fast 456, Sessions.Fast 129, Vulkan.Fast 46; 0 failed)

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
  - [x] 9.3 Heavy suites with `STINGRAY_RUN_HEAVY_TESTS=1` against real checkpoints in `models/`; check per-class wall time (CLAUDE.md rule 12: ~0.1s "pass" = silent no-op, not evidence)
    - PARTIAL 2026-10-08 (classes run one per process, post-migration; whole-project run hung then was OOM-reaped twice): REAL+PASS: SmolLm3GreedyParity 1/1 69s, Gemma4PrefillConsistency 3/3 49s, OlmoeGreedyParity 2 pass+1 skip 17s, Qwen2MoeGreedyParity 2/2 90s. FAIL (pre-existing, bugstofix #24, token 9 divergence from Q3_K kernel rewrite e7b7aa8a; hyperparam asserts pass): DeepSeek2GreedyParity 1/2. SKIPPED (no checkpoint/reference, UNVERIFIED): Maincoder, Falcon, GptNeox, Olmo, Olmo2, GraniteMoe, Gemma4Cpu (3/4). NOT RUN (150s cap hit under memory pressure, not a result): Afmoe, Apertus, Glm4, GlmMoe, Gpt2, Granite, GraniteHybrid, HunyuanMoe, LlamaFour, Mixtral, PhiMoe; Exaone45, Lfm2, NemotronH never started.
    - UPDATE 2026-10-08 (one class per process, real weights, 0 failed): Afmoe 2/2, Lfm2 3/3, NemotronH 2/2, GraniteHybrid 5/5, Mixtral 2/2, GlmMoe 2/2. Skipped in ~0.1s = no checkpoint (UNVERIFIED): Gpt2, Glm4, Apertus, Granite. Hit the 900s cap (large models, two copies of the runner overlapped; inconclusive, not failures): Exaone45, HunyuanMoe, LlamaFour, PhiMoe. Perf sweep of 11 text checkpoints in PerformanceLeague.md ("Full re-sweep ... 2026-10-08"): one confirmed regression, Phi-3-mini prefill 40.1 -> 35.4 t/s (-12%), not bisected; CLOSED 2026-10-08 as won't-investigate (Phi-3 has tiny monthly downloads; documented in PerformanceLeague.md).

> from: ## 5. Phases (tick as completed; each phase ends with a green build + the named tests)
  - [x] 9.4 Per-pilot/per-migrated-arch real-weight receipt re-run; list archs with no checkpoint on disk as UNVERIFIED rather than green
    - FINAL 2026-10-08 (ForwardPass heavy classes, one per process, real weights): **PASS** SmolLm3 1/1, Gemma4PrefillConsistency 3/3, Olmoe 2/3 (1 skip), Qwen2Moe 2/2, Afmoe 2/2, Lfm2 3/3, NemotronH 2/2, GraniteHybrid 5/5, Mixtral 2/2, GlmMoe 2/2, PhiMoe 2/2 (544 s), Exaone45 2/2 (870 s), HunyuanMoe 2/2 (627 s), GptNeox 2/2 (after fixing the pythia filename lookup; it had been skipping silently). **KNOWN FAIL (pre-existing, bugstofix #24):** DeepSeek2 greedy parity 1/2. **DEFERRED by the user:** LlamaFour (Llama 4, ~44 GB mapped vs ~23 GB free RAM; cancelled after ~5 min, not a result). **UNVERIFIED (no checkpoint or reference on disk, tests skip):** Gpt2, Glm4, Apertus, Granite, Maincoder, Falcon, Olmo, Olmo2, GraniteMoe, 3 of 4 Gemma4Cpu tests. Not run: Cuda, Vulkan, Sessions, Server heavy projects, and the other heavy ForwardPass classes outside the migrated architectures. Qwen3.8/Qwen3.6/Ornith (qwen35 family) additionally run end to end on CPU after the factory fix (and Qwen3.8 with MTP).

