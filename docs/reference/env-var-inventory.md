# `STINGRAY_*` environment-variable inventory — name-complete and classified (re-verified 2026-10-31)

> **Status, 2026-10-31: name-complete again.** The table below has a row for all **287** names in
> `KnownEnvironmentVariables.All`. It had fallen behind (192 rows at the 2026-10-31 external review, 95 names missing); the gap was
> pinned by a test and closed the same day in four batches (PRs 65 to 68), each row written from the variable's actual read-site
> rather than its name. `KnownEnvironmentVariablesInventoryTests.InventoryTable_CoversTheRegistry_ExceptTheListedGap` now fails if a
> variable is registered without a row, or a row names something the registry does not contain. The "name-complete" claims in the
> history below were true on the dates given.

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

**Reconciled again 2026-10-29 — `KnownEnvironmentVariables.All` now contains **287** names** (added `STINGRAY_ENCODER_REQUEST_PATHS`: `1` lets `/v1/embeddings`, `/api/embed` and `/v1/rerank` requests name a model path on disk; off by default, so only `STINGRAY_EMBEDDING_MODEL` / `STINGRAY_RERANK_MODEL` are loaded)
(2026-10-13: 286 names; added `STINGRAY_TELEMETRY` and `STINGRAY_STATS_JSON`; earlier, on 2026-10-11, the Qwen Image and Vulkan performance switches: `STINGRAY_QWEN_PROFILE`, `STINGRAY_QWEN_MLP_CHUNKS`, `STINGRAY_VK_QGEMM_SMALLM`, `STINGRAY_VK_MXFP4_V2`, `STINGRAY_VK_MATVEC_V2`, `STINGRAY_VK_PATH3_CHUNK`, `STINGRAY_VULKAN_DECODE_ATTN_GQA`, `STINGRAY_VULKAN_DECODE_PROFILE`, `STINGRAY_VULKAN_FLASH_PREFILL`, `STINGRAY_VULKAN_HYBRID_PREFILL_CHUNK`)
(2026-10-09: 274 names; added `STINGRAY_ALLOW_EXTERNAL`, `STINGRAY_CONFIG_DIR`, `STINGRAY_IGNORE_PREFLIGHT`)
(2026-10-05: registered `STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL` and `STINGRAY_MOE_PHASE_TIMING`, both read in `src/` but missing from the registry. Removed `STINGRAY_HANDOFF_TEST_MODEL`: it is read only by `HybridCpuPrefillHandoffTests` under `tests/`, never in `src/`, so the registry — whose tests scan `src/` — cannot justify it, and registering it would make `doctor` accept a name the engine itself never reads. It stays a test-only switch, like `STINGRAY_RUN_HEAVY_TESTS`; optional `STINGRAY_HANDOFF_TEST_GPU_LAYERS` (default 4) and `STINGRAY_HANDOFF_TEST_CTX` (default 1024) are read by the same test. Net +1 from the 267 the registry held after the earlier, un-documented edits.)

**Reconciled 2026-10-03 — `KnownEnvironmentVariables.All` contained **266** names**
(2026-10-03, later: `STINGRAY_MOE_SLOTS`, `STINGRAY_MOE_TIMING`, `STINGRAY_OFFLINE`, `STINGRAY_HYBRID_CPU_PREFILL`, `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS`, `STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB` and `STINGRAY_PREFILL_TIMING`, `STINGRAY_GPU_CPU_PREFILL` and `STINGRAY_CUDA_HYBRID_CPU_PREFILL` and `STINGRAY_HYBRID_CPU_PREFILL_WARM`, added by the TensorSharp selective-port work.)
(2026-10-02: `STINGRAY_CPU_POOL`, `STINGRAY_CPU_POOL_BLOCKS`, `STINGRAY_CPU_SPIN` for the opt-in spin worker
pool, registered in 8f45d133 without a count update; and `STINGRAY_EXPERIMENTAL_PRISM`, the "ported, not
verified" Bonsai2 PRISM gate.)

**Reconciled 2026-09-27 — the registry then contained 251 names**
(2026-09-28: `STINGRAY_MODEL_HOME`, the `stingray setup` model home, docs/103 item 16; plus three added earlier the same day without a count update: `STINGRAY_RECURRENT_BATCHED_PREFILL` (item 12 fallback switch) and the ACE-Step parity hooks `STINGRAY_ACESTEP_DUMP` / `STINGRAY_ACESTEP_NOISE` (item 14).)
(2026-09-27: the reference-parity debug hooks from docs/102 #11/#13 — `STINGRAY_SD3_DUMP_LATENT_PATH`,
`STINGRAY_FLUX2_INJECT_NOISE_PATH` / `_DUMP_LATENT_PATH` / `_DUMP_COND_PATH`,
`STINGRAY_QWENIMAGE_INJECT_NOISE_PATH` / `_DUMP_LATENT_PATH` / `_DUMP_COND_PATH`, `STINGRAY_HUNYUAN_INJECT_NOISE_PATH` / `_DUMP_COND_PATH` / `_DUMP_VEC_PATH`, `STINGRAY_SA3_NOISE` / `STINGRAY_SA3_DUMP`
— and `STINGRAY_VISION_GEMM`, the `0|nomm|noattn` A/B switch for the batched vision-encoder GEMM paths.)

**Reconciled 2026-09-26 — `KnownEnvironmentVariables.All` contained **229** names**
(2026-09-26: `STINGRAY_Q6K_GEMM`, the A/B toggle for the group-paired Q6_K prefill GEMM;
`STINGRAY_HUNYUAN_GPU`, `0` keeps HunyuanVideo's transformer blocks on the CPU with a GPU backend.)

**Reconciled 2026-09-25 — `KnownEnvironmentVariables.All` contained **227** names**
(2026-09-25b: `STINGRAY_EMBEDDING_MODEL` and `STINGRAY_RERANK_MODEL`, the server defaults for `/v1/embeddings` and `/v1/rerank`. Earlier the same day, 22 added: diffusion GPU/FP16 toggles for FLUX/FLUX.2/Qwen-Image/SD3, SD3/Hunyuan/T5 debug dumps,
`STINGRAY_ATTN128_LEGACY`, `STINGRAY_FORCE_MHA64_F32`, the ASR/TTS/RVC profiling switches and
`STINGRAY_SILERO_VAD_PATH`).

**Reconciled 2026-09-16 — `KnownEnvironmentVariables.All` contained **203** names**
(reconciled recent diffusion, audio, profiling, and Wan2.1 / LTX / CosyVoice3 / XTTS diagnostic environment variables into the source registry).

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
admitting it -- see `docs/1-correctness/bugstofix.md`.)

**Reconciled again 2026-08-24 — `KnownEnvironmentVariables.All` now contains **170** names**
(added `STINGRAY_AUDIO_DIAGNOSTIC_DUMP`: a pre-existing `Environment.GetEnvironmentVariable` read in
`ChatterboxPipeline`/`ChatterboxDecoder` that was never registered, caught by CI failing
`KnownEnvironmentVariablesTests.ListMatchesSource`.)

**Reconciled again 2026-08-21 — `KnownEnvironmentVariables.All` now contains **169** names**
(added `STINGRAY_MLA_TRACE`, a temporary diagnostic env var for the deepseek2 MLA
ground-truth-diffing investigation, and `STINGRAY_GGML_F16_DOT`, an opt-in switch that rounds
F16-weight matmul activations to fp16 before dotting to match ggml's `vec_dot_type=F16` pairing
for bit-parity comparisons — see `docs/1-correctness/bugstofix.md` — plus one further pre-existing count drift
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
| `STINGRAY_ASR_PROFILE` | diagnostic | `=1` prints the prefill and decode wall time (and the raw decoded text) of the VibeVoice ASR generator to stdout. |
| `STINGRAY_AUDIO_DIAGNOSTIC_DUMP` | expert | Enables `Generate()` timing-breakdown diagnostic logging in `ChatterboxPipeline`/`ChatterboxDecoder` (`=1`). |
| `STINGRAY_CFG_TRACE` | diagnostic | `=1` prints classifier-free-guidance trace output to stderr from the CosyVoice 3 flow encoder, DiT model and pipeline (the pipeline reports the frame with the largest deviation). |
| `STINGRAY_COSYVOICE3_TANH_GELU` | experimental | `=1` uses the tanh GELU approximation instead of the exact erf GELU in the CosyVoice 3 DiT feed-forward. Added for A/B listening because two reference implementations disagree; off by default and not to become a default without a listening decision. |
| `STINGRAY_DEBUG_COSYVOICE3` | diagnostic | Any non-empty value prints CosyVoice 3 token and frame counts, speaker-embedding values and mel statistics to stderr after the flow-matching solve. |
| `STINGRAY_DUMP_CONVPOST_PATH` | diagnostic | Path. The HiFT vocoder writes its post-convolution tensor there (parity debugging against the reference). |
| `STINGRAY_DUMP_CONVPRE_PATH` | diagnostic | Path. The HiFT vocoder writes its pre-convolution tensor there (parity debugging against the reference). |
| `STINGRAY_DUMP_STAGE0_PATH` | diagnostic | Path. The HiFT vocoder writes the stage-0 tensor of its first iteration there (parity debugging against the reference). |
| `STINGRAY_FA_DUMP_DIR` | diagnostic | Directory. The Qwen ASR audio encoder and forced aligner write intermediate tensors there (mel, conv-position and transformer outputs) for parity debugging. |
| `STINGRAY_FA_READ_SHIFT` | diagnostic | Integer (default 0). Shifts the position from which the forced aligner reads its timestamp class logits; an experiment handle for alignment-offset debugging. |
| `STINGRAY_FA_TRACE` | diagnostic | `=1` enables per-layer hidden-state taps and logit statistics tracing in the Qwen forced aligner (needs a forward pass that supports hidden taps). |
| `STINGRAY_FORCEDALIGNER_PERF` | diagnostic | `=1` prints timing stages of the Qwen audio encoder used by the forced aligner (convolution stages and others) to stdout. |
| `STINGRAY_FORCEDALIGNER_TRACE` | diagnostic | `=1` prints forced-aligner trace output; the audio encoder also treats it, like `STINGRAY_FORCEDALIGNER_PERF`, as a reason to print its stage trace. |
| `STINGRAY_QWENASR_DIAG` | diagnostic | `=1` prints statistics of the Qwen ASR audio soft tokens and compares their magnitude with a real text-embedding row (a check for a scale mismatch between the encoder output and the text embeddings). |
| `STINGRAY_RVC_PROFILE` | diagnostic | `=1` prints stage timings of the RVC voice-conversion pipeline and its RMVPE pitch encoder. |
| `STINGRAY_RVC_TRACE` | diagnostic | `=1` enables trace output from the RVC RMVPE pitch encoder. |
| `STINGRAY_SILERO_VAD_PATH` | stable | Path to the Silero VAD ONNX model. If the file exists it is loaded first; otherwise the default search runs (the working directory and the application directory, walking up their parents). |
| `STINGRAY_TTS_PROFILE` | diagnostic | `=1` collects and prints per-stage timing of the VibeVoice TTS generation loop. |
| `STINGRAY_TTS_TRACE` | diagnostic | Set to any value (even empty) to print the VibeVoice TTS control logits (speech start, end and diffusion, and the end-of-sequence logit) at every generation step to stdout. |
| `STINGRAY_XTTS_LAYER_TRACE` | diagnostic | `=1` prints a per-layer trace of the XTTS GPT trunk. |
| `STINGRAY_XTTS_TRACE` | diagnostic | `=1` prints XTTS pipeline trace output, including the latent-length to output-sample-count arithmetic. |

## OpenTail.Stingray.Cli

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CONFIG_DIR` | stable | Overrides per-user configuration directory containing favourites.json. |
| `STINGRAY_DBG_TOKEN_RANK` | diagnostic | Token id. At the first generated step the CLI prints that token's logit and its rank among all logits to stderr (`[DBG-RANK]`). |
| `STINGRAY_DSPARK_TIMING` | experimental | Diagnostic (`=1`): print the DSpark block timing split (host time, GPU wait including the D2H download sync, and the host Markov/confidence head chain) from the speculative run command. |
| `STINGRAY_RAW_PROMPT` | expert | `=1` bypasses the chat template and sends the prompt as raw text (llama.cpp `--no-conversation` parity testing). Not for normal use. |
| `STINGRAY_SDCPP` | expert | Path to the `sd-cli` / `sd` executable used by `image` when no explicit path is given and the file exists. Otherwise `tools`, `tools/sd` and the working or app directory are searched. |
| `STINGRAY_SPEC_SAMPLE` | experimental | `=0` disables distribution-preserving sampled speculative decoding (temperature above 0 on the model-draft path); greedy speculation is unaffected. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Core, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_MTP_BATCH_MAX` | expert | Integer clamped to 2..8, default 4. Maximum tokens in one batched MTP verify (speculation settings; also used by the CUDA hybrid-GDN pass). |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Cpu

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_PREFILL_Q8` | experimental | Default **ON** (user decision 2026-10-02, [ADR-0003](adr-0003-cpu-int8-prefill-default.md); was off 2026-10-01..02) int8 batched-prefill dispatch tier, 2.2-2.9x prefill on Q4_K dense models (llama.cpp uses the same technique). `=0` is the rollback: per-token prefill, bit-identical to token-by-token decode (not "F32": the decode matvec also int8-quantizes activations for K-quant weights). Routed MoE experts additionally need `STINGRAY_MOE_PREFILL_Q8=1` (default off). |
| `STINGRAY_Q5K_DECODE_Q8K` | experimental | Default-OFF. `=1` quantizes the activation to Q8_K for the Q5_K attention-output (`wo`) matvec in non-MLA CPU decode, as ggml does (`ggml_vec_dot_q5_K_q8_K`). Experiment for the GLM-4.5 PPL gap (bugstofix item 09). Measured 2026-10-01: improves paired-NLL fidelity on 5 of 6 Q5_K models but costs decode speed (SmolLM2 -37%, phi-2 -16%), so it stays default-off. |
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
| `STINGRAY_DISABLE_MTP` | experimental | `=1` turns the native MTP head off (speculation setting `MtpEnabled`, default on). |
| `STINGRAY_DSPARK_MIN_CONFIDENCE` | experimental | DSpark confidence floor in [0, 1]. Resolution order: `--dspark-min-confidence` flag, then this variable, then 0 (disabled: verify the whole block). |
| `STINGRAY_DSPARK_VERIFY_LEN` | experimental | DSpark per-step verify-length cap (positive integer). Resolution order: `--dspark-verify-len` flag, then this variable, then 0 (uncapped; the confidence trim decides). |
| `STINGRAY_MTP_DRAFT_N` | expert | MTP draft length (positive integer). Default 3; the CPU hybrid-GDN pass defaults to 1 when neither this variable nor an explicit maximum is set. |
| `STINGRAY_PROFILE_GPU_SPLIT` | diagnostic | `=1` enables the Vulkan GPU-phase profile counters (`opentail.profile.phase.duration`) and marks the run-stats JSON `perturbed: true`, because profiling changes the timing. Never use such a run for benchmark comparisons. |
| `STINGRAY_TRACE_MTP` | diagnostic | Diagnostic (`=1`): print the MTP phase times (draft, verify, commit, in ms) after a speculative run, when drafts were emitted. |
| `STINGRAY_GC_STATS` | diagnostic | CLI: print GC collections, bytes allocated and total pause at process exit (2026-10-08) |
| `STINGRAY_EVENTS` | diagnostic | CLI: stream EngineEvents (model loaded, prefill/decode completed, MTP step, GC stats) to stderr as JSON lines (2026-10-08) |
| `STINGRAY_TELEMETRY` | diagnostic | CLI: attach an in-process listener to the engine `Meter`/`ActivitySource` and print a short instrument report at exit (stderr); needs no collector (2026-10-13; see docs/guides/observability.md) |
| `STINGRAY_STATS_JSON` | diagnostic | CLI: write one versioned machine-readable result file (`schema_version` 1: load, prefill, decode tokens and times, speculative counts, `perturbed`) to the given path at exit; the structured alternative to regex-matching printed lines (2026-10-13; see docs/guides/observability.md) |
| `STINGRAY_MODEL_DIRS` | diagnostic | tests/tools: extra folders (path list) searched first by `ModelLocator` when looking for a checkpoint by file name (2026-10-08) |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_MOE` | expert | Mirrors `OpenTailStingrayServerOptions.CpuMoe`. |
| `STINGRAY_CPU_POOL` | experimental | `spin` runs the CPU decode-path loops (matvecs, decode attention, row kernels, MoE decode sweeps) on a persistent spin-then-park worker pool (`SpinParkWorkerPool`, ported from TensorSharp) instead of `Parallel.For`. Opt-in; outputs are bit-identical. Measured 2026-10-02, interleaved: dense-model decode +19..27% (SmolLM2-135M 0.52x -> 0.66x, 360M 0.68x -> 0.81x, Qwen2.5-0.5B 0.73x -> 0.87x of llama.cpp); Mistral-7B tie. **Not for MoE**: OLMoE/LFM2 prefill -18..19%. Prefill/batched kernels always stay on `Parallel.For`. `docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md` §1. |
| `STINGRAY_CPU_POOL_BLOCKS` | experimental | With `STINGRAY_CPU_POOL=spin`: blocks handed to the pool per thread (default 4; 8 measured slightly better on small models, 2 worse; 0 = one block per index, which measured -25..35% decode). |
| `STINGRAY_DSPARK_PLACE` | experimental | Mirrors `OpenTailStingrayServerOptions.DSparkPlace`; DSpark itself is parked (docs/00-current-work.md), not scheduled. |
| `STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL` | experimental | `1` lets a real DiffusionGemma checkpoint (Gemma-4 MoE layout: fused gate/up experts, router scale, `self_cond_*` MLP) load; without it such files are refused. **Ported, not verified** (CLAUDE.md rule 14): real-weight execution stays guarded until independent verification passes. |
| `STINGRAY_EXPERIMENTAL_PRISM` | experimental | `1` lets Bonsai2 PRISM GGUFs (PQ2_0/PTQ1_0 weights + `prism.hadamard.*` transforms) load on the CPU hybrid-GDN pass. **Ported, not verified** (CLAUDE.md rule 14): without it such files are refused. Not verified against the publisher reference (PrismML-Eng/llama.cpp). `docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md` §3b. |
| `STINGRAY_EXPERT_STATS` | diagnostic | Mirrors `OpenTailStingrayServerOptions.ExpertStatsPath` (writes SLRU hit-rate stats to a file on exit — troubleshooting only). |
| `STINGRAY_MOE_GPU_PREFILL` | experimental | Mirrors `OpenTailStingrayServerOptions.GpuMoePrefill`. |
| `STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS` | experimental | Integer, default 64: minimum prompt length for the CUDA hybrid-GDN pass to run MoE prefill on the GPU when op-offload is on. Below it, and for all single-token decode and MTP-verify steps, the byte-exact CPU MoE runs. `0` always engages. Measured in register mode: the CPU wins below about 50 tokens, the GPU wins 28-67% from about 120 up. |
| `STINGRAY_MOE_PHASE_TIMING` | diagnostic | `1`: CPU MoE FFN decode only. Accumulates wall time of each phase of one `ForwardPass.MoeFfn` call (router, shared expert, activation quantization, the two expert sweeps, SiLU-mul, sequential experts) plus two empty-sweep fork/join probes, and prints them per call at exit. Measurement only; no kernel changes. |
| `STINGRAY_MOE_PIN_MODE` | experimental | `register` (default) pins the expert mmap pages in place with cudaHostRegister (no RAM copy, no page-cache eviction, no decode regression, DMA about 13 GB/s). `copy` makes the original cudaMallocHost duplicate (DMA about 26 GB/s but evicts the page cache the CPU decode streams experts from, about -25% decode). |
| `STINGRAY_MOE_PREDICT_PREFETCH` | expert | Mirrors `OpenTailStingrayServerOptions.MoePredictPrefetch`. |
| `STINGRAY_MOE_WARMPIN` | expert | Mirrors `OpenTailStingrayServerOptions.MoeWarmPin`. |
| `STINGRAY_GPU_CPU_PREFILL` | experimental | Full-offload Vulkan pass (`-g -1`), MoE models only: a fresh prompt of at least `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS` tokens is prefilled by the CPU batched pass and its K/V (F32, or packed fp16 when the GPU KV is narrowed) handed to the GPU caches. On by default for MoE; `0` turns it off, `all` admits families without a parity test. |
| `STINGRAY_CUDA_HYBRID_CPU_PREFILL` | experimental | CUDA hybrid: the same handoff, **opt-in and untested on CUDA hardware** (written 2026-10-03 without an NVIDIA GPU). `1` enables it for admitted families, `all` for any; only for F32 GPU KV. |
| `STINGRAY_HYBRID_CPU_PREFILL` | experimental | Vulkan hybrid (`-g N`): a fresh prompt of at least `..._MIN_TOKENS` tokens is prefilled by the CPU batched pass and its K/V handed to the hybrid. `0` turns it off, `all` admits model families without a parity test. Fresh sequences only; refused for TurboQuant, `STINGRAY_KV_DTYPE=bf16`, per-layer head dims. |
| `STINGRAY_HYBRID_CPU_PREFILL_WARM` | experimental | `0` turns off the post-handoff expert-cache warm-up: after a CPU prefill the hybrid preloads the experts the prompt used most into the GPU slot cache (up to its capacity) so decode does not start cold. |
| `STINGRAY_HYBRID_CPU_PREFILL_MIN_TOKENS` | experimental | Shortest prompt that takes the CPU-prefill handoff (default 32). |
| `STINGRAY_HYBRID_CPU_PREFILL_KV_BUDGET_MB` | experimental | Largest temporary CPU KV cache the handoff may allocate (default 4096 MiB); above it the sequential prefill runs. |
| `STINGRAY_IGNORE_PREFLIGHT` | stable | `1` lets the server load a CPU-only GGUF even when the memory check says it will not fit (the `--ignore-preflight` flag of `chat`/`run`). |
| `STINGRAY_PREFILL_TIMING` | experimental | `1`: Vulkan hybrid prints, per prefill, which path ran and the CPU-prefill / KV-handoff / end-to-end times. |
| `STINGRAY_MOE_TIMING` | experimental | `1`: Vulkan hybrid (`-g N`) MoE only. On exit prints, per GPU MoE layer-step, where the time went: GPU wait + router, cache lookup, CPU fallback for missed experts, record tail; and per-expert gate/up/down fallback time. |
| `STINGRAY_MOE_SLOTS` | experimental | Vulkan hybrid (`-g N`) only: caps the GPU expert cache at this many (layer, expert) slots instead of sizing it to hold every expert. Misses are computed on the CPU. For testing eviction on models that would otherwise fit; `VulkanHybridOlmoeParityTests` uses the constructor argument for the same thing. |
| `STINGRAY_ALLOW_EXTERNAL` | stable | `0`/`false`/`off`/`no`/`deny`: nothing may touch the network (remote scout, quant picker, backlog, triage, and the offline check used by catalog installs). Default (unset) and `1`/`true`/`on`/`yes`/`allow`: allowed. Deny always wins: `STINGRAY_OFFLINE` or `HF_HUB_OFFLINE` still deny, and an unrecognised value is treated as off. Only Hugging Face hosts are ever contacted; `HF_TOKEN` is sent to huggingface.co only. |
| `STINGRAY_OFFLINE` | stable | `1`/`true`: catalog installs never touch the network (files already present are still hash-checked; a missing one fails with a clear message). `HF_HUB_OFFLINE` is honoured the same way. |
| `STINGRAY_MOE_WARMPIN_AFTER` | expert | Mirrors `OpenTailStingrayServerOptions.MoeWarmPinAfter`. |

## OpenTail.Stingray.Cli, OpenTail.Stingray.Engine, OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_PREFILL_CHUNK` | expert | Mirrors `OpenTailStingrayServerOptions.PrefillChunkTokens`. |
| `STINGRAY_PREFILL_DEQUANT_MB` | expert | Legacy setting mirroring `OpenTailStingrayServerOptions.PrefillDequantCacheMb`; currently dormant with the CPU prefill kernels. |

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
| `STINGRAY_BATCHED_MATVEC_TIER` | experimental | `=0` disables the CPU non-BLAS batched path that consumes a batch in quads and pairs (`MatVec4In`/`MatVec2In`) instead of N single `MatVec` calls. Bit-identical either way; a weight-reuse switch. Default on (4-way concurrent decode 28.90 to 30.28 tok/s aggregate, per the session runtime plan section 3.4.9). |
| `STINGRAY_CPU_DECODE_VIA_GEMM` | experimental | `=1` runs single-token CPU decode through the batched Q8 GEMM path (ggml Q8_K activations instead of Q8_KS), which changes decode numerics. Opt-in until measured and decided; needs the Q8 prefill path to be enabled (`SimdKernels.DecodeViaGemm`, also settable by tests). |
| `STINGRAY_GEMM_PATH` | experimental | CPU GEMM path selector. Default Path 2 (Q8_K activations, repacked weights); `1` or `path1` selects Path 1 (Q8_KS activations). Path 2 measured 1.83x faster on the isolated matmul, 1.11x end to end, and no worse on wikitext-2 PPL (16.0484 vs 16.0870), per the comment in `GemmPath.cs`. |
| `STINGRAY_LEGACY_DOTQ4K` | expert | `=1` forces the fully-F32 legacy Q4_K dot product instead of the default int8-quantized-activation kernel, for the Q4_K matvec paths. Needed for diffusion models with wide-dynamic-range activations; the default was validated for LLM decode perplexity only. |
| `STINGRAY_MATVEC_WIDE8` | experimental | `=1` selects the experimental 8-wide Q4_K dot (`DotQ4K_Wide8`) for an A/B on the real CLI. Default off. Marked in the source as temporary: to be removed once the A/B is settled. |
| `STINGRAY_TRACE_GDN_INTERNAL` | diagnostic | Diagnostic (`=1`): dump head-0 GDN internals (decay scalar, state L2 after decay and after the rank-1 update, p-readout) in the CPU GDN kernels, written for the qwen3.6-27B-MTP position-12 parity bug. Layers from `STINGRAY_TRACE_GDN_LAYERS`, position from `STINGRAY_TRACE_GDN_POS`. Fires only where the caller passes layer and position. |
| `STINGRAY_TRACE_GDN_LAYERS` | diagnostic | Diagnostic: comma-separated layer indices for `STINGRAY_TRACE_GDN_INTERNAL`. Default `0,20,40,60`. |
| `STINGRAY_TRACE_GDN_POS` | diagnostic | Diagnostic: token position for `STINGRAY_TRACE_GDN_INTERNAL`. Default 12. |

## OpenTail.Stingray.Cpu, OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_CPU_SPIN` | experimental | With `STINGRAY_CPU_POOL=spin`: spin rounds (`Thread.SpinWait(64)` each) a pool worker makes between jobs before parking (default 4096, TensorSharp's measured value). |
| `STINGRAY_CPU_THREADS` | expert | Mirrors `OpenTailStingrayServerOptions.CpuThreads`. Default since 2026-10-02: physical core count (`CpuTopology.PhysicalCores`), not logical processors; measured never slower and up to ~7% faster on an 8-core/16-thread CPU (fcfb5c57). |

## OpenTail.Stingray.Cuda

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_ACT_SOA` | experimental | `=1` makes the CUDA prefill MMQ over SoA-repacked weights read activations from the SoA Q8_1 layout. Bit-identical; default off until an end-to-end win is proven. No effect for AoS-weight prefill. |
| `STINGRAY_ACT_SOA_CPA` | experimental | `=1` routes the Q8_0/Q4_0 CUDA prefill MMQ through cp.async double-buffered kernels (sm_80 and newer; scalar fallback below). Bit-identical. Default off: a +10-15% kernel-level win on probe shapes but end-to-end neutral on the real 48-layer prefill (matmul 74 to 73 ms), per the source comment. |
| `STINGRAY_ATTN_WAVE_BUDGET_MB` | experimental | Positive integer, default 256: scratch budget in MiB for the CUDA attention wave scratch buffer. |
| `STINGRAY_BATCH_DECODE_WS_V2` | experimental | `=0` restores the earlier byte-gather Q6_K weight-stationary batched-decode kernel instead of the word-load plus funnel-shift one (bit-identical either way). Default on. |
| `STINGRAY_CUDA13` | experimental | `=1` makes the CUDA library resolver probe the CUDA 13 DLL pair first and fall back to the other major only if it is absent. Otherwise the other major is probed first. |
| `STINGRAY_CUDA_PRECISION` | experimental | Overrides the cuBLAS GEMM compute type: `fp32`, `fp16`, `bf16`, `fp8` or `fp8e4m3`. Unrecognised values fall back to auto-detection by compute capability (fp8 on sm_90+ with CUDA 12 or newer, bf16 on sm_80+, fp16 on sm_53+, else fp32). Affects only the cuBLAS path; custom NVRTC kernels keep fp32 accumulators. |
| `STINGRAY_DECODE_MMQ_BM32` | experimental | `=0` forces the BM=64 decode-MMQ tile for all shapes. BM=32 is on by default for grid-starved low-row shapes; output is bit-identical. |
| `STINGRAY_GPU_ARGMAX` | experimental | `=0` forces the legacy full-vocab logits download instead of the on-device greedy argmax (8 bytes per token, bit-exact for finite logits). Default on. |
| `STINGRAY_Q40_DP4A` | experimental | `=0` routes Q4_0 CUDA decode matvecs through the fp32 matvec instead of the dp4a/Q8_1 kernel. Default on; the dp4a path is argmax-stable, not bit-exact. |
| `STINGRAY_Q6K_DECODE_MMQ` | experimental | `=0` disables the Q6_K decode-MMQ tile; Q6_K trunk shapes then use the bit-exact weight-stationary matvec. Default on. |
| `STINGRAY_Q80_DP4A` | experimental | `=0` routes Q8_0 CUDA decode matvecs through the fp32-decode kernel instead of the dp4a/Q8_1 kernel. Default on; the dp4a path is argmax-stable, not bit-exact. |

## OpenTail.Stingray.Cuda, OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCH_DECODE_MMQ` | experimental | Tri-state for the MMQ path in CUDA batched decode. Unset: on for multi-user continuous batching, off for speculative verify (which must stay bit-exact). `0`: off everywhere. `1`: on everywhere including spec-verify (argmax-stable; perf A/B only). |
| `STINGRAY_PREFILL_MMQ` | experimental | `=0` reverts CUDA prefill from the MMQ path to the cuBLAS GEMM path (MMQ drops the fp16 weight temporary). Default on, under the prefill-GEMM gate. |
| `STINGRAY_TRUNK_MATVEC_FAST` | experimental | `=1` enables the fast Q8_0 trunk matvec in the CUDA hybrid-GDN plain-decode path. Default off. |

## OpenTail.Stingray.Engine

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_BATCHED_ATTN` | experimental | `=0` forces the per-position attention path in the CUDA hybrid-GDN batched trunk. Default on. |
| `STINGRAY_BATCHED_CPU_FFN` | experimental | `=0` is the kill-switch for the batched CPU dense FFN in CUDA hybrid-GDN prefill. Default on. Decode is untouched. |
| `STINGRAY_BATCHED_FFN` | experimental | `=0` forces the per-token FFN path (the bit-exact reference) in the CUDA hybrid-GDN batched trunk. Default on. |
| `STINGRAY_BATCHED_GDN_SCAN` | experimental | `=0` falls back to the per-position loop inside `GdnBlockBatched` of the CUDA hybrid-GDN pass. Default on. |
| `STINGRAY_BATCHED_PREFILL` | experimental | `=0` forces the bit-exact per-token prefill loop in `CudaForwardPass` (A/B and parity debugging). Default on (batched-trunk prefill). |
| `STINGRAY_BATCHED_TRUNK` | experimental | `=0` falls back to the per-token trunk loop in CUDA hybrid-GDN prefill. Default on. |
| `STINGRAY_BATCH_DECODE_GEMM` | experimental | `=1` routes CUDA batched decode through the compute-bound GEMM/MMQ path (argmax-stable, not bit-exact). Default off: measured slower than GEMM-N at realistic batch sizes (Qwen3-8B Q4_K_M on a 4070 Ti, N=4 99 vs 57 t/s aggregate). |
| `STINGRAY_BATCH_DECODE_RAGGED` | experimental | `=0` restores the per-sequence QK-norm/RoPE/KV-append/attention launches in CUDA batched decode instead of the ragged kernels whose grid covers all sequences at their own positions (bit-identical per sequence, issue 197). Default on. |
| `STINGRAY_BATCH_DECODE_WS` | experimental | `=0` falls back to the GEMM-N matvec (`MatMulBatched`) in CUDA batched decode instead of the weight-stationary path. Default on. |
| `STINGRAY_BYPASS_ATTN` | experimental | Bisection only (`=1`): zero the attention block's contribution in the CPU hybrid forward pass. Default off. Also disables the optimized chunked prefill. |
| `STINGRAY_BYPASS_GDN` | experimental | Bisection only (`=1`): zero the Gated-DeltaNet block's contribution in the CPU hybrid forward pass to localise a bug. Default off. Also disables the optimized chunked prefill. |
| `STINGRAY_BYPASS_MOE` | experimental | Bisection only (`=1`): zero the MoE block's contribution in the CPU hybrid forward pass. Default off. Also disables the optimized chunked prefill. |
| `STINGRAY_CPU_GDN` | experimental | `=1` runs the GDN block on the CPU in the CUDA hybrid-GDN pass instead of the full-GPU block (default). For bisecting parity bugs. |
| `STINGRAY_CUDA_GRAPH` | experimental | `=0` reverts `CudaForwardPass` decode from CUDA graphs to direct launches. Default on (measured +9-10% on the all-GPU Gemma 4 path at low and about 1K context). |
| `STINGRAY_CUDA_MATVEC_BENCH` | bench | Benchmark (`=1`): after CUDA warm-up, time the Q4_K matvec kernel in isolation at the three FFN shapes (`BenchMatVec` in `CudaForwardPass`). |
| `STINGRAY_CUDA_PROFILE` | diagnostic | Diagnostic (`=1`): use the profiled CUDA forward path (graphs off, which inflates launch-bound phases) and disable GPU argmax. |
| `STINGRAY_DECODE_CUDA_GRAPH` | experimental | `=1` captures the GPU trunk of CUDA hybrid-GDN plain decode into a per-layer graph and replays it; the CPU MoE stays outside the graph. Default off. Falls back to direct launches on any capture failure. |
| `STINGRAY_DECODE_PROFILE` | diagnostic | Diagnostic (`=1`): per-token GPU-trunk versus CPU-MoE time split in the CUDA hybrid-GDN pass (adds a sync after the trunk; negligible perturbation). The trunk:moe ratio shows the decode pole. |
| `STINGRAY_DECODE_REGIONS` | experimental | Diagnostic (`=1`): bracket the CUDA decode regions with synchronisations so the wall-clock is attributable per region; prints every 64 tokens. Adds a few microseconds of overhead. |
| `STINGRAY_DENSE_FFN_GPU_MARGIN_MB` | experimental | Integer MiB, default 64: VRAM safety margin kept free when uploading dense FFN weights to the GPU in the CUDA hybrid-GDN pass. `0` pushes to the limit. |
| `STINGRAY_DIAGNOSTIC_ALLOW_UNSUPPORTED_ARCH` | diagnostic | `=1` lets a GGUF whose architecture is not supported for text generation get past the compatibility check (normally refused with a message). For triage only; the output is not claimed to be correct. |
| `STINGRAY_DISABLE_BATCH_VERIFY` | experimental | `=1` clears the `BatchVerify` speculation setting (default on). |
| `STINGRAY_DISABLE_DUAL_Q8` | experimental | `=1` disables the dual (gate and up) int8 prefill GEMM in the CPU forward pass, leaving the separate path. |
| `STINGRAY_FLASH64_STRIDED_GEMM` | experimental | `=0` turns off the strided GEMM calls in the CPU flash64 prefill attention path (`ForwardPass.Attention.cs`; only used when the head dimension equals the 64 tile). Default on. |
| `STINGRAY_FORCE_CPU_EMBED` | experimental | `=1` forces the CPU-resident embedding path in the CUDA hybrid pass so it can be exercised and benchmarked on a model whose embedding fits in VRAM (issue 218). Default off. |
| `STINGRAY_FORCE_NO_BLAS` | experimental | Diagnostic (`=1`): treat the BLAS library as absent regardless of what is on disk (perf investigation). |
| `STINGRAY_GDN_CHUNKED_PREFILL` | expert | Hybrid-GDN optimized prefill (chunked recurrence + batched projections/FFN). Default **on** for every hybrid model, including MTP models since 2026-10-02 ([ADR-0002](adr-0002-hybrid-gdn-mtp-chunked-prefill.md)). `=0` forces the StrictSequential per-token path (bit-identical to decode, reproducible but not more accurate): the diagnostic control and the rollback switch. |
| `STINGRAY_GDN_DECODE_FAST` | experimental | `=1` enables the faster GDN recurrence kernel for CUDA hybrid-GDN plain decode. Default off; the MTP draft sub-decode and the bit-parity oracle keep the original byte-exact kernel. |
| `STINGRAY_GDN_PREFILL_COMPUTE` | experimental | `=0` reverts CUDA hybrid-GDN prefill to the byte-exact per-token matvec instead of the MMQ/GEMM kernels. Default on (argmax-stable, not byte-exact). |
| `STINGRAY_GDN_RAW_Q8_0` | experimental | `=0` reverts to the F32-dequant weight upload (the byte-exact reference) instead of raw Q8_0 weights with MMQ/dp4a kernels in the CUDA hybrid-GDN pass. Default on (argmax-stable, not byte-exact). |
| `STINGRAY_GEMMA4_PROBE` | diagnostic | Diagnostic: bisect probe for the Gemma 4 Vulkan defect in `GpuForwardPass`. Unset in normal use. |
| `STINGRAY_GRANITE4_DIAG` | diagnostic | `=1` prints a `[GRANITE4-DIAG]` first-token logit summary (top-10, EOG margin) after Granite 4 Vision prefill in the CLI. |
| `STINGRAY_GRANITE4_PREFILL` | diagnostic | Only echoed in the `[GRANITE4-DIAG]` line; the engine-side prefill selector it once drove was removed with the item-04 fix. |
| `STINGRAY_GGML_F16_DOT` | experimental | Opt-in: rounds F16-weight matmul activations to fp16 before dotting, matching ggml's `vec_dot_type=F16` pairing, for bit-parity comparisons against llama.cpp (see `docs/1-correctness/bugstofix.md`). Default off — full-F32-precision activation is more accurate. |
| `STINGRAY_HYBRID_BATCHED_MOE` | experimental | `=0` is the kill-switch for batched CPU MoE prefill in the CUDA hybrid pass. Default on; byte-exact on AVX-512, argmax-stable where the int8 tier auto-enables. |
| `STINGRAY_HYBRID_PREFILL_COMPUTE` | experimental | `=0` reverts CUDA hybrid prefill to the byte-exact per-token matvec for the q/k/v/o matmuls. Default on (argmax-stable, not byte-exact). |
| `STINGRAY_KVARN_BATCHED_PREFILL` | experimental | `=0` reverts KVarN prompts in `CudaForwardPass` to the per-token prefill loop instead of chunked batched prefill. Default on. |
| `STINGRAY_KV_BF16_MIN_TOKENS` | expert | Positive integer, default 1024: the context length at which `STINGRAY_KV_STORE=auto` uses BF16 storage. |
| `STINGRAY_KV_STORE` | expert | KV-cache store layout: `bf16` stores BF16, `auto` switches by length (see `STINGRAY_KV_BF16_MIN_TOKENS`), any other non-empty value means FP32. Unset leaves the backend default (FP32). |
| `STINGRAY_LFM2_MOE_BATCHED_PREFILL` | expert | `=0` makes LFM2-MoE use the exact per-token trunk for prefill instead of the batched one. On by default (about 2.3x faster prefill; with `STINGRAY_PREFILL_ATTN_FLASH64=0` the batched trunk matches token-by-token bit for bit). |
| `STINGRAY_MLA_TRACE` | diagnostic | Temporary: prints per-layer MLA/attention/MoE intermediate sums for ground-truth diffing against llama.cpp (see `docs/1-correctness/bugstofix.md`'s deepseek2 investigation). |
| `STINGRAY_MMQ_SOA` | experimental | `=0` reverts the CUDA forward pass to the interleaved (AoS) weight layout instead of the SoA-repacked Q8_0 weights, which use aligned loads (+10-12% prefill, bit-identical). Default on; the backend routes per repacked handle. |
| `STINGRAY_MOE_BATCHED_PREFILL` | experimental | `=0` is the master switch that forces MoE prompts back onto the per-token sequential trunk instead of the batched MoE prefill FFN (the sequential arm is the parity oracle). Default on. |
| `STINGRAY_MOE_GPU_ROUTER` | experimental | `=0` runs the CUDA hybrid-GDN CPU-MoE prefill router matvec on the CPU instead of as a batched GPU GEMM (raw logits downloaded; softmax and top-k stay on the host; argmax-stable). Default on. |
| `STINGRAY_MOE_PREFILL_Q8` | experimental | `=1` re-enables the approximate Q8-activation path in batched MoE expert matmuls (only effective when the general Q8 prefill path is also on). Off by default because exact batched/per-token parity is the default; for explicit MoE-Q8 investigations. |
| `STINGRAY_MOE_THREADS` | experimental | Positive integer: worker threads for the CUDA hybrid-GDN pass's CPU MoE. Default `Environment.ProcessorCount`. |
| `STINGRAY_MTP_BATCHED_MOE_VERIFY` | experimental | `=0` disables the batched MoE verify for MTP in the CUDA hybrid-GDN pass. Default on. |
| `STINGRAY_MTP_MIN_ACCEPT` | experimental | Break-even acceptance for adaptive MTP, a value in [0, 1], default 0.55. The first `STINGRAY_MTP_PROBE_STEPS` steps probe with MTP; if the measured acceptance is below this, the rest of the call uses plain per-token decode (output is unchanged for greedy). `0` means always MTP. |
| `STINGRAY_MTP_PROBE_STEPS` | diagnostic | Number of probe steps for adaptive MTP (integer of at least 1, default 8). See `STINGRAY_MTP_MIN_ACCEPT`. |
| `STINGRAY_PER_LAYER_HD_PREFILL` | experimental | `=1` forces batched prefill for per-layer-head-dim models (Gemma 4). Default off and must stay off until the batched core supports sliding-window attention, per-layer KV head counts and KV-layer sharing. For exercising and measuring only, not a supported configuration. |
| `STINGRAY_PLE_GPU_DEQUANT` | experimental | `=0` keeps the per-layer-embedding (PLE) batched pre-pass dequant on the CPU in `CudaForwardPass`. Default on, and used only for Q8_0 or Q6_K PLE tensors whose stacked row is a multiple of 256; other types always use the CPU dequant. |
| `STINGRAY_PREFAULT` | experimental | Weight pre-fault of memory-mapped checkpoints. `0` disables; `1` forces; unset runs automatically unless the mapped size exceeds 80% of RAM (or of free RAM for the free-RAM gate), in which case it is skipped and the reason is logged to stderr. |
| `STINGRAY_PREFILL_ATTN_FLASH64` | experimental | `=0` disables the CPU flash64 prefill attention. Default on. It engages only when startPos plus the token count is at least 256, with no per-layer head dim, no sliding window, no ALiBi and no logit softcap. |
| `STINGRAY_PREFILL_ATTN_FLASH64_TILE_JOBS` | experimental | `=1` parallelises the CPU flash64 prefill attention (when the KV-outer path is not used) over per-head tile jobs instead of one job per head that walks the tiles in order. Default off. |
| `STINGRAY_PREFILL_ATTN_REGISTER_VALUES` | experimental | `=0` disables the register-values variant of the CPU prefill attention. Default on; it also needs FMA support, a head dimension that is a multiple of 8, and no sliding window. |
| `STINGRAY_PREFILL_FLASH` | experimental | `=0` reverts CUDA batched-prefill attention from the flash kernel to the scalar kernels. Default on (attention 929 to 411 ms at N=1848; Gemma 4 prefill about 1389 to 2180 t/s). |
| `STINGRAY_PREFILL_FLASH_TC` | experimental | `=0` reverts CUDA flash prefill from the tensor-core kernel to the half2 kernel. Default on, and only when the head dimension is a multiple of 16 (the multi-warp kernel is +27-40% over half2 on Gemma 4 at d=512). |
| `STINGRAY_PREFILL_FLASH_TC1` | experimental | `=1` forces the single-warp tensor-core flash kernel even where the multi-warp one would apply (A/B). Default off. |
| `STINGRAY_PREFILL_GEMM` | experimental | `=0` routes Q8_0 trunk matmuls in CUDA batched prefill through the memory-bound matvec GEMM-N instead of the compute-bound cuBLAS GEMM. Default on; non-Q8_0 weights always use the matvec GEMM-N. |
| `STINGRAY_PREFILL_PROFILE` | diagnostic | Diagnostic (`=1`): accumulate synchronous CPU-MoE wall time versus total Forward wall time, to size the batching opportunity (CUDA passes). |
| `STINGRAY_PREFIX_SCRATCH_TOKENS` | experimental | Positive integer, default 4096: size in tokens of the bounded scratch slot used by the multi-slot prefix cache (`STINGRAY_PREFIX_SLOTS=2`) so short interleaved requests do not evict slot 0's long prefix. Clamped to the model's maximum context. |
| `STINGRAY_PREFIX_SLOTS` | experimental | `=2` opts into the multi-slot prefix cache (issue 212): dense, rewindable forward passes only. Any other value, or unset, keeps the exact single-slot behaviour. Also suppresses SnapKV auto-enable; an explicit SnapKV budget above 0 still wins and disables multi-slot. |
| `STINGRAY_PROBE_IDS` | diagnostic | Diagnostic: comma-separated token ids for `STINGRAY_PROBE_LOGITS`. Default `198,248046,248045,271`. |
| `STINGRAY_PROBE_LOGITS` | diagnostic | Diagnostic (`=1`): per-layer logit probe in the CPU hybrid-GDN pass. Projects each trunk-layer residual through output_norm and lm_head and prints the rank and logit of the ids in `STINGRAY_PROBE_IDS`. |
| `STINGRAY_PROBE_POS` | diagnostic | Diagnostic: limit the logit probe to token position `<n>` (otherwise every position, which is unmanageable). |
| `STINGRAY_PROFILE_DECODE` | diagnostic | Diagnostic (`=1`): temporary decode profiler in the CPU hybrid-GDN pass, split by Gdn, Attention, Moe and GdnRecurrence. Reported by the run command. |
| `STINGRAY_PROFILE_PREFILL` | diagnostic | Diagnostic (`=1`): per-category prefill timers (QkvProj, Attention, OutProj, Ffn, RmsNorm, RoPE, GdnBlock, MoeFfn, GdnRecurrence) and the routed-expert stage timing. |
| `STINGRAY_PROFILE_VULKAN_UPLOADS` | diagnostic | `=1` times the stages of uploading each weight tensor in the Vulkan hybrid Gated-DeltaNet pass (host conversion and GPU upload). Changes timing slightly; do not use such a run for benchmark comparisons. |
| `STINGRAY_Q3K_DEQUANT_GEMM` | experimental | `=0` sends prefill-scale Q3_K routed MoE experts in the CUDA hybrid-GDN pass through the per-token-re-reading GEMM-N instead of the dequant, fp16, cuBLAS GEMM (weights read once, argmax-stable). Default on. |
| `STINGRAY_Q3K_Q8K` | experimental | `1` or `0` (also `true` or `false`) forces the Q3_K int8 (Q8_K activation) routed-expert kernel on or off in the routed-MoE dot kernels of the hybrid passes (CPU hybrid-GDN and the CPU MoE of the CUDA hybrid passes). Unset: on when the model has Q3_K routed experts. |
| `STINGRAY_Q4KX8_CACHE_MB` | experimental | Diffusion quantized-weight cache budget in MiB. A positive number sets it; `0` or a negative number disables the cache; unset or unparsable uses one third of the available memory (`GC.GetGCMemoryInfo().TotalAvailableMemoryBytes`). |
| `STINGRAY_Q4K_Q8K` | experimental | `1` or `0` (also `true` or `false`) forces the Q4_K int8 routed-expert kernel on or off in the routed-MoE dot kernels of the hybrid passes (CPU hybrid-GDN and the CPU MoE of the CUDA hybrid passes). Unset: on only when the model has Q4_K routed experts and AVX-512 is not available (the f32 AVX-512 dot measured about 8% faster on a Q4_K_M model). |
| `STINGRAY_Q4K_SOA` | experimental | `=0` keeps Q4_K trunk weights in the AoS layout in `CudaForwardPass` instead of the SoA repack (+7% decode, +5% prefill, bit-identical; Qwen3-8B 70.0 to 74.7 t/s). Default on, dense models only (MoE Q4_K is not SoA-converted). |
| `STINGRAY_HUNYUAN_GPU` | experimental | `0` keeps HunyuanVideo's double/single blocks on the CPU even when a Vulkan backend is passed (default: GPU blocks, `HunyuanVideoModel.Gpu.cs`). |
| `STINGRAY_Q6K_GEMM` | experimental | `0` sends Q6_K prefill matmuls back to the row-major int8 tier instead of the group-paired `Q6KPrefillGemm` (A/B toggle). |
| `STINGRAY_Q6K_SOA` | experimental | `=0` keeps Q6_K trunk weights in the AoS layout everywhere and also disables the Q6_K decode-MMQ tile. Default on, dense models only. |
| `STINGRAY_Q8_0_Q8K` | experimental | `1` or `0` (also `true` or `false`) forces the Q8_0 int8 routed-expert kernel on or off in the routed-MoE dot kernels of the hybrid passes (CPU hybrid-GDN and the CPU MoE of the CUDA hybrid passes). Unset: on when the model has Q8_0 routed experts. |
| `STINGRAY_RECURRENT_BATCHED_PREFILL` | expert | `=0` restores per-token prefill for models with Mamba-2 or short-convolution layers (Granite 4.0-H, Nemotron-H, LFM2) instead of the default batched prefill, which batches the projections and keeps the conv and scan in token order. |
| `STINGRAY_SNAPKV_RECENCY` | expert | Sibling of `STINGRAY_SNAPKV_BUDGET` (above) — one feature, classified together. |
| `STINGRAY_SNAPKV_WINDOW` | expert | Sibling of `STINGRAY_SNAPKV_BUDGET` (above) — one feature, classified together. |
| `STINGRAY_SPEC_BATCH_VERIFY` | experimental | `=0` forces the sequential verify fallback in `SpeculativeDecoder` (model-draft and prompt-lookup speculation) even when the target supports batch verify (kill-switch, issue #207). Default on. |
| `STINGRAY_SPLIT_DECODE_GROUPED` | experimental | GQA head-sharing in CUDA split-KV decode attention: `0` forces per-head, `1` forces grouped whenever eligible (for bisecting), unset auto-selects (fp32 KV at long context, from 24576 tokens, with an eligible head config). The heuristic is tuned to E4B-class shapes. |
| `STINGRAY_TRACE_DSPARK` | diagnostic | Diagnostic (`=1`): print DSpark accepted/emitted draft counts and the acceptance rate after a run. |
| `STINGRAY_TRACE_LAYERS` | diagnostic | Diagnostic (`=1`): per-layer activation trace in the CUDA hybrid-GDN forward pass. |
| `STINGRAY_TRACE_NORMS` | diagnostic | Diagnostic (`=1`): per-layer residual L2-norm trace in the CPU forward pass. |
| `STINGRAY_TRACE_POS` | diagnostic | Diagnostic: restrict the MoE router trace (`STINGRAY_TRACE_ROUTERS`) to one token position `<n>`. Default -1 (every position). |
| `STINGRAY_TRACE_ROUTERS` | diagnostic | Diagnostic (`=1`): dump the top-k experts chosen for every MoE layer. Combine with `STINGRAY_TRACE_POS` to limit the output on large MoE models. |
| `STINGRAY_TRACE_SNAPSHOT` | diagnostic | Diagnostic (`=1`): log to stderr when a session's canonical-history prefix is not a strict token prefix of the generating prompt. |
| `STINGRAY_TRACE_VRAM` | diagnostic | Diagnostic (`=1`): print free VRAM (MiB) at labelled points while the CUDA forward pass loads. |
| `STINGRAY_VULKAN_BATCHED_PREFILL` | experimental | `=0` forces the byte-identical sequential per-token prefill loop in the Vulkan hybrid-GDN pass. Default on. |
| `STINGRAY_VULKAN_DECODE_ATTN_GQA` | expert | `=0` keeps the original one-key-per-thread split-KV decode attention kernel instead of the default GQA-aware one, on the full-GPU pass and the gpt-oss GPU pass (rollback switch). |
| `STINGRAY_VULKAN_DECODE_PROFILE` | diagnostic | `=1` prints a `[DecodeProfile]` line to stderr for the full-GPU Vulkan pass: dispatches and barriers per token, plus stage timing. |
| `STINGRAY_VULKAN_GDN_CHUNKED_PREFILL` | experimental | `=1` swaps the fused byte-exact GDN recurrence scan for the FlashQLA chunk-parallel scan during Vulkan batched prefill. Default off. Argmax-stable, not byte-exact. Needs a device that fits the ~34 KB shared-memory tile; the end-to-end win is small while the batched trunk is capped at 8 tokens (comment in `VulkanHybridGdnForwardPass.cs`). |
| `STINGRAY_VULKAN_HYBRID_PREFILL_CHUNK` | expert | Positive integer (default 128). Prefill chunk size, in tokens, of the Vulkan hybrid Gated-DeltaNet pass, clamped between the pass's minimum batch chunk and the Vulkan matmul path's per-dispatch maximum. |
| `STINGRAY_VULKAN_MIN_FREE_RAM_MB` | expert | Non-negative integer in MiB (default 2048). Before a mandatory GPU-resident upload the Vulkan hybrid Gated-DeltaNet pass checks that this much memory is still available and aborts with an error otherwise; setting a smaller value lowers the safety margin. |
| `STINGRAY_VULKAN_NO_BATCHED_PREFILL` | experimental | `=1` is a field kill-switch that disables batched prefill in `GpuForwardPass` (Vulkan) without a rebuild. |
| `STINGRAY_VULKAN_NO_FLASH_ATTN` | experimental | `=1` falls back from the Vulkan flash attention to the per-(head,query) batched kernel without a rebuild. |
| `STINGRAY_VULKAN_PREFILL_CHUNK` | experimental | Positive integer: tokens per Vulkan batched-prefill chunk. Default 0 (as many as the active path allows); any value is clamped to that cap. Distinct from the CPU-side `STINGRAY_PREFILL_CHUNK`. |
| `STINGRAY_VULKAN_SPLIT_DECODE_MIN` | experimental | Positive integer, default 256: the position from which Vulkan split-KV decode attention engages. `STINGRAY_VULKAN_SPLIT_DECODE=0` disables the path outright. |

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
| `STINGRAY_SPLIT_DECODE` | experimental | `=0` reverts CUDA decode attention to a single block per head (the A/B baseline). Default on (split-KV). |
| `STINGRAY_VULKAN_SPLIT_DECODE` | experimental | `=0` reverts Vulkan decode attention to the VRAM score-spill path (kill switch). Default on (about 2x decode at 10.5K context). |

## OpenTail.Stingray.Server, OpenTail.Stingray.Server.Host

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_KV_BUDGET_MB` | stable | Mirrors `OpenTailStingrayServerOptions.KvBudgetMb`. |
| `STINGRAY_MAX_CONCURRENT` | stable | Mirrors `OpenTailStingrayServerOptions.MaxConcurrentRequests`. |
| `STINGRAY_MAX_QUEUE` | stable | Mirrors `OpenTailStingrayServerOptions.MaxQueuedRequests`. See the `STINGRAY_MAX_QUEUED_REQUESTS` dead-name note above. |
| `STINGRAY_MODEL` | stable | Mirrors `OpenTailStingrayServerOptions.ModelPath`. |
| `STINGRAY_MODEL_HOME` | stable | Where `stingray setup` stores catalog models (`ModelHome`); default `%LOCALAPPDATA%\stingray\models` / `~/.cache/stingray/models`. |
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
| `STINGRAY_ATTN128_LEGACY` | expert | `=1` routes head dimension 128 back to the older `MultiHeadAttentionTiled128` kernel instead of the newer one (A/B comparison and fallback). |
| `STINGRAY_FORCE_MHA64_F32` | expert | `=1` forces the head-dimension-64 attention kernel to the FP32 variant even when the device has shader float16/int8 and 16-bit storage (A/B comparison and fallback). |
| `STINGRAY_VK_MATVEC_V2` | expert | `=0` restores the original one-byte-per-lane Q8_0 and Q4_0 mat-vec kernels instead of the default newer ones (rollback switch). |
| `STINGRAY_VK_MXFP4_V2` | expert | `=0` selects the original MXFP4 mat-vec shader instead of the default V2 shader (rollback switch). |
| `STINGRAY_VK_PATH3_CHUNK` | expert | Integer from 16 to 1024 (default 256). The prefill chunk, in tokens, that Path 3 of the tiled quantized GEMM handles per dispatch; bounded by scratch memory. Out-of-range or unparseable values fall back to 256. |
| `STINGRAY_VK_QGEMM_SMALLM` | expert | Integer (default 32). The largest number of tokens routed to the small-M quantized GEMM (16-row by 64-weight-row tiles); `0` disables that kernel. |
| `STINGRAY_VULKAN_DP4A` | experimental | `=0` forces the hand-written Q4_K int8 matvec; `=1` forces the dp4a intrinsic variant and skips the probe. Unset: the backend runs both variants on a tiny problem and uses the intrinsic only if they agree exactly (some drivers corrupt it). |
| `STINGRAY_VULKAN_FLASH_PREFILL` | expert | `=0` disables the grouped-query flash-attention prefill kernel (`AttentionFlashPrefillGqa`) so batched prefill uses the older attention kernel (rollback switch). |
| `STINGRAY_VULKAN_MM_PATH` | experimental | Vulkan quantized batched matmul path. Unset or `3` or `path3`: Path 3 (default). `1` or `path1`: Path 1. `2` or `path2`: Path 2. Any other value falls back to Path 1. |
| `STINGRAY_VULKAN_MM_STATS` | diagnostic | Diagnostic (`=1`): count what the Vulkan quantized batched matmul does (notably tokens per weight-matrix read). Default off so the default path is unchanged. |
| `STINGRAY_VULKAN_PATH2` | experimental | Gates the Vulkan Path 2 cooperative-tiled-GEMM kernel (`MatMulTiledQ4K`) in `VulkanPath2Dispatcher.ReadFromEnvironment`. Defaults off. **Consolidated 2026-09-02**: was two aliases for this one flag (`STINGRAY_VULKAN_PATH2_EXPERIMENTAL`) — collapsed to this single canonical name, the other removed from the registry and no longer read. |
| `STINGRAY_VULKAN_UMA_FRACTION` | experimental | Fraction of the reported UMA heap size used for weight and KV placement, accepted in 0.05 to 2.0, default 0.5. Values above 1.0 are experimental: they ask for more than the advertised heap, and a mandatory upload past what the OS can back may crash. |
| `STINGRAY_VULKAN_VALIDATION` | diagnostic | `=1` or `true` enables the Khronos validation layer and VK_EXT_debug_utils so invalid buffer, descriptor or pipeline use is reported. Default off with no overhead. |

## OpenTail.Stingray.Diffusion

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_ACESTEP_DUMP` | diagnostic | Directory. When set, the ACE-Step pipeline writes its encoder and condition tensors there as raw row-major `.f32` files (for example `encoder_hidden.f32`) so they can be compared with the reference implementation. |
| `STINGRAY_ACESTEP_NOISE` | diagnostic | Path to a raw float32 file that replaces the sampled initial noise in the ACE-Step flow scheduler (parity hook, `DiffusionParityHooks.TryLoadNoise`). The file must hold exactly the latent's number of floats, otherwise the run throws. |
| `STINGRAY_FLUX2_DUMP_COND_PATH` | diagnostic | Path. The FLUX.2 pipeline writes its text-conditioning embeddings there as raw float32 (parity debugging). |
| `STINGRAY_FLUX2_DUMP_LATENT_PATH` | diagnostic | Path. The FLUX.2 pipeline writes the final latent there as raw float32 (parity debugging). |
| `STINGRAY_FLUX2_GPU_BATCH_SIZE` | experimental | Positive integer (default 1). How many FLUX.2 double blocks the GPU path processes per batch; kept at 1 until measured for this workload. |
| `STINGRAY_FLUX2_GPU_FP16` | expert | `=1` stops the FLUX.2 Q4_K weights from being kept as raw quantized blocks on the GPU: they are uploaded in the normal FP16 or FP32 form instead (A/B comparison and fallback). Unset keeps them quantized when the backend can multiply them directly and the column count is a multiple of 256. |
| `STINGRAY_FLUX2_GPU_SINGLE_BLOCKS` | experimental | `=1` also runs the FLUX.2 single blocks on the GPU (the double blocks already run there when the backend supports it). Off by default. |
| `STINGRAY_FLUX2_INJECT_NOISE_PATH` | diagnostic | Path to a raw float32 file that replaces the FLUX.2 initial noise (parity hook, `DiffusionParityHooks.TryLoadNoise`); a size mismatch throws. |
| `STINGRAY_FLUX2_PERF_TRACE` | diagnostic | `=1` accumulates and prints FLUX.2 DiT timing counters (dequantization and matmul time, among others). |
| `STINGRAY_FLUX_GPU_FP16` | expert | `=1` stops FLUX weights from being kept as raw Q3_K, Q4_K or Q5_K blocks on the GPU: they are uploaded in the normal FP16, BF16 or FP32 form instead (A/B comparison and fallback). Unset keeps them quantized when the backend can multiply them directly. |
| `STINGRAY_FLUX_T5_DEVICE` | expert | `cpu` or `gpu`. Forces where the FLUX T5 text encoder runs. By default it runs on the GPU only on a discrete GPU; on an integrated GPU it runs on the CPU, which shares the same RAM. |
| `STINGRAY_HUNYUAN_DUMP_COND_PATH` | diagnostic | Path. HunyuanVideo writes its text-conditioning context there as raw float32 (parity debugging). |
| `STINGRAY_HUNYUAN_DUMP_LATENT` | diagnostic | `=1` prints HunyuanVideo diagnostics to stderr, such as the count of non-finite values in the initial noise, plus further diagnostics from the HunyuanVideo model. |
| `STINGRAY_HUNYUAN_DUMP_LATENT_PATH` | diagnostic | Path. HunyuanVideo writes the final latent there as raw float32 before the VAE decode (parity debugging). |
| `STINGRAY_HUNYUAN_DUMP_VEC_PATH` | diagnostic | Path. HunyuanVideo writes the pooled CLIP-L conditioning vector there as raw float32 when a CLIP-L is loaded (parity debugging). |
| `STINGRAY_HUNYUAN_INJECT_NOISE_PATH` | diagnostic | Path to a raw float32 file that replaces the HunyuanVideo initial noise (parity hook, `DiffusionParityHooks.TryLoadNoise`); a size mismatch throws. |
| `STINGRAY_LTX_DEBUG_STEPFRAMES` | diagnostic | `=1` decodes and saves a PNG after every LTX-Video denoising step (`ltx_stepframe_NN.png`), which makes a run much slower. Directory from `STINGRAY_LTX_DEBUG_STEPFRAMES_DIR`. |
| `STINGRAY_LTX_DEBUG_STEPFRAMES_DIR` | diagnostic | Directory for the per-step PNGs written when `STINGRAY_LTX_DEBUG_STEPFRAMES=1`. Default `docs/diffusion-samples`, relative to the working directory. |
| `STINGRAY_LTX_DEBUG_STEPSTATS` | diagnostic | `=1` prints per-step statistics of the LTX-Video latents and predicted velocity (sums, maximum absolute values). |
| `STINGRAY_PROFILE_VAE` | diagnostic | `=1` prints per-stage timing of the VAE decoder. |
| `STINGRAY_QWENIMAGE_DUMP_COND_PATH` | diagnostic | Path. Qwen Image writes its text-conditioning context there as raw float32 (parity debugging). |
| `STINGRAY_QWENIMAGE_DUMP_LATENT_PATH` | diagnostic | Path. Qwen Image writes the final latent there as raw float32 (parity debugging). |
| `STINGRAY_QWENIMAGE_INJECT_NOISE_PATH` | diagnostic | Path to a raw float32 file that replaces the Qwen Image initial noise (parity hook, `DiffusionParityHooks.TryLoadNoise`); a size mismatch throws. |
| `STINGRAY_QWEN_GPU_BLOCKS` | expert | Integer, clamped to 0 up to the number of blocks. Caps how many Qwen Image DiT blocks are uploaded to stay resident on the GPU (upload also stops early if the device refuses a block, and some headroom is left for runtime buffers); the pass prints how many ended up resident. Unset or unparseable means all blocks are tried. |
| `STINGRAY_QWEN_GPU_FP16` | expert | `=1` stops Qwen Image weights from being kept as raw quantized blocks on the GPU: they are uploaded in the normal FP16, BF16 or FP32 form instead (A/B comparison and fallback). |
| `STINGRAY_QWEN_MLP_CHUNKS` | expert | `=0` keeps the original single 12288-wide Qwen Image MLP instead of the default four chunks (rollback switch). |
| `STINGRAY_QWEN_PROFILE` | diagnostic | `1` logs Qwen Image GPU workspace creation and timing to stderr (`[QwenProfile]`); `ops` also splits time by operation group (modulation, QKV, attention, MLP) by ending and restarting the GPU batch between groups, which perturbs total time, so use it for proportions only. |
| `STINGRAY_SA3_DUMP` | diagnostic | Directory. Stable Audio 3 writes named intermediate tensors there as raw float32 `.f32` files (parity debugging). |
| `STINGRAY_SA3_NOISE` | diagnostic | Path to a raw float32 file that replaces the Stable Audio 3 initial noise (parity hook, `DiffusionParityHooks.TryLoadNoise`); a size mismatch throws. |
| `STINGRAY_SD3_DUMP_LATENT_PATH` | diagnostic | Path. SD3 writes the final latent there as raw float32 (parity debugging). |
| `STINGRAY_SD3_FORCE_GPU_WEIGHTS_F32` | expert | `=1` uploads SD3 weights to the GPU as exact FP32 instead of the backend's preferred FP16 form (precision A/B and fallback; uses more GPU memory). |
| `STINGRAY_SD3_GPU_FP16` | expert | `=1` stops SD3 weights from being kept as raw quantized blocks on the GPU: they are uploaded in the normal FP16, BF16 or FP32 form instead (A/B comparison and fallback). |
| `STINGRAY_SD3_INJECT_NOISE_PATH` | diagnostic | Path to a raw float32 file that replaces the SD3 initial noise (parity hook). Used only if the file exists; a byte-length mismatch throws. |
| `STINGRAY_SD3_POSEMBED_LEFT` | diagnostic | Integer. Overrides the left origin of the SD3 positional-embedding crop, used to investigate a left-shift in generated images. No effect unless set. |
| `STINGRAY_SD3_POSEMBED_TOP` | diagnostic | Integer. Overrides the top origin of the SD3 positional-embedding crop, used to investigate a left-shift in generated images. No effect unless set. |
| `STINGRAY_T5_DUMP_PREFIX` | diagnostic | File-name prefix. The T5 text encoder writes diagnostic checkpoint tensors to `<prefix>_<checkpoint>_<call number>.bin`, with a call counter that separates the conditional and unconditional passes (parity debugging). |
| `STINGRAY_WAN_DEBUG_ATTNMAP` | diagnostic | `=1` runs the Wan self-attention map inspection (`InspectSelfAttentionMap`) in each self-attention layer and forces the CPU path (the GPU path is not used while it is set). |
| `STINGRAY_WAN_DEBUG_CROSSREF` | diagnostic | `=1` loads reference dumps produced by the stable-diffusion.cpp Wan implementation and compares the Wan model's intermediate tensors with them. The dump directory comes from `STINGRAY_WAN_DEBUG_CROSSREF_DIR`. |
| `STINGRAY_WAN_DEBUG_CROSSREF_DIR` | diagnostic | Directory holding the reference dumps read when `STINGRAY_WAN_DEBUG_CROSSREF=1`. Default `examples/stable-diffusion.cpp`, relative to the working directory. |
| `STINGRAY_WAN_DEBUG_EXACT_GELU` | diagnostic | `=1` uses the exact erf-based GELU in the Wan text-embedding projection instead of the default variant, to test empirically which one the reference model uses. |
| `STINGRAY_WAN_DEBUG_PERBLOCK` | diagnostic | `=1` prints per-block debug output while the Wan transformer blocks run. |
| `STINGRAY_WAN_DEBUG_VELOCITY` | diagnostic | `=1` prints statistics (mean, mean square, maximum absolute value) of the predicted velocity at each Wan denoising step. |
| `STINGRAY_WAN_PROFILE` | diagnostic | `=1` prints Wan DiT stage timings (latent packing, uploads and the rest of the forward pass). |
| `STINGRAY_WAN_SIGN` | experimental | Integer (default -1). The sign applied to the predicted velocity in the Wan latent update (`latent += sign * dt * velocity`). A debugging handle for the sampler direction; leave unset. |
| `STINGRAY_ZIMAGE_DUMP_LATENT` | diagnostic | `=1` prints statistics (minimum, maximum, sum, sum of squares) of the Z-Image latent. |

## OpenTail.Stingray.Server

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_EMBEDDING_MODEL` | stable | Model path served by `/v1/embeddings`. Unset means nothing real is configured and the endpoint answers 404 rather than returning synthetic vectors. A request's `model` field is only a name unless `STINGRAY_ENCODER_REQUEST_PATHS` is on. |
| `STINGRAY_ENABLE_RESOURCE_ADMISSION` | expert | `1` or `true` (case-insensitive) turns on resource-based admission in the multi-model runtime manager when the `EnableResourceAdmission` option is not set. Anything else, or unset, leaves it off. |
| `STINGRAY_ENCODER_REQUEST_PATHS` | expert | `1` or `true` lets a request to `/v1/embeddings` or `/v1/rerank` name a model file path in its `model` field. Off by default because a request path is loaded and kept resident, so any client could otherwise make the server load any GGUF on its disk and probe which paths exist. |
| `STINGRAY_MODEL_RESIDENCY_MODE` | expert | `single` (also `singleslot`, `single-slot`) or `multi` (also `multislot`, `multi-slot`), case-insensitive: whether the multi-model manager keeps one model resident at a time or lets several be resident. Used when the `ModelResidencyMode` option is unset; empty means multi-slot, and any other value throws at startup. |
| `STINGRAY_RERANK_MODEL` | stable | Model path served by `/v1/rerank`. Unset means nothing real is configured and the endpoint answers 404 rather than returning synthetic scores. A request's `model` field is only a name unless `STINGRAY_ENCODER_REQUEST_PATHS` is on. |

## OpenTail.Stingray.Vision

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_GEMMA3_DEBUG` | diagnostic | `=1` prints `[GEMMA3-DBG]` progress lines (patch counts, stage markers) from the Gemma 3 vision encoder to stderr. |
| `STINGRAY_LLAMA4_DEBUG` | diagnostic | `=1` prints `[LLAMA4-DBG]` progress lines (patch counts, stage markers) from the Llama 4 vision encoder to stderr. |
| `STINGRAY_VISION_GEMM` | expert | Selects the batched GEMM paths of the vision ops. `0` disables both the batched matmuls and the batched attention (per-token fallback), `nomm` or `noattn-nomm` disables the batched matmuls, `noattn` or `noattn-nomm` disables the batched attention. Unset keeps both on. For A/B checks. |

## OpenTail.Stingray.Core

| Variable | Class | Notes |
|---|---|---|
| `STINGRAY_PROFILE_TOKENIZE` | diagnostic | `=1` prints a `[TokenizeProfile]` line to stderr for every tokenizer `Encode` call: characters in, tokens out and elapsed milliseconds. |
| `STINGRAY_PROFILE_WEIGHT_READ` | diagnostic | `=1` accumulates and reports where SafeTensors weight reading spends its time (tensor lookup, disk read, conversion) and how many bytes and floats were produced. |
