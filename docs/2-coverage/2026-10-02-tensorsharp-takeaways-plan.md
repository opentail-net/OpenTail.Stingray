# TensorSharp takeaways (2026-10-02)

Source: `examples/TensorSharp/TensorSharp` (BSD 3-Clause; https://github.com/zhongkaifu/TensorSharp).
Any ported code adds its file to the existing TensorSharp section of `THIRD_PARTY_NOTICES.md`.
TensorSharp's published speed numbers come from its GGML (llama.cpp) native backend, so they say
nothing about Stingray's pure-C# paths. Every item below is measured here before it is believed.

**Order (user, 2026-10-02):** 1, 2, 3, then 4.

## 1. Spin-then-park CPU worker pool (performance experiment)

- **Reference:** `TensorSharp.Models/CpuWorkerPool.cs` (~300 lines), `CpuWorkers.cs`,
  `CpuParallelBinding.cs`; its tests are `InferenceWeb.Tests/CpuWorkerPoolTests.cs`.
- **Design:**
  - Workers spin on a generation counter, so submitting a job is one interlocked write, and the
    submitting thread works too.
  - Blocks are claimed with an atomic counter.
  - Each worker spins `TS_CPU_SPIN` = 4096 times, then parks on a monitor; the park happens under
    the lock the submitter pulses, so a wakeup can't be missed.
  - Pool width is half the logical CPUs above 8.
  - Claimed: ~2.8x decode and ~15% prefill over `Parallel.For` per matmul, on a 122-core host.
- **Why it isn't a re-test of something closed:**
  - Stingray's `PersistentThreadPool` experiments (`docs/done/perf-loop-progress.md`, iteration 2,
    `DecodeMatVecDispatchPerfTests`) measured an `AutoResetEvent` OS-wait pool (a kernel wakeup on
    every call) and an unbounded pure-spin pool, on SmolLM2-1.7B shapes.
  - Bounded spin-then-park on **tiny** models was never measured. That is where the remaining decode
    gap is: SmolLM2-135M ~0.55x and Qwen2.5-0.5B ~0.75x of llama.cpp, attributed to per-call
    overhead (`perf-sweep-plan.md`).
- **Plan:**
  - [x] Port behind a switch (e.g. `STINGRAY_CPU_POOL=spin`) and route `SimdKernels`'s matvec
     `Parallel.For` sites through it.
  - [x] Prove the dispatch happened (a call counter).
  - [x] Run an interleaved A/B on 135M / 360M / Qwen2.5-0.5B decode and one 7B; outputs must be
     bit-identical, since only scheduling changes.
  - [x] Keep it only if it is measurably better; record the result either way.
- **Expect less than claimed:** this machine has 8 cores, not 122.

### Result, 2026-10-02: shipped **opt-in** (`STINGRAY_CPU_POOL=spin`)

- **Code:**
  - The pool is `src/OpenTail.Stingray.Cpu/SpinParkWorkerPool.cs`, with `TryFor`: a busy pool
    (another thread's job in flight) makes the caller fall back to `Parallel.For` instead of
    running serialized.
  - Routing is `SimdKernels.KernelFor` / `ParallelForCapped` / `ParallelForUncapped`, used by
    every `MatVec*` kernel, decode attention (`ForwardPass.Attention`, hybrid `Attention` /
    `MtpAttnBlock`), `RowKernels`, the MoE decode sweeps (`ForwardPass.Moe`, hybrid `MoeFfnCore`)
    and Mamba2 heads.
  - Prefill/batched kernels (`MatMulBatchedF32`, `TryMatMulBatchedQ8`, `TryMatMulBatchedDualQ8`,
    the GEMM classes, `MoeBatchedExperts`, chunked GDN/attention) stay on `Parallel.For`.
  - Default (off) behaviour is unchanged: the same caps as before.
- **Tests:** `SpinParkWorkerPoolTests` (blocks run exactly once at every width, nesting,
  exceptions, concurrent submitters, park/wake, dispose, `TryFor` busy, empty ranges, bit identity
  of `MatVec`/`MatVecDual` and concurrent `MatVec` vs `Parallel.For`). ForwardPass.Fast is 771/771
  with the pool off **and** on, and running the suite with the pool on caught a real empty-range
  divide-by-zero.
- **Measurements** (interleaved off/on, Ryzen 5700G, 8 threads, generated text identical every pair):

| Variant | Finding |
|---|---|
| v1: one pool block per row pair | decode **-25..35%** (an interlocked claim per row pair) |
| v2: + attention/row kernels on the pool | still -15..30% |
| v3: coarse blocks (`STINGRAY_CPU_POOL_BLOCKS`, ~4-8 per thread) | decode +14..23% on small models; Mistral-7B tie |
| all CPU loops on the pool | decode +14..24% on every model incl. MoE, but prefill **-9..28%** (starves OpenBLAS threads, inlines nested loops) |
| **shipped: decode loops only** | see the table below |

| Model | Decode default -> spin | llama.cpp tg128 | Ratio | Prefill |
|---|---|---|---|---|
| SmolLM2-135M | 152 -> 193 t/s (+27%) | 293.8 | 0.52x -> 0.66x | neutral |
| SmolLM2-360M | 86 -> 103 t/s (+21%) | 126.4 | 0.68x -> 0.81x | neutral |
| Qwen2.5-0.5B | 69 -> 82 t/s (+19%) | 94.0 | 0.73x -> 0.87x | neutral (629-token prompt) |
| SmolLM2-1.7B | +7..20% | | | mixed |
| Mistral-7B | tie | | | tie |
| OLMoE-1B-7B (MoE) | -5% | | | **-19%** |
| LFM2-8B-A1B (MoE) | tie | | | **-18%** |
| Ornith-9B (hybrid) | -4% | | | -2% |

- **Why opt-in, not default:** a clear win for dense-model decode, but a loss for MoE prompt
  processing. MoE prefill still issues decode-style matvecs (the per-token router) between its
  batched expert loops, so the spinners compete with them. A concurrent multi-user server
  workload has also not been measured.
- **Next, if pursued:**
  - [ ] Evaluate a per-model or per-phase policy (pool for dense decode only).
  - [ ] Check concurrent-server throughput before any default change.
  - [ ] Measure on a many-core host, where TensorSharp measured its gains.
- **Measurement caution recorded:** mid-session this machine's memory-bound throughput dropped
  about 25% for reasons outside the code (a HEAD build showed the same drop). Only same-run
  interleaved pairs count.

## 2. Q8_K reference for the int8 alignment follow-up

- **Context:** ADR-0003's known gap plus the 2026-10-02 row check. Our per-token int8 matvec has
  1.2e-2 relative error per projection, and our int8 prefill is 0.69% further from llama.cpp PPL
  than the per-token path on SmolLM2.
- **Reference:** `TensorSharp.Models/ManagedQuantGemm*.cs` and `ManagedQuantizedOps.cs`, managed
  AVX2/AVX-512 kernels on ggml's Q8_K layout (one scale per 256, per-16 bsums). Every SIMD variant
  produces the same bits as the scalar loop.
- **Also worth copying:** `TS_CPU_QGEMM_VERIFY`, which compares a finished GEMM against the
  per-row path.
- **Use:** a second, readable implementation next to the vendored ggml for the cross-check in
  `perf-sweep-plan.md` 10.2 follow-ups (Q8_KS vs Q8_K, Q3_K batched vs decode format).

### Result, 2026-10-02

The cross-check used the vendored ggml directly rather than TensorSharp's port, because it is the
primary reference. Findings are recorded in ADR-0003 "Known gap":
- both engines' kernels are exact;
- our decode Q8_KS is about 2.5x more precise than ggml's Q8_K on Q4_K;
- Q6_K is bit-identical to ggml;
- our batched Q4_K prefill uses ggml's Q8_K.

TensorSharp's managed Q8_K kernels remain a readable reference if a Q8_KS variant of the repacked
GEMM is built.

## 3. `Q1_0` and the Bonsai2 quant types

**Result, 2026-10-02 (3a):** already supported since the initial release (`DType.Q1_0/Q2_0`,
block tables, `Dequantize`, dedicated `MatVecQ1_0`/`MatVecQ2_0`). Now independently verified:
dequantization is bit-identical to ggml's `to_float` on 786k random values (`harness/ggmlx`). No
local `Q1_0` checkpoint exists, so no end-to-end or speed check.

**Result, 2026-10-03 (3b): ported, not verified.** Bonsai2 PRISM runs on the CPU hybrid-GDN pass
behind `STINGRAY_EXPERIMENTAL_PRISM=1`; without it such files are refused with a "ported but not
verified" message. Design:
- PQ2_0/PTQ1_0 are transcoded losslessly to ggml Q2_0 at load.
- The forward transform (grouped-GDN head reorder first for `ssm_out`) is applied to a copy of each
  listed projection's input; the inverse is applied on embedding rows.
- MoE, MTP-head and tied-embedding PRISM checkpoints are refused (paths the transform doesn't
  cover). CUDA and Vulkan refuse PRISM.

Tests (`BonsaiPrismTests`, 13):
- FWHT vs a dense Sylvester Hadamard matrix, both directions;
- inverse∘forward = identity at width 5120 / block 1024;
- grouped reorder vs an explicit index map;
- PTQ1_0 decode of an independent test encoder;
- transcode-to-Q2_0 bit-exact against the direct decode;
- GGUF type IDs.

Real checkpoint: coherent, correct greedy answers (see the "Ported, not verified" table).
Remaining: the publisher-reference comparison and a fast ternary kernel (0.4 t/s now).

- **Types:** GGML type 41 `Q1_0` (one F16 scale plus 128 one-bit signs per block, 1.125
  bits/weight), and the Bonsai2 publisher types `PQ2_0` (142) and `PTQ1_0` (143).
- **TensorSharp's handling:** it transcodes the Bonsai2 types losslessly to GGML `Q2_0` at load
  (`ModelBase.Bonsai.cs`, `docs/models/bonsai2.md`).
- **Bonsai2 27B** is the `qwen35` architecture Stingray already runs, so this is coverage through
  dequant/matvec kernels alone.
- **Verify:** dequantize against gguf-py / llama.cpp (once it supports the type), then a real
  Bonsai2 checkpoint if one fits on disk.

## 4. Model families: port now, prove later ("ported, not verified")

Working todo with per-family details and checklists: [ported-families-todo.md](ported-families-todo.md).

The user's policy:
- **Port** the families from their references now.
- **Prove** them when there is capacity to check real checkpoints.
- **Don't** "open the floodgates" or advertise them in the meantime.

| Family | GGUF arch | Primary reference | Secondary (TensorSharp) | Notes |
|---|---|---|---|---|
| Qwen 3.8 Flash Next | `qwen4exp` | vendored llama.cpp, if present | `Models/Qwen4Exp` | Size not yet checked |
| GLM-5.x | `glm-dsa`, `glm5next` | llama.cpp `src/models/glm-dsa.cpp` | `Models/GlmDsa` | TensorSharp: llama.cpp is not a valid reference for `glm5next` |
| DeepSeek V4 / V4.1 Flash | `deepseek4`, `deepseek41` | llama.cpp `deepseek4.cpp` | `Models/DeepSeek4` (pure-C# `DeepSeek4CpuExecutor`) | Stingray already has alpha code (plan 058); use TensorSharp's to review it. Too large for this PC **Revised 2026-10-03:** V4 smoke test uses ratio 0 only (CSA unexercised), no llama.cpp comparison; V4.1 is partial (raw-attention trunk, refuses ratios 1/2 and YaRN, no indexer, F32-only Engram) with corrected `moe_intermediate_size` 2304 and indexer heads 32. V4 (98.6 GB) can be paged from disk and checked against b10306; V4.1 (335 GB) does not fit. |
| Muse-Glimmer | `muse-glimmer` | llama.cpp `src/models/muse-glimmer.cpp` | `Models/MuseGlimmer` | |
| DiffusionGemma | - | HF reference | `Models/` (text diffusion) | No llama.cpp run of its output is recorded |
| MiniMax-H3 (video + 32 kHz stereo audio) | - | upstream | `Models/MiniMaxH3` | Diffusion project **Revised 2026-10-03:** foundation only, not a complete port: no Qwen3-VL hidden-state conditioning, no real DiT loader, no VAE encode, no conditioning modes, no upstream parity. |

### Rules for a "ported, not verified" family

1. **Not admitted:** it gets a `// <arch> — NOT admitted` comment block in
   `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` (precedent: `deepseek4`, `deepseek32`),
   saying what was ported, from which reference, and what verification is missing. A user loading
   one gets the normal unsupported-architecture refusal.
2. **Not advertised:** it doesn't appear in `docs/STATUS.md`, the README, `docs/WHAT-YOU-CAN-DO.md`,
   `docs/RUNNING.md`, the model catalog, or any supported-models list.
3. **Internal record:** this file's table, one line per family (status `ported YYYY-MM-DD, not
   verified`, code location, reference used and its commit or file), plus the family's own
   detail doc if it needs one.
4. **Tests:** structural and synthetic tests are fine, but they must not be named or described as
   real-weight verification. `RealWeights` tests that no-op without a checkpoint must say so in
   their name or skip message (CLAUDE.md rule 12).
5. **Promotion to "supported"** follows the normal path: a real checkpoint, an independent
   reference (llama.cpp / `admit-arch`), timed test runs, then the admission and the STATUS row in
   the same pass.

### Ported, not verified (fill in as ported)

| Family | Ported | Code | Reference used | Missing for admission |
|---|---|---|---|---|
| Bonsai2 PRISM (`qwen35` + PQ2_0/PTQ1_0 + `prism.hadamard.*`) | 2026-10-03 | `Cpu/BonsaiQuant.cs`, `Cpu/PrismHadamard.cs`, `Engine/PrismHadamardMetadata.cs`, `HybridGdnForwardPass` (ResolveTensor transcode, FusedMatVec / BatchedProjection / GateUpDual transforms, embedding inverse); gate `STINGRAY_EXPERIMENTAL_PRISM=1` in `ModelCompatibility`; CUDA/Vulkan refuse | TensorSharp `bonsai_quant.cpp`, `BonsaiHadamardMetadata.cs`, `ggml_ops_bonsai.cpp` (BSD-3) + its model card | Publisher-reference comparison (`PrismML-Eng/llama.cpp`, branch `prism`): exact token references, PPL. Sanity so far: `Ternary-Bonsai-2-27B-PTQ1_0` (SHA-256 matches) answers "The capital of France is **Paris**." and gives a correct two-sentence Rayleigh-scattering explanation, greedy. Speed 0.4 t/s (generic Q2_0 kernel), a ternary kernel is the lever |
| Qwen 3.8 Flash Next (`qwen4exp`) | 2026-10-03 | `Engine/Qwen4ExpAlpha.cs`, `Engine/Qwen4ExpTensorSet.cs`, `Engine/Qwen4ExpForwardPass.cs`, `Tests.Core/Qwen4ExpAlphaTests.cs` (9 synthetic/unit tests); `// qwen4exp — NOT admitted` block in `ModelCompatibility.cs` | llama.cpp `src/models/qwen4exp.cpp` (`bed0a8566`) + TensorSharp `Models/Qwen4Exp/` | Real-checkpoint verification (>64 GB RAM hardware required; smallest quant ~72.5 GB); MTP, Vision, GPU paths deferred **Revised 2026-10-03:** routed MoE was missing and is now executed (`Qwen4ExpMoeRoutingTests`, `Qwen4ExpRoutedMoeTests`); still missing: PLE n-gram table, RoPE in the QSA mixer, QSA indexer + K-pool selection. The forward pass refuses real configs (`Qwen4ExpGuardTests`); tests are now 9 + 2 + 1 + 1. The 72.5 GB checkpoint can be paged from disk once those pieces exist. |
| Muse-Glimmer (`muse-glimmer`) | 2026-10-03 | `ModelGraph.cs`, `ForwardPass.cs`, `ForwardPass.Decode.cs`, `Tests.ForwardPass.Fast/MuseGlimmerSyntheticTests.cs` (specification test passing); `// muse-glimmer — NOT admitted` block in `ModelCompatibility.cs` | llama.cpp `src/models/muse-glimmer.cpp` (`bed0a8566`) + TensorSharp `Models/MuseGlimmer/` | Real-checkpoint parity test; vision tower (50L ViT), ATEM tool-calling, DFlash speculative drafter deferred **Revised 2026-10-03:** real-checkpoint greedy parity vs llama.cpp `bed0a8566` done for the text tower on CPU (UD-Q4_K_XL; 3 prompts, token-identical to the compared length except one 0.006-nat near-tie; see the Muse-Glimmer plan). Still missing: admit-arch verdict, timed runs, long-context SWA check, vision, DFlash. The standalone SWA-boundary test was removed (tautological). **Admitted 2026-10-03 (text, CPU only):** `admit-arch` 8/8 exact, a 3,748-token prompt past the real sliding window identical, STATUS row added; prefill about 10x slower than llama.cpp. |
| MiniMax-H3 (video + 32 kHz stereo audio) | 2026-10-03 | `Diffusion/MiniMaxH3/*.cs` (config, dual flow schedulers, continuous AdaLN curve table, packed sequence layout, 3D RoPE, DiT denoiser, 3D video VAE chunked decoder, audio VAE decoder, pipeline), `Tests.Diffusion/MiniMaxH3Tests.cs` (13 unit/pipeline tests passing); not advertised | upstream MiniMax-H3 HF code + TensorSharp `Models/MiniMaxH3/` | Real-checkpoint end-to-end sample generation; unadvertised in CLI/STATUS until verified **Revised 2026-10-03:** foundation only, not a complete port: no Qwen3-VL hidden-state conditioning, no real DiT loader, no VAE encode, no conditioning modes, no upstream parity. |
| GLM-5.x (`glm-dsa`, `glm5next`) | 2026-10-03 | `Engine/GlmDsa*.cs`, `Engine/Glm5Next*.cs`, `Tests.ForwardPass.Fast/GlmDsaSyntheticTests.cs` (4 tests), `Tests.ForwardPass.Fast/Glm5NextSyntheticTests.cs` (3 tests); `// glm-dsa`, `// glm5next — NOT admitted` blocks in `ModelCompatibility.cs` | llama.cpp `src/models/glm-dsa.cpp`, `src/models/glm5-next.cpp` (`bed0a8566`, b10306), HF `Glm5NextForConditionalGeneration`, TensorSharp `Models/GlmDsa/` | Real-checkpoint verification (>236 GiB RAM hardware required); MTP heads, vision tower deferred **Revised 2026-10-03:** smoke tests only (finite logits), no numeric oracle run. `glm5next` mHC token-state bug fixed (regression test added); K-pool sparse selection missing (throws past `indexer.top_k` keys). GLM-5.3 Q2_K_XL is downloading for a paged real-weight run. |
| DiffusionGemma (`diffusion-gemma`) | 2026-10-03 | `Diffusion/DiffusionGemma/*.cs` (config, self-conditioning, sampler, forward pass, pipeline), `Tests.Diffusion/DiffusionGemmaTests.cs` (7 tests passing); `// diffusion-gemma — NOT admitted` block in `ModelCompatibility.cs` | upstream HF reference, vLLM DiffusionGemma, TensorSharp `Models/DiffusionGemma/` | Real-weight checkpoint verification on `unsloth/diffusiongemma-26B-A4B-it-GGUF` (Q4_K_M ~16.8 GB); unadvertised in CLI/STATUS **Revised 2026-10-03: NOT DONE.** The real checkpoint uses the Gemma-4 MoE layer layout the port does not implement (tensor names, fused experts, router scale, parallel norms, layer scales, `self_cond_*` MLP, RoPE, V-norm, prefill attention); the forward pass now refuses it. See the DiffusionGemma plan. |
| DeepSeek V4 / V4.1 Flash (`deepseek4`, `deepseek41`) | 2026-10-03 | `Engine/DeepSeek4*.cs`, `Engine/DeepSeek41*.cs`, `Tests.Core/DeepSeek4AlphaTests.cs` (32 tests), `Tests.Core/DeepSeek41AlphaTests.cs` (7 tests); `// deepseek4`, `// deepseek41 — NOT admitted` blocks in `ModelCompatibility.cs` | llama.cpp `src/models/deepseek4.cpp` (b10306), TensorSharp `Models/DeepSeek4/` (103-case PyTorch oracle) | Real-checkpoint verification (>64 GB RAM hardware required; V4 ~98 GB, V4.1 ~335 GB); DSpark, vision, MTP deferred **Revised 2026-10-03:** V4 smoke test uses ratio 0 only (CSA unexercised), no llama.cpp comparison; V4.1 is partial (raw-attention trunk, refuses ratios 1/2 and YaRN, no indexer, F32-only Engram) with corrected `moe_intermediate_size` 2304 and indexer heads 32. V4 (98.6 GB) can be paged from disk and checked against b10306; V4.1 (335 GB) does not fit. |
| DeepSeek-V3.2 (`deepseek32`) | 2026-10-03 | `Engine/DeepSeek32Alpha.cs`, `Engine/DeepSeek32TensorSet.cs`, `Engine/DeepSeek32ForwardPass.cs`, `Tests.Core/DeepSeek32AlphaTests.cs`; `// deepseek32 — NOT admitted` block in `ModelCompatibility.cs` | llama.cpp `src/models/deepseek32.cpp`; see [058 DeepSeek lineage plan](058-deepseek-full-lineage-implementation-plan.md), Phase 1 | No real-checkpoint run; indexer Hadamard rotation and MTP are unimplemented, and MLA absorption math has not been independently verified. |
