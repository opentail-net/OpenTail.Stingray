# llama.cpp refresh and TensorSharp survey (2026-11-25)

**Question.** Refresh the two reference checkouts under `examples/`, then say what Stingray can reuse from them.

**Companion file.** Which of the newly added llama.cpp architectures fit this host: [2026-11-25-new-llama-cpp-archs-hardware-fit.md](2026-11-25-new-llama-cpp-archs-hardware-fit.md). Earlier reference notes: [2026-11-24-recent-open-weight-models-research.md](2026-11-24-recent-open-weight-models-research.md).

## 1. Checkouts after the refresh

- **`examples/llama.cpp/llama.cpp`** fast-forwarded from `bed0a8566` (1,357 commits behind) to `36d818e16` (origin `master`). Uncommitted local work was kept: `examples/CMakeLists.txt`, `examples/eval-callback/eval-callback.cpp`, the untracked `examples/logit-dump/` and `examples/mtp-check/` directories, and `tools/mtmd/test-1.png`. The restored copies match a backup taken before the update. The backup sits in the session scratchpad, which is not durable.
- **`examples/TensorSharp/TensorSharp`** was deleted and re-cloned from `https://github.com/gbaydin/TensorSharp` at `c5494f2` (origin `master`, nothing behind). That repository is **not** the LLM engine that the takeaways plan describes. It is Alex Weaver's 2016 N-dimensional array library for .NET: Windows x64 only, .NET Framework 4.0, Visual Studio 2015, CPU operations plus CUDA through NVRTC, about 13,700 lines of C#. Nothing in it maps onto Stingray's LLM engine.
- **The LLM engine TensorSharp** (`https://github.com/zhongkaifu/TensorSharp`, last seen at `2cef684f`) was **not** refreshed. Its old checkout, with 34 uncommitted documentation changes including deleted `zh-cn` files, is kept in the scratchpad as `TensorSharp-old-2026-11-25`. Which repository is the intended reference is still open.

## 2. llama.cpp at `36d818e16` compared with `bed0a8566`

- GGUF architecture names: **154 → 181**. Files under `src/models/`: **157 → 188**.
- Architecture names added since the earlier snapshot: mistral-large4, kimi-k31, k2-horizon, zaya1, muse-spark, xing4_0, ricursive, limite, flint, kokoro, moshi, zonos3, tiny-voca, kolibri1, metis16, beam, clef, clover_moe, domynedge, inkling2, inkling2_loop, maion-coder, sensenova_u1, vllm-decision2, gemma-embedding2. None of the new names is in Stingray's descriptor set, except where noted in the companion file.
- Speculation types in `common/speculative.cpp`: NONE, SIMPLE, DRAFT_SIMPLE, NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE, DRAFT_MTP, DRAFT_DSPARK, DRAFT_DFLASH, DRAFT_DFLASH2, DRAFT_EAGLE3, DRAFT_LILICORR.
  - Stingray already has prompt lookup (`PromptLookupDraft`), MTP (`MtpDecoder`) and DSpark (`DSparkDecoder`).
  - Not found in Stingray: DFlash. `MuseGlimmerArchitecture.cs` says its drafter is not ported. DFlash2, EAGLE3, LILICORR, NGRAM_MOD, NGRAM_CACHE and NGRAM_MAP_K are not present as far as a grep shows.
- Quantisation types: Stingray's `DType` already covers every ggml type that llama.cpp lists through `Q2_0` (NVFP4, TQ1_0, TQ2_0, MXFP4, Q1_0, IQ1_M and Q2_0 included). Whether each type has a real CPU kernel was only spot-checked by counting the CPU files that mention it. NVFP4 is in one file.
- Size of the speculation code, measured at `bed0a8566` and **not yet re-measured** at `36d818e16`: `common/speculative.cpp` 3,063 lines, `common/ngram-cache.cpp` 285, `common/ngram-map.cpp` 536, `common/ngram-mod.cpp` 62, `src/models/dflash.cpp` 1,046.

## 3. TensorSharp (LLM engine, `zhongkaifu`) compared with Stingray

Source: its `FEATURES.md` at the old checkout's commit.

- **Already in Stingray:** n-gram prompt lookup, the radix prefix cache, CPU MoE offload, layer split, grammar-constrained JSON, YaRN, Jinja chat templates, Wan video, and a pluggable speculative `spec-type`. Its takeaways plan (`docs/2-coverage/2026-10-02-tensorsharp-takeaways-plan.md`) has no open boxes.
- **Still open:**
  - The **DFlash block drafter**: a separate 5-layer GGUF that proposes a whole speculative window in one forward. Not ported. The drafter file is not on disk.
  - A **`q4_0` KV-cache dtype** (about 0.56 bytes per element). Stingray offers q8_0, bf16, TurboQuant (`kvarn`/`turbo`) and SnapKV, but not q4_0.
  - **Distributed TCP tensor parallelism.** Not useful on one Windows machine.
- **Not relevant to an engine repository:** agent skills, code execution, sub-agents, TensorAgent, and the MLX and Metal backends.

## 4. Open checks

- Re-measure the speculation files at `36d818e16`, and look at the EAGLE3 and DFLASH2 drivers.
- Measure the n-gram variants' acceptance rates on the local Qwen models. This needs no new checkpoint.
- Decide which TensorSharp repository is the intended reference, and refresh that one if it is the LLM engine.
