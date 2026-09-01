# CLI option inventory — generated from source, classification complete

**Generated:** by `scripts/gen-cli-option-inventory.ps1`, which scans `[CommandOption]` /
`[Description]` pairs under `src/OpenTail.Stingray.Cli`. Last regenerated **2026-09-01**, recording
**197 option declarations** across 16 command files — the same count the
`StaticPlanConfigurationTests` guard enforces against source. (Reconciled three rows of drift:
`ImageCommand` gained `--umt5-encoder`/`--umt5-tokenizer` for Wan's real UMT5-XXL text encoder, and
`TtsCommand` lost `--cfg` and gained `--backend` — caught by CI failing
`CliOptionInventory_DeclaredCountMatchesSource`. New rows classified `stable`, matching every other
encoder/tokenizer-path and backend-selection option in this table.)

The tables below are no longer hand-maintained. Regenerate with the script rather than editing rows
by hand; `-Check` exits non-zero when they are stale.

**Generator bug found and fixed 2026-08-15.** The description parser only matched
`[Description("...")]` on a single line, so any multi-line concatenated attribute
(`[Description("..." + "..." + "...")]`, used for the longer option descriptions) silently produced
a blank cell â three rows (`RunCommand`'s `--device`, `--thinking`, `--tq-mode`) had real, detailed
descriptions in source that never made it into this doc. The parser now accumulates lines from the
attribute's opening to its closing `)]` and concatenates every quoted segment, which also correctly
handles the ordinary single-line case. Re-running after the fix reproduced the same 153-row count
with all three descriptions now populated â a pure information gain, no row/count change.

The **Class** column is hand classification, and the generator preserves existing values by command
+ option name, so a regeneration never discards it. **Classification is now complete for every row**
(see below) â this was the outstanding half of Phase 0 deliverable 1 of
`04-quality-of-life-improvements-plan.md`, alongside the same pass on `env-var-inventory.md`.

The declared source count is test-guarded by
`StaticPlanConfigurationTests.CliOptionInventory_DeclaredCountMatchesSource`. That guard detects the
next drift immediately; it does not substitute for re-running the classification pass on any new row
that arrives blank.

Companion to `env-var-inventory.md`; together they are the starting point for the configuration work
in `04-quality-of-life-improvements-plan.md`. A current source-backed inventory is required before
the precedence chain (`CLI pin > profile > host > environment > default`) can be specified.

Class meanings (deliberately prose, not a table â a `` | `class-name` | `` row here would be
indistinguishable from an actual option row to the row-count guard below, which is exactly the
kind of accidental collision that guard exists to catch):

- **stable** â supported user configuration, the ordinary/documented way to use this flag or
  command. Belongs in profiles and effective-config output where the underlying setting is itself
  profile-worthy (RunCommand's generation/serving flags); for the introspection/diagnostic commands
  (Doctor, InspectKv, ListEnv/Metadata/Models/Tensors, ShowTemplate, Status) it means "the normal
  invocation," not "belongs in a generation profile" â those commands aren't part of the profile
  surface at all.
- **expert** â supported but sharp-edged; documented, not surfaced by default; a real operational
  lever for a sophisticated operator.
- **diagnostic** â troubleshooting only. Never written to a profile.
- **experimental** â may change or vanish without notice. Must never become a default. Used here
  specifically for the DSpark flags (parked, not scheduled per `docs/00-current-work.md`) and
  `--n-cpu-moe` (explicitly "DEFERRED / not yet supported" in its own description).

**On the "(llama.cpp compat) Not supported/Not implemented/No effect" flags** (`--batch-size`,
`--flash-attn`, `--main-gpu`, `--mlock`, `--no-mmap`, `--no-warmup`, `--numa`, `--split-mode`,
`--tensor-split`, `--ubatch-size`): classified `stable`, not `experimental` â the CONTRACT here
(accept the flag, then either warn-and-continue or explain-and-refuse) is itself the deliberate,
permanent, documented behavior, not a placeholder likely to change. `--n-cpu-moe` is the one
exception: its own description calls it "DEFERRED / not yet supported," an explicit statement that
real behavior is still pending, which is what `experimental` means here.

**Aliases share one row.** A declaration like `-m|--model` is ONE option with two spellings, so the
count above is declarations, not distinct user-visible spellings.

**The overlap with the environment surface is the point.** Where a flag and a variable set the same
thing, the precedence between them is exactly what Â§7.3 must pin down â and today it is decided ad
hoc at each read site rather than in one place.

## DoctorCommand

| Option | Class | Description |
|---|---|---|
| `--bundle <PATH>` | diagnostic | Write a redacted support bundle (.zip) for attaching to a bug report |
| `--deep` | diagnostic | Run memory allocation smoke tests and backend pipeline verification |
| `--json` | stable | Write machine-readable JSON to stdout |
| `--model <PATH>` | stable | Optional GGUF to validate structurally |
| `--no-gpu-probe` | diagnostic | Do not initialize CUDA/Vulkan; report GPU checks as not probed |

## EmbedCommand

| Option | Class | Description |
|---|---|---|
| `--dimensions <N>` | stable | Matryoshka representation dimension reduction (e.g. 512, 768, 1536). |
| `--file <PATH>` | stable | Optional path to text file containing input text or lines to embed. |
| `--model <MODEL>` | stable | Embedding model name or GGUF path. Default: text-embedding-3-small. |
| `--no-norm` | stable | Disable unit L2 vector normalization. |
| `--output <PATH>` | stable | Optional output file path to write embedding vectors as JSON. |
| `--pooling <TYPE>` | stable | Sequence pooling strategy: mean (default), cls, or last. |
| `--prompt <TEXT>` | stable | Input text prompt to embed into a dense semantic vector. |

## ImageCommand

| Option | Class | Description |
|---|---|---|
| `--backend` | stable | (Z-Image) Force compute backend: auto (default), cuda, vulkan, cpu |
| `--cfg-scale` | stable | Guidance scale — not used for Z-Image (distilled), 1.0 for FLUX schnell (default: auto) |
| `--clip-l` | stable | (FLUX) Path to CLIP-L encoder safetensors |
| `--clip-tokenizer` | stable | (FLUX) Path to CLIP tokenizer.json |
| `--control-image` |  | Path to ControlNet condition hint image (Canny edge, depth map, openpose, etc.) |
| `--control-net` |  | Path to ControlNet weights (.safetensors or .gguf) |
| `--control-strength` |  | ControlNet conditioning scale/strength (default: 1.0) |
| `--device` | stable | (Z-Image) GPU to use: auto (default), none/cpu, an index (0, 1), or a named device (CUDA0, Vulkan1) |
| `--height` | stable | Output image height in pixels — must be divisible by 16 (default: 512) |
| `--init-image` |  | Path to initial image for image-to-image (img2img) generation |
| `--mask-image` |  | Path to inpainting mask image (white = inpaint region, black = preserve original) |
| `--model` | stable | Path to diffusion model GGUF or safetensors directory (FLUX.1, Z-Image-Turbo, …) |
| `--n-gpu-layers` | stable | (Z-Image) GPU acceleration: -1 = auto (CUDA→Vulkan→CPU, default), 0 = CPU only |
| `--negative-prompt` | stable | Negative prompt — what to avoid in the generated image |
| `--output` | stable | Output PNG file path (default: output.png) |
| `--prompt` | stable | Text prompt describing the image to generate |
| `--qwen-encoder` | stable | (Z-Image) Path to Qwen3-4B GGUF text encoder (from Qwen/Qwen3-4B-GGUF) |
| `--qwen-tokenizer` | stable | (Z-Image) Path to Qwen3 tokenizer.json |
| `--sampler` |  | Diffusion scheduler algorithm: euler (default), euler-a, ddim, dpm2m, dpm2m-karras, lcm |
| `--sd-cli` | expert | Path to sd-cli executable used when --use-sdcpp is set (overrides STINGRAY_SDCPP env var) |
| `--seed` | stable | RNG seed (-1 = random, default: -1) |
| `--steps` | stable | Denoising steps (default: 4 for Z-Image-Turbo, 4 for FLUX schnell, 20 for dev) |
| `--strength` |  | Strength for img2img generation (0.0 to 1.0, default: 0.75). Higher values add more variation from the initial image. |
| `--t5-tokenizer` | stable | (FLUX) Path to T5 tokenizer.json |
| `--t5xxl` | stable | (FLUX) Path to T5-XXL encoder safetensors |
| `--text-encoder` | stable | (sd-cli mode only) Path to LLM-style text encoder GGUF |
| `--umt5-encoder` | stable | (Wan) Path to the real UMT5-XXL text encoder safetensors (converted from Wan-AI/Wan2.1-T2V-1.3B's models_t5_umt5-xxl-enc-bf16.pth) |
| `--umt5-tokenizer` | stable | (Wan) Path to the real UMT5 tokenizer.json (Wan-AI/Wan2.1-T2V-1.3B's google/umt5-xxl/tokenizer.json) |
| `--upscale-blend` | expert | Blend factor for the upscaled result (0.0–1.0). 1.0 = full RRDB (sharpest), <1.0 softens by blending with bicubic. Default 1.0. |
| `--upscaler` | stable | Path to ESRGAN/Real-ESRGAN upscaler weights (.safetensors). Upscales the generated image by ×2 or ×4 before saving. |
| `--use-sdcpp` | expert | Delegate to stable-diffusion.cpp sd-cli instead of native pipeline (for comparison) |
| `--vae` | stable | Path to VAE safetensors file or directory (ae.safetensors or vae/ dir) |
| `--verbose` | stable | Show per-step timing and progress |
| `--video-frames` |  | Number of video frames to generate for Wan video diffusion models (default: 1) |
| `--width` | stable | Output image width in pixels — must be divisible by 16 (default: 512) |

## InspectKvCommand

| Option | Class | Description |
|---|---|---|
| `--json` | stable | Write machine-readable JSON snapshot to stdout |
| `--page-size <TOKENS>` | diagnostic | Tokens per page (default: 32) |
| `--pages <PAGES>` | diagnostic | Simulate total page capacity (default: 65536) |

## ListEnvCommand

| Option | Class | Description |
|---|---|---|
| `--all` | stable | Also list known settings that are NOT set (the full surface) |
| `--json` | stable | Emit machine-readable JSON instead of text |

## ListMetadataCommand

| Option | Class | Description |
|---|---|---|
| `--model` | stable | Path to GGUF model file |

## ListModelsCommand

| Option | Class | Description |
|---|---|---|
| `--deep` | diagnostic | Open each GGUF index to report architecture and tensor count (slower) |
| `--dir <PATH>` | stable | Directory to scan (default: ./models, then the current directory) |

## ListTensorsCommand

| Option | Class | Description |
|---|---|---|
| `--filter` | stable | Case-insensitive substring filter on tensor name |
| `--layer` | stable | Show only tensors for this layer index (matches blk.<N>.*) |
| `--model` | stable | Path to GGUF model file |
| `--summary` | stable | Group tensors by name suffix; show count and total bytes per group |

## PerplexityCommand

| Option | Class | Description |
|---|---|---|
| `--backend` | stable | With -g -1, which GPU backend to score on: 'cuda' or 'vulkan'. Default: CUDA when present, else Vulkan. Needed on machines with both to gate the Vulkan path explicitly. |
| `--batch-chunk-size` | expert | Tokens per Prefill() call in --batched mode (default: 256, matching the engine's STINGRAY_PREFILL_CHUNK default). Smaller chunks exercise more chunk-boundary KV-cache transitions; larger chunks are closer to a single-shot prompt. |
| `--batched` | expert | Score every position through batched ForwardPass.Prefill (docs/cpu-prefill-plan.md §14) instead of token-by-token Forward. Default mode NEVER calls MatMulBatched, so it cannot see STINGRAY_CPU_PREFILL_Q8's effect at all -- this flag is required to actually measure that path's perplexity impact. Not supported with --tq, -g -1, or per-layer-head-dim models (those still fall back to sequential Forward inside PrefillCore); MoE models ARE supported and route through the batched per-expert FFN. Prompts are evaluated in --batch-chunk-size chunks so KV-cache truncation matches real multi-chunk prefill. |
| `--ctx-size` | stable | Number of tokens to evaluate (default: 2048). Clamped to the model context length and the corpus length. |
| `--file` | stable | UTF-8 text file to evaluate (llama.cpp -f/--file). Tokenized raw (no chat template); the first -c tokens are scored. |
| `--model` | stable | Path to GGUF model file |
| `--n-gpu-layers` | stable | Layers on GPU: 0 (default, CPU forward pass) or -1 (full offload — CUDA via CudaForwardPass, else Vulkan via GpuForwardPass). Partial offload is not supported. |
| `--tq` | stable | Enable TurboQuant KV cache compression (same flag as the run command) |
| `--tq-mode` | stable | TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a quality warning), kvarn (issue #180: 4-bit K / 2-bit V, 128-token tiles), or lloydmax (3-bit codebooks; severely degrades quality on QK-norm models such as Qwen3 — issue #432). |
| `--tq-window` | expert | FP32 recent-token window before compression kicks in (default: 256; min 128 for kvarn — one full tile). Also sets the first position-bucket edge of the report, so pass the same value to the fp32 baseline for bucket-comparable numbers. |

## RerankCommand

| Option | Class | Description |
|---|---|---|
| `--document <TEXT>` | stable | Candidate document string (can be specified multiple times). |
| `--file <PATH>` | stable | Optional file containing candidate documents (one per line). |
| `--model <MODEL>` | stable | Reranker model name or GGUF path. Default: bge-reranker-large. |
| `--output <PATH>` | stable | Optional output file path to write reranked results as JSON. |
| `--query <TEXT>` | stable | Search query to rank candidate documents against. |
| `--top-n <N>` | stable | Number of top most relevant documents to return. |

## RunCommand

| Option | Class | Description |
|---|---|---|
| `--allow-unverified-arch` | expert | Attempt a GGUF whose architecture has no validated forward-pass profile. Output correctness is UNVERIFIED: GGUF tensor naming does not establish compatible attention, RoPE, normalization or FFN semantics, so the model may produce plausible but wrong tokens. Without this flag such a model is refused. |
| `--auto` | stable | Automatically resolve execution plan based on hardware and target goal |
| `--backend` | stable | GPU backend: auto, vulkan, cuda. Default: auto (prefers CUDA when -g is set and CUDA is available, otherwise Vulkan). |
| `--batch-size <N>` | stable | (llama.cpp compat) Not supported — OpenTail does not expose a configurable batch size. |
| `--cache-type-k` | expert | KV-cache element type for the CUDA backend: fp32 (default), bf16 (half the KV VRAM → ~2x context), or q8_0 (quarter → ~4x). OpenTail applies one dtype to both K and V, so -ctk and -ctv must agree. Mirrors llama.cpp --cache-type-k/-ctk. Env: STINGRAY_KV_DTYPE. |
| `--cache-type-v` | expert | KV-cache V-cache element type. Must match --kv-type/--cache-type-k/-ctk: OpenTail applies one dtype to both K and V. Mirrors llama.cpp --cache-type-v/-ctv. |
| `--chat-template <TEMPLATE>` | expert | Override the model's built-in chat template with a raw Jinja2 source string. Named shortcuts (chatml, llama3, …) are refused — hand-written approximations degrade output silently. Mirrors llama.cpp's --chat-template. |
| `--cpu-moe` | expert | MoE: keep ALL routed expert weights on the CPU (llama.cpp --cpu-moe). Sets STINGRAY_CPU_MOE=1, overriding the VRAM-fit auto-select; STINGRAY_CPU_MOE=0 in the env still forces on-GPU experts. Alias --cmoe (llama.cpp's single-dash -cmoe isn't representable: Spectre short options must be one character). |
| `--ctx-size` | stable | Context size / max sequence length (0 = model default) |
| `--device` | stable | GPU device to offload to: index (0,1,…), name (CUDA0, Vulkan1), or 'none' for CPU. Default: auto. Single-device only (no multi-GPU split). Mirrors llama.cpp's --device. |
| `--draft-lookup` | expert | Speculative decoding via prompt-lookup (n-gram) drafting — proposes tokens by matching the generated tail against prompt+history; no draft model needed (greedy only, requires --temp 0) |
| `--dspark-min-confidence <P>` | experimental | Floor on the DSpark confidence head's predicted acceptance probability; positions below it are trimmed from the verify batch. Unset resolves via STINGRAY_DSPARK_MIN_CONFIDENCE, then 0 = verify the whole block. |
| `--dspark-model <PATH>` | experimental | Path to a DSpark draft-head model.safetensors (deepseek-ai/DeepSpec, e.g. dspark_qwen3_4b_block7) with its config.json alongside. Enables DSpark block-speculative decoding (greedy only, CPU target for now — PR #413 spec). |
| `--dspark-place <MODE>` | experimental | Where the DSpark draft head runs: auto (default; planner decides from VRAM/RAM headroom), gpu, cpu, off. Unset resolves via STINGRAY_DSPARK_PLACE. An explicit value pins the mode outright, like -g pins the layer split. |
| `--dspark-verify-len <N>` | experimental | Cap on draft tokens verified per DSpark step. Unset resolves via STINGRAY_DSPARK_VERIFY_LEN, then 0 = the confidence scheduler decides (up to the head's block size). |
| `--escape` | stable | Process escape sequences (\\n, \\t, \\r, \\\\) in -p/--prompt. Mirrors llama.cpp's -e/--escape. |
| `--expert-stats` | diagnostic | MoE: write GPU expert-cache (SLRU) hit-rate stats to this file on exit. Env: STINGRAY_EXPERT_STATS. |
| `--explain` | stable | Print full decision trace for the resolved execution plan before starting generation |
| `--file` | stable | Read the prompt from a file (llama.cpp -f/--file). Overrides -p when both are given; useful for prompts longer than the shell's command-line limit. |
| `--flash-attn` | stable | (llama.cpp compat) No effect — attention is already fused in the OpenTail backends. Accepted with a warning. |
| `--frequency-penalty <P>` | stable | Subtract once per prior occurrence from a token's logit (0 = disabled). |
| `--goal <GOAL>` | stable | Optimization goal for execution planning: balanced (default), quality, throughput, long-context, low-memory |
| `--gpu-moe-prefill <BOOL>` | expert | CPU-MoE: run the routed-expert prefill matmuls on the GPU (transient weight upload, like llama.cpp's op-offload) instead of CPU dots. Default ON (#390); pass 'false' to force the CPU MoE prefill. Sets STINGRAY_MOE_GPU_PREFILL. ~+28-67% PREFILL on the CUDA GDN-hybrid CPU-MoE models, with DECODE within noise of the CPU path — the register-in-place pin mode (STINGRAY_MOE_PIN_MODE, default 'register') cudaHostRegisters the expert mmap pages instead of a ~14 GB copy, so no RAM duplicate and no page-cache eviction; a token gate (STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS, default 64) keeps tiny prefills + decode on the CPU path. Argmax-stable (GPU runs the MoE in F32), not bit-identical to CPU. Auto-falls-back to the CPU path if the GPU scratch can't allocate. |
| `--hide-thinking` | stable | Hide reasoning output (the model still reasons; only the answer is shown) |
| `--image <PATH>` | stable | Path to a PNG image for multimodal input (Gemma 4 encoder-free vision). Repeatable for multiple images; reference each with an <image> marker in -p (left-to-right), or omit markers to prepend them. Requires --mmproj and a text prompt (-p). Runs on CPU, CUDA (full + partial offload), and Vulkan (full offload). |
| `--json-schema <SCHEMA>` | stable | JSON schema to constrain the entire response to (https://json-schema.org/), e.g. '{"type":"object","properties":{...},"required":[[...]]}' (llama.cpp -j/--json-schema). Root must be an object schema declaring at least one property; unsupported keywords ($ref, oneOf/anyOf, pattern, minLength/maxLength, minimum/maximum) degrade to unconstrained. Mutually exclusive with --json-schema-file. |
| `--json-schema-file` | stable | File containing a JSON schema to constrain the entire response to (llama.cpp --json-schema-file/-jf; alias --jf since llama.cpp's single-dash -jf isn't representable: Spectre short options must be one character). Mutually exclusive with --json-schema. |
| `--json-schema-ordered` | stable | With --json-schema/--json-schema-file: require properties in declaration order (issue #425) -- optional properties may be skipped but never reordered. Lets a streaming consumer act on an early field before a later, larger one finishes. |
| `--logit-bias <BIAS>` | stable | Additive logit bias for a token. Format: TOKEN_ID+BIAS or TOKEN_ID-BIAS, e.g. '1234+1.5' or '5678-100'. Repeatable. Mirrors llama.cpp's --logit-bias. |
| `--main-gpu <N>` | stable | (llama.cpp compat) Not supported — use --device to select the target GPU. |
| `--max-thinking-tokens` | stable | Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path. |
| `--min-batch-blas` | expert | Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via STINGRAY_MIN_BATCH_BLAS env var. |
| `--min-p` | stable | Min-p sampling (default: 0.05) |
| `--mlock` | stable | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--mmproj` | expert | Path to the multimodal projector GGUF (mmproj-*.gguf). Required with --image. Mirrors llama.cpp's --mmproj. |
| `--model` | stable | Path to GGUF model file |
| `--model-draft` | expert | Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0). Mirrors llama.cpp's --model-draft. |
| `--moe-warmpin` | expert | MoE: also pin the top-N hottest experts per layer into the GPU cache after warmup (default 0 = off; frequency-aware eviction already retains hot experts). Env: STINGRAY_MOE_WARMPIN. |
| `--moe-warmpin-after` | expert | MoE: expert accesses to observe before warm-pinning selects the hot set (default 512). Only used with --moe-warmpin. Env: STINGRAY_MOE_WARMPIN_AFTER. |
| `--n-cpu-moe` | experimental | MoE: keep the routed experts of N layers on the CPU (llama.cpp --n-cpu-moe). DEFERRED / not yet supported — OpenTail.Stingray's expert placement is all-or-nothing (no per-layer split in the engine), so passing any value errors with that rationale. Use --cpu-moe (all on CPU) or omit (auto). |
| `--n-gpu-layers` | stable | Layers on GPU (0=CPU only, -1=all). Mirrors llama.cpp's --n-gpu-layers/--ngl. |
| `--n-predict` | stable | Number of tokens to predict (default: 512) |
| `--no-display-prompt` | stable | Don't echo the prompt |
| `--no-mmap` | stable | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--no-moe-predict-prefetch` | expert | MoE: disable next-layer predictive expert prefetch (Vulkan; on by default). Env: STINGRAY_MOE_PREDICT_PREFETCH=0. |
| `--no-thinking` | stable | Disable reasoning mode (sets enable_thinking=false in the chat template) |
| `--no-warmup` | stable | (llama.cpp compat) No effect — OpenTail has no separate warmup step. Accepted with a warning. |
| `--numa <MODE>` | stable | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--prefill-dequant-cache-mb` | expert | Dequant-once BLAS weight-cache budget in MiB for CPU prefill (issue #189): caches the F32 dequant per projection weight so chunked prefill re-pays no dequant (bit-identical). Auto (env STINGRAY_PREFILL_DEQUANT_MB / fit-25%-RAM) by default; 0 = off, negative = unlimited. CPU only. |
| `--presence-penalty <P>` | stable | Subtract once from logits of tokens already generated (0 = disabled). |
| `--prompt` | stable | Input prompt (default: interactive chat) |
| `--repeat-last-n` | stable | Number of recent tokens the repetition penalty considers (default: 64; 0 = disabled; -1 = full context). Mirrors llama.cpp's --repeat-last-n. |
| `--repeat-penalty` | stable | Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1). Mirrors llama.cpp's --repeat-penalty/--repeat_penalty. |
| `--seed` | stable | RNG seed (-1 = random, default: -1) |
| `--single-turn` | stable | Generate one response and exit |
| `--spec-draft-n-max` | expert | Max draft tokens per MTP step (issue #30 batched verify). Unset resolves via STINGRAY_MTP_DRAFT_N, then defaults to 1 (a 2-token verify batch — the measured optimum). Values > 1 also need snapshot-ring slots: set STINGRAY_MTP_BATCH_MAX >= drafts+1 (default 2; each extra slot costs ~150 MiB VRAM on 27B). Mirrors llama.cpp. |
| `--spec-draft-n-min` | expert | Min draft tokens per MTP step (default: 0). Mirrors llama.cpp. Currently rejected at parse time when > 0 since N=1 is the only supported draft length; issue #37. |
| `--spec-draft-p-min` | expert | Min draft probability for MTP probabilistic accept (default: 1.0 = strict argmax-match, byte-identical to no-MTP baseline). 0.75 mirrors llama.cpp; values in (0, 1) accept drafts whose softmax probability under the verifier meets the threshold even when they aren't argmax (issue #38). |
| `--spec-lookahead` | expert | Number of draft tokens per speculative step with --draft-model (default: 4) |
| `--spec-type` | stable | Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp), dspark (requires --dspark-model). Mirrors llama.cpp. |
| `--split-mode <MODE>` | stable | (llama.cpp compat) Not supported — use --auto or -g <N> for layer placement. |
| `--system-prompt` | stable | System prompt |
| `--temp` | stable | Temperature (0 = greedy, default: 0.7) |
| `--tensor-split <SPLIT>` | stable | (llama.cpp compat) Not supported — OpenTail places layers with --auto or an explicit -g <N>. |
| `--thinking` | stable | Enable reasoning mode (sets enable_thinking=true). Needed for Gemma 4 reasoning finetunes, which default off because stock Gemma 4 instruct models aren't reasoning-trained. |
| `--threads` | expert | CPU worker threads for the SIMD kernels (default: logical processor count, or STINGRAY_CPU_THREADS). Mirrors llama.cpp's -t/--threads. |
| `--tool-grammar` | expert | Constrain tool-call arguments to the --tools JSON Schemas (issue #374): required keys can't be dropped, only declared keys/enum values appear, value shapes match the declared type. Needs --tools and a model family with constraint support (Gemma 4, Qwen/Qwen3-Coder, Llama-3, DeepSeek). Default off → byte-identical to unconstrained decoding. |
| `--tools <PATH>` | stable | Path to a JSON file of OpenAI-format tool definitions ([[{type:"function", function:{name, description, parameters}}, ...]], or a {"tools":[[...]]} wrapper). Advertised to the model via its chat template; on a single-prompt (-p) run the parsed tool calls are printed after generation. |
| `--top-k` | stable | Top-k sampling (0 = disabled, default: 40) |
| `--top-p` | stable | Top-p nucleus sampling (default: 0.95) |
| `--tq` | expert | Enable TurboQuant KV cache compression (reduces KV memory ~4-8x; quantizer picked by --tq-mode) |
| `--tq-mode` | expert | TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a quality warning), kvarn (issue #180: Sinkhorn-normalized asymmetric RTN, 4-bit K / 2-bit V, 128-token tiles; CPU (-g 0, any power-of-2 head dim ≤ 1024) or full-CUDA-offload dense (-g -1, head dim ≤ 256); no SnapKV), or lloydmax (3-bit Lloyd-Max codebooks; severely degrades quality on QK-norm models such as Qwen3 — issue #432). |
| `--ubatch-size <N>` | stable | (llama.cpp compat) Not supported — OpenTail does not expose a configurable micro-batch size. |
| `--verbose-prompt` | diagnostic | Print token IDs before generating |

## ShowTemplateCommand

| Option | Class | Description |
|---|---|---|
| `--model <PATH>` | stable | Path to a GGUF model file |
| `--no-thinking` | stable | Render with enable_thinking = false |
| `--prompt <TEXT>` | stable | Sample user message (default: a short placeholder) |
| `--raw` | stable | Print the raw Jinja template source instead of a rendered sample |
| `--system <TEXT>` | stable | Optional system message to include |

## StaticPlanCommand

| Option | Class | Description |
|---|---|---|
| `--backend <NAME>` | stable | Backend preference: auto, cpu, cuda, or vulkan |
| `--ctx-size <N>` | stable | Context size (0 = planner/model default) |
| `--device <NAME>` | stable | Requested device (none forces CPU; a named/indexed GPU is reported but not selected by inspect) |
| `--explain` | stable | Include the full selected/rejected decision trace in text output |
| `--gpu-layers <N>` | stable | GPU layers: 0 = CPU, -1 = planner-selected, omitted = default 0 |
| `--json` | stable | Write machine-readable JSON to stdout |
| `--kv-type <NAME>` | stable | KV element type: fp32, bf16, or q8_0 |
| `--max-batch <N>` | stable | Requested maximum batch size |
| `--model <PATH>` | stable | Path to a GGUF model file |
| `--no-gpu-probe` | diagnostic | Do not initialize CUDA/Vulkan; report GPU availability as not probed |
| `--print-effective-config` | stable | Print the resolved planning configuration and exit; a model is not required |
| `--print-profile-schema` | stable | Write the strict JSON Schema for --profile and exit; a model is not required |
| `--profile <PATH>` | stable | Optional JSON planning profile; CLI values override profile, environment, then defaults |
| `--save-profile <PATH>` | stable | Write the resolved strict planning profile; may be used without a model |
| `--spec-type <NAME>` | stable | Speculation type: auto, none, or mtp |
| `--target <NAME>` | stable | Eligibility target: cli (default) or server |
| `--tool-grammar <BOOL>` | stable | Whether tool grammar is requested |
| `--tq <BOOL>` | stable | Whether TurboQuant KV compression is requested |
| `--tq-mode <NAME>` | stable | TurboQuant mode: auto, kvarn, or lloydmax |

## StatusCommand

| Option | Class | Description |
|---|---|---|
| `--json` | stable | Write machine-readable JSON snapshot to stdout |
| `--url <URL>` | stable | Server URL (default: http://127.0.0.1:8080) |
| `--watch` | stable | Continuously refresh status every second |

## SttCommand

| Option | Class | Description |
|---|---|---|
| `--input <PATH>` | stable | Input 16kHz WAV audio file path for Speech-to-Text transcription or translation. |
| `--language <LANG>` | stable | Spoken language code (e.g. en, es, fr, de, zh, ja). Default: auto/en. |
| `--model <VARIANT>` | stable | Whisper model architecture preset: tiny (default), base, small, medium, large-v3, or turbo. |
| `--model-file <PATH>` | stable | Path to a whisper.cpp GGML .bin checkpoint with real weights. If omitted, a file matching --model's preset name is searched for under ./models (e.g. ggml-tiny.bin); if none is found, the pipeline runs with untrained placeholder weights and a warning is printed. |
| `--no-timestamps` | stable | Disable timestamp-aligned subtitle segment generation. |
| `--output <PATH>` | stable | Optional output file path to write the transcribed text or subtitle segments. |
| `--task <TASK>` | stable | ASR task: 'transcribe' (default) or 'translate' (translate to English). |
| `--temperature <TEMP>` | stable | Decoding temperature (0.0 for greedy argmax). Default: 0.0. |
| `--vad` | stable | Enable Silero VAD neural speech boundary detection and silence filtering. |

## TtsCommand

| Option | Class | Description |
|---|---|---|
| `--backend <BACKEND>` | stable | Compute backend: auto (default), vulkan, or cpu. |
| `--engine <ENGINE>` | stable | TTS architecture engine: kokoro (default), piper, f5tts, chatterbox, or melo. |
| `--model <PATH>` | stable | Custom model checkpoint path (.gguf, .onnx, or .safetensors). |
| `--nfe <NFE>` | stable | Number of Function Evaluations / ODE solver steps for Flow-Matching DiT (default: 32). |
| `--output <PATH>` | stable | Output destination path (.wav). Default: speech.wav. |
| `--prompt <TEXT>` | stable | Input text to synthesize into speech audio. |
| `--ref-audio <PATH>` | stable | Reference audio path (.wav) for Zero-Shot Voice Cloning (F5-TTS). |
| `--ref-text <TEXT>` | stable | Reference audio transcript text for Zero-Shot Voice Cloning (F5-TTS). |
| `--speed <SPEED>` | stable | Speech generation speed multiplier. Default: 1.0. |
| `--vocab <PATH>` | stable | Custom vocabulary / token file path (F5-TTS vocab.txt). |
| `--voice <VOICE>` | stable | Voice persona style preset (e.g. af_heart, af_bella, resemble_default, EN-US, EN-BR, ZH). Default: af_heart. |
| `--voices-dir <PATH>` | stable | Kokoro voice directory containing .bin / .gguf voice vectors. |

