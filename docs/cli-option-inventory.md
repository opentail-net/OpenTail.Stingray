# CLI option inventory — historical snapshot, regeneration required

**Snapshot generated:** 2026-07-26 by scanning `[CommandOption]` attributes in
`src/OpenTail.Stingray.Cli`. It records **94 option declarations**.

**Drift found 2026-08-07:** the same source scan now finds **149** `[CommandOption]`
attributes. The 94 rows below are therefore useful historical classification input, but are
**not a complete current option inventory**. Do not use this document alone to define a public
configuration surface or precedence contract. Regenerate it from source (with a checked-in
generator or test-backed extraction), then classify the new rows before doing either.

The declared source count is test-guarded by
`StaticPlanConfigurationTests.CliOptionInventory_DeclaredCountMatchesSource`. That guard detects
the next drift immediately; it does not substitute for the still-required row-level refresh and
human ownership classification.

Companion to `env-var-inventory.md`; together they are the starting point for the configuration
work in `quality-of-life-improvements-plan.md`. A current source-backed inventory is required
before the precedence chain (`CLI pin > profile > host > environment > default`) can be specified.

**Class is unfilled deliberately**, same reasoning as the environment inventory: it is a judgement
call for whoever owns the behaviour, and inferring it from a flag name is how a diagnostic switch
quietly becomes public API. Use `stable` / `expert` / `diagnostic` / `bench` / `experimental`;
unclassified means `experimental`.

**Aliases share one row.** A declaration like `-m|--model` is ONE option with two spellings, so the
count above is declarations, not distinct user-visible spellings.

**The overlap with the environment surface is the point.** Where a flag and a variable set the same
thing, the precedence between them is exactly what §7.3 must pin down — and today it is decided ad
hoc at each read site rather than in one place.


## ImageCommand

| Option | Class | Description |
|---|---|---|
| `--backend` | | (Z-Image) Force compute backend: auto (default), cuda, vulkan, cpu |
| `--cfg-scale` | | Guidance scale — not used for Z-Image (distilled), 1.0 for FLUX schnell (default: auto) |
| `--clip-l` | | (FLUX) Path to CLIP-L encoder safetensors |
| `--clip-tokenizer` | | (FLUX) Path to CLIP tokenizer.json |
| `--device` | | (Z-Image) GPU to use: auto (default), none/cpu, an index (0, 1), or a named device (CUDA0, Vulkan1) |
| `--negative-prompt` | | Negative prompt — what to avoid in the generated image |
| `--ngl|--n-gpu-layers|--gpu-layers|-g` | | (Z-Image) GPU acceleration: -1 = auto (CUDA→Vulkan→CPU, default), 0 = CPU only |
| `--qwen-encoder` | | (Z-Image) Path to Qwen3-4B GGUF text encoder (from Qwen/Qwen3-4B-GGUF) |
| `--qwen-tokenizer` | | (Z-Image) Path to Qwen3 tokenizer.json |
| `--sd-cli` | | Path to sd-cli executable used when --use-sdcpp is set (overrides STINGRAY_SDCPP env var) |
| `--steps` | | Denoising steps (default: 4 for Z-Image-Turbo, 4 for FLUX schnell, 20 for dev) |
| `--t5-tokenizer` | | (FLUX) Path to T5 tokenizer.json |
| `--t5xxl` | | (FLUX) Path to T5-XXL encoder safetensors |
| `--text-encoder` | | (sd-cli mode only) Path to LLM-style text encoder GGUF |
| `--upscale-blend` | | Blend factor for the upscaled result (0.0–1.0). 1.0 = full RRDB (sharpest), <1.0 softens by blending with bicubic. Default 1.0. |
| `--upscaler` | | Path to ESRGAN/Real-ESRGAN upscaler weights (.safetensors). Upscales the generated image by ×2 or ×4 before saving. |
| `--use-sdcpp` | | Delegate to stable-diffusion.cpp sd-cli instead of native pipeline (for comparison) |
| `--vae` | | Path to VAE safetensors file or directory (ae.safetensors or vae/ dir) |
| `-H|--height` | | Output image height in pixels — must be divisible by 16 (default: 512) |
| `-W|--width` | | Output image width in pixels — must be divisible by 16 (default: 512) |
| `-m|--model` | | Path to diffusion model GGUF or safetensors directory (FLUX.1, Z-Image-Turbo, …) |
| `-o|--output` | | Output PNG file path (default: output.png) |
| `-p|--prompt` | | Text prompt describing the image to generate |
| `-s|--seed` | | RNG seed (-1 = random, default: -1) |
| `-v|--verbose` | | Show per-step timing and progress |

## ListEnvCommand

| Option | Class | Description |
|---|---|---|
| `--all` | | Also list known settings that are NOT set (the full surface) |
| `--json` | | Emit machine-readable JSON instead of text |

## ListMetadataCommand

| Option | Class | Description |
|---|---|---|
| `-m|--model` | | Path to GGUF model file |

## ListTensorsCommand

| Option | Class | Description |
|---|---|---|
| `--filter` | | Case-insensitive substring filter on tensor name |
| `--layer` | | Show only tensors for this layer index (matches blk.<N>.*) |
| `--summary` | | Group tensors by name suffix; show count and total bytes per group |
| `-m|--model` | | Path to GGUF model file |

## PerplexityCommand

| Option | Class | Description |
|---|---|---|
| `--backend` | | With -g -1, which GPU backend to score on: 'cuda' or 'vulkan'. Default: CUDA when present, else Vulkan. Needed on machines with both to gate the Vulkan path explicitly. |
| `--batch-chunk-size` | | Tokens per Prefill() call in --batched mode (default: 256, matching the engine's STINGRAY_PREFILL_CHUNK default). Smaller chunks exercise more chunk-boundary KV-cache transitions; larger chunks are closer to a single-shot prompt. |
| `--batched` | | Score every position through batched ForwardPass.Prefill (docs/cpu-prefill-plan.md §14) instead of token-by-token Forward. Default mode NEVER calls MatMulBatched, so it cannot see STINGRAY_CPU_PREFILL_Q8's effect at all -- this flag is required to actually measure that path's perplexity impact. Not supported with --tq, -g -1, or MoE/per-layer-head-dim models (they don't route through the batched dense path either); prompts are evaluated in --batch-chunk-size chunks so KV-cache truncation matches real multi-chunk prefill. |
| `--ngl|--n-gpu-layers|--gpu-layers|-g` | | Layers on GPU: 0 (default, CPU forward pass) or -1 (full offload — CUDA via CudaForwardPass, else Vulkan via GpuForwardPass). Partial offload is not supported. |
| `--tq` | | Enable TurboQuant KV cache compression (same flag as the run command) |
| `--tq-mode` | | TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a quality warning), kvarn (issue #180: 4-bit K / 2-bit V, 128-token tiles), or lloydmax (3-bit codebooks; severely degrades quality on QK-norm models such as Qwen3 — issue #432). |
| `--tq-window` | | FP32 recent-token window before compression kicks in (default: 256; min 128 for kvarn — one full tile). Also sets the first position-bucket edge of the report, so pass the same value to the fp32 baseline for bucket-comparable numbers. |
| `-c|--ctx-size` | | Number of tokens to evaluate (default: 2048). Clamped to the model context length and the corpus length. |
| `-f|--file` | | UTF-8 text file to evaluate (llama.cpp -f/--file). Tokenized raw (no chat template); the first -c tokens are scored. |
| `-m|--model` | | Path to GGUF model file |

## RunCommand

| Option | Class | Description |
|---|---|---|
| `--backend` | | GPU backend: auto, vulkan, cuda. Default: auto (prefers CUDA when -g is set and CUDA is available, otherwise Vulkan). |
| `--cpu-moe|--cmoe` | | MoE: keep ALL routed expert weights on the CPU (llama.cpp --cpu-moe). Sets STINGRAY_CPU_MOE=1, overriding the VRAM-fit auto-select; STINGRAY_CPU_MOE=0 in the env still forces on-GPU experts. Alias --cmoe (llama.cpp's single-dash -cmoe isn't representable: Spectre short options must be one character). |
| `--device` | | GPU device to offload to: index (0,1,…), name (CUDA0, Vulkan1), or 'none' for CPU.  |
| `--draft-lookup` | | Speculative decoding via prompt-lookup (n-gram) drafting — proposes tokens by matching the generated tail against prompt+history; no draft model needed (greedy only, requires --temp 0) |
| `--dspark-min-confidence <P>` | | Floor on the DSpark confidence head's predicted acceptance probability; positions below it are trimmed from the verify batch. Unset resolves via STINGRAY_DSPARK_MIN_CONFIDENCE, then 0 = verify the whole block. |
| `--dspark-model <PATH>` | | Path to a DSpark draft-head model.safetensors (deepseek-ai/DeepSpec, e.g. dspark_qwen3_4b_block7) with its config.json alongside. Enables DSpark block-speculative decoding (greedy only, CPU target for now — PR #413 spec). |
| `--dspark-place <MODE>` | | Where the DSpark draft head runs: auto (default; planner decides from VRAM/RAM headroom), gpu, cpu, off. Unset resolves via STINGRAY_DSPARK_PLACE. An explicit value pins the mode outright, like -g pins the layer split. |
| `--dspark-verify-len <N>` | | Cap on draft tokens verified per DSpark step. Unset resolves via STINGRAY_DSPARK_VERIFY_LEN, then 0 = the confidence scheduler decides (up to the head's block size). |
| `--expert-stats` | | MoE: write GPU expert-cache (SLRU) hit-rate stats to this file on exit. Env: STINGRAY_EXPERT_STATS. |
| `--gpu-moe-prefill <BOOL>` | | CPU-MoE: run the routed-expert prefill matmuls on the GPU (transient weight upload, like llama.cpp's op-offload) instead of CPU dots. Default ON (#390); pass 'false' to force the CPU MoE prefill. Sets STINGRAY_MOE_GPU_PREFILL. ~+28-67% PREFILL on the CUDA GDN-hybrid CPU-MoE models, with DECODE within noise of the CPU path — the register-in-place pin mode (STINGRAY_MOE_PIN_MODE, default 'register') cudaHostRegisters the expert mmap pages instead of a ~14 GB copy, so no RAM duplicate and no page-cache eviction; a token gate (STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS, default 64) keeps tiny prefills + decode on the CPU path. Argmax-stable (GPU runs the MoE in F32), not bit-identical to CPU. Auto-falls-back to the CPU path if the GPU scratch can't allocate. |
| `--hide-thinking` | | Hide reasoning output (the model still reasons; only the answer is shown) |
| `--image <PATH>` | | Path to a PNG image for multimodal input (Gemma 4 encoder-free vision). Repeatable for multiple images; reference each with an <image> marker in -p (left-to-right), or omit markers to prepend them. Requires --mmproj and a text prompt (-p). Runs on CPU, CUDA (full + partial offload), and Vulkan (full offload). |
| `--json-schema-file|--jf <PATH>` | | File containing a JSON schema to constrain the entire response to (llama.cpp --json-schema-file/-jf; alias --jf since llama.cpp's single-dash -jf isn't representable: Spectre short options must be one character). Mutually exclusive with --json-schema. |
| `--json-schema-ordered` | | With --json-schema/--json-schema-file: require properties in declaration order (issue #425) -- optional properties may be skipped but never reordered. Lets a streaming consumer act on an early field before a later, larger one finishes. |
| `--kv-type` | | KV-cache element type for the CUDA backend: fp32 (default), bf16 (half the KV VRAM → ~2x context), or q8_0 (quarter → ~4x). Like llama.cpp --cache-type-k/v. Env: STINGRAY_KV_DTYPE. |
| `--max-thinking-tokens` | | Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path. |
| `--min-batch-blas` | | Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via STINGRAY_MIN_BATCH_BLAS env var. |
| `--min-p` | | Min-p sampling (default: 0.05) |
| `--mmproj` | | Path to the multimodal projector GGUF (mmproj-*.gguf). Required with --image. Mirrors llama.cpp's --mmproj. |
| `--model-draft|--draft-model` | | Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0). Mirrors llama.cpp's --model-draft. |
| `--moe-warmpin` | | MoE: also pin the top-N hottest experts per layer into the GPU cache after warmup (default 0 = off; frequency-aware eviction already retains hot experts). Env: STINGRAY_MOE_WARMPIN. |
| `--moe-warmpin-after` | | MoE: expert accesses to observe before warm-pinning selects the hot set (default 512). Only used with --moe-warmpin. Env: STINGRAY_MOE_WARMPIN_AFTER. |
| `--n-cpu-moe|--ncmoe <N>` | | MoE: keep the routed experts of N layers on the CPU (llama.cpp --n-cpu-moe). DEFERRED / not yet supported — OpenTail.Stingray's expert placement is all-or-nothing (no per-layer split in the engine), so passing any value errors with that rationale. Use --cpu-moe (all on CPU) or omit (auto). |
| `--ngl|--n-gpu-layers|--gpu-layers|-g` | | Layers on GPU (0=CPU only, -1=all, default: 0). Mirrors llama.cpp's --n-gpu-layers/--ngl. |
| `--no-display-prompt` | | Don't echo the prompt |
| `--no-moe-predict-prefetch` | | MoE: disable next-layer predictive expert prefetch (Vulkan; on by default). Env: STINGRAY_MOE_PREDICT_PREFETCH=0. |
| `--no-thinking` | | Disable reasoning mode (sets enable_thinking=false in the chat template) |
| `--prefill-dequant-cache-mb` | | Dequant-once BLAS weight-cache budget in MiB for CPU prefill (issue #189): caches the F32 dequant per projection weight so chunked prefill re-pays no dequant (bit-identical). Auto (env STINGRAY_PREFILL_DEQUANT_MB / fit-25%-RAM) by default; 0 = off, negative = unlimited. CPU only. |
| `--repeat-penalty|--rep-penalty` | | Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1). Mirrors llama.cpp's --repeat-penalty. |
| `--single-turn` | | Generate one response and exit |
| `--spec-draft-n-max` | | Max draft tokens per MTP step (issue #30 batched verify). Unset resolves via STINGRAY_MTP_DRAFT_N, then defaults to 1 (a 2-token verify batch — the measured optimum). Values > 1 also need snapshot-ring slots: set STINGRAY_MTP_BATCH_MAX >= drafts+1 (default 2; each extra slot costs ~150 MiB VRAM on 27B). Mirrors llama.cpp. |
| `--spec-draft-n-min` | | Min draft tokens per MTP step (default: 0). Mirrors llama.cpp. Currently rejected at parse time when > 0 since N=1 is the only supported draft length; issue #37. |
| `--spec-draft-p-min` | | Min draft probability for MTP probabilistic accept (default: 1.0 = strict argmax-match, byte-identical to no-MTP baseline). 0.75 mirrors llama.cpp; values in (0, 1) accept drafts whose softmax probability under the verifier meets the threshold even when they aren't argmax (issue #38). |
| `--spec-lookahead|--draft-tokens` | | Number of draft tokens per speculative step with --draft-model (default: 4) |
| `--spec-type` | | Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp), dspark (requires --dspark-model). Mirrors llama.cpp. |
| `--system-prompt` | | System prompt |
| `--temp` | | Temperature (0 = greedy, default: 0.7) |
| `--thinking` | | Enable reasoning mode (sets enable_thinking=true). Needed for Gemma 4 reasoning  |
| `--tool-grammar` | | Constrain tool-call arguments to the --tools JSON Schemas (issue #374): required keys can't be dropped, only declared keys/enum values appear, value shapes match the declared type. Needs --tools and a model family with constraint support (Gemma 4, Qwen/Qwen3-Coder, Llama-3, DeepSeek). Default off → byte-identical to unconstrained decoding. |
| `--tools <PATH>` | | Path to a JSON file of OpenAI-format tool definitions ([[{type:\"function\", function:{name, description, parameters}}, ...]], or a {\"tools\":[[...]]} wrapper). Advertised to the model via its chat template; on a single-prompt (-p) run the parsed tool calls are printed after generation. |
| `--top-k` | | Top-k sampling (0 = disabled, default: 40) |
| `--top-p` | | Top-p nucleus sampling (default: 0.95) |
| `--tq` | | Enable TurboQuant KV cache compression (reduces KV memory ~4-8x; quantizer picked by --tq-mode) |
| `--tq-mode` | | TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a  |
| `--verbose-prompt` | | Print token IDs before generating |
| `-c|--ctx-size` | | Context size / max sequence length (0 = model default) |
| `-f|--file` | | Read the prompt from a file (llama.cpp -f/--file). Overrides -p when both are given; useful for prompts longer than the shell's command-line limit. |
| `-j|--json-schema <SCHEMA>` | | JSON schema to constrain the entire response to (https://json-schema.org/), e.g. '{\"type\":\"object\",\"properties\":{...},\"required\":[[...]]}' (llama.cpp -j/--json-schema). Root must be an object schema declaring at least one property; unsupported keywords ($ref, oneOf/anyOf, pattern, minLength/maxLength, minimum/maximum) degrade to unconstrained. Mutually exclusive with --json-schema-file. |
| `-m|--model` | | Path to GGUF model file |
| `-n|--n-predict` | | Number of tokens to predict (default: 512) |
| `-p|--prompt` | | Input prompt (default: interactive chat) |
| `-s|--seed` | | RNG seed (-1 = random, default: -1) |
