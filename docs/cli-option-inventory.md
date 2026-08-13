# CLI option inventory — generated from source

**Generated:** by `scripts/gen-cli-option-inventory.ps1`, which scans `[CommandOption]` /
`[Description]` pairs under `src/OpenTail.Stingray.Cli`. Last regenerated **2026-08-11**, recording
**153 option declarations** across 12 command files — the same count the
`StaticPlanConfigurationTests` guard enforces against source.

The tables below are no longer hand-maintained. They had drifted to 96 rows against 149 declared
options, which is why this document previously carried a "do not use to define the public
configuration surface" warning. Regenerate with the script rather than editing rows by hand;
`-Check` exits non-zero when they are stale.

The **Class** column is deliberately still hand-written — it is the classification work this
inventory exists for — and the generator preserves existing values by command + option name, so a
regeneration never discards it. New rows arrive blank.

The declared source count is test-guarded by
`StaticPlanConfigurationTests.CliOptionInventory_DeclaredCountMatchesSource`. That guard detects
the next drift immediately; it does not substitute for the still-required row-level refresh and
human ownership classification.

Companion to `env-var-inventory.md`; together they are the starting point for the configuration
work in `04-quality-of-life-improvements-plan.md`. A current source-backed inventory is required
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


## DoctorCommand

| Option | Class | Description |
|---|---|---|
| `--bundle <PATH>` |  | Write a redacted support bundle (.zip) for attaching to a bug report |
| `--deep` |  | Run memory allocation smoke tests and backend pipeline verification |
| `--json` |  | Write machine-readable JSON to stdout |
| `--model <PATH>` |  | Optional GGUF to validate structurally |
| `--no-gpu-probe` |  | Do not initialize CUDA/Vulkan; report GPU checks as not probed |

## ImageCommand

| Option | Class | Description |
|---|---|---|
| `--backend` |  | (Z-Image) Force compute backend: auto (default), cuda, vulkan, cpu |
| `--cfg-scale` |  | Guidance scale — not used for Z-Image (distilled), 1.0 for FLUX schnell (default: auto) |
| `--clip-l` |  | (FLUX) Path to CLIP-L encoder safetensors |
| `--clip-tokenizer` |  | (FLUX) Path to CLIP tokenizer.json |
| `--device` |  | (Z-Image) GPU to use: auto (default), none/cpu, an index (0, 1), or a named device (CUDA0, Vulkan1) |
| `--height` |  | Output image height in pixels — must be divisible by 16 (default: 512) |
| `--model` |  | Path to diffusion model GGUF or safetensors directory (FLUX.1, Z-Image-Turbo, …) |
| `--negative-prompt` |  | Negative prompt — what to avoid in the generated image |
| `--n-gpu-layers` |  | (Z-Image) GPU acceleration: -1 = auto (CUDA→Vulkan→CPU, default), 0 = CPU only |
| `--output` |  | Output PNG file path (default: output.png) |
| `--prompt` |  | Text prompt describing the image to generate |
| `--qwen-encoder` |  | (Z-Image) Path to Qwen3-4B GGUF text encoder (from Qwen/Qwen3-4B-GGUF) |
| `--qwen-tokenizer` |  | (Z-Image) Path to Qwen3 tokenizer.json |
| `--sd-cli` |  | Path to sd-cli executable used when --use-sdcpp is set (overrides STINGRAY_SDCPP env var) |
| `--seed` |  | RNG seed (-1 = random, default: -1) |
| `--steps` |  | Denoising steps (default: 4 for Z-Image-Turbo, 4 for FLUX schnell, 20 for dev) |
| `--t5-tokenizer` |  | (FLUX) Path to T5 tokenizer.json |
| `--t5xxl` |  | (FLUX) Path to T5-XXL encoder safetensors |
| `--text-encoder` |  | (sd-cli mode only) Path to LLM-style text encoder GGUF |
| `--upscale-blend` |  | Blend factor for the upscaled result (0.0–1.0). 1.0 = full RRDB (sharpest), <1.0 softens by blending with bicubic. Default 1.0. |
| `--upscaler` |  | Path to ESRGAN/Real-ESRGAN upscaler weights (.safetensors). Upscales the generated image by ×2 or ×4 before saving. |
| `--use-sdcpp` |  | Delegate to stable-diffusion.cpp sd-cli instead of native pipeline (for comparison) |
| `--vae` |  | Path to VAE safetensors file or directory (ae.safetensors or vae/ dir) |
| `--verbose` |  | Show per-step timing and progress |
| `--width` |  | Output image width in pixels — must be divisible by 16 (default: 512) |

## InspectKvCommand

| Option | Class | Description |
|---|---|---|
| `--json` |  | Write machine-readable JSON snapshot to stdout |
| `--pages <PAGES>` |  | Simulate total page capacity (default: 65536) |
| `--page-size <TOKENS>` |  | Tokens per page (default: 32) |

## ListEnvCommand

| Option | Class | Description |
|---|---|---|
| `--all` |  | Also list known settings that are NOT set (the full surface) |
| `--json` |  | Emit machine-readable JSON instead of text |

## ListMetadataCommand

| Option | Class | Description |
|---|---|---|
| `--model` |  | Path to GGUF model file |

## ListModelsCommand

| Option | Class | Description |
|---|---|---|
| `--deep` |  | Open each GGUF index to report architecture and tensor count (slower) |
| `--dir <PATH>` |  | Directory to scan (default: ./models, then the current directory) |

## ListTensorsCommand

| Option | Class | Description |
|---|---|---|
| `--filter` |  | Case-insensitive substring filter on tensor name |
| `--layer` |  | Show only tensors for this layer index (matches blk.<N>.*) |
| `--model` |  | Path to GGUF model file |
| `--summary` |  | Group tensors by name suffix; show count and total bytes per group |

## PerplexityCommand

| Option | Class | Description |
|---|---|---|
| `--backend` |  | With -g -1, which GPU backend to score on: 'cuda' or 'vulkan'. Default: CUDA when present, else Vulkan. Needed on machines with both to gate the Vulkan path explicitly. |
| `--batch-chunk-size` |  | Tokens per Prefill() call in --batched mode (default: 256, matching the engine's STINGRAY_PREFILL_CHUNK default). Smaller chunks exercise more chunk-boundary KV-cache transitions; larger chunks are closer to a single-shot prompt. |
| `--batched` |  | Score every position through batched ForwardPass.Prefill (docs/cpu-prefill-plan.md §14) instead of token-by-token Forward. Default mode NEVER calls MatMulBatched, so it cannot see STINGRAY_CPU_PREFILL_Q8's effect at all -- this flag is required to actually measure that path's perplexity impact. Not supported with --tq, -g -1, or per-layer-head-dim models (those still fall back to sequential Forward inside PrefillCore); MoE models ARE supported and route through the batched per-expert FFN. Prompts are evaluated in --batch-chunk-size chunks so KV-cache truncation matches real multi-chunk prefill. |
| `--ctx-size` |  | Number of tokens to evaluate (default: 2048). Clamped to the model context length and the corpus length. |
| `--file` |  | UTF-8 text file to evaluate (llama.cpp -f/--file). Tokenized raw (no chat template); the first -c tokens are scored. |
| `--model` |  | Path to GGUF model file |
| `--n-gpu-layers` |  | Layers on GPU: 0 (default, CPU forward pass) or -1 (full offload — CUDA via CudaForwardPass, else Vulkan via GpuForwardPass). Partial offload is not supported. |
| `--tq` |  | Enable TurboQuant KV cache compression (same flag as the run command) |
| `--tq-mode` |  | TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a quality warning), kvarn (issue #180: 4-bit K / 2-bit V, 128-token tiles), or lloydmax (3-bit codebooks; severely degrades quality on QK-norm models such as Qwen3 — issue #432). |
| `--tq-window` |  | FP32 recent-token window before compression kicks in (default: 256; min 128 for kvarn — one full tile). Also sets the first position-bucket edge of the report, so pass the same value to the fp32 baseline for bucket-comparable numbers. |

## RunCommand

| Option | Class | Description |
|---|---|---|
| `--allow-unverified-arch` |  | Attempt a GGUF whose architecture has no validated forward-pass profile. Output correctness is UNVERIFIED: GGUF tensor naming does not establish compatible attention, RoPE, normalization or FFN semantics, so the model may produce plausible but wrong tokens. Without this flag such a model is refused. |
| `--auto` |  | Automatically resolve execution plan based on hardware and target goal |
| `--backend` |  | GPU backend: auto, vulkan, cuda. Default: auto (prefers CUDA when -g is set and CUDA is available, otherwise Vulkan). |
| `--batch-size <N>` |  | (llama.cpp compat) Not supported — OpenTail does not expose a configurable batch size. |
| `--cache-type-k` |  | KV-cache element type for the CUDA backend: fp32 (default), bf16 (half the KV VRAM → ~2x context), or q8_0 (quarter → ~4x). OpenTail applies one dtype to both K and V, so -ctk and -ctv must agree. Mirrors llama.cpp --cache-type-k/-ctk. Env: STINGRAY_KV_DTYPE. |
| `--cache-type-v` |  | KV-cache V-cache element type. Must match --kv-type/--cache-type-k/-ctk: OpenTail applies one dtype to both K and V. Mirrors llama.cpp --cache-type-v/-ctv. |
| `--chat-template <TEMPLATE>` |  | Override the model's built-in chat template with a raw Jinja2 source string. Named shortcuts (chatml, llama3, …) are refused — hand-written approximations degrade output silently. Mirrors llama.cpp's --chat-template. |
| `--cpu-moe` |  | MoE: keep ALL routed expert weights on the CPU (llama.cpp --cpu-moe). Sets STINGRAY_CPU_MOE=1, overriding the VRAM-fit auto-select; STINGRAY_CPU_MOE=0 in the env still forces on-GPU experts. Alias --cmoe (llama.cpp's single-dash -cmoe isn't representable: Spectre short options must be one character). |
| `--ctx-size` |  | Context size / max sequence length (0 = model default) |
| `--device` |  |  |
| `--draft-lookup` |  | Speculative decoding via prompt-lookup (n-gram) drafting — proposes tokens by matching the generated tail against prompt+history; no draft model needed (greedy only, requires --temp 0) |
| `--draft-model` |  | Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0). Mirrors llama.cpp's --model-draft. |
| `--dspark-min-confidence <P>` |  | Floor on the DSpark confidence head's predicted acceptance probability; positions below it are trimmed from the verify batch. Unset resolves via STINGRAY_DSPARK_MIN_CONFIDENCE, then 0 = verify the whole block. |
| `--dspark-model <PATH>` |  | Path to a DSpark draft-head model.safetensors (deepseek-ai/DeepSpec, e.g. dspark_qwen3_4b_block7) with its config.json alongside. Enables DSpark block-speculative decoding (greedy only, CPU target for now — PR #413 spec). |
| `--dspark-place <MODE>` |  | Where the DSpark draft head runs: auto (default; planner decides from VRAM/RAM headroom), gpu, cpu, off. Unset resolves via STINGRAY_DSPARK_PLACE. An explicit value pins the mode outright, like -g pins the layer split. |
| `--dspark-verify-len <N>` |  | Cap on draft tokens verified per DSpark step. Unset resolves via STINGRAY_DSPARK_VERIFY_LEN, then 0 = the confidence scheduler decides (up to the head's block size). |
| `--escape` |  | Process escape sequences (\\n, \\t, \\r, \\\\) in -p/--prompt. Mirrors llama.cpp's -e/--escape. |
| `--expert-stats` |  | MoE: write GPU expert-cache (SLRU) hit-rate stats to this file on exit. Env: STINGRAY_EXPERT_STATS. |
| `--explain` |  | Print full decision trace for the resolved execution plan before starting generation |
| `--file` |  | Read the prompt from a file (llama.cpp -f/--file). Overrides -p when both are given; useful for prompts longer than the shell's command-line limit. |
| `--flash-attn` |  | (llama.cpp compat) No effect — attention is already fused in the OpenTail backends. Accepted with a warning. |
| `--frequency-penalty <P>` |  | Subtract once per prior occurrence from a token's logit (0 = disabled). |
| `--goal <GOAL>` |  | Optimization goal for execution planning: balanced (default), quality, throughput, long-context, low-memory |
| `--gpu-moe-prefill <BOOL>` |  | CPU-MoE: run the routed-expert prefill matmuls on the GPU (transient weight upload, like llama.cpp's op-offload) instead of CPU dots. Default ON (#390); pass 'false' to force the CPU MoE prefill. Sets STINGRAY_MOE_GPU_PREFILL. ~+28-67% PREFILL on the CUDA GDN-hybrid CPU-MoE models, with DECODE within noise of the CPU path — the register-in-place pin mode (STINGRAY_MOE_PIN_MODE, default 'register') cudaHostRegisters the expert mmap pages instead of a ~14 GB copy, so no RAM duplicate and no page-cache eviction; a token gate (STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS, default 64) keeps tiny prefills + decode on the CPU path. Argmax-stable (GPU runs the MoE in F32), not bit-identical to CPU. Auto-falls-back to the CPU path if the GPU scratch can't allocate. |
| `--hide-thinking` |  | Hide reasoning output (the model still reasons; only the answer is shown) |
| `--image <PATH>` |  | Path to a PNG image for multimodal input (Gemma 4 encoder-free vision). Repeatable for multiple images; reference each with an <image> marker in -p (left-to-right), or omit markers to prepend them. Requires --mmproj and a text prompt (-p). Runs on CPU, CUDA (full + partial offload), and Vulkan (full offload). |
| `--json-schema <SCHEMA>` |  | JSON schema to constrain the entire response to (https://json-schema.org/), e.g. '{"type":"object","properties":{...},"required":[[...]]}' (llama.cpp -j/--json-schema). Root must be an object schema declaring at least one property; unsupported keywords ($ref, oneOf/anyOf, pattern, minLength/maxLength, minimum/maximum) degrade to unconstrained. Mutually exclusive with --json-schema-file. |
| `--json-schema-file` |  | File containing a JSON schema to constrain the entire response to (llama.cpp --json-schema-file/-jf; alias --jf since llama.cpp's single-dash -jf isn't representable: Spectre short options must be one character). Mutually exclusive with --json-schema. |
| `--json-schema-ordered` |  | With --json-schema/--json-schema-file: require properties in declaration order (issue #425) -- optional properties may be skipped but never reordered. Lets a streaming consumer act on an early field before a later, larger one finishes. |
| `--logit-bias <BIAS>` |  | Additive logit bias for a token. Format: TOKEN_ID+BIAS or TOKEN_ID-BIAS, e.g. '1234+1.5' or '5678-100'. Repeatable. Mirrors llama.cpp's --logit-bias. |
| `--main-gpu <N>` |  | (llama.cpp compat) Not supported — use --device to select the target GPU. |
| `--max-thinking-tokens` |  | Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path. |
| `--min-batch-blas` |  | Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via STINGRAY_MIN_BATCH_BLAS env var. |
| `--min-p` |  | Min-p sampling (default: 0.05) |
| `--mlock` |  | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--mmproj` |  | Path to the multimodal projector GGUF (mmproj-*.gguf). Required with --image. Mirrors llama.cpp's --mmproj. |
| `--model` |  | Path to GGUF model file |
| `--moe-warmpin` |  | MoE: also pin the top-N hottest experts per layer into the GPU cache after warmup (default 0 = off; frequency-aware eviction already retains hot experts). Env: STINGRAY_MOE_WARMPIN. |
| `--moe-warmpin-after` |  | MoE: expert accesses to observe before warm-pinning selects the hot set (default 512). Only used with --moe-warmpin. Env: STINGRAY_MOE_WARMPIN_AFTER. |
| `--ncmoe <N>` |  | MoE: keep the routed experts of N layers on the CPU (llama.cpp --n-cpu-moe). DEFERRED / not yet supported — OpenTail.Stingray's expert placement is all-or-nothing (no per-layer split in the engine), so passing any value errors with that rationale. Use --cpu-moe (all on CPU) or omit (auto). |
| `--n-gpu-layers` |  | Layers on GPU (0=CPU only, -1=all). Mirrors llama.cpp's --n-gpu-layers/--ngl. |
| `--no-display-prompt` |  | Don't echo the prompt |
| `--no-mmap` |  | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--no-moe-predict-prefetch` |  | MoE: disable next-layer predictive expert prefetch (Vulkan; on by default). Env: STINGRAY_MOE_PREDICT_PREFETCH=0. |
| `--no-thinking` |  | Disable reasoning mode (sets enable_thinking=false in the chat template) |
| `--no-warmup` |  | (llama.cpp compat) No effect — OpenTail has no separate warmup step. Accepted with a warning. |
| `--n-predict` |  | Number of tokens to predict (default: 512) |
| `--numa <MODE>` |  | (llama.cpp compat) Not implemented in OpenTail.Stingray. |
| `--prefill-dequant-cache-mb` |  | Dequant-once BLAS weight-cache budget in MiB for CPU prefill (issue #189): caches the F32 dequant per projection weight so chunked prefill re-pays no dequant (bit-identical). Auto (env STINGRAY_PREFILL_DEQUANT_MB / fit-25%-RAM) by default; 0 = off, negative = unlimited. CPU only. |
| `--presence-penalty <P>` |  | Subtract once from logits of tokens already generated (0 = disabled). |
| `--prompt` |  | Input prompt (default: interactive chat) |
| `--repeat_penalty` |  | Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1). Mirrors llama.cpp's --repeat-penalty/--repeat_penalty. |
| `--repeat-last-n` |  | Number of recent tokens the repetition penalty considers (default: 64; 0 = disabled; -1 = full context). Mirrors llama.cpp's --repeat-last-n. |
| `--seed` |  | RNG seed (-1 = random, default: -1) |
| `--single-turn` |  | Generate one response and exit |
| `--spec-draft-n-max` |  | Max draft tokens per MTP step (issue #30 batched verify). Unset resolves via STINGRAY_MTP_DRAFT_N, then defaults to 1 (a 2-token verify batch — the measured optimum). Values > 1 also need snapshot-ring slots: set STINGRAY_MTP_BATCH_MAX >= drafts+1 (default 2; each extra slot costs ~150 MiB VRAM on 27B). Mirrors llama.cpp. |
| `--spec-draft-n-min` |  | Min draft tokens per MTP step (default: 0). Mirrors llama.cpp. Currently rejected at parse time when > 0 since N=1 is the only supported draft length; issue #37. |
| `--spec-draft-p-min` |  | Min draft probability for MTP probabilistic accept (default: 1.0 = strict argmax-match, byte-identical to no-MTP baseline). 0.75 mirrors llama.cpp; values in (0, 1) accept drafts whose softmax probability under the verifier meets the threshold even when they aren't argmax (issue #38). |
| `--spec-lookahead` |  | Number of draft tokens per speculative step with --draft-model (default: 4) |
| `--spec-type` |  | Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp), dspark (requires --dspark-model). Mirrors llama.cpp. |
| `--split-mode <MODE>` |  | (llama.cpp compat) Not supported — use --auto or -g <N> for layer placement. |
| `--system-prompt` |  | System prompt |
| `--temp` |  | Temperature (0 = greedy, default: 0.7) |
| `--tensor-split <SPLIT>` |  | (llama.cpp compat) Not supported — OpenTail places layers with --auto or an explicit -g <N>. |
| `--thinking` |  |  |
| `--threads` |  | CPU worker threads for the SIMD kernels (default: logical processor count, or STINGRAY_CPU_THREADS). Mirrors llama.cpp's -t/--threads. |
| `--tool-grammar` |  | Constrain tool-call arguments to the --tools JSON Schemas (issue #374): required keys can't be dropped, only declared keys/enum values appear, value shapes match the declared type. Needs --tools and a model family with constraint support (Gemma 4, Qwen/Qwen3-Coder, Llama-3, DeepSeek). Default off → byte-identical to unconstrained decoding. |
| `--tools <PATH>` |  | Path to a JSON file of OpenAI-format tool definitions ([[{type:"function", function:{name, description, parameters}}, ...]], or a {"tools":[[...]]} wrapper). Advertised to the model via its chat template; on a single-prompt (-p) run the parsed tool calls are printed after generation. |
| `--top-k` |  | Top-k sampling (0 = disabled, default: 40) |
| `--top-p` |  | Top-p nucleus sampling (default: 0.95) |
| `--tq` |  | Enable TurboQuant KV cache compression (reduces KV memory ~4-8x; quantizer picked by --tq-mode) |
| `--tq-mode` |  |  |
| `--ubatch-size <N>` |  | (llama.cpp compat) Not supported — OpenTail does not expose a configurable micro-batch size. |
| `--verbose-prompt` |  | Print token IDs before generating |

## ShowTemplateCommand

| Option | Class | Description |
|---|---|---|
| `--model <PATH>` |  | Path to a GGUF model file |
| `--no-thinking` |  | Render with enable_thinking = false |
| `--prompt <TEXT>` |  | Sample user message (default: a short placeholder) |
| `--raw` |  | Print the raw Jinja template source instead of a rendered sample |
| `--system <TEXT>` |  | Optional system message to include |

## StaticPlanCommand

| Option | Class | Description |
|---|---|---|
| `--backend <NAME>` |  | Backend preference: auto, cpu, cuda, or vulkan |
| `--ctx-size <N>` |  | Context size (0 = planner/model default) |
| `--device <NAME>` |  | Requested device (none forces CPU; a named/indexed GPU is reported but not selected by inspect) |
| `--explain` |  | Include the full selected/rejected decision trace in text output |
| `--gpu-layers <N>` |  | GPU layers: 0 = CPU, -1 = planner-selected, omitted = default 0 |
| `--json` |  | Write machine-readable JSON to stdout |
| `--kv-type <NAME>` |  | KV element type: fp32, bf16, or q8_0 |
| `--max-batch <N>` |  | Requested maximum batch size |
| `--model <PATH>` |  | Path to a GGUF model file |
| `--no-gpu-probe` |  | Do not initialize CUDA/Vulkan; report GPU availability as not probed |
| `--print-effective-config` |  | Print the resolved planning configuration and exit; a model is not required |
| `--print-profile-schema` |  | Write the strict JSON Schema for --profile and exit; a model is not required |
| `--profile <PATH>` |  | Optional JSON planning profile; CLI values override profile, environment, then defaults |
| `--save-profile <PATH>` |  | Write the resolved strict planning profile; may be used without a model |
| `--spec-type <NAME>` |  | Speculation type: auto, none, or mtp |
| `--target <NAME>` |  | Eligibility target: cli (default) or server |
| `--tool-grammar <BOOL>` |  | Whether tool grammar is requested |
| `--tq <BOOL>` |  | Whether TurboQuant KV compression is requested |
| `--tq-mode <NAME>` |  | TurboQuant mode: auto, kvarn, or lloydmax |

## StatusCommand

| Option | Class | Description |
|---|---|---|
| `--json` |  | Write machine-readable JSON snapshot to stdout |
| `--url <URL>` |  | Server URL (default: http://127.0.0.1:8080) |
| `--watch` |  | Continuously refresh status every second |
