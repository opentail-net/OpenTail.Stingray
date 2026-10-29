namespace OpenTail.Stingray.Cli;

public sealed partial class RunCommand
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("-p|--prompt")]
        [Description("Input prompt (default: interactive chat)")]
        public string? Prompt { get; set; }

        [CommandOption("-f|--file")]
        [Description("Read the prompt from a file (llama.cpp -f/--file). Overrides -p when both are given; useful for prompts longer than the shell's command-line limit.")]
        public string? PromptFile { get; init; }

        [CommandOption("--image <PATH>")]
        [Description("Path to a PNG image for multimodal input (Gemma 4 encoder-free vision). Repeatable for multiple images; reference each with an <image> marker in -p (left-to-right), or omit markers to prepend them. Requires --mmproj and a text prompt (-p). Runs on CPU, CUDA (full + partial offload), and Vulkan (full offload).")]
        public string[]? ImagePaths { get; init; }

        [CommandOption("--mmproj")]
        [Description("Path to the multimodal projector GGUF (mmproj-*.gguf). Required with --image. Mirrors llama.cpp's --mmproj.")]
        public string? MmprojPath { get; set; }

        [CommandOption("-n|--n-predict|-npredict")]
        [Description("Number of tokens to predict (default: 512)")]
        [DefaultValue(512)]
        public int NPredict { get; init; }

        [CommandOption("--temp")]
        [Description("Temperature (0 = greedy, default: 0.7)")]
        [DefaultValue(0.7f)]
        public float Temperature { get; init; }

        [CommandOption("--top-k")]
        [Description("Top-k sampling (0 = disabled, default: 40)")]
        [DefaultValue(40)]
        public int TopK { get; init; }

        [CommandOption("--top-p")]
        [Description("Top-p nucleus sampling (default: 0.95)")]
        [DefaultValue(0.95f)]
        public float TopP { get; init; }

        [CommandOption("--min-p")]
        [Description("Min-p sampling (default: 0.05)")]
        [DefaultValue(0.05f)]
        public float MinP { get; init; }

        [CommandOption("-s|--seed")]
        [Description("RNG seed (-1 = random, default: -1)")]
        [DefaultValue(-1)]
        public int Seed { get; init; }

        [CommandOption("--single-turn")]
        [Description("Generate one response and exit")]
        [DefaultValue(false)]
        public bool SingleTurn { get; init; }

        [CommandOption("--system-prompt")]
        [Description("System prompt")]
        public string? SystemPrompt { get; init; }

        [CommandOption("--no-display-prompt")]
        [Description("Don't echo the prompt")]
        [DefaultValue(false)]
        public bool NoDisplayPrompt { get; init; }

        [CommandOption("--verbose-prompt")]
        [Description("Print token IDs before generating")]
        [DefaultValue(false)]
        public bool VerbosePrompt { get; init; }

        [CommandOption("--ngl|--n-gpu-layers|--gpu-layers|-g|-ngl")]
        [Description("Layers on GPU (0=CPU only, -1=all). Mirrors llama.cpp's --n-gpu-layers/--ngl.")]
        public int? NGpuLayers { get; init; }

        [CommandOption("--device")]
        [Description("GPU device to offload to: index (0,1,…), name (CUDA0, Vulkan1), or 'none' for CPU. " +
            "Default: auto. Single-device only (no multi-GPU split). Mirrors llama.cpp's --device.")]
        public string? Device { get; init; }

        [CommandOption("-c|--ctx-size|-ctx|--n-ctx")]
        [Description("Context size / max sequence length (0 = model default)")]
        [DefaultValue(0)]
        public int CtxSize { get; init; }

        [CommandOption("--tq")]
        [Description("Enable TurboQuant KV cache compression (reduces KV memory ~4-8x; quantizer picked by --tq-mode)")]
        [DefaultValue(false)]
        public bool TurboQuant { get; init; }

        [CommandOption("--tq-mode")]
        [Description("TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a " +
            "quality warning), kvarn (issue #180: Sinkhorn-normalized asymmetric RTN, 4-bit K / 2-bit V, 128-token " +
            "tiles; CPU (-g 0, any power-of-2 head dim ≤ 1024) or full-CUDA-offload dense (-g -1, head dim ≤ 256); " +
            "no SnapKV), or lloydmax (3-bit Lloyd-Max codebooks; severely degrades quality on QK-norm models " +
            "such as Qwen3 — issue #432).")]
        [DefaultValue("auto")]
        public string TqModeStr { get; init; } = "auto";

        [CommandOption("--kv-type|--cache-type-k|-ctk")]
        [Description("KV-cache element type for the CUDA backend: fp32 (default), bf16 (half the KV VRAM → ~2x context), or q8_0 (quarter → ~4x). OpenTail applies one dtype to both K and V, so -ctk and -ctv must agree. Mirrors llama.cpp --cache-type-k/-ctk. Env: STINGRAY_KV_DTYPE.")]
        public string? KvTypeK { get; init; }

        [CommandOption("--cache-type-v|-ctv")]
        [Description("KV-cache V-cache element type. Must match --kv-type/--cache-type-k/-ctk: OpenTail applies one dtype to both K and V. Mirrors llama.cpp --cache-type-v/-ctv.")]
        public string? KvTypeV { get; init; }

        /// <summary>
        /// The single KV dtype to apply to both K and V. Null when unset.
        /// </summary>
        /// <remarks>
        /// llama.cpp lets K and V differ; OpenTail does not. Rather than silently merging two
        /// different requests — which would run happily with a cache the user did not ask for —
        /// <see cref="Validate"/> rejects a disagreement and this property returns the agreed value.
        /// </remarks>
        public string? KvType => KvTypeK ?? KvTypeV;

        [CommandOption("--model-draft|--draft-model|-md")]
        [Description("Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0). Mirrors llama.cpp's --model-draft.")]
        public string? DraftModelPath { get; init; }

        [CommandOption("--spec-lookahead|--draft-tokens|--draft")]
        [Description("Number of draft tokens per speculative step with --draft-model (default: 4)")]
        [DefaultValue(4)]
        public int SpecLookahead { get; init; }

        [CommandOption("--draft-lookup")]
        [Description("Speculative decoding via prompt-lookup (n-gram) drafting — proposes tokens by matching the generated tail against prompt+history; no draft model needed (greedy only, requires --temp 0)")]
        [DefaultValue(false)]
        public bool DraftLookup { get; init; }

        [CommandOption("--spec-type")]
        [Description("Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp), dspark (requires --dspark-model). Mirrors llama.cpp.")]
        [DefaultValue("auto")]
        public string SpecTypeStr { get; init; } = "auto";

        [CommandOption("--dspark-model <PATH>")]
        [Description("Path to a DSpark draft-head model.safetensors (deepseek-ai/DeepSpec, e.g. dspark_qwen3_4b_block7) with its config.json alongside. Enables DSpark block-speculative decoding (greedy only, CPU target for now — PR #413 spec).")]
        public string? DSparkModelPath { get; init; }

        [CommandOption("--dspark-place <MODE>")]
        [Description("Where the DSpark draft head runs: auto (default; planner decides from VRAM/RAM headroom), gpu, cpu, off. Unset resolves via STINGRAY_DSPARK_PLACE. An explicit value pins the mode outright, like -g pins the layer split.")]
        public string? DSparkPlaceStr { get; init; }

        [CommandOption("--dspark-verify-len <N>")]
        [Description("Cap on draft tokens verified per DSpark step. Unset resolves via STINGRAY_DSPARK_VERIFY_LEN, then 0 = the confidence scheduler decides (up to the head's block size).")]
        [DefaultValue(0)]
        public int DSparkVerifyLen { get; init; }

        [CommandOption("--dspark-min-confidence <P>")]
        [Description("Floor on the DSpark confidence head's predicted acceptance probability; positions below it are trimmed from the verify batch. Unset resolves via STINGRAY_DSPARK_MIN_CONFIDENCE, then 0 = verify the whole block.")]
        [DefaultValue(-1f)]
        public float DSparkMinConfidence { get; init; }

        [CommandOption("--spec-draft-n-max")]
        [Description("Max draft tokens per MTP step (issue #30 batched verify). Unset resolves via STINGRAY_MTP_DRAFT_N, then defaults to 1 (a 2-token verify batch — the measured optimum). Values > 1 also need snapshot-ring slots: set STINGRAY_MTP_BATCH_MAX >= drafts+1 (default 2; each extra slot costs ~150 MiB VRAM on 27B). Mirrors llama.cpp.")]
        [DefaultValue(0)]
        public int SpecDraftNMax { get; init; }

        [CommandOption("--spec-draft-n-min")]
        [Description("Min draft tokens per MTP step (default: 0). Mirrors llama.cpp. Currently rejected at parse time when > 0 since N=1 is the only supported draft length; issue #37.")]
        [DefaultValue(0)]
        public int SpecDraftNMin { get; init; }

        [CommandOption("--spec-draft-p-min")]
        [Description("Min draft probability for MTP probabilistic accept (default: 1.0 = strict argmax-match, byte-identical to no-MTP baseline). 0.75 mirrors llama.cpp; values in (0, 1) accept drafts whose softmax probability under the verifier meets the threshold even when they aren't argmax (issue #38).")]
        [DefaultValue(1.0f)]
        public float SpecDraftPMin { get; init; }

        [CommandOption("--min-batch-blas")]
        [Description("Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via STINGRAY_MIN_BATCH_BLAS env var.")]
        [DefaultValue(0)]
        public int MinBatchBlas { get; init; }

        [CommandOption("--prefill-dequant-cache-mb")]
        [Description("Legacy dequant-once BLAS weight-cache setting (issue #189). Currently dormant because the CPU prefill kernels do not use the OpenBLAS route; this option does not change inference math. CPU only.")]
        [DefaultValue(long.MinValue)]
        public long PrefillDequantCacheMb { get; init; }

        [CommandOption("--repeat-penalty|--rep-penalty|--repeat_penalty")]
        [Description("Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1). Mirrors llama.cpp's --repeat-penalty/--repeat_penalty.")]
        [DefaultValue(1.1f)]
        public float RepPenalty { get; init; }

        [CommandOption("--repeat-last-n")]
        [Description("Number of recent tokens the repetition penalty considers (default: 64; 0 = disabled; -1 = full context). Mirrors llama.cpp's --repeat-last-n.")]
        [DefaultValue(64)]
        public int RepeatLastN { get; init; }

        [CommandOption("-t|--threads")]
        [Description("CPU worker threads for the SIMD kernels (default: logical processor count, or STINGRAY_CPU_THREADS). Mirrors llama.cpp's -t/--threads.")]
        [DefaultValue(0)]
        public int Threads { get; init; }

        [CommandOption("-e|--escape")]
        [Description("Process escape sequences (\\n, \\t, \\r, \\\\) in -p/--prompt. Mirrors llama.cpp's -e/--escape.")]
        [DefaultValue(false)]
        public bool Escape { get; init; }

        [CommandOption("--logit-bias <BIAS>")]
        [Description("Additive logit bias for a token. Format: TOKEN_ID+BIAS or TOKEN_ID-BIAS, e.g. '1234+1.5' or '5678-100'. Repeatable. Mirrors llama.cpp's --logit-bias.")]
        public string[]? LogitBias { get; init; }

        [CommandOption("--chat-template <TEMPLATE>")]
        [Description("Override the model's built-in chat template with a raw Jinja2 source string. Named shortcuts (chatml, llama3, …) are refused — hand-written approximations degrade output silently. Mirrors llama.cpp's --chat-template.")]
        public string? ChatTemplateOverride { get; init; }

        [CommandOption("--presence-penalty <P>")]
        [Description("Subtract once from logits of tokens already generated (0 = disabled).")]
        public float? PresencePenalty { get; init; }

        [CommandOption("--frequency-penalty <P>")]
        [Description("Subtract once per prior occurrence from a token's logit (0 = disabled).")]
        public float? FrequencyPenalty { get; init; }

        // ── llama.cpp flags recognised and refused ───────────────────────────
        // Bound so the user gets a message naming the OpenTail alternative rather than
        // "unexpected argument". Never silently accepted: see docs/llamacpp-onramp-plan.md.

        [CommandOption("-ts|--tensor-split <SPLIT>")]
        [Description("(llama.cpp compat) Not supported — OpenTail places layers with --auto or an explicit -g <N>.")]
        public string? TensorSplit { get; init; }

        [CommandOption("-sm|--split-mode <MODE>")]
        [Description("(llama.cpp compat) Not supported — use --auto or -g <N> for layer placement.")]
        public string? SplitMode { get; init; }

        [CommandOption("-mg|--main-gpu <N>")]
        [Description("(llama.cpp compat) Not supported — use --device to select the target GPU.")]
        public string? MainGpu { get; init; }

        [CommandOption("--mlock")]
        [Description("(llama.cpp compat) Not implemented in OpenTail.Stingray.")]
        [DefaultValue(false)]
        public bool Mlock { get; init; }

        [CommandOption("--no-mmap")]
        [Description("(llama.cpp compat) Not implemented in OpenTail.Stingray.")]
        [DefaultValue(false)]
        public bool NoMmap { get; init; }

        [CommandOption("--numa <MODE>")]
        [Description("(llama.cpp compat) Not implemented in OpenTail.Stingray.")]
        public string? Numa { get; init; }

        [CommandOption("-b|--batch-size <N>")]
        [Description("(llama.cpp compat) Not supported — OpenTail does not expose a configurable batch size.")]
        public int? BatchSize { get; init; }

        [CommandOption("-ub|--ubatch-size <N>")]
        [Description("(llama.cpp compat) Not supported — OpenTail does not expose a configurable micro-batch size.")]
        public int? UBatchSize { get; init; }

        // ── llama.cpp flags that are INERT here ───────────────────────────────
        // Accepted and warned about rather than refused: their absence changes nothing about
        // the output, and `-ngl 99 -fa` is close to the most-copied llama.cpp command there is.
        // Erroring on them would make the on-ramp fail at precisely the moment it should work.

        [CommandOption("-fa|--flash-attn")]
        [Description("(llama.cpp compat) No effect — attention is already fused in the OpenTail backends. Accepted with a warning.")]
        [DefaultValue(false)]
        public bool FlashAttn { get; init; }

        [CommandOption("--no-warmup")]
        [Description("(llama.cpp compat) No effect — OpenTail has no separate warmup step. Accepted with a warning.")]
        [DefaultValue(false)]
        public bool NoWarmup { get; init; }

        /// <summary>
        /// Cross-option validation for the llama.cpp compatibility surface.
        /// </summary>
        /// <remarks>
        /// Every flag here is refused because honouring it is impossible, not because it is
        /// unimplemented paperwork: each one would otherwise change placement, memory residency or
        /// sampling in a way the user asked for and would not get. Flags that are merely inert are
        /// warned about in <c>Execute</c> instead — see the INERT block above.
        /// </remarks>
        public override string? Validate()
        {
            if (NPredict < 0)
                return "--n-predict must be zero or greater; OpenTail.Stingray does not support llama.cpp's -1 (until EOS) sentinel.";
            if (KvTypeK is not null && KvTypeV is not null
                && !string.Equals(KvTypeK, KvTypeV, StringComparison.OrdinalIgnoreCase))
                return $"-ctk/--cache-type-k ({KvTypeK}) and -ctv/--cache-type-v ({KvTypeV}) must agree: " +
                       "OpenTail applies one KV dtype to both K and V.";
            if (TensorSplit is not null)
                return "-ts/--tensor-split is not supported; OpenTail places layers automatically with --auto or an explicit -g <N>.";
            if (SplitMode is not null)
                return "-sm/--split-mode is not supported; use --auto or -g <N> for layer placement.";
            if (MainGpu is not null)
                return "-mg/--main-gpu is not supported; use --device to select the target GPU.";
            if (Mlock)
                return "--mlock is not implemented in OpenTail.Stingray.";
            if (NoMmap)
                return "--no-mmap is not implemented in OpenTail.Stingray.";
            if (Numa is not null)
                return "--numa is not implemented in OpenTail.Stingray.";
            if (BatchSize is not null)
                return "-b/--batch-size is not supported: OpenTail.Stingray does not expose a configurable batch size.";
            if (UBatchSize is not null)
                return "-ub/--ubatch-size is not supported: OpenTail.Stingray does not expose a configurable micro-batch size.";
            return null;
        }

        [CommandOption("--ignore-preflight")]
        [Description("Load even if the memory check says the model will not fit this machine (CPU runs only).")]
        public bool IgnorePreflight { get; init; }

        [CommandOption("--backend")]
        [Description("GPU backend: auto, vulkan, cuda. Default: auto (prefers CUDA when -g is set and CUDA is available, otherwise Vulkan).")]
        [DefaultValue("auto")]
        public string Backend { get; init; } = "auto";

        [CommandOption("--no-thinking")]
        [Description("Disable reasoning mode (sets enable_thinking=false in the chat template)")]
        [DefaultValue(false)]
        public bool NoThinking { get; init; }

        [CommandOption("--allow-unverified-arch")]
        [Description("Attempt a GGUF whose architecture has no validated forward-pass profile. Output correctness is UNVERIFIED: GGUF tensor naming does not establish compatible attention, RoPE, normalization or FFN semantics, so the model may produce plausible but wrong tokens. Without this flag such a model is refused.")]
        [DefaultValue(false)]
        public bool AllowUnverifiedArch { get; init; }

        [CommandOption("--thinking")]
        [Description("Enable reasoning mode (sets enable_thinking=true). Needed for Gemma 4 reasoning " +
            "finetunes, which default off because stock Gemma 4 instruct models aren't reasoning-trained.")]
        [DefaultValue(false)]
        public bool Thinking { get; init; }

        [CommandOption("--hide-thinking")]
        [Description("Hide reasoning output (the model still reasons; only the answer is shown)")]
        [DefaultValue(false)]
        public bool HideThinking { get; init; }

        [CommandOption("--max-thinking-tokens")]
        [Description("Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path.")]
        [DefaultValue(0)]
        public int MaxThinkingTokens { get; init; }

        // ── Tool calling ──
        [CommandOption("--tools <PATH>")]
        // Square brackets are Spectre markup delimiters and must be escaped ([[ / ]]) even in
        // help text, or rendering --help throws "Could not find color or style".
        [Description("Path to a JSON file of OpenAI-format tool definitions ([[{type:\"function\", function:{name, description, parameters}}, ...]], or a {\"tools\":[[...]]} wrapper). Advertised to the model via its chat template; on a single-prompt (-p) run the parsed tool calls are printed after generation.")]
        public string? ToolsPath { get; init; }

        [CommandOption("--tool-grammar")]
        [Description("Constrain tool-call arguments to the --tools JSON Schemas (issue #374): required keys can't be dropped, only declared keys/enum values appear, value shapes match the declared type. Needs --tools and a model family with constraint support (Gemma 4, Qwen/Qwen3-Coder, Llama-3, DeepSeek). Default off → byte-identical to unconstrained decoding.")]
        [DefaultValue(false)]
        public bool ToolGrammar { get; init; }

        // ── JSON-Schema-constrained output (issue #423 follow-up) ──
        // Whole-turn structured output, the OpenTail.Stingray analogue of OpenAI/llama.cpp's
        // response_format:json_schema. Mirrors llama.cpp's own flag names (-j/--json-schema,
        // --json-schema-file) where Spectre.Console.Cli can represent them.
        [CommandOption("-j|--json-schema <SCHEMA>")]
        [Description("JSON schema to constrain the entire response to (https://json-schema.org/), e.g. '{\"type\":\"object\",\"properties\":{...},\"required\":[[...]]}' (llama.cpp -j/--json-schema). Root must be an object schema declaring at least one property; unsupported keywords ($ref, oneOf/anyOf, pattern, minLength/maxLength, minimum/maximum) degrade to unconstrained. Mutually exclusive with --json-schema-file.")]
        public string? JsonSchema { get; init; }

        [CommandOption("--json-schema-file|--jf <PATH>")]
        [Description("File containing a JSON schema to constrain the entire response to (llama.cpp --json-schema-file/-jf; alias --jf since llama.cpp's single-dash -jf isn't representable: Spectre short options must be one character). Mutually exclusive with --json-schema.")]
        public string? JsonSchemaFile { get; init; }

        [CommandOption("--json-schema-ordered")]
        [Description("With --json-schema/--json-schema-file: require properties in declaration order (issue #425) -- optional properties may be skipped but never reordered. Lets a streaming consumer act on an early field before a later, larger one finishes.")]
        [DefaultValue(false)]
        public bool JsonSchemaOrdered { get; init; }

        // ── MoE expert-cache tuning (offloaded MoE models) ──
        // Good defaults are automatic: frequency-aware SLRU eviction, VRAM-sized cache,
        // and next-layer predictive prefetch are all ON without any flag. These knobs only
        // tune/disable that behaviour. Each is also settable via the named env var.
        [CommandOption("--no-moe-predict-prefetch")]
        [Description("MoE: disable next-layer predictive expert prefetch (Vulkan; on by default). Env: STINGRAY_MOE_PREDICT_PREFETCH=0.")]
        [DefaultValue(false)]
        public bool NoMoePredictPrefetch { get; init; }

        [CommandOption("--moe-warmpin")]
        [Description("MoE: also pin the top-N hottest experts per layer into the GPU cache after warmup (default 0 = off; frequency-aware eviction already retains hot experts). Env: STINGRAY_MOE_WARMPIN.")]
        public int? MoeWarmPin { get; init; }

        // ── Execution Plan & Auto-Tuning (§5.3 & §7.1) ──
        [CommandOption("--auto")]
        [Description("Automatically resolve execution plan based on hardware and target goal")]
        [DefaultValue(false)]
        public bool Auto { get; init; }

        [CommandOption("--goal <GOAL>")]
        [Description("Optimization goal for execution planning: balanced (default), quality, throughput, long-context, low-memory")]
        [DefaultValue("balanced")]
        public string Goal { get; init; } = "balanced";

        [CommandOption("--explain")]
        [Description("Print full decision trace for the resolved execution plan before starting generation")]
        [DefaultValue(false)]
        public bool Explain { get; init; }

        [CommandOption("--moe-warmpin-after")]
        [Description("MoE: expert accesses to observe before warm-pinning selects the hot set (default 512). Only used with --moe-warmpin. Env: STINGRAY_MOE_WARMPIN_AFTER.")]
        [DefaultValue(0L)]
        public long MoeWarmPinAfter { get; init; }

        [CommandOption("--expert-stats")]
        [Description("MoE: write GPU expert-cache (SLRU) hit-rate stats to this file on exit. Env: STINGRAY_EXPERT_STATS.")]
        public string? ExpertStatsPath { get; init; }

        // ── MoE expert placement (CPU vs GPU), issue #80. Wraps the existing all-or-nothing
        // STINGRAY_CPU_MOE override the engine reads at forward-pass construction.
        [CommandOption("--cpu-moe|--cmoe")]
        [Description("MoE: keep ALL routed expert weights on the CPU (llama.cpp --cpu-moe). Sets STINGRAY_CPU_MOE=1, overriding the VRAM-fit auto-select; STINGRAY_CPU_MOE=0 in the env still forces on-GPU experts. Alias --cmoe (llama.cpp's single-dash -cmoe isn't representable: Spectre short options must be one character).")]
        [DefaultValue(false)]
        public bool CpuMoe { get; init; }

        [CommandOption("--n-cpu-moe|--ncmoe <N>")]
        [Description("MoE: keep the routed experts of N layers on the CPU (llama.cpp --n-cpu-moe). DEFERRED / not yet supported — OpenTail.Stingray's expert placement is all-or-nothing (no per-layer split in the engine), so passing any value errors with that rationale. Use --cpu-moe (all on CPU) or omit (auto).")]
        public int? NCpuMoe { get; init; }

        // ── GPU op-offload of the CPU-MoE routed prefill. Wraps STINGRAY_MOE_GPU_PREFILL.
        [CommandOption("--gpu-moe-prefill <BOOL>")]
        [Description("CPU-MoE: run the routed-expert prefill matmuls on the GPU (transient weight upload, like llama.cpp's op-offload) instead of CPU dots. Default ON (#390); pass 'false' to force the CPU MoE prefill. Sets STINGRAY_MOE_GPU_PREFILL. ~+28-67% PREFILL on the CUDA GDN-hybrid CPU-MoE models, with DECODE within noise of the CPU path — the register-in-place pin mode (STINGRAY_MOE_PIN_MODE, default 'register') cudaHostRegisters the expert mmap pages instead of a ~14 GB copy, so no RAM duplicate and no page-cache eviction; a token gate (STINGRAY_MOE_GPU_PREFILL_MIN_TOKENS, default 64) keeps tiny prefills + decode on the CPU path. Argmax-stable (GPU runs the MoE in F32), not bit-identical to CPU. Auto-falls-back to the CPU path if the GPU scratch can't allocate.")]
        public bool? GpuMoePrefill { get; init; }
    }

    /// <summary>
    /// Translates the llama.cpp-style MoE placement flags (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>,
    /// issue #80) into the <c>STINGRAY_CPU_MOE</c> override the engine reads when it builds the
    /// hybrid forward pass. <c>--cpu-moe</c> (carried to the runtime via the plan) forces every routed expert onto the CPU
    /// (equivalent to <c>STINGRAY_CPU_MOE=1</c> and the server's <c>CpuMoe=true</c>, issue #93); an
    /// explicit flag wins over an inherited env var, and its absence leaves the env (hence the
    /// engine's VRAM-fit auto-select) untouched. <paramref name="nCpuMoe"/> (partial per-layer
    /// placement) is <b>deferred</b>: the engine override is all-or-nothing, so any value is
    /// rejected via <paramref name="error"/>. Returns <c>false</c> (with <paramref name="error"/>
    /// set) when the caller should abort. Validation only: the flag itself travels in the ExecutionRequest.
    /// </summary>
    internal static bool TryValidateCpuMoeFlags(int? nCpuMoe, out string? error)
    {
        if (nCpuMoe is int n)
        {
            error =
                $"--n-cpu-moe/--ncmoe ({n}) is not supported yet: OpenTail.Stingray places routed MoE " +
                "experts all-or-nothing (the STINGRAY_CPU_MOE override the engine reads has no per-layer " +
                "granularity), so a partial per-layer split can't be honored. Use --cpu-moe to keep all " +
                "routed experts on the CPU, or omit it to let VRAM fit auto-select (STINGRAY_CPU_MOE=0 " +
                "forces on-GPU experts). Tracked in issue #80.";
            return false;
        }

        error = null;
        return true;
    }
}
