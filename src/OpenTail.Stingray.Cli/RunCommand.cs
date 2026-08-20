using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using OpenTail.Stingray.Cli.Terminal;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Core.Grammar;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vision;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Main inference command. Parameter names match llama-cli where applicable.
/// Usage: opentail-llm-cli -m model.gguf -p "Hello" -n 128 --temp 0.7
/// </summary>
public sealed class RunCommand : Command<RunCommand.Settings>
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
        public string? MmprojPath { get; init; }

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
        [Description("Dequant-once BLAS weight-cache budget in MiB for CPU prefill (issue #189): caches the F32 dequant per projection weight so chunked prefill re-pays no dequant (bit-identical). Auto (env STINGRAY_PREFILL_DEQUANT_MB / fit-25%-RAM) by default; 0 = off, negative = unlimited. CPU only.")]
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
    /// hybrid forward pass. <paramref name="cpuMoe"/> forces every routed expert onto the CPU
    /// (equivalent to <c>STINGRAY_CPU_MOE=1</c> and the server's <c>CpuMoe=true</c>, issue #93); an
    /// explicit flag wins over an inherited env var, and its absence leaves the env (hence the
    /// engine's VRAM-fit auto-select) untouched. <paramref name="nCpuMoe"/> (partial per-layer
    /// placement) is <b>deferred</b>: the engine override is all-or-nothing, so any value is
    /// rejected via <paramref name="error"/>. Returns <c>false</c> (with <paramref name="error"/>
    /// set) when the caller should abort; the env side effect mirrors <see cref="GpuDevice.Resolve"/>.
    /// </summary>
    internal static bool TryApplyCpuMoeFlags(bool cpuMoe, int? nCpuMoe, out string? error)
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

        if (cpuMoe)
            Environment.SetEnvironmentVariable("STINGRAY_CPU_MOE", "1");

        error = null;
        return true;
    }

    /// <summary>
    /// Builds a whole-turn <see cref="JsonSchemaOutputConstraint"/> (issue #423 follow-up) from
    /// <c>--json-schema</c> (inline) or <c>--json-schema-file</c>/<c>--jf</c> (file), or leaves
    /// <paramref name="constraint"/> <c>null</c> when neither flag is given.
    /// <paramref name="ordered"/> maps <c>--json-schema-ordered</c> (issue #425: properties in
    /// declaration order). Returns <c>false</c> (with <paramref name="error"/> set) on mutual
    /// exclusivity, a missing/unreadable file, malformed JSON, or a schema that can't be compiled
    /// to a constraint -- an explicit schema request is a hard requirement, so failures are
    /// reported rather than silently ignored.
    /// </summary>
    internal static bool TryLoadJsonSchemaConstraint(
        string? inlineSchema, string? schemaFilePath, GrammarVocabulary vocab,
        out ITokenConstraint? constraint, out string? error, bool ordered = false)
    {
        constraint = null;
        error = null;

        if (inlineSchema is { Length: > 0 } && schemaFilePath is { Length: > 0 })
        {
            error = "--json-schema and --json-schema-file/--jf are mutually exclusive; pass only one.";
            return false;
        }

        string schemaText;
        if (schemaFilePath is { Length: > 0 })
        {
            if (!File.Exists(schemaFilePath))
            {
                error = $"schema file not found: {schemaFilePath}";
                return false;
            }
            try
            {
                schemaText = File.ReadAllText(schemaFilePath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException)
            {
                error = $"could not read --json-schema-file: {ex.Message}";
                return false;
            }
        }
        else if (inlineSchema is { Length: > 0 })
        {
            schemaText = inlineSchema;
        }
        else
        {
            // An explicit ordered request without a schema is user error, not a no-op -- silently
            // generating unconstrained output would violate the fail-loudly contract above.
            if (ordered)
            {
                error = "--json-schema-ordered requires --json-schema or --json-schema-file/--jf.";
                return false;
            }
            return true;   // neither flag given -- no-op
        }

        JsonElement root;
        try
        {
            using var doc = JsonDocument.Parse(schemaText);
            root = doc.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            error = $"could not parse JSON schema: {ex.Message}";
            return false;
        }

        try
        {
            var schemaObject = ToolSchema.FromOpenAiFunction("_", root).Arguments;
            constraint = new JsonSchemaOutputConstraint(vocab, schemaObject, ordered);
            return true;
        }
        catch (ArgumentException ex)
        {
            error = $"--json-schema could not be compiled: {ex.Message}";
            return false;
        }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        ExecutionPlan? resolvedPlan = null;
        if (settings.Explain && !settings.Auto)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] --explain requires --auto so the displayed plan is the plan that will execute.");
            return 1;
        }

        if (settings.Auto && !string.IsNullOrEmpty(settings.ModelPath) && File.Exists(settings.ModelPath))
        {
            try
            {
                AutoPlanInputs planInputs = ResolveAutoPlanInputs(settings);
                resolvedPlan = ExecutionPlanBuilder.Build(
                    settings.ModelPath,
                    settings.Goal,
                    planInputs.Backend,
                    planInputs.GpuLayers,
                    planInputs.ContextSize,
                    settings.KvType);

                Console.WriteLine(resolvedPlan.CompactSummary());

                if (settings.Explain)
                {
                    Console.WriteLine("\n[ExecutionPlan Decision Trace]");
                    foreach (var d in resolvedPlan.Decisions)
                    {
                        Console.WriteLine($"  - [{d.Code}] {d.SelectedValue} ({d.Reason} - {d.Source})");
                    }
                    Console.WriteLine();
                }
            }
            catch
            {
                // Soft fallback if plan building fails before loading
            }
        }

        // --file/-f (llama.cpp): load the prompt from a file. Overrides -p; lets prompts exceed
        // the shell command-line length limit. Read as-is (no trailing-newline stripping).
        if (settings.PromptFile is { Length: > 0 } promptFile)
        {
            if (!File.Exists(promptFile))
            {
                AnsiConsole.MarkupLine($"[red]Prompt file not found:[/] {Markup.Escape(promptFile)}");
                return 1;
            }
            // Read failures (locked file, permissions, bad path) should fail loud + clean, not
            // throw a stack trace; Escape the message since paths can carry Spectre markup chars.
            try
            {
                settings.Prompt = File.ReadAllText(promptFile);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException)
            {
                AnsiConsole.MarkupLine($"[red]Error reading prompt file:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        if (settings.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = settings.MinBatchBlas;

        // -t/--threads must be applied before any forward pass is constructed, since the SIMD
        // kernels capture the parallel options. An explicit flag beats STINGRAY_CPU_THREADS.
        if (settings.Threads > 0)
            SimdKernels.CpuThreads = settings.Threads;

        // Inert llama.cpp flags: accepted so common command lines run, but they do nothing here.
        // Warned rather than refused — a warning is not silent, so this does not breach the
        // "never accept and ignore" rule in docs/llamacpp-onramp-plan.md.
        if (settings.FlashAttn)
            AnsiConsole.MarkupLine("[yellow]Note:[/] -fa/--flash-attn has no effect: attention is already fused in the OpenTail backends.");
        if (settings.NoWarmup)
            AnsiConsole.MarkupLine("[yellow]Note:[/] --no-warmup has no effect: OpenTail has no separate warmup step.");

        // Resolve --device before any GPU call (it may set CUDA_VISIBLE_DEVICES, which the CUDA
        // driver only reads at first init; Vulkan takes the index explicitly below). `--device none`
        // forces the CPU path, overriding --n-gpu-layers.
        int gpuDeviceIndex;
        bool deviceNone;
        try
        {
            gpuDeviceIndex = GpuDevice.Resolve(settings.Device, out deviceNone);
        }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        // `is > 0`, not `!= 0`. NGpuLayers became int? so the planner could tell "unset" from an
        // explicit "-g 0"; under the old int/default-0 shape `!= 0` meant "user asked for GPU
        // layers", but with a nullable, unset is null and `null != 0` is TRUE — which made this
        // note fire on every `--device none` run, claiming to have overridden a flag the user
        // never passed.
        if (deviceNone && settings.NGpuLayers is > 0 or -1)
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --ngl/-g; running on CPU.");
        // Precedence: --device none > explicit --ngl/-g > ExecutionPlan (only under --auto) > CPU.
        //
        // The null test is load-bearing. NGpuLayers is int?, so "unset" is null and "-g 0" is an
        // EXPLICIT request for CPU-only. Testing `== 0` (as the first wiring did) inverted this:
        // null == 0 is false, so the plan was ignored in the default case and applied only when the
        // user had explicitly asked for CPU — overriding the one value they were most deliberate
        // about, which Phase 1's "explicit pins are never silently overridden" criterion forbids.
        //
        // --auto alone gates planning. --goal is a parameter OF the plan, not a second trigger for
        // it; letting `--goal quality` silently enable placement changes would make an opt-in
        // feature reachable without opting in.
        int effNGpuLayers = deviceNone
            ? 0
            : settings.NGpuLayers ?? (settings.Auto && resolvedPlan is not null ? resolvedPlan.GpuLayers : 0);

        // MoE expert-cache knobs are read from the environment inside the engine
        // (WarmPinConfig / HybridForwardPass / slot-manager dispose). Surface them as
        // CLI flags by setting the env var here — before any forward pass is built —
        // so an explicit flag overrides, and env-only use still works.
        if (settings.MoeWarmPin is int warmPin)  // explicitly passed (incl. 0 to force off)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN", warmPin.ToString());
        if (settings.MoeWarmPinAfter > 0)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_WARMPIN_AFTER", settings.MoeWarmPinAfter.ToString());
        if (settings.NoMoePredictPrefetch)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_PREDICT_PREFETCH", "0");

        // MoE expert placement (#80): --cpu-moe sets STINGRAY_CPU_MOE=1; --n-cpu-moe is deferred
        // (the engine override is all-or-nothing) and fails fast with the rationale. Done here,
        // before any forward pass is built, so the engine constructor sees the override.
        if (!TryApplyCpuMoeFlags(settings.CpuMoe, settings.NCpuMoe, out string? cpuMoeError))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(cpuMoeError!)}");
            return 1;
        }

        // GPU op-offload of the CPU-MoE routed prefill (default on in the engine, #390). An explicit
        // --gpu-moe-prefill wins over an inherited STINGRAY_MOE_GPU_PREFILL; absence leaves the
        // env (hence the engine default) untouched.
        if (settings.GpuMoePrefill is bool gpuMoePrefill)
            Environment.SetEnvironmentVariable("STINGRAY_MOE_GPU_PREFILL", gpuMoePrefill ? "1" : "0");

        // KV-cache dtype (issue #179): surface STINGRAY_KV_DTYPE as a flag. Set before
        // any forward pass is built so an explicit flag overrides; env-only use still
        // works. The CudaForwardPass constructor validates the value (fp32|bf16|q8_0).
        string? effectiveKvType = settings.KvType is { Length: > 0 }
            ? settings.KvType
            : settings.Auto ? resolvedPlan?.KvDtype : null;
        if (effectiveKvType is { Length: > 0 })
            Environment.SetEnvironmentVariable("STINGRAY_KV_DTYPE", effectiveKvType);
        if (!string.IsNullOrEmpty(settings.ExpertStatsPath))
            Environment.SetEnvironmentVariable("STINGRAY_EXPERT_STATS", settings.ExpertStatsPath);

        var modelPath = settings.ModelPath;
        if (modelPath is null)
        {
            foreach (var candidate in new[] { "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "model.gguf" })
                if (File.Exists(candidate) || Directory.Exists(candidate)) { modelPath = candidate; break; }
        }
        if (modelPath is null || (!File.Exists(modelPath) && !Directory.Exists(modelPath)))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No model file or package directory found. Use [yellow]-m <path>[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[dim]Loading model:[/] {modelPath}");
        var sw = Stopwatch.StartNew();

        // Pre-declare shared variables so the goto from the SafeTensors branch is valid.
        // In the GGUF path these are assigned in their original locations below, unchanged.
        ModelHyperparams hp;
        GgufTokenizer tokenizer;
        int ctxSize;
        int nGpuLayers;
        // Not a `using var`: the SafeTensors branch jumps past this declaration, and C# forbids a
        // goto that skips a using declaration (CS8648). Disposed explicitly in the finally below.
        GgufModel? model = null;
        ForwardPass? fwd = null;
        HybridGdnForwardPass? hybridFwd = null;
        IForwardPass? mtpFwd = null;
        IDisposable? gpuBackend = null;
        IDisposable? gpuFwd = null;
        // Tracks SafetensorsTensorSource lifetime in the package path; null in the GGUF path.
        // Disposed alongside fwd at every cleanup site below.
        SafetensorsTensorSource? stTensorSource = null;
        // Shared by both paths, and not a `using var` for the same CS8648 reason as `model`.
        CpuBackend? cpuBackend = null;
        Func<int, int, ReadOnlySpan<float>> forward;
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill;
        Action resetCache;

        // ── SafeTensors package branch ────────────────────────────────────────
        // A directory path or bare .safetensors file routes here; GGUF falls
        // through to the GgufModel.Open path below, unchanged.
        bool isPackage = Directory.Exists(modelPath)
            || modelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            || modelPath.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase);

        if (isPackage)
        {
            // ── 1. Capability check — fail before allocating anything expensive.
            var pkgReport = ModelPackageInspector.Inspect(modelPath);
            if (!pkgReport.IsSupported)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] SafeTensors package not supported:");
                foreach (var r in pkgReport.Rejections)
                    AnsiConsole.MarkupLine($"  [red]·[/] {Markup.Escape(r.Detail)}");
                AnsiConsole.MarkupLine("[dim]GGUF is the recommended deployment format for quantized models.[/]");
                return 1;
            }

            // ── 2. Refuse features that require GgufModel or GPU backends.
            // Explicit errors — the user must know why, not receive a cryptic exception.
            if (effNGpuLayers != 0)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] GPU offload ([yellow]--ngl[/] / [yellow]-g[/]) is not yet supported for SafeTensors packages. " +
                    "Run on CPU (omit [yellow]-g[/] or pass [yellow]-g 0[/]), or convert to GGUF for GPU execution.");
                return 1;
            }
            if (settings.TurboQuant)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] [yellow]--tq[/] (TurboQuant) is not supported for SafeTensors packages. " +
                    "Only GGUF models support KV-cache quantization via this flag.");
                return 1;
            }
            if (settings.DraftModelPath is not null || settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Speculative decoding ([yellow]--draft-model[/] / [yellow]--draft-lookup[/]) " +
                    "is not supported for SafeTensors packages.");
                return 1;
            }
            if (settings.DSparkModelPath is not null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] DSpark ([yellow]--dspark-model[/]) is not supported for SafeTensors packages.");
                return 1;
            }
            if (settings.ImagePaths is { Length: > 0 })
            {
                AnsiConsole.MarkupLine("[red]Error:[/] [yellow]--image[/] (multimodal input) is not supported for SafeTensors packages.");
                return 1;
            }

            // ── 3. Open the package and tokenizer.
            try
            {
                stTensorSource = SafetensorsTensorSource.Open(modelPath);
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Failed to open SafeTensors package: {Markup.Escape(ex.Message)}");
                return 1;
            }

            var tokResult = HuggingFaceTokenizerSource.Load(modelPath);
            if (!tokResult.IsUsable || tokResult.Source is null)
            {
                stTensorSource.Dispose();
                stTensorSource = null;
                AnsiConsole.MarkupLine("[red]Error:[/] Failed to load tokenizer:");
                foreach (var r in tokResult.Rejections)
                    AnsiConsole.MarkupLine($"  [red]·[/] {Markup.Escape(r.Detail)}");
                return 1;
            }

            // ── 4. Build engine objects.
            hp = ModelHyperparams.FromGgufMetadata(stTensorSource.Metadata, stTensorSource);
            tokenizer = GgufTokenizer.FromSource(tokResult.Source);
            // Resolve the requested scratch/context ceiling BEFORE constructing the CPU pass.
            // RunSinglePrompt rejects a prompt that would leave no decode slot, so this cannot
            // turn an oversized prompt into an unsafe undersized-scratch prefill.
            ctxSize = settings.CtxSize > 0 ? settings.CtxSize
                : settings.Auto && resolvedPlan is not null ? resolvedPlan.ContextSize : 0;
            cpuBackend = new CpuBackend();
            fwd = new ForwardPass(stTensorSource, cpuBackend, hp, maxContextLength: ctxSize);

            // Populate shared state the decode loop reads (mirrors the GGUF path below).
            s_arch = stTensorSource.Metadata.TryGetValue("general.architecture", out var stArchVal)
                ? (string)stArchVal : "llama";
            s_jinja = tokenizer.ChatTemplate;
            (s_thinkTokenId, s_endThinkTokenId) = tokenizer.ReasoningTokens;

            if (settings.Thinking && settings.NoThinking)
                AnsiConsole.MarkupLine("[yellow]Warning:[/] both --thinking and --no-thinking given; --no-thinking wins.");
            s_noThinking = ResolveThinkingOff(s_arch, settings.Thinking, settings.NoThinking);

            if (s_thinkTokenId > 0 && settings.Temperature == 0f && !s_noThinking)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Greedy decoding (--temp 0) on a reasoning model often produces");
                AnsiConsole.MarkupLine("infinite \"wait, but actually\" loops. Consider [yellow]--temp 0.6 --top-p 0.95 --top-k 20[/].");
            }

            // ── 5. Wire forward/prefill/resetCache — same shape as the GGUF CPU path.
            // GPU offload is refused above, so the shared decode loop must see a CPU-only count.
            nGpuLayers = 0;
            forward = fwd.Forward;
            prefill = tokens => fwd.Prefill(tokens);
            resetCache = fwd.ResetCache;

            goto backendConfigured;
        }

        // ── GGUF path (unchanged) ─────────────────────────────────────────────
        model = GgufModel.Open(modelPath);

        // Apply the same compatibility gate the server loader applies. Until 2026-08-08 this path
        // did not, so the CLI would attempt any architecture while `doctor`, `static-plan` and the
        // server all refused everything outside the supported set — one entry point ran models
        // another would not admit. Disagreeing on which models are supported is worse than either
        // answer alone, so the gate now runs everywhere and --allow-unverified-arch is the explicit
        // way to override it.
        if (settings.AllowUnverifiedArch)
        {
            string requested = model.Metadata.TryGetValue("general.architecture", out var reqArch)
                ? Convert.ToString(reqArch) ?? "" : "";
            if (!ModelCompatibility.IsTextGenerationArchitectureSupported(requested))
                AnsiConsole.MarkupLine(
                    $"[yellow]Warning:[/] architecture '[bold]{requested.EscapeMarkup()}[/]' has no validated "
                    + "forward-pass profile. Running because --allow-unverified-arch was given; output may be "
                    + "wrong in ways that still look plausible. Do not use this run as evidence of support.");
        }
        else
        {
            ModelCompatibility.ValidateForTextGeneration(model);
        }

        hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        s_arch = model.Metadata.TryGetValue("general.architecture", out var archVal) ? (string)archVal : "qwen2";
        // Explicit --ctx-size wins. Under --auto the immutable plan owns this choice, so
        // the printed context is the one passed to the forward pass and TierPlanner.
        ctxSize = settings.CtxSize > 0
            ? settings.CtxSize
            : settings.Auto && resolvedPlan is not null ? resolvedPlan.ContextSize : 0;
        tokenizer = GgufTokenizer.FromGgufModel(model);
        s_jinja = tokenizer.ChatTemplate;

        // Reasoning boundary tokens: ChatML <think>/</think> (Qwen3, DeepSeek-R1, SmolLM3, ...)
        // or Gemma 4's <|channel>thought … <channel|>. Resolved once on the tokenizer so the CLI,
        // server, and engine share one definition. The decode loops no-op when these IDs are -1.
        (s_thinkTokenId, s_endThinkTokenId) = tokenizer.ReasoningTokens;

        // Resolve reasoning on/off (see ResolveThinkingOff for the full precedence). --no-thinking
        // and --thinking are opposites; if both are passed --no-thinking wins, so warn rather than
        // silently dropping --thinking.
        if (settings.Thinking && settings.NoThinking)
            AnsiConsole.MarkupLine("[yellow]Warning:[/] both --thinking and --no-thinking given; --no-thinking wins.");
        s_noThinking = ResolveThinkingOff(s_arch, settings.Thinking, settings.NoThinking);
        // Gemma 4's stock instruct models (E4B-it, 12B-it) bracket a <|channel>thought block in
        // their chat template but are NOT trained to reason, so Gemma 4 defaults thinking off.
        // Surface that default (and how to override it) only when we actually defaulted off.
        if (s_arch == "gemma4" && !settings.Thinking && !settings.NoThinking)
            AnsiConsole.MarkupLine("[dim]Gemma 4 defaults to --no-thinking (stock instruct models aren't " +
                "reasoning-trained). For a reasoning finetune pass --thinking " +
                "(recommended: --temp 1.0 --top-k 64 --top-p 0.95).[/]");

        // Greedy on a reasoning model tends to "wait, but actually" itself into infinite
        // loops; --no-thinking sidesteps the issue since the model won't reason at all.
        if (s_thinkTokenId > 0 && settings.Temperature == 0f && !s_noThinking)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Greedy decoding (--temp 0) on a reasoning model often produces");
            AnsiConsole.MarkupLine("infinite \"wait, but actually\" loops. Consider [yellow]--temp 0.6 --top-p 0.95 --top-k 20[/].");
        }

        // Image input: multimodal vision via --mmproj. CPU-only single-prompt path.
        // Validate the preconditions up front so we fail fast before building any forward pass.
        //
        // No longer gated to gemma4 (found 2026-08-20: RunCommand's actual embedding path below
        // already goes through UnifiedVisionPipeline.Open, which dispatches ALL 22+ supported
        // architectures generically from the mmproj's own clip.vision.projector_type metadata --
        // it never was gemma4-specific. This stale guard rejected every other architecture before
        // that generic path ever ran, even though the encoders themselves were implemented and
        // working -- see docs/perf-loop-project-review-progress.md. UnifiedVisionPipeline.Open
        // still throws a clear NotSupportedException for a genuinely unrecognized mmproj, so this
        // isn't removing validation, just the redundant and wrong one.
        if (settings.ImagePaths is { Length: > 0 } imagePaths)
        {
            if (settings.MmprojPath is not { Length: > 0 })
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --image requires --mmproj <mmproj.gguf> (the multimodal projector).");
                return 1;
            }
            if (!File.Exists(settings.MmprojPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] mmproj file not found: {Markup.Escape(settings.MmprojPath)}");
                return 1;
            }
            foreach (var imgPath in imagePaths)
            {
                if (!File.Exists(imgPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] image file not found: {Markup.Escape(imgPath)}");
                    return 1;
                }
            }
            if (settings.Prompt is not { Length: > 0 })
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --image requires a text prompt ([yellow]-p \"...\"[/]); interactive image chat is not supported yet.");
                return 1;
            }
        }

        cpuBackend = new CpuBackend();

        // Hybrid GDN models (qwen35moe) run via the dedicated HybridGdnForwardPass
        // (CPU) or CudaHybridGdnForwardPass (GPU). Features that touch the per-token
        // GDN state are not supported because the rank-1 recurrence is destructive.
        if (hp.IsHybridSsm && settings.TurboQuant)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers).");
            return 1;
        }
        if (hp.IsHybridSsm && (settings.DraftModelPath is not null || settings.DraftLookup))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Speculative decoding is not supported for hybrid GDN models (GDN state is destructively updated and cannot be rewound).");
            return 1;
        }

        // Build the appropriate CPU forward pass. The GPU branches below construct
        // their own (CudaHybridGdnForwardPass for hybrid + CUDA; the existing Cuda/
        // CudaHybrid paths for non-hybrid). For hybrid + GPU we still build the
        // CPU baseline so the bridge can reuse small helpers, but it stays unused.
        // IForwardPass handle for MtpDecoder integration (issue #32). Captured when the
        // chosen forward pass ships an MTP head. The actual MTP gating happens later in
        // RunSinglePrompt / RunInteractive based on sp.SpecType.
        if (hp.IsHybridSsm && effNGpuLayers == 0)
        {
            hybridFwd = new HybridGdnForwardPass(model, cpuBackend, hp);
            if (hybridFwd.HasMtpHead) mtpFwd = hybridFwd;
        }
        else if (!hp.IsHybridSsm)
        {
            // #189 dequant cache: only the pure-CPU path (no GPU offload) runs the batched
            // CPU prefill that consults it; under -g it would be a wasted F32 model copy.
            long dequantBytes = effNGpuLayers != 0
                ? 0
                : settings.PrefillDequantCacheMb == long.MinValue
                    ? long.MinValue // auto / STINGRAY_PREFILL_DEQUANT_MB
                    : ForwardPass.MbToBudgetBytes(settings.PrefillDequantCacheMb);
            fwd = new ForwardPass(model, cpuBackend, hp, maxContextLength: ctxSize,
                prefillDequantCacheBytes: dequantBytes);
        }


        // Resolve the TurboQuant quantizer (issue #180). KVarN runs on the CPU forward
        // pass (-g 0, P0) or on the full-CUDA-offload dense path (-g -1 / -g NumLayers,
        // Task 5a); SnapKV, Vulkan, partial offload, and MoE-on-GPU are rejected
        // up-front with actionable errors (ForwardPass.EnableTurboQuant and the
        // CudaForwardPass constructor re-check as guards of last resort).
        // The default mode is auto (issue #432): prefer KVarN wherever it is supported
        // and fall back to Lloyd-Max with a loud warning elsewhere — Lloyd-Max 3-bit
        // collapses on QK-norm models (Qwen3-0.6B wikitext-2 PPL: 15.47 fp32 /
        // 15.67 KVarN / 945.6 Lloyd-Max 3-bit).
        TqQuantizer tqQuantizer;
        bool tqModeIsAuto = false;
        switch (settings.TqModeStr.Trim().ToLowerInvariant())
        {
            case "" or "auto":
                tqModeIsAuto = true;
                tqQuantizer = TqQuantizer.LloydMax; // resolved below when --tq is set
                break;
            case "lloydmax" or "lloyd-max":
                tqQuantizer = TqQuantizer.LloydMax;
                break;
            case "kvarn":
                tqQuantizer = TqQuantizer.KVarN;
                break;
            default:
                AnsiConsole.MarkupLine($"[red]Error:[/] Unknown --tq-mode value '{Markup.Escape(settings.TqModeStr)}'. Expected one of: auto, lloydmax, kvarn.");
                return 1;
        }
        if (tqModeIsAuto && settings.TurboQuant)
        {
            // Resolve to KVarN only when every precondition holds (TqSupport is the shared
            // matrix — issue #437), so the auto path can never hit a kvarn error. Partial
            // CUDA offload is only knowable after TierPlanner runs (-g -1); that branch
            // downgrades an auto-resolved KVarN to Lloyd-Max instead of erroring like an
            // explicit --tq-mode kvarn does.
            string? kvarnBlocked = TqSupport.KVarNBlockedReason(
                hp.HeadDim, SnapKvConfig.FromEnvironment().Enabled, onGpu: effNGpuLayers != 0,
                isVulkan: (settings.Backend ?? "auto").Trim().ToLowerInvariant() == "vulkan",
                cudaAvailable: CudaBackend.IsAvailable(), isMoE: hp.IsMoE);
            if (kvarnBlocked is null)
            {
                tqQuantizer = TqQuantizer.KVarN;
            }
            else
            {
                AnsiConsole.MarkupLine(
                    $"[yellow]Warning:[/] --tq is falling back to the Lloyd-Max 3-bit quantizer ({Markup.Escape(kvarnBlocked)}). " +
                    $"{Markup.Escape(TqSupport.QualityWarningReason)}; " +
                    "pass [yellow]--tq-mode lloydmax[/] explicitly to silence this warning.");
            }
        }
        if (tqQuantizer == TqQuantizer.KVarN)
        {
            if (!settings.TurboQuant)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn requires [yellow]--tq[/].");
                return 1;
            }
            if (SnapKvConfig.FromEnvironment().Enabled)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn does not compose with SnapKV eviction yet (issue #180 follow-up); unset [yellow]STINGRAY_SNAPKV_BUDGET[/].");
                return 1;
            }
            if (effNGpuLayers != 0)
            {
                // GPU KVarN is the CUDA full-offload dense path only (issue #180 Task 5a).
                string kvarnBackend = (settings.Backend ?? "auto").Trim().ToLowerInvariant();
                if (kvarnBackend == "vulkan")
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn is not supported on the Vulkan backend; use [yellow]--backend cuda -g -1[/] (full offload) or [yellow]-g 0[/] (CPU).");
                    return 1;
                }
                if (!CudaBackend.IsAvailable())
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn with GPU offload requires a CUDA device (issue #180 Task 5a); use [yellow]-g 0[/] for the CPU path.");
                    return 1;
                }
                if (hp.IsMoE)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use [yellow]-g 0[/] for MoE.");
                    return 1;
                }
            }
        }

        // Validate TurboQuant head-dimension compatibility before any GPU allocation.
        // Lloyd-Max ships hardcoded codebooks for 128/256; KVarN is calibration-free
        // and accepts any power-of-2 head dim in [8, 1024] on the CPU path, [8, 256]
        // on CUDA (the shared-memory WHT cap — 512/1024 stay CPU-only for now).
        if (settings.TurboQuant)
        {
            int headDim = hp.HeadDim;
            if (tqQuantizer == TqQuantizer.KVarN)
            {
                if (!TqSupport.IsKVarNHeadDim(headDim))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] --tq-mode kvarn requires a power-of-2 head dimension in [[8, 1024]]; this model has head dim {headDim}.");
                    return 1;
                }
                if (effNGpuLayers != 0 && headDim > TqSupport.KVarNCudaMaxHeadDim)
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] --tq-mode kvarn on CUDA requires head dim ≤ 256 (shared-memory WHT cap); this model has head dim {headDim}. Use [yellow]-g 0[/] for the CPU path.");
                    return 1;
                }
            }
            else if (!TqSupport.IsLloydMaxHeadDim(headDim))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] TurboQuant requires head dimension 128 or 256; this model has head dim {headDim}. Remove [yellow]--tq[/] to run without KV compression.");
                return 1;
            }
        }

        nGpuLayers = effNGpuLayers;

        // Issue #2 (MoE on hybrid GPU+CPU produced NaN/garbled output) was resolved by
        // fixing the descriptor-set reuse hazard in ComputePipeline.RecordWith.
        // The MoE+hybrid path is now exercised by
        // OpenTail.Stingray.Tests.ForwardPass.VulkanShaderTests.HybridForwardPass_MoE_ProducesFiniteLogits.

        // Resolve which GPU backend to use when -g is non-zero. CUDA is preferred only
        // when the user explicitly opted in (--backend cuda) or auto-detection finds it
        // and Vulkan is not the explicit choice. The CUDA forward pass currently covers
        // dense (non-MoE) models with all layers on GPU; MoE and hybrid -g N stay on the
        // Vulkan path.
        bool wantCuda = false;
        string backendStr = (settings.Backend ?? "auto").Trim().ToLowerInvariant();
        if (nGpuLayers != 0)
        {
            switch (backendStr)
            {
                case "cuda":
                    wantCuda = true;
                    break;
                case "vulkan":
                    wantCuda = false;
                    break;
                case "auto":
                case "":
                    // Auto: pick CUDA when available. CudaForwardPass handles full-offload
                    // (dense + MoE); CudaHybridForwardPass handles partial-offload (dense or
                    // MoE; routed experts stream through the CudaExpertSlotManager SLRU).
                    // TQ on CUDA requires head_dim ∈ {128, 256}; KVarN pow2 in [8, 256]
                    // (already validated above, so kvarn implies CUDA-compatible here).
                    bool tqHeadDimOk = tqQuantizer == TqQuantizer.KVarN
                        || hp.HeadDim is 128 or 256;
                    wantCuda = (!settings.TurboQuant || tqHeadDimOk)
                        && CudaBackend.IsAvailable();
                    break;
                default:
                    AnsiConsole.MarkupLine($"[red]Error:[/] Unknown --backend value '{settings.Backend}'. Expected one of: auto, vulkan, cuda.");
                    return 1;
            }
            if (wantCuda && settings.TurboQuant && tqQuantizer == TqQuantizer.LloydMax
                && hp.HeadDim is not 128 and not 256)
            {
                AnsiConsole.MarkupLine($"[yellow]Note:[/] --backend cuda TurboQuant requires head_dim ∈ {{128, 256}} (model head_dim={hp.HeadDim}); falling back to Vulkan.");
                wantCuda = false;
            }
        }

        if (nGpuLayers == 0)
        {
            // CPU only
            if (hybridFwd is not null)
            {
                forward = hybridFwd.Forward;
                prefill = tokens => hybridFwd.Prefill(tokens);
                resetCache = hybridFwd.ResetCache;
                string ffnKindCpu = hp.IsMoE ? "MoE" : "dense FFN";
                AnsiConsole.MarkupLine($"[dim]Backend: [blue]CPU[/] (hybrid GDN + {ffnKindCpu})[/]");
            }
            else
            {
                if (settings.TurboQuant)
                {
                    fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: tqQuantizer);
                    AnsiConsole.MarkupLine(tqQuantizer == TqQuantizer.KVarN
                        ? "[dim]TurboQuant: [green]enabled[/] (KVarN K4V2, window=256)[/]"
                        : "[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                }
                forward = fwd!.Forward;
                prefill = tokens => fwd.Prefill(tokens);
                resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/][/]");
            }
        }
        else if (wantCuda)
        {
            var cuda = CudaBackend.Create();
            gpuBackend = cuda;
            try
            {
                // qwen35moe (hybrid GDN+MoE) takes a dedicated CUDA forward pass that
                // routes the 30 recurrent blocks to CPU and the 10 attention layers +
                // MoE FFN to GPU via the CudaExpertSlotManager SLRU. Layer placement is
                // implicit (driven by hp.LayerTypes), so we skip TierPlanner here.
                if (hp.IsHybridSsm)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = new LayerPlacement(
                        GpuLayers: hp.NumLayers,
                        CpuLayers: 0,
                        GpuWeightBytes: 0,
                        GpuKvBytes: 0,
                        RecommendedCtxSize: ctxSize > 0 ? ctxSize : Math.Min(hp.ContextLength, 4096));
                    var chgdn = new CudaHybridGdnForwardPass(model, cuda, hp, placement);
                    gpuFwd = chgdn;
                    if (chgdn.HasMtpHead) mtpFwd = chgdn;
                    forward = chgdn.Forward;
                    prefill = tokens => chgdn.Prefill(tokens);
                    resetCache = chgdn.ResetCache;
                    int gdnLayers = 0, attnLayers = 0;
                    for (int i = 0; i < hp.NumLayers; i++)
                        if (hp.LayerTypes![i] == LayerType.Attention) attnLayers++; else gdnLayers++;
                    string ffnKind = hp.IsMoE
                        ? (chgdn.IsMoeOnCpu ? "MoE on CPU" : "MoE on GPU")
                        : "dense FFN on CPU";
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA hybrid GDN[/] ({cuda.Name}, {gdnLayers} GDN + {attnLayers} attn on GPU + {ffnKind})[/]");
                }
                else
                {

                // For -g -1 (auto), run TierPlanner against CUDA's VRAM and use the
                // resulting layer split — same logic as the Vulkan branch. Without this,
                // a model bigger than VRAM (e.g. Qwen3-Coder 30B-A3B in 12 GB) would
                // attempt full-offload via CudaForwardPass and silently OOM.
                int cudaGpuLayers;
                bool moeAutoNeedsHybrid = false;
                if (nGpuLayers == -1)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, kvDtype: CudaForwardPass.ResolveConfiguredKvDType());
                    cudaGpuLayers = placement.GpuLayers;

                    // Gemma 4 KV-share constraint: the shared-KV source layers (E4B:
                    // 22 and 23) must live on the same tier as the shared-KV tail
                    // layers (24..41) because cross-tier KV reads are not wired.
                    // TierPlanner doesn't model this and may return a value that
                    // straddles the boundary (e.g. 30). Clamp UP to NumLayers when
                    // possible — TierPlanner's per-layer KV budget ignores that
                    // shared-KV-aliased layers don't grow their own cache, so it's
                    // pessimistic by ~18 layers × full-ctx-KV; full offload almost
                    // always fits when the auto value already exceeded the safe max.
                    if (hp.KvSourceLayer is { } ksl)
                    {
                        int minSrc = int.MaxValue;
                        for (int i = 0; i < hp.NumLayers; i++)
                            if (ksl[i] >= 0 && ksl[i] < minSrc) minSrc = ksl[i];
                        if (minSrc != int.MaxValue
                            && cudaGpuLayers > minSrc
                            && cudaGpuLayers < hp.NumLayers)
                        {
                            AnsiConsole.MarkupLine(
                                $"[dim]TierPlanner returned -g {cudaGpuLayers}, which would " +
                                $"cross the Gemma 4 KV-share boundary (sources <= {minSrc}); " +
                                $"promoting to full offload (-g {hp.NumLayers}). " +
                                $"Pass -g {minSrc} explicitly if VRAM is tight.[/]");
                            cudaGpuLayers = hp.NumLayers;
                        }
                    }

                    // Issue #215: a MoE model whose routed experts can't all stay resident must use the
                    // hybrid path (which streams experts via SLRU or runs them on CPU), even though the
                    // planner places the whole attention trunk on GPU (GpuLayers == NumLayers). Without
                    // this, auto (-g -1) falls through to full-offload CudaForwardPass and thrashes/OOMs —
                    // the very case TierPlanner was added to avoid.
                    moeAutoNeedsHybrid = hp.IsMoE
                        && cudaGpuLayers == hp.NumLayers
                        && placement.MoeRoutedExpertBytes > placement.ExpertCacheBudgetBytes;
                    if (moeAutoNeedsHybrid)
                    {
                        AnsiConsole.MarkupLine(
                            $"[dim]MoE routed experts ({placement.MoeRoutedExpertBytes / (1024.0 * 1024):F0} MB) " +
                            $"exceed the GPU expert-cache budget ({placement.ExpertCacheBudgetBytes / (1024.0 * 1024):F0} MB); " +
                            $"using the hybrid path (CPU-MoE / SLRU streaming) instead of full offload.[/]");
                    }
                }
                else
                {
                    cudaGpuLayers = nGpuLayers;
                }

                bool wantHybrid = (cudaGpuLayers > 0 && cudaGpuLayers < hp.NumLayers) || moeAutoNeedsHybrid;
                if (wantHybrid && settings.TurboQuant && tqQuantizer == TqQuantizer.KVarN)
                {
                    // KVarN on GPU is the full-offload CudaForwardPass path only (issue
                    // #180 Task 5a) — the hybrid pass has no KVarN ring/tile machinery.
                    if (tqModeIsAuto)
                    {
                        // Auto-resolved KVarN, but TierPlanner picked a partial split.
                        // The partial-offload path has no KVarN machinery, so the only
                        // TQ codec available here is Lloyd-Max — which ships codebooks
                        // for head dim 128/256 only. For any other (pow-2) head dim the
                        // auto path reached here precisely because the Lloyd-Max 128/256
                        // gate above was skipped under the KVarN assumption; downgrading
                        // now would crash in the forward-pass constructor. Fail cleanly
                        // instead, pointing at the CPU KVarN path that does support it.
                        if (!TqSupport.IsLloydMaxHeadDim(hp.HeadDim))
                        {
                            AnsiConsole.MarkupLine(
                                $"[red]Error:[/] --tq with head dim {hp.HeadDim} requires KVarN (Lloyd-Max has no " +
                                $"codebook for this head dim), but KVarN needs full CUDA offload and only " +
                                $"{cudaGpuLayers}/{hp.NumLayers} layers fit this GPU. Use [yellow]-g 0[/] for the CPU KVarN path.");
                            return 1;
                        }
                        // downgrade with the #432 quality warning instead of erroring.
                        tqQuantizer = TqQuantizer.LloydMax;
                        AnsiConsole.MarkupLine(
                            $"[yellow]Warning:[/] --tq is falling back to the Lloyd-Max 3-bit quantizer (KVarN requires " +
                            $"full CUDA offload; only {cudaGpuLayers}/{hp.NumLayers} layers fit this GPU). " +
                            $"{Markup.Escape(TqSupport.QualityWarningReason)}; " +
                            "use [yellow]-g 0[/] for CPU KVarN or pass [yellow]--tq-mode lloydmax[/] to silence this warning.");
                    }
                    else
                    {
                        AnsiConsole.MarkupLine(
                            $"[red]Error:[/] --tq-mode kvarn requires full CUDA offload, but only " +
                            $"{cudaGpuLayers}/{hp.NumLayers} layers fit this GPU. Use [yellow]-g 0[/] for the CPU path.");
                        return 1;
                    }
                }
                if (wantHybrid)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    // pinGpuLayers prices the expert-cache budget (read by the MoE CPU-vs-SLRU
                    // auto-decision) for this exact split. cudaGpuLayers equals the auto value on
                    // the -g -1 path, so pinning is a no-op there and only matters on explicit -g N
                    // (#224). A `with { GpuLayers = }` override would leave the budget stale.
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, kvDtype: CudaForwardPass.ResolveConfiguredKvDType(),
                        pinGpuLayers: cudaGpuLayers);

                    var chfwd = new CudaHybridForwardPass(model, cuda, hp, placement, settings.TurboQuant);
                    gpuFwd = chfwd;
                    forward = chfwd.Forward;
                    prefill = tokens => chfwd.Prefill(tokens);
                    resetCache = chfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA hybrid[/] ({cuda.Name}, {placement.GpuLayers} GPU + {placement.CpuLayers} CPU layers)[/]");
                }
                else if (cudaGpuLayers == 0)
                {
                    // Model doesn't fit any GPU layer — fall back to CPU forward pass.
                    // (Hybrid GDN models were rejected before reaching here.)
                    cuda.Dispose();
                    gpuBackend = null;
                    if (settings.TurboQuant)
                    {
                        fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: tqQuantizer);
                        AnsiConsole.MarkupLine(tqQuantizer == TqQuantizer.KVarN
                            ? "[dim]TurboQuant: [green]enabled[/] (KVarN K4V2, window=256)[/]"
                            : "[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                    }
                    forward = fwd!.Forward;
                    prefill = tokens => fwd.Prefill(tokens);
                    resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                    AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/] (CUDA fallback: no GPU-capable layers)[/]");
                }
                else
                {
                    var cfwd = new CudaForwardPass(model, cuda, hp, ctxSize,
                        enableTurboQuant: settings.TurboQuant, tqQuantizer: tqQuantizer);
                    if (settings.TurboQuant)
                        AnsiConsole.MarkupLine(tqQuantizer == TqQuantizer.KVarN
                            ? $"[dim]TurboQuant: [green]enabled[/] (KVarN K4V2, window=256, context: {cfwd.MaxSeqLen})[/]"
                            : $"[dim]TurboQuant: [green]enabled[/] (3-bit, context: {cfwd.MaxSeqLen})[/]");
                    gpuFwd = cfwd;
                    forward = cfwd.Forward;
                    prefill = tokens => cfwd.Prefill(tokens);
                    resetCache = cfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA[/] ({cuda.Name}, all {hp.NumLayers} layers)[/]");
                }
                } // end !IsHybridSsm
            }
            catch
            {
                gpuFwd?.Dispose();
                gpuBackend?.Dispose();
                gpuFwd = null;
                gpuBackend = null;
                throw;
            }
        }
        else
        {
            var gpu = new VulkanBackend(gpuDeviceIndex);
            gpuBackend = gpu;
            try
            {
                gpu.PrintDeviceInfo();

                var hwProfile = HardwareProfile.Detect(gpu);
                AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");

                // qwen35moe / qwen36 (hybrid GDN + attention) takes a dedicated Vulkan forward
                // pass. Layer placement is implicit (driven by hp.LayerTypes — GDN + attn on GPU,
                // FFN per-layer GPU/CPU), so TierPlanner is skipped, mirroring the CUDA branch above.
                // (PR4 Round 1 — dense FFN; Round 2 — MoE FFN via CPU-MoE / GPU-SLRU.)
                if (hp.IsHybridSsm)
                {
                    var placement = new LayerPlacement(
                        GpuLayers: hp.NumLayers,
                        CpuLayers: 0,
                        GpuWeightBytes: 0,
                        GpuKvBytes: 0,
                        RecommendedCtxSize: ctxSize > 0 ? ctxSize : Math.Min(hp.ContextLength, 4096));
                    var vhgdn = new VulkanHybridGdnForwardPass(model, gpu, hp, placement);
                    gpuFwd = vhgdn;
                    // #357 PR4: the Vulkan GDN hybrid now ships an MTP/NEXTN head (HasMtpHead +
                    // SupportsBatchVerify), so admit it into the MtpDecoder path exactly like the
                    // CUDA branch above. ResolveCliMtp gates the rest on --spec-type / greedy / no-think.
                    if (vhgdn.HasMtpHead) mtpFwd = vhgdn;
                    forward = vhgdn.Forward;
                    prefill = tokens => vhgdn.Prefill(tokens);
                    resetCache = vhgdn.ResetCache;
                    int gdnLayers = 0, attnLayers = 0;
                    for (int i = 0; i < hp.NumLayers; i++)
                        if (hp.LayerTypes![i] == LayerType.Attention) attnLayers++; else gdnLayers++;
                    string vkFfnKind = hp.IsMoE ? "MoE FFN (CPU/SLRU)" : "dense FFN GPU/CPU";
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]Vulkan hybrid GDN[/] ({gpu.Name}, {gdnLayers} GDN + {attnLayers} attn on GPU + {vkFfnKind})[/]");
                    goto backendConfigured;
                }

                // Auto-detect layer count when -g -1
                if (nGpuLayers == -1)
                {
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant, requestedCtxSize: ctxSize);
                    nGpuLayers = placement.GpuLayers;
                    if (nGpuLayers == 0)
                    {
                        // Hybrid GDN models were rejected before reaching this Vulkan branch.
                        // KVarN cannot reach the Vulkan backend (explicit --tq-mode kvarn is
                        // rejected up front, auto falls back to Lloyd-Max), so tqQuantizer is
                        // always Lloyd-Max here.
                        if (settings.TurboQuant)
                        {
                            fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3, quantizer: tqQuantizer);
                            AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                        }

                        forward = fwd!.Forward;
                        prefill = tokens => fwd.Prefill(tokens);
                        resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                        AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/] (auto fallback: no GPU-capable layers for this model/path)[/]");
                        goto backendConfigured;
                    }
                }

                if (nGpuLayers >= hp.NumLayers)
                {
                    // All layers on GPU. Pass the configured KV dtype (issues #311 / #325): fp32
                    // default, bf16 = half-width KV, q8_0 = block-quantized (~quarter) KV. Reuses
                    // the same --kv-type/STINGRAY_KV_DTYPE parser the CUDA path uses.
                    var gfwd = new GpuForwardPass(model, gpu, hp, ctxSize,
                        enableTurboQuant: settings.TurboQuant,
                        kvDtype: CudaForwardPass.ResolveConfiguredKvDTypeOrNull());
                    if (settings.TurboQuant)
                        AnsiConsole.MarkupLine($"[dim]TurboQuant: [green]enabled[/] (3-bit, context: {gfwd.MaxSeqLen})[/]");
                    gpuFwd = gfwd;
                    forward = gfwd.Forward;
                    prefill = tokens => gfwd.Prefill(tokens);
                    resetCache = gfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]GPU[/] ({gpu.Name}, all {hp.NumLayers} layers)[/]");
                }
                else
                {
                    // Hybrid: N layers GPU, rest CPU. nGpuLayers is the auto value on -g -1 and the
                    // explicit count otherwise; pinGpuLayers prices weights/KV/budget for it (#224).
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, pinGpuLayers: nGpuLayers);

                    var hfwd = new HybridForwardPass(model, gpu, hp, placement, settings.TurboQuant);
                    gpuFwd = hfwd;
                    forward = hfwd.Forward;
                    prefill = tokens => hfwd.Prefill(tokens);
                    resetCache = hfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [yellow]Hybrid[/] ({gpu.Name}, {placement.GpuLayers} GPU + {placement.CpuLayers} CPU layers)[/]");
                }
            }
            catch
            {
                gpuFwd?.Dispose();
                gpuBackend?.Dispose();
                gpuFwd = null;
                gpuBackend = null;
                throw;
            }
        }

    backendConfigured:
        int activeContextLength = (gpuFwd as IForwardPass)?.MaxSeqLen
            ?? (fwd as IForwardPass)?.MaxSeqLen
            ?? (hybridFwd as IForwardPass)?.MaxSeqLen
            ?? hp.ContextLength;
        AnsiConsole.MarkupLine($"[dim]Model loaded in {sw.Elapsed.TotalSeconds:F1}s — " +
            $"{hp.NumLayers}L, {hp.EmbeddingDim}d, headDim={hp.HeadDim}, {hp.VocabSize} vocab, ctx={activeContextLength}[/]");

        // ── Tool calling (optional) ───────────────────────────────────────────────
        // --tools advertises OpenAI-format tool definitions to the model via its chat template;
        // --tool-grammar additionally constrains the argument bytes to the supplied JSON Schemas
        // (issue #374) for families with constraint support. Both are single-prompt features.
        // Shared by --json-schema/--json-schema-file below (issue #423 follow-up) -- constructing
        // this is free even when unused (the expensive per-token byte table builds lazily).
        var grammarVocab = new GrammarVocabulary(tokenizer);
        List<ToolSchema>? toolSchemas = null;
        ITokenConstraint? toolConstraint = null;
        int[] toolBoundaryStops = [];
        s_tools = null;   // reset: this run advertises tools only if --tools is given (no leak across in-process runs)
        if (settings.ToolsPath is { Length: > 0 } toolsPath)
        {
            if (!File.Exists(toolsPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] tools file not found: {Markup.Escape(toolsPath)}");
                return 1;
            }
            try
            {
                (s_tools, toolSchemas) = LoadTools(toolsPath);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException
                                          or JsonException or FormatException)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] could not parse --tools file: {Markup.Escape(ex.Message)}");
                return 1;
            }
            AnsiConsole.MarkupLine($"[dim]Loaded {toolSchemas.Count} tool(s) from {Markup.Escape(Path.GetFileName(toolsPath))}.[/]");

            var adapter = ToolCallAdapterRegistry.Get(s_arch);

            // Halt right after the tool call(s) instead of running into a hallucinated trailing
            // turn (issue #304): add the adapter's tool-boundary markers (Gemma 4: <|tool_response>)
            // to the stop set, resolved against the vocab.
            toolBoundaryStops = adapter.ToolBoundaryStopMarkers
                .Select(m => tokenizer.SpecialTokens.TryGetValue(m, out int id) ? id : -1)
                .Where(id => id > 0)
                .Distinct()
                .ToArray();

            if (settings.ToolGrammar)
            {
                toolConstraint = adapter.BuildArgumentConstraint(toolSchemas, grammarVocab);
                AnsiConsole.MarkupLine(toolConstraint is not null
                    ? "[dim]Tool-call arguments are grammar-constrained (issue #374).[/]"
                    : $"[yellow]Warning:[/] --tool-grammar has no effect for arch '{s_arch}' (no constraint support, or no supplied tool is constrainable); arguments generate unconstrained.");
            }
        }
        else if (settings.ToolGrammar)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --tool-grammar requires --tools (a schema to constrain against); ignoring.");
        }

        // ── JSON-Schema-constrained output (issue #423 follow-up) ─────────────────
        if (!TryLoadJsonSchemaConstraint(settings.JsonSchema, settings.JsonSchemaFile, grammarVocab,
                out ITokenConstraint? jsonSchemaConstraint, out string? jsonSchemaError,
                ordered: settings.JsonSchemaOrdered))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(jsonSchemaError!)}");
            return 1;
        }
        if (jsonSchemaConstraint is not null)
        {
            AnsiConsole.MarkupLine("[dim]Response is constrained to the supplied JSON schema.[/]");
            if (toolConstraint is not null)
                AnsiConsole.MarkupLine(
                    "[yellow]Warning:[/] --json-schema/--json-schema-file combined with --tool-grammar likely " +
                    "makes tool calls unreachable (the schema constrains the whole response from the first " +
                    "token) — use one or the other.");
        }

        // -e/--escape: expand escape sequences in the prompt before it is templated or tokenized.
        if (settings.Escape && settings.Prompt is not null)
            settings.Prompt = ProcessEscapeSequences(settings.Prompt);

        // --chat-template: raw Jinja only. Named shortcuts are refused rather than approximated —
        // a hand-written chatml/llama3 template loads, runs, and degrades output with no error,
        // which is the worst failure mode available here.
        if (settings.ChatTemplateOverride is { Length: > 0 } tmplOverride)
        {
            string trimmed = tmplOverride.Trim();
            if (!trimmed.Contains("{{", StringComparison.Ordinal) && !trimmed.Contains("{%", StringComparison.Ordinal))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] --chat-template '{Markup.Escape(trimmed)}' is not Jinja source. " +
                    "Named shortcuts are not supported: pass the model's raw Jinja2 template, or omit the flag to use " +
                    "the one embedded in the model.");
                return 1;
            }
            s_jinja = new JinjaChatTemplate(trimmed);
            AnsiConsole.MarkupLine("[dim]Chat template overridden via --chat-template.[/]");
        }

        // --logit-bias: TOKEN_ID+BIAS / TOKEN_ID-BIAS entries into a bias map.
        IReadOnlyDictionary<int, float>? logitBiasMap = null;
        if (settings.LogitBias is { Length: > 0 } biasEntries
            && !TryParseLogitBias(biasEntries, out logitBiasMap, out string? biasError))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] --logit-bias: {Markup.Escape(biasError!)}");
            return 1;
        }

        var sp = new SamplingParams
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            MaxNewTokens = settings.NPredict,
            StopTokenIds = toolBoundaryStops.Length > 0
                ? [.. BuildStopTokenIds(tokenizer), .. toolBoundaryStops]
                : [.. BuildStopTokenIds(tokenizer)],
            RepetitionPenalty = settings.RepPenalty,
            PresencePenalty = settings.PresencePenalty ?? 0f,
            FrequencyPenalty = settings.FrequencyPenalty ?? 0f,
            LogitBias = logitBiasMap,
            SpecType = ParseSpecType(settings.SpecTypeStr),
            SpecDraftNMax = settings.SpecDraftNMax,
            SpecDraftNMin = settings.SpecDraftNMin,
            SpecDraftPMin = settings.SpecDraftPMin,
            Constraint = TokenConstraints.Combine(jsonSchemaConstraint, toolConstraint),
        };
        var rng = settings.Seed >= 0 ? new Random(settings.Seed) : new Random();

        // Speculative decoding path (requires --draft-model and --temp 0). Supported
        // targets: pure CPU (-g 0) and full CUDA offload of a dense model (issue #207 —
        // packed k-token verify via CudaForwardPass.BatchVerify). Vulkan and the partial-
        // offload hybrids fall back to normal generation: without a batched verify,
        // speculation costs k sequential target forwards per step and is never a win.
        bool specRequested = settings.DraftModelPath is not null || settings.DraftLookup;
        if (specRequested && sp.Constraint is not null)
        {
            // Any active constraint (--tool-grammar and/or --json-schema/--json-schema-file, issue
            // #423 follow-up) masks one token at a time against running grammar state; a multi-token
            // speculative verify can't honor it. Drop speculation so the constraint actually applies
            // (the standard decode path below reads sp.Constraint).
            AnsiConsole.MarkupLine("[yellow]Warning:[/] --tool-grammar/--json-schema is not applied with speculative decoding (--draft-model/--draft-lookup); generating without speculation so the constraint takes effect.");
            specRequested = false;
        }
        if (specRequested)
        {
            bool cudaSpecTarget = gpuFwd is CudaForwardPass { SupportsBatchVerify: true };
            // Vulkan full-offload of a dense Q4_K/Q6_K model exposes the same weight-amortized
            // BatchVerify (issue #308). gemma4/TurboQuant report SupportsBatchVerify=false (no spec);
            // MoE/bias models report true but stay on the bit-exact K-loop fallback (still lossless,
            // just not weight-amortized). --draft-LOOKUP only on Vulkan; --draft-model (needs a 2nd
            // GpuForwardPass + VRAM mgmt) is a CUDA-only follow-up.
            bool vulkanSpecTarget = gpuFwd is GpuForwardPass { SupportsBatchVerify: true };
            bool gpuSpecTarget = cudaSpecTarget || vulkanSpecTarget;
            // Sampled speculative decoding (issue #178): temp>0 now drives distribution-preserving
            // spec sampling on the model-draft path (greedy at temp 0 stays byte-stable). Gated to
            // model drafts (lookup proposals expose no q), to non-penalized/-biased sampling (draft
            // and target must agree on the distribution), and bypassable via STINGRAY_SPEC_SAMPLE=0.
            bool sampledSpec = settings.Temperature > 0f;
            bool specSampleDisabled = Environment.GetEnvironmentVariable("STINGRAY_SPEC_SAMPLE") == "0";
            bool hasPenalty = sp.RepetitionPenalty != 1f && sp.PreviousTokens is { Count: > 0 };
            bool hasBias = sp.LogitBias is { Count: > 0 };
            if (settings.DraftModelPath is not null && settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --draft-model and --draft-lookup are mutually exclusive.");
                return 1;
            }
            if (nGpuLayers != 0 && !gpuSpecTarget)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Speculative decoding requires pure CPU (-g 0), full CUDA offload of a dense or Gemma-4 model, or full Vulkan offload of a dense Q4_K/Q6_K model (--draft-lookup). Falling back to normal generation.");
            }
            else if (vulkanSpecTarget && settings.DraftModelPath is not null)
            {
                // --draft-model needs a 2nd GpuForwardPass + VRAM management on Vulkan; not yet
                // wired. Guard before the draft-model branch's (CudaForwardPass)gpuFwd cast.
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --draft-model speculative decoding is not yet supported on Vulkan (use --draft-lookup, or CUDA); falling back to normal generation.");
            }
            else if (sampledSpec && settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --draft-lookup supports greedy (--temp 0) only; sampled speculative decoding needs --draft-model. Falling back to normal generation.");
            }
            else if (sampledSpec && specSampleDisabled)
            {
                AnsiConsole.MarkupLine("[yellow]Note:[/] STINGRAY_SPEC_SAMPLE=0 — sampled speculative decoding disabled; using normal sampled generation.");
            }
            else if (sampledSpec && (hasPenalty || hasBias))
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] sampled speculative decoding does not yet support --repeat-penalty / logit bias (draft and target must share the same distribution); falling back to normal generation.");
            }
            else if (settings.DraftLookup)
            {
                // Prompt-lookup drafting (issue #207): no draft model — proposals come from
                // n-gram matches against prompt + generated history, verified by the same
                // batched-verify step. Floor is ~baseline (no match → plain decode step).
                try
                {
                    // gpuFwd is the CudaForwardPass or GpuForwardPass (both IForwardPass) on a GPU
                    // spec target; fall back to the CPU pass otherwise.
                    IForwardPass lookupTarget = gpuSpecTarget ? (IForwardPass)gpuFwd! : fwd!;
                    AnsiConsole.MarkupLine($"[dim]Speculative decoding: prompt-lookup (n-gram) drafting | Lookahead k={settings.SpecLookahead}[/]");
                    if (settings.Prompt is not null)
                        return RunSpeculativeSinglePrompt(settings, lookupTarget, null, tokenizer, sp, rng);
                    return RunSpeculativeInteractive(settings, lookupTarget, null, tokenizer, sp, rng);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    return 1;
                }
                finally
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                }
            }
            else if (!File.Exists(settings.DraftModelPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Draft model not found: {settings.DraftModelPath}");
                return 1;
            }
            else
            {
                try
                {
                    AnsiConsole.MarkupLine($"[dim]Loading draft model:[/] {settings.DraftModelPath}");
                    using var draftModel = GgufModel.Open(settings.DraftModelPath);
                    var draftHp = ModelHyperparams.FromGgufMetadata(draftModel.Metadata, draftModel);
                    if (cudaSpecTarget)
                    {
                        var target = (CudaForwardPass)gpuFwd!;
                        // The draft gets its OWN CudaBackend: graph capture state is one
                        // exec graph per backend instance, so sharing the target's backend
                        // would have the draft's decode graph clobber the target's.
                        //
                        // Clamp the draft's context: the decoder advances both passes in
                        // lockstep, so the draft never sees a position past the target's
                        // window — and unless the user pinned -c explicitly, cap it at 4096
                        // (the decode runners bound generation by BOTH windows, so a smaller
                        // draft ring only caps session length, never indexes out of range).
                        // Passing 0 would size the draft's KV from the VRAM left AFTER the
                        // target loaded — measured on the 12 GB 4070 Ti: the 0.6B draft
                        // grabbed a 34K-ctx / ~7 GB ring next to the 8B target (decode
                        // 75 → 13 t/s, WDDM paging); even a target-matched 12K fp32 ring
                        // (~2.8 GB) left so little headroom that the draft's weights paged
                        // in and out every step (draft forward 2.9 → ~15 ms, decode 34 t/s).
                        int draftCtx = ctxSize > 0 ? target.MaxSeqLen : Math.Min(target.MaxSeqLen, 4096);
                        using var draftCuda = CudaBackend.Create();
                        using var draftFwd = new CudaForwardPass(draftModel, draftCuda, draftHp, draftCtx);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([green]CUDA[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, target, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, target, draftFwd, tokenizer, sp, rng);
                    }
                    else
                    {
                        using var draftCpuBackend = new CpuBackend();
                        using var draftFwd = new ForwardPass(draftModel, draftCpuBackend, draftHp);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([blue]CPU[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, fwd!, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, fwd!, draftFwd, tokenizer, sp, rng);
                    }
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    return 1;
                }
                finally
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                }
            }
        }

        // DSpark block-speculative decoding (docs/dspark-plan.md, PR #413): a DeepSpec
        // draft head conditioned on target hidden-state taps. Greedy-only and CPU-target
        // for now (spec Phases 1–3; the CUDA draft path is Phase 4). Placement Off (auto
        // or explicit) falls through to normal generation rather than erroring.
        bool dsparkRequested = settings.DSparkModelPath is not null || sp.SpecType == SpecType.DSpark;
        if (dsparkRequested)
        {
            if (settings.DSparkModelPath is null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --spec-type dspark requires --dspark-model <path-to-model.safetensors>.");
                return 1;
            }
            if (settings.DraftModelPath is not null || settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --dspark-model and --draft-model/--draft-lookup are mutually exclusive.");
                return 1;
            }
            if (sp.SpecType == SpecType.Mtp)
            {
                // An explicit conflicting --spec-type must not be silently outranked
                // by the presence of --dspark-model.
                AnsiConsole.MarkupLine("[red]Error:[/] --spec-type mtp conflicts with --dspark-model; pick one.");
                return 1;
            }
            if (settings.DSparkMinConfidence > 1f)
            {
                // Same [0,1] contract the --spec-draft-p-min validation enforces;
                // a threshold above any sigmoid output would silently disable all
                // drafting instead of doing what the user meant.
                AnsiConsole.MarkupLine($"[red]Error:[/] --dspark-min-confidence={settings.DSparkMinConfidence} must be in [0, 1].");
                return 1;
            }

            // Supported targets: pure CPU (-g 0) and dense full CUDA offload (-g -1,
            // Phase 4). Vulkan and the partial-offload hybrids fall back — no tap
            // capture there yet.
            IForwardPass? dsparkTarget = null;
            CudaBackend? dsparkCuda = null;
            if (nGpuLayers == 0 && fwd is not null)
            {
                dsparkTarget = fwd;
            }
            else if (gpuFwd is CudaForwardPass cudaTarget && gpuBackend is CudaBackend cudaBk)
            {
                dsparkTarget = cudaTarget;
                dsparkCuda = cudaBk;
            }

            string? dsparkReject = null;
            if (sp.SpecType == SpecType.None)
                dsparkReject = "--spec-type none explicitly disables speculation";
            else if (sp.Constraint is not null)
                dsparkReject = "--tool-grammar/--json-schema is active (a multi-token verify can't honor a token-level constraint)";
            else if (settings.ToolsPath is not null)
                dsparkReject = "--tools capture is not wired on the DSpark path (same restriction as MTP)";
            else if (settings.Temperature > 0f)
                dsparkReject = "DSpark is greedy-only for now; pass --temp 0";
            else if (dsparkTarget is null)
                dsparkReject = "DSpark requires a pure CPU target (-g 0) or dense full CUDA offload (-g -1); Vulkan and partial-offload hybrids have no tap capture";
            else if (!dsparkTarget.SupportsHiddenTaps)
                dsparkReject = "the target pass can't capture hidden taps (SnapKV eviction, TurboQuant KV, MoE, or Gemma-4 transforms active)";

            if (dsparkReject is not null)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] DSpark disabled — {dsparkReject}. Falling back to normal generation.");
            }
            else if (settings.Prompt is null)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] DSpark is wired for single-prompt runs only (like MTP); interactive mode falls back to normal generation.");
            }
            else
            {
                int rc;
                try
                {
                    // DSpark is refused for SafeTensors packages above, so this is only ever
                    // reached on the GGUF path where `model` is assigned.
                    rc = TryRunDSparkSinglePrompt(settings, model!, hp, dsparkTarget!, dsparkCuda,
                        tokenizer, sp, ctxSize);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    rc = 1;
                }
                if (rc >= 0)
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                    return rc;
                }
                // rc < 0: placement said Off — fall through to normal generation.
            }
        }

        try
        {
            IForwardPass activeForwardPass = (gpuFwd as IForwardPass) ?? (fwd as IForwardPass) ?? (hybridFwd as IForwardPass)
                ?? throw new InvalidOperationException("No forward pass was configured.");
            if (settings.ImagePaths is { Length: > 0 })
                return RunImagePrompt(settings, activeForwardPass, tokenizer, hp, sp, rng,
                    activeForwardPass.MaxSeqLen);
            if (settings.Prompt is not null)
                return RunSinglePrompt(settings, forward, prefill, tokenizer, sp, rng, mtpFwd,
                    activeForwardPass.MaxSeqLen);
            return RunInteractive(settings, forward, prefill, resetCache, tokenizer, sp, rng, mtpFwd,
                activeForwardPass.MaxSeqLen);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
        finally
        {
            gpuFwd?.Dispose();
            gpuBackend?.Dispose();
            fwd?.Dispose();
            hybridFwd?.Dispose();
            // Package-path resources: null in the GGUF path. Disposed after fwd so that
            // ForwardPass finishes reading tensor data before the memory maps are closed.
            cpuBackend?.Dispose();
            stTensorSource?.Dispose();
            // Was a `using var` before the package branch needed to jump past it.
            model?.Dispose();
        }
    }

    /// <summary>
    /// Converts CLI pins into the planner's narrower input surface. Kept separate so the
    /// precedence rules are testable without loading a model or initializing a backend.
    /// </summary>
    internal static AutoPlanInputs ResolveAutoPlanInputs(Settings settings)
    {
        bool deviceNone = string.Equals(settings.Device, "none", StringComparison.OrdinalIgnoreCase);
        return new(
            deviceNone ? "cpu" : settings.Backend,
            deviceNone ? 0 : settings.NGpuLayers,
            settings.CtxSize > 0 ? settings.CtxSize : null);
    }

    internal sealed record AutoPlanInputs(string? Backend, int? GpuLayers, int? ContextSize);

    /// <summary>
    /// True when a prompt of <paramref name="promptTokens"/> tokens leaves no room to
    /// speculate inside BOTH context windows (prompt + lookahead + 1 correction token).
    /// Prints an actionable error: the typical trigger is the CUDA draft's 4096-token
    /// KV ring cap when <c>-c</c> isn't pinned, where prefilling past the ring would
    /// write K/V out of range and a tail prompt would silently emit zero tokens.
    /// </summary>
    private static bool SpecWindowExhausted(int promptTokens,
        IForwardPass target, IForwardPass? draft, int lookahead)
    {
        int window = Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue);
        if (promptTokens + lookahead + 1 < window) return false;
        AnsiConsole.MarkupLine(
            $"[red]Error:[/] prompt ({promptTokens} tokens) + lookahead ({lookahead}) does not fit the " +
            $"speculative context window ({window} tokens" +
            (draft is not null && draft.MaxSeqLen < target.MaxSeqLen
                ? $", limited by the draft model's KV ring — pass -c to size it explicitly"
                : "") +
            "). Shorten the prompt, raise -c, or drop --draft-model/--draft-lookup.");
        return true;
    }

    private static int RunSpeculativeSinglePrompt(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        // The prompt must fit BOTH context windows BEFORE any prefill runs — the
        // draft's ring may be much smaller than the target's (the CUDA spec path
        // caps it at 4096 when -c isn't pinned), and a too-long prompt would write
        // K/V past the ring's end during draft.Prefill, not merely cap generation.
        if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
            return 1;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        // Prefill (batched-trunk path — the per-token Forward loop this replaces was
        // ~30× slower on the CUDA target). A null draft means prompt-lookup mode.
        ReadOnlySpan<float> targetLogits = target.Prefill(tokens);
        ReadOnlySpan<float> draftLogits = draft is not null ? draft.Prefill(tokens) : default;
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        SpeculativeDecoder spec;
        if (draft is not null)
        {
            // temp>0 → sampled (distribution-preserving) accept; temp 0 → greedy (byte-stable).
            spec = sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead)
                : new SpeculativeDecoder(target, draft, s.SpecLookahead);
            spec.Initialize(tokens.Count, targetLogits, draftLogits);
        }
        else
        {
            spec = new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead);
            spec.Initialize(tokens, targetLogits);
        }

        // Bound generation by BOTH context windows (the draft's may be smaller — the CUDA
        // spec path caps its KV ring), leaving lookahead headroom for the last spec step.
        // The guard above ensures maxNew >= 1 here.
        int maxNew = Math.Min(sp.MaxNewTokens,
            Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);
        if (maxNew < sp.MaxNewTokens)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] generation capped at {maxNew} tokens by the context window.");

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
        {
            if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
        });
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking)) Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            $" | Acceptance rate: {spec.AcceptanceRate:P0} | " +
            $"draft {spec.DraftMs:F0}ms / verify {spec.VerifyMs:F0}ms / commit {spec.CommitMs:F0}ms[/]");
        return 0;
    }

    private static int RunSpeculativeInteractive(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        AnsiConsole.MarkupLine("[green]Interactive chat (speculative decoding).[/] Type a message, or [yellow]/exit[/] to quit.\n");
        var spec = draft is not null
            ? (sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead)
                : new SpeculativeDecoder(target, draft, s.SpecLookahead))
            : new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead);

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
            var tokens = tok.Encode(prompt);

            // Same pre-prefill window guard as the single-prompt runner: the draft
            // ring may be smaller than the target's window, and prefilling past it
            // writes K/V out of range rather than just capping generation.
            if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
                continue;

            target.ResetCache();
            draft?.ResetCache();

            var sw = Stopwatch.StartNew();
            ReadOnlySpan<float> targetLogits = target.Prefill(tokens);

            if (draft is not null)
                spec.Initialize(tokens.Count, targetLogits, draft.Prefill(tokens));
            else
                spec.Initialize(tokens, targetLogits);

            int maxNew = Math.Min(sp.MaxNewTokens,
                Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);

            sw.Restart();
            int generated = 0;
            int totalDecoded = 0;
            bool inThinking = false;
            var streamDec = new Utf8StreamDecoder();
            bool hideThinking = s.HideThinking;
            spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
            {
                if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
                totalDecoded++;
            });
            var tail = streamDec.Flush();
            if (!(hideThinking && inThinking)) Console.Write(tail);
            if (inThinking) Console.Write("\x1b[0m");
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                $" | Accept: {spec.AcceptanceRate:P0}[/]\n");

            if (OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled)
                OpenTail.Stingray.Engine.DecodeProfileTimers.Report(Console.Out);

            if (s.SingleTurn) break;
        }
        return 0;
    }

    /// <summary>
    /// DSpark single-prompt runner (docs/dspark-plan.md, PR #413): resolves the draft-head
    /// files (model.safetensors + sibling config.json), validates head↔target compatibility,
    /// runs the placement planner, loads the head (GPU or CPU per the decision — Phase 4),
    /// enables hidden taps on the target, and drives <see cref="DSparkDecoder"/>. Returns
    /// 0/1 like the other runners, or -1 when the placement decision is Off — the caller
    /// falls back to normal generation.
    /// </summary>
    private static int TryRunDSparkSinglePrompt(Settings s, GgufModel model, ModelHyperparams hp,
        IForwardPass target, CudaBackend? cuda, GgufTokenizer tok, SamplingParams sp, int ctxSize)
    {
        string stPath = s.DSparkModelPath!;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        if (!File.Exists(stPath))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] DSpark model not found: {stPath}");
            return 1;
        }
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        if (!File.Exists(cfgPath))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] DSpark config.json not found next to the safetensors: {cfgPath}");
            return 1;
        }

        var cfg = DSparkConfig.FromJsonFile(cfgPath);
        if (cfg.VocabSize != hp.VocabSize || cfg.NumTargetLayers != hp.NumLayers
            || cfg.HiddenSize != hp.EmbeddingDim)
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] DSpark head/target mismatch — head expects vocab {cfg.VocabSize}, " +
                $"{cfg.NumTargetLayers} target layers, hidden {cfg.HiddenSize}; target has " +
                $"vocab {hp.VocabSize}, {hp.NumLayers} layers, hidden {hp.EmbeddingDim}. " +
                "The head must be trained for this target model.");
            return 1;
        }

        DSparkPlacement userPlace;
        try
        {
            userPlace = DSparkPlacementPlanner.ResolvePlacement(s.DSparkPlaceStr);
        }
        catch (ArgumentException ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }

        // Static placement estimate (spec §4). With a CUDA target the profile carries
        // real VRAM + a measured PCIe probe; a CPU target detects no GPU, so the
        // planner decides Cpu vs Off from RAM headroom. The CPU-side figure also
        // carries the target's hidden-tap buffer, which grows to ctx × TapDim × 4
        // bytes on the host regardless of where the draft runs.
        var hwProfile = cuda is not null ? HardwareProfile.Detect(cuda) : HardwareProfile.Detect();
        var targetPlacement = TierPlanner.Plan(model, hp, hwProfile, s.TurboQuant,
            requestedCtxSize: ctxSize);
        long headBytesGpu = CudaDSparkDraftModel.EstimateGpuResidentBytes(cfg);
        long headBytesCpu = DSparkDraftModel.EstimateResidentBytes(cfg);
        long tapBytes = (long)targetPlacement.RecommendedCtxSize * cfg.TapDim * sizeof(float);
        var decision = DSparkPlacementPlanner.Plan(
            hwProfile, targetPlacement, headBytesGpu, headBytesCpu, userPlace,
            hostTapBytes: tapBytes);
        AnsiConsole.MarkupLine($"[dim]DSpark placement: {decision.Placement} — {decision.Reason.EscapeMarkup()}[/]");
        if (decision.Placement == DSparkPlacement.Off)
        {
            AnsiConsole.MarkupLine("[yellow]Note:[/] DSpark placement is off; falling back to normal generation.");
            return -1;
        }
        if (decision.Placement == DSparkPlacement.Gpu && cuda is null)
        {
            // A GPU draft needs the target's CudaBackend (shared stream orders the tap
            // producer and draft consumer); with a CPU target the head runs on CPU.
            // Re-plan in AUTO over a GPU-less profile so the RAM budget is actually
            // checked (a Cpu OVERRIDE would skip it by contract) — the spec's
            // Gpu → Cpu → Off graceful fallback.
            var cpuCheck = DSparkPlacementPlanner.Plan(
                hwProfile with { VramBytes = 0 }, targetPlacement,
                headBytesGpu, headBytesCpu, DSparkPlacement.Auto,
                hostTapBytes: tapBytes);
            AnsiConsole.MarkupLine(
                "[yellow]Note:[/] a GPU DSpark draft requires a CUDA target (-g -1); " +
                $"re-planned for CPU — {cpuCheck.Reason.EscapeMarkup()}");
            if (cpuCheck.Placement == DSparkPlacement.Off)
            {
                AnsiConsole.MarkupLine("[yellow]Note:[/] DSpark placement is off; falling back to normal generation.");
                return -1;
            }
            decision = cpuCheck;
        }

        AnsiConsole.MarkupLine($"[dim]Loading DSpark draft head:[/] {stPath}");
        using var st = SafetensorsLoader.Open(stPath);
        using IDSparkDraft draft = decision.Placement == DSparkPlacement.Gpu
            ? new CudaDSparkDraftModel(cfg, st, cuda!, target.MaxSeqLen)
            : new DSparkDraftModel(cfg, st, target.MaxSeqLen);
        AnsiConsole.MarkupLine(
            $"[dim]DSpark draft: {cfg.NumLayers}L block-{cfg.BlockSize} " +
            $"([{(decision.Placement == DSparkPlacement.Gpu ? "green]GPU" : "blue]CPU")}[/])[/]");
        target.EnableHiddenTaps(cfg.TargetLayerIds);

        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);
        // The head's RoPE window (max_position_embeddings) can be smaller than the
        // target's — bound the whole session by BOTH, like the spec-decode runner does.
        int window = Math.Min(target.MaxSeqLen, draft.MaxContext);
        if (tokens.Count + cfg.BlockSize + 1 >= window)
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] prompt ({tokens.Count} tokens) + DSpark block ({cfg.BlockSize}) " +
                $"does not fit the context window ({window} tokens" +
                (draft.MaxContext < target.MaxSeqLen ? ", limited by the draft head's RoPE window" : "") +
                ").");
            return 1;
        }

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = target.Prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(tokens.Count, logits);

        int maxNew = Math.Min(sp.MaxNewTokens,
            window - tokens.Count - cfg.BlockSize - 1);
        if (maxNew < sp.MaxNewTokens)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] generation capped at {maxNew} tokens by the context window.");

        float minConfidence = DSparkDecoder.ResolveMinConfidence(s.DSparkMinConfidence);
        int verifyLenCap = DSparkDecoder.ResolveVerifyLen(s.DSparkVerifyLen);

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        decoder.Decode(maxNew, sp.StopTokenIds ?? [], token =>
        {
            if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
        }, minConfidence, verifyLenCap);
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking)) Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            $" | DSpark accept: {decoder.AcceptanceRate:P0} ({decoder.TotalDraftsAccepted}/{decoder.TotalDraftsEmitted}) | " +
            $"draft {decoder.DraftMs:F0}ms / verify {decoder.VerifyMs:F0}ms / commit {decoder.CommitMs:F0}ms[/]");
        // Draft-internal breakdown for the #428 perf work: enqueue = launch-issue CPU
        // time, gpu-wait = GPU execution + D2H collapsed into the per-block download
        // sync, heads = host Markov/confidence chain.
        if (Environment.GetEnvironmentVariable("STINGRAY_DSPARK_TIMING") == "1"
            && draft is CudaDSparkDraftModel cudaDraft)
            AnsiConsole.MarkupLine(
                $"[dim]DSpark draft breakdown: enqueue {decoder.DraftMs - cudaDraft.GpuWaitMs - cudaDraft.HostHeadsMs:F0}ms / " +
                $"gpu-wait {cudaDraft.GpuWaitMs:F0}ms / heads {cudaDraft.HostHeadsMs:F0}ms[/]");
        return 0;
    }

    /// <summary>
    /// Bounds generation so prompt + output fit the active context. ForwardPass sizes its
    /// attention-score and RoPE scratch from the same ceiling but its KV cache is far larger, so
    /// decoding past it is an out-of-bounds native access; the engine now throws, and this is what
    /// keeps ordinary "context full" from reaching that throw. Returns <paramref name="sp"/>
    /// unchanged when the request already fits.
    /// </summary>
    internal static SamplingParams ClampToRemainingContext(SamplingParams sp, int promptTokens, int maxContextLength)
    {
        int room = maxContextLength - promptTokens;
        if (room >= sp.MaxNewTokens) return sp;

        AnsiConsole.MarkupLine(
            $"[yellow]Note:[/] --n-predict {sp.MaxNewTokens} exceeds the {maxContextLength}-token context " +
            $"with a {promptTokens}-token prompt; generating at most {room} token{(room == 1 ? "" : "s")}. " +
            $"Raise [yellow]--ctx-size[/] for a longer response.");
        return sp with { MaxNewTokens = room };
    }

    private static int RunSinglePrompt(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd, int maxContextLength)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        // STINGRAY_RAW_PROMPT bypasses the chat template, so we need to add BOS
        // manually for models that expect it (e.g. Gemma 4 with add_bos_token=true).
        // The chat-template path already injects bos_token via Jinja.
        bool isRaw = Environment.GetEnvironmentVariable("STINGRAY_RAW_PROMPT") == "1";
        if (isRaw && tok.AddBosToken && tok.BosTokenId >= 0
            && (tokens.Count == 0 || tokens[0] != tok.BosTokenId))
        {
            var withBos = new List<int>(tokens.Count + 1) { tok.BosTokenId };
            withBos.AddRange(tokens);
            tokens = withBos;
        }

        // The final prompt token occupies one context slot and ordinary generation needs at
        // least one more. Check after every tokenizer/BOS transformation and before Prefill:
        // ForwardPass uses this same bound for its scratch allocation, so admitting an oversized
        // prompt after wiring --ctx-size would be an unsafe write, not a harmless truncation.
        if (tokens.Count >= maxContextLength)
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] prompt has {tokens.Count} tokens but the active context is " +
                $"{maxContextLength}; shorten the prompt or raise [yellow]--ctx-size[/] so at least one token can be generated.");
            return 1;
        }

        // The prompt check above is necessary but not sufficient: generation continues from the
        // prompt's last position, so the context bounds prompt + output together. With
        // --ctx-size 512 and the default --n-predict 512, any prompt at all runs off the end.
        // Stop at context-full (llama.cpp's -2 semantics) rather than decoding past MaxSeqLen.
        sp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

        if (s.VerbosePrompt)
        {
            var escaped = prompt.Replace("\n", "\\n").Replace("\r", "\\r");
            AnsiConsole.MarkupLine($"[dim]Prompt (escaped): {Markup.Escape(escaped)}[/]");
            AnsiConsole.MarkupLine($"[dim]Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}[/]");
        }

        var sw = Stopwatch.StartNew();
        var logits = prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (OpenTail.Stingray.Engine.PrefillProfileTimers.Enabled)
            OpenTail.Stingray.Engine.PrefillProfileTimers.Report(Console.Out);

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        // MTP self-speculative decode (issue #32). Activates when the model ships an
        // MTP head AND sampling is greedy AND the user disabled thinking on the chat
        // template (--no-thinking) AND sp.SpecType permits. STINGRAY_DISABLE_MTP=1 is
        // a back-compat off-switch that wins.
        bool useMtp = ResolveCliMtp(mtpFwd, sp, s_noThinking, out string? mtpReject);
        if (mtpReject != null)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(mtpReject)}");
            return 1;
        }

        // When --tools is active, capture the raw token stream so the calls can be parsed and shown
        // afterward, and route through the standard decode loop: the MTP fast path can't honor the
        // argument-grammar constraint (sp.Constraint) and has no capture hook.
        List<int>? toolCapture = s.ToolsPath is { Length: > 0 } ? new List<int>() : null;
        if (toolCapture is not null) useMtp = false;

        sw.Restart();
        int generated, totalDecoded;
        float? acceptanceRate = null;
        long mtpAccepted = 0, mtpEmitted = 0;
        if (useMtp)
        {
            (generated, totalDecoded, acceptanceRate, mtpAccepted, mtpEmitted) =
                DecodeLoopMtp(mtpFwd!, tokens, logits, tok, sp, s.HideThinking, s.VerbosePrompt);
        }
        else
        {
            (generated, totalDecoded) =
                DecodeLoop(forward, logits, tokens.Count, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens, toolCapture, repeatLastN: s.RepeatLastN);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            (acceptanceRate is float ar ? $" | MTP accept: {ar:P0} ({mtpAccepted}/{mtpEmitted})" : "") +
            "[/]");

        if (OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled)
            OpenTail.Stingray.Engine.DecodeProfileTimers.Report(Console.Out);

        if (toolCapture is not null)
            PrintToolCalls(tok, toolCapture);
        return 0;
    }

    /// <summary>User-facing prompt marker for an image position (mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder before templating). One per <c>--image</c>, left-to-right.</summary>
    private const string ImageMarker = "<image>";

    /// <summary>
    /// Single-prompt image→text for Gemma 4 (issue #250), one or more images. Each image is
    /// preprocessed and run through the encoder-free projector to soft tokens, then spliced
    /// into the decoder via <see cref="ForwardPass.ForwardEmbedding"/>, wrapped in the runtime
    /// markers (<c>&lt;|image&gt;</c> … soft tokens … <c>&lt;image|&gt;</c>).
    ///
    /// Placement: each <c>&lt;image&gt;</c> marker in the prompt is mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder (id 258880), the prompt is rendered through the model's
    /// own chat template (so BOS / <c>&lt;|turn&gt;</c> / thinking handling matches the text path —
    /// Gemma 4 uses <c>&lt;|turn&gt;role\n…&lt;turn|&gt;</c>, NOT Gemma 3's <c>&lt;start_of_turn&gt;</c>),
    /// then each placeholder token in the token stream is expanded with its image's soft tokens
    /// in order. With no markers, the images are prepended to the user turn. The
    /// embedding-injection seam (<see cref="IForwardPass.ForwardEmbedding"/>) is implemented by
    /// <see cref="ForwardPass"/> (CPU), <see cref="CudaForwardPass"/> (full CUDA offload), and
    /// <see cref="CudaHybridForwardPass"/> (CUDA partial offload, issue #252).
    /// </summary>
    private static int RunImagePrompt(Settings s,
        IForwardPass fwd, GgufTokenizer tok, ModelHyperparams hp,
        SamplingParams sp, Random rng, int maxContextLength)
    {
        if (!fwd.SupportsEmbeddingInput)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] the selected backend does not support image embedding input. " +
                "Image input runs on CPU ([yellow]-g 0[/]), full CUDA offload ([yellow]-g -1[/]), or CUDA " +
                "partial-offload ([yellow]-g N[/]); the Vulkan partial-offload hybrid is not supported yet.");
            return 1;
        }

        var imagePaths = s.ImagePaths!;
        int nImages = imagePaths.Length;

        IVisionEmbedder vision;
        try
        {
            vision = UnifiedVisionPipeline.Open(s.MmprojPath!);
        }
        catch (NotSupportedException ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        using var __ = vision; // matches this method's existing risk tolerance: other early
                                // returns below (e.g. the EmbedImageFile catch) also don't dispose
                                // vision explicitly; process exit reclaims it for a CLI invocation.
        int embd = hp.EmbeddingDim;

        // Reconcile the number of <image> markers in the prompt with the number of --image
        // files. No markers → prepend one placeholder per image (in --image order). Otherwise
        // the counts must match so the i-th marker pairs with the i-th --image.
        //
        // The text substituted in must be the model's own placeholder marker (vision.
        // PlaceholderMarker), not a hardcoded literal: this method originally only supported
        // Gemma 4, whose marker happens to literally be "<|image|>", but every other architecture
        // uses its own text (DotsOcr/Granite4 = "<image_pad>", InternVL = "<IMG_CONTEXT>", etc.).
        // Hardcoding "<|image|>" for all of them meant the substituted text tokenized back to
        // ordinary characters instead of the model's real placeholder token id, so the later
        // placeholder-count check always found 0 -- see docs/vl-migration-plan-2026-08-20.md.
        int markerCount = CountOccurrences(s.Prompt!, ImageMarker);
        string userMsg;
        if (markerCount == 0)
        {
            userMsg = string.Concat(Enumerable.Repeat(vision.PlaceholderMarker, nImages)) + s.Prompt;
        }
        else if (markerCount == nImages)
        {
            userMsg = s.Prompt!.Replace(ImageMarker, vision.PlaceholderMarker);
        }
        else
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] prompt has {markerCount} '{ImageMarker}' marker(s) but " +
                $"{nImages} --image file(s) were given; the counts must match (or omit markers to prepend the images).");
            return 1;
        }

        // Project every image to its soft-token block up front, in --image order.
        var blocks = new (float[] Soft, int NTok)[nImages];
        int totalSoft = 0;
        for (int i = 0; i < nImages; i++)
        {
            float[] soft;
            int nTok;
            try
            {
                soft = vision.EmbedImageFile(imagePaths[i], out nTok);
            }
            catch (Exception ex) when (ex is IOException or NotSupportedException or InvalidDataException
                                          or UnauthorizedAccessException or System.Security.SecurityException)
            {
                AnsiConsole.MarkupLine($"[red]Error reading image[/] {Markup.Escape(imagePaths[i])}: {Markup.Escape(ex.Message)}");
                return 1;
            }
            blocks[i] = (soft, nTok);
            totalSoft += nTok;
            AnsiConsole.MarkupLine($"[dim]Image {i + 1}/{nImages}: {vision.ProjectorType} -> {nTok} soft tokens ({embd}-dim)[/]");
        }

        int imgOpen = !string.IsNullOrEmpty(vision.ImageOpenMarker) && tok.SpecialTokens.TryGetValue(vision.ImageOpenMarker, out var o) ? o : -1;
        int imgClose = !string.IsNullOrEmpty(vision.ImageCloseMarker) && tok.SpecialTokens.TryGetValue(vision.ImageCloseMarker, out var c) ? c : -1;
        int placeholder = tok.SpecialTokens.TryGetValue(vision.PlaceholderMarker, out var ph) ? ph :
                          (tok.SpecialTokens.TryGetValue("<|image|>", out var ph2) ? ph2 :
                          (tok.SpecialTokens.TryGetValue("<image_soft_token>", out var ph3) ? ph3 : 258880));

        var prompt = FormatPrompt(userMsg, s.SystemPrompt, enableThinking: !s_noThinking);
        var allTokens = tok.Encode(prompt).ToList();
        int placeholdersFound = allTokens.Count(t => t == placeholder);
        if (placeholdersFound != nImages)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] expected {nImages} image placeholder token(s) ({vision.PlaceholderMarker}, {placeholder}) " +
                $"after templating but found {placeholdersFound}; this model may not support image input.");
            return 1;
        }

        // Each placeholder expands to [open] + its soft tokens + [close], so the prefill is
        // longer than the token list — an image is easily hundreds of positions. Check the
        // expanded length before the first Forward: ForwardPass sizes its attention-score and
        // RoPE scratch from the context bound but not its KV cache, so running past it writes
        // out of bounds instead of failing.
        int markerTokens = (imgOpen >= 0 ? 1 : 0) + (imgClose >= 0 ? 1 : 0);
        int plannedPrefill = allTokens.Count + (nImages * markerTokens) + totalSoft - nImages;
        if (plannedPrefill >= maxContextLength)
        {
            AnsiConsole.MarkupLine(
                $"[red]Error:[/] prompt plus images expand to {plannedPrefill} tokens ({totalSoft} image) " +
                $"but the active context is {maxContextLength}; use fewer/smaller images or raise " +
                $"[yellow]--ctx-size[/] so at least one token can be generated.");
            return 1;
        }
        sp = ClampToRemainingContext(sp, plannedPrefill, maxContextLength);

        var sw = Stopwatch.StartNew();
        int pos = 0;
        int imgIdx = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in allTokens)
        {
            if (id == placeholder)
            {
                var (soft, nTok) = blocks[imgIdx++];
                if (imgOpen >= 0) logits = fwd.Forward(imgOpen, pos++);
                for (int t = 0; t < nTok; t++)
                    logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
                if (imgClose >= 0) logits = fwd.Forward(imgClose, pos++);
            }
            else
            {
                logits = fwd.Forward(id, pos++);
            }
        }
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        sw.Restart();
        var (generated, totalDecoded) =
            DecodeLoop(fwd.Forward, logits, pos, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens, repeatLastN: s.RepeatLastN);
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {pos} tokens ({totalSoft} image + {pos - totalSoft} text), " +
            $"{pos / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            "[/]");
        return 0;
    }

    /// <summary>Count non-overlapping occurrences of <paramref name="needle"/> in <paramref name="haystack"/>.</summary>
    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, idx = 0;
        while ((idx = haystack.IndexOf(needle, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += needle.Length;
        }
        return count;
    }

    // Decides whether to engage the MTP self-speculative path on the CLI side.
    // Mirrors the InferenceEngine gate but reads `--no-thinking` from CLI settings
    // (vs. the engine which inspects whether the model has think tokens registered).
    // The CLI path can engage MTP even on models that registered <think>/</think>,
    // as long as the user passed --no-thinking — in that case the template renders
    // with enable_thinking=false and no think tokens appear in the prompt or output.
    private static bool ResolveCliMtp(IForwardPass? mtpFwd, SamplingParams sp, bool noThinking, out string? rejectReason)
    {
        rejectReason = null;
        bool envDisabled = Environment.GetEnvironmentVariable("STINGRAY_DISABLE_MTP") == "1";
        bool eligible = mtpFwd is not null
                        && mtpFwd.HasMtpHead
                        && sp.Temperature <= 0f
                        && !sp.HasHistoryPenalty
                        && noThinking
                        && !envDisabled;

        // Spec-decode CLI flag validation:
        //   --spec-draft-p-min: clamped to [0, 1] at the MtpDecoder boundary. Accepted
        //     on any spec path. 1.0 (or 0 / unset) = pure argmax-match; p ∈ (0,1) is
        //     llama.cpp's probabilistic-accept rule from #38. Reject obviously bad input.
        //   --spec-draft-n-min: accepted as a no-op under N=2 batched verify.
        //   --spec-draft-n-max == 2 enables #30 batched verify; > 2 still rejected.
        if (sp.SpecType != SpecType.None)
        {
            if (sp.SpecDraftPMin > 1f)
            {
                rejectReason = $"--spec-draft-p-min={sp.SpecDraftPMin} must be in [0, 1].";
                return false;
            }
        }

        // Max drafts per step = batch capacity − 1 (the certain token rides in the
        // batch). The pass's snapshot-ring capacity bounds the batch (issue #30);
        // without batched verify the sequential path drafts exactly 1 per step.
        int maxDraftN = (mtpFwd is not null && mtpFwd.SupportsBatchVerify)
            ? Math.Max(1, mtpFwd.MaxBatchVerifyTokens - 1)
            : 1;

        switch (sp.SpecType)
        {
            case SpecType.None:
                return false;
            case SpecType.DSpark:
                // The user asked for DSpark; if that path already rejected/fell back
                // upstream, silently swapping in MTP would contradict the printed
                // "falling back to normal generation" warning.
                return false;
            case SpecType.Mtp:
                if (envDisabled) { rejectReason = "--spec-type mtp conflicts with STINGRAY_DISABLE_MTP=1."; return false; }
                if (mtpFwd is null || !mtpFwd.HasMtpHead) { rejectReason = "--spec-type mtp requires a model with an MTP head (nextn tensors)."; return false; }
                if (sp.Temperature > 0f) { rejectReason = "--spec-type mtp requires greedy sampling (--temp 0)."; return false; }
                if (!noThinking) { rejectReason = "--spec-type mtp requires --no-thinking (chat template must render with enable_thinking=false)."; return false; }
                WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return true;
            default: // Auto
                if (eligible)
                    WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return eligible;
        }
    }

    /// <summary>
    /// A draft chain deeper than the snapshot ring's capacity is CLAMPED, not rejected
    /// (rejecting would disable MTP entirely and run SLOWER — the silent-baseline trap
    /// the old SpecDraftNMax&gt;1 throw existed to prevent). Warn so the user knows the
    /// effective depth and the knob that raises it; MtpDecoder clamps per step.
    /// </summary>
    private static void WarnIfDraftNClamped(int requested, int maxDraftN)
    {
        if (requested > maxDraftN)
            AnsiConsole.MarkupLine(
                $"[yellow]Note:[/] --spec-draft-n-max={requested} exceeds the snapshot-ring capacity; " +
                $"running {maxDraftN} draft(s)/step. Set STINGRAY_MTP_BATCH_MAX={requested + 1} to go deeper " +
                "(each ring slot costs ~150 MiB VRAM on 27B-class models).");
    }

    // MTP self-speculative decode path. Reuses the same UTF-8 streaming + EmitToken
    // logic as the baseline DecodeLoop but drives token emission via MtpDecoder.
    // Requires --no-thinking, so no thinking-mode bookkeeping here.
    private static (int generated, int totalDecoded, float acceptanceRate, long accepted, long emitted) DecodeLoopMtp(
        IForwardPass mtpFwd, IReadOnlyList<int> promptTokens, ReadOnlySpan<float> initialLogits,
        GgufTokenizer tok, SamplingParams sp, bool hideThinking, bool verbosePromptLogging = false)
    {
        var mtpDec = new MtpDecoder(mtpFwd);
        mtpDec.Initialize(promptTokens.Count, initialLogits);
        // Populate the MTP KV cache for the full prompt. Cost: ~1.6%/token; only paid
        // on the MTP-enabled run.
        mtpFwd.PrefillMtp(promptTokens, 0);

        var streamDec = new Utf8StreamDecoder();
        bool inThinking = false;
        int generated = 0;
        int totalDecoded = 0;

        // Materialize stop ids once (MtpDecoder takes ReadOnlySpan<int>).
        int[] stopIds = sp.StopTokenIds ?? [];

        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), next =>
        {
            if (verbosePromptLogging)
                Console.Error.WriteLine($"[DBG] tok={totalDecoded} next={next}('{tok.Decode([next])}')");
            totalDecoded++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking)) generated++;
        }, pMin: sp.SpecDraftPMin, draftN: MtpDecoder.ResolveDraftN(sp.SpecDraftNMax),
           ct: CancellationToken.None);

        if (Environment.GetEnvironmentVariable("STINGRAY_TRACE_MTP") == "1" && mtpDec.TotalDraftsEmitted > 0)
            Console.Error.WriteLine(
                $"[mtp] phase ms: draft={mtpDec.DraftMs:F0} verify={mtpDec.VerifyMs:F0} commit={mtpDec.CommitMs:F0}");

        // Flush the UTF-8 decoder tail, applying the same hide-thinking gate as DecodeLoop.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
            Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");

        return (generated, totalDecoded, mtpDec.AcceptanceRate, mtpDec.TotalDraftsAccepted, mtpDec.TotalDraftsEmitted);
    }

    private static int RunInteractive(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        Action resetCache,
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd, int maxContextLength)
    {
        // mtpFwd reserved for interactive MTP wiring (follow-up to #32). Today the
        // interactive loop stays on the baseline decode path; the bench surface and
        // single-prompt runs (RunSinglePrompt above) are what exercise MTP.
        _ = mtpFwd;

        // Full-screen chat UI when attached to a real terminal. It takes over the alternate
        // screen buffer, so a redirected stdin/stdout (scripted use) keeps the plain
        // line-oriented loop below, which stays pipe-friendly.
        //
        // Compiled only when -p:EnableTuiChat=true. The UI depends on OpenTail.TUI, which is not
        // yet published with the API this uses (IModel/IMsg/Cmd, TranscriptEntry, TextInputState,
        // ViewportState, CellBuffer), so the default build has no such dependency and the plain
        // loop below is the interactive path. Nothing else in the CLI touches Tui/.
#if STINGRAY_TUI
        if (Tui.ChatTui.IsSupported && !s.SingleTurn)
        {
            var engine = new Tui.ChatEngine((message, onText, ct) =>
            {
                var prompt = FormatPrompt(message, s.SystemPrompt, enableThinking: !s_noThinking);
                var tokens = tok.Encode(prompt);

                // Same context bound as the single-prompt path: the scratch buffers are sized
                // from it and the KV cache is not, so overrunning is an out-of-bounds write.
                // Decline the turn rather than prefill it; the transcript keeps its history.
                if (tokens.Count >= maxContextLength) return 0;
                var turnSp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

                resetCache();
                var logits = prefill(tokens);

                var (_, totalDecoded) = DecodeLoop(
                    forward, logits, tokens.Count, tok, turnSp, rng,
                    hideThinking: s.HideThinking, maxThinkingTokens: s.MaxThinkingTokens,
                    sink: onText, cancellation: ct, repeatLastN: s.RepeatLastN);

                return totalDecoded;
            });

            return Tui.ChatTui.Run(engine, Path.GetFileName(s.ModelPath) ?? "model");
        }
#endif

        AnsiConsole.MarkupLine("[green]Interactive chat.[/] Type a message, or [yellow]/exit[/] to quit.\n");

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
            var tokens = tok.Encode(prompt);

            // Reject rather than truncate, and stay in the loop so the session survives a single
            // oversized paste. ForwardPass sizes its scratch from this bound but its KV cache is
            // far larger, so prefilling past it would be an out-of-bounds write.
            if (tokens.Count >= maxContextLength)
            {
                AnsiConsole.MarkupLine(
                    $"[red]Error:[/] message is {tokens.Count} tokens but the active context is " +
                    $"{maxContextLength}; shorten it or restart with a larger [yellow]--ctx-size[/].");
                continue;
            }
            var turnSp = ClampToRemainingContext(sp, tokens.Count, maxContextLength);

            resetCache();
            var sw = Stopwatch.StartNew();
            var logits = prefill(tokens);

            sw.Restart();
            var (generated, totalDecoded) = DecodeLoop(forward, logits, tokens.Count, tok, turnSp, rng, hideThinking: s.HideThinking, maxThinkingTokens: s.MaxThinkingTokens, repeatLastN: s.RepeatLastN);
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                "[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static (int generated, int totalDecoded) DecodeLoop(
        Func<int, int, ReadOnlySpan<float>> forward,
        ReadOnlySpan<float> initialLogits,
        int startPos,
        GgufTokenizer tok,
        SamplingParams sp,
        Random rng,
        bool verbosePromptLogging = false,
        bool hideThinking = false,
        int maxThinkingTokens = 0,
        List<int>? captureTokens = null,
        Action<string>? sink = null,
        CancellationToken cancellation = default,
        int repeatLastN = 64)
    {
        var logits = initialLogits;
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        int thinkingTokenCount = 0;
        // Presence/frequency count the WHOLE completion (OpenAI), so history may only be trimmed to
        // the repetition window when neither is active — which is the common case, so the usual
        // memory profile is unchanged. The repetition window itself is applied by the sampler now,
        // not by evicting this list, so the two families no longer share one window.
        int historyCap = (sp.PresencePenalty != 0f || sp.FrequencyPenalty != 0f)
            ? int.MaxValue
            : ResolvePenaltyHistoryCap(repeatLastN);

        // --repeat-last-n and SamplingParams.RepeatLastN encode "no limit" differently. Translate
        // here so the flag's 0/-1 convention stays a CLI concern:
        //   0  → repetition penalty off entirely (neutralise the multiplier)
        //   <0 → whole history        (sampler: 0)
        //   >0 → that many trailing tokens
        float effectiveRepetition = repeatLastN == 0 ? 1.0f : sp.RepetitionPenalty;
        int samplerRepeatLastN = repeatLastN < 0 ? 0 : repeatLastN;
        var recentTokens = new List<int>(Math.Min(historyCap, 256));
        var streamDec = new Utf8StreamDecoder();
        // Tool-call grammar constraint (issue #374): start each response from the watching state so
        // a reused instance serves this generation fresh. No-op when no constraint is attached.
        sp.Constraint?.Reset();
        bool profDecode = OpenTail.Stingray.Engine.DecodeProfileTimers.Enabled;
        for (int i = 0; i < sp.MaxNewTokens; i++)
        {
            // Stop between tokens when the caller cancels (the TUI quitting mid-generation).
            // Breaking rather than throwing keeps whatever has already been decoded, and leaves
            // the KV cache consistent with the tokens actually emitted.
            if (cancellation.IsCancellationRequested) break;

            long nonTrunkStart = profDecode ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            var spWithHistory = (effectiveRepetition != 1.0f || sp.PresencePenalty != 0f || sp.FrequencyPenalty != 0f)
                && recentTokens.Count > 0
                ? sp with
                {
                    PreviousTokens = recentTokens,
                    RepetitionPenalty = effectiveRepetition,
                    RepeatLastN = samplerRepeatLastN,
                }
                : sp;
            int next;
            if (inThinking && maxThinkingTokens > 0 && thinkingTokenCount >= maxThinkingTokens && s_endThinkTokenId > 0)
            {
                // Force </think> to exit a runaway reasoning block; the close tag still
                // goes through forward() below so the model continues from the post-think state.
                next = s_endThinkTokenId;
            }
            else
            {
                // While the grammar is restricting the vocabulary, sample from the masked logits so
                // only a grammar-legal token can be chosen; otherwise sample exactly as before.
                var sampleLogits = sp.Constraint is { IsConstraining: true } ctr ? ctr.Filter(logits) : logits;
                next = sp.Temperature <= 0 && !spWithHistory.HasHistoryPenalty
                    ? Sampler.Greedy(sampleLogits)
                    : Sampler.Sample(sampleLogits, spWithHistory, rng);
            }
            if (verbosePromptLogging)
            {
                Console.Error.WriteLine($"[DBG] tok={i} next={next}('{tok.Decode([next])}') stop={sp.StopTokenIds.Contains(next)} top5:{FormatTopLogits(logits, 5)}");
            }
            if (sp.StopTokenIds.Contains(next)) break;
            // Advance the constraint by every emitted token (so it can detect a tool call beginning
            // and ending) and capture the raw stream for post-generation tool-call parsing.
            sp.Constraint?.Accept(next);
            captureTokens?.Add(next);
            // Counter resets on each <think> open (in case the model opens multiple blocks)
            // and counts every token emitted while inThinking is true on entry, including
            // the boundary tokens themselves — that keeps the budget predictable: N tokens
            // of reasoning content trip the force-close on iteration N+1.
            if (next == s_thinkTokenId) thinkingTokenCount = 0;
            else if (inThinking) thinkingTokenCount++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking, sink)) generated++;
            totalDecoded++;
            recentTokens.Add(next);
            if (recentTokens.Count > historyCap) recentTokens.RemoveAt(0);
            if (profDecode) OpenTail.Stingray.Engine.DecodeProfileTimers.AddNonTrunk(System.Diagnostics.Stopwatch.GetTimestamp() - nonTrunkStart);
            logits = forward(next, startPos + i);
        }
        // When hiding reasoning, the decoder may still hold an in-thinking tail —
        // flush it through the same gate so nothing leaks to stdout.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
        {
            if (sink is null) Console.Write(tail);
            else if (tail.Length > 0) sink(tail);
        }
        // Style reset is stdout-only; a sink receives plain text.
        if (inThinking && sink is null) Console.Write(AnsiCodes.Esc + "[0m");
        return (generated, totalDecoded);
    }

    /// <summary>
    /// Writes <paramref name="next"/> to <paramref name="sink"/> (stdout when null), handling the
    /// &lt;think&gt;/&lt;/think&gt; boundary tokens and dim-styling everything inside. Returns true when
    /// the emitted token counts toward the visible decode total (i.e. not a thinking-mode token
    /// and not a boundary marker).
    /// </summary>
    /// <param name="sink">
    /// Receives decoded text instead of stdout. The interactive TUI passes its own sink so tokens
    /// stream into the transcript; ANSI styling is suppressed for sinks since the TUI applies its
    /// own styling per transcript role.
    /// </param>
    private static bool EmitToken(int next, GgufTokenizer tok, Utf8StreamDecoder streamDec, ref bool inThinking, bool hideThinking = false, Action<string>? sink = null)
    {
        void Emit(string text)
        {
            if (sink is null) Console.Write(text);
            else if (text.Length > 0) sink(text);
        }

        // Styling escapes are stdout-only; a sink receives plain text.
        void EmitStyle(string ansi)
        {
            if (sink is null) Console.Write(ansi);
        }

        // Reasoning boundary tokens are always consumed (never printed), with the state flip
        // gated on the current mode so a malformed double-open or a bare close — e.g. Gemma 4's
        // post-tool prompt primes <|channel>, so the answer pass emits a lone <channel|> with no
        // open — is swallowed rather than rendered as a literal marker (issue #304).
        if (next == s_thinkTokenId)
        {
            if (!inThinking)
            {
                inThinking = true;
                // No trailing \n: the model often emits its own leading newline inside the block,
                // and a double break before the reasoning starts looks noisy.
                EmitStyle(AnsiCodes.Esc + "[2m");
                Emit("[Thinking...] ");
            }
            return false;
        }
        if (next == s_endThinkTokenId)
        {
            if (inThinking)
            {
                inThinking = false;
                EmitStyle(AnsiCodes.Esc + "[0m");
                Emit(((char)10).ToString());
            }
            return false;
        }
        // Stream through the same UTF-8 decoder regardless of mode so multibyte
        // sequences split across thinking/visible boundaries stay intact. When
        // hideThinking is set we still consume the bytes (so the decoder stays in
        // sync across the boundary) but discard the rendered output.
        var rendered = streamDec.Append(tok.DecodeBytes(next));
        if (!(hideThinking && inThinking))
            Emit(rendered);
        return !inThinking;
    }

    /// <summary>
    /// Formats the <paramref name="k"/> highest-logit token candidates as
    /// <c>idx(value) idx(value) …</c> (descending by value, ties broken by lower index) for the
    /// <c>--verbose-prompt</c> debug line. Issue #155: a single O(V·k) pass over the logits span
    /// keeping the k best — no <c>logits.ToArray()</c> copy and no full O(V·logV) vocab sort per
    /// decode token (a ~1 MB alloc + sort on Gemma 4's 262144-token vocab that badly skewed
    /// decode t/s under <c>--verbose-prompt</c>).
    /// </summary>
    internal static string FormatTopLogits(ReadOnlySpan<float> logits, int k)
    {
        k = Math.Min(k, logits.Length);
        if (k <= 0) return "";

        // count-tracked insertion (not a value sentinel) so a real -Infinity logit still occupies
        // a slot and is printed — the old LINQ path showed it, and --verbose-prompt is exactly the
        // tool you reach for when a model emits non-finite garbage (#155 review). Stackalloc only
        // for small k (matches Sampler.FindKthLargest); heap-fall back guards against a large k
        // passed by some other caller stack-overflowing (#155 review).
        Span<float> bestVal = k <= 256 ? stackalloc float[k] : new float[k];
        Span<int> bestIdx = k <= 256 ? stackalloc int[k] : new int[k];
        int count = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            // Skip only when the set is full AND v doesn't outrank the current worst. Ranking
            // matches a stable OrderByDescending: higher value first, NaN sorts last, equal values
            // keep the earlier (lower) index — so a later equal value never displaces.
            if (count == k && !SortsBefore(v, bestVal[k - 1])) continue;

            int pos = count < k ? count : k - 1;
            while (pos > 0 && SortsBefore(v, bestVal[pos - 1]))
            {
                bestVal[pos] = bestVal[pos - 1];
                bestIdx[pos] = bestIdx[pos - 1];
                pos--;
            }
            bestVal[pos] = v;
            bestIdx[pos] = i;
            if (count < k) count++;
        }

        var sb = new System.Text.StringBuilder(k * 12);
        for (int j = 0; j < count; j++)
        {
            if (j > 0) sb.Append(' ');
            sb.Append(bestIdx[j]).Append('(').Append($"{bestVal[j]:F2}").Append(')');
        }
        return sb.ToString();

        // True when a ranks strictly above b in a descending sort (higher value first; NaN is
        // least), matching Comparer<float> as used by OrderByDescending.
        static bool SortsBefore(float a, float b)
        {
            if (float.IsNaN(a)) return false;
            if (float.IsNaN(b)) return true;
            return a > b;
        }
    }

    private static string s_arch = "qwen2"; // set during model load
    // Effective "thinking off" state: --no-thinking OR a model whose recommended config
    // disables reasoning (Gemma 4 E4B-it is not a reasoning model). Set during model load.
    private static bool s_noThinking;
    private static int s_thinkTokenId = -1;    // <think> token for any model using the <think>/</think> special-token convention
    private static int s_endThinkTokenId = -1; // </think> token for any model using the <think>/</think> special-token convention
    private static JinjaChatTemplate? s_jinja;  // parsed from GGUF tokenizer.chat_template
    // Tool definitions loaded from --tools (template-facing object graph: a list of
    // {type, function:{…}} dicts), rendered into the chat template's `tools` variable. Null
    // unless --tools was given, in which case the prompt advertises no tools (legacy behaviour).
    private static IReadOnlyList<object?>? s_tools;

    /// <summary>
    /// Builds the stop token ID list. Delegates to <see cref="GgufTokenizer.EogTokenIds"/> —
    /// the single source of truth for end-of-generation tokens (EOS plus the end-of-turn
    /// variants used by Llama 3/4, Mistral, Phi, Gemma, etc.) — so the CLI and server stop on
    /// exactly the same set. Notably this is what lets the CLI halt on Gemma 4's <c>&lt;eos&gt;</c>
    /// (id 1, distinct from its configured EOS <c>&lt;turn|&gt;</c> at id 106) instead of decoding
    /// it as literal text.
    /// </summary>
    private static IReadOnlyList<int> BuildStopTokenIds(GgufTokenizer tokenizer) => tokenizer.EogTokenIds;

    /// <summary>
    /// Resolves <c>--repeat-last-n</c> into the size the decode loop's history buffer must hold.
    /// </summary>
    /// <remarks>
    /// This exists as a named, testable function because the previous version hard-coded the
    /// buffer at 64 while accepting any value: <c>--repeat-last-n 256</c> and <c>-1</c> both
    /// silently behaved as 64, and the tests could not see it because they only asserted that the
    /// flag BOUND. The buffer size is the behaviour; assert this, not the parsed integer.
    /// </remarks>
    /// <param name="repeatLastN">0 = disabled, -1 (or any negative) = full context, otherwise the window.</param>
    internal static int ResolvePenaltyHistoryCap(int repeatLastN) => repeatLastN switch
    {
        0 => 0,                 // no history retained; the penalty never sees a token
        < 0 => int.MaxValue,    // full context: never evict
        _ => repeatLastN,
    };

    /// <summary>
    /// Process llama.cpp-style escape sequences in <paramref name="s"/>: <c>\n</c>, <c>\t</c>,
    /// <c>\r</c>, <c>\\</c>. Mirrors llama.cpp's <c>-e/--escape</c>. Unknown escapes and a
    /// trailing backslash are left exactly as written.
    /// </summary>
    internal static string ProcessEscapeSequences(string s)
    {
        if (s.IndexOf('\\', StringComparison.Ordinal) < 0) return s;

        var sb = new System.Text.StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '\\' && i + 1 < s.Length)
            {
                switch (s[i + 1])
                {
                    case 'n':  sb.Append('\n'); i++; continue;
                    case 't':  sb.Append('\t'); i++; continue;
                    case 'r':  sb.Append('\r'); i++; continue;
                    case '\\': sb.Append('\\'); i++; continue;
                }
            }
            sb.Append(s[i]);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Parse llama.cpp-style logit-bias entries (<c>TOKEN_ID+BIAS</c> / <c>TOKEN_ID-BIAS</c>,
    /// e.g. <c>1234+1.5</c>, <c>5678-100</c>) into a token-id → bias map. Invariant culture, so a
    /// machine's locale cannot change how a command line is interpreted.
    /// </summary>
    internal static bool TryParseLogitBias(
        string[] entries,
        out IReadOnlyDictionary<int, float>? result,
        out string? error)
    {
        var map = new Dictionary<int, float>(entries.Length);
        foreach (string entry in entries)
        {
            // Scan back for the sign that separates id from bias. Start at 1, not 0, so a leading
            // '-' is treated as part of a (rejected) negative token id rather than as the separator.
            int sep = -1;
            for (int i = entry.Length - 1; i > 0; i--)
                if (entry[i] is '+' or '-') { sep = i; break; }

            if (sep < 0)
            {
                result = null;
                error = $"invalid entry '{entry}': expected TOKEN_ID+BIAS or TOKEN_ID-BIAS (e.g. '1234+1.5').";
                return false;
            }

            if (!int.TryParse(entry[..sep], System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture, out int tokenId) || tokenId < 0)
            {
                result = null;
                error = $"invalid token id '{entry[..sep]}' in '{entry}': must be a non-negative integer.";
                return false;
            }
            // entry[sep..] keeps the sign, so "+1.5" and "-100" both parse directly.
            if (!float.TryParse(entry[sep..], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float bias))
            {
                result = null;
                error = $"invalid bias value '{entry[sep..]}' in '{entry}': must be a number (e.g. +1.5 or -100).";
                return false;
            }
            map[tokenId] = bias;
        }
        result = map;
        error = null;
        return true;
    }

    // ── Tool calling (--tools / --tool-grammar) ────────────────────────────────

    /// <summary>
    /// Loads OpenAI-format tool definitions from a JSON file: a bare array of
    /// <c>{type:"function", function:{name, description, parameters}}</c> objects, or a
    /// <c>{ "tools": [ … ] }</c> wrapper. Returns the template-facing object graph (passed to the
    /// chat template's <c>tools</c> variable) and the parsed <see cref="ToolSchema"/>s (used to build
    /// the argument-grammar constraint). Eagerly materialised so the backing <see cref="JsonDocument"/>
    /// can be disposed here. Throws <see cref="FormatException"/> / <see cref="JsonException"/> on a
    /// malformed file.
    /// </summary>
    private static (IReadOnlyList<object?> Tools, List<ToolSchema> Schemas) LoadTools(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        JsonElement arr = root.ValueKind == JsonValueKind.Object && root.TryGetProperty("tools", out var t) ? t : root;
        if (arr.ValueKind != JsonValueKind.Array)
            throw new FormatException("expected a JSON array of tool definitions (or a { \"tools\": [ … ] } wrapper).");

        var tools = new List<object?>();
        var schemas = new List<ToolSchema>();
        foreach (var el in arr.EnumerateArray())
        {
            tools.Add(JsonToObject(el));   // detached object graph for the template
            if (el.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object
                && fn.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String
                && n.GetString() is { Length: > 0 } name)
            {
                JsonElement? parameters = fn.TryGetProperty("parameters", out var p) ? p : null;
                schemas.Add(ToolSchema.FromOpenAiFunction(name, parameters));
            }
        }
        if (schemas.Count == 0)
            throw new FormatException("no tool with a function.name was found.");
        return (tools, schemas);
    }

    /// <summary>Recursively converts a <see cref="JsonElement"/> into a detached object graph
    /// (<see cref="Dictionary{TKey,TValue}"/> / <see cref="List{T}"/> / scalar) the Jinja engine
    /// consumes — reflection-free, so it survives NativeAOT trimming.</summary>
    private static object? JsonToObject(JsonElement el) => el.ValueKind switch
    {
        JsonValueKind.Object => el.EnumerateObject().ToDictionary(p => p.Name, p => JsonToObject(p.Value), StringComparer.Ordinal),
        JsonValueKind.Array  => el.EnumerateArray().Select(JsonToObject).ToList(),
        JsonValueKind.String => el.GetString(),
        JsonValueKind.Number => el.TryGetInt64(out long l) ? l : el.TryGetDouble(out double d) ? d : (object?)el.GetRawText(),
        JsonValueKind.True   => true,
        JsonValueKind.False  => false,
        _                    => null,
    };

    /// <summary>Parses the captured raw output with the model's tool-call adapter and prints the
    /// structured calls (name + JSON arguments). No-op when the model emitted no tool call.</summary>
    private static void PrintToolCalls(GgufTokenizer tok, List<int> tokens)
    {
        if (tokens.Count == 0) return;
        var (_, calls) = ToolCallAdapterRegistry.Get(s_arch).Parse(tok.Decode(tokens));
        if (calls.Count == 0) return;
        AnsiConsole.MarkupLine($"\n[green]Parsed {calls.Count} tool call(s):[/]");
        foreach (var c in calls)
            AnsiConsole.MarkupLine($"  [bold]{Markup.Escape(c.Name)}[/]({Markup.Escape(RenderArgs(c.Arguments))})");
    }

    /// <summary>Renders a parsed tool call's argument object as compact JSON via a manual
    /// <see cref="Utf8JsonWriter"/> walk (no reflection-based serialization → AOT-safe).</summary>
    private static string RenderArgs(IReadOnlyDictionary<string, object?> args)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var w = new Utf8JsonWriter(buffer))
            WriteJsonValue(w, args);
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static void WriteJsonValue(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null:                                       w.WriteNullValue(); break;
            case bool b:                                     w.WriteBooleanValue(b); break;
            case string s:                                   w.WriteStringValue(s); break;
            case long l:                                     w.WriteNumberValue(l); break;
            case int i:                                      w.WriteNumberValue(i); break;
            case double d:                                   w.WriteNumberValue(d); break;
            case IReadOnlyDictionary<string, object?> map:
                w.WriteStartObject();
                foreach (var kv in map) { w.WritePropertyName(kv.Key); WriteJsonValue(w, kv.Value); }
                w.WriteEndObject();
                break;
            case System.Collections.IEnumerable seq:
                w.WriteStartArray();
                foreach (var item in seq) WriteJsonValue(w, item);
                w.WriteEndArray();
                break;
            default:                                         w.WriteStringValue(value.ToString() ?? ""); break;
        }
    }

    /// <summary>
    /// Resolves whether reasoning ("thinking") should be OFF for this run.
    /// Precedence: <c>--no-thinking</c> forces it off and wins if both flags are passed;
    /// <c>--thinking</c> forces it on; with neither, Gemma 4 defaults off (its stock instruct
    /// models aren't reasoning-trained — nothing in the GGUF metadata distinguishes a reasoning
    /// finetune from a stock model, so reasoning is opt-in there) while every other architecture
    /// defaults on. Pure and side-effect-free so the precedence is unit-testable.
    /// </summary>
    internal static bool ResolveThinkingOff(string arch, bool thinking, bool noThinking)
    {
        if (noThinking) return true;   // explicit off wins over a conflicting --thinking
        if (thinking)   return false;  // explicit on
        return arch == "gemma4";       // default: off only for Gemma 4
    }

    // Accept llama.cpp's "draft-mtp" alongside the shorter "mtp" so existing command
    // lines copy-paste over. Unknown values fall back to auto with a console warning.
    private static SpecType ParseSpecType(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return SpecType.Auto;
        return value.Trim().ToLowerInvariant() switch
        {
            "auto" or "" => SpecType.Auto,
            "none" or "off" or "disabled" => SpecType.None,
            "mtp" or "draft-mtp" => SpecType.Mtp,
            "dspark" => SpecType.DSpark,
            _ => WarnUnknownSpecType(value),
        };

        static SpecType WarnUnknownSpecType(string v)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Unknown --spec-type [yellow]{Markup.Escape(v)}[/]; expected auto|none|mtp|dspark. Falling back to auto.");
            return SpecType.Auto;
        }
    }

    private static string FormatPrompt(string userMessage, string? systemPrompt, bool enableThinking = true)
    {
        // STINGRAY_RAW_PROMPT=1 bypasses the chat template entirely. Used for parity testing
        // against llama.cpp's --no-conversation mode (raw text completion). Not for normal use.
        if (Environment.GetEnvironmentVariable("STINGRAY_RAW_PROMPT") == "1")
            return userMessage;

        // Use the model's own Jinja2 chat template when available (read from GGUF metadata).
        if (s_jinja != null)
        {
            // Qwen3 (dense) chat models behave poorly without a system message — they
            // end the turn after a few tokens for short prompts. The hardcoded fallback
            // path (below) injects this default; mirror it here for the same arch.
            // Note: qwen3moe is intentionally excluded — Qwen3-Coder appears to be
            // tuned to operate without a system prompt and gets HIGH-confidence on
            // <|endoftext|> when one is forced (logit ~29 vs ~14 with no system).
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3" ? "You are a helpful assistant." : null);
            var messages = JinjaChatTemplate.BuildMessages(userMessage, systemContent: effectiveSystemPrompt);
            return s_jinja.Render(new Dictionary<string, object?>
            {
                ["messages"]             = messages,
                ["add_generation_prompt"] = true,
                ["tools"]                = (object?)s_tools,
                ["enable_thinking"]      = enableThinking,
            });
        }

        // Fallback: hardcoded templates for known architectures.
        var sb = new System.Text.StringBuilder();

        if (s_arch is "llama4")
        {
            // Llama 4: <|begin_of_text|><|header_start|>role<|header_end|>\n\nmessage<|eot|>
            sb.Append("<|begin_of_text|>");
            if (systemPrompt is not null)
                sb.Append($"<|header_start|>system<|header_end|>\n\n{systemPrompt}<|eot|>");
            sb.Append($"<|header_start|>user<|header_end|>\n\n{userMessage}<|eot|>");
            sb.Append("<|header_start|>assistant<|header_end|>\n\n");
        }
        else if (s_arch is "llama")
        {
            // Llama 3/3.1: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\nmessage<|eot_id|>
            sb.Append("<|begin_of_text|>");
            if (systemPrompt is not null)
                sb.Append($"<|start_header_id|>system<|end_header_id|>\n\n{systemPrompt}<|eot_id|>");
            sb.Append($"<|start_header_id|>user<|end_header_id|>\n\n{userMessage}<|eot_id|>");
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        else
        {
            // ChatML (Qwen, SmolLM, default): <|im_start|>role\nmessage<|im_end|>
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3moe" or "qwen3" ? "You are a helpful assistant." : null);
            if (effectiveSystemPrompt is not null)
                sb.Append($"<|im_start|>system\n{effectiveSystemPrompt}<|im_end|>\n");
            sb.Append($"<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
