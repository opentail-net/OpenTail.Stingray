using OpenTail.Stingray.Core;
using OpenTail.Stingray.Server;

// Same guard as the CLI: an STINGRAY_* name the engine never reads is indistinguishable from
// "unset", so the process starts having silently ignored the operator's configuration. That is
// worse here than on the CLI — a server runs for days, and the misconfiguration is only noticed
// as unexplained behaviour. Written to stderr before the host starts so it appears even when the
// logging pipeline is reconfigured. Warn, never fail: an unknown name may belong to a different
// OpenTail version, and refusing to boot over it would be worse than the typo.
foreach (string unknown in KnownEnvironmentVariables.FindUnknown())
{
    string? suggestion = KnownEnvironmentVariables.SuggestClosest(unknown);
    Console.Error.WriteLine(suggestion is null
        ? $"warning: {unknown} is set but is not read by this build — it will have no effect."
        : $"warning: {unknown} is set but is not read by this build — did you mean {suggestion}?");
}

var builder = WebApplication.CreateSlimBuilder(args);

// Per-developer overrides (not committed) layered on top of appsettings.json. Anything you
// don't want in git — local model paths, credentials, port pinning — goes here. The file is
// listed in .gitignore so it never accidentally ships.
builder.Configuration.AddJsonFile("appsettings.Local.json", optional: true, reloadOnChange: true);

// Bind OpenTailStingrayServerOptions from the "OpenTail.Stingray" config section first,
// then layer environment-variable overrides for backward compatibility with the original
// STINGRAY_MODEL / STINGRAY_MAX_BATCH knobs. Inline configure runs last → wins.
builder.Services.AddOpenTailStingray(builder.Configuration, opts =>
{
    var envModel = Environment.GetEnvironmentVariable("STINGRAY_MODEL");
    if (!string.IsNullOrWhiteSpace(envModel))
        opts.ModelPath = envModel;

    // Multimodal projector for image input (issue #253). Mirrors the CLI's --mmproj.
    var envMmproj = Environment.GetEnvironmentVariable("STINGRAY_MMPROJ");
    if (!string.IsNullOrWhiteSpace(envMmproj))
        opts.MmprojPath = envMmproj;

    if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_MAX_BATCH"), out int maxBatch) && maxBatch > 0)
        opts.MaxBatchSize = maxBatch;

    if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_MAX_QUEUE"), out int maxQueue) && maxQueue >= 0)
        opts.MaxQueuedRequests = maxQueue;

    // Legacy fixed in-flight cap. When set it overrides MaxQueuedRequests; otherwise the
    // bounded active+waiting queue controls overload.
    if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_MAX_CONCURRENT"), out int maxConcurrent) && maxConcurrent > 0)
        opts.MaxConcurrentRequests = maxConcurrent;

    // Continuous-batching scheduling knobs (issue #183): prefill chunk size (Gap 1)
    // and KV admission budget in MiB (Gap 3). Same precedence as STINGRAY_MAX_BATCH.
    if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_PREFILL_CHUNK"), out int prefillChunk) && prefillChunk >= 0)
        opts.PrefillChunkTokens = prefillChunk;

    if (long.TryParse(Environment.GetEnvironmentVariable("STINGRAY_KV_BUDGET_MB"), out long kvBudgetMb) && kvBudgetMb != 0)
        opts.KvBudgetMb = kvBudgetMb;

    if (long.TryParse(Environment.GetEnvironmentVariable("STINGRAY_PREFIX_CACHE_MB"), out long prefixCacheMb))
        opts.PrefixCacheMb = prefixCacheMb;

    // Dequant-once BLAS weight-cache budget in MiB (issue #189). null/unset = auto.
    if (long.TryParse(Environment.GetEnvironmentVariable("STINGRAY_PREFILL_DEQUANT_MB"), out long dequantMb))
        opts.PrefillDequantCacheMb = dequantMb;

    // STINGRAY_BACKEND ∈ {auto, cpu, cuda, vulkan} — case-insensitive. Lets a
    // smoke test or ad-hoc run override the appsettings.Local.json backend
    // without editing the file (matches the STINGRAY_MODEL pattern above).
    var envBackend = Environment.GetEnvironmentVariable("STINGRAY_BACKEND");
    if (!string.IsNullOrWhiteSpace(envBackend)
        && Enum.TryParse<OpenTail.Stingray.Server.ServerBackend>(envBackend, ignoreCase: true, out var backend))
    {
        opts.Backend = backend;
    }

    if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_N_GPU_LAYERS"), out int nGpuLayers))
        opts.NGpuLayers = nGpuLayers;

    // STINGRAY_KV_DTYPE ∈ {fp32, bf16, q8_0} — CUDA dense KV-cache element type (#179).
    // Mirrors the STINGRAY_MODEL/STINGRAY_BACKEND override pattern; the loader forwards it
    // back to the env var the forward pass reads. Validated at model load.
    var envKvType = Environment.GetEnvironmentVariable("STINGRAY_KV_DTYPE");
    if (!string.IsNullOrWhiteSpace(envKvType))
        opts.KvType = envKvType;

    // STINGRAY_TQ ∈ {1, true} enables TurboQuant KV-cache compression (mirrors --tq);
    // STINGRAY_TQ_MODE ∈ {auto, kvarn, lloydmax} picks the quantizer (mirrors --tq-mode,
    // default auto — issue #432: KVarN where supported, Lloyd-Max fallback with warning).
    var envTq = Environment.GetEnvironmentVariable("STINGRAY_TQ");
    if (!string.IsNullOrWhiteSpace(envTq)
        && (envTq == "1" || envTq.Equals("true", StringComparison.OrdinalIgnoreCase)))
    {
        opts.TurboQuant = true;
    }
    var envTqMode = Environment.GetEnvironmentVariable("STINGRAY_TQ_MODE");
    if (!string.IsNullOrWhiteSpace(envTqMode))
        opts.TqMode = envTqMode;

    // STINGRAY_NO_THINKING ∈ {1, true} globally disables reasoning (server-side --no-thinking),
    // for agentic clients that never send the per-request opt-out.
    var envNoThink = Environment.GetEnvironmentVariable("STINGRAY_NO_THINKING");
    if (!string.IsNullOrWhiteSpace(envNoThink)
        && (envNoThink == "1" || envNoThink.Equals("true", StringComparison.OrdinalIgnoreCase)))
    {
        opts.DisableThinking = true;
    }

    // STINGRAY_PRESERVE_THINKING ∈ {1, true} globally keeps prior assistant turns' reasoning in
    // the chat-template history instead of stripping it, for agentic clients that never send the
    // per-request preserve_thinking opt-in.
    var envPreserveThink = Environment.GetEnvironmentVariable("STINGRAY_PRESERVE_THINKING");
    if (!string.IsNullOrWhiteSpace(envPreserveThink)
        && (envPreserveThink == "1" || envPreserveThink.Equals("true", StringComparison.OrdinalIgnoreCase)))
    {
        opts.PreserveThinking = true;
    }

    // STINGRAY_TOOL_GRAMMAR ∈ {1, true} enables schema/grammar-constrained tool-call argument
    // decoding (issue #374). Off by default → byte-identical to unconstrained decoding.
    var envToolGrammar = Environment.GetEnvironmentVariable("STINGRAY_TOOL_GRAMMAR");
    if (!string.IsNullOrWhiteSpace(envToolGrammar)
        && (envToolGrammar == "1" || envToolGrammar.Equals("true", StringComparison.OrdinalIgnoreCase)))
    {
        opts.ToolGrammar = true;
    }
});

var app = builder.Build();

app.MapOpenTailStingray();

app.Run();

// Required for WebApplicationFactory<Program> in integration tests.
public partial class Program { }
