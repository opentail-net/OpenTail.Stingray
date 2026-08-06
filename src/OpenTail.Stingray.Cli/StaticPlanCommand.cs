using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cuda;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Read-only GGUF and runtime planning report.  This command opens only the GGUF index
/// (memory mapped); it does not load weights, allocate a KV cache, or enter inference.
/// </summary>
public sealed class StaticPlanCommand : Command<StaticPlanCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("Path to a GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("--profile <PATH>")]
        [Description("Optional JSON planning profile; CLI values override profile, environment, then defaults")]
        public string? ProfilePath { get; init; }

        [CommandOption("--save-profile <PATH>")]
        [Description("Write the resolved strict planning profile; may be used without a model")]
        public string? SaveProfilePath { get; init; }

        [CommandOption("--backend <NAME>")]
        [Description("Backend preference: auto, cpu, cuda, or vulkan")]
        public string? Backend { get; init; }

        [CommandOption("--device <NAME>")]
        [Description("Requested device (none forces CPU; a named/indexed GPU is reported but not selected by inspect)")]
        public string? Device { get; init; }

        [CommandOption("--target <NAME>")]
        [Description("Eligibility target: cli (default) or server")]
        public string? Target { get; init; }

        [CommandOption("-g|--gpu-layers <N>")]
        [Description("GPU layers: 0 = CPU, -1 = planner-selected, omitted = default 0")]
        public int? GpuLayers { get; init; }

        [CommandOption("-c|--ctx-size <N>")]
        [Description("Context size (0 = planner/model default)")]
        public int? ContextSize { get; init; }

        [CommandOption("--tq <BOOL>")]
        [Description("Whether TurboQuant KV compression is requested")]
        public bool? TurboQuant { get; init; }

        [CommandOption("--tq-mode <NAME>")]
        [Description("TurboQuant mode: auto, kvarn, or lloydmax")]
        public string? TqMode { get; init; }

        [CommandOption("--kv-type <NAME>")]
        [Description("KV element type: fp32, bf16, or q8_0")]
        public string? KvType { get; init; }

        [CommandOption("--spec-type <NAME>")]
        [Description("Speculation type: auto, none, or mtp")]
        public string? SpecType { get; init; }

        [CommandOption("--max-batch <N>")]
        [Description("Requested maximum batch size")]
        public int? MaxBatchSize { get; init; }

        [CommandOption("--tool-grammar <BOOL>")]
        [Description("Whether tool grammar is requested")]
        public bool? ToolGrammar { get; init; }

        [CommandOption("--no-gpu-probe")]
        [Description("Do not initialize CUDA/Vulkan; report GPU availability as not probed")]
        public bool NoGpuProbe { get; init; }

        [CommandOption("--json")]
        [Description("Write machine-readable JSON to stdout")]
        public bool Json { get; init; }

        [CommandOption("--print-effective-config")]
        [Description("Print the resolved planning configuration and exit; a model is not required")]
        public bool PrintEffectiveConfig { get; init; }

        [CommandOption("--print-profile-schema")]
        [Description("Write the strict JSON Schema for --profile and exit; a model is not required")]
        public bool PrintProfileSchema { get; init; }

        [CommandOption("--explain")]
        [Description("Include the full selected/rejected decision trace in text output")]
        public bool Explain { get; init; }

        public override string? Validate() => ModelPath is null && !PrintEffectiveConfig && !PrintProfileSchema && SaveProfilePath is null ? "Use -m <model.gguf>." : null;
    }

    protected override int Execute(Settings settings, CancellationToken cancellation) => Execute(settings, inspectOnly: false);

    internal static int Execute(Settings settings, bool inspectOnly)
    {
        try
        {
            if (settings.PrintProfileSchema)
            {
                Console.WriteLine(StaticPlanProfileSchema.Json);
                return 0;
            }
            var profile = StaticPlanProfile.Load(settings.ProfilePath);
            var config = StaticPlanConfiguration.Resolve(settings, profile, Environment.GetEnvironmentVariable);
            if (settings.SaveProfilePath is { } savePath)
            {
                var resolvedProfile = StaticPlanProfile.FromEffectiveConfiguration(config);
                File.WriteAllText(savePath, JsonSerializer.Serialize(resolvedProfile, StaticPlanJsonContext.Indented.StaticPlanProfile));
                Console.Error.WriteLine($"Saved resolved planning profile: {Path.GetFullPath(savePath)}");
                if (settings.ModelPath is null)
                {
                    if (settings.Json)
                        Console.WriteLine(JsonSerializer.Serialize(config, StaticPlanJsonContext.Indented.EffectiveConfigurationSnapshot));
                    else
                        WriteEffectiveConfiguration(config);
                    return 0;
                }
            }
            if (settings.PrintEffectiveConfig)
            {
                if (settings.Json)
                    Console.WriteLine(JsonSerializer.Serialize(config, StaticPlanJsonContext.Indented.EffectiveConfigurationSnapshot));
                else
                    WriteEffectiveConfiguration(config);
                return 0;
            }
            using var model = GgufModel.Open(settings.ModelPath!);
            var report = StaticPlanReport.Create(settings.ModelPath!, model, config, settings.NoGpuProbe, includePlacement: !inspectOnly);
            if (settings.Json)
            {
                if (inspectOnly)
                {
                    var inspect = StaticInspectReport.FromPlan(report);
                    Console.WriteLine(JsonSerializer.Serialize(inspect, StaticPlanJsonContext.Indented.StaticInspectReport));
                }
                else
                    Console.WriteLine(JsonSerializer.Serialize(report, StaticPlanJsonContext.Indented.StaticPlanReport));
            }
            else
            {
                if (inspectOnly) StaticInspectReport.WriteHuman(StaticInspectReport.FromPlan(report));
                else StaticPlanReport.WriteHuman(report, settings.Explain);
            }
            return report.Compatibility.Selected ? 0 : 2;
        }
        catch (Exception ex) when (ex is IOException or JsonException or NotSupportedException or ArgumentException or FormatException)
        {
            Console.Error.WriteLine($"plan: {ex.Message}");
            return 1;
        }
    }

    private static void WriteEffectiveConfiguration(EffectiveConfigurationSnapshot configuration)
    {
        foreach (var (name, value) in configuration.Values.OrderBy(x => x.Key, StringComparer.Ordinal))
            Console.WriteLine($"{name} = {value.Value.GetRawText()} ({value.Source})");
        foreach (var diagnostic in configuration.Diagnostics)
            Console.WriteLine($"{diagnostic.Kind}: {diagnostic.Field}: {diagnostic.Message}");
    }
}

internal static class StaticPlanProfileSchema
{
    // Kept beside the strict profile record so the accepted surface is reviewable without
    // a reflection-based schema package (the CLI is NativeAOT-published).
    internal const string Json = """
        {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
          "title": "OpenTail.Stingray static planning profile",
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "backend": { "type": "string", "enum": ["auto", "cpu", "cuda", "vulkan"] },
            "device": { "type": "string" },
            "target": { "type": "string", "enum": ["cli", "server"] },
            "gpu_layers": { "type": "integer", "minimum": -1 },
            "context_size": { "type": "integer", "minimum": 0 },
            "turbo_quant": { "type": "boolean" },
            "tq_mode": { "type": "string", "enum": ["auto", "kvarn", "lloydmax"] },
            "kv_type": { "type": "string", "enum": ["fp32", "bf16", "q8_0"] },
            "spec_type": { "type": "string", "enum": ["auto", "none", "mtp"] },
            "max_batch_size": { "type": "integer", "minimum": 1 },
            "tool_grammar": { "type": "boolean" },
            "snap_kv_budget": { "type": "integer", "minimum": 0 }
          }
        }
        """;
}

/// <summary>JSON profile intentionally covers only static planning knobs, never credentials or prompts.</summary>
 [JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
public sealed record StaticPlanProfile(
    string? Backend = null, string? Device = null, string? Target = null, int? GpuLayers = null, int? ContextSize = null,
    bool? TurboQuant = null, string? TqMode = null, string? KvType = null,
    string? SpecType = null, int? MaxBatchSize = null, bool? ToolGrammar = null, int? SnapKvBudget = null)
{
    public static StaticPlanProfile? Load(string? path) => path is null ? null :
        JsonSerializer.Deserialize(File.ReadAllText(path), StaticPlanJsonContext.Default.StaticPlanProfile)
        as StaticPlanProfile ?? throw new JsonException("Profile was empty.");

    internal static StaticPlanProfile FromEffectiveConfiguration(EffectiveConfigurationSnapshot config) => new(
        Backend: config.Get<string>("backend"),
        Device: config.Get<string>("device"),
        Target: config.Get<string>("target"),
        GpuLayers: config.Get<int>("gpu_layers"),
        ContextSize: config.Get<int>("context_size"),
        TurboQuant: config.Get<bool>("turbo_quant"),
        TqMode: config.Get<string>("tq_mode"),
        KvType: config.Get<string>("kv_type"),
        SpecType: config.Get<string>("spec_type"),
        MaxBatchSize: config.Get<int>("max_batch"),
        ToolGrammar: config.Get<bool>("tool_grammar"),
        SnapKvBudget: config.Get<int>("snapkv_budget") is -1 ? null : config.Get<int>("snapkv_budget"));
}

internal static class StaticPlanConfiguration
{
    internal static EffectiveConfigurationSnapshot Resolve(StaticPlanCommand.Settings cli, StaticPlanProfile? profile,
        Func<string, string?> environment)
    {
        return EffectiveConfigurationResolver.Resolve(
        [
            Setting("backend", cli.Backend, profile?.Backend, environment("STINGRAY_BACKEND"), "auto"),
            Setting("device", cli.Device, profile?.Device, null, "auto"),
            Setting("target", cli.Target, profile?.Target, null, "cli"),
            Setting("gpu_layers", cli.GpuLayers, profile?.GpuLayers, environment("STINGRAY_N_GPU_LAYERS"), 0),
            // Run has --ctx-size and --spec-type, but does not consume environment variables
            // for either. Do not invent environment compatibility in a diagnostic report.
            Setting("context_size", cli.ContextSize, profile?.ContextSize, null, 0),
            Setting("turbo_quant", cli.TurboQuant, profile?.TurboQuant, environment("STINGRAY_TQ"), false),
            Setting("tq_mode", cli.TqMode, profile?.TqMode, environment("STINGRAY_TQ_MODE"), "auto"),
            Setting("kv_type", cli.KvType, profile?.KvType, environment("STINGRAY_KV_DTYPE"), "fp32"),
            Setting("spec_type", cli.SpecType, profile?.SpecType, null, "auto"),
            Setting("max_batch", cli.MaxBatchSize, profile?.MaxBatchSize, environment("STINGRAY_MAX_BATCH"), 1),
            Setting("tool_grammar", cli.ToolGrammar, profile?.ToolGrammar, environment("STINGRAY_TOOL_GRAMMAR"), false),
            Setting("snapkv_budget", null, profile?.SnapKvBudget, environment("STINGRAY_SNAPKV_BUDGET"), -1),
        ]);

        static EffectiveConfigurationSetting Setting(string name, object? cli, object? profile, string? environment, object defaultValue) =>
            new(name, defaultValue,
            [new("cli", cli), new("profile", profile), new("environment", environment)]);
    }
}

public sealed record StaticPlanDecision(string Area, bool Selected, string Why);
public sealed record StaticPlanBackend(string Name, string Status, long? VramBytes, long? FreeVramBytes, string Detail);
public sealed record StaticPlanModel(string Path, string Filename, uint GgufVersion, string Architecture,
    int TensorCount, int MetadataCount, int NumLayers, int ContextLength, int HeadDim, int NumKvHeads,
    bool IsMoE, int NumMtpLayers, string? Name, string? FileType, string? QuantizationVersion,
    bool HasChatTemplate, bool HasReasoningTokens, bool SupportsVisionInput, long ParameterElements,
    int VocabularySize, IReadOnlyDictionary<string, int> TensorDtypes);
public sealed record StaticPlanHardware(long RamBytes, int CpuCores, bool Avx512, string Detail);
public sealed record StaticPlanPlacement(string SelectedBackend, int GpuLayers, int CpuLayers,
    long GpuWeightBytes, long GpuKvBytes, int RecommendedCtxSize, long CpuWeightBytes, string Rationale);

/// <summary>Process-boundary hardware facts shared by inspect, plan, and doctor.</summary>
internal sealed record StaticPlanRuntimeFacts(
    IReadOnlyList<StaticPlanBackend> Backends, StaticPlanHardware Hardware, HardwareProfile HardwareProfile)
{
    internal static StaticPlanRuntimeFacts Detect(bool noGpuProbe)
    {
        bool cuda = false;
        long cudaVram = 0, cudaFreeVram = 0;
        string cudaDetail = noGpuProbe ? "Not probed (--no-gpu-probe)." : "No usable CUDA runtime/device detected.";
        if (!noGpuProbe && CudaBackend.IsAvailable())
        {
            try
            {
                using var backend = CudaBackend.Create();
                cuda = true;
                cudaVram = (long)backend.VramBytes;
                cudaFreeVram = (long)backend.FreeVramBytes;
                cudaDetail = $"{backend.Name}; {cudaVram / (1024.0 * 1024 * 1024):F1} GiB VRAM detected.";
            }
            catch (Exception ex) { cudaDetail = ex.GetType().Name + ": " + ex.Message; }
        }

        bool vulkan = false;
        long vulkanVram = 0;
        string vkDetail = noGpuProbe ? "Not probed (--no-gpu-probe)." : "No usable Vulkan device detected.";
        if (!noGpuProbe)
        {
            try
            {
                using var backend = new VulkanBackend();
                vulkan = true;
                vulkanVram = (long)backend.VramBytes;
                vkDetail = $"{backend.Name}; {vulkanVram / (1024.0 * 1024 * 1024):F1} GiB placement budget detected.";
            }
            catch (Exception ex) { vkDetail = ex.GetType().Name + ": " + ex.Message; }
        }

        HardwareProfile hardware = HardwareProfile.Detect();
        return new(
        [
            new StaticPlanBackend("cpu", "available", null, null, "Portable CPU backend is always available."),
            new StaticPlanBackend("cuda", noGpuProbe ? "not_probed" : cuda ? "available" : "unavailable", cuda ? cudaVram : null, cuda ? cudaFreeVram : null, cudaDetail),
            new StaticPlanBackend("vulkan", noGpuProbe ? "not_probed" : vulkan ? "available" : "unavailable", vulkan ? vulkanVram : null, null, vkDetail),
        ], new StaticPlanHardware(hardware.RamBytes, hardware.CpuCores, hardware.HasAvx512, hardware.Summary()), hardware);
    }
}

public sealed record StaticPlanReport(
    int SchemaVersion, StaticPlanModel Model, StaticPlanDecision Compatibility,
    IReadOnlyList<StaticPlanBackend> Backends, StaticPlanHardware Hardware,
    EffectiveConfigurationSnapshot EffectiveConfiguration, StaticPlanPlacement? Placement,
    IReadOnlyList<StaticPlanDecision> Decisions,
    ExecutionPlan ExecutionPlan)
{
    internal static StaticPlanReport Create(string path, GgufModel model, EffectiveConfigurationSnapshot config, bool noGpuProbe, bool includePlacement)
    {
        string arch = model.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) ?? "unknown" : "llama";
        ModelHyperparams hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        StaticPlanDecision compatibility;
        try { ModelCompatibility.ValidateForTextGeneration(model); compatibility = new("gguf_compatibility", true, "Architecture and tensor storage types are supported."); }
        catch (NotSupportedException ex) { compatibility = new("gguf_compatibility", false, ex.Message); }

        var runtimeFacts = StaticPlanRuntimeFacts.Detect(noGpuProbe);
        var backends = runtimeFacts.Backends;
        bool cuda = backends.Any(x => x.Name == "cuda" && x.Status == "available");
        bool vulkan = backends.Any(x => x.Name == "vulkan" && x.Status == "available");
        long cudaVram = backends.First(x => x.Name == "cuda").VramBytes ?? 0;
        long vulkanVram = backends.First(x => x.Name == "vulkan").VramBytes ?? 0;
        int gpuLayers = config.Get<int>("gpu_layers");
        string device = config.Get<string>("device").Trim();
        bool deviceNone = string.Equals(device, "none", StringComparison.OrdinalIgnoreCase);
        if (deviceNone) gpuLayers = 0;
        string requested = config.Get<string>("backend").ToLowerInvariant();
        string selected = requested == "auto" ? (gpuLayers == 0 ? "cpu" : cuda ? "cuda" : vulkan ? "vulkan" : "cpu") : requested;
        var decisions = new List<StaticPlanDecision>();
        bool selectedAvailable = backends.Any(x => x.Name == selected && x.Status == "available");
        bool backendNameValid = requested is "auto" or "cpu" or "cuda" or "vulkan";
        decisions.Add(new("backend", backendNameValid && selectedAvailable, !backendNameValid ? $"Rejected unknown backend '{requested}'." : selectedAvailable ? $"Selected {selected} from {config.Values["backend"].Source}." : $"Rejected {selected}: it is unavailable."));
        var configurationDiagnostics = config.Diagnostics.ToList();
        if (!backendNameValid)
            configurationDiagnostics.Add(new("invalid", "backend", "Expected auto, cpu, cuda, or vulkan."));
        if (deviceNone && config.Get<int>("gpu_layers") != 0)
            configurationDiagnostics.Add(new("ignored", "gpu_layers", "--device none forces CPU placement."));
        else if (!string.Equals(device, "auto", StringComparison.OrdinalIgnoreCase))
            configurationDiagnostics.Add(new("conditional", "device", "Inspect reports the default CUDA/Vulkan device; a named/indexed device is selected only by run/server."));
        bool tq = config.Get<bool>("turbo_quant");
        string tqMode = config.Get<string>("tq_mode").Trim().ToLowerInvariant();
        bool tqModeValid = tqMode is "auto" or "kvarn" or "lloydmax" or "lloyd-max";
        int snapKvBudget = config.Get<int>("snapkv_budget");
        bool snapKvEnabled = snapKvBudget > 0;
        string? kvarnBlocked = TqSupport.KVarNBlockedReason(hp.HeadDim, snapKvEnabled,
            onGpu: selected != "cpu", isVulkan: selected == "vulkan", cudaAvailable: cuda, isMoE: hp.IsMoE);
        bool kvarn = tq && (tqMode == "kvarn" || (tqMode == "auto" && kvarnBlocked is null));
        bool lloyd = tq && (tqMode is "lloydmax" or "lloyd-max" || (tqMode == "auto" && kvarnBlocked is not null));
        bool tqApplicable = tq && tqModeValid && selectedAvailable &&
            ((kvarn && kvarnBlocked is null) || (lloyd && TqSupport.IsLloydMaxHeadDim(hp.HeadDim)));
        string tqWhy = !tq ? "Not selected: TurboQuant is disabled."
            : !tqModeValid ? $"Rejected unknown TurboQuant mode '{tqMode}'."
            : kvarn && kvarnBlocked is null ? "Selected: KVarN is eligible on this model/backend."
            : lloyd && TqSupport.IsLloydMaxHeadDim(hp.HeadDim) ? $"Selected: Lloyd-Max is eligible ({(tqMode == "auto" ? "KVarN was unavailable: " + kvarnBlocked : "explicitly requested")})."
            : $"Rejected: Lloyd-Max requires head dim 128 or 256; this model has {hp.HeadDim}.";
        decisions.Add(new("kv_turbo_quant", tqApplicable, tqWhy));
        if (tq && !tqApplicable)
            configurationDiagnostics.Add(new("inapplicable", "turbo_quant", tqWhy));
        if (!tqModeValid)
            configurationDiagnostics.Add(new("invalid", "tq_mode", "Expected auto, kvarn, or lloydmax."));
        if (!tq && tqMode is "kvarn" or "lloydmax" or "lloyd-max")
            configurationDiagnostics.Add(new("ignored", "tq_mode", "TurboQuant is disabled, so its quantizer selection has no effect."));
        decisions.Add(new("snapkv", snapKvEnabled, snapKvBudget > 0
            ? $"Selected: explicit budget {snapKvBudget}."
            : snapKvBudget == 0 ? "Not selected: explicitly disabled."
            : "Not explicitly configured; supported CUDA paths may choose an automatic budget."));
        string kvType = config.Get<string>("kv_type").ToLowerInvariant();
        DType kvDtype = kvType switch
        {
            "fp32" => DType.Float32,
            "bf16" => DType.BFloat16,
            "q8_0" => DType.Q8_0,
            _ => DType.Float32,
        };
        if (kvType is not ("fp32" or "bf16" or "q8_0"))
            configurationDiagnostics.Add(new("invalid", "kv_type", "Expected fp32, bf16, or q8_0."));
        if (kvType != "fp32" && selected != "cuda")
            configurationDiagnostics.Add(new("inapplicable", "kv_type", $"KV type '{kvType}' applies only to CUDA and is ignored for '{selected}'."));
        string spec = config.Get<string>("spec_type").ToLowerInvariant();
        bool mtp = hp.NumMtpLayers > 0;
        bool speculationSelected = mtp && (spec is "auto" or "mtp");
        decisions.Add(new("speculation", speculationSelected, spec == "none" ? "Not selected." : mtp ? "MTP head detected; eligible for MTP speculation." : "No MTP head detected; MTP is unavailable."));
        if (spec == "mtp" && !mtp)
            configurationDiagnostics.Add(new("inapplicable", "spec_type", "MTP was requested but this GGUF has no MTP head."));
        string target = config.Get<string>("target").Trim().ToLowerInvariant();
        bool targetValid = target is "cli" or "server";
        if (!targetValid) configurationDiagnostics.Add(new("invalid", "target", "Expected cli or server."));
        int batch = config.Get<int>("max_batch");
        bool batchingSelected = target == "server" && batch > 1;
        decisions.Add(new("batching", batchingSelected, target == "cli" && batch > 1 ? "Ignored: the CLI does not construct continuous batching." : batchingSelected ? "Selected: continuous batching requested." : "Not selected: max_batch is 1."));
        if (target == "cli" && batch > 1)
            configurationDiagnostics.Add(new("ignored", "max_batch", "Continuous batching is a server-only setting."));
        bool grammar = config.Get<bool>("tool_grammar");
        decisions.Add(new("tool_grammar", grammar && !batchingSelected, !grammar ? "Not selected." : batchingSelected ? "Rejected: tool grammar is incompatible with continuous batching." : "Eligible subject to the selected tool schema and model template."));
        if (grammar && batchingSelected)
            configurationDiagnostics.Add(new("inapplicable", "tool_grammar", "Tool grammar is ignored when continuous batching is selected."));
        // This is deliberately a product-surface report, not a claim about internal cache types.
        // The sessions assembly contains experimental retained-state primitives, but neither the
        // CLI nor the HTTP host can yet open, persist, then resume a named inference session after
        // a process restart. Saying "available" merely because PagedKvCache can export bytes would
        // turn a useful capability report into an accidental promise.
        decisions.Add(new("session_restart_continuation", false,
            "Not exposed by the CLI or server yet; retained-session persistence is experimental and requires an end-to-end restart proof before it is a supported deployment feature."));

        StaticPlanPlacement? placement = null;
        HardwareProfile hardware = runtimeFacts.HardwareProfile;
        if (includePlacement && compatibility.Selected)
        {
            // No PCIe benchmark or model upload occurs here. The planner uses the detected
            // capacity, while the report marks PCIe as an estimate rather than measurement.
            // A rejected backend cannot be presented as the selected placement. Report a CPU
            // baseline instead; the backend decision preserves the requested/rejected reason.
            string placementBackend = selectedAvailable ? selected : "cpu";
            long selectedVram = placementBackend == "cuda" ? cudaVram : placementBackend == "vulkan" ? vulkanVram : 0;
            var planHardware = new HardwareProfile(selectedAvailable ? selectedVram : 0,
                hardware.RamBytes, hardware.CpuCores, 0, hardware.HasAvx512);
            // A rejected request must not leak into the plan: show the executable baseline
            // and leave the rejection in Decisions/Diagnostics instead of claiming a layout
            // the runtime will refuse to construct.
            DType plannerKvDtype = placementBackend == "cuda" && (kvType is "fp32" or "bf16" or "q8_0")
                ? kvDtype : DType.Float32;
            var placementPlan = TierPlanner.Plan(model, hp, planHardware, tqApplicable,
                requestedCtxSize: config.Get<int>("context_size"), kvDtype: plannerKvDtype,
                pinGpuLayers: gpuLayers < 0 ? null : gpuLayers);
            placement = new StaticPlanPlacement(placementBackend, placementPlan.GpuLayers, placementPlan.CpuLayers,
                placementPlan.GpuWeightBytes, placementPlan.GpuKvBytes, placementPlan.RecommendedCtxSize,
                placementPlan.CpuWeightBytes, !tqApplicable && tq
                    ? "Baseline placement; requested TurboQuant was rejected (see decisions)."
                    : !selectedAvailable ? $"CPU baseline because requested backend '{selected}' is unavailable (see decisions)."
                    : placementBackend == "cpu" ? "CPU placement calculated from GGUF index."
                    : "Placement uses detected VRAM capacity; no upload, KV allocation, or PCIe microbenchmark was run.");
        }
        var effectiveConfiguration = config with { Diagnostics = configurationDiagnostics };
        foreach (var d in effectiveConfiguration.Diagnostics) decisions.Add(new("configuration_" + d.Field, false, d.Message));
        string? Metadata(string key) => model.Metadata.TryGetValue(key, out var value) ? Convert.ToString(value) : null;
        bool HasToken(string value) => model.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokens)
            && tokens is object[] array && array.Any(token => string.Equals(token as string, value, StringComparison.Ordinal));
        int vocabularySize = model.Metadata.TryGetValue("tokenizer.ggml.tokens", out var vocabulary)
            && vocabulary is object[] vocabularyTokens ? vocabularyTokens.Length : hp.VocabSize;
        var tensorDtypes = model.Tensors
            .GroupBy(tensor => tensor.DType.ToString(), StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
        long parameterElements = model.Tensors.Sum(tensor => tensor.ElementCount);
        var modelInfo = new StaticPlanModel(Path.GetFullPath(path), Path.GetFileName(path), model.Header.Version,
            arch, model.Tensors.Count, model.Metadata.Count, hp.NumLayers, hp.ContextLength, hp.HeadDim,
            hp.NumKvHeads, hp.IsMoE, hp.NumMtpLayers, Metadata("general.name"), Metadata("general.file_type"),
            Metadata("general.quantization_version"), !string.IsNullOrWhiteSpace(Metadata("tokenizer.chat_template")),
            HasToken("<think>") && HasToken("</think>"), arch == "gemma4", parameterElements,
            vocabularySize, tensorDtypes);
        var executionPlan = BuildExecutionPlan(config, compatibility,
            config.Get<string>("backend").ToLowerInvariant(), selected, selectedAvailable, placement, decisions, effectiveConfiguration);
        return new(1, modelInfo, compatibility, backends, runtimeFacts.Hardware, effectiveConfiguration, placement, decisions, executionPlan);
    }

    private static ExecutionPlan BuildExecutionPlan(
        EffectiveConfigurationSnapshot configuration,
        StaticPlanDecision compatibility,
        string requestedBackend,
        string selectedBackend,
        bool backendAvailable,
        StaticPlanPlacement? placement,
        IReadOnlyList<StaticPlanDecision> decisions,
        EffectiveConfigurationSnapshot effectiveConfiguration)
    {
        var request = new PlanRequest(
            configuration.Get<string>("target"), requestedBackend,
            configuration.Get<int>("gpu_layers"), configuration.Get<int>("context_size"),
            configuration.Get<bool>("turbo_quant"), configuration.Get<string>("kv_type"),
            configuration.Get<string>("spec_type"), configuration.Get<int>("max_batch"),
            configuration.Get<bool>("tool_grammar"));
        var planDecisions = new List<ExecutionPlanDecision>
        {
            new(ExecutionPlanDecisionCodes.ModelCompatibility,
                compatibility.Selected ? PlanDecisionDisposition.Selected : PlanDecisionDisposition.Rejected,
                compatibility.Selected ? PlanDiagnosticSeverity.Info : PlanDiagnosticSeverity.Error,
                compatibility.Why)
        };
        planDecisions.AddRange(decisions.Select(decision => new ExecutionPlanDecision(
            DecisionCode(decision.Area), DecisionDisposition(decision), DecisionSeverity(decision), decision.Why)));
        bool invalidConfiguration = effectiveConfiguration.Diagnostics.Any(x => x.Kind == "invalid");
        return new(1, request, placement?.SelectedBackend ?? selectedBackend,
            placement?.GpuLayers ?? 0, placement?.CpuLayers ?? 0, placement?.RecommendedCtxSize ?? request.ContextSize,
            compatibility.Selected && backendAvailable && !invalidConfiguration, planDecisions, effectiveConfiguration);
    }

    private static string DecisionCode(string area) => area switch
    {
        "backend" => ExecutionPlanDecisionCodes.BackendSelection,
        "kv_turbo_quant" => ExecutionPlanDecisionCodes.KvTurboQuant,
        "snapkv" => ExecutionPlanDecisionCodes.SnapKv,
        "speculation" => ExecutionPlanDecisionCodes.Speculation,
        "batching" => ExecutionPlanDecisionCodes.Batching,
        "tool_grammar" => ExecutionPlanDecisionCodes.ToolGrammar,
        _ when area.StartsWith("configuration_", StringComparison.Ordinal) => ExecutionPlanDecisionCodes.Configuration + "." + area["configuration_".Length..],
        _ => area
    };

    private static PlanDecisionDisposition DecisionDisposition(StaticPlanDecision decision)
    {
        if (decision.Selected) return PlanDecisionDisposition.Selected;
        if (decision.Why.Contains("ignored", StringComparison.OrdinalIgnoreCase)) return PlanDecisionDisposition.Ignored;
        if (decision.Why.Contains("conditional", StringComparison.OrdinalIgnoreCase)) return PlanDecisionDisposition.Conditional;
        return PlanDecisionDisposition.Rejected;
    }

    private static PlanDiagnosticSeverity DecisionSeverity(StaticPlanDecision decision) => decision.Selected
        ? PlanDiagnosticSeverity.Info
        : decision.Area == "backend" || decision.Why.Contains("invalid", StringComparison.OrdinalIgnoreCase)
            ? PlanDiagnosticSeverity.Error
            : PlanDiagnosticSeverity.Warning;

    internal static void WriteHuman(StaticPlanReport report, bool explain)
    {
        Console.WriteLine($"Model: {report.Model}");
        Console.WriteLine($"Compatibility: {(report.Compatibility.Selected ? "selected" : "rejected")} — {report.Compatibility.Why}");
        if (report.Placement is { } placement)
            Console.WriteLine($"Placement: {placement.GpuLayers} GPU layers / {placement.CpuLayers} CPU layers; context {placement.RecommendedCtxSize}.");
        if (explain)
        {
            Console.WriteLine("Decisions:");
            foreach (var decision in report.Decisions)
                Console.WriteLine($"  {decision.Area}: {(decision.Selected ? "selected" : "rejected")} — {decision.Why}");
            Console.WriteLine("Effective configuration:");
            foreach (var (name, value) in report.EffectiveConfiguration.Values.OrderBy(x => x.Key, StringComparer.Ordinal))
                Console.WriteLine($"  {name} = {value.Value.GetRawText()} ({value.Source})");
        }
        else Console.WriteLine("Use --explain or --json for the complete selected/rejected/why report.");
    }
}

/// <summary>Model-centred inspect view. It deliberately excludes TierPlanner placement.</summary>
public sealed record StaticInspectReport(
    int SchemaVersion, StaticPlanModel Model, StaticPlanDecision Compatibility,
    IReadOnlyList<StaticPlanBackend> Backends, StaticPlanHardware Hardware,
    IReadOnlyList<StaticPlanDecision> Features)
{
    internal static StaticInspectReport FromPlan(StaticPlanReport plan) =>
        new(plan.SchemaVersion, plan.Model, plan.Compatibility, plan.Backends, plan.Hardware, plan.Decisions);

    internal static void WriteHuman(StaticInspectReport report)
    {
        Console.WriteLine($"Model: {report.Model.Name ?? report.Model.Filename} ({report.Model.Architecture}, GGUF v{report.Model.GgufVersion})");
        Console.WriteLine($"Compatibility: {(report.Compatibility.Selected ? "supported" : "rejected")} — {report.Compatibility.Why}");
        Console.WriteLine($"Model facts: {report.Model.ParameterElements:N0} parameter elements; vocab {report.Model.VocabularySize:N0}; {report.Model.TensorCount} tensors.");
        Console.WriteLine($"Capabilities: chat template={report.Model.HasChatTemplate}, reasoning tokens={report.Model.HasReasoningTokens}, MTP={report.Model.NumMtpLayers > 0}, vision={report.Model.SupportsVisionInput}");
        Console.WriteLine("Use --json for the complete machine-readable report.");
    }
}

/// <summary>Dedicated inspect command; it shares options/facts with plan but never calculates placement.</summary>
public sealed class InspectCommand : Command<StaticPlanCommand.Settings>
{
    protected override int Execute(StaticPlanCommand.Settings settings, CancellationToken cancellation)
    {
        // A SafeTensors package is a directory, not a GGUF file, so the GGUF planner cannot read it.
        // Report the capability verdict instead — Phase 0's requirement that support be decidable
        // without constructing a forward pass.
        if (settings.ModelPath is { Length: > 0 } path && ModelPackageReporting.LooksLikePackage(path))
            return ModelPackageReporting.PrintReport(path);

        return StaticPlanCommand.Execute(settings, inspectOnly: true);
    }
}

/// <summary>Prints the published model-package capability rows. Takes no model.</summary>
public sealed class CapabilitiesCommand : Command<CapabilitiesCommand.Settings>
{
    public sealed class Settings : CommandSettings;

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        ModelPackageReporting.PrintCapabilityTable();
        return 0;
    }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    PropertyNameCaseInsensitive = true, WriteIndented = false, UseStringEnumConverter = true)]
[JsonSerializable(typeof(StaticPlanReport))]
[JsonSerializable(typeof(StaticInspectReport))]
[JsonSerializable(typeof(EffectiveConfigurationSnapshot))]
[JsonSerializable(typeof(StaticPlanProfile))]
[JsonSerializable(typeof(ExecutionPlan))]
internal partial class StaticPlanJsonContext : JsonSerializerContext
{
    internal static StaticPlanJsonContext Indented => new(new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    });
}
