using System.Diagnostics;
using System.Globalization;
using System.Text;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Measurement, not a gate: compares the planner's CPU RAM estimate (<c>ExecutionPlan.Memory.EstimatedRamMb</c>, which is
/// <c>PlacementPlan.CpuWeightBytes</c> on the SafeTensors facts path) with the process's real private memory across a CPU load and four
/// greedy tokens, for one tied-embedding SafeTensors checkpoint (Qwen3-0.6B, <c>tie_word_embeddings</c> true).
/// Opt in with <c>STINGRAY_RUN_SAFETENSORS_PROBE=1</c>. The checkpoint folder is <c>hf/Qwen__Qwen3-0.6B</c> (or <c>Qwen__Qwen3-0.6B</c>) under a
/// <see cref="ModelLocator.Roots"/> folder, or the folder named by <c>STINGRAY_SAFETENSORS_PROBE_MODEL</c> when that is set. The one-row TSV summary goes to
/// <c>STINGRAY_SAFETENSORS_PROBE_OUT</c> (default: the system temp folder). Nothing is written to the console.
/// The probe asserts only that it reached a terminal stage; it makes no assertion on the figures.
/// </summary>
public sealed class SafeTensorsRamProbeTests
{
    private const string CheckpointFolder = "Qwen__Qwen3-0.6B";
    private const string Prompt = "The capital of France is";
    private const int MaxNewTokens = 4;
    private const int ContextSize = 512;
    private const double MiB = 1024.0 * 1024.0;
    private static readonly string[] TerminalStages = ["generated", "plan_failed", "load_failed", "generate_failed"];

    private readonly record struct MemoryReading(double PrivateMb, double WorkingSetMb);

    private sealed class ProbeResult
    {
        public string Stage { get; set; } = "not_started";
        public double? PlannedRamMb { get; set; }
        public long? PlannedCpuWeightBytes { get; set; }
        public double? PlannedRamMbLoadedPlan { get; set; }
        public MemoryReading? Before { get; set; }
        public MemoryReading? AfterLoad { get; set; }
        public MemoryReading? AfterGen { get; set; }
        public double? LoadSeconds { get; set; }
        public int GeneratedTokens { get; set; }
        public string GeneratedText { get; set; } = "";
        public string? Error { get; set; }
        public LoadedEngine? Loaded { get; set; }
    }

    [Fact]
    public async Task Qwen3_0_6B_TiedSafeTensors_PlannedRamVsMeasuredMemory_Probe()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_SAFETENSORS_PROBE") == "1",
            "set STINGRAY_RUN_SAFETENSORS_PROBE=1 to run this probe");
        string? modelDir = FindCheckpointDirectory();
        Assert.SkipUnless(modelDir is not null,
            "the Qwen3-0.6B SafeTensors folder (hf/Qwen__Qwen3-0.6B containing model.safetensors) was not found under a models folder");

        var opts = new OpenTailStingrayServerOptions
        {
            ModelPath = modelDir,
            Backend = ServerBackend.Cpu,
            NGpuLayers = 0,
            ContextSize = ContextSize,
            MaxBatchSize = 1,
        };

        var result = new ProbeResult();
        try
        {
            await RunProbeAsync(opts, result);
        }
        finally
        {
            WriteTsv(result);
            Dispose(result.Loaded);
        }

        Assert.Contains(result.Stage, TerminalStages);
    }

    private static async Task RunProbeAsync(OpenTailStingrayServerOptions opts, ProbeResult result)
    {
        // Planned figure: the same plan InferenceEngineLoader.Load builds from these options. BuildPlan is internal and this assembly is in
        // the Server assembly's InternalsVisibleTo, so the test reads the plan the load would run, not a re-derivation of it.
        ExecutionPlan plan;
        try
        {
            plan = InferenceEngineLoader.BuildPlan(opts);
        }
        catch (Exception ex)
        {
            result.Stage = "plan_failed";
            result.Error = Describe(ex);
            return;
        }
        result.PlannedRamMb = plan.Memory?.EstimatedRamMb;
        result.PlannedCpuWeightBytes = plan.Placement?.CpuWeightBytes;

        CollectGarbage();
        result.Before = ReadMemory();

        var clock = Stopwatch.StartNew();
        try
        {
            result.Loaded = InferenceEngineLoader.Load(opts);
        }
        catch (Exception ex)
        {
            result.LoadSeconds = clock.Elapsed.TotalSeconds;
            result.Stage = "load_failed";
            result.Error = Describe(ex);
            return;
        }
        result.LoadSeconds = clock.Elapsed.TotalSeconds;
        LoadedEngine loaded = result.Loaded!;
        result.PlannedRamMbLoadedPlan = loaded.Instance?.Plan.Memory?.EstimatedRamMb;

        CollectGarbage();
        result.AfterLoad = ReadMemory();
        result.Stage = "loaded";

        // Four greedy tokens. The memory reading after generation is taken without a forced collection.
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        var text = new StringBuilder();
        int tokens = 0;
        string? genError = null;
        try
        {
            await foreach (var chunk in loaded.Engine.GenerateChunksAsync(Prompt, sp))
            {
                if (chunk.Kind is GenerateChunkKind.Usage or GenerateChunkKind.Stop) continue;
                tokens++;
                text.Append(chunk.Text);
            }
        }
        catch (Exception ex)
        {
            genError = Describe(ex);
        }
        result.GeneratedTokens = tokens;
        result.GeneratedText = text.ToString();
        result.AfterGen = ReadMemory();
        if (genError is null)
        {
            result.Stage = "generated";
        }
        else
        {
            result.Stage = "generate_failed";
            result.Error = genError;
        }
    }

    /// <summary>Checkpoint folder: <c>STINGRAY_SAFETENSORS_PROBE_MODEL</c> when set, else the first match under a <see cref="ModelLocator.Roots"/> folder.</summary>
    private static string? FindCheckpointDirectory()
    {
        string? explicitDir = Environment.GetEnvironmentVariable("STINGRAY_SAFETENSORS_PROBE_MODEL");
        if (!string.IsNullOrWhiteSpace(explicitDir))
            return HasWeights(explicitDir) ? explicitDir : null;

        foreach (string root in ModelLocator.Roots())
        {
            foreach (string candidate in new[] { Path.Combine(root, "hf", CheckpointFolder), Path.Combine(root, CheckpointFolder) })
            {
                if (HasWeights(candidate)) return candidate;
            }
        }
        return null;
    }

    private static bool HasWeights(string dir) => File.Exists(Path.Combine(dir, "model.safetensors"));

    private static void CollectGarbage()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static MemoryReading ReadMemory()
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return new MemoryReading(process.PrivateMemorySize64 / MiB, process.WorkingSet64 / MiB);
    }

    /// <summary>The engine and forward pass first (RuntimeInstance does not dispose the model), then the model, which the loader leaves to the caller.</summary>
    private static void Dispose(LoadedEngine? loaded)
    {
        if (loaded is null) return;
        if (loaded.Instance is { } instance)
        {
            try
            {
                instance.Dispose();
            }
            finally
            {
                instance.Model.Dispose();
            }
        }
        else
        {
            (loaded.Engine as IDisposable)?.Dispose();
        }
    }

    private static string Describe(Exception ex) => $"{ex.GetType().Name}: {ex.Message}";

    private static string Num(double? value, string format = "F1") =>
        value is { } v ? v.ToString(format, CultureInfo.InvariantCulture) : "NA";

    /// <summary>Escapes backslash, tab, CR and LF so a free-text field stays on one TSV cell.</summary>
    private static string Escape(string s) =>
        s.Replace("\\", "\\\\", StringComparison.Ordinal)
         .Replace("\t", "\\t", StringComparison.Ordinal)
         .Replace("\r", "\\r", StringComparison.Ordinal)
         .Replace("\n", "\\n", StringComparison.Ordinal);

    private static void WriteTsv(ProbeResult r)
    {
        string path = Environment.GetEnvironmentVariable("STINGRAY_SAFETENSORS_PROBE_OUT")
            ?? Path.Combine(Path.GetTempPath(), "safetensors-ram-probe.tsv");
        string? dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        string[] header =
        [
            "stage", "planned_ram_mb", "planned_cpu_weight_bytes", "planned_ram_mb_loaded_plan",
            "private_before_mb", "private_after_load_mb", "private_after_gen_mb",
            "working_set_before_mb", "working_set_after_load_mb", "working_set_after_gen_mb",
            "load_seconds", "generated_tokens", "generated_text", "exception",
        ];
        string[] values =
        [
            r.Stage,
            Num(r.PlannedRamMb),
            r.PlannedCpuWeightBytes is { } cpuBytes ? cpuBytes.ToString(CultureInfo.InvariantCulture) : "NA",
            Num(r.PlannedRamMbLoadedPlan),
            Num(r.Before?.PrivateMb),
            Num(r.AfterLoad?.PrivateMb),
            Num(r.AfterGen?.PrivateMb),
            Num(r.Before?.WorkingSetMb),
            Num(r.AfterLoad?.WorkingSetMb),
            Num(r.AfterGen?.WorkingSetMb),
            Num(r.LoadSeconds, "F3"),
            r.GeneratedTokens.ToString(CultureInfo.InvariantCulture),
            Escape(r.GeneratedText),
            Escape(r.Error ?? ""),
        ];
        File.WriteAllLines(path, [string.Join('\t', header), string.Join('\t', values)]);
    }
}
