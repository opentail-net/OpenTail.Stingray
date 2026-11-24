using System.Diagnostics;
using System.Globalization;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Diagnosis for the Queue 3 item "Memory returned to the OS after unload" (docs/00-current-work.md, open item (a)): on a single-slot host
/// that alternates between two real models, which managed objects are still alive after their runtime was disposed and evicted?
/// Measurement, not a gate: it writes one TSV row per (unload, tracked object) and asserts only that each generation produced tokens.
/// Opt in with <c>STINGRAY_RUN_MEMORY_PROBE=1</c>; the TSV goes to <c>STINGRAY_RESIDUE_PROBE_OUT</c> (default: the system temp folder).
/// The number of cycles (one A and one B acquisition each) is 3 by default and set by <c>STINGRAY_RESIDUE_PROBE_CYCLES</c>.
/// <para>Under <see cref="ModelResidencyMode.SingleSlot"/> an idle runtime stays resident until the other model is acquired, and that
/// acquisition evicts it. So the acquisitions run A, B, A, B, A, B; every acquisition after the first evicts the one before it, and a
/// sample is taken after each such eviction. A row with <c>alive = 1</c> and <c>object_model = unload_model</c> is an object of an
/// evicted runtime that is still reachable, i.e. a leak candidate. Rows whose object belongs to the resident model are expected alive.</para>
/// </summary>
public sealed class ManagedResidueProbeTests
{
    private const int DefaultCycles = 3;
    private const int MaxNewTokens = 8;
    private const int ContextSize = 2048;
    private const string Prompt = "The capital of France is";

    private sealed record Tracked(string ObjectModel, int CycleCreated, string Kind, WeakReference Ref);

    private sealed record Row(int Cycle, string UnloadModel, string ObjectKind, int CycleCreated, bool Alive, string ObjectModel,
        int ResidentModels, long ManagedBytes, long PrivateBytes);

    [Fact]
    public async Task ManagedObjectsAfterUnload_SingleSlotTwoModels_Probe()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_MEMORY_PROBE") == "1", "set STINGRAY_RUN_MEMORY_PROBE=1 to run this probe");
        string? pathA = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        string? pathB = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        Assert.SkipUnless(pathA is not null && pathB is not null, "the two probe models were not found under a models folder");

        var ids = new[] { ModelId.Canonicalize(pathA!), ModelId.Canonicalize(pathB!) };
        var labels = new[] { "0.5b", "1.5b" };
        var tracked = new List<Tracked>();
        var rows = new List<Row>();
        var generated = new List<int>();
        ModelRuntimeManager? manager = null;

        try
        {
            manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(OptionsFor(id.Value)), ModelResidencyMode.SingleSlot);

            // A, B, A, B, ...: each cycle is one A and one B acquisition. The cycle count comes from STINGRAY_RESIDUE_PROBE_CYCLES.
            int acquisitions = 2 * ReadCycles();
            for (int k = 0; k < acquisitions; k++)
            {
                int which = k % 2;
                int cycle = k / 2 + 1;
                tracked.AddRange(await UseAsync(manager, ids[which], labels[which], cycle, generated));
                if (k == 0) continue;

                // This acquisition evicted the model acquired just before it (idle, single-slot policy).
                int evicted = 1 - which;
                AddRows(rows, tracked, cycle, labels[evicted], manager.GetStats().ResidentModels);
            }
        }
        finally
        {
            WriteTsv(rows);
            manager?.Dispose();
        }

        Assert.All(generated, tokens => Assert.True(tokens > 0, "a generation produced no tokens"));
    }

    /// <summary>
    /// Acquires <paramref name="id"/>, records weak references to the runtime's objects, generates <see cref="MaxNewTokens"/> tokens
    /// greedily, then releases the handle. Only weak references leave this method: nothing strong is kept by the caller.
    /// </summary>
    private static async Task<List<Tracked>> UseAsync(ModelRuntimeManager manager, ModelId id, string label, int cycle, List<int> generated)
    {
        using var handle = await manager.AcquireAsync(id);
        var tracked = Track(label, cycle, handle.Runtime);
        int tokens = await GenerateAsync(handle.Runtime.Engine);
        generated.Add(tokens);
        return tracked;
    }

    private static async Task<int> GenerateAsync(IInferenceEngine engine)
    {
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        int tokens = 0;
        await foreach (var chunk in engine.GenerateChunksAsync(Prompt, sp))
        {
            if (chunk.Kind is GenerateChunkKind.Usage or GenerateChunkKind.Stop) continue;
            tokens++;
        }
        return tokens;
    }

    /// <summary>Weak references to every object a runtime holds that is a candidate for surviving its own disposal.</summary>
    private static List<Tracked> Track(string label, int cycle, ModelRuntime runtime)
    {
        var list = new List<Tracked>();
        void Add(string kind, object? obj)
        {
            if (obj is not null) list.Add(new Tracked(label, cycle, kind, new WeakReference(obj)));
        }

        var loaded = runtime.Loaded;
        Add("loaded_engine", loaded);
        Add("engine", loaded.Engine);
        Add("model_runtime", runtime);
        Add("runtime_instance", loaded.Instance);
        // Public on LoadedEngine (BuildLoadedEngine passes instance.Tokenizer), so no reflection is needed.
        Add("tokenizer", loaded.Tokenizer);
        Add("grammar_vocab", loaded.Grammar);
        Add("chat_template", loaded.ChatTemplate);
        if (loaded.Instance is { } instance)
        {
            Add("gguf_model", instance.Model);
            Add("forward_pass", instance.ForwardPass);
            if (!ReferenceEquals(instance.Engine, loaded.Engine)) Add("instance_engine", instance.Engine);
        }
        return list;
    }

    /// <summary>Forces collection, reads the managed and process figures once, and emits one row per tracked object created so far.</summary>
    private static void AddRows(List<Row> rows, IReadOnlyList<Tracked> tracked, int cycle, string unloadModel, int residentModels)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long managed = GC.GetTotalMemory(true);
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        long privateBytes = process.PrivateMemorySize64;
        foreach (var t in tracked)
        {
            rows.Add(new Row(cycle, unloadModel, t.Kind, t.CycleCreated, t.Ref.IsAlive, t.ObjectModel, residentModels, managed, privateBytes));
        }
    }

    /// <summary>
    /// Cycles from <c>STINGRAY_RESIDUE_PROBE_CYCLES</c>: unset or empty gives <see cref="DefaultCycles"/>; a value that is not a
    /// positive integer fails the run rather than silently falling back, so a typo cannot pass for the default.
    /// </summary>
    private static int ReadCycles()
    {
        string? raw = Environment.GetEnvironmentVariable("STINGRAY_RESIDUE_PROBE_CYCLES");
        if (string.IsNullOrWhiteSpace(raw)) return DefaultCycles;
        if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int cycles) && cycles > 0) return cycles;
        throw new InvalidOperationException($"STINGRAY_RESIDUE_PROBE_CYCLES must be a positive integer, got '{raw}'.");
    }

    private static OpenTailStingrayServerOptions OptionsFor(string modelPath) => new()
    {
        Backend = ServerBackend.Cpu,
        NGpuLayers = 0,
        ContextSize = ContextSize,
        MaxBatchSize = 1,
        ModelPath = modelPath,
    };

    private static void WriteTsv(List<Row> rows)
    {
        string path = Environment.GetEnvironmentVariable("STINGRAY_RESIDUE_PROBE_OUT")
            ?? Path.Combine(Path.GetTempPath(), "managed-residue-probe.tsv");
        string? dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var lines = new List<string> { "cycle\tunload_model\tobject_kind\tcycle_created\talive\tmanaged_bytes\tprivate_bytes\tobject_model\tresident_models" };
        foreach (var r in rows)
        {
            lines.Add(string.Join('\t',
                r.Cycle.ToString(CultureInfo.InvariantCulture),
                r.UnloadModel,
                r.ObjectKind,
                r.CycleCreated.ToString(CultureInfo.InvariantCulture),
                r.Alive ? "1" : "0",
                r.ManagedBytes.ToString(CultureInfo.InvariantCulture),
                r.PrivateBytes.ToString(CultureInfo.InvariantCulture),
                r.ObjectModel,
                r.ResidentModels.ToString(CultureInfo.InvariantCulture)));
        }
        File.WriteAllLines(path, lines);
    }
}
