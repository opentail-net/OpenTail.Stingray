using System.Diagnostics;
using System.Globalization;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Queue 3 probe, "Memory returned to the OS after unload": on a single-slot host that switches between two real models, does
/// private memory come back after each unload? Measurement, not a gate: it writes one TSV row per sample and asserts only that
/// each generation produced tokens and that each unload left no resident runtime. Opt in with <c>STINGRAY_RUN_MEMORY_PROBE=1</c>;
/// the TSV goes to <c>STINGRAY_MEMORY_PROBE_OUT</c> (default: the system temp folder).
/// <para>Under <see cref="ModelResidencyMode.SingleSlot"/> a released handle does not evict anything: the idle runtime stays resident
/// until the other model is acquired (<see cref="ModelRuntimeManager.AcquireAsync"/> evicts it synchronously, before its own load
/// starts). So each unload is sampled inside the loader of the model that replaces it: after the old runtime has been disposed and
/// before the new model's weights are read. This is the only point where "old model gone, new model not yet loaded" is observable.</para>
/// </summary>
public sealed class MemoryReturnAfterUnloadProbeTests
{
    private const int Cycles = 4;
    private const int MaxNewTokens = 8;
    private const int ContextSize = 2048;
    private const string Prompt = "The capital of France is";

    private sealed record Row(int Index, int Cycle, string Event, long PrivateBytes, long WorkingSetBytes, long ManagedBytes, int ResidentAfter);

    [Fact]
    public async Task UnloadReturnsMemory_SingleSlotTwoModels_Probe()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_MEMORY_PROBE") == "1", "set STINGRAY_RUN_MEMORY_PROBE=1 to run this probe");
        string? pathA = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        string? pathB = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        Assert.SkipUnless(pathA is not null && pathB is not null, "the two probe models were not found under a models folder");

        var idA = ModelId.Canonicalize(pathA!);
        var idB = ModelId.Canonicalize(pathB!);
        var rows = new List<Row>();
        var generated = new List<int>();

        // Set by the test before an acquisition that is expected to evict; read by the loader, which runs after that eviction.
        string? pendingEvent = null;
        int pendingCycle = 0;
        ModelRuntimeManager? manager = null;

        LoadedEngine Load(ModelId id)
        {
            if (pendingEvent is { } evt)
            {
                pendingEvent = null;
                rows.Add(Sample(manager!, rows.Count, pendingCycle, evt));
            }
            return InferenceEngineLoader.Load(OptionsFor(id.Value));
        }

        try
        {
            manager = new ModelRuntimeManager(Load, ModelResidencyMode.SingleSlot);
            rows.Add(Sample(manager, rows.Count, 0, "baseline"));

            for (int cycle = 1; cycle <= Cycles; cycle++)
            {
                if (cycle > 1)
                {
                    pendingEvent = "unload_B";
                    pendingCycle = cycle - 1;
                }
                await UseAsync(manager, idA, generated);

                pendingEvent = "unload_A";
                pendingCycle = cycle;
                await UseAsync(manager, idB, generated);
            }

            // Acquiring A once more evicts the last B, which is the unload_B row of cycle Cycles.
            pendingEvent = "unload_B";
            pendingCycle = Cycles;
            using (await manager.AcquireAsync(idA))
            {
            }
        }
        finally
        {
            WriteTsv(rows);
            manager?.Dispose();
        }

        Assert.All(generated, tokens => Assert.True(tokens > 0, "a generation produced no tokens"));
        foreach (var row in rows.Where(r => r.Event.StartsWith("unload_", StringComparison.Ordinal)))
        {
            Assert.True(row.ResidentAfter == 0, $"{row.Event} cycle {row.Cycle}: {row.ResidentAfter} runtime(s) still resident after the eviction");
        }
        Assert.Equal(Cycles * 2 + 1, rows.Count);
    }

    /// <summary>Acquires <paramref name="id"/>, generates <see cref="MaxNewTokens"/> tokens greedily, then releases the handle.</summary>
    private static async Task UseAsync(ModelRuntimeManager manager, ModelId id, List<int> generated)
    {
        using var handle = await manager.AcquireAsync(id);
        var engine = handle.Runtime.Engine;
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        int tokens = 0;
        await foreach (var chunk in engine.GenerateChunksAsync(Prompt, sp))
        {
            if (chunk.Kind is GenerateChunkKind.Usage or GenerateChunkKind.Stop) continue;
            tokens++;
        }
        generated.Add(tokens);
    }

    /// <summary>Forces collection, then reads the process and managed figures. Also reports how many runtimes are resident.</summary>
    private static Row Sample(ModelRuntimeManager manager, int index, int cycle, string evt)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long managed = GC.GetTotalMemory(true);
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return new Row(index, cycle, evt, process.PrivateMemorySize64, process.WorkingSet64, managed, manager.GetStats().ResidentModels);
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
        string path = Environment.GetEnvironmentVariable("STINGRAY_MEMORY_PROBE_OUT")
            ?? Path.Combine(Path.GetTempPath(), "memory-unload-probe.tsv");
        string? dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var lines = new List<string> { "row\tcycle\tevent\tprivate_bytes\tworking_set_bytes\tmanaged_bytes\tresident_after" };
        foreach (var r in rows)
        {
            lines.Add(string.Join('\t',
                r.Index.ToString(CultureInfo.InvariantCulture),
                r.Cycle.ToString(CultureInfo.InvariantCulture),
                r.Event,
                r.PrivateBytes.ToString(CultureInfo.InvariantCulture),
                r.WorkingSetBytes.ToString(CultureInfo.InvariantCulture),
                r.ManagedBytes.ToString(CultureInfo.InvariantCulture),
                r.ResidentAfter.ToString(CultureInfo.InvariantCulture)));
        }
        File.WriteAllLines(path, lines);
    }
}
