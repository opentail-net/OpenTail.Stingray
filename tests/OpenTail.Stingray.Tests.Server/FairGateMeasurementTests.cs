using System.Diagnostics;
using System.Text;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Plan 2 P4: what does <c>FairModelScheduling</c> (the opt-in <see cref="FairExecutionGate"/>) do to latency and throughput when two resident
/// models share one device? Measurement, not a gate: it prints one line per run and asserts only that the work completed and that the generated
/// text does not depend on the setting. Opt in with <c>STINGRAY_FAIR_GATE_MEASURE=1</c>.
/// <para>Scenario: a decode-heavy model ("B", small, 200 tokens) generates while a prefill-heavy model ("A", larger) processes several long
/// prompts. B's inter-token gaps show whether A's prefill starves it; the wall time for both to finish is the throughput proxy. Runs
/// alternate gate off and on so drift in the machine's load hits both alike.</para>
/// Settings: <c>STINGRAY_FAIR_GATE_MODEL_A</c> / <c>_B</c> (file names under the model folders, defaults qwen2.5-1.5b / qwen2.5-0.5b),
/// <c>STINGRAY_FAIR_GATE_PROMPT_FILE</c> and <c>_PROMPT_CHARS</c> (default 5000) for A's prompt, <c>STINGRAY_FAIR_GATE_RUNS</c> (pairs, default 3),
/// <c>STINGRAY_FAIR_GATE_PREFILL_CHUNK</c> (prompt tokens prefilled per batcher iteration; the server default is automatic, 256). One iteration is the
/// unit the gate hands out, so this bounds how long the other model can wait.
/// </summary>
public sealed class FairGateMeasurementTests
{
    private const int DecodeTokens = 200;
    private const int LongRequests = 3;

    private sealed record RunResult(
        int Stalls, double StallMs,
        bool Fair, double BTotalMs, double BP50GapMs, double BP95GapMs, double BMaxGapMs, double ATotalMs, double AFirstMs, double WallMs,
        int BTokens, int ATokens, int APromptTokens, string BText, string AText);

    [Fact]
    public async Task TwoModels_LongPrefillBesideADecode_GateOffVersusOn()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_MEASURE") == "1", "set STINGRAY_FAIR_GATE_MEASURE=1 to run this measurement");
        string? a = FindModel(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_MODEL_A") ?? "qwen2.5-1.5b-instruct-q4_k_m.gguf");
        string? b = FindModel(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_MODEL_B") ?? "qwen2.5-0.5b-instruct-q4_k_m.gguf");
        string? promptFile = Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_PROMPT_FILE");
        Assert.SkipUnless(a is not null && b is not null, "the two models were not found");
        Assert.SkipUnless(promptFile is not null && File.Exists(promptFile), "set STINGRAY_FAIR_GATE_PROMPT_FILE to a text file");
        int chars = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_PROMPT_CHARS"), out int c) ? c : 5000;
        int pairs = int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_RUNS"), out int r) ? r : 3;
        string text = File.ReadAllText(promptFile!);
        text = text[..Math.Min(chars, text.Length)];

        var results = new List<RunResult>();
        for (int i = 0; i < pairs; i++)
        {
            foreach (bool fair in new[] { false, true })
            {
                var result = await OneRun(a!, b!, text, fair);
                results.Add(result);
                Console.WriteLine(
                    $"[fairgate] run={i + 1} gate={(fair ? "on " : "off")} wall={result.WallMs:F0}ms | B: total={result.BTotalMs:F0}ms tokens={result.BTokens} " +
                    $"gap p50={result.BP50GapMs:F1}ms p95={result.BP95GapMs:F1}ms max={result.BMaxGapMs:F1}ms stalls>500ms={result.Stalls} (sum {result.StallMs:F0}ms) | A: total={result.ATotalMs:F0}ms first={result.AFirstMs:F0}ms " +
                    $"promptTokens={result.APromptTokens} tokens={result.ATokens}");
            }
        }

        foreach (bool fair in new[] { false, true })
        {
            var group = results.Where(x => x.Fair == fair).ToArray();
            Console.WriteLine(
                $"[fairgate] MEAN gate={(fair ? "on " : "off")} wall={group.Average(x => x.WallMs):F0}ms | B: total={group.Average(x => x.BTotalMs):F0}ms " +
                $"p95gap={group.Average(x => x.BP95GapMs):F1}ms maxgap={group.Average(x => x.BMaxGapMs):F1}ms | A: total={group.Average(x => x.ATotalMs):F0}ms");
        }

        // Taking turns must not change what a model generates (greedy), and every run finished its work.
        Assert.All(results, x => Assert.Equal(DecodeTokens, x.BTokens));
        Assert.Single(results.Select(x => x.BText).Distinct());
        Assert.Single(results.Select(x => x.AText).Distinct());
    }

    private static async Task<RunResult> OneRun(string modelA, string modelB, string longPrompt, bool fair)
    {
        var services = new Microsoft.Extensions.DependencyInjection.ServiceCollection();
        services.AddOpenTailStingray(opts =>
        {
            opts.Backend = ServerBackend.Cpu;
            opts.NGpuLayers = 0;
            opts.ContextSize = 4096;
            opts.MaxBatchSize = 2;
            opts.FairModelScheduling = fair;
            if (int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_FAIR_GATE_PREFILL_CHUNK"), out int chunk)) opts.PrefillChunkTokens = chunk;
            opts.Models = [new NamedModelOptions { Alias = "a", ModelPath = modelA }, new NamedModelOptions { Alias = "b", ModelPath = modelB }];
        });
        await using var provider = services.BuildServiceProvider();
        var manager = Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<IModelRuntimeManager>(provider);
        using var handleA = await manager.AcquireAsync(ModelId.Canonicalize(modelA));
        using var handleB = await manager.AcquireAsync(ModelId.Canonicalize(modelB));
        var engineA = EngineOf(handleA);
        var engineB = EngineOf(handleB);

        var warm = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        await Drain(engineA, "Hello", warm);   // load cost, first-use setup and thread spin-up stay out of the measurement
        await Drain(engineB, "Hello", warm);

        var spB = new SamplingParams { Temperature = 0f, MaxNewTokens = DecodeTokens };
        var spA = new SamplingParams { Temperature = 0f, MaxNewTokens = 8 };
        var gaps = new List<double>();
        var bText = new StringBuilder();
        var aText = new StringBuilder();
        int bTokens = 0, aTokens = 0, aPrompt = 0;
        double aFirst = 0;
        var wall = Stopwatch.StartNew();

        var runB = Task.Run(async () =>
        {
            var sw = Stopwatch.StartNew();
            double last = 0;
            await foreach (var chunk in engineB.GenerateChunksAsync("Write a long story about a lighthouse keeper.", spB))
            {
                if (chunk.Kind is GenerateChunkKind.Usage or GenerateChunkKind.Stop) continue;
                double now = sw.Elapsed.TotalMilliseconds;
                if (bTokens > 0) gaps.Add(now - last);
                last = now;
                bTokens++;
                bText.Append(chunk.Text);
            }
            return sw.Elapsed.TotalMilliseconds;
        });
        var runA = Task.Run(async () =>
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < LongRequests; i++)
            {
                await foreach (var chunk in engineA.GenerateChunksAsync(longPrompt + $"\n\nSummarise part {i + 1} in one sentence:", spA))
                {
                    if (chunk.Kind == GenerateChunkKind.Usage) { aPrompt = chunk.PromptTokens; continue; }
                    if (chunk.Kind == GenerateChunkKind.Stop) continue;
                    if (i == 0 && aTokens == 0) aFirst = sw.Elapsed.TotalMilliseconds;
                    aTokens++;
                    aText.Append(chunk.Text);
                }
            }
            return sw.Elapsed.TotalMilliseconds;
        });
        var done = await Task.WhenAll(runB, runA).WaitAsync(TimeSpan.FromMinutes(15));
        wall.Stop();

        var sorted = gaps.OrderBy(x => x).ToArray();
        int stalls = gaps.Count(g => g > 500);
        double stallMs = gaps.Where(g => g > 500).Sum();
        double Percentile(double p) => sorted.Length == 0 ? 0 : sorted[Math.Min(sorted.Length - 1, (int)(p * sorted.Length))];
        return new RunResult(stalls, stallMs, fair, done[0], Percentile(0.50), Percentile(0.95), sorted.Length == 0 ? 0 : sorted[^1], done[1], aFirst, wall.Elapsed.TotalMilliseconds,
            bTokens, aTokens, aPrompt, bText.ToString(), aText.ToString());
    }

    private static ContinuousBatchingEngine EngineOf(ModelRuntimeHandle handle)
        => Assert.IsType<ContinuousBatchingEngine>(Assert.IsType<OpenTail.Stingray.Engine.Runtime.RuntimeInstance>(handle.Runtime.Loaded.Instance).Engine);

    private static async Task Drain(ContinuousBatchingEngine engine, string prompt, SamplingParams sp)
    {
        await foreach (var _ in engine.GenerateChunksAsync(prompt, sp)) { }
    }

    private static string? FindModel(string fileName)
    {
        foreach (string root in new[] { @"F:\_models", @"E:\_models", @"H:\_models" })
        {
            string candidate = Path.Combine(root, fileName);
            if (File.Exists(candidate)) return candidate;
        }
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", fileName);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
