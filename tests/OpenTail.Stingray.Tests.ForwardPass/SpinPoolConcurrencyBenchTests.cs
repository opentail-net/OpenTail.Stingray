using System.Diagnostics;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Plan 6 D1: is the opt-in spin-then-park CPU worker pool (<c>STINGRAY_CPU_POOL=spin</c>) safe to make the default for dense decode when more than one
/// engine decodes in the same process? A single engine decodes on one thread at a time (the batcher thread), so the pool serves one caller. Two engines in
/// one process (multi-model serving) are two callers of the one static pool: the second finds it busy and falls back to <c>Parallel.For</c>
/// (<c>SimdKernels.SpinPoolBusyFallbacks</c>). This measures aggregate decode throughput with 1, 2 and 3 engines decoding at once, pool off against on,
/// toggled in-process (<c>SimdKernels.SpinPoolEnabled</c>) so every pair is interleaved in the same run, and requires the generated tokens to be identical
/// in every mode. It prints a table; the only assertion is equal output, because the verdict is a number to read, not a threshold to pass.
/// Real weights, timed; needs a quiet machine (the interleaving removes drift, not a busy neighbour), so the background load is the reader's to check.
/// </summary>
public sealed class SpinPoolConcurrencyBenchTests : HeavyTestBase
{
    private const int NewTokens = 64;
    private const int Rounds = 5;
    private static readonly int[] EngineCounts = [1, 2, 3];

    private static string? FindModel(string fileName)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null; dir = dir.Parent)
            foreach (string c in new[] { Path.Combine(dir.FullName, "models", fileName), Path.Combine(dir.FullName, "models", "_models", fileName) })
                if (File.Exists(c)) return c;
        string home = Path.Combine(OpenTail.Stingray.Core.Catalog.ModelHome.DefaultRoot(), fileName);
        if (File.Exists(home)) return home;
        foreach (var drive in DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.Fixed && d.IsReady))
        {
            string c = Path.Combine(drive.RootDirectory.FullName, "_models", fileName);
            if (File.Exists(c)) return c;
        }
        return null;
    }

    [Theory]
    [InlineData("qwen2.5-0.5b-instruct-q4_k_m.gguf")]
    [InlineData("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")]
    public void DenseDecode_AggregateThroughput_SpinPoolOffVsOn_AtOneToThreeEngines(string modelFile)
    {
        string? path = FindModel(modelFile);
        Assert.SkipUnless(path is not null, $"{modelFile} is required for this benchmark");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        int[] prompt = tokenizer.Encode("The capital of France is").ToArray();
        using var backend = new CpuBackend();

        var engines = new List<OpenTail.Stingray.Engine.ForwardPass>();
        try
        {
            for (int i = 0; i < EngineCounts.Max(); i++) engines.Add(new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp));

            int[] Decode(OpenTail.Stingray.Engine.ForwardPass fwd)
            {
                fwd.ResetCache();
                var logits = fwd.Prefill(prompt);
                int token = 0; float best = float.NegativeInfinity;
                for (int i = 0; i < logits.Length; i++) if (logits[i] > best) { best = logits[i]; token = i; }
                var generated = new int[NewTokens];
                int pos = prompt.Length;
                for (int n = 0; n < NewTokens; n++)
                {
                    generated[n] = token;
                    (token, _) = ((IForwardPass)fwd).ForwardArgmax(token, pos++);
                }
                return generated;
            }

            // One round: n engines each decode NewTokens in their own thread, released together. Returns the wall time and every engine's tokens.
            (TimeSpan Elapsed, int[][] Tokens) Round(int n, bool pool)
            {
                SimdKernels.SpinPoolEnabled = pool;
                var results = new int[n][];
                using var start = new Barrier(n + 1);
                var threads = Enumerable.Range(0, n).Select(i => new Thread(() => { start.SignalAndWait(); results[i] = Decode(engines[i]); })).ToArray();
                foreach (var t in threads) t.Start();
                start.SignalAndWait();
                var clock = Stopwatch.StartNew();
                foreach (var t in threads) t.Join();
                return (clock.Elapsed, results);
            }

            bool original = SimdKernels.SpinPoolEnabled;
            try
            {
                // Warm both modes (JIT, pool threads, caches) before anything is timed.
                foreach (bool pool in new[] { false, true }) Round(EngineCounts.Max(), pool);

                int[] reference = Round(1, pool: false).Tokens[0];
                var samples = new Dictionary<(int N, bool Pool), List<double>>();
                long dispatchedBefore = SimdKernels.SpinPoolDispatches, busyBefore = SimdKernels.SpinPoolBusyFallbacks;
                for (int round = 0; round < Rounds; round++)
                    foreach (int n in EngineCounts)
                        foreach (bool pool in round % 2 == 0 ? new[] { false, true } : new[] { true, false })   // alternate which mode goes first
                        {
                            var (elapsed, tokens) = Round(n, pool);
                            foreach (var t in tokens) Assert.Equal(reference, t);   // the same greedy text, with or without the pool, alone or concurrent
                            if (!samples.TryGetValue((n, pool), out var list)) samples[(n, pool)] = list = [];
                            list.Add(n * NewTokens / elapsed.TotalSeconds);   // aggregate tokens per second
                        }

                double Median(List<double> v) { var s = v.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
                Console.Error.WriteLine($"[SpinPoolConcurrencyBench] {modelFile}, {NewTokens} greedy tokens per engine, {Rounds} interleaved rounds, {Environment.ProcessorCount} logical CPUs");
                Console.Error.WriteLine("  engines | pool off (t/s, aggregate) | pool on | on / off | min..max on, off");
                foreach (int n in EngineCounts)
                {
                    var off = samples[(n, false)]; var on = samples[(n, true)];
                    Console.Error.WriteLine($"  {n,7} | {Median(off),25:F1} | {Median(on),7:F1} | {Median(on) / Median(off),7:F2}x | {on.Min():F1}..{on.Max():F1}, {off.Min():F1}..{off.Max():F1}");
                }
                Console.Error.WriteLine($"  pool dispatches {SimdKernels.SpinPoolDispatches - dispatchedBefore}, busy fallbacks {SimdKernels.SpinPoolBusyFallbacks - busyBefore} during the timed rounds");
                Assert.True(SimdKernels.SpinPoolDispatches - dispatchedBefore > 0, "the pool never ran: the 'on' rows measured nothing");   // not vacuous
            }
            finally { SimdKernels.SpinPoolEnabled = original; }
        }
        finally { foreach (var e in engines) e.Dispose(); }
    }
}
