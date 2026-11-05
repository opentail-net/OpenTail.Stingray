using System.Diagnostics;
using System.Text;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// E1, long-running stability on a real model. One process, one loaded model, many greedy generations: the output must be
/// identical on every repeat, and working memory must not keep growing after warm-up. Opt-in (STINGRAY_SOAK=1) because it
/// runs a real model on CPU for a minute or so; it is not part of the regular suite.
/// </summary>
public sealed class LongRunSoakTests
{
    private const int Repeats = 30;
    private const long MaxGrowthBytes = 200L * 1024 * 1024;

    [Fact]
    public async Task GreedyRepeats_OnOneLoadedModel_AreIdentical_AndMemoryIsFlat()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("STINGRAY_SOAK") != "1", "set STINGRAY_SOAK=1 to run the soak");
        string? path = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipWhen(path is null, "Qwen3-0.6B-Q8_0.gguf not present");

        using var model = Model.Load(new ModelParams(path!) { Backend = "cpu", GpuLayerCount = 0 });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = 1024, ThreadCount = 4 });
        IInferenceEngine engine = ctx.Engine;
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 48 };
        const string prompt = "Write one sentence about the sea.";

        string first = await Collect(engine, prompt, sp);
        Assert.False(string.IsNullOrWhiteSpace(first), "the first generation produced no text");
        long warm = WorkingSet();

        var failures = new List<string>();
        for (int i = 1; i < Repeats; i++)
        {
            string again = await Collect(engine, prompt, sp);
            if (again != first) failures.Add($"repeat {i} differs: '{Trim(again)}' vs '{Trim(first)}'");
        }
        long end = WorkingSet();

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        Assert.True(end - warm < MaxGrowthBytes,
            $"working set grew by {(end - warm) / 1048576} MB over {Repeats - 1} repeats (limit {MaxGrowthBytes / 1048576} MB)");
    }

    private static async Task<string> Collect(IInferenceEngine engine, string prompt, SamplingParams sp)
    {
        var sb = new StringBuilder();
        await foreach (var piece in engine.GenerateAsync(prompt, sp)) sb.Append(piece);
        return sb.ToString();
    }

    private static long WorkingSet()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        return Process.GetCurrentProcess().WorkingSet64;
    }

    private static string Trim(string s) => s.Length > 80 ? s[..80] + "..." : s;

    private static string? FindModel(string file)
    {
        foreach (var dir in new[] { @"F:\_models", @"E:\_models", @"H:\_models" })
        {
            var p = Path.Combine(dir, file);
            if (File.Exists(p)) return p;
        }
        return null;
    }
}
