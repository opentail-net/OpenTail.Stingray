using System.Diagnostics;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Throwaway before/after probe for docs/done/vision-raw-pointer-reduction-probe-2026-08-20.md --
/// times repeated real-weight InternVlVisionEncoder.Forward() calls to check whether converting
/// norm/bias fields from raw float* to managed float[] (pinned per-call via `fixed`) costs anything
/// measurable. Skips if the local model fixtures aren't present (same convention as the other
/// real-model tests in this project).
/// </summary>
public sealed class InternVlRawPointerProbeBenchmark
{
    private readonly ITestOutputHelper _output;
    public InternVlRawPointerProbeBenchmark(ITestOutputHelper output) => _output = output;

    private static readonly string BaseModelPath = FindRepoFile("models/InternVL3-2B.Q4_K_M.gguf");
    private static readonly string MmprojPath = FindRepoFile("models/mmproj-internvl3-2b-q8_0.gguf");
    private static readonly string ImagePath = FindRepoFile("tests/OpenTail.Stingray.Tests.Diffusion/bin/Release/net10.0/test_sd15_output.png");

    private static string FindRepoFile(string relative)
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir != null; i++)
        {
            var candidate = Path.Combine(dir, relative);
            if (File.Exists(candidate)) return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        return Path.Combine(AppContext.BaseDirectory, relative);
    }

    [Fact(Timeout = 300_000)]
    public void Probe_Forward_RepeatedTiming()
    {
        if (!File.Exists(MmprojPath) || !File.Exists(BaseModelPath) || !File.Exists(ImagePath))
        {
            _output.WriteLine($"Skipping: local fixtures not present ({MmprojPath}, {BaseModelPath}, {ImagePath})");
            return;
        }

        var model = InternVlVisionModel.Open(MmprojPath);
        var encoder = new InternVlVisionEncoder(model);

        var rgb = ImageIO.LoadRgb(ImagePath, out int width, out int height);
        var pre = InternVlImagePreprocessor.Preprocess(rgb, width, height, model.ImageSize, model.PatchSize);

        // Warmup (JIT, first-touch paging).
        for (int i = 0; i < 2; i++)
        {
            encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out _);
        }

        const int iterations = 8;
        var timesMs = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            encoder.Forward(pre.Chw, pre.TargetWidth, pre.TargetHeight, pre.PatchesX, pre.PatchesY, out int tokenCount);
            sw.Stop();
            timesMs[i] = sw.Elapsed.TotalMilliseconds;
            _output.WriteLine($"iter {i}: {timesMs[i]:F2}ms ({tokenCount} tokens)");
        }

        Array.Sort(timesMs);
        double median = timesMs[iterations / 2];
        double best = timesMs[0];
        var summary = $"median={median:F2}ms best={best:F2}ms raw=[{string.Join(",", Array.ConvertAll(timesMs, t => t.ToString("F2")))}]";
        _output.WriteLine(summary);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "internvl_probe_result.txt"), summary);
    }
}
