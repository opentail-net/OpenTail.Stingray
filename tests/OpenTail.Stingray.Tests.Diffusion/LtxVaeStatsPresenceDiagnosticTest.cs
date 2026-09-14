using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic test (docs/077, 2026-09-14): confirms the real
/// `vae.per_channel_statistics.std-of-means`/`mean-of-means` tensors `LtxVideoPipeline.GenerateVideo`
/// relies on to un-normalize DiT-space latents before VAE decode are actually present in the real
/// checkpoint -- if either key is silently missing, `LtxVideoPipeline`'s own null-check silently
/// skips un-normalization entirely (feeding DiT-space latents straight into a VAE decoder trained on
/// a different, rescaled input distribution), which would produce exactly the kind of structured
/// (not random) but visually-wrong artifact seen in the post-timestep-fix real generation run
/// (`docs/diffusion-samples/ltx_video_apple_flipfix.png`).
/// </summary>
public sealed class LtxVaeStatsPresenceDiagnosticTest
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void PerChannelStatistics_TensorsArePresent_AndHaveExpectedShape()
    {
        string? modelPath = FindModelPath(ModelFileName);
        if (modelPath is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);

        bool hasStd = loader.Contains("vae.per_channel_statistics.std-of-means");
        bool hasMean = loader.Contains("vae.per_channel_statistics.mean-of-means");
        Console.WriteLine($"[LTX VAE stats] hasStd={hasStd} hasMean={hasMean}");

        Assert.True(hasStd, "vae.per_channel_statistics.std-of-means MISSING from checkpoint -- un-normalization is silently skipped");
        Assert.True(hasMean, "vae.per_channel_statistics.mean-of-means MISSING from checkpoint -- un-normalization is silently skipped");

        var std = loader.ReadF32("vae.per_channel_statistics.std-of-means");
        var mean = loader.ReadF32("vae.per_channel_statistics.mean-of-means");
        Console.WriteLine($"[LTX VAE stats] std.Length={std.Length} mean.Length={mean.Length}");
        Console.WriteLine($"[LTX VAE stats] std[0..4]={std[0]:F4},{std[1]:F4},{std[2]:F4},{std[3]:F4}");
        Console.WriteLine($"[LTX VAE stats] mean[0..4]={mean[0]:F4},{mean[1]:F4},{mean[2]:F4},{mean[3]:F4}");

        Assert.Equal(128, std.Length);
        Assert.Equal(128, mean.Length);
    }
}
