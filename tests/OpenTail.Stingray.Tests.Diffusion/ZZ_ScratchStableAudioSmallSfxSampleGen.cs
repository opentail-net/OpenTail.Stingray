using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class ZZ_ScratchStableAudioSmallSfxSampleGen
{
    [Fact]
    public void GenerateRealSfxSample()
    {
        string ditDir = @"C:\Git-Public\OpenTail.Stingray\models\stable-audio-3-small-sfx-base";
        string t5gemmaDir = @"C:\Git-Public\OpenTail.Stingray\models\stable-audio-3-t5gemma";
        if (!Directory.Exists(ditDir) || !Directory.Exists(t5gemmaDir)) return;

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir);

        string outPath = @"C:\Git-Public\OpenTail.Stingray\docs\diffusion-samples\sample_stable_audio3_small_sfx_glass-shatter_4s_2026-09-18.wav";
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "a glass bottle shattering on a hard floor",
            DurationSeconds = 4f,
            Steps = 16,
            CfgScale = 6.0f,
            Seed = 1234,
            OutputPath = outPath,
        });
        sw.Stop();
        Console.Error.WriteLine($"[StableAudio3-SmallSfx sample] generated in {sw.Elapsed.TotalSeconds:F1}s, saved to {outPath}");

        Assert.True(pcm.Length > 0);
        foreach (var v in pcm) Assert.True(float.IsFinite(v));
    }
}
