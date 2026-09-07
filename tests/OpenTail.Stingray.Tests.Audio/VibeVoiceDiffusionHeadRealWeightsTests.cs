using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke tests for <see cref="VibeVoiceDiffusionHead"/>,
/// <see cref="VibeVoiceDpmSolverScheduler"/> and <see cref="VibeVoiceDiffusionSampler"/>.</summary>
public sealed class VibeVoiceDiffusionHeadRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    // Real config numbers dumped from the checkpoint's own config.json (2026-09-07, not guessed).
    private const int HiddenSize = 1536, LatentSize = 64, HeadLayers = 4;
    private const float HeadFfnRatio = 3.0f, RmsNormEps = 1e-5f;
    private const int DdpmNumSteps = 1000;

    private static (VibeVoiceDiffusionHeadWeights, GgufModel)? LoadReal()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        if (path == null) return null;
        var model = GgufModel.Open(path);
        var source = new RvcPackedTensorSource(model);
        var w = VibeVoiceDiffusionHeadWeights.Load(HiddenSize, LatentSize, HeadLayers, HeadFfnRatio, RmsNormEps, source.GetTensor);
        return (w, model);
    }

    [Fact]
    public void Predict_OnRealCheckpoint_ProducesFiniteOutput()
    {
        var loaded = LoadReal();
        Assert.SkipUnless(loaded != null, "vibevoice-1.5b-q8_0.gguf not found");
        var (w, model) = loaded!.Value;
        using var _ = model;

        var rng = new Random(9);
        float[] RandRow(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        const int frames = 3;
        var noisy = Enumerable.Range(0, frames).Select(_ => RandRow(LatentSize)).ToArray();
        var condition = Enumerable.Range(0, frames).Select(_ => RandRow(HiddenSize)).ToArray();

        var output = VibeVoiceDiffusionHead.Predict(w, noisy, condition, timestep: 500f);

        Assert.Equal(frames, output.Length);
        foreach (var row in output)
        {
            Assert.Equal(LatentSize, row.Length);
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
        }
    }

    [Fact]
    public void Sample_OnRealCheckpoint_RunsFullCfgLoop_ProducesFiniteLatent()
    {
        var loaded = LoadReal();
        Assert.SkipUnless(loaded != null, "vibevoice-1.5b-q8_0.gguf not found");
        var (w, model) = loaded!.Value;
        using var _ = model;

        var rng = new Random(13);
        float[] RandRow(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        var positive = RandRow(HiddenSize);
        var negative = RandRow(HiddenSize);
        var initial = RandRow(LatentSize);

        var scheduler = new VibeVoiceDpmSolverScheduler(DdpmNumSteps);
        scheduler.SetTimesteps(inferenceSteps: 5);

        var result = VibeVoiceDiffusionSampler.Sample(w, scheduler, positive, negative, initial, guidanceScale: 1.5f);

        Assert.Equal(LatentSize, result.Length);
        Assert.All(result, v => Assert.True(float.IsFinite(v)));
    }
}
