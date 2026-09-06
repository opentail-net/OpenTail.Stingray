
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: A/B comparison between the shared HiFTVocoderKernels SineGen
/// (continuous per-sample phase accumulation, currently used by both Chatterbox and CosyVoice2/3)
/// and CosyVoiceSineGenFrameHoldExperiment (frame-rate cumulative-phase-then-hold, matching real
/// reference implementations), on the exact same CosyVoice3-generated mel, to test whether the
/// frame-hold algorithm fixes CosyVoice3's long-standing "wobbling distortion" symptom.</summary>
public sealed class CosyVoice3SineGenFrameHoldABTest : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void CurrentContinuous_Vs_FrameHoldExperiment()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "CosyVoice3 GGUF model not found");

        const string prompt = "Hello, I will make some lunch, darling!";

        using var pipelineCurrent = OpenTail.Stingray.Audio.CosyVoice.CosyVoice3Pipeline.Load(modelPath!);
        var wavCurrent = pipelineCurrent.Generate(prompt, seed: 42, useFrameHoldSineGenExperiment: false);

        using var pipelineExperiment = OpenTail.Stingray.Audio.CosyVoice.CosyVoice3Pipeline.Load(modelPath!);
        var wavExperiment = pipelineExperiment.Generate(prompt, seed: 42, useFrameHoldSineGenExperiment: true);

        Assert.NotEmpty(wavCurrent);
        Assert.NotEmpty(wavExperiment);

        string? outDir = FindRepoFile("docs/audio-samples");
        if (outDir != null)
        {
            new OpenTail.Stingray.Audio.AudioGenerationResult(wavCurrent, 24000).SaveWav(Path.Combine(outDir, "cosyvoice3-sinegen-CURRENT-continuous.wav"));
            new OpenTail.Stingray.Audio.AudioGenerationResult(wavExperiment, 24000).SaveWav(Path.Combine(outDir, "cosyvoice3-sinegen-EXPERIMENT-framehold.wav"));
        }

        Console.Error.WriteLine($"[CosyVoice3SineGenAB] current samples={wavCurrent.Length} experiment samples={wavExperiment.Length}");
    }
}
