using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Conditioning;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using OpenTail.Stingray.Diffusion.AceStep.Transformer;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class GenerateAudioDiffusionSamplesTests
{
    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string GetOutputDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "docs", "diffusion-samples");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        string fallback = Path.Combine(Directory.GetCurrentDirectory(), "docs", "diffusion-samples");
        Directory.CreateDirectory(fallback);
        return fallback;
    }

    [Fact]
    public void Generate_StableAudioSmall_Sample()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-small-music-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-small-music-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outPath = Path.Combine(GetOutputDir(), "sample_stable_audio_small_lofi_4s.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "A smooth lo-fi jazz beat with soft electric piano and mellow vinyl crackle",
            DurationSeconds = 4f,
            Steps = 16,
            CfgScale = 6.0f,
            Seed = 42,
            OutputPath = outPath,
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");
        Assert.True(File.Exists(outPath), $"Output file {outPath} does not exist");
        Console.WriteLine($"[Sample Generated] Stable Audio Small -> {outPath} ({pcm.Length} samples)");
    }

    [Fact]
    public void Generate_StableAudioMedium_Sample()
    {
        string? ditDir = FindRepoDir("models/stable-audio-3-medium-base");
        string? t5gemmaDir = FindRepoDir("models/stable-audio-3-t5gemma");
        Assert.SkipUnless(ditDir != null, "models/stable-audio-3-medium-base not found");
        Assert.SkipUnless(t5gemmaDir != null, "models/stable-audio-3-t5gemma not found");

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir!);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir!);
        using var pipeline = new StableAudioMediumPipeline(ditWeights, textEncoderWeights, t5gemmaDir!);

        string outPath = Path.Combine(GetOutputDir(), "sample_stable_audio_medium_orchestral_4s.wav");

        var pcm = pipeline.Generate(new StableAudioRequest
        {
            Prompt = "A cinematic orchestral crescendo with sweeping strings and timpani",
            DurationSeconds = 4f,
            Steps = 12,
            CfgScale = 6.0f,
            Seed = 42,
            OutputPath = outPath,
        });

        Assert.True(pcm.Length > 0, "generated zero samples");
        foreach (var v in pcm) Assert.True(float.IsFinite(v), "PCM contains NaN/Inf");
        Assert.True(File.Exists(outPath), $"Output file {outPath} does not exist");
        Console.WriteLine($"[Sample Generated] Stable Audio Medium -> {outPath} ({pcm.Length} samples)");
    }

    [Fact]
    public void Generate_AceStep_Sample()
    {
        string? turboPath = FindRepoFile("models/acestep-v15/turbo.safetensors");
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        string? ggufPath = FindRepoFile("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        Assert.SkipUnless(turboPath != null, "models/acestep-v15/turbo.safetensors not found");
        Assert.SkipUnless(vaePath != null, "models/acestep-v15/vae.safetensors not found");
        Assert.SkipUnless(ggufPath != null, "models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf not found");

        using var turboLoader = SafetensorsLoader.Open(turboPath!);
        var ditWeights = AceStepDiTWeights.Load(turboLoader);
        var conditionWeights = AceStepConditionEncoderWeights.Load(turboLoader);
        var timbreWeights = AceStepTimbreEncoderWeights.Load(turboLoader);

        using var vaeLoader = SafetensorsLoader.Open(vaePath!);
        var vaeWeights = AceStepOobleckDecoderWeights.Load(vaeLoader);
        var vaeEncoderWeights = AceStepOobleckEncoderWeights.Load(vaeLoader);

        using var textEncoder = new AceStepQwen3TextEncoder(ggufPath!);

        var model = new AceStepModel
        {
            Transformer = ditWeights,
            Vae = vaeWeights,
            VaeEncoder = vaeEncoderWeights,
            TextEncoder = textEncoder,
            ConditionEncoder = conditionWeights,
            TimbreEncoder = timbreWeights,
        };
        using var pipeline = new AceStepPipeline(model);

        var result = pipeline.Generate(new AceStepGenerationParams
        {
            Prompt = "An upbeat electronic synthwave track with retro bass and energetic drums",
            Lyrics = "",
            Instrumental = true,
            DurationSeconds = 2.5f,
            Seed = 1234,
        });

        Assert.Equal(AceStepConfig.VaeSampleRate, result.SampleRate);
        Assert.True(result.SampleCount > 0, "generated zero samples");
        Assert.Equal(result.Left.Length, result.Right.Length);

        // Interleave stereo channels for standard WAV file format
        var interleaved = new float[result.SampleCount * 2];
        for (int i = 0; i < result.SampleCount; i++)
        {
            interleaved[i * 2] = result.Left[i];
            interleaved[i * 2 + 1] = result.Right[i];
        }

        string outPath = Path.Combine(GetOutputDir(), "sample_acestep_turbo_synthwave_2.5s.wav");
        WavWriter.WriteWav(outPath, interleaved, result.SampleRate, channels: 2, DitherMode.Tpdf);

        Assert.True(File.Exists(outPath), $"Output file {outPath} does not exist");
        Console.WriteLine($"[Sample Generated] ACE-Step Turbo -> {outPath} ({result.SampleCount} samples)");
    }
}
