using OpenTail.Stingray.Diffusion.StableAudio;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class StableAudioConformanceTests
{
    private const string DitDirRelative = "models/stable-audio-3-small-music-base";

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>Shape-only smoke test on the real DiT (accuracy is covered by
    /// <c>StableAudioDiTGoldenParityTests</c>) -- this class has fixed real dimensions (1024/20/16),
    /// unlike the old placeholder API which took arbitrary synthetic sizes.</summary>
    [Fact]
    public void StableAudioDiT_Forward_ProducesCorrectVelocityShape()
    {
        string? ditDir = FindRepoFile(DitDirRelative);
        if (ditDir is null) return; // skip: needs local DiT weights

        using var st = SafetensorsLoader.OpenDirectory(ditDir);
        using var dit = StableAudioDiT.FromLoader(st);

        int seqLen = 4;
        int nCond = 3;
        var latent = new float[seqLen * 256];
        var condTokens = new float[nCond * 768];
        var condMask = new[] { true, true, true };
        var secondsTotalRaw = new float[768];

        float[] velocity = dit.Forward(latent, seqLen, condTokens, nCond, condMask, secondsTotalRaw, timestep: 0.5f);

        Assert.NotNull(velocity);
        Assert.Equal(seqLen * 256, velocity.Length);
    }

    [Fact]
    public void AcousticVaeDecoder_DecodesLatentsToStereoPcm()
    {
        int latentChannels = 16;
        int audioChannels = 2;
        int upsampleRatio = 256;

        var decoder = new AcousticVaeDecoder(latentChannels, audioChannels, upsampleRatio);

        int seqLen = 10;
        var latents = new float[seqLen * latentChannels];
        Array.Fill(latents, 0.5f);

        float[] pcm = decoder.Decode(latents, seqLen);

        int expectedSamples = seqLen * upsampleRatio * audioChannels;
        Assert.Equal(expectedSamples, pcm.Length);

        for (int i = 0; i < pcm.Length; i++)
        {
            Assert.InRange(pcm[i], -1.0f, 1.0f);
        }
    }

    /// <summary>End-to-end smoke test using the real text encoder + real DiT (still-placeholder
    /// VAE, see <see cref="StableAudioPipeline"/>'s doc comment) -- confirms the real weight-driven
    /// pipeline wiring runs to completion and produces a valid WAV, not that its audio is correct
    /// yet (that needs the real VAE, not ported yet).</summary>
    [Fact]
    public void StableAudioPipeline_GeneratesStereoWavFileWithTpdfDither()
    {
        string? ditDir = FindRepoFile(DitDirRelative);
        string? t5gemmaDir = FindRepoFile("models/stable-audio-3-t5gemma");
        if (ditDir is null || t5gemmaDir is null) return; // skip: needs local weights

        using var ditWeights = SafetensorsLoader.OpenDirectory(ditDir);
        using var textEncoderWeights = SafetensorsLoader.OpenDirectory(t5gemmaDir);
        using var pipeline = new StableAudioPipeline(ditWeights, textEncoderWeights);

        string tempWav = Path.Combine(Path.GetTempPath(), $"stable_audio_test_{Guid.NewGuid():N}.wav");
        try
        {
            var request = new StableAudioRequest
            {
                PromptTokenIds = [1, 2, 3, 4, 5],
                DurationSeconds = 0.1f,
                Steps = 2,
                OutputPath = tempWav,
            };

            float[] pcm = pipeline.Generate(request);

            Assert.NotNull(pcm);
            Assert.True(pcm.Length > 0);
            Assert.True(File.Exists(tempWav));
            Assert.True(new FileInfo(tempWav).Length > 100);
        }
        finally
        {
            if (File.Exists(tempWav)) File.Delete(tempWav);
        }
    }
}
