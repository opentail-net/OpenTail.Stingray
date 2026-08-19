using OpenTail.Stingray.Diffusion.StableAudio;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class StableAudioConformanceTests
{
    [Fact]
    public void StableAudioDiT_Forward_ProducesCorrectVelocityShape()
    {
        var @params = new StableAudioParams
        {
            LatentChannels = 16,
            HiddenSize = 64,
            Depth = 2,
            NumHeads = 4,
            TextContextDim = 32,
            TimingFeaturesDim = 64
        };

        var dit = new StableAudioDiT(@params);

        int seqLen = 20; // ~0.5 seconds of acoustic latents
        var latent = new float[seqLen * @params.LatentChannels];
        var txtEmbeds = new float[16 * @params.TextContextDim];

        float[] velocity = dit.Forward(
            latent,
            seqLen,
            txtEmbeds,
            timestep: 0.5f,
            secondsStart: 0.0f,
            secondsTotal: 5.0f,
            guidance: 5.0f);

        Assert.NotNull(velocity);
        Assert.Equal(seqLen * @params.LatentChannels, velocity.Length);
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

        // Verify bounds in [-1.0, 1.0]
        for (int i = 0; i < pcm.Length; i++)
        {
            Assert.InRange(pcm[i], -1.0f, 1.0f);
        }
    }

    [Fact]
    public void StableAudioPipeline_GeneratesStereoWavFileWithTpdfDither()
    {
        var @params = new StableAudioParams
        {
            LatentChannels = 8,
            HiddenSize = 32,
            Depth = 1,
            NumHeads = 2,
            TextContextDim = 16,
            TimingFeaturesDim = 32,
            SampleRate = 44100,
            AudioChannels = 2,
            LatentFrameRate = 10.0f
        };

        using var pipeline = new StableAudioPipeline(@params);

        string tempWav = Path.Combine(Path.GetTempPath(), $"stable_audio_test_{Guid.NewGuid():N}.wav");
        try
        {
            var request = new StableAudioRequest
            {
                Prompt = "80s synthwave drum loop with punchy analog kick",
                DurationSeconds = 1.0f,
                Steps = 2,
                Guidance = 3.0f,
                OutputPath = tempWav
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
