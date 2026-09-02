using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep;
using OpenTail.Stingray.Diffusion.AceStep.Vae;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real-weight smoke test for ACE-Step's `AutoencoderOobleck` VAE decoder. Non-degeneracy
/// receipt (finite, non-silent, real 25Hz-latent-to-48kHz-PCM shape math checked), not yet a
/// numeric golden-parity test against a real `diffusers` `AutoencoderOobleck.decode` reference
/// run.
/// </summary>
public sealed class AceStepOobleckDecoderTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Decode_RealWeights_ProducesCorrectShapeAndNonDegenerateAudio()
    {
        string? vaePath = FindRepoFile("models/acestep-v15/vae.safetensors");
        Assert.SkipUnless(vaePath != null, "models/acestep-v15/vae.safetensors not found");

        using var loader = SafetensorsLoader.Open(vaePath!);
        var weights = AceStepOobleckDecoderWeights.Load(loader);

        // Real hop length = product(downsampling_ratios) = 2*4*4*6*10 = 1920.
        // 48000 / 1920 = 25 exactly -- this is the real derivation of ACE-Step's "25Hz latent"
        // claim, not assumed: confirmed via this decoder's own real config numbers.
        int hopLength = AceStepConfig.VaeDownsamplingRatios.Aggregate(1, (a, b) => a * b);
        Assert.Equal(1920, hopLength);
        Assert.Equal(25, AceStepConfig.VaeSampleRate / hopLength);

        int latentFrames = 50; // 2 real seconds @ 25Hz
        var rng = new Random(0);
        var latent = new float[AceStepConfig.VaeDecoderInputChannels * latentFrames];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var pcm = AceStepOobleckDecoder.Decode(weights, latent, latentFrames);

        int expectedSamples = latentFrames * hopLength;
        Assert.Equal(AceStepConfig.VaeAudioChannels * expectedSamples, pcm.Length);

        foreach (var v in pcm)
            Assert.True(float.IsFinite(v), "PCM contains NaN/Inf -- decoder produced degenerate output");

        double sumSq = 0;
        foreach (var v in pcm) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM RMS energy ({rms}) is near-silent -- likely a wiring bug, not real audio");
    }
}
