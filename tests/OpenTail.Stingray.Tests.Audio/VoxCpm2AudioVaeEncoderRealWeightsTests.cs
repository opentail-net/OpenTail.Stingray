using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for VoxCPM2's AudioVAE ENCODER (voice cloning / reference-audio
/// conditioning), ported this session -- loads the real `voxcpm2-q8_0.gguf` checkpoint's
/// `audiovae_weights/encoder.*` tensors and encodes a synthetic waveform, confirming finite,
/// correctly-shaped output on real weights.
/// </summary>
public sealed class VoxCpm2AudioVaeEncoderRealWeightsTests : HeavyTestBase
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

    private static VoxCpm2AudioVaeConfig RealConfig() => new()
    {
        LatentDim = 64,
        DecoderDim = 2048,
        DecoderRates = [8, 6, 5, 2, 2, 2],
        OutputSampleRate = 48000,
        SampleRateBinBoundaries = [20000, 30000, 40000],
        EncoderDim = 128,
        EncoderRates = [2, 5, 8, 8],
    };

    [Fact]
    public void Encode_OnRealCheckpoint_ProducesFiniteCorrectlyShapedLatent()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var config = RealConfig();
        var weights = VoxCpm2AudioVaeDecoderWeights.Load(config, name => source.GetTensor($"audiovae_weights/{name}"));

        int encoderStride = config.EncoderRates.Aggregate(1, (a, b) => a * b);
        int samples = encoderStride * 20; // a few real downsample-stride's worth of samples
        var rng = new Random(11);
        var pcm = new float[samples];
        for (int i = 0; i < samples; i++) pcm[i] = (float)(rng.NextDouble() * 2 - 1);

        var latent = VoxCpm2AudioVaeDecoder.Encode(config, weights, pcm);

        int expectedFrames = samples / encoderStride;
        Assert.Equal(config.LatentDim * expectedFrames, latent.Length);
        Assert.All(latent, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(latent, v => v != 0f);
    }
}
