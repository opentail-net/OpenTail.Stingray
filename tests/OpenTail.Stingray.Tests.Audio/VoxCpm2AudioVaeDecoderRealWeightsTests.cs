using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for VoxCPM2's AudioVAE decoder: loads the real
/// `voxcpm2-q8_0.gguf` checkpoint's `audiovae_weights/*` tensors and decodes synthetic random
/// latent features (this pipeline's LM/DiT/CFM feature generator is not yet ported -- see
/// docs/audio-review-progress.md -- so a real latent input from the real feature generator is
/// not yet available; this test therefore checks the decoder alone: finite, correctly-shaped,
/// in-range output on the real weights, not numeric golden parity).
/// </summary>
public sealed class VoxCpm2AudioVaeDecoderRealWeightsTests : HeavyTestBase
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

    // Real config.audio_vae_config from the checkpoint's embedded config.json (confirmed via
    // VoxCpm2DumpDebugTest, not guessed).
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
    public void Decode_OnRealCheckpoint_ProducesFiniteBoundedAudio()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var config = RealConfig();
        var weights = VoxCpm2AudioVaeDecoderWeights.Load(config, name => source.GetTensor($"audiovae_weights/{name}"));

        int frames = 4;
        var rng = new Random(7);
        var latent = new float[config.LatentDim * frames];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        var pcm = VoxCpm2AudioVaeDecoder.Decode(config, weights, latent, frames);

        int expectedStride = config.DecoderRates.Aggregate(1, (a, b) => a * b);
        Assert.Equal(frames * expectedStride, pcm.Length);
        Assert.All(pcm, s => Assert.True(float.IsFinite(s)));
        Assert.All(pcm, s => Assert.InRange(s, -1f, 1f));
    }
}
