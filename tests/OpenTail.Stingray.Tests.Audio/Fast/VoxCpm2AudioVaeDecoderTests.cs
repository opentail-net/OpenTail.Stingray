using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural (synthetic-weight) tests for <see cref="VoxCpm2AudioVaeDecoder"/>.</summary>
public sealed class VoxCpm2AudioVaeDecoderTests
{
    private static VoxCpm2AudioVaeConfig MakeConfig() => new()
    {
        LatentDim = 8,
        DecoderDim = 32, // must be divisible by 2^len(DecoderRates)
        DecoderRates = [2, 2],
        OutputSampleRate = 48000,
        SampleRateBinBoundaries = [24000],
    };

    private static VoxCpm2AudioVaeDecoderWeights MakeWeights(VoxCpm2AudioVaeConfig config, Random rng)
    {
        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            return a;
        }

        var tensors = new Dictionary<string, float[]>();

        void PutConv1d(string prefix, int outCh, int inCh, int kernel, bool depthwise)
        {
            int storedIn = depthwise ? 1 : inCh;
            tensors[$"{prefix}.weight_v"] = Rand(outCh * storedIn * kernel);
            tensors[$"{prefix}.weight_g"] = Rand(outCh).Select(MathF.Abs).ToArray();
            tensors[$"{prefix}.bias"] = Rand(outCh);
        }

        void PutConvTranspose1d(string prefix, int inCh, int outCh, int kernel)
        {
            tensors[$"{prefix}.weight_v"] = Rand(inCh * outCh * kernel);
            tensors[$"{prefix}.weight_g"] = Rand(inCh).Select(MathF.Abs).ToArray();
            tensors[$"{prefix}.bias"] = Rand(outCh);
        }

        void PutResidualUnit(string prefix, int channels)
        {
            tensors[$"{prefix}.block.0.alpha"] = Rand(channels);
            PutConv1d($"{prefix}.block.1", channels, channels, 7, depthwise: true);
            tensors[$"{prefix}.block.2.alpha"] = Rand(channels);
            PutConv1d($"{prefix}.block.3", channels, channels, 1, depthwise: false);
        }

        PutConv1d("decoder.model.0", config.LatentDim, config.LatentDim, 7, depthwise: true);
        PutConv1d("decoder.model.1", config.DecoderDim, config.LatentDim, 1, depthwise: false);

        int buckets = config.SampleRateBucketCount;
        for (int i = 0; i < config.DecoderRates.Length; i++)
        {
            int inputCh = config.DecoderDim >> i;
            int outputCh = config.DecoderDim >> (i + 1);
            int stride = config.DecoderRates[i];
            int modelIndex = i + 2;
            string p = $"decoder.model.{modelIndex}";

            tensors[$"decoder.sr_cond_model.{modelIndex}.scale_embed.weight"] = Rand(buckets * inputCh);
            tensors[$"decoder.sr_cond_model.{modelIndex}.bias_embed.weight"] = Rand(buckets * inputCh);
            tensors[$"{p}.block.0.alpha"] = Rand(inputCh);
            PutConvTranspose1d($"{p}.block.1", inputCh, outputCh, 2 * stride);
            PutResidualUnit($"{p}.block.2", outputCh);
            PutResidualUnit($"{p}.block.3", outputCh);
            PutResidualUnit($"{p}.block.4", outputCh);
        }

        int finalChannels = config.DecoderDim >> config.DecoderRates.Length;
        tensors[$"decoder.model.{config.DecoderRates.Length + 2}.alpha"] = Rand(finalChannels);
        PutConv1d($"decoder.model.{config.DecoderRates.Length + 3}", 1, finalChannels, 7, depthwise: false);

        return VoxCpm2AudioVaeDecoderWeights.Load(config, name => tensors[name]);
    }

    [Fact]
    public void Decode_ProducesCorrectSampleCount_AndFiniteBoundedOutput()
    {
        var config = MakeConfig();
        var rng = new Random(1);
        var weights = MakeWeights(config, rng);

        int frames = 5;
        var latent = new float[config.LatentDim * frames];
        for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);

        var pcm = VoxCpm2AudioVaeDecoder.Decode(config, weights, latent, frames);

        int expectedStride = config.DecoderRates.Aggregate(1, (a, b) => a * b);
        Assert.Equal(frames * expectedStride, pcm.Length);
        Assert.All(pcm, s => Assert.True(float.IsFinite(s)));
        Assert.All(pcm, s => Assert.InRange(s, -1f, 1f)); // post-Tanh
    }

    [Fact]
    public void Decode_IsCausal_LongerInputReproducesShorterInputsPrefixExactly()
    {
        var config = MakeConfig();
        var rng = new Random(2);
        var weights = MakeWeights(config, rng);

        int shortFrames = 3;
        int longFrames = 5;
        var longLatent = new float[config.LatentDim * longFrames];
        for (int i = 0; i < longLatent.Length; i++) longLatent[i] = (float)(rng.NextDouble() * 2 - 1);

        // Channel-major layout: [channel][frame]. A shorter input is the same data truncated per
        // channel (first `shortFrames` columns), not a flat prefix.
        var shortLatent = new float[config.LatentDim * shortFrames];
        for (int c = 0; c < config.LatentDim; c++)
            Array.Copy(longLatent, c * longFrames, shortLatent, c * shortFrames, shortFrames);

        var shortPcm = VoxCpm2AudioVaeDecoder.Decode(config, weights, shortLatent, shortFrames);
        var longPcm = VoxCpm2AudioVaeDecoder.Decode(config, weights, longLatent, longFrames);

        Assert.Equal(shortPcm.Length, longPcm.AsSpan(0, shortPcm.Length).ToArray().Length);
        for (int i = 0; i < shortPcm.Length; i++)
            Assert.Equal(shortPcm[i], longPcm[i], precision: 3);
    }
}
