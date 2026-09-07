using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural (synthetic-weight) tests for <see cref="VibeVoiceTokenizerEncoder"/>.</summary>
public sealed class VibeVoiceTokenizerEncoderTests
{
    private static VibeVoiceTokenizerConfig MakeConfig() => new()
    {
        Channels = 1,
        VaeDim = 4,
        EncoderNFilters = 8,
        EncoderRatios = [2, 2], // reversed internally -> [2,2] (symmetric here for simplicity)
        EncoderDepths = [1, 1, 1], // 3 stages (ratios.Length + 1)
        DisableLastNorm = false,
        LayerNormEps = 1e-6f,
    };

    private static VibeVoiceTokenizerEncoderWeights MakeWeights(VibeVoiceTokenizerConfig config, Random rng)
    {
        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            return a;
        }

        var tensors = new Dictionary<string, float[]>();

        tensors["prefix.downsample_layers.0.0.conv.conv.weight"] = Rand(config.EncoderNFilters * config.Channels * 7);
        tensors["prefix.downsample_layers.0.0.conv.conv.bias"] = Rand(config.EncoderNFilters);

        var ratios = (int[])config.EncoderRatios.Clone();
        Array.Reverse(ratios);
        for (int i = 0; i < ratios.Length; i++)
        {
            int inCh = config.EncoderNFilters * (1 << i);
            int outCh = config.EncoderNFilters * (1 << (i + 1));
            int kernel = ratios[i] * 2;
            string p = $"prefix.downsample_layers.{i + 1}.0.conv.conv";
            tensors[$"{p}.weight"] = Rand(outCh * inCh * kernel);
            tensors[$"{p}.bias"] = Rand(outCh);
        }

        for (int stage = 0; stage < config.EncoderDepths.Length; stage++)
        {
            int channels = config.EncoderNFilters * (1 << stage);
            for (int block = 0; block < config.EncoderDepths[stage]; block++)
            {
                string bp = $"prefix.stages.{stage}.{block}";
                tensors[$"{bp}.norm.weight"] = Enumerable.Repeat(1f, channels).ToArray();
                tensors[$"{bp}.mixer.conv.conv.conv.weight"] = Rand(channels * 1 * 7);
                tensors[$"{bp}.mixer.conv.conv.conv.bias"] = Rand(channels);
                tensors[$"{bp}.gamma"] = Rand(channels);
                tensors[$"{bp}.ffn_norm.weight"] = Enumerable.Repeat(1f, channels).ToArray();
                tensors[$"{bp}.ffn.linear1.weight"] = Rand(4 * channels * channels);
                tensors[$"{bp}.ffn.linear1.bias"] = Rand(4 * channels);
                tensors[$"{bp}.ffn.linear2.weight"] = Rand(channels * 4 * channels);
                tensors[$"{bp}.ffn.linear2.bias"] = Rand(channels);
                tensors[$"{bp}.ffn_gamma"] = Rand(channels);
            }
        }

        int finalChannels = config.EncoderNFilters * (1 << (config.EncoderDepths.Length - 1));
        tensors["prefix.norm.weight"] = Enumerable.Repeat(1f, finalChannels).ToArray();
        tensors["prefix.head.conv.conv.weight"] = Rand(config.VaeDim * finalChannels * 7);
        tensors["prefix.head.conv.conv.bias"] = Rand(config.VaeDim);

        return VibeVoiceTokenizerEncoderWeights.Load(config, "prefix", name => tensors[name]);
    }

    [Fact]
    public void Encode_ProducesFiniteOutput_WithExpectedChannelCount()
    {
        var config = MakeConfig();
        var rng = new Random(1);
        var weights = MakeWeights(config, rng);

        var waveform = new float[4096];
        for (int i = 0; i < waveform.Length; i++) waveform[i] = (float)(rng.NextDouble() * 2 - 1);

        var latent = VibeVoiceTokenizerEncoder.Encode(weights, waveform, config.LayerNormEps);

        Assert.Equal(config.VaeDim, latent.Length);
        Assert.True(latent[0].Length > 0);
        foreach (var row in latent)
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Encode_IsCausal_LongerInputReproducesShorterInputsPrefixApproximately()
    {
        var config = MakeConfig();
        var rng = new Random(2);
        var weights = MakeWeights(config, rng);

        var longWaveform = new float[4096];
        for (int i = 0; i < longWaveform.Length; i++) longWaveform[i] = (float)(rng.NextDouble() * 2 - 1);
        var shortWaveform = longWaveform.AsSpan(0, 2048).ToArray();

        var shortLatent = VibeVoiceTokenizerEncoder.Encode(weights, shortWaveform, config.LayerNormEps);
        var longLatent = VibeVoiceTokenizerEncoder.Encode(weights, longWaveform, config.LayerNormEps);

        // Causal convs mean the early output frames should match closely between the short and
        // long runs (allowing for the real Encodec-style "extra padding to land frame count
        // exactly" term, which can shift the last couple of frames -- check a safe prefix).
        int checkFrames = Math.Min(shortLatent[0].Length, longLatent[0].Length) - 2;
        Assert.True(checkFrames > 0);
        for (int c = 0; c < config.VaeDim; c++)
            for (int t = 0; t < checkFrames; t++)
                Assert.Equal(shortLatent[c][t], longLatent[c][t], precision: 3);
    }
}
