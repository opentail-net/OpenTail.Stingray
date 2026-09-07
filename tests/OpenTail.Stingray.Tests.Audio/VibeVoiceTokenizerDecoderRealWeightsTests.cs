using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="VibeVoiceTokenizerDecoder"/> (the TTS-side
/// acoustic-latent-to-waveform decoder, distinct from the ASR-side encoder already verified).</summary>
public sealed class VibeVoiceTokenizerDecoderRealWeightsTests : HeavyTestBase
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

    // Real config numbers dumped from the checkpoint's own config.json (2026-09-07, not
    // guessed): acoustic_tokenizer_config.decoder_depths is null, so the real reference falls
    // back to reverse(encoder_depths) -- encoder_depths="3-3-3-3-3-3-8" reversed = [8,3,3,3,3,3,3].
    private const int DecoderNFilters = 32, VaeDim = 64, Channels = 1;
    private static readonly int[] DecoderRatios = [8, 5, 5, 4, 2, 2];
    private static readonly int[] DecoderDepths = [8, 3, 3, 3, 3, 3, 3];
    private const float LayerNormEps = 1e-5f;

    [Fact]
    public void Decode_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var config = new VibeVoiceTokenizerConfig
        {
            Channels = Channels,
            VaeDim = VaeDim,
            EncoderNFilters = DecoderNFilters,
            EncoderRatios = DecoderRatios,
            EncoderDepths = DecoderDepths,
            DisableLastNorm = true,
            LayerNormEps = LayerNormEps,
        };
        var w = VibeVoiceTokenizerDecoderWeights.Load(config, DecoderNFilters, DecoderRatios, DecoderDepths,
            "model.acoustic_tokenizer.decoder", source.GetTensor);

        var rng = new Random(21);
        const int latentFrames = 6;
        var latent = new float[VaeDim][];
        for (int c = 0; c < VaeDim; c++)
        {
            latent[c] = new float[latentFrames];
            for (int t = 0; t < latentFrames; t++) latent[c][t] = (float)(rng.NextDouble() * 0.2 - 0.1);
        }

        var waveform = VibeVoiceTokenizerDecoder.Decode(w, latent, LayerNormEps);

        Assert.Single(waveform);
        int totalUpsample = 1;
        foreach (int r in DecoderRatios) totalUpsample *= r;
        Assert.True(waveform[0].Length > latentFrames * totalUpsample / 2, "waveform should be roughly latentFrames*totalUpsample samples long");
        Assert.All(waveform[0], v => Assert.True(float.IsFinite(v)));
        Assert.Contains(waveform[0], v => v != 0f);
    }
}
