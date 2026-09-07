using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for VibeVoice ASR's tokenizer encoders (acoustic + semantic), connector,
/// and Gaussian latent sampler -- loads the real checkpoint and runs the full
/// <see cref="VibeVoiceSpeechFeatures.Extract"/> chain on a synthetic waveform. Not a numeric
/// golden-parity check (no captured reference trace) -- confirms real weight shapes resolve and
/// produce finite, correctly-shaped output.
/// </summary>
public sealed class VibeVoiceTokenizerEncoderRealWeightsTests : HeavyTestBase
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

    // Real config from the checkpoint's embedded config.json (confirmed via VibeVoiceAsrDumpDebugTest).
    private static VibeVoiceTokenizerConfig AcousticConfig() => new()
    {
        Channels = 1,
        VaeDim = 64,
        EncoderNFilters = 32,
        EncoderRatios = [8, 5, 5, 4, 2, 2],
        EncoderDepths = [3, 3, 3, 3, 3, 3, 8],
        DisableLastNorm = true,
        LayerNormEps = 1e-5f,
        FixStd = 0.5f,
    };

    private static VibeVoiceTokenizerConfig SemanticConfig() => new()
    {
        Channels = 1,
        VaeDim = 128,
        EncoderNFilters = 32,
        EncoderRatios = [8, 5, 5, 4, 2, 2],
        EncoderDepths = [3, 3, 3, 3, 3, 3, 8],
        DisableLastNorm = true,
        LayerNormEps = 1e-5f,
        FixStd = 0f,
    };

    [Fact]
    public void ExtractSpeechFeatures_OnRealCheckpoint_ProducesFiniteCorrectlyShapedOutput()
    {
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var acousticEncoder = VibeVoiceTokenizerEncoderWeights.Load(AcousticConfig(), "model.acoustic_tokenizer.encoder", source.GetTensor);
        var semanticEncoder = VibeVoiceTokenizerEncoderWeights.Load(SemanticConfig(), "model.semantic_tokenizer.encoder", source.GetTensor);
        var acousticConnector = VibeVoiceConnectorWeights.Load("model.acoustic_connector", inputDim: 64, hiddenSize: 3584, source.GetTensor);
        var semanticConnector = VibeVoiceConnectorWeights.Load("model.semantic_connector", inputDim: 128, hiddenSize: 3584, source.GetTensor);

        // Real downsample factor: product(encoder_ratios) * downsample_layers[0]'s stride(1) --
        // use a waveform long enough to survive the full 6-stage downsample cleanly.
        int waveformLength = 8 * 5 * 5 * 4 * 2 * 2 * 8; // a few real output frames
        var waveform = new float[waveformLength];
        var rng = new Random(11);
        for (int i = 0; i < waveform.Length; i++) waveform[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        var combined = VibeVoiceSpeechFeatures.Extract(
            acousticEncoder, AcousticConfig(), acousticConnector,
            semanticEncoder, SemanticConfig(), semanticConnector,
            waveform, new Random(7));

        Assert.Equal(3584, combined.Length);
        Assert.True(combined[0].Length > 0);
        foreach (var row in combined)
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
    }
}
