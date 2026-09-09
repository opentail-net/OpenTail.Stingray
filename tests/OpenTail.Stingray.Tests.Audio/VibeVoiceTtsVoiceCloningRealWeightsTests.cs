using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for VibeVoice TTS's voice-cloning path (`Voice input:` prompt
/// section + real per-speaker acoustic-latent splicing), the real gap flagged in
/// <see cref="VibeVoiceTtsPromptBuilderRealWeightsTests"/>'s doc comment ("Deliberately does NOT
/// implement the reference's real `Voice input:` speaker-audio section"). First real reference-
/// audio-conditioned generation run for this model.
/// </summary>
public sealed class VibeVoiceTtsVoiceCloningRealWeightsTests : HeavyTestBase
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

    private static GgufTokenizer LoadTokenizerFromPackedGguf(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("VibeVoice TTS packed GGUF has no embedded_files metadata.");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string dir = Path.Combine(Path.GetTempPath(), "stingray-vibevoice-tts-tokenizer");
        Directory.CreateDirectory(dir);
        string[] wanted = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"];
        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            if (Array.IndexOf(wanted, name) < 0) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            File.WriteAllBytes(Path.Combine(dir, name), bytes[(int)start..(int)end]);
        }

        var result = HuggingFaceTokenizerSource.Load(dir);
        if (result.Source is null)
            throw new InvalidOperationException($"VibeVoice TTS tokenizer load failed: {string.Join("; ", result.Rejections)}");
        return GgufTokenizer.FromSource(result.Source);
    }

    private const int NumLayers = 28, HiddenDim = 1536, NumHeads = 12, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 8960, VocabSize = 151936;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    private const int DecoderNFilters = 32, AcousticVaeDim = 64, SemanticVaeDim = 128, Channels = 1;
    private static readonly int[] TokenizerRatios = [8, 5, 5, 4, 2, 2];
    private static readonly int[] EncoderDepths = [3, 3, 3, 3, 3, 3, 8];
    private static readonly int[] DecoderDepths = [8, 3, 3, 3, 3, 3, 3];
    private const float TokenizerLayerNormEps = 1e-5f;
    private const float FixStd = 0.5f; // real, confirmed via VibeVoiceTokenizerEncoderRealWeightsTests (same checkpoint family)
    private const int SpeechTokCompressRatio = 3200; // real product(TokenizerRatios) -- one real compressed acoustic-encoder frame per this many real 24kHz samples

    private const int HeadLayers = 4;
    private const float HeadFfnRatio = 3.0f, HeadRmsNormEps = 1e-5f;
    private const int DdpmNumSteps = 1000;

    private const int SpeechStartId = 151652, SpeechEndId = 151653, SpeechDiffusionId = 151654, EosId = 151643;

    [Fact]
    public void BuildVoiceCloningPrompt_ThenGenerate_OnRealCheckpoint_ProducesFiniteNonSilentAudio()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var tokenizer = LoadTokenizerFromPackedGguf(model);

        // One real synthetic 24kHz speaker-reference waveform, long enough for a handful of real
        // compressed acoustic frames (matches SpeechTokCompressRatio's real product-of-ratios).
        int speakerSamples = SpeechTokCompressRatio * 4;
        var rng = new Random(41);
        var speakerWaveform = new float[speakerSamples];
        for (int i = 0; i < speakerSamples; i++) speakerWaveform[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        string script = "Speaker 0: This is a real voice-cloning prompt test.";
        var (promptTokenIds, speechMask, tokenCounts) = VibeVoiceTtsPromptBuilder.BuildPromptWithVoiceCloning(
            text => [.. tokenizer.Encode(text)], script, SpeechStartId, SpeechEndId, SpeechDiffusionId,
            [speakerSamples], SpeechTokCompressRatio);
        Assert.NotEmpty(promptTokenIds);
        Assert.Equal(promptTokenIds.Length, speechMask.Length);
        Assert.Contains(true, speechMask);
        Assert.Equal(tokenCounts[0], speechMask.Count(m => m));

        using var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);
        using var negativeFwd = new ForwardPass(llm, backend, hp);

        var textEmbeddingTable = source.GetTensor("model.language_model.embed_tokens.weight");
        var diffusionHeadWeights = VibeVoiceDiffusionHeadWeights.Load(HiddenDim, AcousticVaeDim, HeadLayers, HeadFfnRatio, HeadRmsNormEps, source.GetTensor);

        var acousticEncoderConfig = new VibeVoiceTokenizerConfig
        {
            Channels = Channels,
            VaeDim = AcousticVaeDim,
            EncoderNFilters = DecoderNFilters,
            EncoderRatios = TokenizerRatios,
            EncoderDepths = EncoderDepths,
            DisableLastNorm = true,
            LayerNormEps = TokenizerLayerNormEps,
        };
        var acousticEncoderWeights = VibeVoiceTokenizerEncoderWeights.Load(acousticEncoderConfig, "model.acoustic_tokenizer.encoder", source.GetTensor);

        var acousticDecoderConfig = new VibeVoiceTokenizerConfig
        {
            Channels = Channels,
            VaeDim = AcousticVaeDim,
            EncoderNFilters = DecoderNFilters,
            EncoderRatios = TokenizerRatios,
            EncoderDepths = DecoderDepths,
            DisableLastNorm = true,
            LayerNormEps = TokenizerLayerNormEps,
        };
        var acousticDecoderWeights = VibeVoiceTokenizerDecoderWeights.Load(acousticDecoderConfig, DecoderNFilters, TokenizerRatios, DecoderDepths,
            "model.acoustic_tokenizer.decoder", source.GetTensor);

        var semanticConfig = new VibeVoiceTokenizerConfig
        {
            Channels = Channels,
            VaeDim = SemanticVaeDim,
            EncoderNFilters = DecoderNFilters,
            EncoderRatios = TokenizerRatios,
            EncoderDepths = EncoderDepths,
            DisableLastNorm = true,
            LayerNormEps = TokenizerLayerNormEps,
        };
        var semanticEncoderWeights = VibeVoiceTokenizerEncoderWeights.Load(semanticConfig, "model.semantic_tokenizer.encoder", source.GetTensor);

        var acousticConnectorWeights = VibeVoiceConnectorWeights.Load("model.acoustic_connector", AcousticVaeDim, HiddenDim, source.GetTensor);
        var semanticConnectorWeights = VibeVoiceConnectorWeights.Load("model.semantic_connector", SemanticVaeDim, HiddenDim, source.GetTensor);

        float speechScalingFactor = source.GetTensor("model.speech_scaling_factor")[0];
        float speechBiasFactor = source.GetTensor("model.speech_bias_factor")[0];

        // Real acoustic-tokenizer encode of the speaker reference waveform -> [dim][frames] mean.
        var speakerMean = VibeVoiceTokenizerEncoder.Encode(acousticEncoderWeights, speakerWaveform, TokenizerLayerNormEps);
        Assert.True(speakerMean[0].Length >= tokenCounts[0], "encoder produced fewer real frames than the prompt builder's own real speech_token_count expects");

        var result = VibeVoiceGenerator.GenerateWithVoiceCloning(
            fwd, negativeFwd, promptTokenIds, speechMask, [speakerMean], tokenCounts,
            textEmbeddingTable, HiddenDim,
            SpeechStartId, SpeechEndId, SpeechDiffusionId, EosId,
            diffusionHeadWeights, acousticDecoderWeights, semanticEncoderWeights,
            acousticConnectorWeights, semanticConnectorWeights,
            speechScalingFactor, speechBiasFactor, FixStd, TokenizerLayerNormEps,
            DdpmNumSteps, inferenceSteps: 4, guidanceScale: 1.5f,
            maxSteps: 6, new Random(31));

        Assert.NotEmpty(result.GeneratedTokens);
        Assert.NotEmpty(result.AudioSamples);
        Assert.All(result.AudioSamples, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(result.AudioSamples, v => v != 0f);
    }
}
