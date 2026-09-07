using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real, live end-to-end smoke test for VibeVoice TTS: real LLM prefill through the
/// (confirmed directly reusable) VibeVoiceLlmTensorSource/ForwardPass bridge, the real
/// interleaved text/diffusion generation loop (VibeVoiceGenerator), producing real audio through
/// the real diffusion head + acoustic decoder + semantic re-encoder + connectors -- chaining
/// every VibeVoice TTS piece real-weight verified individually this session into one live
/// run.</summary>
public sealed class VibeVoiceGeneratorRealWeightsTests : HeavyTestBase
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

    // Real numbers confirmed from the checkpoint's own config.json / tokenizer.json (2026-09-07,
    // not guessed).
    private const int NumLayers = 28, HiddenDim = 1536, NumHeads = 12, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 8960, VocabSize = 151936;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    private const int DecoderNFilters = 32, AcousticVaeDim = 64, SemanticVaeDim = 128, Channels = 1;
    private static readonly int[] TokenizerRatios = [8, 5, 5, 4, 2, 2];
    // Real reference: decoder_depths falls back to reverse(encoder_depths) when null (see
    // VibeVoiceTokenizerDecoderRealWeightsTests) -- the encoder itself uses the RAW,
    // un-reversed "3-3-3-3-3-3-8" order directly (VibeVoiceTokenizerEncoderWeights.Load never
    // reverses depths, only ratios). Using the decoder's reversed array for the encoder was an
    // earlier bug in this test, caught immediately by a real "missing tensor" error.
    private static readonly int[] EncoderDepths = [3, 3, 3, 3, 3, 3, 8];
    private static readonly int[] DecoderDepths = [8, 3, 3, 3, 3, 3, 3];
    private const float TokenizerLayerNormEps = 1e-5f;

    private const int HeadLayers = 4;
    private const float HeadFfnRatio = 3.0f, HeadRmsNormEps = 1e-5f;
    private const int DdpmNumSteps = 1000;

    private const int SpeechStartId = 151652, SpeechEndId = 151653, SpeechDiffusionId = 151654, EosId = 151643;

    [Fact]
    public void Generate_OnRealCheckpoint_ProducesFiniteNonSilentAudio()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        using var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var textEmbeddingTable = source.GetTensor("model.language_model.embed_tokens.weight");

        var diffusionHeadWeights = VibeVoiceDiffusionHeadWeights.Load(HiddenDim, AcousticVaeDim, HeadLayers, HeadFfnRatio, HeadRmsNormEps, source.GetTensor);

        var acousticConfig = new VibeVoiceTokenizerConfig
        {
            Channels = Channels,
            VaeDim = AcousticVaeDim,
            EncoderNFilters = DecoderNFilters,
            EncoderRatios = TokenizerRatios,
            EncoderDepths = DecoderDepths, // unused by the decoder loader (explicit decoderDepths param below), kept for field completeness
            DisableLastNorm = true,
            LayerNormEps = TokenizerLayerNormEps,
        };
        var acousticDecoderWeights = VibeVoiceTokenizerDecoderWeights.Load(acousticConfig, DecoderNFilters, TokenizerRatios, DecoderDepths,
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
        Assert.NotEqual(0f, speechScalingFactor);

        // A minimal real prompt: a few in-range token ids ending in the real speech_start token
        // (the real full prompt-template port -- "Text input:\n Speaker N: ...\n Speech
        // output:\n" -- is separate, unstarted work; this exercises the real GENERATION LOOP
        // machinery, not the real prompt construction).
        var promptTokenIds = new[] { 1, 100, 200, 300, SpeechStartId };

        var result = VibeVoiceGenerator.Generate(
            fwd, promptTokenIds, textEmbeddingTable, HiddenDim,
            SpeechStartId, SpeechEndId, SpeechDiffusionId, EosId,
            diffusionHeadWeights, acousticDecoderWeights, semanticEncoderWeights,
            acousticConnectorWeights, semanticConnectorWeights,
            speechScalingFactor, speechBiasFactor, TokenizerLayerNormEps,
            DdpmNumSteps, inferenceSteps: 4, guidanceScale: 1.5f,
            maxSteps: 6, new Random(31));

        Assert.NotEmpty(result.GeneratedTokens);
        Assert.NotEmpty(result.AudioSamples);
        Assert.All(result.AudioSamples, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(result.AudioSamples, v => v != 0f);
    }
}
