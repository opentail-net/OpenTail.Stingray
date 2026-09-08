using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real VibeVoice TTS wav end-to-end for informal
/// listening (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per
/// CLAUDE.md). Not a golden/parity test.</summary>
public sealed class VibeVoiceTtsGenerateWavDebugTest : HeavyTestBase
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
    private const int OutputSampleRate = 24000;

    private const int HeadLayers = 4;
    private const float HeadFfnRatio = 3.0f, HeadRmsNormEps = 1e-5f;
    private const int DdpmNumSteps = 1000;

    private const int SpeechStartId = 151652, SpeechEndId = 151653, SpeechDiffusionId = 151654, EosId = 151643;

    [Fact]
    public void Generate_RealVibeVoiceTtsWav()
    {
        string? path = FindRepoFile("models/_models/vibevoice-tts/VibeVoice-1.5B-GGUF/vibevoice-1.5b-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-1.5b-q8_0.gguf not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        var tokenizer = LoadTokenizerFromPackedGguf(model);
        string script = "Speaker 1: Hello there, this is a real end to end test of speech synthesis.";
        var promptTokenIds = VibeVoiceTtsPromptBuilder.BuildPrompt(text => [.. tokenizer.Encode(text)], script, SpeechStartId);

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
            EncoderDepths = DecoderDepths,
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

        var result = VibeVoiceGenerator.Generate(
            fwd, promptTokenIds, textEmbeddingTable, HiddenDim,
            SpeechStartId, SpeechEndId, SpeechDiffusionId, EosId,
            diffusionHeadWeights, acousticDecoderWeights, semanticEncoderWeights,
            acousticConnectorWeights, semanticConnectorWeights,
            speechScalingFactor, speechBiasFactor, TokenizerLayerNormEps,
            DdpmNumSteps, inferenceSteps: 10, guidanceScale: 1.5f,
            maxSteps: 60, new Random(31), llm.NormWeight, RmsNormEps);

        Assert.True(result.AudioSamples.Length > 0);

        var wavResult = new OpenTail.Stingray.Audio.AudioGenerationResult(result.AudioSamples, OutputSampleRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "vibevoice-tts-real-check.wav");
        wavResult.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {result.AudioSamples.Length} samples, {result.AudioSamples.Length / (double)OutputSampleRate:F2}s");
    }
}
