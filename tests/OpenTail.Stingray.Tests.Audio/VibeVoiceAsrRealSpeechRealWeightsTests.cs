using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end VibeVoice ASR transcription on REAL speech audio (a real LibriSpeech clip, the
/// same corpus used to golden-verify Voxtral Realtime/Nemotron 3.5 ASR earlier this session) --
/// distinct from <see cref="VibeVoiceAsrGeneratorRealWeightsTests"/>'s synthetic-noise smoke test,
/// which only proved the pipeline doesn't crash. This is a real listening/transcription-quality
/// check, not yet a byte-for-byte golden-parity comparison against a captured reference
/// transcription (no independent oracle run captured for this exact clip yet).
/// </summary>
public sealed class VibeVoiceAsrRealSpeechRealWeightsTests : HeavyTestBase
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

    private static VibeVoiceAsrTextTokenizer LoadTokenizer(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("VibeVoice ASR packed GGUF has no embedded_files metadata.");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string dir = Path.Combine(Path.GetTempPath(), "stingray-vibevoice-asr-tokenizer");
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
            throw new InvalidOperationException($"VibeVoice ASR tokenizer load failed: {string.Join("; ", result.Rejections)}");
        return new VibeVoiceAsrTextTokenizer(result.Source);
    }

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

    private const int NumLayers = 28, HiddenDim = 3584, NumHeads = 28, NumKvHeads = 4, HeadDim = 128;
    private const int FfDim = 18944, VocabSize = 152064;
    private const float RopeTheta = 1000000f, RmsNormEps = 1e-6f;
    private const int RealSampleRate = 24000; // real config.audio_processor.sample_rate default

    [Fact]
    public void Generate_OnRealLibriSpeechClip_ProducesTranscript()
    {
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");
        string? wavPath = FindRepoFile("examples/audio.cpp/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
        Assert.SkipUnless(wavPath != null, "librispeech reference clip not found");

        var (rawSamples, rawRate, rawChannels) = WavReader.ReadWav(wavPath!);
        Assert.Equal(1, rawChannels);
        var waveform = rawRate == RealSampleRate ? rawSamples : AudioResampler.Resample(rawSamples, rawRate, RealSampleRate);

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var tokenizer = LoadTokenizer(model);

        var acousticEncoder = VibeVoiceTokenizerEncoderWeights.Load(AcousticConfig(), "model.acoustic_tokenizer.encoder", source.GetTensor);
        var semanticEncoder = VibeVoiceTokenizerEncoderWeights.Load(SemanticConfig(), "model.semantic_tokenizer.encoder", source.GetTensor);
        var acousticConnector = VibeVoiceConnectorWeights.Load("model.acoustic_connector", inputDim: 64, hiddenSize: HiddenDim, source.GetTensor);
        var semanticConnector = VibeVoiceConnectorWeights.Load("model.semantic_connector", inputDim: 128, hiddenSize: HiddenDim, source.GetTensor);

        var speechEmbeddingsChannelMajor = VibeVoiceSpeechFeatures.Extract(
            acousticEncoder, AcousticConfig(), acousticConnector,
            semanticEncoder, SemanticConfig(), semanticConnector,
            waveform, new Random(7));
        int speechFrames = speechEmbeddingsChannelMajor[0].Length;

        double audioSeconds = waveform.Length / (double)RealSampleRate;
        var prompt = tokenizer.BuildPrompt(audioSeconds, speechFrames);

        var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);

        var speechEmbeddingsFrameMajor = new float[speechFrames * HiddenDim];
        for (int f = 0; f < speechFrames; f++)
            for (int c = 0; c < HiddenDim; c++)
                speechEmbeddingsFrameMajor[f * HiddenDim + c] = speechEmbeddingsChannelMajor[c][f];
        llm.EnableSpeechConditioning(speechEmbeddingsFrameMajor, speechFrames);

        var inputIds = (int[])prompt.InputIds.Clone();
        for (int slot = 0; slot < prompt.SpeechPositions.Length; slot++)
            inputIds[prompt.SpeechPositions[slot]] = llm.SpeechTokenIdOffset + slot;
        var remappedPrompt = new VibeVoiceAsrPrompt(inputIds, prompt.SpeechPositions);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        string transcript = VibeVoiceAsrGenerator.GenerateTranscript(
            fwd, tokenizer, remappedPrompt, new VibeVoiceAsrGenerationOptions { MaxNewTokens = 96 });

        Assert.NotNull(transcript);
        Console.WriteLine($"[VibeVoiceAsr] real speech ({audioSeconds:F2}s) transcript='{transcript}'");
    }
}
