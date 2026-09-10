using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for VibeVoice-ASR for PerformanceLeague.md backfill,
/// against the standard 14.1s b.wav reference used elsewhere in this doc. Pipeline setup mirrors
/// VibeVoiceAsrRealSpeechRealWeightsTests (minus its debug taps). No C++ reference attempted this
/// pass. Not part of the permanent suite, delete after use.
/// </summary>
public sealed class VibeVoiceAsrPerfBaselineDebugTest : HeavyTestBase
{
    private const int NumLayers = 28, HiddenDim = 3584, NumHeads = 28, NumKvHeads = 4, HeadDim = 128;
    private const int FfDim = 18944, VocabSize = 152064;
    private const float RopeTheta = 1000000f, RmsNormEps = 1e-6f;
    private const int RealSampleRate = 24000;
    private const int Runs = 3;

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

    [Fact]
    public void Bench_Transcribe_BWav()
    {
        string? path = FindRepoFile("models/_models/vibevoice_asr/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf");
        Assert.SkipUnless(path != null, "vibevoice-asr-q8_0.gguf not found");
        string? wavPath = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(wavPath != null, "reference b.wav not found");

        var (rawSamples, rawRate, rawChannels) = WavReader.ReadWav(wavPath!);
        var mono = rawChannels == 1 ? rawSamples : rawSamples; // b.wav is mono per other tests' usage
        var resampled = rawRate == RealSampleRate ? mono : AudioResampler.Resample(mono, rawRate, RealSampleRate, channels: 1, ResampleQuality.BestQuality);
        var waveform = VibeVoiceAudioNormalizer.Normalize(resampled);
        double audioSeconds = waveform.Length / (double)RealSampleRate;

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var tokenizer = LoadTokenizer(model);

        var acousticEncoder = VibeVoiceTokenizerEncoderWeights.Load(AcousticConfig(), "model.acoustic_tokenizer.encoder", source.GetTensor);
        var semanticEncoder = VibeVoiceTokenizerEncoderWeights.Load(SemanticConfig(), "model.semantic_tokenizer.encoder", source.GetTensor);
        var acousticConnector = VibeVoiceConnectorWeights.Load("model.acoustic_connector", inputDim: 64, hiddenSize: HiddenDim, source.GetTensor);
        var semanticConnector = VibeVoiceConnectorWeights.Load("model.semantic_connector", inputDim: 128, hiddenSize: HiddenDim, source.GetTensor);
        var llm = new VibeVoiceLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps);

        string RunOnce()
        {
            var speechEmbeddingsChannelMajor = VibeVoiceSpeechFeatures.Extract(
                acousticEncoder, AcousticConfig(), acousticConnector,
                semanticEncoder, SemanticConfig(), semanticConnector,
                waveform, new Random(7));
            int speechFrames = speechEmbeddingsChannelMajor[0].Length;

            var prompt = tokenizer.BuildPrompt(audioSeconds, speechFrames);

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

            return VibeVoiceAsrGenerator.GenerateTranscript(
                fwd, tokenizer, remappedPrompt, new VibeVoiceAsrGenerationOptions { MaxNewTokens = 96 });
        }

        var warm = RunOnce();

        double[] elapsedSec = new double[Runs];
        string lastText = warm;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            lastText = RunOnce();
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
        }

        double meanSec = elapsedSec.Average();
        double rtf = meanSec / audioSeconds;
        string msg = $"[VibeVoice-ASR] audio={audioSeconds:F2}s text=\"{lastText}\"\n" +
                     $"[VibeVoice-ASR] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
