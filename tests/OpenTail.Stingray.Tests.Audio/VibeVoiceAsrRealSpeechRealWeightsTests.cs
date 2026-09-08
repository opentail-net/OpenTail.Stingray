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
        var resampled = rawRate == RealSampleRate ? rawSamples : AudioResampler.Resample(rawSamples, rawRate, RealSampleRate, channels: 1, ResampleQuality.BestQuality);
        // Real, previously-missing step found via this session's per-op bisection: the reference's
        // `VibeVoiceASRFrontend::normalize` (frontend.cpp) RMS-normalizes to a target dBFS AFTER
        // resampling and BEFORE the tokenizer encoders run -- our pipeline never applied it.
        var waveform = VibeVoiceAudioNormalizer.Normalize(resampled);

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

        {
            // DEBUG: per-stage taps inside the semantic encoder itself, matching the reference's
            // own `STINGRAY_ASR_TRACE`-gated `stageN_downsample`/`stageN_blockM` dump added to
            // `speech_tokenizer.cpp`'s `build_encoder` for this bisection -- localizes the
            // divergence to a specific stage/block instead of only the encoder's final output.
            // Replicates the reference's `sample_point_indices(count, target=40)` (trace.cpp)
            // exactly so the flat indices printed here line up byte-for-byte with the
            // `STINGRAY_ASR_TRACE`-gated dump added to `speech_tokenizer.cpp`'s `build_encoder`.
            static long[] SamplePointIndices(long count, long target = 40)
            {
                if (count == 0) return [];
                if (count <= target)
                {
                    var all = new long[count];
                    for (long i = 0; i < count; i++) all[i] = i;
                    return all;
                }
                const long firstCount = 14, middleCount = 12, lastCount = 14;
                long firstEnd = count / 3, middleEnd = count * 2 / 3;
                var points = new List<long>();
                void AppendRange(long begin, long end, long samples)
                {
                    if (begin >= end || samples == 0) return;
                    long span = end - begin;
                    if (span <= samples)
                    {
                        for (long i = begin; i < end; i++)
                            if (points.Count == 0 || points[^1] != i) points.Add(i);
                        return;
                    }
                    for (long i = 0; i < samples; i++)
                    {
                        double pos = samples == 1 ? 0.0 : (double)i / (samples - 1);
                        long offset = (long)(pos * (span - 1));
                        long index = begin + offset;
                        if (points.Count == 0 || points[^1] != index) points.Add(index);
                    }
                }
                AppendRange(0, firstEnd, firstCount);
                AppendRange(firstEnd, middleEnd, middleCount);
                AppendRange(middleEnd, count, lastCount);
                if (points.Count == 0 || points[0] != 0) points.Insert(0, 0);
                return points.ToArray();
            }
            void Tap(string name, float[][] h)
            {
                int channels = h.Length, frames = h[0].Length;
                long total = (long)channels * frames;
                var sb = new System.Text.StringBuilder($"[DEBUG] encoder.{name} shape=[{frames},{channels}] samples=[");
                foreach (long idx in SamplePointIndices(total))
                {
                    int c = (int)(idx / frames), f = (int)(idx % frames);
                    sb.Append($"{idx}:{h[c][f]:G6},");
                }
                sb.Append(']');
                Console.WriteLine(sb.ToString());
            }
            // DEBUG: isolate the deterministic semantic-only branch (no RNG) for a clean
            // comparison against the reference's own trace.
            var semanticLatent = VibeVoiceTokenizerEncoder.Encode(semanticEncoder, waveform, SemanticConfig().LayerNormEps, Tap);
            var semanticProjected = VibeVoiceConnector.Project(semanticConnector, semanticLatent);
            var flat = new float[speechFrames * HiddenDim];
            for (int f = 0; f < speechFrames; f++)
                for (int c = 0; c < HiddenDim; c++)
                    flat[f * HiddenDim + c] = semanticProjected[c][f];
            int[] idx = [0, 2481, 4962, 7443, 9924, 12405, 14886, 17368, 19849, 22330, 24811, 27292, 29773, 32255, 32256, 35188, 38120, 41052, 43985, 46917, 49849, 52781, 55714, 58646, 61578, 64511, 64512, 66993, 69474, 71955, 74436, 76917, 79398, 81880, 84361, 86842, 89323, 91804, 94285, 96767];
            var sb = new System.Text.StringBuilder("[DEBUG] semantic.values samples=[");
            foreach (int i in idx) sb.Append($"{i}:{flat[i]:G6},");
            sb.Append(']');
            Console.WriteLine(sb.ToString());
        }

        {
            var sb = new System.Text.StringBuilder("[DEBUG] semantic_tokenizer.raw_waveform samples=[");
            int[] idx = [0, 2156, 4313, 6470, 8627, 10784, 12941, 15097, 17254, 19411, 21568, 23725, 25882, 28039, 28040, 30589, 33138, 35686, 38236, 40785, 43333, 45883, 48432, 50981, 53530, 56079, 56080, 58236, 60393, 62550, 64707, 66864, 69021, 71177, 73334, 75491, 77648, 79805, 81962, 84119];
            foreach (int i in idx) sb.Append($"{i}:{waveform[i]:G6},");
            sb.Append(']');
            Console.WriteLine(sb.ToString());
        }
        double audioSeconds = waveform.Length / (double)RealSampleRate;
        var prompt = tokenizer.BuildPrompt(audioSeconds, speechFrames);
        Console.WriteLine($"[DEBUG] speechFrames={speechFrames} promptTokens={prompt.InputIds.Length} rawSamples={rawSamples.Length} rawRate={rawRate} resampledLen={waveform.Length}");
        {
            var flat = new float[speechFrames * HiddenDim];
            for (int f = 0; f < speechFrames; f++)
                for (int c = 0; c < HiddenDim; c++)
                    flat[f * HiddenDim + c] = speechEmbeddingsChannelMajor[c][f];
            int[] idx = [0, 2481, 4962, 7443, 9924, 12405, 14886, 17368, 19849, 22330, 24811, 27292, 29773, 32255, 32256, 35188, 38120, 41052, 43985, 46917, 49849, 52781, 55714, 58646, 61578, 64511, 64512, 66993, 69474, 71955, 74436, 76917, 79398, 81880, 84361, 86842, 89323, 91804, 94285, 96767];
            var sb = new System.Text.StringBuilder("[DEBUG] speech.values samples=[");
            foreach (int i in idx) sb.Append($"{i}:{flat[i]:G6},");
            sb.Append(']');
            Console.WriteLine(sb.ToString());
        }

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
