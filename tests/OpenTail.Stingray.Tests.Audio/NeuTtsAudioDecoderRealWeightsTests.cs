using OpenTail.Stingray.Audio.NeuTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, full end-to-end smoke test for NeuTTS: text -&gt; real AR speech-token generation -&gt;
/// real FSQ acoustic-decoder codec decode -&gt; a real, finite 24kHz mono waveform. Closes the last
/// missing piece scoped in this session's NeuTTS handoff notes (docs/audio-review-progress.md).
/// Not yet a numeric golden-parity check against a captured reference waveform.
/// </summary>
public sealed class NeuTtsAudioDecoderRealWeightsTests : HeavyTestBase
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

    private static (TokenizerSource Source, byte[] TokenizerJsonBytes) LoadTokenizerSource(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("no embedded_files metadata");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        string dir = Path.Combine(Path.GetTempPath(), "stingray-neutts-tokenizer");
        Directory.CreateDirectory(dir);
        string[] wanted = ["tokenizer.json", "tokenizer_config.json"];
        byte[]? tokenizerJsonBytes = null;
        for (int i = 0; i < names.Length; i++)
        {
            string name = (string)names[i];
            if (Array.IndexOf(wanted, name) < 0) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            var slice = bytes[(int)start..(int)end];
            File.WriteAllBytes(Path.Combine(dir, name), slice);
            if (name == "tokenizer.json") tokenizerJsonBytes = slice;
        }

        var result = HuggingFaceTokenizerSource.Load(dir);
        if (result.Source is null)
            throw new InvalidOperationException($"NeuTTS tokenizer load failed: {string.Join("; ", result.Rejections)}");
        return (result.Source, tokenizerJsonBytes!);
    }

    [Fact]
    public void FullPipeline_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/neutts/NeuTTS-2E-GGUF/neutts-2e-orig.gguf");
        Assert.SkipUnless(path != null, "neutts-2e-orig.gguf not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var model = GgufModel.Open(path!);
        var (tokenizerSource, tokenizerJsonBytes) = LoadTokenizerSource(model);
        var tokenizer = GgufTokenizer.FromSource(tokenizerSource);
        var addedTokens = NeuTtsPromptBuilder.LoadAddedTokenIds(tokenizerJsonBytes);

        var source = new RvcPackedTensorSource(model);
        var emilyInfo = source.GetRawInfo("speaker_prompts/emily");
        Assert.Equal(DType.Int32, emilyInfo.DType);
        int[] emilyCodesInt;
        unsafe
        {
            byte* raw = source.GetRawDataPtr("speaker_prompts/emily");
            emilyCodesInt = new ReadOnlySpan<int>(raw, checked((int)emilyInfo.ElementCount)).ToArray();
        }

        string? emilyRefText = null;
        if (model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var nObj) && nObj is object[] fNames)
        {
            var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
            var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
            var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
            for (int i = 0; i < fNames.Length; i++)
            {
                string fn = (string)fNames[i];
                if (fn.EndsWith("emily.txt", StringComparison.OrdinalIgnoreCase) || fn.EndsWith("emily", StringComparison.OrdinalIgnoreCase))
                {
                    long start = Convert.ToInt64(offsets[i]);
                    long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
                    emilyRefText = System.Text.Encoding.UTF8.GetString(bytes[(int)start..(int)end]).Trim();
                    break;
                }
            }
        }
        Console.WriteLine($"[NeuTts] Emily ref text: '{emilyRefText}'");

        var prompt = NeuTtsPromptBuilder.Build(
            tokenizer, tokenizerSource, addedTokens,
            referenceText: emilyRefText ?? "This is a reference recording.",
            inputText: "Hello there, this is a full test of the NeuTTS speech synthesis system.",
            speakerSpeechCodes: emilyCodesInt);

        using var llm = new NeuTtsBackboneTensorSource(source);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var codes = NeuTtsGenerator.GenerateSpeechCodes(fwd, prompt, new NeuTtsGenerationOptions
        {
            MaxNewTokens = 300,
            Sampling = new SamplingParams { Temperature = 0.8f, TopK = 50, TopP = 0.95f },
            Rng = new Random(42)
        });
        Assert.NotEmpty(codes);

        var codecWeights = NeuTtsAudioDecoderWeights.Load(source.GetTensor);
        var waveform = NeuTtsAudioDecoder.Decode(codecWeights, codes);
        sw.Stop();

        Assert.NotEmpty(waveform);
        Assert.All(waveform, s => Assert.True(float.IsFinite(s)));

        if (repoRoot != null)
        {
            string outPath = Path.Combine(repoRoot, "audio-samples", "neutts-real-check.wav");
            var result = new AudioGenerationResult(waveform, NeuTtsAudioDecoderWeights.SampleRate);
            result.SaveWav(outPath);
            Console.WriteLine($"Wrote {outPath}, {waveform.Length} samples, {waveform.Length / (double)NeuTtsAudioDecoderWeights.SampleRate:F2}s");
        }
        Console.WriteLine($"[NeuTts] {sw.ElapsedMilliseconds}ms, {codes.Length} codes -> {waveform.Length} samples");
    }
}
