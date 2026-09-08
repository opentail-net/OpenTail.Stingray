using OpenTail.Stingray.Audio.NeuTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt end-to-end smoke test for NeuTTS's AR speech-token generation loop
/// (`NeuTtsGenerator`/`NeuTtsPromptBuilder`, both new this session) against the real downloaded
/// checkpoint. Produces real, in-range codec codes -- NOT yet decoded to audio (the FSQ codec is a
/// separate, not-yet-started task per this session's scoping).
/// </summary>
public sealed class NeuTtsGeneratorRealWeightsTests : HeavyTestBase
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
    public void GenerateSpeechCodes_OnRealCheckpoint_ProducesInRangeCodes()
    {
        string? path = FindRepoFile("models/_models/neutts/NeuTTS-2E-GGUF/neutts-2e-orig.gguf");
        Assert.SkipUnless(path != null, "neutts-2e-orig.gguf not found");

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
        Assert.NotEmpty(emilyCodesInt);

        var prompt = NeuTtsPromptBuilder.Build(
            tokenizer, tokenizerSource, addedTokens,
            referenceText: "This is a reference recording.",
            inputText: "Hello there.",
            speakerSpeechCodes: emilyCodesInt);

        using var llm = new NeuTtsBackboneTensorSource(source);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var codes = NeuTtsGenerator.GenerateSpeechCodes(fwd, prompt, new NeuTtsGenerationOptions { MaxNewTokens = 32 });
        sw.Stop();

        Assert.All(codes, c => Assert.InRange(c, 0, 65535));
        Console.WriteLine($"[NeuTtsGenerator] {sw.ElapsedMilliseconds}ms, promptTokens={prompt.TokenIds.Length}, generatedCodes={codes.Length}");
    }
}
