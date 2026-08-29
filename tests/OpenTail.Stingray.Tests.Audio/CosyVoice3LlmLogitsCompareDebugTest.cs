
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: golden-verifies CosyVoice3Llm's forward pass by comparing our
/// ForwardPass.Prefill's raw first-step logits against the real C++ reference's own dumped
/// `llm_logits.bin` (`cosyvoice-llm.cpp`'s new COSY_DUMP_LLM_LOGITS_PATH hook) for the IDENTICAL
/// composed token sequence (sos+prefix+endofprompt+promptText+text+task+promptSpeechTokens, real
/// prompt tokens/text from an actual reference CLI run). This is the numeric oracle the project's
/// own methodology calls for before trusting/fixing the LLM stage further.</summary>
public sealed class CosyVoice3LlmLogitsCompareDebugTest : HeavyTestBase
{
    private static string? FindModelPath(string relPath)
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

    private static int[] ReadInts(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var ints = new int[bytes.Length / sizeof(int)];
        Buffer.BlockCopy(bytes, 0, ints, 0, bytes.Length);
        return ints;
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private static double Cosine(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    private static int ArgMax(float[] a)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int i = 0; i < a.Length; i++) if (a[i] > bv) { bv = a[i]; best = i; }
        return best;
    }

    [Fact]
    public void FirstStepLogits_MatchReference()
    {
        string dumpDir = FindModelPath("examples/cosyvoice.cpp/llm_logits.bin") is { } p ? Path.GetDirectoryName(p)! : "";
        Assert.SkipUnless(!string.IsNullOrEmpty(dumpDir), "reference LLM logits dump not found");
        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(ggufPath != null, "CosyVoice3 GGUF not found");

        var refLogits = ReadFloats(Path.Combine(dumpDir, "llm_logits.bin"));
        var promptTokens = ReadInts(Path.Combine(dumpDir, "prompttokens.bin"));

        using var rawModel = GgufModel.Open(ggufPath!);
        var llmSource = new CosyVoice3LlmTensorSource(rawModel);
        llmSource.EnableSpeechGenerationMode();

        var ourLogits = CosyVoice3Llm.GetFirstStepLogitsForTest(rawModel, llmSource, "This is a test of voice synthesis.",
            promptText: "this is a test of voice cloning", promptSpeechTokens: promptTokens);

        // Also run token-by-token sequential Forward to see if Prefill vs Decode diverged
        var tokenizer = CosyVoice3Llm.BuildTokenizer(rawModel);
        int sosTokenId = rawModel.GetMetadata("sos_token_id", 0);
        int taskTokenId = rawModel.GetMetadata("task_token_id", 0);
        string instructionPrefix = rawModel.GetMetadata("cosyvoice.instruction_prefix", "You are a helpful assistant.");
        var prefixTokens = tokenizer.Encode(instructionPrefix);
        var endOfPromptTokens = tokenizer.Encode("<|endofprompt|>");
        var promptTextTokens = tokenizer.Encode("this is a test of voice cloning");
        var textTokens = tokenizer.Encode("This is a test of voice synthesis.");

        var prefillIds = new List<int> { llmSource.SpeechTokenIdOffset + sosTokenId };
        prefillIds.AddRange(prefixTokens);
        prefillIds.AddRange(endOfPromptTokens);
        prefillIds.AddRange(promptTextTokens);
        prefillIds.AddRange(textTokens);
        prefillIds.Add(llmSource.SpeechTokenIdOffset + taskTokenId);
        foreach (int t in promptTokens) prefillIds.Add(llmSource.SpeechTokenIdOffset + t);

        Console.WriteLine($"[TOKENS] total={prefillIds.Count} sos={sosTokenId} prefixCount={prefixTokens.Count} endofpromptCount={endOfPromptTokens.Count} promptTextCount={promptTextTokens.Count} textCount={textTokens.Count} taskToken={taskTokenId} promptSpeechCount={promptTokens.Length}");
        Console.WriteLine($"[TOKENS] prefixTokens: [{string.Join(",", prefixTokens)}]");
        Console.WriteLine($"[TOKENS] endofprompt: [{string.Join(",", endOfPromptTokens)}]");
        Console.WriteLine($"[TOKENS] promptTextTokens: [{string.Join(",", promptTextTokens)}]");
        Console.WriteLine($"[TOKENS] textTokens: [{string.Join(",", textTokens)}]");
        Console.WriteLine($"[TOKENS] first 10 promptTokens: [{string.Join(",", promptTokens.Take(10))}] last 5 promptTokens: [{string.Join(",", promptTokens.TakeLast(5))}]");

        var hp = ModelHyperparams.FromGgufMetadata(llmSource.Metadata, llmSource);
        Console.WriteLine($"[HYPERPARAMS] hasAttnBias={hp.HasAttnBias} hasAttnOutputBias={hp.HasAttnOutputBias} hasNormBias={hp.HasNormBias} hasFfnBias={hp.HasFfnBias} isNeoxRope={hp.IsNeoxRope} ropeTheta={hp.RopeTheta}");
        using var backend = new CpuBackend();
        using var seqFwd = new ForwardPass(llmSource, backend, hp);
        float[] seqLogits = [];
        for (int i = 0; i < prefillIds.Count; i++)
        {
            seqLogits = seqFwd.Forward(prefillIds[i], i).ToArray();
        }

        double cosPrefillVsSeq = Cosine(ourLogits, seqLogits);
        double cosRefVsSeq = Cosine(seqLogits, refLogits);
        int seqArgmax = ArgMax(seqLogits);

        double cos = Cosine(ourLogits, refLogits);
        int ourArgmax = ArgMax(ourLogits);
        int refArgmax = ArgMax(refLogits);

        string msg = $"[TOKENS] total={prefillIds.Count} sos={sosTokenId} prefixCount={prefixTokens.Count} endofpromptCount={endOfPromptTokens.Count} promptTextCount={promptTextTokens.Count} textCount={textTokens.Count} taskToken={taskTokenId} promptSpeechCount={promptTokens.Length}\n" +
                     $"[TOKENS] prefixTokens: [{string.Join(",", prefixTokens)}]\n" +
                     $"[TOKENS] endofprompt: [{string.Join(",", endOfPromptTokens)}]\n" +
                     $"[TOKENS] promptTextTokens: [{string.Join(",", promptTextTokens)}]\n" +
                     $"[TOKENS] textTokens: [{string.Join(",", textTokens)}]\n" +
                     $"[LLMLOGITS] our.Length={ourLogits.Length} ref.Length={refLogits.Length} cosine={cos:F6} " +
                     $"ourArgmax={ourArgmax} (val={ourLogits[ourArgmax]:F4}) refArgmax={refArgmax} (val={refLogits[refArgmax]:F4}) " +
                     $"ourAtRefArgmax={ourLogits[refArgmax]:F4} refAtOurArgmax={refLogits[ourArgmax]:F4}\n" +
                     $"[SEQ COMPARE] cos(Prefill, Seq)={cosPrefillVsSeq:F6} cos(Seq, Ref)={cosRefVsSeq:F6} seqArgmax={seqArgmax} (val={seqLogits[seqArgmax]:F4})";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "llmlogits_compare_result.txt"), msg);
        Assert.True(cos > 0.99, $"Expected cosine > 0.99 against reference logits, got {cos:F6}");
        Assert.Equal(refArgmax, ourArgmax);
    }
}
