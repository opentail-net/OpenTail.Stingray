using System;
using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

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

        double cos = Cosine(ourLogits, refLogits);
        int ourArgmax = ArgMax(ourLogits);
        int refArgmax = ArgMax(refLogits);

        string msg = $"[LLMLOGITS] our.Length={ourLogits.Length} ref.Length={refLogits.Length} cosine={cos:F6} " +
                     $"ourArgmax={ourArgmax} (val={ourLogits[ourArgmax]:F4}) refArgmax={refArgmax} (val={refLogits[refArgmax]:F4}) " +
                     $"ourAtRefArgmax={ourLogits[refArgmax]:F4} refAtOurArgmax={refLogits[ourArgmax]:F4}";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "llmlogits_compare_result.txt"), msg);
    }
}
