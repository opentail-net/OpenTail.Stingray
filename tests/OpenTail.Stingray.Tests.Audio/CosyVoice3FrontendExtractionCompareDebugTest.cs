
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: compares OUR OWN frontend extraction (CamPlusSpeakerEncoder,
/// CosyVoiceSpeechTokenizer, CosyVoiceMelExtractor) run on the real reference wav
/// (`cosyvoice3-REFERENCE-nofix-compare.wav`) against the C++ reference's own dumped
/// embedding/prompt-tokens/prompt-mel for the SAME wav -- to find which extractor is the
/// remaining source of the "gibberish" bug, now that the flow encoder/DiT/HiFT math is all
/// proven correct (see CosyVoice3FullChainFromRefInputsDebugTest).</summary>
public sealed class CosyVoice3FrontendExtractionCompareDebugTest : HeavyTestBase
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

    // The reference CLI's --prompt-audio-16k/-24k files are raw 32-bit FLOAT PCM (confirmed via
    // tools/cli/cosyvoice-cli.cpp: "Reference audio in 16kHz PCM float format" + its own
    // sizeof(float)-alignment check), NOT 16-bit int PCM.
    private static float[] ReadRawPcmFloat32Mono(string path) => ReadFloats(path);

    [Fact]
    public void OurFrontendExtraction_ComparedToReference()
    {
        string dumpDir = FindModelPath("examples/cosyvoice.cpp/mu.bin") is { } p ? Path.GetDirectoryName(p)! : "";
        Assert.SkipUnless(!string.IsNullOrEmpty(dumpDir), "reference dumps not found");

        // The REAL prompt audio the reference CLI actually used -- raw headerless PCM16, NOT the
        // synthesized output wav (a much longer, different signal; using it here would silently
        // compare against the wrong audio).
        string? prompt16kPath = FindModelPath("examples/cosyvoice.cpp/prompt16k.pcm");
        string? prompt24kPath = FindModelPath("examples/cosyvoice.cpp/prompt24k.pcm");
        string? campplusPath = FindModelPath("models/campplus.onnx");
        string? speechTokenizerPath = FindModelPath("models/cosyvoice_speech_tokenizer_v2.onnx");
        Assert.SkipUnless(prompt16kPath != null && prompt24kPath != null && campplusPath != null && speechTokenizerPath != null,
            "reference prompt PCM or ONNX frontends not found");

        var refEmbedding = ReadFloats(Path.Combine(dumpDir, "embedding.bin"));
        var refPromptTokens = ReadInts(Path.Combine(dumpDir, "prompttokens.bin"));
        var refPromptFeat = ReadFloats(Path.Combine(dumpDir, "promptfeat.bin"));

        var samples24k = ReadRawPcmFloat32Mono(prompt24kPath!);
        var ourMel = CosyVoiceMelExtractor.Shared.ExtractMel(samples24k);
        double melCos = Cosine(ourMel, refPromptFeat);

        var samples16k = ReadRawPcmFloat32Mono(prompt16kPath!);
        var ourEmbedding = CamPlusSpeakerEncoder.Extract(campplusPath!, samples16k) ?? [];
        double embCos = Cosine(ourEmbedding, refEmbedding);

        var ourPromptTokens = CosyVoiceSpeechTokenizer.Extract(speechTokenizerPath!, samples16k) ?? [];

        int tokMatch = 0;
        int tokMin = Math.Min(ourPromptTokens.Length, refPromptTokens.Length);
        for (int i = 0; i < tokMin; i++) if (ourPromptTokens[i] == refPromptTokens[i]) tokMatch++;

        string msg = $"[FRONTEND] melCos={melCos:F6} (our.Length={ourMel.Length} ref.Length={refPromptFeat.Length}) " +
                     $"embCos={embCos:F6} (our.Length={ourEmbedding?.Length ?? -1} ref.Length={refEmbedding.Length}) " +
                     $"tokens: our.Length={ourPromptTokens.Length} ref.Length={refPromptTokens.Length} exactMatch={tokMatch}/{tokMin} " +
                     $"ourFirst10=[{string.Join(",", ourPromptTokens[..Math.Min(10, ourPromptTokens.Length)])}] refFirst10=[{string.Join(",", refPromptTokens[..Math.Min(10, refPromptTokens.Length)])}]";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "frontend_compare_result.txt"), msg);
    }
}
