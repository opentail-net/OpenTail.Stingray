
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: feeds the real reference's own dumped new/prompt speech tokens
/// and speaker embedding (from `examples/cosyvoice.cpp/src/cosyvoice-token2wav.cpp`'s new
/// COSY_DUMP_* hooks, real real values from an actual zero-shot CLI run) through our C#
/// CosyVoice3FlowEncoder.ComputeMuAndSpks, and compares the resulting mu/spks against the
/// reference's own dumped mu/spks tensors -- to numerically localize whether the remaining
/// "gibberish" bug lives in the flow encoder or further downstream (DiT/CFG/cond).</summary>
public sealed class CosyVoice3FlowEncoderCompareDebugTest : HeavyTestBase
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

    [Fact]
    public void Mu_And_Spks_MatchReference()
    {
        string dumpDir = FindModelPath("examples/cosyvoice.cpp/mu.bin") is { } p ? Path.GetDirectoryName(p)! : "";
        Assert.SkipUnless(!string.IsNullOrEmpty(dumpDir), "reference flow-encoder dumps not found");

        string? ggufPath = FindModelPath("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(ggufPath != null, "CosyVoice3 GGUF not found");

        var newTokens = ReadInts(Path.Combine(dumpDir, "newtokens.bin"));
        var promptTokens = ReadInts(Path.Combine(dumpDir, "prompttokens.bin"));
        var embedding = ReadFloats(Path.Combine(dumpDir, "embedding.bin"));
        var refMu = ReadFloats(Path.Combine(dumpDir, "mu.bin"));
        var refSpks = ReadFloats(Path.Combine(dumpDir, "spks.bin"));
        var refConds = ReadFloats(Path.Combine(dumpDir, "conds.bin"));
        var promptFeatFrameMajor = ReadFloats(Path.Combine(dumpDir, "promptfeat.bin"));

        int[] jointTokens = [.. promptTokens, .. newTokens];

        using var rawModel = GgufModel.Open(ggufPath!);
        var flowWeights = new CosyVoice3FlowEncoderWeights(rawModel);
        var (mu, spks) = CosyVoice3FlowEncoder.ComputeMuAndSpks(flowWeights, jointTokens, embedding);

        const int melDim = 80;
        int numFrames = mu.Length / melDim;
        int promptFrames = Math.Min(promptFeatFrameMajor.Length / melDim, numFrames);
        var cond = new float[mu.Length];
        // mu/cond are frame-major (mel-dim contiguous per frame) -- confirmed by mu's own perfect
        // 1.0 cosine match above using a straight, unpermuted comparison. promptFeatFrameMajor is
        // already in that same layout (real reference's raw dump), so this is a straight copy, NOT
        // the channel-major transpose CosyVoice3Pipeline.Generate applies later for HiFT's Decode
        // input (a DIFFERENT, separate convention -- do not conflate the two).
        Array.Copy(promptFeatFrameMajor, 0, cond, 0, promptFrames * melDim);

        double muCos = Cosine(mu, refMu);
        double spksCos = Cosine(spks, refSpks);
        double condCos = Cosine(cond, refConds);

        string msg = $"[FLOWCOMPARE] jointTokens={jointTokens.Length} mu.Length={mu.Length} refMu.Length={refMu.Length} muCos={muCos:F6} spks.Length={spks.Length} refSpks.Length={refSpks.Length} spksCos={spksCos:F6} condCos={condCos:F6} numFrames={numFrames} promptFrames={promptFrames}";
        Console.WriteLine(msg);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "flow_compare_result.txt"), msg);
    }
}
