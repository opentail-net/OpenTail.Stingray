namespace OpenTail.Stingray.Tests.Audio;

public sealed class QwenForcedAlignerEmbeddingRowDebugTest : HeavyTestBase
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

    [Fact]
    public void CompareRawEmbeddingRow_Token151705_AgainstReferenceDump()
    {
        string? safetensorsPath = FindRepoFile("models/qwen3-forcedaligner/model.safetensors");
        Assert.SkipUnless(safetensorsPath != null, "safetensors not found");
        const string dumpDir = "C:/Users/Dmitri/AppData/Local/Temp/claude/C--Git-Public/e20d4458-3906-46e3-a5d1-af57aa8bcc0d/scratchpad/fa_dump";
        string embPath = Path.Combine(dumpDir, "prompt_embeddings.bin");
        Assert.SkipUnless(File.Exists(embPath), "dump not found");

        using var loader = SafetensorsLoader.Open(safetensorsPath!);
        var table = loader.ReadF32("thinker.model.embed_tokens.weight");
        int[] shape = loader.GetShape("thinker.model.embed_tokens.weight");
        Console.Error.WriteLine($"[DBG] embed_tokens shape=[{string.Join(",", shape)}]");

        const int hiddenDim = 1024;
        const int tokenId = 151705;
        var rawRow = table.AsSpan(tokenId * hiddenDim, hiddenDim);

        var bytes = File.ReadAllBytes(embPath);
        int steps = bytes.Length / 4 / hiddenDim;
        var refRow = new float[hiddenDim];
        Buffer.BlockCopy(bytes, (steps - 1) * hiddenDim * 4, refRow, 0, hiddenDim * 4);

        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < hiddenDim; i++)
        {
            float a = rawRow[i], b = refRow[i];
            dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            maxDiff = Math.Max(maxDiff, Math.Abs(a - b));
        }
        double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.Error.WriteLine($"[DBG] raw safetensors row {tokenId} vs reference dump last-row: cosine={cosine:F6} maxDiff={maxDiff:F6}");
        Console.Error.WriteLine($"[DBG] raw row[0..5]=[{string.Join(",", rawRow.Slice(0,5).ToArray())}]");
        Console.Error.WriteLine($"[DBG] ref row[0..5]=[{string.Join(",", refRow.AsSpan(0,5).ToArray())}]");

        Assert.True(double.IsFinite(cosine));
    }
}
