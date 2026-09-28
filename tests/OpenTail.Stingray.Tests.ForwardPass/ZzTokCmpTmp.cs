namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): our tokenisation of a file vs a reference id list. ZZ_TOK=1, ZZ_MODEL, ZZ_FILE, ZZ_REF.
public sealed class ZzTokCmpTmp
{
    [Fact]
    public void Compare()
    {
        if (Environment.GetEnvironmentVariable("ZZ_TOK") != "1") return;
        using var model = GgufModel.Open(Environment.GetEnvironmentVariable("ZZ_MODEL")!);
        var tok = GgufTokenizer.FromGgufModel(model);
        var ours = tok.Encode(File.ReadAllText(Environment.GetEnvironmentVariable("ZZ_FILE")!)).ToList();
        var refIds = File.ReadAllText(Environment.GetEnvironmentVariable("ZZ_REF")!).Split(',').Select(int.Parse).ToList();
        Console.WriteLine($"ours {ours.Count} ref {refIds.Count} pre={Convert.ToString(model.Metadata.GetValueOrDefault("tokenizer.ggml.pre"))}");
        int i = 0;
        while (i < Math.Min(ours.Count, refIds.Count) && ours[i] == refIds[i]) i++;
        Console.WriteLine($"first mismatch at {i}");
        if (i < Math.Min(ours.Count, refIds.Count))
        {
            Console.WriteLine("ours: " + string.Join(" | ", ours.Skip(Math.Max(0, i - 3)).Take(8).Select(t => $"{t}:{tok.Decode([t])}")));
            Console.WriteLine("ref:  " + string.Join(" | ", refIds.Skip(Math.Max(0, i - 3)).Take(8).Select(t => $"{t}:{tok.Decode([t])}")));
        }
    }
}
