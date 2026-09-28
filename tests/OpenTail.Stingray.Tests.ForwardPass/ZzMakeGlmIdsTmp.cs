namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): tokenizes the first N bytes of a text file with the real model's own
// tokenizer and writes the first K token ids (comma-separated) to ZZ_OUT, for feeding into
// ZzLayerDumpTmp / llama-eval-callback as a matched prompt. ZZ_MKIDS=1, ZZ_MODEL, ZZ_FILE,
// ZZ_BYTES, ZZ_COUNT, ZZ_OUT.
public sealed class ZzMakeGlmIdsTmp
{
    [Fact]
    public void MakeIds()
    {
        if (Environment.GetEnvironmentVariable("ZZ_MKIDS") != "1") return;
        using var model = GgufModel.Open(Environment.GetEnvironmentVariable("ZZ_MODEL")!);
        var tok = GgufTokenizer.FromGgufModel(model);

        int nBytes = int.Parse(Environment.GetEnvironmentVariable("ZZ_BYTES") ?? "1400");
        int nCount = int.Parse(Environment.GetEnvironmentVariable("ZZ_COUNT") ?? "326");

        var bytes = File.ReadAllBytes(Environment.GetEnvironmentVariable("ZZ_FILE")!);
        var text = System.Text.Encoding.UTF8.GetString(bytes, 0, Math.Min(nBytes, bytes.Length));

        var ids = tok.Encode(text).ToList();
        Console.WriteLine($"tokenized {text.Length} chars -> {ids.Count} tokens (addBos={tok.AddBosToken}, first={ids[0]})");
        var take = ids.Take(nCount).ToList();
        Console.WriteLine($"writing {take.Count} ids");

        File.WriteAllText(Environment.GetEnvironmentVariable("ZZ_OUT")!, string.Join(",", take));
    }
}
