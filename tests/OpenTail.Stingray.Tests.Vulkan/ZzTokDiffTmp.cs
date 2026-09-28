namespace OpenTail.Stingray.Tests.Vulkan;
public sealed class ZzTokDiffTmp
{
    [Fact]
    public void Diff()
    {
        if (Environment.GetEnvironmentVariable("ZZ_TOK") != "1") return;
        string sp = Environment.GetEnvironmentVariable("ZZ_SP")!;
        string text = File.ReadAllText(Path.Combine(sp, "wiki20k.txt"));
        foreach (var m in (Environment.GetEnvironmentVariable("ZZ_MODELS") ?? "").Split(","))
        {
            using var model = GgufModel.Open($@"C:/Git-Public/OpenTail.Stingray/models/_models/{m}.gguf");
            var tok = GgufTokenizer.FromGgufModel(model);
            int[] ours = tok.Encode(text).ToArray();
            if (tok.AddBosToken && tok.BosTokenId >= 0 && (ours.Length == 0 || ours[0] != tok.BosTokenId)) ours = new[] { tok.BosTokenId }.Concat(ours).ToArray();
            int[] ref_ = File.ReadAllText(Path.Combine(sp, $"tok_llama_{m}.txt")).Trim().Trim('[', ']').Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x.Trim())).ToArray();
            int first = -1;
            for (int i = 0; i < Math.Min(ours.Length, ref_.Length); i++) if (ours[i] != ref_[i]) { first = i; break; }
            Console.WriteLine($"{m}: ours {ours.Length} llama {ref_.Length} first-diff {first}");
            if (first >= 0)
            {
                int a = Math.Max(0, first - 3);
                Console.WriteLine($"   ours  [{a}..]: {string.Join(",", ours.Skip(a).Take(10))}  = '{tok.Decode(ours.Skip(a).Take(10).ToArray()).Replace("\n", "\n")}'");
                Console.WriteLine($"   llama [{a}..]: {string.Join(",", ref_.Skip(a).Take(10))}  = '{tok.Decode(ref_.Skip(a).Take(10).ToArray()).Replace("\n", "\n")}'");
            }
        }
    }
}
