namespace OpenTail.Stingray.Tests.ForwardPass;

// Scratch (untracked): our top-5 log-probs after a token prefix, to compare with llama-server n_probs. ZZ_NH=1.
public sealed class ZzNemotronTopKTmp
{
    [Fact]
    public void TopK()
    {
        if (Environment.GetEnvironmentVariable("ZZ_NH") != "1") return;
        string path = Environment.GetEnvironmentVariable("ZZ_MODEL") ?? @"C:\Git-Public\OpenTail.Stingray\models\_models\nemotron-nano-12b-v2-vl-Q2_K.gguf";
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tok = GgufTokenizer.FromGgufModel(model);
        var ids = Environment.GetEnvironmentVariable("ZZ_IDS") is { Length: > 0 } raw
            ? raw.Split(',').Select(int.Parse).ToList()
            : tok.Encode(Environment.GetEnvironmentVariable("ZZ_PROMPT") ?? "The capital of France is").ToList();
        Console.WriteLine("prompt ids: " + string.Join(",", ids));
        foreach (var extra in (Environment.GetEnvironmentVariable("ZZ_EXTRA") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries))
            ids.Add(int.Parse(extra));
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 1024);
        var logits = fwd.Prefill(ids).ToArray();
        double max = logits.Max();
        double lse = max + Math.Log(logits.Sum(l => Math.Exp(l - max)));
        foreach (var (l, i) in logits.Select((l, i) => (l, i)).OrderByDescending(p => p.l).Take(5))
            Console.WriteLine($"  {i,7} {tok.Decode([i]),-16} logprob {l - lse:F3}");
    }
}
