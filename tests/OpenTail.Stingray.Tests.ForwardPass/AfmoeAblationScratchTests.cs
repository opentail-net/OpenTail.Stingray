namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>Temporary diagnostic: where does the reference token rank under feature toggles? Not committed.</summary>
public sealed class AfmoeAblationScratchTests : HeavyTestBase
{
    [Fact]
    public void Ablation()
    {
        var path = new[] { @"H:\_models\Trinity-Mini-Q4_K_M.gguf" }.FirstOrDefault(File.Exists);
        Assert.SkipWhen(path is null, "model missing");
        using var handle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = handle.Model;
        var hp0 = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tok = GgufTokenizer.FromGgufModel(model);
        var ids = tok.Encode("The capital of France is").ToList();
        Console.WriteLine($"[abl] prompt ids: {string.Join(",", ids)} embScale={hp0.EmbeddingScale} gate={hp0.AttentionOutputGate} layers={hp0.NumLayers}");
        var variants = new (string Name, ModelHyperparams Hp)[]
        {
            ("current", hp0),
            ("embScale=1", hp0 with { EmbeddingScale = 1f }),
            ("gate off", hp0 with { AttentionOutputGate = false }),
            ("embScale=1 + gate off", hp0 with { EmbeddingScale = 1f, AttentionOutputGate = false }),
        };
        foreach (var (name, hp) in variants)
        {
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256);
            var l = fwd.Prefill(ids).ToArray();
            int arg = 0; for (int i = 1; i < l.Length; i++) if (l[i] > l[arg]) arg = i;
            int rank = 0; float r = l[8849]; for (int i = 0; i < l.Length; i++) if (l[i] > r) rank++;
            Console.WriteLine($"[abl] {name}: argmax {arg} (logit {l[arg]:F2}); ref token 8849 logit {r:F2} rank {rank}");
        }
    }
}
