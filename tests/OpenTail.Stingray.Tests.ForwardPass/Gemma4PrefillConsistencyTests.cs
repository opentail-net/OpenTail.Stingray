namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Gemma-4 E4B (per-layer head dims 256/512 + KV-shared tail layers): batched Prefill must give the
/// same last-position logits as feeding the prompt token by token through Forward. Pins two
/// prefill-only bugs found 2026-09-26 (docs/101, "GPU for LLMs" audit):
/// <list type="number">
/// <item>K/V were staged at a <c>_maxHeadDim</c> head stride while every cache reader uses the
///       layer's own head dim, so all KV heads but the first read zero padding;</item>
/// <item>KV-shared layers left the cache length at the chunk start, so they attended to none of
///       the chunk.</item>
/// </list>
/// Before the fix: cosine 0.966-0.973 at N = 2..8 and 0.908 on the 13-token prompt; after:
/// ≥ 0.9999 / 0.9993. The token-by-token path matches the independent Vulkan pass (0.99966) and
/// llama-server's greedy tokens, so it is the reference here.
/// </summary>
public sealed class Gemma4PrefillConsistencyTests : HeavyTestBase
{
    private const string ModelFile = "gemma-4-E4B-it-Q4_K_M.gguf";

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(13)]
    public void BatchedPrefill_MatchesTokenByToken(int n)
    {
        string? path = FindModelPath();
        Assert.SkipUnless(path is not null, $"{ModelFile} not present");

        using var model = GgufModel.Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.KvSourceLayer);

        int[] all = GgufTokenizer.FromGgufModel(model)
            .Encode("The scheduler assigns runnable threads to cores, balancing throughput.").ToArray();
        int[] prompt = all.Take(Math.Min(n, all.Length)).ToArray();

        using var backend = new CpuBackend();
        float[] batched, sequential = [];
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
            batched = fwd.Prefill(prompt).ToArray();
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
            for (int i = 0; i < prompt.Length; i++)
                sequential = fwd.Forward(prompt[i], i).ToArray();

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < batched.Length; i++)
        {
            dot += (double)batched[i] * sequential[i];
            na += (double)batched[i] * batched[i];
            nb += (double)sequential[i] * sequential[i];
        }
        double cos = dot / Math.Sqrt(na * nb);
        // int8 prefill activations vs the decode path's own quantisation leave ~1e-4..1e-3 of
        // noise; the bugs this guards against measured 0.91-0.97.
        Assert.True(cos > 0.998, $"N={prompt.Length}: prefill vs token-by-token cosine {cos:F6}");
    }

    private static string? FindModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            foreach (var p in new[] { Path.Combine(dir.FullName, "models", ModelFile), Path.Combine(dir.FullName, "models", "_models", ModelFile) })
                if (File.Exists(p)) return p;
            dir = dir.Parent;
        }
        return null;
    }
}
