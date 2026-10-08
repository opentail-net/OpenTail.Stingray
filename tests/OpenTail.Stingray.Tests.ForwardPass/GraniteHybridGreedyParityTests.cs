
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// IBM Granite 4.0-H (granitehybrid: Mamba-2 mixer layers + NoPE attention layers) greedy parity against
/// llama-server (vendored tools/llama.cpp, same GGUF, raw completion, temperature 0, top_k 1, ignore_eos where
/// the reference ran past EOS; captured 2026-09-27). Exercises the Mamba-2 recurrent state (conv + SSM) across
/// prefill and decode, which a KV-cache model never touches. Admission evidence is second-half perplexity
/// (ModelCompatibility.cs); these pin exact tokens where llama.cpp's top-1 is not a near-tie.
/// </summary>
public sealed class GraniteHybridGreedyParityTests : HeavyTestBase
{
    private const string Model350M = "granite-4.0-h-350m-Q8_0.gguf";
    private const string Model1B = "granite-4.0-h-1b-Q8_0.gguf";

    private const string EinsteinPrompt =
        "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric effect, and";

    [Fact]
    public void Granite4H_350M_ShortPrompt_GreedyMatchesLlamaServer()
    {
        // " Paris." then EOS.
        AssertGreedy(Model350M, "The capital of France is", 5, [12366, 13, 100257]);
    }

    [Fact]
    public void Granite4H_1B_ShortPrompt_GreedyMatchesLlamaServer()
    {
        // " Paris. Paris is located in the north-central part of France, in the region" (ignore_eos: llama-server
        // masks EOS, which our greedy pick would otherwise take after " Paris.").
        AssertGreedy(Model1B, "The capital of France is", 5,
            [12366, 13, 12366, 374, 7559, 304, 279, 10411, 85181, 961, 315, 9822, 11, 304, 279, 5654], maskEos: true);
    }

    [Fact]
    public void Granite4H_1B_LongerPrompt_GreedyMatchesLlamaServer()
    {
        // " the second explained the special theory of relativity. The third paper explained the". The 16th token is
        // a near-tie in llama.cpp (" equivalence" logprob -1.288 vs " general" -1.352; ours picks " general"), so
        // only the first 15 are pinned.
        AssertGreedy(Model1B, EinsteinPrompt, 23,
            [279, 2132, 11497, 279, 3361, 10334, 315, 1375, 44515, 13, 578, 4948, 5684, 11497, 279], maskEos: true);
    }

    /// <summary>The Mamba-2 state is not a KV cache: ResetCache must zero it, so the same prompt after a reset gives
    /// the same logits as on a fresh instance.</summary>
    [Fact]
    public void Granite4H_ResetCache_ClearsRecurrentState()
    {
        var path = FindModel(Model350M);
        Assert.SkipWhen(path is null, $"{Model350M} is required.");
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokens = GgufTokenizer.FromGgufModel(model).Encode(EinsteinPrompt);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        var first = fwd.Prefill(tokens).ToArray();
        fwd.ResetCache();
        var second = fwd.Prefill(tokens).ToArray();
        Assert.Equal(first, second);
    }

    /// <summary>Layers without attention must not allocate KV pages (docs/103 item 9): only the attention layers own
    /// pages after a prefill, and the output is unchanged (the greedy tests above still pass).</summary>
    [Fact]
    public void Granite4H_MambaLayers_AllocateNoKvPages()
    {
        var path = FindModel(Model1B);
        Assert.SkipWhen(path is null, $"{Model1B} is required.");
        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokens = GgufTokenizer.FromGgufModel(model).Encode(EinsteinPrompt);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);
        fwd.Prefill(tokens);

        int attnLayers = 0;
        long used = 0;
        for (int l = 0; l < hp.NumLayers; l++)
        {
            int pages = fwd.Cache.AllocatedPages(l);
            if (hp.IsMamba2Layer![l]) Assert.Equal(0, pages);
            else { Assert.True(pages > 0, $"attention layer {l} has no KV pages"); attnLayers++; }
            used += pages * fwd.Cache.PageBytes;
        }
        long allLayers = (long)hp.NumLayers * fwd.Cache.AllocatedPages(Array.FindIndex(hp.IsMamba2Layer!.ToArray(), r => !r)) * fwd.Cache.PageBytes;
        Console.WriteLine($"[granite4h kv] {attnLayers}/{hp.NumLayers} attention layers own pages: {used / 1024} KiB vs {allLayers / 1024} KiB if every layer stored KV");
    }

    private const int EndOfText = 100257; // <|end_of_text|>

    private static void AssertGreedy(string modelFile, string prompt, int expectedPromptLen, int[] expected, bool maskEos = false)
    {
        var path = FindModel(modelFile);
        Assert.SkipWhen(path is null, $"{modelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Assert.Equal("granitehybrid", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        Assert.NotNull(hp.Mamba2);
        Assert.Contains(true, hp.IsMamba2Layer!);
        Assert.Contains(false, hp.IsMamba2Layer!);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var promptTokens = tokenizer.Encode(prompt);
        Assert.Equal(expectedPromptLen, promptTokens.Count);

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        var logits = fwd.Prefill(promptTokens);
        var generated = new List<int>(expected.Length);
        int pos = promptTokens.Count;
        for (int i = 0; i < expected.Length; i++)
        {
            int next;
            if (maskEos)
            {
                var masked = logits.ToArray();
                masked[EndOfText] = float.NegativeInfinity;
                next = Sampler.Greedy(masked);
            }
            else next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < expected.Length) logits = fwd.Forward(next, pos++);
        }

        Console.WriteLine(tokenizer.Decode(generated));
        Assert.Equal(expected, generated);
    }

    private static string? FindModel(string modelFile) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindOrReport(modelFile);
}
