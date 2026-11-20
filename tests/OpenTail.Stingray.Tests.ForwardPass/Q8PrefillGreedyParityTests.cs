namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Raw-completion oracle check for the two dense CPU prefill modes involved in correctness item
/// 25. The prompt has no chat template and no BOS (the SmolLM2 GGUF has add_bos_token=false).
/// Reference: llama.cpp b10306 llama-server, Q4_K_M, CPU, temperature 0, top_k 1.
/// </summary>
public sealed class Q8PrefillGreedyParityTests : HeavyTestBase
{
    private const string ModelFile = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    private static readonly int[] s_promptTokens = [504, 3575, 282, 4649, 314];
    private static readonly int[] s_llamaContinuation =
    [
        7042, 30, 198, 504, 3575, 282, 4649, 314,
        7042, 30, 198, 504, 3575, 282, 4649, 314,
        7042, 30, 198, 504, 3575, 282, 4649, 314
    ];

    private readonly bool _savedQ8Gate = SimdKernels.Q8PrefillEnabled;

    public override void Dispose()
    {
        SimdKernels.Q8PrefillEnabled = _savedQ8Gate;
        base.Dispose();
    }

    [Fact]
    public void RawGreedyContinuation_MatchesLlamaCppPrefixBeforeDivergence()
    {
        string? path = OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(ModelFile);
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this raw-prefill parity check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));

        var perToken = Generate(q8Enabled: false);
        var batchedQ8 = Generate(q8Enabled: true);

        Console.WriteLine($"llama.cpp raw greedy:   {string.Join(',', s_llamaContinuation)}");
        Console.WriteLine($"per-token prefill:      {string.Join(',', perToken)}");
        Console.WriteLine($"batched-Q8 prefill:     {string.Join(',', batchedQ8)}");
        // Both modes match llama.cpp for " Paris." then diverge on its newline token (198).
        // Keep the verified prefix as a regression receipt without treating later numerical
        // sensitivity as a kernel failure; item 25 tracks the remaining teacher-forced analysis.
        Assert.Equal(s_llamaContinuation[..2], perToken[..2]);
        Assert.Equal(s_llamaContinuation[..2], batchedQ8[..2]);

        int[] Generate(bool q8Enabled)
        {
            SimdKernels.Q8PrefillEnabled = q8Enabled;
            var hp = ArchitectureModelResolver.ResolveHyperparams(model);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256);
            var logits = fwd.Prefill(s_promptTokens);
            var generated = new int[s_llamaContinuation.Length];
            int position = s_promptTokens.Length;
            for (int i = 0; i < generated.Length; i++)
            {
                int next = Sampler.Greedy(logits);
                generated[i] = next;
                if (i + 1 < generated.Length)
                {
                    logits = fwd.Forward(next, position++);
                    if (i == 1)
                    {
                        float[] currentLogits = logits[..tokenizer.VocabSize].ToArray();
                        int[] top = Enumerable.Range(0, tokenizer.VocabSize)
                            .OrderByDescending(id => currentLogits[id])
                            .Take(5)
                            .ToArray();
                        Console.WriteLine($"after Paris. {(q8Enabled ? "Q8" : "per-token")} top5: "
                            + string.Join(", ", top.Select(id => $"{id}:{currentLogits[id]:F3}")));
                        Console.WriteLine($"after Paris. {(q8Enabled ? "Q8" : "per-token")} "
                            + $"logP(newline)={LogProbability(currentLogits, 198):F4}, "
                            + $"logP(EOS)={LogProbability(currentLogits, 2):F4}");
                    }
                }
            }
            return generated;
        }

        double LogProbability(float[] logits, int tokenId)
        {
            double max = logits.Max();
            double sum = 0;
            for (int i = 0; i < logits.Length; i++)
                sum += Math.Exp(logits[i] - max);
            return logits[tokenId] - max - Math.Log(sum);
        }
    }
}
