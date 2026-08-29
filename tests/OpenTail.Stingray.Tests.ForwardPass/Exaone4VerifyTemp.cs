
namespace OpenTail.Stingray.Tests.ForwardPass;

public sealed class Exaone4VerifyTemp : HeavyTestBase
{
    private const string ModelFile = "exaone4-1.2b-Q8_0.gguf";

    private static readonly int[] s_promptTokens = [1320, 7304, 670, 9776, 772];

    private static readonly int[] s_referenceContinuationTokens =
        [619, 7304, 670, 9776, 772, 619, 7304, 670, 9776, 772, 619, 7304, 670, 9776, 772, 619, 7304, 670, 9776, 772, 619, 7304, 670, 9776];

    [Fact]
    public void Exaone4_GreedyContinuation_MatchesLlamaCpp()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity receipt.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Assert.Equal("exaone4", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.Equal(s_promptTokens, tokenizer.Encode("The capital of France is"));
        Assert.True(hp.HasPostAttnNorm, "exaone4 (this size) has no pre-norm at all, only post-norm");
        Assert.True(hp.HasPostFfwNorm, "exaone4 (this size) has no pre-norm at all, only post-norm");
        Assert.True(hp.HasQkNorm, "exaone4 has learned attn_q_norm/attn_k_norm");

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048);

        int n = s_referenceContinuationTokens.Length;
        var logits = fwd.Prefill(s_promptTokens);
        var generated = new List<int>(n);
        int pos = s_promptTokens.Length;
        for (int i = 0; i < n; i++)
        {
            int next = Sampler.Greedy(logits);
            generated.Add(next);
            if (i + 1 < n) logits = fwd.Forward(next, pos++);
        }

        for (int i = 0; i < n; i++)
            Assert.Equal(s_referenceContinuationTokens[i], generated[i]);
    }

    [Fact]
    public void Exaone4_DecodeStepwise_AgreesWithSinglePassPrefill()
    {
        var path = FindModel();
        Assert.SkipWhen(path is null, $"{ModelFile} is required for this consistency check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        int[] full = [.. s_promptTokens, s_referenceContinuationTokens[0], s_referenceContinuationTokens[1]];

        using var backend = new CpuBackend();

        float[] stepwise;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            fwd.Prefill(s_promptTokens);
            var logits = fwd.Forward(full[^2], s_promptTokens.Length);
            logits = fwd.Forward(full[^1], s_promptTokens.Length + 1);
            stepwise = logits[..tokenizer.VocabSize].ToArray();
        }

        float[] singlePass;
        using (var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 2048))
        {
            singlePass = fwd.Prefill(full)[..tokenizer.VocabSize].ToArray();
        }

        int argmaxStep = Array.IndexOf(stepwise, stepwise.Max());
        int argmaxFull = Array.IndexOf(singlePass, singlePass.Max());

        Assert.True(argmaxStep == argmaxFull,
            $"prefill/decode disagree on argmax: stepwise {argmaxStep} vs single-pass {argmaxFull}");
    }

    private static string? FindModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        return null;
    }
}
