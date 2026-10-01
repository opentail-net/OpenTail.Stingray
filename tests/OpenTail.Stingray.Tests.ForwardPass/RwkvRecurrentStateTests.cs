using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Recurrent-state contracts of the RWKV passes on real weights: a reset returns the pass to its
/// initial state exactly, batched (chunked) prefill agrees with token-by-token stepping, and
/// rewinding — which a recurrent state cannot do — fails loudly instead of returning stale logits.
/// </summary>
public sealed class RwkvRecurrentStateTests : HeavyTestBase
{
    // "Once upon a time, in a small village by the sea, there lived an old fisherman who"
    private static readonly int[] s_prompt =
        [23977, 32350, 332, 32251, 45, 4596, 332, 39720, 53287, 4450, 22590, 22446, 45, 39934, 38917, 4419, 22221, 45998, 8148, 22762];

    public static TheoryData<string, string> Models => new()
    {
        { @"E:\_models\rwkv7-goose-world3-1b5", "RWKV7-Goose-World3-1.5B-HF.Q8_0.gguf" },
        { @"E:\_models\rwkv6-world-1b6", "rwkv-6-world-1.6b-Q8_0.gguf" },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void ResetCache_RestoresInitialStateExactly(string dir, string file)
    {
        var path = Path.Combine(dir, file);
        Assert.SkipUnless(File.Exists(path), $"{file} is required.");
        using var model = GgufModel.Open(path);
        using var fwd = RwkvForwardPassBase.Create(model);

        float[] first = fwd.Prefill(s_prompt).ToArray();
        fwd.Forward(Sampler.Greedy(first), s_prompt.Length);
        fwd.ResetCache();
        float[] again = fwd.Prefill(s_prompt).ToArray();
        Assert.Equal(first, again);
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void BatchedPrefill_AgreesWithStepwiseDecode(string dir, string file)
    {
        var path = Path.Combine(dir, file);
        Assert.SkipUnless(File.Exists(path), $"{file} is required.");
        using var model = GgufModel.Open(path);

        float[] batched, stepwise;
        using (var f = RwkvForwardPassBase.Create(model)) batched = f.Prefill(s_prompt).ToArray();
        using (var f = RwkvForwardPassBase.Create(model))
        {
            ReadOnlySpan<float> l = default;
            for (int i = 0; i < s_prompt.Length; i++) l = f.Forward(s_prompt[i], i);
            stepwise = l.ToArray();
        }
        // Different kernels (batched tier vs single-row matvec), same math. Measured 2026-10-01 over
        // all 65536 logits: rwkv7 < 0.05, rwkv6 0.565 — RWKV-6's decay is exp(-exp(·)), which
        // amplifies rounding. Neither path is the less faithful one: teacher-forced against
        // llama-server, a stepwise prompt gives mean |Δlogprob| 0.029, a batched one 0.031.
        Assert.Equal(Sampler.Greedy(stepwise), Sampler.Greedy(batched));
        float maxDiff = 0f;
        for (int i = 0; i < batched.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(batched[i] - stepwise[i]));
        Assert.True(maxDiff < 1.0f, $"max |batched - stepwise| logit = {maxDiff}");
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void RewindAndOutOfOrderPositions_Throw(string dir, string file)
    {
        var path = Path.Combine(dir, file);
        Assert.SkipUnless(File.Exists(path), $"{file} is required.");
        using var model = GgufModel.Open(path);
        using var fwd = RwkvForwardPassBase.Create(model);

        fwd.Prefill(s_prompt[..4]);
        fwd.TruncateTo(4);                                          // no-op is allowed
        Assert.Throws<NotSupportedException>(() => fwd.TruncateTo(2));
        Assert.Throws<NotSupportedException>(() => fwd.Forward(s_prompt[4], 7));
        fwd.TruncateTo(0);                                          // full reset is allowed
        fwd.Forward(s_prompt[0], 0);
    }
}
