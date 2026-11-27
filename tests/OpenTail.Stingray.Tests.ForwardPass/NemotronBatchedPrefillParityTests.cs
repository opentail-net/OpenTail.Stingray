namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Nemotron-3.5 Lightning 30B-A3B (nemotron_h_moe, non-gated ReLU² Q4_0 experts, hybrid Mamba-2 / attention / FFN-only
/// layers): the batched prefill trunk must agree with the stepwise decode trunk at every prompt position of the real
/// checkpoint. Two fresh instances, so no recurrent or KV state is shared between the two arms.
/// </summary>
public sealed class NemotronBatchedPrefillParityTests : HeavyTestBase
{
    private const string ModelFile = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf";
    private const string Prompt = "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric effect, and";

    [Fact]
    public void BatchedPrefill_MatchesStepwiseDecode_AtEveryPosition()
    {
        var path = TeacherForcedParity.FindModel(ModelFile);
        Xunit.Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Xunit.Assert.Equal("nemotron_h_moe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokens = TeacherForcedParity.Tokenize(ModelFile, Prompt).ToArray();

        using var backend = new CpuBackend();
        var stepwise = new float[tokens.Length][];
        using (var fwdA = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
        {
            for (int pos = 0; pos < tokens.Length; pos++)
                stepwise[pos] = fwdA.Forward(tokens[pos], pos).ToArray();
        }

        var batched = new float[tokens.Length][];
        using (var fwdB = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256))
        {
            fwdB.PrefillWithPerPositionLogits(tokens, 0, (pos, logits) => batched[pos] = logits.ToArray());
        }

        int argmaxMismatches = 0;
        double worst = 0;
        int worstPos = -1;
        for (int pos = 0; pos < tokens.Length; pos++)
        {
            Xunit.Assert.NotNull(batched[pos]);
            if (ArgMax(batched[pos]) != ArgMax(stepwise[pos])) argmaxMismatches++;
            double d = 0;
            for (int i = 0; i < stepwise[pos].Length; i++)
                d = Math.Max(d, Math.Abs(batched[pos][i] - stepwise[pos][i]));
            if (d > worst) { worst = d; worstPos = pos; }
        }
        Console.WriteLine($"PREFILL-PARITY positions={tokens.Length} argmaxMismatches={argmaxMismatches} maxAbsLogitDiff={worst:G6} at pos {worstPos}");
        Xunit.Assert.Equal(0, argmaxMismatches);
        // Observed exactly 0 on 2026-11-27; the tolerance only guards against a future kernel change that reorders sums.
        Xunit.Assert.True(worst < 1e-3, $"max |batched - stepwise| logit diff {worst:G6} at position {worstPos} exceeds 1e-3");
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
