namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Nemotron-3.5 Lightning 30B-A3B (nemotron_h_moe, non-gated ReLU² Q4_0 experts): the batched routed-expert path
/// (<c>ForwardPass.MoeNonGatedBatchedEnabled</c>) must produce bit-identical logits to the per-expert sequential loop
/// it replaces, at every position of a teacher-forced decode on the real checkpoint.
/// </summary>
public sealed class NemotronBatchedExpertParityTests : HeavyTestBase
{
    private const string ModelFile = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf";
    private const string Prompt = "The capital of France is";

    [Fact]
    public void BatchedNonGatedExperts_MatchSequentialLoop_BitForBit()
    {
        var path = TeacherForcedParity.FindModel(ModelFile);
        Xunit.Assert.SkipWhen(path is null, $"{ModelFile} is required for this parity check.");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path!);
        var model = modelHandle.Model;
        Xunit.Assert.Equal("nemotron_h_moe", Convert.ToString(model.Metadata["general.architecture"]));
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        var tokens = TeacherForcedParity.Tokenize(ModelFile, Prompt).ToArray();

        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 256);
        bool previous = Engine.ForwardPass.MoeNonGatedBatchedEnabled;
        try
        {
            Engine.ForwardPass.MoeNonGatedBatchedEnabled = true;
            var batched = RunLogits(fwd, tokens);

            // ResetCache clears the Mamba-2 and short-conv recurrent state as well as the KV cache;
            // ResetKvCache alone would leave the first arm's SSM state in place for the second.
            fwd.ResetCache();
            Engine.ForwardPass.MoeNonGatedBatchedEnabled = false;
            var sequential = RunLogits(fwd, tokens);

            Xunit.Assert.Equal(sequential.Length, batched.Length);
            for (int pos = 0; pos < batched.Length; pos++)
            {
                Xunit.Assert.Equal(sequential[pos].Length, batched[pos].Length);
                for (int i = 0; i < batched[pos].Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(batched[pos][i]) != BitConverter.SingleToInt32Bits(sequential[pos][i]))
                        Xunit.Assert.Fail($"logit mismatch at position {pos}, index {i}: batched {batched[pos][i]:R} vs sequential {sequential[pos][i]:R}");
                }
            }
        }
        finally
        {
            Engine.ForwardPass.MoeNonGatedBatchedEnabled = previous;
        }
    }

    private static float[][] RunLogits(Engine.ForwardPass fwd, IReadOnlyList<int> tokens)
    {
        var rows = new float[tokens.Count][];
        for (int pos = 0; pos < tokens.Count; pos++)
            rows[pos] = fwd.Forward(tokens[pos], pos).ToArray();
        return rows;
    }
}
