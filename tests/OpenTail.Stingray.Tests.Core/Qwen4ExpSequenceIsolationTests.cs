using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 1 acceptance (Qwen 3.8 plan): two sequences on one model must not share state. Real checkpoint only;
/// skipped visibly when the UD-IQ1_S checkpoint is absent.
/// </summary>
public sealed class Qwen4ExpSequenceIsolationTests
{
    // Five-token prompts, same length so they can be stepped in lockstep.
    private static readonly int[] SeqA = { 760, 6511, 314, 9338, 369 };   // "The capital of France is"
    private static readonly int[] SeqB = { 760, 7526, 11247, 303, 279 }; // "The largest planet in the"

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        Qwen4ExpTestSupport.TrunkFileName);

    [Fact]
    public void TwoSequences_Interleaved_MatchSeparateRunsBitwise()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var fp = new Qwen4ExpForwardPass(model, hp, tensors);

        float[][] soloA = RunSolo(fp, SeqA);
        float[][] soloB = RunSolo(fp, SeqB);

        var stateA = fp.CreateSequenceState();
        var stateB = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++)
        {
            var la = fp.Forward(SeqA[pos], pos, stateA).ToArray();
            var lb = fp.Forward(SeqB[pos], pos, stateB).ToArray();
            Assert.Equal(soloA[pos], la);
            Assert.Equal(soloB[pos], lb);
        }
    }

    [Fact]
    public void ResetOfOneSequence_DoesNotChangeTheOther()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var fp = new Qwen4ExpForwardPass(model, hp, tensors);

        // Solo continuation of A: prompt, then one extra token.
        float[][] soloA = RunSolo(fp, SeqA);
        var stateA0 = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++) fp.Forward(SeqA[pos], pos, stateA0);
        int nextToken = 1;
        var expectedA = fp.Forward(nextToken, SeqA.Length, stateA0).ToArray();

        // Same continuation, but B runs and is reset in between.
        var stateA = fp.CreateSequenceState();
        var stateB = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++)
        {
            fp.Forward(SeqA[pos], pos, stateA);
            fp.Forward(SeqB[pos], pos, stateB);
        }
        stateB.Reset();
        fp.Forward(SeqB[0], 0, stateB); // B starts over, at position 0 of a fresh sequence
        var actualA = fp.Forward(nextToken, SeqA.Length, stateA).ToArray();

        Assert.Equal(expectedA, actualA);
        Assert.NotEqual(soloA[SeqA.Length - 1], actualA); // sanity: the extra token changes the distribution
    }

    [Fact]
    public void Clone_ContinuesIdentically_AndDivergesIndependently()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var fp = new Qwen4ExpForwardPass(model, hp, tensors);

        var original = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++) fp.Forward(SeqA[pos], pos, original);
        var clone = original.Clone();

        // Same next token on both: identical continuation.
        var fromOriginal = fp.Forward(1, SeqA.Length, original).ToArray();
        var fromClone = fp.Forward(1, SeqA.Length, clone).ToArray();
        Assert.Equal(fromOriginal, fromClone);

        // Different next tokens: the two continue independently and each keeps its own history.
        var originalNext = fp.Forward(2, SeqA.Length + 1, original).ToArray();
        var cloneNext = fp.Forward(3, SeqA.Length + 1, clone).ToArray();
        Assert.NotEqual(originalNext, cloneNext);

        var soloOriginal = RunSolo(fp, SeqA.Append(1).Append(2).ToArray());
        Assert.Equal(soloOriginal[SeqA.Length + 1], originalNext);
    }

    [Fact]
    public void DisposedSequence_RejectsUse_AndLeavesOthersIntact()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var fp = new Qwen4ExpForwardPass(model, hp, tensors);

        float[][] soloA = RunSolo(fp, SeqA);
        var stateA = fp.CreateSequenceState();
        var stateB = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++)
        {
            Assert.Equal(soloA[pos], fp.Forward(SeqA[pos], pos, stateA).ToArray());
            fp.Forward(SeqB[pos], pos, stateB);
        }

        stateB.Dispose();
        Assert.True(stateB.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => fp.Forward(SeqB[0], 0, stateB));

        // A keeps going as if B had never existed: one more token, compared with its solo run.
        var expected = RunSolo(fp, SeqA.Append(1).ToArray())[SeqA.Length];
        Assert.Equal(expected, fp.Forward(1, SeqA.Length, stateA).ToArray());
    }

    [Fact]
    public void Restore_ReturnsTheSequenceToTheSnapshot()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var fp = new Qwen4ExpForwardPass(model, hp, tensors);

        var state = fp.CreateSequenceState();
        for (int pos = 0; pos < SeqA.Length; pos++) fp.Forward(SeqA[pos], pos, state);
        var snapshot = state.Snapshot();
        var first = fp.Forward(1, SeqA.Length, state).ToArray();
        fp.Forward(2, SeqA.Length + 1, state);          // move the sequence on
        state.Restore(snapshot);                         // back to the snapshot
        Assert.Equal(first, fp.Forward(1, SeqA.Length, state).ToArray());
    }

    private static float[][] RunSolo(Qwen4ExpForwardPass fp, int[] tokens)
    {
        var state = fp.CreateSequenceState();
        var result = new float[tokens.Length][];
        for (int pos = 0; pos < tokens.Length; pos++)
            result[pos] = fp.Forward(tokens[pos], pos, state).ToArray();
        return result;
    }
}
