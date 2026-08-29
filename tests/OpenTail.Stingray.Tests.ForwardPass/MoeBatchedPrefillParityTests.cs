
namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Parity gate for the batched MoE prefill FFN
/// (<c>ForwardPass.MoeFfnBatched</c>, docs/cpu-architecture-kernel-opportunities.md item 6).
///
/// <para>Until this landed, every MoE prompt prefilled one token at a time: the batched cores
/// bailed to sequential <c>Forward</c> because routing is per token, so the dense
/// "one GEMM per weight matrix across N tokens" shape does not apply. The batched path buckets
/// the (token, slot) pairs by selected expert and runs one batch of GEMMs per expert instead.
/// The oracle here is the sequential path it replaced — reached by flipping
/// <see cref="Engine.ForwardPass.MoeBatchedPrefillEnabled"/> off — because that is what
/// production did and what every other MoE result on this model was measured against.</para>
///
/// <para><b>Two arms, deliberately.</b> With <c>Q8PrefillEnabled</c> pinned off the expert GEMMs
/// fall back to the same F32 fused MatVec the sequential path uses, and the batched path is then
/// BIT-IDENTICAL to sequential — nothing about bucketing tokens by expert changes the arithmetic,
/// provided the per-token reduce runs in top-k slot order. That exactness is worth asserting as
/// exactness: an earlier revision reduced in expert order instead and moved the final logits by
/// up to 0.20, which a tolerance-based test would have waved through. With the int8 path on
/// (production's default) activations are additionally quantised per block, so that arm can only
/// assert argmax stability. Conflating the two would let a real bucketing bug hide inside the
/// quantisation budget, so the strict arm is checked separately and first.</para>
///
/// <para>Skipped silently when the MoE model is not on disk; a failure inside prefill must FAIL.</para>
/// </summary>
public sealed class MoeBatchedPrefillParityTests : HeavyTestBase
{
    /// <summary>OLMoE: 16 layers, 64 experts, top-8, no shared expert — the smallest real MoE here.</summary>
    private static string? FindMoePath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>Deterministic pseudo-prompt; the content is irrelevant, the routing spread is the point.</summary>
    private static int[] MakeTokens(int count)
    {
        var t = new int[count];
        for (int i = 0; i < count; i++) t[i] = 1 + ((i * 7919) % 4096);
        return t;
    }

    private static float[] PrefillLogits(GgufModel model, CpuBackend backend, ModelHyperparams hp,
        int[] tokens, bool batchedMoe)
    {
        bool prev = Engine.ForwardPass.MoeBatchedPrefillEnabled;
        Engine.ForwardPass.MoeBatchedPrefillEnabled = batchedMoe;
        try
        {
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            return fwd.Prefill(tokens).ToArray();
        }
        finally { Engine.ForwardPass.MoeBatchedPrefillEnabled = prev; }
    }

    /// <summary>
    /// Strict arm: int8 activation quantisation off on both sides, so the expert dots are the
    /// same F32 kernel the sequential path runs. The contract is bit equality, not a tolerance —
    /// any difference at all means the bucketing, the gather, the scatter or the reduce ORDER
    /// changed the arithmetic, and the whole point of this path is that it must not.
    /// </summary>
    [Fact]
    public void BatchedMoePrefill_MatchesSequential_F32()
    {
        var path = FindMoePath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        bool prevQ8 = SimdKernels.Q8PrefillEnabled;
        // Each expert's batched GEMM (MatMulBatched, allowQ8: true) still runs its bucket size
        // through the ordinary batch-size gate: >= MinBatchForBlas routes to OpenBLAS sgemm
        // regardless of Q8PrefillEnabled. On a machine with OpenBLAS actually loaded, 96 tokens
        // over 64 experts can easily bucket >=16 tokens onto a single expert, and BLAS's summation
        // order is not bit-identical to the per-token FusedMatVec the sequential oracle uses.
        // Push the threshold out of reach so this arm's "must agree bit for bit" contract holds
        // regardless of what's installed on the machine running it.
        int prevMinBatchForBlas = SimdKernels.MinBatchForBlas;
        SimdKernels.Q8PrefillEnabled = false;
        SimdKernels.MinBatchForBlas = int.MaxValue;
        try
        {
            using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
            var model = modelHandle.Model;
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            Assert.True(hp.IsMoE, "expected an MoE model — the batched path under test is MoE-only");
            using var backend = new CpuBackend();

            int[] tokens = MakeTokens(96);
            float[] seq = PrefillLogits(model, backend, hp, tokens, batchedMoe: false);
            float[] bat = PrefillLogits(model, backend, hp, tokens, batchedMoe: true);

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length && firstDiff < 0; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(bat[i]))
                    firstDiff = i;
            Assert.True(firstDiff < 0,
                firstDiff < 0 ? "" :
                $"logit[{firstDiff}] differs: sequential {seq[firstDiff]} vs batched MoE {bat[firstDiff]} "
                + "— the F32 arms must agree bit for bit");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
        }
        finally
        {
            SimdKernels.Q8PrefillEnabled = prevQ8;
            SimdKernels.MinBatchForBlas = prevMinBatchForBlas;
        }
    }

    /// <summary>
    /// Production arm: the int8 expert GEMMs the batched path actually uses, against the F32
    /// sequential trunk. This is a quantisation-error budget, not a parity proof — its job is to
    /// catch a batched path that has drifted far enough to pick a different next token, which is
    /// the level at which the divergence stops being a rounding question.
    /// </summary>
    [Fact]
    public void BatchedMoePrefill_MatchesSequential_Int8_ArgmaxStable()
    {
        var path = FindMoePath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        if (!hp.IsMoE) return;
        using var backend = new CpuBackend();

        int[] tokens = MakeTokens(96);
        float[] seq = PrefillLogits(model, backend, hp, tokens, batchedMoe: false);
        float[] bat = PrefillLogits(model, backend, hp, tokens, batchedMoe: true);

        Assert.Equal(seq.Length, bat.Length);
        Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
    }

    /// <summary>
    /// The batched path must survive chunking: a prompt admitted as 64 + 32 has to agree with the
    /// same prompt admitted in one pass. This is the defect class every threshold in the batched
    /// prefill carries — a bucket size that falls below some kernel's minimum on one chunk and not
    /// the other gives two different answers for one prompt.
    /// </summary>
    [Fact]
    public void BatchedMoePrefill_Chunked_MatchesFull()
    {
        var path = FindMoePath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        using var modelHandle = SharedModelCacheFixture.Instance.Acquire(path);
        var model = modelHandle.Model;
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        if (!hp.IsMoE) return;
        using var backend = new CpuBackend();

        int[] tokens = MakeTokens(96);

        bool prev = Engine.ForwardPass.MoeBatchedPrefillEnabled;
        Engine.ForwardPass.MoeBatchedPrefillEnabled = true;
        try
        {
            float[] full;
            using (var fwd = new Engine.ForwardPass(model, backend, hp))
            using (var cache = fwd.CreateCache())
                full = fwd.PrefillWithCache(tokens, cache).ToArray();

            float[] split;
            using (var fwd = new Engine.ForwardPass(model, backend, hp))
            using (var cache = fwd.CreateCache())
            {
                fwd.PrefillWithCache(new ArraySegment<int>(tokens, 0, 64), cache, startPos: 0);
                split = fwd.PrefillWithCache(new ArraySegment<int>(tokens, 64, 32), cache, startPos: 64)
                           .ToArray();
            }

            Assert.Equal(full.Length, split.Length);
            Assert.Equal(Sampler.Greedy(full), Sampler.Greedy(split));
        }
        finally { Engine.ForwardPass.MoeBatchedPrefillEnabled = prev; }
    }
}
