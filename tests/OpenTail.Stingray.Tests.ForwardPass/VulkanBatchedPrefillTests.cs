using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Parity gate for the weight-amortizing batched Vulkan prefill against the per-token
/// <c>Forward</c> loop it replaces.
///
/// <para>The batched trunk reads each weight matrix from VRAM once per chunk of up to 8 tokens
/// rather than once per token, and its attention is causal with token i attending [0, startPos+i]
/// — the same semantics the sequential loop produces. Prefill takes the FP path (it passes
/// <c>allowInt8: false</c>), so the two agree to FP32 reassociation noise.</para>
///
/// <para>These asserted BIT-EXACT equality until the single-row <c>MatVecQ4K</c> was rewritten for
/// a coalesced weight read (2.4x, perf-loop iteration 28). That rewrite changes the order the
/// per-lane products are summed, and <c>MatVecBatchedQ4K</c> deliberately kept the old order (the
/// same rewrite was a 2x LOSS there — it uncoalesces the input, which the batched kernel re-reads
/// nTok times). So the reference and batched paths no longer share a summation order and differ in
/// the last few mantissa bits — measured 6.1e-5 on a logit of ~3, i.e. ~2e-5 relative. The bound
/// below is deliberately tight: it is ~80x the observed delta but ~4000x smaller than the logit
/// scale, so it still fails loudly on a real divergence rather than absorbing one.</para>
///
/// <para>The int8 activation path is deliberately NOT used for prefill and is NOT what these
/// cover — it perturbs logits enough to flip the argmax on near-tied short prompts, which would
/// change the first generated token.</para>
/// </summary>
public sealed class VulkanBatchedPrefillTests
{
    /// <summary>See the class remarks: ~80x the measured delta, ~4000x below the logit scale.</summary>
    private const double FpReassociationTolerance = 5e-3;

    /// <summary>
    /// Tolerance for the ACTIVE Vulkan matmul path's disagreement with the per-token path.
    /// </summary>
    /// <remarks>
    /// <para>Path 1's batched matvec keeps the single-row kernel's element iteration order exactly,
    /// so it lands inside <see cref="FpReassociationTolerance"/>. <b>Path 2 cannot.</b> Its tiled
    /// GEMM carries one running sum per output across the whole K dimension, where Path 1 forms a
    /// per-lane partial sum and then a shared-memory tree reduction — a different association of the
    /// same additions. Measured on the reference model: 9.4e-3 max abs logit delta, against 5e-3 for
    /// Path 1.</para>
    ///
    /// <para>This is why Path 1 remains the default. A delta at this scale is well inside the noise
    /// that flipped 2 of 6 short-prompt argmaxes when int8 activations were tried, so <b>Path 2 needs
    /// a perplexity gate before it could take the default</b> — exactly the bar the CPU's Path 2 had
    /// to clear (wikitext-2, 8191 scored tokens) and has not been run for this kernel yet. The
    /// greedy-argmax assertion below still applies to both paths and is the sharper check.</para>
    /// </remarks>
    private static double ActivePathTolerance =>
        VulkanMatMulPathConfig.UsePath2 ? 2e-2 : FpReassociationTolerance;

    private static string? FindModelPath()
    {
        var dir = AppDomain.CurrentDomain.BaseDirectory;
        while (dir is not null)
        {
            var p = Path.Combine(dir, "models", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    private static VulkanBackend? TryCreateBackend()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    /// <summary>
    /// Tokens chosen to span several chunk boundaries (17 tokens = chunks of 8, 8, 1) so the
    /// ragged final chunk and the chunk-to-chunk KV handoff are both exercised, not just one
    /// full-width batch.
    /// </summary>
    private static readonly int[] Prompt =
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

    [Fact]
    public void BatchedPrefill_MatchesPerTokenPrefill()
    {
        var path = FindModelPath();
        if (path is null) return;
        using var gpu = TryCreateBackend();
        if (gpu is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        float[] reference, batched;
        bool usedBatchedPath;

        using (var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
        {
            if (!fwd.CanBatchedTrunk) return; // model not eligible; nothing to compare
            fwd.DisableBatchedPrefill = true;
            reference = fwd.Prefill(Prompt).ToArray();
        }

        using (var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
        {
            usedBatchedPath = fwd.CanBatchedTrunk;
            batched = fwd.Prefill(Prompt).ToArray();
        }

        Assert.True(usedBatchedPath, "expected the batched prefill path to be eligible for this model");
        Assert.Equal(reference.Length, batched.Length);

        double maxAbs = 0;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(reference[i] - batched[i]));
        Assert.True(maxAbs < FpReassociationTolerance,
            $"batched prefill diverged from the per-token path beyond FP reassociation noise: " +
            $"max abs logit delta {maxAbs:R} (tolerance {FpReassociationTolerance})");
        Assert.Equal(Sampler.Greedy(reference), Sampler.Greedy(batched));
    }

    /// <summary>
    /// A prompt shorter than one chunk must still agree — this is the boundary where the batched
    /// path takes a single ragged chunk and does the whole prompt in one trunk pass.
    /// </summary>
    [Fact]
    public void BatchedPrefill_ShortPromptBelowOneChunk_MatchesPerToken()
    {
        var path = FindModelPath();
        if (path is null) return;
        using var gpu = TryCreateBackend();
        if (gpu is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        int[] shortPrompt = [1, 2, 3, 5, 7];

        float[] reference, batched;
        using (var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
        {
            if (!fwd.CanBatchedTrunk) return;
            fwd.DisableBatchedPrefill = true;
            reference = fwd.Prefill(shortPrompt).ToArray();
        }
        using (var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 512))
            batched = fwd.Prefill(shortPrompt).ToArray();

        double maxAbs = 0;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(reference[i] - batched[i]));
        Assert.True(maxAbs < ActivePathTolerance,
            $"max abs logit delta {maxAbs:R} (tolerance {ActivePathTolerance}, "
            + $"path {VulkanMatMulPathConfig.Current})");
        Assert.Equal(Sampler.Greedy(reference), Sampler.Greedy(batched));
    }

    /// <summary>
    /// Prefill must leave the KV cache in a state that continued decoding reads correctly — a
    /// batched append that wrote the right logits but the wrong cache slots would pass the logit
    /// comparison above and then produce garbage on the next token.
    /// </summary>
    [Fact]
    public void BatchedPrefill_ThenDecode_MatchesPerTokenPrefillThenDecode()
    {
        var path = FindModelPath();
        if (path is null) return;
        using var gpu = TryCreateBackend();
        if (gpu is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        static int[] DecodeAfterPrefill(GgufModel m, VulkanBackend g, ModelHyperparams h, bool disableBatched)
        {
            using var fwd = new GpuForwardPass(m, g, h, maxContextLength: 512);
            fwd.DisableBatchedPrefill = disableBatched;
            var logits = fwd.Prefill(Prompt);
            int next = Sampler.Greedy(logits);
            var produced = new int[6];
            for (int i = 0; i < produced.Length; i++)
            {
                produced[i] = next;
                next = Sampler.Greedy(fwd.Forward(next, Prompt.Length + i));
            }
            return produced;
        }

        int[] reference = DecodeAfterPrefill(model, gpu, hp, disableBatched: true);
        int[] batched = DecodeAfterPrefill(model, gpu, hp, disableBatched: false);

        Assert.Equal(reference, batched);
    }
}
