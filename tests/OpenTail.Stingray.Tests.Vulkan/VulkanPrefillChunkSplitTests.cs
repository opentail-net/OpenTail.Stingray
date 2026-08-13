using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// The batched-prefill chunk size used to be <c>MaxBatchVerifyK</c>, the same constant that bounds
/// speculative-decode draft length. They are now separate: prefill follows the active matmul path's
/// per-dispatch cap, verify keeps its own.
/// </summary>
/// <remarks>
/// <para>The split is a prerequisite for a wider Path 2 tile ever showing a win — a tiled GEMM
/// clamped to 16 tokens would be measured on exactly the workload Path 1 already handles well and
/// would correctly look like nothing. It is also a hazard: changing the chunk size changes how a
/// prompt is partitioned, and <b>a kernel selected by a size threshold is how the CPU side broke
/// chunked-vs-unchunked prefill parity</b>. So the load-bearing test here is not that the constants
/// are separate, it is <see cref="Prefill_IsChunkSizeIndependent"/>.</para>
/// </remarks>
public sealed class VulkanPrefillChunkSplitTests
{
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

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static int[] BuildPrompt(int n)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 5 + (i * 7) % 2000;
        return t;
    }

    // ── The split itself ─────────────────────────────────────────────────────

    /// <summary>
    /// Both paths cap at 16 today, so the split is behaviour-neutral. That is deliberate: a split
    /// that also changed the chunk size would confound the measurement it exists to enable.
    /// </summary>
    [Fact]
    public void BothPaths_StillCapAt16_SoTheSplitChangesNothingYet()
    {
        Assert.Equal(16, VulkanMatMulPathConfig.Path1MaxTokensPerDispatch);
        Assert.Equal(16, VulkanMatMulPathConfig.Path2MaxTokensPerDispatch);
    }

    [Fact]
    public void MaxTokensPerDispatch_FollowsTheActivePath()
    {
        var prior = VulkanMatMulPathConfig.Current;
        try
        {
            VulkanMatMulPathConfig.Current = VulkanMatMulPath.Path1;
            Assert.Equal(VulkanMatMulPathConfig.Path1MaxTokensPerDispatch, VulkanMatMulPathConfig.MaxTokensPerDispatch);
            VulkanMatMulPathConfig.Current = VulkanMatMulPath.Path2;
            Assert.Equal(VulkanMatMulPathConfig.Path2MaxTokensPerDispatch, VulkanMatMulPathConfig.MaxTokensPerDispatch);
        }
        finally { VulkanMatMulPathConfig.Current = prior; }
    }

    /// <summary>
    /// A Path 2 kernel must never advertise more tokens per dispatch than
    /// <c>VulkanBackend.MatMulBatched</c> accepts, because Path 2 is allowed to decline any shape
    /// and fall through to Path 1 — which would then be handed a chunk it throws on. Whoever raises
    /// <c>Path2MaxTokensPerDispatch</c> must relax that range check in the same change.
    /// </summary>
    [Fact]
    public void Path2Cap_CannotExceed_WhatPath1CanFallBackTo()
    {
        Assert.True(
            VulkanMatMulPathConfig.Path2MaxTokensPerDispatch <= VulkanMatMulPathConfig.Path1MaxTokensPerDispatch,
            "Path 2 declines fall through to Path 1, so its cap must stay within Path 1's until "
            + "MatMulBatched's nTok range check and the MatVecBatched shaders' MAX_NTOK are raised too.");
    }

    // ── The hazard the split introduces ──────────────────────────────────────

    /// <summary>
    /// Prefill must return the same logits regardless of how the prompt was partitioned. Each
    /// chunk is mathematically independent — every token's accumulator is its own, and Q8 activation
    /// quantization is per-token — so this should hold exactly, and any drift means a kernel is
    /// keyed on batch size somewhere.
    /// </summary>
    /// <remarks>
    /// 5 and 7 are deliberately chosen not to divide the 96-token prompt, so the final short chunk
    /// (which forces a scratch reallocation at a different k) is exercised at several sizes.
    /// </remarks>
    [Theory]
    [InlineData(16)]   // the cap, and the default
    [InlineData(8)]
    [InlineData(7)]
    [InlineData(5)]
    [InlineData(1)]    // degenerate: one token per chunk, still through the batched trunk
    public void Prefill_IsChunkSizeIndependent(int chunkSize)
    {
        var path = FindModelPath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");
        using var gpu = TryCreate();
        Assert.SkipUnless(gpu is not null, "no usable GPU backend in this environment");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        int[] prompt = BuildPrompt(96);

        static float[] Run(GgufModel m, VulkanBackend g, ModelHyperparams h, int[] prompt, int chunk)
        {
            using var fwd = new GpuForwardPass(m, g, h, maxContextLength: 512);
            if (!fwd.CanBatchedTrunk) return [];
            fwd.PrefillChunkTokens = chunk;
            return fwd.Prefill(prompt).ToArray();
        }

        float[] reference = Run(model, gpu, hp, prompt, 0);   // 0 = "whatever the path allows"
        float[] chunked = Run(model, gpu, hp, prompt, chunkSize);
        if (reference.Length == 0) return;   // model not batched-trunk eligible on this device

        Assert.Equal(reference.Length, chunked.Length);
        for (int i = 0; i < reference.Length; i++)
            Assert.True(reference[i] == chunked[i],
                $"logit[{i}] diverged at chunk={chunkSize}: {reference[i]} vs {chunked[i]}");
    }

    /// <summary>
    /// <c>PrefillChunkTokens = 0</c> means "the active path's cap", and an over-large request is
    /// clamped to it rather than reaching <c>MatMulBatched</c> and throwing.
    /// </summary>
    [Fact]
    public void OversizedChunkRequest_IsClampedNotThrown()
    {
        var path = FindModelPath();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");
        using var gpu = TryCreate();
        Assert.SkipUnless(gpu is not null, "no usable GPU backend in this environment");

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        int[] prompt = BuildPrompt(40);

        using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 512);
        if (!fwd.CanBatchedTrunk) return;

        fwd.PrefillChunkTokens = 4096;   // far beyond any kernel's MAX_NTOK
        var logits = fwd.Prefill(prompt).ToArray();
        Assert.Equal(hp.VocabSize, logits.Length);
        Assert.All(logits, v => Assert.False(float.IsNaN(v)));
    }
}
