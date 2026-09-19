using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Isolated GPU-vs-CPU parity check for <c>VulkanBackend.AdaLNModulate</c>'s <c>isRmsNorm: false</c>
/// (affine-free LayerNorm) branch -- docs/091's blocker 3, 2026-09-19. Every existing GPU caller
/// (FLUX.1's <c>DoubleBlockGpu</c>/<c>SingleBlockGpu</c>, and <c>FluxGpuParityTests.
/// AdaLNModulate_MatchesCpuReference</c>) exercises ONLY <c>isRmsNorm: true</c> (RMSNorm), because
/// FLUX.1's real AdaLN-Zero uses RMSNorm-style modulation. FLUX.2's real CPU path
/// (<c>Flux2DiT.ModulateCopy</c>) uses <c>DiffusionOps.LayerNormNoAffine</c> -- mean-subtracted,
/// affine-free LayerNorm, NOT RMSNorm -- which requires the shader's <c>isRmsNorm: false</c> branch
/// (<c>Shaders.AdaLNModulate</c>'s <c>else</c> block). That branch has never been exercised by any
/// test in this codebase before this one. This closes that gap BEFORE any FLUX.2 double-block GPU
/// forward pass is built on top of it (per docs/091's revised implementation order).
/// </summary>
public sealed class Flux2AdaLNModulateLayerNormGpuTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    [Fact]
    public void AdaLNModulate_IsRmsNormFalse_MatchesCpuLayerNormReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int nTokens = 64;
        const int dim = 128;
        var rng = new Random(2026);

        var input = new float[nTokens * dim];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var shift = new float[dim];
        var scale = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            shift[i] = (float)(rng.NextDouble() * 2 - 1);
            scale[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Real CPU reference: DiffusionOps.AdaLNModulate's isRmsNorm=false branch (mean-subtracted,
        // affine-free LayerNorm -- the exact math FLUX.2's ModulateCopy uses).
        var cpuOut = new float[nTokens * dim];
        DiffusionOps.AdaLNModulate(cpuOut, input, shift, scale, nTokens, dim, isRmsNorm: false, eps: 1e-6f);

        var inGpu = backend.Upload(input, TensorShape.D2(nTokens, dim));
        var shiftGpu = backend.Upload(shift, TensorShape.D1(dim));
        var scaleGpu = backend.Upload(scale, TensorShape.D1(dim));
        var outGpu = backend.Allocate(TensorShape.D2(nTokens, dim));

        try
        {
            backend.AdaLNModulate(outGpu, inGpu, shiftGpu, scaleGpu, nTokens, dim, isRmsNorm: false, eps: 1e-6f);

            var gpuOut = new float[nTokens * dim];
            backend.Download(outGpu, gpuOut);

            double maxDiff = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(cpuOut[i] - gpuOut[i]));

            Console.WriteLine($"[Flux2 AdaLNModulate isRmsNorm=false] maxDiff={maxDiff:E4}");
            Assert.True(maxDiff < 1e-3, $"isRmsNorm=false GPU branch diverges from CPU LayerNorm reference: maxDiff={maxDiff}");
        }
        finally
        {
            backend.Free(inGpu);
            backend.Free(shiftGpu);
            backend.Free(scaleGpu);
            backend.Free(outGpu);
        }
    }

    /// <summary>Sanity check that isRmsNorm=true and isRmsNorm=false genuinely diverge on the same
    /// non-trivial input (i.e. the shader branch is actually selected by the push constant, not a
    /// no-op/dead branch) -- guards against a trivial "both branches happen to agree" false pass.</summary>
    [Fact]
    public void AdaLNModulate_RmsNormAndLayerNormBranches_GenuinelyDiffer()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int nTokens = 8;
        const int dim = 32;
        var rng = new Random(7);

        // A nonzero-mean input is required to distinguish LayerNorm (mean-subtracted) from RMSNorm
        // (not mean-subtracted) -- a zero-mean input would make both branches agree trivially.
        var input = new float[nTokens * dim];
        for (int i = 0; i < input.Length; i++) input[i] = 5.0f + (float)(rng.NextDouble() * 2 - 1);

        var shift = new float[dim];
        var scale = new float[dim];
        for (int i = 0; i < dim; i++) scale[i] = 0.1f;

        var inGpu = backend.Upload(input, TensorShape.D2(nTokens, dim));
        var shiftGpu = backend.Upload(shift, TensorShape.D1(dim));
        var scaleGpu = backend.Upload(scale, TensorShape.D1(dim));
        var outRmsGpu = backend.Allocate(TensorShape.D2(nTokens, dim));
        var outLnGpu = backend.Allocate(TensorShape.D2(nTokens, dim));

        try
        {
            backend.AdaLNModulate(outRmsGpu, inGpu, shiftGpu, scaleGpu, nTokens, dim, isRmsNorm: true, eps: 1e-6f);
            backend.AdaLNModulate(outLnGpu, inGpu, shiftGpu, scaleGpu, nTokens, dim, isRmsNorm: false, eps: 1e-6f);

            var rmsOut = new float[nTokens * dim];
            var lnOut = new float[nTokens * dim];
            backend.Download(outRmsGpu, rmsOut);
            backend.Download(outLnGpu, lnOut);

            double maxDiff = 0;
            for (int i = 0; i < rmsOut.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(rmsOut[i] - lnOut[i]));

            Console.WriteLine($"[Flux2 AdaLNModulate branch divergence] maxDiff={maxDiff:E4}");
            Assert.True(maxDiff > 0.1, $"RMSNorm and LayerNorm branches should genuinely differ on a nonzero-mean input, but maxDiff={maxDiff}");
        }
        finally
        {
            backend.Free(inGpu);
            backend.Free(shiftGpu);
            backend.Free(scaleGpu);
            backend.Free(outRmsGpu);
            backend.Free(outLnGpu);
        }
    }
}
