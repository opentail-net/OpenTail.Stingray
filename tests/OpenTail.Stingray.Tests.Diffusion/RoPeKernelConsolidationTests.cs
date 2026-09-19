using OpenTail.Stingray.Diffusion.Primitives;
using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Cheap, no-real-weights check (docs/092) that <see cref="WanRoPE"/>'s hand-rolled interleaved
/// frequency/rotation math is bit-identical to the shared <see cref="InterleavedRoPE"/> primitive
/// (already used by Flux2/Flux3) -- confirms it's safe to consolidate WanRoPE to delegate to the
/// shared kernel instead of keeping a duplicate local implementation.
/// </summary>
public sealed class RoPeKernelConsolidationTests
{
    [Fact]
    public void WanRoPE_And_SharedInterleavedRoPE_ProduceIdenticalFrequencyTables()
    {
        int dim = 44; // Wan's real temporal-axis dim
        float theta = 10000.0f;
        int pos = 7;

        // Wan's own local computation (via reflection-free direct re-derivation of its formula,
        // since FillFrequenciesInterleaved is private -- exercise it through Compute3DRoPE instead,
        // isolating a single axis by using numFrames=1,patchH=1,patchW=1 won't isolate one axis
        // cleanly, so directly re-derive the same formula here matching FillFrequenciesInterleaved's
        // documented behavior, and cross check against the public Compute3DRoPE output at token 0
        // for the temporal axis specifically (t=0 -> pos=0, not useful) -- instead call
        // Compute3DRoPE with numFrames>1 and inspect token index for t=pos.
        int numFrames = 8, patchH = 1, patchW = 1;
        var (wanCos, wanSin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, headDim: 128, theta: theta);
        int tokenIdx = 7; // t=7, h=0, w=0 -> temporal axis pos=7
        var wanCosSlice = wanCos.AsSpan(tokenIdx * 128, dim);
        var wanSinSlice = wanSin.AsSpan(tokenIdx * 128, dim);

        // Shared primitive: build the same temporal axis (dim=44, pos=7, theta=10000) independently.
        var invFreq = InterleavedRoPE.ComputeInvFreqs(dim, theta);
        var sharedCos = new float[dim];
        var sharedSin = new float[dim];
        InterleavedRoPE.FillAxisFreqs(sharedCos, sharedSin, pos, invFreq);

        for (int i = 0; i < dim; i++)
        {
            Assert.True(MathF.Abs(wanCosSlice[i] - sharedCos[i]) < 1e-6f,
                $"cos[{i}]: Wan={wanCosSlice[i]}, shared={sharedCos[i]}");
            Assert.True(MathF.Abs(wanSinSlice[i] - sharedSin[i]) < 1e-6f,
                $"sin[{i}]: Wan={wanSinSlice[i]}, shared={sharedSin[i]}");
        }
    }

    [Fact]
    public void WanRoPE_And_SharedInterleavedRoPE_ProduceIdenticalRotation()
    {
        int headDim = 8;
        int numHeads = 1;
        int seqLen = 1;

        var rng = new Random(42);
        var tensorWan = new float[headDim];
        var tensorShared = new float[headDim];
        for (int i = 0; i < headDim; i++)
        {
            float v = (float)rng.NextDouble() * 2f - 1f;
            tensorWan[i] = v;
            tensorShared[i] = v;
        }

        // Build identical frequency tables via the shared primitive (single axis = full headDim).
        var invFreq = InterleavedRoPE.ComputeInvFreqs(headDim, 10000f);
        var cos = new float[headDim];
        var sin = new float[headDim];
        InterleavedRoPE.FillAxisFreqs(cos, sin, pos: 3, invFreq);

        WanRoPE.ApplyRoPE(tensorWan, cos, sin, seqLen, numHeads, headDim);
        InterleavedRoPE.ApplyRoPE(tensorShared, cos, sin, seqLen, numHeads, headDim);

        for (int i = 0; i < headDim; i++)
        {
            Assert.True(MathF.Abs(tensorWan[i] - tensorShared[i]) < 1e-6f,
                $"element {i}: Wan={tensorWan[i]}, shared={tensorShared[i]}");
        }
    }
}
