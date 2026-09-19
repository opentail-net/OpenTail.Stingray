using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// GPU-vs-CPU parity check for the new <c>VulkanBackend.SiluGateMul</c> shader (docs/091's blocker
/// 2, 2026-09-19): FLUX.2's gated FFN activation, needed for the double-block GPU forward pass and
/// not previously implemented anywhere in this codebase's GPU op set (checked: FLUX.1's GPU MLP
/// only has plain GELU via <c>VisionGeluInPlace</c>, since FLUX.1's own MLP isn't gated).
///
/// Reference matches <c>Flux2DiT.cs</c>'s own CPU <c>GatedFfn</c> method exactly: given a
/// <c>[nTokens, 2*mlpHidden]</c> up-projection output, the FIRST half (`u1`) is SiLU'd (the gate),
/// the SECOND half (`u2`) is the value, and the output is <c>silu(u1) * u2</c>, `[nTokens,
/// mlpHidden]`. Confirmed against `examples/flux2/src/flux2/model.py`'s real `SiLUActivation`.
/// </summary>
public sealed class Flux2SiluGateMulGpuTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    [Fact]
    public void SiluGateMul_MatchesCpuGatedFfnReference()
    {
        using var backend = TryCreateVulkan();
        if (backend is null) return;

        const int nTokens = 37; // deliberately not a multiple of the workgroup size
        const int mlpHidden = 256;
        var rng = new Random(2091);

        var input = new float[nTokens * 2 * mlpHidden];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 6 - 3);

        // Real CPU reference: same math as Flux2DiT.cs's GatedFfn (first half = gate, second = value).
        var cpuOut = new float[nTokens * mlpHidden];
        for (int t = 0; t < nTokens; t++)
        {
            int inOff = t * 2 * mlpHidden;
            int outOff = t * mlpHidden;
            for (int c = 0; c < mlpHidden; c++)
            {
                float gate = input[inOff + c];
                float value = input[inOff + mlpHidden + c];
                cpuOut[outOff + c] = Silu(gate) * value;
            }
        }

        var inGpu = backend.Upload(input, TensorShape.D2(nTokens, 2 * mlpHidden));
        var outGpu = backend.Allocate(TensorShape.D2(nTokens, mlpHidden));

        try
        {
            backend.SiluGateMul(outGpu, inGpu, nTokens, mlpHidden);

            var gpuOut = new float[nTokens * mlpHidden];
            backend.Download(outGpu, gpuOut);

            double maxDiff = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(cpuOut[i] - gpuOut[i]));

            Console.WriteLine($"[Flux2 SiluGateMul] maxDiff={maxDiff:E4}");
            Assert.True(maxDiff < 1e-4, $"SiluGateMul GPU output diverges from CPU GatedFfn reference: maxDiff={maxDiff}");
        }
        finally
        {
            backend.Free(inGpu);
            backend.Free(outGpu);
        }
    }
}
