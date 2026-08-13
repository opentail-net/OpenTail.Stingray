using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Pins the GLSL <c>tanh</c> overflow that silently killed Gemma 4 on Vulkan.
///
/// <para>GLSL spec-defines <c>tanh(x)</c> as <c>(e^x - e^-x)/(e^x + e^-x)</c> and drivers
/// implement it literally. float32 <c>exp</c> overflows past ~88, so an argument above ~44
/// evaluates <c>inf/inf = NaN</c>. Gemma 4 E4B's wide FFN produced a single gate value of
/// g = 20.31, giving a tanh argument of 0.7978*(g + 0.044715*g^3) = 315.1 — <b>one</b> NaN in a
/// 10240-wide activation. The next op is <c>ffn_down</c>, a matmul, so that one element
/// contaminated all 2560 output rows and rode the residual through all 42 layers, producing
/// <c>&lt;pad&gt;</c> output with no error anywhere.</para>
///
/// <para>The CPU kernel (<see cref="SimdKernels.GeluTanhMul"/>) already clamped for exactly this
/// reason; the Vulkan shader had not been updated to match. These tests are the cross-backend
/// guard that keeps the two from drifting apart again — and they are deliberately kernel-level
/// and model-free, so they run on any machine with a Vulkan device instead of gating on a 4 GB
/// GGUF that most checkouts do not have.</para>
///
/// <para>Silent-skip when no Vulkan device is present, matching the other Vulkan suites.</para>
/// </summary>
public sealed class VulkanTanhOverflowTests
{
    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    /// <summary>
    /// The exact value observed in Gemma 4 E4B layer 0, plus a spread around it. Anything above
    /// roughly g = 10.75 overflows the unclamped formula, so the small values in this list are
    /// the control: they must be unaffected by the clamp.
    /// </summary>
    private static readonly float[] GateValues =
        [0f, 0.5f, -0.5f, 1f, -1f, 3f, -3f, 10f, -10f, 20.309921f, -20.309921f, 50f, -50f, 200f, -200f];

    [Fact]
    public void GeluTanhMul_LargeGate_StaysFinite()
    {
        using var gpu = TryCreate();
        // Assert.Skip, not `return`: a silent early return is indistinguishable from a pass in the
        // runner summary, which is how the Gemma4 Vulkan defect this file guards went unnoticed for
        // as long as the GGUF was absent. A skip is reported as a skip.
        if (gpu is null) Assert.Skip("no Vulkan device available");

        var gate = GateValues;
        var up = new float[gate.Length];
        for (int i = 0; i < up.Length; i++) up[i] = 1f;

        var gGpu = gpu.Upload(gate, TensorShape.D1(gate.Length));
        var uGpu = gpu.Upload(up, TensorShape.D1(up.Length));
        gpu.GeluTanhMul(gGpu, uGpu);

        var got = new float[gate.Length];
        gpu.Download(gGpu, got);

        for (int i = 0; i < got.Length; i++)
            Assert.True(float.IsFinite(got[i]),
                $"GeluTanhMul produced {got[i]} for gate={gate[i]} (tanh arg "
                + $"{0.7978845608028654 * (gate[i] + 0.044715 * Math.Pow(gate[i], 3)):F1}). "
                + "The GLSL tanh argument is not clamped — see Shaders.GeluTanhMul.");
    }

    [Fact]
    public void GeluTanhMul_MatchesCpuKernel()
    {
        using var gpu = TryCreate();
        // Assert.Skip, not `return`: a silent early return is indistinguishable from a pass in the
        // runner summary, which is how the Gemma4 Vulkan defect this file guards went unnoticed for
        // as long as the GGUF was absent. A skip is reported as a skip.
        if (gpu is null) Assert.Skip("no Vulkan device available");

        var gate = GateValues;
        var up = new float[gate.Length];
        for (int i = 0; i < up.Length; i++) up[i] = (i % 2 == 0) ? 1.5f : -26.330753f;

        var gGpu = gpu.Upload(gate, TensorShape.D1(gate.Length));
        var uGpu = gpu.Upload(up, TensorShape.D1(up.Length));
        gpu.GeluTanhMul(gGpu, uGpu);
        var got = new float[gate.Length];
        gpu.Download(gGpu, got);

        // Independent oracle: the CPU kernel, which has carried the clamp all along.
        var expected = new float[gate.Length];
        unsafe
        {
            fixed (float* g = gate, u = up, o = expected)
                SimdKernels.GeluTanhMul(g, u, o, gate.Length);
        }

        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])),
                $"gate={gate[i]} up={up[i]}: Vulkan={got[i]} CPU={expected[i]}");
    }

    /// <summary>
    /// <c>Softcap</c> carries the identical unguarded <c>tanh(x / cap)</c>. Gemma is the only user
    /// of both kernels, so a logit large enough to overflow it would fail the same silent way.
    /// This was fixed on inspection rather than on a reproduction, which is precisely why it needs
    /// a test of its own.
    /// </summary>
    [Fact]
    public void SoftcapInPlace_ExtremeLogits_StayFiniteAndBounded()
    {
        using var gpu = TryCreate();
        // Assert.Skip, not `return`: a silent early return is indistinguishable from a pass in the
        // runner summary, which is how the Gemma4 Vulkan defect this file guards went unnoticed for
        // as long as the GGUF was absent. A skip is reported as a skip.
        if (gpu is null) Assert.Skip("no Vulkan device available");

        const float cap = 30f;
        float[] logits = [0f, 1f, -1f, 30f, -30f, 1000f, -1000f, 1e6f, -1e6f, 3e38f, -3e38f];

        var xGpu = gpu.Upload(logits, TensorShape.D1(logits.Length));
        gpu.SoftcapInPlace(xGpu, cap);
        var got = new float[logits.Length];
        gpu.Download(xGpu, got);

        for (int i = 0; i < got.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]),
                $"SoftcapInPlace produced {got[i]} for logit {logits[i]} (cap {cap}).");
            // tanh is bounded by ±1, so the capped value can never leave [-cap, cap].
            Assert.True(Math.Abs(got[i]) <= cap + 1e-3f,
                $"SoftcapInPlace produced {got[i]} for logit {logits[i]}, outside +/-{cap}.");
            Assert.True(Math.Sign(got[i]) == Math.Sign(logits[i]) || logits[i] == 0f,
                $"SoftcapInPlace flipped the sign of {logits[i]} to {got[i]}.");
        }
    }
}
