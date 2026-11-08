using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Regression for the Qwen 3.8 Q5_K matvec numerics. llama.cpp quantises the activation to Q8_K for Q5_K matmuls
/// (ggml_vec_dot_q5_K_q8_K); Stingray's generic MatVec keeps f32 activations. Found 2026-11-08: the ssm_out
/// projection differed from llama.cpp by 2.2% until the Qwen path used the Q8_K kernel. Verified against llama.cpp's
/// own ssm_out-0 output (relative error 5.6e-7 with Q8_K activations, 2.2% with f32).
/// </summary>
public sealed class Qwen4ExpQ5KDispatchTests
{
    private const int Rows = 8;
    private const int Cols = 512; // two Q5_K super-blocks per row

    [Fact]
    public unsafe void MatVecLlamaNumerics_Q5K_UsesQ8KActivations()
    {
        var rng = new Random(1);
        byte[] weights = MakeQ5KWeights(rng, Rows, Cols);
        float[] input = new float[Cols];
        for (int i = 0; i < Cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] viaQwen = new float[Rows], viaQ8K = new float[Rows], viaF32 = new float[Rows];
        fixed (byte* w = weights)
        fixed (float* x = input, y1 = viaQwen, y2 = viaQ8K, y3 = viaF32)
        {
            Qwen4ExpForwardPass.MatVecLlamaNumerics(y1, w, x, Rows, Cols, DType.Q5_K);
            SimdKernels.MatVecQ5K_Q8K(y2, w, x, Rows, Cols);
            SimdKernels.MatVecQ5K(y3, w, x, Rows, Cols);
        }

        for (int r = 0; r < Rows; r++)
        {
            Assert.True(float.IsFinite(viaQwen[r]));
            Assert.Equal(viaQ8K[r], viaQwen[r], 6);
        }

        // The f32-activation kernel must differ, otherwise this test would not detect a dispatch regression.
        double maxDiff = 0;
        for (int r = 0; r < Rows; r++) maxDiff = Math.Max(maxDiff, Math.Abs(viaQ8K[r] - viaF32[r]));
        Assert.True(maxDiff > 1e-5, $"Q8_K and f32 Q5_K kernels agree (max diff {maxDiff}); the test cannot tell them apart.");
    }

    /// <summary>
    /// Builds Q5_K super-blocks (176 bytes each: d, dmin as fp16, 12 scale bytes, 32 high-bit bytes, 128 low-nibble
    /// bytes) with finite, small d and dmin so every value is well defined.
    /// </summary>
    private static byte[] MakeQ5KWeights(Random rng, int rows, int cols)
    {
        int blocksPerRow = cols / 256;
        byte[] bytes = new byte[rows * blocksPerRow * 176];
        rng.NextBytes(bytes);
        for (int b = 0; b < rows * blocksPerRow; b++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 176, 2), (Half)0.02f);
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 176 + 2, 2), (Half)0.01f);
        }
        return bytes;
    }
}
