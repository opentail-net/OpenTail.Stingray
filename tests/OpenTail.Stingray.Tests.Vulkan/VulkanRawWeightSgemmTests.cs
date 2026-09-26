namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// <c>Shaders.SgemmBf16W</c> / <c>Shaders.SgemmFp8W</c> (fp32 activations x raw bf16 / fp8-E4M3 weights) against a double-precision CPU
/// reference on the same bf16-rounded weights. bf16 widens to fp32 exactly, so the only error is
/// fp32 accumulation order. Covers the vec4 path (K % 4 == 0) and the scalar/odd-K tail, plus
/// M/N that are not tile multiples.
/// </summary>
public sealed class VulkanRawWeightSgemmTests : HeavyTestBase
{
    [Theory]
    [InlineData(37, 256, 300)]
    [InlineData(130, 1024, 520)]
    [InlineData(5, 67, 19)]
    public void SgemmBf16W_MatchesCpuReference(int M, int K, int N)
    {
        global::OpenTail.Stingray.Vulkan.VulkanBackend backend;
        try { backend = new global::OpenTail.Stingray.Vulkan.VulkanBackend(); }
        catch (Exception ex) { Assert.Skip($"Vulkan device could not be created: {ex.Message}"); throw; }
        using var _ = backend;

        var rng = new Random(M * 7 + K * 13 + N);
        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        var bBits = new ushort[N * K];
        var bF = new float[N * K];
        for (int i = 0; i < bBits.Length; i++)
        {
            float w = (float)((rng.NextDouble() * 2 - 1) * (i % 101 == 0 ? 300.0 : 0.05));
            bBits[i] = (ushort)(BitConverter.SingleToUInt32Bits(w) >> 16);
            bF[i] = BitConverter.UInt32BitsToSingle((uint)bBits[i] << 16);
        }

        var ta = backend.Upload(a, TensorShape.D2(M, K));
        var tb = backend.UploadBf16(bBits, TensorShape.D2(N, K));
        var tc = backend.Allocate(TensorShape.D2(M, N));
        backend.Sgemm(tc, ta, tb, M, K, N);
        var got = new float[M * N];
        backend.Download(tc, got);

        double worst = 0, scale = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double r = 0, mag = 0;
                for (int k = 0; k < K; k++) { double t = (double)a[m * K + k] * bF[n * K + k]; r += t; mag += Math.Abs(t); }
                worst = Math.Max(worst, Math.Abs(got[m * N + n] - r) / (mag + 1e-30));
                scale = Math.Max(scale, Math.Abs(r));
            }
        Assert.True(scale > 0);
        Assert.True(worst < 1e-5, $"worst error relative to sum|a*b|: {worst}");
    }

    private static float E4M3(byte b)
    {
        int e = (b >> 3) & 0xF, m = b & 7;
        float v = e == 0 ? m / 512f : (1 + m / 8f) * MathF.Pow(2, e - 7);
        return (b & 0x80) != 0 ? -v : v;
    }

    [Theory]
    [InlineData(37, 256, 300)]
    [InlineData(130, 1024, 520)]
    [InlineData(5, 67, 19)]
    public void SgemmFp8W_MatchesCpuReference(int M, int K, int N)
    {
        global::OpenTail.Stingray.Vulkan.VulkanBackend backend;
        try { backend = new global::OpenTail.Stingray.Vulkan.VulkanBackend(); }
        catch (Exception ex) { Assert.Skip($"Vulkan device could not be created: {ex.Message}"); throw; }
        using var _ = backend;

        var rng = new Random(M * 5 + K * 11 + N);
        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        // Every byte value except the two NaN encodings, so subnormals, zero, signs and the
        // largest exponents are all exercised.
        var bBytes = new byte[N * K];
        var bF = new float[N * K];
        for (int i = 0; i < bBytes.Length; i++)
        {
            byte v;
            do v = (byte)rng.Next(256); while ((v & 0x7F) == 0x7F);
            bBytes[i] = v; bF[i] = E4M3(v);
        }

        var ta = backend.Upload(a, TensorShape.D2(M, K));
        var tb = backend.UploadFp8(bBytes, TensorShape.D2(N, K));
        var tc = backend.Allocate(TensorShape.D2(M, N));
        backend.Sgemm(tc, ta, tb, M, K, N);
        var got = new float[M * N];
        backend.Download(tc, got);

        double worst = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                double r = 0, mag = 0;
                for (int k = 0; k < K; k++) { double t = (double)a[m * K + k] * bF[n * K + k]; r += t; mag += Math.Abs(t); }
                worst = Math.Max(worst, Math.Abs(got[m * N + n] - r) / (mag + 1e-30));
            }
        Assert.True(worst < 1e-5, $"worst error relative to sum|a*b|: {worst}");
    }
}
