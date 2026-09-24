using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

// Vulkan dequant-in-shader GEMM (Shaders.SgemmQ3K / SgemmQ4K) vs the CPU's exact
// PackedSgemmF32.GemmQuant, on REAL Qwen Image block-quantized tensors. Also times the kernel
// against SgemmF16 on the same weights (FP16 upload), for the record.
public sealed unsafe class VulkanSgemmQuantParityTests
{
    private readonly ITestOutputHelper _out;
    public VulkanSgemmQuantParityTests(ITestOutputHelper o) => _out = o;

    private const string ModelPath = @"C:\Git-Public\OpenTail.Stingray\models\_models\qwen-image-Q3_K_S.gguf";

    [Theory]
    [InlineData("transformer_blocks.10.img_mlp.net.0.proj.weight", 271)] // Q3_K [12288 x 3072]
    [InlineData("transformer_blocks.10.img_mlp.net.2.weight", 271)]      // Q3_K [3072 x 12288]
    [InlineData("transformer_blocks.0.attn.to_q.weight", 271)]           // Q4_K [3072 x 3072]
    [InlineData("transformer_blocks.10.img_mod.1.weight", 1)]            // M=1 modulation shape
    public void SgemmQuant_MatchesCpuExactDequant(string tensor, int m)
    {
        Assert.SkipUnless(File.Exists(ModelPath), "qwen-image-Q3_K_S.gguf not found");
        using var w = GgufWeightLoader.Open(ModelPath);
        Assert.True(w.TryGetRaw(tensor, out nint data, out long byteLen, out var dt, out int n, out int k));

        var rng = new Random(9);
        var x = new float[m * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        var cpu = new float[m * n];
        fixed (float* px = x, pc = cpu)
            PackedSgemmF32.GemmQuant(pc, px, (byte*)data, dt, m, n, k);

        using var vk = new VulkanBackend();
        var raw = new ReadOnlySpan<byte>((void*)data, checked((int)byteLen));
        var bq = vk.UploadRaw(raw, TensorShape.D2(n, k), dt);
        var a = vk.Upload(x, TensorShape.D2(m, k), exact: true);
        var c = vk.Upload(new float[m * n], TensorShape.D2(m, n), exact: true);

        vk.Sgemm(c, a, bq, m, k, n); // warm-up (pipeline compile)
        var gpu = new float[m * n];
        vk.Download(c, gpu);
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++)
        {
            var sw = Stopwatch.StartNew();
            vk.Sgemm(c, a, bq, m, k, n);
            vk.Download(c, gpu);
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }

        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++) { double d = gpu[i] - cpu[i]; num += d * d; den += (double)cpu[i] * cpu[i]; }
        double rel = Math.Sqrt(num / den);

        // Same weights as FP16 through SgemmF16, for timing context.
        var f32 = w.ReadF32(tensor);
        var half = new Half[f32.Length];
        for (int i = 0; i < f32.Length; i++) half[i] = (Half)f32[i];
        var bh = vk.UploadHalf(half, TensorShape.D2(n, k));
        vk.Sgemm(c, a, bh, m, k, n);
        vk.Download(c, new float[m * n]);
        double bestF16 = 1e9;
        for (int rep = 0; rep < 3; rep++)
        {
            var sw = Stopwatch.StartNew();
            vk.Sgemm(c, a, bh, m, k, n);
            vk.Download(c, new float[m * n]);
            bestF16 = Math.Min(bestF16, sw.Elapsed.TotalMilliseconds);
        }

        string msg = $"{tensor} {dt} [{n}x{k}] m={m}: relErr vs CPU {rel:E2}; quant {best:F1}ms ({byteLen / 1048576.0:F1} MiB) vs F16 {bestF16:F1}ms ({half.Length * 2 / 1048576.0:F1} MiB)";
        _out.WriteLine(msg);
        Console.WriteLine(msg);
        vk.Free(bq); vk.Free(bh); vk.Free(a); vk.Free(c);

        Assert.True(rel < 1e-4, $"relErr {rel:E2}");
    }

    private const string Flux2Path = @"C:\Git-Public\OpenTail.Stingray\models\_models\flux2-dev-Q4_K_S.gguf";

    // FLUX.2's MLP down-projection runs through the fused SgemmSiluGate: A is [M, 2K] (gate | value)
    // and the GEMM input is silu(gate) * value.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Flux2Q4K_SgemmAndSiluGate_MatchCpu(bool siluGate)
    {
        Assert.SkipUnless(File.Exists(Flux2Path), "flux2-dev-Q4_K_S.gguf not found");
        using var w = GgufWeightLoader.Open(Flux2Path);
        string tensor = siluGate ? "double_blocks.0.img_mlp.2.weight" : "double_blocks.0.img_attn.qkv.weight";
        Assert.True(w.TryGetRaw(tensor, out nint data, out long byteLen, out var dt, out int n, out int k));
        Assert.Equal(DType.Q4_K, dt);
        const int m = 300;
        int aCols = siluGate ? 2 * k : k;
        var rng = new Random(4);
        var a = new float[m * aCols];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        var gemmIn = a;
        if (siluGate)
        {
            gemmIn = new float[m * k];
            for (int r = 0; r < m; r++)
                for (int j = 0; j < k; j++)
                {
                    float g = a[r * aCols + j], v = a[r * aCols + k + j];
                    gemmIn[r * k + j] = g / (1f + MathF.Exp(-g)) * v;
                }
        }
        var cpu = new float[m * n];
        fixed (float* px = gemmIn, pc = cpu)
            PackedSgemmF32.GemmQuant(pc, px, (byte*)data, dt, m, n, k);

        using var vk = new VulkanBackend();
        var bq = vk.UploadRaw(new ReadOnlySpan<byte>((void*)data, checked((int)byteLen)), TensorShape.D2(n, k), dt);
        var ag = vk.Upload(a, TensorShape.D2(m, aCols), exact: true);
        var c = vk.Upload(new float[m * n], TensorShape.D2(m, n), exact: true);
        if (siluGate) vk.SgemmSiluGate(c, ag, bq, m, k, n); else vk.Sgemm(c, ag, bq, m, k, n);
        var gpu = new float[m * n];
        vk.Download(c, gpu);
        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++) { double d = gpu[i] - cpu[i]; num += d * d; den += (double)cpu[i] * cpu[i]; }
        double rel = Math.Sqrt(num / den);
        string msg = $"FLUX.2 {tensor} [{n}x{k}] m={m} siluGate={siluGate}: relErr vs CPU {rel:E2}";
        _out.WriteLine(msg);
        Console.WriteLine(msg);
        vk.Free(bq); vk.Free(ag); vk.Free(c);
        Assert.True(rel < 1e-4, $"relErr {rel:E2}");
    }
}
