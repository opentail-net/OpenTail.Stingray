using System.Diagnostics;
using OpenTail.Stingray.Diffusion.QwenImage;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): Qwen Image 256^2 generate with real conditioning. Env: ZZ_QI=1,
// ZZ_STEPS (8), ZZ_GPU=1 for Vulkan, ZZ_OUT (dir), ZZ_TAG (file suffix).
public sealed class ZzQwenImageProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzQwenImageProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public unsafe void QuantGemmMicrobench()
    {
        if (Environment.GetEnvironmentVariable("ZZ_QIBENCH") != "1") return;
        string path = Environment.GetEnvironmentVariable("ZZ_GGUF") ?? @"C:\Git-Public\OpenTail.Stingray\models\_models\qwen-image-Q3_K_S.gguf";
        int m = int.TryParse(Environment.GetEnvironmentVariable("ZZ_M"), out var mm) ? mm : 271;
        using var w = OpenTail.Stingray.Diffusion.GgufWeightLoader.Open(path);
        var names = (Environment.GetEnvironmentVariable("ZZ_TENSORS") ?? "transformer_blocks.0.attn.to_q.weight,transformer_blocks.0.img_mlp.net.0.proj.weight,transformer_blocks.0.img_mlp.net.2.weight,transformer_blocks.0.img_mod.1.weight").Split(',');
        var rng = new Random(3);
        foreach (var name in names)
        {
            if (!w.TryGetRaw(name, out nint data, out _, out var dt, out int rows, out int cols)) { Log($"{name}: no raw"); continue; }
            var x = new float[m * cols];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            var wf = w.ReadF32(name);
            var oRef = new float[m * rows]; var oA = new float[m * rows]; var oB = new float[m * rows]; var oC = new float[m * rows];
            double tA = 1e9, tB = 1e9, tC = 1e9;
            fixed (float* px = x, pw = wf, pr = oRef, pa = oA, pb = oB, pc = oC)
            {
                float* packed = OpenTail.Stingray.Cpu.PackedSgemmF32.PackWeights(pw, rows, cols);
                OpenTail.Stingray.Cpu.PackedSgemmF32.Gemm(pr, px, packed, null, m, rows, cols);
                System.Runtime.InteropServices.NativeMemory.AlignedFree(packed);
                for (int rep = 0; rep < 3; rep++)
                {
                    var sw = Stopwatch.StartNew();
                    OpenTail.Stingray.Cpu.SimdKernels.MatMulBatched(pa, (byte*)data, px, m, rows, cols, dt, allowQ8: true);
                    tA = Math.Min(tA, sw.Elapsed.TotalSeconds); sw.Restart();
                    OpenTail.Stingray.Cpu.SimdKernels.MatMulBatched(pb, (byte*)data, px, m, rows, cols, dt, allowQ8: false);
                    tB = Math.Min(tB, sw.Elapsed.TotalSeconds); sw.Restart();
                    OpenTail.Stingray.Cpu.PackedSgemmF32.GemmQuant(pc, px, (byte*)data, dt, m, rows, cols);
                    tC = Math.Min(tC, sw.Elapsed.TotalSeconds);
                }
            }
            float Rel(float[] o) { double num = 0, den = 0; for (int i = 0; i < o.Length; i++) { double d = o[i] - oRef[i]; num += d * d; den += (double)oRef[i] * oRef[i]; } return (float)Math.Sqrt(num / den); }
            double gf = 2.0 * m * rows * cols / 1e9;
            Log($"{name} {dt} [{rows}x{cols}] m={m}: q8path {tA * 1e3:F0}ms ({gf / tA:F0} GF/s, relErr {Rel(oA):E1}) | f32dot {tB * 1e3:F0}ms ({gf / tB:F0}, {Rel(oB):E1}) | gemmQuant {tC * 1e3:F0}ms ({gf / tC:F0}, {Rel(oC):E1})");
        }
    }

    // Our Vulkan GEMMs at ggml test-backend-ops' own perf shape (m=4096 out, n=512 tokens, k=14336).
    [Fact]
    public unsafe void VulkanGemmVsGgmlShape()
    {
        if (Environment.GetEnvironmentVariable("ZZ_VKBENCH") != "1") return;
        int outN = int.TryParse(Environment.GetEnvironmentVariable("ZZ_N"), out var nn) ? nn : 4096;
        int tokM = int.TryParse(Environment.GetEnvironmentVariable("ZZ_M"), out var mm) ? mm : 512;
        int kk = int.TryParse(Environment.GetEnvironmentVariable("ZZ_K"), out var k0) ? k0 : 14336;
        var rng = new Random(2);
        using var vk = new OpenTail.Stingray.Vulkan.VulkanBackend();
        var a = vk.Upload(Enumerable.Range(0, tokM * kk).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray(), TensorShape.D2(tokM, kk), exact: true);
        var c = vk.Upload(new float[tokM * outN], TensorShape.D2(tokM, outN), exact: true);
        var half = new Half[outN * kk];
        for (int i = 0; i < half.Length; i++) half[i] = (Half)((rng.NextDouble() - 0.5) * 0.05);
        var bh = vk.UploadHalf(half, TensorShape.D2(outN, kk));
        var q = new byte[outN * (kk / 256) * 144];
        rng.NextBytes(q);
        for (int b = 0; b < q.Length; b += 144) { q[b] = 0x00; q[b + 1] = 0x20; q[b + 2] = 0x00; q[b + 3] = 0x20; } // d = dmin = 2^-7
        var bq = vk.UploadRaw(q, TensorShape.D2(outN, kk), DType.Q4_K);
        double flop = 2.0 * tokM * outN * kk;
        foreach (var (label, b) in new[] { ("F16", bh), ("Q4_K", bq) })
        {
            vk.Sgemm(c, a, b, tokM, kk, outN); vk.Download(c, new float[1]);
            double best = 1e9;
            for (int rep = 0; rep < 5; rep++)
            {
                var sw = Stopwatch.StartNew();
                vk.Sgemm(c, a, b, tokM, kk, outN);
                vk.Download(c, new float[1]);
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            Log($"[ZZ vk] ours {label} m(out)={outN} n(tok)={tokM} k={kk}: {best:F1} ms, {flop / best / 1e6:F0} GFLOP/s");
        }
    }

    // Our Vulkan attention at FLUX.1's shape (1280 tokens, 24 heads x 128).
    [Fact]
    public void VulkanAttentionFluxShape()
    {
        if (Environment.GetEnvironmentVariable("ZZ_VKATTN") != "1") return;
        int seq = 1280, heads = 24, hd = 128, d = heads * hd;
        var rng = new Random(3);
        using var vk = new OpenTail.Stingray.Vulkan.VulkanBackend();
        float[] R() => Enumerable.Range(0, seq * d).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var q = vk.Upload(R(), TensorShape.D2(seq, d), exact: true);
        var k = vk.Upload(R(), TensorShape.D2(seq, d), exact: true);
        var v = vk.Upload(R(), TensorShape.D2(seq, d), exact: true);
        var o = vk.Upload(new float[seq * d], TensorShape.D2(seq, d), exact: true);
        var ops = (IImageOpsBackend)vk;
        ops.MultiHeadAttentionTiled(o, q, k, v, seq, seq, heads, hd); vk.Download(o, new float[1]);
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++)
        {
            var sw = Stopwatch.StartNew();
            ops.MultiHeadAttentionTiled(o, q, k, v, seq, seq, heads, hd);
            vk.Download(o, new float[1]);
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        double flop = 4.0 * seq * seq * d;
        Log($"[ZZ vk] ours attention seq={seq} heads={heads} hd={hd}: {best:F1} ms, {flop / best / 1e6:F0} GFLOP/s; x57 blocks = {best * 57 / 1000:F1}s per step");
    }

    [Fact]
    public void QwenImageGenerate()
    {
        if (Environment.GetEnvironmentVariable("ZZ_QI") != "1") return;
        string m = @"C:\Git-Public\OpenTail.Stingray\models\_models";
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string tag = Environment.GetEnvironmentVariable("ZZ_TAG") ?? "run";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var s) ? s : 8;
        using var vk = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new OpenTail.Stingray.Vulkan.VulkanBackend() : null;
        var sw = Stopwatch.StartNew();
        using var pipe = QwenImagePipeline.Load(Path.Combine(m, "qwen-image-Q3_K_S.gguf"),
            Path.Combine(m, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"), Path.Combine(m, "qwen_image_vae.safetensors"), vk);
        Log($"[ZZ] load {sw.Elapsed.TotalSeconds:F1}s");
        sw.Restart();
        float cfg = float.TryParse(Environment.GetEnvironmentVariable("ZZ_CFG"), System.Globalization.CultureInfo.InvariantCulture, out var g) ? g : 4.0f;
        float[]? ctx = null;
        if (Environment.GetEnvironmentVariable("ZZ_COND") is { Length: > 0 } condPath)
        {
            var bytes = File.ReadAllBytes(condPath);
            ctx = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, ctx, 0, bytes.Length);
        }
        pipe.Generate(prompt: "a red apple on a wooden table", negativePrompt: "", width: 256, height: 256,
            steps: steps, guidance: cfg, seed: 42, outputPath: Path.Combine(outDir, $"zz_qi_{tag}.png"), textContext: ctx,
            progress: (i, n) => { Log($"[ZZ] step {i}/{n} at {sw.Elapsed.TotalSeconds:F1}s"); });
        Log($"[ZZ] generate {sw.Elapsed.TotalSeconds:F1}s");
    }
}
