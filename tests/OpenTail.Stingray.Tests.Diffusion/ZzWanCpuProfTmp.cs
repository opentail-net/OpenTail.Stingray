using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): Wan CPU GEMM microbench. Env ZZ_M = token count (default 1024).
public sealed unsafe class ZzWanCpuProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzWanCpuProfTmp(ITestOutputHelper o) => _out = o;

    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    // Full CPU pipeline (text encode on CPU too). Env: ZZ_RES (256), ZZ_FRAMES (1), ZZ_STEPS (20),
    // ZZ_SKIPTEXT=1 reuses a cached encoding from the scratch dir if present.
    [Fact]
    public void WanCpuEndToEnd()
    {
        if (Environment.GetEnvironmentVariable("ZZ_E2E") != "1") return;
        int res = int.TryParse(Environment.GetEnvironmentVariable("ZZ_RES"), out var r) ? r : 256;
        int frames = int.TryParse(Environment.GetEnvironmentVariable("ZZ_FRAMES"), out var f) ? f : 1;
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var s) ? s : 20;
        string root = @"C:\Git-Public\OpenTail.Stingray\models\wan2.1";
        string scratch = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string condCache = Path.Combine(scratch, "wan_cond.bin"), uncondCache = Path.Combine(scratch, "wan_uncond.bin");

        var total = Stopwatch.StartNew();
        float[] rawCond, rawUncond;
        var sw = Stopwatch.StartNew();
        if (Environment.GetEnvironmentVariable("ZZ_SKIPTEXT") == "1" && File.Exists(condCache))
        {
            rawCond = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(condCache)).ToArray();
            rawUncond = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(uncondCache)).ToArray();
        }
        else
        {
            var tok = OpenTail.Stingray.Diffusion.TextEncoders.T5Tokenizer.FromFile(Path.Combine(root, "umt5-tokenizer.json"), maxLen: 512);
            using var umt5 = new OpenTail.Stingray.Diffusion.TextEncoders.UMT5Encoder(Path.Combine(root, "models_t5_umt5-xxl-enc-bf16.safetensors"));
            rawCond = umt5.Encode(tok.Tokenize("a red apple on a wooden table, photorealistic, high quality"));
            rawUncond = umt5.Encode(tok.Tokenize(""));
            File.WriteAllBytes(condCache, MemoryMarshal.AsBytes(rawCond.AsSpan()).ToArray());
            File.WriteAllBytes(uncondCache, MemoryMarshal.AsBytes(rawUncond.AsSpan()).ToArray());
        }
        Log($"[ZZ] text encode {sw.Elapsed.TotalSeconds:F1}s");

        const int fixedLen = 226, txtDim = 4096;
        var cond = new float[fixedLen * txtDim]; Array.Copy(rawCond, cond, Math.Min(rawCond.Length, cond.Length));
        var uncond = new float[fixedLen * txtDim]; Array.Copy(rawUncond, uncond, Math.Min(rawUncond.Length, uncond.Length));

        sw.Restart();
        using var pipe = OpenTail.Stingray.Diffusion.Wan.WanPipeline.Load(Path.Combine(root, "wan2.1-t2v-1.3b-dit.safetensors"), Path.Combine(root, "Wan2.1_VAE.safetensors"), backend: null);
        Log($"[ZZ] load {sw.Elapsed.TotalSeconds:F1}s");
        sw.Restart();
        int stepN = 0; var stepSw = Stopwatch.StartNew();
        pipe.Generate("", "", res, res, frames, steps, 6.0f, 3.0f, 42, Path.Combine(scratch, $"zz_wan_cpu_{res}.png"),
            progress: (i, n) => { Log($"[ZZ] step {i}/{n} {stepSw.Elapsed.TotalSeconds:F2}s"); stepSw.Restart(); stepN = i; },
            textContext: cond, negativeTextContext: uncond);
        Log($"[ZZ] generate {sw.Elapsed.TotalSeconds:F1}s, total {total.Elapsed.TotalSeconds:F1}s");
    }

    [Fact]
    public void GemmMicrobench()
    {
        if (Environment.GetEnvironmentVariable("ZZ_E2E") == "1") return;
        int m = int.TryParse(Environment.GetEnvironmentVariable("ZZ_M"), out var mm) ? mm : 1024;
        var rng = new Random(1);
        foreach (var (k, n) in new[] { (1536, 1536), (1536, 8960), (8960, 1536) })
        {
            var x = new float[m * k]; var w = new float[n * k];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() - 0.5);
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5) * 0.05f;
            var o1 = new float[m * n]; var o2 = new float[m * n];
            double flop = 2.0 * m * n * k;
            fixed (float* px = x, pw = w, p1 = o1, p2 = o2)
            {
                float* packed = PackedSgemmF32.PackWeights(pw, n, k);
                double best1 = 1e9, best2 = 1e9;
                for (int rep = 0; rep < 5; rep++)
                {
                    var sw = Stopwatch.StartNew();
                    SimdKernels.MatMulBatchedF32(p1, pw, px, m, n, k);
                    best1 = Math.Min(best1, sw.Elapsed.TotalSeconds);
                    sw.Restart();
                    PackedSgemmF32.Gemm(p2, px, packed, null, m, n, k);
                    best2 = Math.Min(best2, sw.Elapsed.TotalSeconds);
                }
                NativeMemory.AlignedFree(packed);
                float md = 0; for (int i = 0; i < o1.Length; i++) md = MathF.Max(md, MathF.Abs(o1[i] - o2[i]));
                Log($"m={m} k={k} n={n}: dot-kernel {best1 * 1e3:F1}ms ({flop / best1 / 1e9:F0} GFLOP/s)  packed {best2 * 1e3:F1}ms ({flop / best2 / 1e9:F0} GFLOP/s)  maxDiff={md:E2}");
            }
        }
    }
}
