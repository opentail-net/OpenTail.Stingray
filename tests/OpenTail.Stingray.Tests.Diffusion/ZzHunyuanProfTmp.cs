using System.Diagnostics;
using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): HunyuanVideo 256^2 single-frame generate with real LLaMA-3 + CLIP-L
// conditioning. Env: ZZ_HY=1, ZZ_STEPS (8), ZZ_NOCLIP=1 to skip CLIP-L, ZZ_OUT, ZZ_TAG.
public sealed class ZzHunyuanProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzHunyuanProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void Fp8DecodeCheck()
    {
        if (Environment.GetEnvironmentVariable("ZZ_FP8") != "1") return;
        string p = @"C:\Git-Public\OpenTail.Stingray\models\hunyuanvideo\hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors";
        string name = "double_blocks.0.img_attn_qkv.weight";
        using var fs = File.OpenRead(p);
        var lenB = new byte[8]; fs.ReadExactly(lenB);
        long hl = BitConverter.ToInt64(lenB);
        var hb = new byte[hl]; fs.ReadExactly(hb);
        using var doc = System.Text.Json.JsonDocument.Parse(hb);
        var off = doc.RootElement.GetProperty(name).GetProperty("data_offsets");
        long start = off[0].GetInt64(), end = off[1].GetInt64();
        fs.Position = 8 + hl + start;
        var raw = new byte[end - start]; fs.ReadExactly(raw);
        static float Dec(byte b)
        {
            if ((b & 0x7F) == 0x7F) return float.NaN;
            int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
            float v = e == 0 ? m / 8f * MathF.Pow(2, -6) : (1 + m / 8f) * MathF.Pow(2, e - 7);
            return s == 1 ? -v : v;
        }
        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(p);
        var f = loader.ReadF32(name);
        int bad = 0; double sumSq = 0; float maxAbs = 0;
        for (int i = 0; i < raw.Length; i++)
        {
            float r = Dec(raw[i]);
            if (!(r == f[i] || (float.IsNaN(r) && float.IsNaN(f[i])))) { if (bad < 5) Log($"[ZZ fp8] mismatch at {i}: byte 0x{raw[i]:X2} ref {r} got {f[i]}"); bad++; }
            sumSq += (double)f[i] * f[i]; maxAbs = MathF.Max(maxAbs, MathF.Abs(f[i]));
        }
        Log($"[ZZ fp8] {name}: {raw.Length} elems, mismatches {bad}, rms {Math.Sqrt(sumSq / raw.Length):E3}, maxAbs {maxAbs}");
    }

    [Fact]
    public void HunyuanVaeOnly()
    {
        if (Environment.GetEnvironmentVariable("ZZ_HYVAE") != "1") return;
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(@"C:\Git-Public\OpenTail.Stingray\models\hunyuanvideo\hunyuan_video_vae_bf16.safetensors");
        using var vae = new HunyuanVaeDecoder3D(loader);
        int lh = 32, lw = 32;
        string? latPath = Environment.GetEnvironmentVariable("ZZ_LATENT");
        if (latPath is not null)
        {
            var lat = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(latPath)).ToArray();
            // Channels 0..2 of the latent as a false-colour image (per-channel min/max stretched).
            var vis = new float[3 * lh * lw];
            for (int c = 0; c < 3; c++)
            {
                var ch = lat.AsSpan(c * lh * lw, lh * lw);
                float mn = float.MaxValue, mx = float.MinValue;
                foreach (var v in ch) { mn = MathF.Min(mn, v); mx = MathF.Max(mx, v); }
                for (int i = 0; i < ch.Length; i++) vis[c * lh * lw + i] = (ch[i] - mn) / (mx - mn + 1e-6f);
            }
            OpenTail.Stingray.Diffusion.PngWriter.Write(Path.Combine(outDir, "zz_hylat_vis.png"), vis, lw, lh);
            string? perm = Environment.GetEnvironmentVariable("ZZ_PERM");
            if (perm == "cinner")
            {
                // Re-pack with our (c, dy, dx) order, then unpack assuming (dy, dx, c).
                var packed = HunyuanVideoModel.PackLatents(lat, 1, lh, lw);
                var p2 = new float[lat.Length];
                for (int py = 0; py < lh / 2; py++) for (int px = 0; px < lw / 2; px++)
                {
                    int tok = py * (lw / 2) + px;
                    for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) for (int c = 0; c < 16; c++)
                        p2[(c * lh + py * 2 + dy) * lw + px * 2 + dx] = packed[tok * 64 + (dy * 2 + dx) * 16 + c];
                }
                lat = p2;
            }
            if (perm is "swapx" or "swapy")
            {
                var p2 = new float[lat.Length];
                for (int c = 0; c < 16; c++) for (int y = 0; y < lh; y++) for (int x = 0; x < lw; x++)
                {
                    int sy = perm == "swapy" ? (y ^ 1) : y, sx = perm == "swapx" ? (x ^ 1) : x;
                    p2[(c * lh + y) * lw + x] = lat[(c * lh + sy) * lw + sx];
                }
                lat = p2;
            }
            var swd = Stopwatch.StartNew();
            var fr = vae.Decode(lat, 1, lh, lw)[0];
            Log($"[ZZ vae] decode {swd.Elapsed.TotalSeconds:F1}s");
            OpenTail.Stingray.Diffusion.PngWriter.Write(Path.Combine(outDir, "zz_hylat_decoded.png"), fr, lw * 8, lh * 8);
            if (Environment.GetEnvironmentVariable("ZZ_CMP") == "1")
            {
                Environment.SetEnvironmentVariable("ZZ_HYVAE_SCALAR", "1");
                using var vae2 = new HunyuanVaeDecoder3D(loader);
                swd.Restart();
                var fr2 = vae2.Decode(lat, 1, lh, lw)[0];
                Log($"[ZZ vae] scalar decode {swd.Elapsed.TotalSeconds:F1}s, maxDiff {fr.Zip(fr2, (a, b) => MathF.Abs(a - b)).Max():E2}");
                Environment.SetEnvironmentVariable("ZZ_HYVAE_SCALAR", null);
            }
            Log("[ZZ vae] decoded saved latent");
            return;
        }
        foreach (var (name, gen) in new (string, Func<int, int, int, float>)[] {
            ("zero", (c, y, x) => 0f),
            ("smooth", (c, y, x) => (float)(Math.Sin(y * 0.2 + c) * Math.Cos(x * 0.15 - c * 0.5))) })
        {
            var lat = new float[16 * lh * lw];
            for (int c = 0; c < 16; c++) for (int y = 0; y < lh; y++) for (int x = 0; x < lw; x++) lat[(c * lh + y) * lw + x] = gen(c, y, x);
            var f = vae.Decode(lat, 1, lh, lw)[0];
            double mn = f.Min(), mx = f.Max(), mean = f.Average();
            Log($"[ZZ vae] {name}: min {mn:F3} max {mx:F3} mean {mean:F3}");
            OpenTail.Stingray.Diffusion.PngWriter.Write(Path.Combine(outDir, $"zz_hyvae_{name}.png"), f, lw * 8, lh * 8);
        }
    }

    [Fact]
    public void HunyuanGenerate()
    {
        if (Environment.GetEnvironmentVariable("ZZ_HY") != "1") return;
        string m = @"C:\Git-Public\OpenTail.Stingray\models";
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string tag = Environment.GetEnvironmentVariable("ZZ_TAG") ?? "run";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var s) ? s : 8;
        int res = int.TryParse(Environment.GetEnvironmentVariable("ZZ_RES"), out var rr) ? rr : 256;
        bool clip = Environment.GetEnvironmentVariable("ZZ_NOCLIP") != "1";
        var sw = Stopwatch.StartNew();
        using var gpuBackend = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new OpenTail.Stingray.Vulkan.VulkanBackend() : null;
        using var pipe = HunyuanVideoPipeline.Load(
            Path.Combine(m, "hunyuanvideo", "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"),
            Path.Combine(m, "_models", "llava-llama-3-8b-v1_1-int4.gguf"),
            Path.Combine(m, "hunyuanvideo", "hunyuan_video_vae_bf16.safetensors"),
            gpuBackend,
            clipLPath: clip ? Path.Combine(m, "flux1-schnell", "clip_l.safetensors") : null,
            clipTokenizerPath: clip ? Path.Combine(m, "flux1-schnell", "tokenizer_clip", "tokenizer.json") : null);
        Log($"[ZZ] load {sw.Elapsed.TotalSeconds:F1}s");
        sw.Restart();
        var frames = pipe.Generate("a red apple on a wooden table", "", res, res, numFrames: 1, steps: steps, guidance: 6.0f, seed: 42,
            outputPath: Path.Combine(outDir, $"zz_hy_{tag}_raw.png"),
            progress: (i, n) => Log($"[ZZ] step {i}/{n} at {sw.Elapsed.TotalSeconds:F1}s"));
        Log($"[ZZ] generate {sw.Elapsed.TotalSeconds:F1}s");
        var f = frames[0];
        var rescaled = new float[f.Length];
        for (int i = 0; i < f.Length; i++) rescaled[i] = Math.Clamp((f[i] + 1f) * 0.5f, 0f, 1f);
        OpenTail.Stingray.Diffusion.PngWriter.Write(Path.Combine(outDir, $"zz_hy_{tag}.png"), rescaled, res, res);
    }
}
