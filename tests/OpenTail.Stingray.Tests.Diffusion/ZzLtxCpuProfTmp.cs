using System.Diagnostics;
using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): LTX-Video CPU end-to-end timing. Env: ZZ_RES (512), ZZ_STEPS (20),
// ZZ_CFG (3.0), ZZ_OUT (output dir).
public sealed class ZzLtxCpuProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzLtxCpuProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void LtxCpuEndToEnd()
    {
        if (Environment.GetEnvironmentVariable("ZZ_LTX") != "1") return;
        int res = int.TryParse(Environment.GetEnvironmentVariable("ZZ_RES"), out var r) ? r : 512;
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var s) ? s : 20;
        float cfg = float.TryParse(Environment.GetEnvironmentVariable("ZZ_CFG"), out var c) ? c : 3.0f;
        string models = @"C:\Git-Public\OpenTail.Stingray\models";
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();

        var total = Stopwatch.StartNew();
        var sw = Stopwatch.StartNew();
        using var vk = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new OpenTail.Stingray.Vulkan.VulkanBackend() : null;
        using var pipe = LtxVideoPipeline.Load(Path.Combine(models, "ltx-video-2b-v0.9.1.safetensors"), backend: vk,
            textEncoderDir: Path.Combine(models, "ltx-t5", "text_encoder"),
            tokenizerJsonPath: Path.Combine(models, "ltx-t5", "tokenizer", "tokenizer.json"));
        Log($"[ZZ] load {sw.Elapsed.TotalSeconds:F1}s");
        sw.Restart();
        var stepSw = Stopwatch.StartNew();
        var frames = pipe.GenerateVideo("a red apple on a wooden table", res, res, numFrames: 1, steps: steps, guidance: cfg, seed: int.TryParse(Environment.GetEnvironmentVariable("ZZ_SEED"), out var sd) ? sd : 42,
            progress: (i, n) => { Log($"[ZZ] step {i}/{n} {stepSw.Elapsed.TotalSeconds:F2}s"); stepSw.Restart(); });
        Log($"[ZZ] after last step -> end (VAE) {stepSw.Elapsed.TotalSeconds:F1}s");
        Log($"[ZZ] generate {sw.Elapsed.TotalSeconds:F1}s, total {total.Elapsed.TotalSeconds:F1}s");
        var f0 = frames[0];
        double mean = 0; foreach (var v in f0) mean += v; mean /= f0.Length;
        Log($"[ZZ] frame0 mean {mean:F4}");
        OpenTail.Stingray.Diffusion.PngWriter.Write(Path.Combine(outDir, $"zz_ltx_{(vk is null ? "cpu" : "gpu")}_{res}.png"), f0, res, res);
    }
}
