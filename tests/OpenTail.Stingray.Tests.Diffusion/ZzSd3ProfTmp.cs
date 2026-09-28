using System.Diagnostics;
using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

// Scratch harness (untracked): SD3.5 256^2/20-step generate. Env: ZZ_SD3=1, ZZ_GPU=1 for Vulkan,
// ZZ_OUT (dir), ZZ_TAG (file suffix), STINGRAY_SD3_INJECT_NOISE_PATH for the C++ noise.
public sealed class ZzSd3ProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzSd3ProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void Sd3Generate()
    {
        if (Environment.GetEnvironmentVariable("ZZ_SD3") != "1") return;
        string m = @"C:\Git-Public\OpenTail.Stingray\models";
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string tag = Environment.GetEnvironmentVariable("ZZ_TAG") ?? "run";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("ZZ_STEPS"), out var s) ? s : 20;
        using var vk = Environment.GetEnvironmentVariable("ZZ_GPU") == "1" ? new OpenTail.Stingray.Vulkan.VulkanBackend() : null;
        var sw = Stopwatch.StartNew();
        using var pipe = Sd3Pipeline.LoadSeparate(
            Path.Combine(m, "sd35-medium-aux", "text_encoder", "model.fp16.safetensors"),
            Path.Combine(m, "sd35-medium-aux", "text_encoder_2", "model.fp16.safetensors"),
            Path.Combine(m, "sd3.5_medium-Q4_K_M.gguf"),
            Path.Combine(m, "sd35-medium-aux", "vae", "diffusion_pytorch_model.safetensors"),
            Path.Combine(m, "flux1-schnell", "tokenizer_clip", "tokenizer.json"),
            backend: vk,
            t5EncoderPath: Path.Combine(m, "flux1-schnell", "t5xxl_fp8_e4m3fn.safetensors"),
            t5TokenizerPath: Path.Combine(m, "flux1-schnell", "tokenizer_t5", "tokenizer.json"));
        Log($"[ZZ] load {sw.Elapsed.TotalSeconds:F1}s");
        sw.Restart();
        pipe.Generate(prompt: "a red apple on a wooden table", negativePrompt: "", width: 256, height: 256,
            steps: steps, guidance: 4.5f, seed: 42, outputPath: Path.Combine(outDir, $"zz_sd3_{tag}.png"));
        Log($"[ZZ] generate {sw.Elapsed.TotalSeconds:F1}s");
    }
}
