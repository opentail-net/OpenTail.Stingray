using OpenTail.Stingray.Diffusion.QwenImage;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Qwen Image DiT against stable-diffusion.cpp (<c>sd-cli --backend cpu</c>, the vendored
/// <c>examples/stable-diffusion.cpp</c> built locally with its git-ignored dump patches; docs/102 #13).
/// Fixture (<c>TestData/QwenImageSdCppGolden</c>): the reference's own initial noise [16, 32, 32], its Qwen2.5-VL
/// conditioning for "a red apple on a wooden table" [12, 3584], and its latent after ONE Euler step at cfg 1
/// (sigma 1 -> 0, so velocity = noise - latent), real <c>qwen-image-Q3_K_S.gguf</c>, 256², seed 42.
/// </summary>
/// <remarks>
/// Noise floor: sd.cpp's own CPU and Vulkan backends differ by cosine 0.9984 (relative L2 6.6%) on this step.
/// Ours was 0.9933 before the <c>txt_norm</c> fix and is 0.99968 after it.
/// </remarks>
public sealed class QwenImageSdCppParityTests : HeavyTestBase
{
    private const int LatH = 32, LatW = 32;

    private static string? FindRepoPath(params string[] parts)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine([dir, .. parts]);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    [Fact]
    public void OneStepVelocity_MatchesSdCppCpu()
    {
        string? model = FindRepoPath("models", "_models", "qwen-image-Q3_K_S.gguf");
        string? golden = FindRepoPath("tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "QwenImageSdCppGolden");
        Assert.SkipUnless(model is not null && golden is not null, "qwen-image-Q3_K_S.gguf or the sd.cpp fixture not found");

        var noise = ReadFloats(Path.Combine(golden!, "noise.f32"));
        var cond = ReadFloats(Path.Combine(golden!, "cond.f32"));
        var refLatent = ReadFloats(Path.Combine(golden!, "latent_1step_cfg1_cpu.f32"));
        Assert.Equal(16 * LatH * LatW, noise.Length);

        using var weights = OpenTail.Stingray.Diffusion.GgufWeightLoader.Open(model!);
        using var dit = new QwenImageModel(weights, prefix: "");
        var velocity = dit.Forward(noise, 1000f, cond, LatH, LatW);

        double dot = 0, nr = 0, no = 0, diff = 0;
        for (int i = 0; i < noise.Length; i++)
        {
            double r = noise[i] - refLatent[i], o = velocity[i];
            dot += r * o; nr += r * r; no += o * o; diff += (r - o) * (r - o);
        }
        double cos = dot / Math.Sqrt(nr * no), rel = Math.Sqrt(diff / nr);
        Console.WriteLine($"[QwenImage vs sd.cpp CPU] 1-step velocity cosine {cos:F6}, relative L2 {rel:F4}");
        Assert.True(cos > 0.999, $"cosine {cos:F6} (sd.cpp CPU vs Vulkan floor is 0.9984; ours was 0.9933 without txt_norm)");
    }
}
