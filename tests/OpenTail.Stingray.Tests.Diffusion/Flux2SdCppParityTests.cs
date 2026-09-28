using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// FLUX.2 [dev] end to end (real Mistral-Small-3.2 conditioning + DiT) against stable-diffusion.cpp
/// (<c>sd-cli --backend cpu</c>, the vendored <c>examples/stable-diffusion.cpp</c> with its git-ignored dump patches;
/// docs/102 #13). Fixture (<c>TestData/Flux2SdCppGolden</c>): the reference's initial noise and its latent after ONE
/// Euler step (sigma 1 -> 0, so velocity = noise - latent) for "a red apple on a wooden table", 256², guidance 3.5,
/// seed 42, real <c>flux2-dev-Q4_K_S.gguf</c>. The conditioning (512 x 15360, ~31 MB) is not committed: this test
/// encodes the prompt with the real Mistral checkpoint, so it also covers <see cref="Flux2TextConditioning"/>.
/// </summary>
public sealed class Flux2SdCppParityTests : HeavyTestBase
{
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
        string? dit = FindRepoPath("models", "_models", "flux2-dev-Q4_K_S.gguf");
        string? mistral = FindRepoPath("models", "_models", "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf");
        string? vae = FindRepoPath("models", "_models", "flux2-vae.safetensors");
        string? golden = FindRepoPath("tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "Flux2SdCppGolden");
        Assert.SkipUnless(dit is not null && mistral is not null && vae is not null && golden is not null,
            "FLUX.2 DiT/Mistral/VAE checkpoints or the sd.cpp fixture not found");

        string noisePath = Path.Combine(golden!, "noise.f32");
        var noise = ReadFloats(noisePath);
        var refLatent = ReadFloats(Path.Combine(golden!, "latent_1step_cpu.f32"));
        string tmp = Path.Combine(Path.GetTempPath(), $"flux2_sdcpp_parity_{Environment.ProcessId}");
        Directory.CreateDirectory(tmp);
        string oursPath = Path.Combine(tmp, "latent.f32");

        Environment.SetEnvironmentVariable("STINGRAY_FLUX2_INJECT_NOISE_PATH", noisePath);
        Environment.SetEnvironmentVariable("STINGRAY_FLUX2_DUMP_LATENT_PATH", oursPath);
        try
        {
            using var pipeline = Flux2Pipeline.Load(dit!, mistral!, vae!);
            pipeline.Generate(new Flux2GenerationRequest
            {
                Prompt = "a red apple on a wooden table",
                Width = 256,
                Height = 256,
                Steps = 1,
                Guidance = 3.5f,
                Seed = 42,
                OutputPath = Path.Combine(tmp, "out.png"),
            });
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_FLUX2_INJECT_NOISE_PATH", null);
            Environment.SetEnvironmentVariable("STINGRAY_FLUX2_DUMP_LATENT_PATH", null);
        }

        var ours = ReadFloats(oursPath);
        Assert.Equal(refLatent.Length, ours.Length);
        double dot = 0, nr = 0, no = 0, diff = 0;
        for (int i = 0; i < noise.Length; i++)
        {
            double r = noise[i] - refLatent[i], o = noise[i] - ours[i];
            dot += r * o; nr += r * r; no += o * o; diff += (r - o) * (r - o);
        }
        double cos = dot / Math.Sqrt(nr * no), rel = Math.Sqrt(diff / nr), normRatio = Math.Sqrt(no / nr);
        Console.WriteLine($"[FLUX.2 vs sd.cpp CPU] 1-step velocity cosine {cos:F6}, relative L2 {rel:F4}, norm ratio {normRatio:F4}");
        Assert.True(cos > 0.999, $"cosine {cos:F6}");
        Assert.InRange(normRatio, 0.98, 1.02);
    }
}
