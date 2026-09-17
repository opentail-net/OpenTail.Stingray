using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class DiffusionBackendResolverTests
{
    [Fact]
    public void Resolve_ExplicitBackend_ReturnsExplicitAndDoesNotOwn()
    {
        using var cpu = new CpuBackend();
        var (backend, ownsBackend) = DiffusionBackendResolver.Resolve(cpu);

        Assert.Same(cpu, backend);
        Assert.False(ownsBackend);
    }

    [Fact]
    public void Resolve_WhenStingrayBackendCpuSet_ReturnsNull()
    {
        string? oldVal = Environment.GetEnvironmentVariable("STINGRAY_BACKEND");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_BACKEND", "cpu");
            var (backend, ownsBackend) = DiffusionBackendResolver.Resolve(null);

            Assert.Null(backend);
            Assert.False(ownsBackend);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_BACKEND", oldVal);
        }
    }

    [Fact]
    public void Resolve_DefaultProbe_ReturnsVulkanWhenSupported()
    {
        string? oldVal = Environment.GetEnvironmentVariable("STINGRAY_BACKEND");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_BACKEND", null);
            var (backend, ownsBackend) = DiffusionBackendResolver.Resolve(null);

            if (backend != null)
            {
                Assert.IsType<VulkanBackend>(backend);
                Assert.True(ownsBackend);
                backend.Dispose();
            }
            else
            {
                // Headless environment without Vulkan ICD
                Assert.False(ownsBackend);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_BACKEND", oldVal);
        }
    }

    [Fact]
    public void AceStepFlowScheduler_OnVulkan_ProducesFiniteOutput()
    {
        string? turboPath = "models/acestep-v15/turbo.safetensors";
        for (int i = 0; i < 6 && !File.Exists(turboPath); i++)
            turboPath = Path.Combine("..", turboPath);
        if (!File.Exists(turboPath)) return;

        using var vk = new VulkanBackend();
        using var loader = SafetensorsLoader.Open(turboPath);
        var weights = OpenTail.Stingray.Diffusion.AceStep.Transformer.AceStepDiTWeights.Load(loader);

        int frames = 20;
        var rng = new Random(42);
        var cond = new float[16][];
        for (int i = 0; i < 16; i++)
        {
            cond[i] = new float[2048];
            for (int d = 0; d < 2048; d++) cond[i][d] = (float)(rng.NextDouble() * 0.2 - 0.1);
        }

        var latent = OpenTail.Stingray.Diffusion.AceStep.Transformer.AceStepFlowScheduler.Generate(
            weights, cond, frames, shift: 1.0f, seed: 1234, srcLatents: null, backend: vk);

        Assert.Equal(frames, latent.Length);
        for (int t = 0; t < frames; t++)
        {
            for (int c = 0; c < 64; c++)
            {
                Assert.True(float.IsFinite(latent[t][c]), $"latent[{t}][{c}] is not finite: {latent[t][c]}");
            }
        }

        string? vaePath = "models/acestep-v15/vae.safetensors";
        for (int i = 0; i < 6 && !File.Exists(vaePath); i++)
            vaePath = Path.Combine("..", vaePath);
        if (File.Exists(vaePath))
        {
            using var vaeLoader = SafetensorsLoader.Open(vaePath);
            var vaeWeights = OpenTail.Stingray.Diffusion.AceStep.Vae.AceStepOobleckDecoderWeights.Load(vaeLoader);
            var latentFlat = new float[64 * frames];
            for (int t = 0; t < frames; t++)
                for (int c = 0; c < 64; c++)
                    latentFlat[c * frames + t] = latent[t][c];

            var pcm = OpenTail.Stingray.Diffusion.AceStep.Vae.AceStepOobleckDecoder.Decode(vaeWeights, latentFlat, frames);
            for (int i = 0; i < pcm.Length; i++)
            {
                Assert.True(float.IsFinite(pcm[i]), $"pcm[{i}] is not finite: {pcm[i]}");
            }
        }
    }

    [Fact]
    public void AceStepOobleckEncoder_Silence_ProducesFinite()
    {
        string? vaePath = "models/acestep-v15/vae.safetensors";
        for (int i = 0; i < 6 && !File.Exists(vaePath); i++)
            vaePath = Path.Combine("..", vaePath);
        if (!File.Exists(vaePath)) return;

        using var vaeLoader = SafetensorsLoader.Open(vaePath);
        var vaeWeights = OpenTail.Stingray.Diffusion.AceStep.Vae.AceStepOobleckEncoderWeights.Load(vaeLoader);

        int sampleCount = 5 * 1920;
        var zeroPcm = new float[2 * sampleCount];
        var flat = OpenTail.Stingray.Diffusion.AceStep.Vae.AceStepOobleckEncoder.EncodeMode(vaeWeights, zeroPcm, 2, sampleCount);

        for (int i = 0; i < flat.Length; i++)
            Assert.True(float.IsFinite(flat[i]), $"flat[{i}] is not finite: {flat[i]}");
    }

    [Fact]
    public void AceStepQwen3TextEncoder_Prompt_ProducesFinite()
    {
        string? ggufPath = "models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf";
        for (int i = 0; i < 6 && !File.Exists(ggufPath); i++)
            ggufPath = Path.Combine("..", ggufPath);
        if (!File.Exists(ggufPath)) return;

        using var textEncoder = new OpenTail.Stingray.Diffusion.AceStep.Text.AceStepQwen3TextEncoder(ggufPath);
        string prompt =
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" +
            "# Caption\nA cinematic orchestral soundtrack with deep drums\n\n" +
            "# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 2 seconds\n<|endoftext|>\n";

        var hidden = textEncoder.Encode(prompt);
        for (int r = 0; r < hidden.Length; r++)
            for (int d = 0; d < hidden[r].Length; d++)
                Assert.True(float.IsFinite(hidden[r][d]), $"hidden[{r}][{d}] is not finite: {hidden[r][d]}");
    }
}
