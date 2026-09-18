namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real smoke test for FLUX.2's VAE decode path (docs/087): confirms the real per-channel
/// BatchNorm un-normalize + 2x2 pixel-unshuffle (<see cref="OpenTail.Stingray.Diffusion.Flux2.Flux2Vae"/>)
/// followed by the existing general-purpose <see cref="VaeDecoder"/> (reused unmodified, same
/// diffusers AutoencoderKL schema as FLUX.1/SD1.5/SDXL/SD3) runs end-to-end against the real
/// checkpoint and produces finite RGB output. NOT a coherence/quality check (synthetic latent
/// input, no real DiT output yet) -- matches the "structurally sound, not yet coherent" milestone
/// this session's other FLUX.2 real-weight tests reached.
/// </summary>
public sealed class Flux2VaeRealWeightsTests
{
    private const string ModelFileName = "flux2-vae.safetensors";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Flux2Vae_RealWeights_DecodeProducesFiniteRgb()
    {
        string? modelPath = FindModelPath(ModelFileName);
        Assert.SkipUnless(modelPath != null, "models/_models/flux2-vae.safetensors not found");

        using var st = SafetensorsLoader.Open(modelPath!);
        Assert.True(st.Contains("bn.running_mean"));
        Assert.True(st.Contains("bn.running_var"));

        var runningMean = st.ReadF32("bn.running_mean");
        var runningVar = st.ReadF32("bn.running_var");
        Assert.Equal(128, runningMean.Length);
        Assert.Equal(128, runningVar.Length);

        // Small synthetic normalized latent -- real DiT output not available in this smoke test.
        int inH = 4, inW = 4;
        var rng = new Random(42);
        var normalizedLatent = new float[128 * inH * inW];
        for (int i = 0; i < normalizedLatent.Length; i++) normalizedLatent[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;

        var (rawLatent, h, w) = OpenTail.Stingray.Diffusion.Flux2.Flux2Vae.UnnormalizeAndUnshuffle(
            normalizedLatent, inH, inW, runningMean, runningVar);

        Assert.Equal(32 * (inH * 2) * (inW * 2), rawLatent.Length);
        Assert.Equal(inH * 2, h);
        Assert.Equal(inW * 2, w);
        foreach (var v in rawLatent) Assert.True(float.IsFinite(v), "un-normalized latent contains NaN/Inf");

        using var vae = new VaeDecoder(st);
        var rgb = vae.Decode(rawLatent, h, w, scaleOverride: 1f, shiftOverride: 0f);

        Assert.True(rgb.Length > 0);
        foreach (var v in rgb) Assert.True(float.IsFinite(v), "FLUX.2 VAE decode output contains NaN/Inf");
    }
}
