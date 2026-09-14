using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Bypasses the DiT/Euler loop entirely to isolate whether the persistent blotchy noise-blob
/// texture seen in every Wan sample (CPU, GPU, guidance on/off) originates in the VAE decoder
/// itself rather than in denoising. Decodes three very different hand-constructed latents
/// (zeros, smooth gradient, N(0,1) noise) and dumps per-frame stats + PNGs for visual comparison.
/// See docs/081-master-gpu-perf-and-accuracy-plan.md Priority 0, "2026-09-14 update".
/// </summary>
public sealed class WanVaeOnlyIsolationTests
{
    private readonly ITestOutputHelper _output;

    public WanVaeOnlyIsolationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private void Log(string msg)
    {
        _output.WriteLine(msg);
        Console.WriteLine(msg);
    }

    private static string FindModelPath(string relativePath)
    {
        var candidates = new[]
        {
            Path.Combine("..", "..", "..", "..", "..", relativePath),
            Path.Combine("..", "..", "..", relativePath),
            relativePath,
            Path.Combine(AppContext.BaseDirectory, relativePath),
            Path.Combine(@"c:\Git-Public\OpenTail.Stingray", relativePath)
        };
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return relativePath;
    }

    [Fact]
    public void Decode_ZerosVsGradientVsNoise_ShouldProduceVisiblyDifferentOutputs()
    {
        string vaePath = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));
        if (!File.Exists(vaePath))
        {
            Log($"VAE not found at {vaePath}, skipping.");
            return;
        }

        using var vae = new WanVaeDecoder3D(SafetensorsLoader.Open(vaePath));

        const int latH = 32, latW = 32, c = 16, t = 1;
        int len = c * t * latH * latW;

        // (a) all zeros -- should decode to a flat/near-flat color per real per-channel mean.
        var zeros = new float[len];

        // (b) smooth low-frequency gradient across width, same value for every channel/row.
        var gradient = new float[len];
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < latH; y++)
                for (int x = 0; x < latW; x++)
                    gradient[(ch * t) * latH * latW + y * latW + x] = (x / (float)(latW - 1)) * 2f - 1f;

        // (c) pure N(0,1) noise, same distribution as the real Euler loop's initial sample.
        var noise = new float[len];
        var rng = new Random(42);
        for (int i = 0; i < len - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1)), theta = 2.0 * Math.PI * u2;
            noise[i] = (float)(radius * Math.Cos(theta));
            noise[i + 1] = (float)(radius * Math.Sin(theta));
        }

        void DecodeAndReport(string name, float[] latent, string outFile)
        {
            var frames = vae.Decode(latent, t, latH, latW);
            var f = frames[0];
            int pixelCount = latH * 8 * latW * 8;
            float rMean = 0, gMean = 0, bMean = 0, rStd = 0;
            for (int i = 0; i < pixelCount; i++) { rMean += f[i]; gMean += f[pixelCount + i]; bMean += f[2 * pixelCount + i]; }
            rMean /= pixelCount; gMean /= pixelCount; bMean /= pixelCount;
            for (int i = 0; i < pixelCount; i++) rStd += (f[i] - rMean) * (f[i] - rMean);
            rStd = MathF.Sqrt(rStd / pixelCount);
            Log($"[VAE-only: {name}] R={rMean:F4} G={gMean:F4} B={bMean:F4} R-channel std(across pixels)={rStd:F4}");
            PngWriter.Write(outFile, f, latW * 8, latH * 8);
        }

        DecodeAndReport("zeros", zeros, "wan_vae_isolation_zeros.png");
        DecodeAndReport("gradient", gradient, "wan_vae_isolation_gradient.png");
        DecodeAndReport("noise", noise, "wan_vae_isolation_noise.png");
    }
}
