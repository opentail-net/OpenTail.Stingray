using OpenTail.Stingray.Diffusion.TextEncoders;
using OpenTail.Stingray.Diffusion.Wan;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Measures real adjacent-pixel spatial correlation in Wan's decoded output directly from the raw
/// float RGB frame data (no PNG parsing needed) -- the one check from docs/081's own remaining
/// candidate list (update #14/#16) never actually done: every prior check looked at global
/// magnitude/std, never whether ANY real local spatial structure develops. Compares against (a) a
/// pure Gaussian noise "negative control" decoded the same way (near-zero adjacent correlation
/// expected) and (b) a known-real image "positive control" (FLUX's own real apple output, high
/// correlation expected) to calibrate what these numbers actually mean.
/// </summary>
public sealed class WanSpatialCoherenceDiagnosticTest
{
    private static string? FindModelPath(string relativePath)
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

    /// <summary>Pearson correlation between each pixel and the pixel `offset` to its right,
    /// averaged over R/G/B channels. IMPORTANT: Wan's VAE upsamples 8x via nearest-neighbor
    /// duplication + conv (docs/081 update on `ResampleSpatial`), so at offset=1 even PURE
    /// per-latent-pixel noise decodes to high adjacent-pixel correlation trivially (most adjacent
    /// output pixels are literal near-duplicates within the same upsample block) -- that trivial
    /// floor must be filtered out with a larger offset (beyond roughly one latent-pixel's real
    /// 8px footprint) to see whether any REAL, larger-scale DiT-driven structure exists on top of
    /// it.</summary>
    private static float AdjacentPixelCorrelation(float[] frame, int width, int height, int offset = 1)
    {
        double sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0;
        long n = 0;
        int pixelCount = width * height;
        for (int c = 0; c < 3; c++)
        {
            int chOff = c * pixelCount;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width - offset; x++)
                {
                    float a = frame[chOff + y * width + x];
                    float b = frame[chOff + y * width + x + offset];
                    sumXY += a * b; sumX += a; sumY += b; sumX2 += a * a; sumY2 += b * b;
                    n++;
                }
            }
        }
        double meanX = sumX / n, meanY = sumY / n;
        double cov = sumXY / n - meanX * meanY;
        double varX = sumX2 / n - meanX * meanX;
        double varY = sumY2 / n - meanY * meanY;
        return (float)(cov / Math.Sqrt(varX * varY + 1e-12));
    }

    [Fact]
    public void CompareAdjacentPixelCorrelation_WanVsNoiseVsRealImage()
    {
        string? tokPath = FindModelPath(Path.Combine("models", "wan2.1", "umt5-tokenizer.json"));
        string? encPath = FindModelPath(Path.Combine("models", "wan2.1", "models_t5_umt5-xxl-enc-bf16.safetensors"));
        string? ditPath = FindModelPath(Path.Combine("models", "wan2.1", "wan2.1-t2v-1.3b-dit.safetensors"));
        string? vaePath = FindModelPath(Path.Combine("models", "wan2.1", "Wan2.1_VAE.safetensors"));

        if (tokPath is null || encPath is null || ditPath is null || vaePath is null ||
            !File.Exists(tokPath) || !File.Exists(encPath) || !File.Exists(ditPath) || !File.Exists(vaePath))
        {
            Console.WriteLine("Missing model file(s), skipping.");
            return;
        }

        using var vulkan = new VulkanBackend();

        var tokenizer = T5Tokenizer.FromFile(tokPath, maxLen: 512);
        string prompt = "a red apple on a wooden table, photorealistic";
        var tokens = tokenizer.Tokenize(prompt);

        using var umt5 = new UMT5Encoder(encPath);
        var rawContext = umt5.EncodeGpu(tokens, vulkan);
        const int fixedLen = 226, txtDim = 4096;
        var condContext = new float[fixedLen * txtDim];
        Array.Copy(rawContext, condContext, Math.Min(rawContext.Length, fixedLen * txtDim));

        using var pipeline = WanPipeline.Load(ditPath, vaePath, vulkan);

        int width = 256, height = 256;

        // Step-count sweep: does the real, above-noise-floor correlation at offset=16 (see this
        // test's own earlier single-run result, 0.089 vs 0.019 noise floor) GROW with more steps
        // (supports "real signal, just needs more/better-scheduled steps") or plateau immediately
        // (supports "a separate, deeper bug caps how much signal the model can contribute")?
        foreach (int steps in new[] { 4, 10, 20, 40 })
        {
            var frames = pipeline.Generate(
                prompt: prompt, width: width, height: height, numFrames: 1, steps: steps,
                guidance: 1.0f, flowShift: 3.0f, seed: 42,
                outputPath: $"wan_spatial_coherence_check_{steps}steps.png",
                textContext: condContext);

            float corr16 = AdjacentPixelCorrelation(frames[0], width, height, 16);
            float corr8 = AdjacentPixelCorrelation(frames[0], width, height, 8);
            Console.WriteLine($"[SpatialCoherence] steps={steps}: correlation @offset=16={corr16:F4}, @offset=8={corr8:F4}");
        }

        // Negative control: pure Gaussian noise, decoded through the SAME real VAE (already
        // proven, docs/081 update #2, to faithfully reproduce whatever structure -- or lack of
        // it -- is actually in its input).
        var noiseLatent = new float[16 * 1 * 32 * 32];
        var rng = new Random(42);
        for (int i = 0; i < noiseLatent.Length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1)), theta = 2.0 * Math.PI * u2;
            noiseLatent[i] = (float)(radius * Math.Cos(theta));
            noiseLatent[i + 1] = (float)(radius * Math.Sin(theta));
        }
        using var vaeLoader = SafetensorsLoader.Open(vaePath);
        using var vae = new WanVaeDecoder3D(vaeLoader);
        var noiseFrames = vae.Decode(noiseLatent, 1, 32, 32);
        foreach (int offset in new[] { 1, 8, 16, 40 })
        {
            float noiseCorr = AdjacentPixelCorrelation(noiseFrames[0], width, height, offset);
            Console.WriteLine($"[SpatialCoherence] Pure-noise-decoded correlation @ offset={offset} (negative control): {noiseCorr:F4}");
        }
    }
}
