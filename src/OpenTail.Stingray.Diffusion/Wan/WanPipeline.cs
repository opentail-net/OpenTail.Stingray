using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Native C# Wan 2.1 / 2.2 Video and Image Diffusion pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::WAN
/// </summary>
public sealed class WanPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly WanModel _transformer;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public string Architecture => "WanVideo";

    public WanPipeline(
        IWeightLoader weights,
        WanModel transformer,
        VaeDecoder vae)
    {
        _weights = weights;
        _transformer = transformer;
        _vae = vae;
    }

    public static WanPipeline Load(string modelPath, string? vaePath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new WanModel(weights, prefix: "", backend: backend);

        IWeightLoader vaeLoader = weights;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }

        var vae = new VaeDecoder(vaeLoader, backend: backend);

        return new WanPipeline(weights, transformer, vae);
    }

    /// <summary>
    /// Generates image or video frames using Wan Flow-Matching DiT.
    /// </summary>
    public void Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 832,
        int height = 480,
        int numFrames = 1,
        int steps = 20,
        float guidance = 6.0f,
        float flowShift = 3.0f,
        int seed = -1,
        string outputPath = "output.png",
        Action<int, int>? progress = null,
        RRDBNet? upscaler = null,
        float upscaleBlend = 1.0f,
        float[]? textContext = null)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException($"Width and height must be divisible by 16 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;
        int latC = 16;

        // 1. Text conditioning context [seqLen, 4096] (or dummy context for conformance)
        int seqLen = 512;
        var condContext = textContext ?? new float[seqLen * WanModel.TextDim];
        var uncondContext = new float[seqLen * WanModel.TextDim];

        // 2. Initial Gaussian noise in video latent space [16, numFrames, latH, latW]
        var latent = SampleGaussianNoise(latC * numFrames * latH * latW, seed);

        // 3. Rectified Flow-Matching Timesteps with Flow Shift s = 3.0:
        var timesteps = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float linearT = 1.0f - (float)i / steps;
            timesteps[i] = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        }

        // 4. Euler Flow trajectory loop
        for (int step = 0; step < steps; step++)
        {
            float t = timesteps[step];
            float tNext = timesteps[step + 1];
            float dt = t - tNext;

            var condVelocity = _transformer.Forward(latent, t * 1000.0f, condContext, numFrames, latH, latW);
            float[] velocity;

            if (guidance > 1.0f)
            {
                var uncondVelocity = _transformer.Forward(latent, t * 1000.0f, uncondContext, numFrames, latH, latW);
                velocity = new float[condVelocity.Length];
                for (int i = 0; i < velocity.Length; i++)
                    velocity[i] = uncondVelocity[i] + guidance * (condVelocity[i] - uncondVelocity[i]);
            }
            else
            {
                velocity = condVelocity;
            }

            for (int i = 0; i < latent.Length; i++)
                latent[i] -= dt * velocity[i];

            progress?.Invoke(step + 1, steps);
        }

        // 5. Decode latents to RGB pixels
        var frameLatent = latent;
        if (numFrames > 1)
        {
            // For first milestone, decode the primary anchor frame
            int frameLen = latC * latH * latW;
            frameLatent = new float[frameLen];
            Array.Copy(latent, 0, frameLatent, 0, frameLen);
        }

        var pixels = _vae.Decode(frameLatent, latH, latW);

        // 6. Optional Super-Resolution Upscaling
        int outWidth = width, outHeight = height;
        if (upscaler is not null)
        {
            var preUpscalePixels = pixels;
            var (up, uw, uh) = upscaler.Upscale(pixels, width, height);
            pixels = up; outWidth = uw; outHeight = uh;
            if (upscaleBlend < 1f)
            {
                var bicubic = DiffusionOps.UpsampleBicubic(preUpscalePixels, 3, height, width, outHeight, outWidth);
                pixels = DiffusionOps.BlendRgb(pixels, bicubic, upscaleBlend);
            }
        }

        // 7. Write output PNG
        PngWriter.Write(outputPath, pixels, outWidth, outHeight);
    }

    private static float[] SampleGaussianNoise(int length, int seed)
    {
        var noise = new float[length];
        var rng = seed >= 0 ? new Random(seed) : new Random();

        for (int i = 0; i < length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            noise[i] = (float)(radius * Math.Cos(theta));
            noise[i + 1] = (float)(radius * Math.Sin(theta));
        }

        if ((length & 1) == 1)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[^1] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return noise;
    }

    public void Generate(ImageGenerationRequest request)
    {
        Generate(
            request.Prompt,
            negativePrompt: null,
            width: request.Width <= 0 ? 832 : request.Width,
            height: request.Height <= 0 ? 480 : request.Height,
            numFrames: 1,
            steps: request.Steps <= 0 ? 20 : request.Steps,
            guidance: request.Guidance == 1.0f ? 6.0f : request.Guidance,
            flowShift: 3.0f,
            seed: request.Seed,
            outputPath: request.OutputPath,
            progress: request.Progress,
            upscaler: request.Upscaler,
            upscaleBlend: request.UpscaleBlend);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
            _transformer.Dispose();
            _vae.Dispose();
        }
    }
}
