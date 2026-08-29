
namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Native C# HunyuanVideo Text-to-Video and Text-to-Image inference pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::HUNYUAN_VIDEO
/// </summary>
public sealed class HunyuanVideoPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly HunyuanVideoModel _transformer;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public string Architecture => "HunyuanVideo";

    public HunyuanVideoPipeline(
        IWeightLoader weights,
        HunyuanVideoModel transformer,
        VaeDecoder vae)
    {
        _weights = weights;
        _transformer = transformer;
        _vae = vae;
    }

    public static HunyuanVideoPipeline Load(string modelPath, string? vaePath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new HunyuanVideoModel(weights, prefix: "", backend: backend);

        IWeightLoader vaeLoader = weights;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }

        var vae = new VaeDecoder(vaeLoader, backend: backend);

        return new HunyuanVideoPipeline(weights, transformer, vae);
    }

    /// <summary>
    /// Generates image or multi-frame video using HunyuanVideo Rectified Flow-Matching.
    /// </summary>
    public List<float[]> Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 1280,
        int height = 720,
        int numFrames = 1,
        int steps = 20,
        float guidance = 6.0f,
        float flowShift = 7.0f,
        int seed = -1,
        string outputPath = "output.png",
        Action<int, int>? progress = null,
        RRDBNet? upscaler = null,
        float upscaleBlend = 1.0f,
        float[]? textContext = null,
        float[]? initImageRgb = null)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException($"Width and height must be divisible by 16 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;
        int latC = 16;

        // 1. Text conditioning context [seqLen, 4096] (or dummy context for conformance)
        int seqLen = 256;
        var condContext = textContext ?? new float[seqLen * HunyuanVideoModel.TextDim];
        var uncondContext = new float[seqLen * HunyuanVideoModel.TextDim];

        // 2. Initial Gaussian noise in video latent space [16, numFrames, latH, latW]
        var latent = SampleGaussianNoise(latC * numFrames * latH * latW, seed);

        // If initial image provided (I2V), blend into starting frame
        if (initImageRgb is not null)
        {
            using var vaeEnc = new VaeEncoder(_weights);
            var initLatent = vaeEnc.Encode(initImageRgb, height, width, latentChannels: 16, seed: seed);
            int frameLen = latC * latH * latW;
            for (int i = 0; i < Math.Min(frameLen, initLatent.Length); i++)
                latent[i] = initLatent[i] + 0.1f * latent[i];
        }

        // 3. Rectified Flow-Matching Timesteps with Flow Shift s = 7.0:
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

        // 5. Decode all video frame latents to RGB pixels via 16-channel VAE
        int singleFrameLen = latC * latH * latW;
        var allFrames = new List<float[]>(numFrames);

        for (int f = 0; f < numFrames; f++)
        {
            var singleFrameLatent = new float[singleFrameLen];
            for (int c = 0; c < latC; c++)
            {
                int srcOff = ((c * numFrames) + f) * (latH * latW);
                int dstOff = c * (latH * latW);
                Array.Copy(latent, srcOff, singleFrameLatent, dstOff, latH * latW);
            }

            var framePixels = _vae.Decode(singleFrameLatent, latH, latW);

            // Optional super-resolution upscaling
            if (upscaler is not null)
            {
                var preUpscale = framePixels;
                var (up, uw, uh) = upscaler.Upscale(framePixels, width, height);
                framePixels = up;
                if (upscaleBlend < 1f)
                {
                    var bicubic = DiffusionOps.UpsampleBicubic(preUpscale, 3, height, width, uh, uw);
                    framePixels = DiffusionOps.BlendRgb(framePixels, bicubic, upscaleBlend);
                }
            }

            allFrames.Add(framePixels);
        }

        // 6. Save primary anchor frame
        PngWriter.Write(outputPath, allFrames[0], width, height);

        // 7. If video sequence (numFrames > 1), save all frames alongside
        if (numFrames > 1)
        {
            string dir = Path.GetDirectoryName(outputPath) ?? ".";
            string stem = Path.GetFileNameWithoutExtension(outputPath);
            string ext = Path.GetExtension(outputPath);
            if (string.IsNullOrEmpty(ext)) ext = ".png";

            for (int f = 0; f < numFrames; f++)
            {
                string framePath = Path.Combine(dir, $"{stem}_frame_{f:D3}{ext}");
                PngWriter.Write(framePath, allFrames[f], width, height);
            }
        }

        return allFrames;
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
            width: request.Width <= 0 ? 1280 : request.Width,
            height: request.Height <= 0 ? 720 : request.Height,
            numFrames: 1,
            steps: request.Steps <= 0 ? 20 : request.Steps,
            guidance: request.Guidance == 1.0f ? 6.0f : request.Guidance,
            flowShift: 7.0f,
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
