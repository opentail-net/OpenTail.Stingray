
namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Native C# Wan 2.1 / 2.2 Video and Image Diffusion pipeline.
/// Supports Text-to-Video (T2V), Image-to-Video (I2V), and Dual-Model Low/High Noise swapping (Wan2.2 A14B).
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::WAN
/// </summary>
public sealed class WanPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly WanModel _transformer;
    // Real Wan VAE: a proper 3D causal VAE (decoder.conv1, decoder.middle.*.residual,
    // decoder.upsamples.*, gamma-based norm, 3D conv kernels [C,C,3,3,3]) -- NOT the generic
    // 2D Stable-Diffusion-style VaeDecoder (decoder.conv_in.weight etc.) this pipeline used to
    // construct for its single-frame path only, while the multi-frame path already (correctly)
    // used this same WanVaeDecoder3D class. Unified onto one decoder for both paths -- numFrames=1
    // is just the degenerate case of the same real 3D decode, not a special path needing its own
    // (wrong) decoder.
    private readonly WanVaeDecoder3D _vae;
    private bool _disposed;

    public string Architecture => "WanVideo";

    public WanPipeline(
        IWeightLoader weights,
        WanModel transformer,
        WanVaeDecoder3D vae)
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

        var vae = new WanVaeDecoder3D(vaeLoader, backend: backend);

        return new WanPipeline(weights, transformer, vae);
    }

    /// <summary>
    /// Generates image or multi-frame video using Wan Flow-Matching DiT.
    /// </summary>
    public List<float[]> Generate(
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
        float[]? textContext = null,
        float[]? negativeTextContext = null,
        float[]? initImageRgb = null,
        WanModel? highNoiseTransformer = null,
        float highNoiseBoundary = 0.5f)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException($"Width and height must be divisible by 16 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;
        int latC = 16;

        // 1. Text conditioning context [seqLen, 4096] (or dummy context for conformance).
        // `prompt`/`negativePrompt` are NOT re-encoded here -- real text encoding (real UMT5,
        // see TextEncoders/UMT5Encoder.cs) is the caller's responsibility, matching this class's
        // existing `textContext` pre-computed-embedding pattern; `negativeTextContext` follows
        // the same convention (falls back to an all-zero context, the prior behavior, if the
        // caller doesn't supply a real negative-prompt encoding -- `negativePrompt` itself was
        // previously accepted but silently discarded entirely).
        int seqLen = 512;
        var condContext = textContext ?? new float[seqLen * WanModel.TextDim];
        var uncondContext = negativeTextContext ?? new float[seqLen * WanModel.TextDim];

        // 2. Initial Gaussian noise in video latent space [16, numFrames, latH, latW]
        var latent = SampleGaussianNoise(latC * numFrames * latH * latW, seed);

        // If initial image provided (I2V), blend encoded image into starting frame
        if (initImageRgb is not null)
        {
            using var vaeEnc = new VaeEncoder(_weights);
            var initLatent = vaeEnc.Encode(initImageRgb, height, width, latentChannels: 16, seed: seed);
            int frameLen = latC * latH * latW;
            for (int i = 0; i < Math.Min(frameLen, initLatent.Length); i++)
                latent[i] = initLatent[i] + 0.1f * latent[i];
        }

        // 3. Rectified Flow-Matching Timesteps with Flow Shift s = 3.0:
        var timesteps = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float linearT = 1.0f - (float)i / steps;
            timesteps[i] = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        }

        // 4. Euler Flow trajectory loop with optional Dual-Model Low/High Noise switching
        for (int step = 0; step < steps; step++)
        {
            float t = timesteps[step];
            float tNext = timesteps[step + 1];
            float dt = t - tNext;

            var activeModel = (highNoiseTransformer is not null && t >= highNoiseBoundary)
                ? highNoiseTransformer
                : _transformer;

            var condVelocity = activeModel.Forward(latent, t * 1000.0f, condContext, numFrames, latH, latW);
            float[] velocity;

            if (guidance > 1.0f)
            {
                var uncondVelocity = activeModel.Forward(latent, t * 1000.0f, uncondContext, numFrames, latH, latW);
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

        // 5. Decode 3D latents to full RGB video frames via the real 3D causal VAE (same decoder
        // for numFrames==1 and numFrames>1 -- see the _vae field's doc comment for why these were
        // previously two different, inconsistently-wired code paths).
        List<float[]> allFrames = _vae.Decode(latent, numFrames, latH, latW);

        // Optional super-resolution upscaling per frame
        if (upscaler is not null)
        {
            for (int f = 0; f < allFrames.Count; f++)
            {
                var preUpscale = allFrames[f];
                var (up, uw, uh) = upscaler.Upscale(allFrames[f], width, height);
                allFrames[f] = up;
                if (upscaleBlend < 1f)
                {
                    var bicubic = DiffusionOps.UpsampleBicubic(preUpscale, 3, height, width, uh, uw);
                    allFrames[f] = DiffusionOps.BlendRgb(allFrames[f], bicubic, upscaleBlend);
                }
            }
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
