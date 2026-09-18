
namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Native C# Qwen Image text-to-image and image-edit inference pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::QWEN_IMAGE
/// </summary>
public sealed class QwenImagePipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly QwenImageModel _transformer;
    private readonly Wan.WanVaeDecoder3D _vae;
    private readonly GgufModel? _textEncoderModel;
    private readonly Engine.ForwardPass? _textEncoderForward;
    private readonly GgufTokenizer? _textEncoderTokenizer;
    private readonly Cpu.CpuBackend? _textEncoderBackend;
    private bool _disposed;

    public string Architecture => "QwenImage";

    public QwenImagePipeline(
        IWeightLoader weights,
        QwenImageModel transformer,
        Wan.WanVaeDecoder3D vae)
    {
        _weights = weights;
        _transformer = transformer;
        _vae = vae;
    }

    private QwenImagePipeline(
        IWeightLoader weights, QwenImageModel transformer, Wan.WanVaeDecoder3D vae,
        GgufModel textEncoderModel, Engine.ForwardPass textEncoderForward, GgufTokenizer textEncoderTokenizer, Cpu.CpuBackend textEncoderBackend)
        : this(weights, transformer, vae)
    {
        _textEncoderModel = textEncoderModel;
        _textEncoderForward = textEncoderForward;
        _textEncoderTokenizer = textEncoderTokenizer;
        _textEncoderBackend = textEncoderBackend;
    }

    /// <summary>
    /// Loads a Qwen-Image pipeline. VAE decode uses <see cref="Wan.WanVaeDecoder3D"/> directly,
    /// unmodified -- Qwen-Image's real VAE is architecturally identical to Wan2.1's (confirmed via
    /// `examples/stable-diffusion.cpp`'s own `sd_version_uses_wan_vae(VERSION_QWEN_IMAGE)`, which
    /// literally routes it through the same `WAN::WanVAERunner`/`wan_vae.hpp` class -- same
    /// dim_mult=[1,2,4,4]/z_dim=16/dim=96 channel progression, same temperal_upsample=[true,true,
    /// false], and even the exact same per-channel latents_mean/latents_std constants as Wan2.1's
    /// 16-channel variant, confirmed directly in `wan_vae.hpp`'s `get_latents_mean_std`). Verified
    /// with a real checkpoint decode (2026-09-18): finite, non-degenerate output. See
    /// docs/086-video-model-verification-plan.md for the full tensor-shape evidence.
    /// </summary>
    public static QwenImagePipeline Load(string modelPath, string? vaePath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new QwenImageModel(weights, prefix: "", backend: backend);

        IWeightLoader vaeLoader = weights;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }

        var vae = new Wan.WanVaeDecoder3D(vaeLoader, backend);

        return new QwenImagePipeline(weights, transformer, vae);
    }

    /// <summary>
    /// Loads a Qwen-Image pipeline WITH real Qwen2.5-VL-7B text conditioning (docs/089) --
    /// <c>Generate</c> will use <see cref="QwenImageTextConditioning.Encode"/> automatically when
    /// the caller doesn't supply an explicit <c>textContext</c>.
    /// </summary>
    public static QwenImagePipeline Load(string modelPath, string textEncoderPath, string? vaePath, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new QwenImageModel(weights, prefix: "", backend: backend);

        IWeightLoader vaeLoader = weights;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }
        var vae = new Wan.WanVaeDecoder3D(vaeLoader, backend);

        var textEncoderModel = GgufModel.Open(textEncoderPath);
        var hp = ModelHyperparams.FromGgufMetadata(textEncoderModel.Metadata, textEncoderModel);
        var tokenizer = GgufTokenizer.FromGgufModel(textEncoderModel);
        var textEncoderBackend = new Cpu.CpuBackend();
        var textEncoderForward = new Engine.ForwardPass(textEncoderModel, textEncoderBackend, hp);

        return new QwenImagePipeline(weights, transformer, vae, textEncoderModel, textEncoderForward, tokenizer, textEncoderBackend);
    }

    /// <summary>
    /// Generates an image using Qwen Image Rectified Flow-Matching.
    /// Supports Text-to-Image and Qwen Image Edit (reference visual conditioning).
    /// </summary>
    public void Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 1024,
        int height = 1024,
        int steps = 20,
        float guidance = 2.5f,
        float flowShift = 3.0f,
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

        // 1. Text conditioning context [seqLen, 3584] -- real Qwen2.5-VL-7B conditioning
        //    (QwenImageTextConditioning.Encode, docs/089) when this pipeline owns a real text
        //    encoder (Load(modelPath, textEncoderPath, ...)) and the caller didn't already supply
        //    an explicit textContext; falls back to zero-conditioning otherwise (unchanged
        //    behavior for existing callers).
        float[] condContext;
        float[] uncondContext;
        if (textContext is not null)
        {
            condContext = textContext;
            uncondContext = new float[condContext.Length];
        }
        else if (_textEncoderForward is not null && _textEncoderTokenizer is not null)
        {
            (condContext, _) = QwenImageTextConditioning.Encode(_textEncoderForward, _textEncoderTokenizer, prompt);
            (uncondContext, _) = QwenImageTextConditioning.Encode(_textEncoderForward, _textEncoderTokenizer, negativePrompt ?? "");
        }
        else
        {
            int seqLen = 77;
            condContext = new float[seqLen * QwenImageModel.ContextDim];
            uncondContext = new float[seqLen * QwenImageModel.ContextDim];
        }

        // 2. Reference image latents for Qwen Image Edit
        float[]? refLatent = null;
        if (initImageRgb is not null)
        {
            using var vaeEnc = new VaeEncoder(_weights);
            refLatent = vaeEnc.Encode(initImageRgb, height, width, latentChannels: 16, seed: seed);
        }

        // 3. Initial Gaussian noise in latent space [16, latH, latW]
        var latent = SampleGaussianNoise(latC * latH * latW, seed);

        // 4. Rectified Flow-Matching Timesteps with Flow Shift s = 3.0:
        var timesteps = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float linearT = 1.0f - (float)i / steps;
            timesteps[i] = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        }

        // 5. Euler Flow trajectory loop
        for (int step = 0; step < steps; step++)
        {
            float t = timesteps[step];
            float tNext = timesteps[step + 1];
            float dt = t - tNext;

            var condVelocity = _transformer.Forward(latent, t * 1000.0f, condContext, latH, latW, refLatent);
            float[] velocity;

            if (guidance > 1.0f)
            {
                var uncondVelocity = _transformer.Forward(latent, t * 1000.0f, uncondContext, latH, latW, refLatent);
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

        // 6. Decode 16-channel latents to RGB pixels via VAE (single-frame, t=1 -- WanVaeDecoder3D
        // already returns [0,1]-clamped pixels, matching PngWriter's expected convention directly).
        var pixels = _vae.Decode(latent, 1, latH, latW)[0];

        // 7. Optional Super-Resolution Upscaling
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

        // 8. Write output PNG
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
            width: request.Width <= 0 ? 1024 : request.Width,
            height: request.Height <= 0 ? 1024 : request.Height,
            steps: request.Steps <= 0 ? 20 : request.Steps,
            guidance: request.Guidance == 1.0f ? 2.5f : request.Guidance,
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
            _textEncoderForward?.Dispose();
            _textEncoderBackend?.Dispose();
            _textEncoderModel?.Dispose();
        }
    }
}
