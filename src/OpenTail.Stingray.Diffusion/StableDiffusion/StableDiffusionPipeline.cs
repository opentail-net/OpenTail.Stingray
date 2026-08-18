using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Native Stable Diffusion 1.5 Image Generation Pipeline.
/// Supports both CPU and GPU (Vulkan/CUDA via IComputeBackend).
/// </summary>
public sealed class StableDiffusionPipeline : IDisposable, IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly ClipTokenizer _tokenizer;
    private readonly ClipLEncoder _textEncoder;
    private readonly UNet2DConditionModel _unet;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public StableDiffusionPipeline(
        IWeightLoader weights,
        ClipTokenizer tokenizer,
        ClipLEncoder textEncoder,
        UNet2DConditionModel unet,
        VaeDecoder vae)
    {
        _weights = weights;
        _tokenizer = tokenizer;
        _textEncoder = textEncoder;
        _unet = unet;
        _vae = vae;
    }

    /// <summary>
    /// Loads a Stable Diffusion 1.5 pipeline from a unified checkpoint.
    /// </summary>
    public static StableDiffusionPipeline Load(string modelPath, string? tokenizerPath = null, IComputeBackend? backend = null)
    {
        var weights = SafetensorsLoader.Open(modelPath);

        tokenizerPath ??= Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", "clip_tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            string candidate = Path.Combine(AppContext.BaseDirectory, "models", "clip_tokenizer.json");
            if (File.Exists(candidate)) tokenizerPath = candidate;
        }

        var tokenizer = ClipTokenizer.FromFile(tokenizerPath);

        var clipLoader = new PrefixWeightLoader(weights, "cond_stage_model.transformer.");
        var clip = new ClipLEncoder(clipLoader);

        var unetLoader = new PrefixWeightLoader(weights, "model.diffusion_model.");
        var unet = new UNet2DConditionModel(unetLoader, prefix: "", backend: backend);

        var vaeLoader = new PrefixWeightLoader(weights, "first_stage_model.");
        var vae = new VaeDecoder(vaeLoader, backend: backend);

        return new StableDiffusionPipeline(weights, tokenizer, clip, unet, vae);
    }

    /// <summary>
    /// Generates an image from a text prompt and saves to disk as PNG.
    /// </summary>
    public void Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int steps = 20,
        float guidance = 7.5f,
        int seed = -1,
        DiffusionSchedulerType schedulerType = DiffusionSchedulerType.Euler,
        string outputPath = "output.png",
        Action<int, int>? progress = null,
        RRDBNet? upscaler = null,
        float upscaleBlend = 1.0f)
    {
        if (width % 8 != 0 || height % 8 != 0)
            throw new ArgumentException($"Width and height must be divisible by 8 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;
        int latC = 4;

        // 1. Text Conditioning:
        var condTokens = _tokenizer.Tokenize(prompt);
        var (condContext, _) = _textEncoder.Encode(condTokens);

        // 2. Negative / Unconditional Conditioning:
        var uncondTokens = _tokenizer.Tokenize(negativePrompt ?? "");
        var (uncondContext, _) = _textEncoder.Encode(uncondTokens);

        // 3. Scheduler & Initial Noise:
        var scheduler = new EulerDiscreteScheduler(steps, schedulerType);
        var latent = scheduler.SampleNoise(latC * latH * latW, seed);

        // 4. Denoising loop with 2-pass CFG:
        var denoised = scheduler.Denoise(latent, (scaledLatent, timestep) =>
        {
            var condPred = _unet.Forward(scaledLatent, timestep, condContext, latH, latW);
            var uncondPred = _unet.Forward(scaledLatent, timestep, uncondContext, latH, latW);
            return scheduler.CombineGuidance(condPred, uncondPred, guidance);
        }, progress);

        // 5. Decode latent to RGB pixels via VAE:
        var pixels = _vae.Decode(denoised, latH, latW);

        // 6. Optional Super-Resolution Upscaler (4x):
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

        // 7. Write to PNG:
        PngWriter.Write(outputPath, pixels, outWidth, outHeight);
    }

    public void Generate(ImageGenerationRequest request)
    {
        Generate(
            request.Prompt,
            negativePrompt: null,
            width: request.Width <= 0 ? 512 : request.Width,
            height: request.Height <= 0 ? 512 : request.Height,
            steps: request.Steps <= 0 ? 20 : request.Steps,
            guidance: request.Guidance == 1.0f ? 7.5f : request.Guidance,
            seed: request.Seed,
            schedulerType: DiffusionSchedulerType.Euler,
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
            _textEncoder.Dispose();
            _unet.Dispose();
            _vae.Dispose();
            _weights.Dispose();
        }
    }
}
