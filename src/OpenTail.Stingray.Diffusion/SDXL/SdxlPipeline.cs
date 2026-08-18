using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.SDXL;

/// <summary>
/// Stable Diffusion XL (SDXL) Image Generation Pipeline.
/// Orchestrates dual text tokenization (CLIP-L + OpenCLIP-bigG), micro-conditioning embeddings,
/// 3-level SDXL UNet denoising, and 4-channel VAE decoding.
/// </summary>
public sealed class SdxlPipeline : IDisposable, IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly ClipLEncoder _clipL;
    private readonly OpenClipGEncoder _clipG;
    private readonly SdxlUNet2DConditionModel _unet;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public SdxlPipeline(
        IWeightLoader weights,
        ClipTokenizer clipTokenizer,
        ClipLEncoder clipL,
        OpenClipGEncoder clipG,
        SdxlUNet2DConditionModel unet,
        VaeDecoder vae)
    {
        _weights = weights;
        _clipTokenizer = clipTokenizer;
        _clipL = clipL;
        _clipG = clipG;
        _unet = unet;
        _vae = vae;
    }

    public static SdxlPipeline Load(string modelPath, string? tokenizerPath = null, IComputeBackend? backend = null)
    {
        var weights = SafetensorsLoader.Open(modelPath);

        tokenizerPath ??= Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", "clip_tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            string candidate = Path.Combine(AppContext.BaseDirectory, "models", "clip_tokenizer.json");
            if (File.Exists(candidate)) tokenizerPath = candidate;
        }

        var tokenizer = ClipTokenizer.FromFile(tokenizerPath);

        var clipLLoader = new PrefixWeightLoader(weights, "conditioner.embedders.0.transformer.");
        if (weights.Contains("text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight"))
            clipLLoader = new PrefixWeightLoader(weights, "text_encoders.clip_l.transformer.");
        var clipL = new ClipLEncoder(clipLLoader);

        var clipGLoader = new PrefixWeightLoader(weights, "conditioner.embedders.1.model.");
        if (weights.Contains("text_encoders.clip_g.transformer.text_model.embeddings.token_embedding.weight"))
            clipGLoader = new PrefixWeightLoader(weights, "text_encoders.clip_g.transformer.");
        var clipG = new OpenClipGEncoder(clipGLoader);

        var unetLoader = new PrefixWeightLoader(weights, "model.diffusion_model.");
        var unet = new SdxlUNet2DConditionModel(unetLoader, prefix: "", backend: backend);

        var vaeLoader = new PrefixWeightLoader(weights, "first_stage_model.");
        var vae = new VaeDecoder(vaeLoader, backend: backend);

        return new SdxlPipeline(weights, tokenizer, clipL, clipG, unet, vae);
    }

    public void Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 1024,
        int height = 1024,
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

        // 1. Dual text conditioning (CLIP-L [77, 768] + OpenCLIP-bigG [77, 1280] -> [77, 2048])
        var condTokens = _clipTokenizer.Tokenize(prompt);
        var (condHiddenL, _) = _clipL.Encode(condTokens);
        var (condHiddenG, condPooledG) = _clipG.Encode(condTokens);
        var condContext = ConcatContext(condHiddenL, condHiddenG);

        var uncondTokens = _clipTokenizer.Tokenize(negativePrompt ?? "");
        var (uncondHiddenL, _) = _clipL.Encode(uncondTokens);
        var (uncondHiddenG, uncondPooledG) = _clipG.Encode(uncondTokens);
        var uncondContext = ConcatContext(uncondHiddenL, uncondHiddenG);

        // 2. Micro-conditioning addition embeddings [2816]
        var condAddEmbeds = BuildAddEmbeddings(condPooledG, height, width, 0, 0, height, width);
        var uncondAddEmbeds = BuildAddEmbeddings(uncondPooledG, height, width, 0, 0, height, width);

        // 3. Scheduler & Noise
        var scheduler = new EulerDiscreteScheduler(steps, schedulerType);
        var latent = scheduler.SampleNoise(latC * latH * latW, seed);

        // 4. Denoising loop
        var denoised = scheduler.Denoise(latent, (scaledLatent, timestep) =>
        {
            var condPred = _unet.Forward(scaledLatent, timestep, condContext, condAddEmbeds, latH, latW);
            var uncondPred = _unet.Forward(scaledLatent, timestep, uncondContext, uncondAddEmbeds, latH, latW);
            return scheduler.CombineGuidance(condPred, uncondPred, guidance);
        }, progress);

        // 5. VAE Decode
        var pixels = _vae.Decode(denoised, latH, latW);

        // 6. Optional Upscaler
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

        PngWriter.Write(outputPath, pixels, outWidth, outHeight);
    }

    private static float[] ConcatContext(float[] hiddenL, float[] hiddenG)
    {
        // Concatenate [77, 768] + [77, 1280] -> [77, 2048]
        var cat = new float[77 * 2048];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(hiddenL, t * 768, cat, t * 2048, 768);
            Array.Copy(hiddenG, t * 1280, cat, t * 2048 + 768, 1280);
        }
        return cat;
    }

    private static float[] BuildAddEmbeddings(float[] pooled, int origH, int origW, int cropH, int cropW, int targetH, int targetW)
    {
        var result = new float[2816];
        Array.Copy(pooled, 0, result, 0, 1280);

        int[] coords = [origH, origW, cropH, cropW, targetH, targetW];
        for (int i = 0; i < coords.Length; i++)
        {
            var sinEmb = ComputeFourierCoordinate(coords[i], 256);
            Array.Copy(sinEmb, 0, result, 1280 + i * 256, 256);
        }

        return result;
    }

    private static float[] ComputeFourierCoordinate(float val, int dim)
    {
        var emb = new float[dim];
        int half = dim / 2;
        float logMaxPeriod = MathF.Log(10000.0f);

        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMaxPeriod * i / half);
            float arg = val * freq;
            emb[i]        = MathF.Cos(arg);
            emb[half + i] = MathF.Sin(arg);
        }
        return emb;
    }

    public void Generate(ImageGenerationRequest request)
    {
        Generate(
            request.Prompt,
            negativePrompt: null,
            width: request.Width <= 0 ? 1024 : request.Width,
            height: request.Height <= 0 ? 1024 : request.Height,
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
            _clipL.Dispose();
            _clipG.Dispose();
            _unet.Dispose();
            _vae.Dispose();
            _weights.Dispose();
        }
    }
}

