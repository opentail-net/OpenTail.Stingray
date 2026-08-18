using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.SDXL;

/// <summary>
/// Native C# Stable Diffusion XL (SDXL) end-to-end inference pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::SDXL
/// </summary>
public sealed class SdxlPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly ClipLEncoder _clipL;
    private readonly OpenClipGEncoder _clipG;
    private readonly SdxlUNet2DConditionModel _unet;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public string Architecture => "SDXL";

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
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

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
        float upscaleBlend = 1.0f,
        float[]? initImageRgb = null,
        float strength = 0.75f)
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
        var scheduler = new EulerDiscreteScheduler(steps, schedulerType: schedulerType);
        int startStep = 0;
        float[] latent;

        if (initImageRgb is not null)
        {
            using var vaeEnc = new VaeEncoder(_weights);
            var cleanLatent = vaeEnc.Encode(initImageRgb, height, width, latentChannels: 4, seed: seed);
            startStep = (int)Math.Clamp(steps * (1.0f - strength), 0, steps - 1);
            var noise = scheduler.CreateInitialLatents(1, latC, latH, latW, seed);
            latent = scheduler.CreateNoisyLatent(cleanLatent, noise, startStep);
        }
        else
        {
            latent = scheduler.CreateInitialLatents(1, latC, latH, latW, seed);
        }

        // 4. Denoising loop
        var denoised = scheduler.Denoise(latent, (scaledLatent, timestep) =>
        {
            var condPred = _unet.Forward(scaledLatent, timestep, condContext, condAddEmbeds, latH, latW);
            var uncondPred = _unet.Forward(scaledLatent, timestep, uncondContext, uncondAddEmbeds, latH, latW);
            return scheduler.CombineGuidance(condPred, uncondPred, guidance);
        }, progress, startStep: startStep);

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
        int seqLen = 77;
        int dimL = 768;
        int dimG = 1280;
        int outDim = dimL + dimG;
        var concat = new float[seqLen * outDim];

        for (int i = 0; i < seqLen; i++)
        {
            Array.Copy(hiddenL, i * dimL, concat, i * outDim, dimL);
            Array.Copy(hiddenG, i * dimG, concat, i * outDim + dimL, dimG);
        }
        return concat;
    }

    public static float[] BuildAddEmbeddings(float[] pooledText, int origH, int origW, int cropTop, int cropLeft, int targetH, int targetW)
    {
        // SDXL addition condition vector layout: [pooledText(1280), origH(256), origW(256), cropTop(256), cropLeft(256), targetH(256), targetW(256)] = 2816
        var addEmbed = new float[1280 + 6 * 256];
        Array.Copy(pooledText, 0, addEmbed, 0, Math.Min(1280, pooledText.Length));

        int offset = 1280;
        AppendFourierEmbedding(addEmbed, offset + 0 * 256, origH, 256);
        AppendFourierEmbedding(addEmbed, offset + 1 * 256, origW, 256);
        AppendFourierEmbedding(addEmbed, offset + 2 * 256, cropTop, 256);
        AppendFourierEmbedding(addEmbed, offset + 3 * 256, cropLeft, 256);
        AppendFourierEmbedding(addEmbed, offset + 4 * 256, targetH, 256);
        AppendFourierEmbedding(addEmbed, offset + 5 * 256, targetW, 256);

        return addEmbed;
    }

    private static void AppendFourierEmbedding(float[] dst, int dstOffset, float val, int dim)
    {
        int half = dim / 2;
        float factor = 10000.0f;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(factor) * i / half);
            dst[dstOffset + i] = MathF.Cos(val * freq);
            dst[dstOffset + half + i] = MathF.Sin(val * freq);
        }
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
            _weights.Dispose();
            _unet.Dispose();
            _vae.Dispose();
        }
    }
}

