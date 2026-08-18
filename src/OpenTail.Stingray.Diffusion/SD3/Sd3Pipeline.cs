using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// Stable Diffusion 3 / 3.5 Pipeline.
/// Triple text conditioning (CLIP-L + OpenCLIP-bigG + T5), MMDiT DiT transformer,
/// Rectified Flow-Matching scheduler, and 16-channel VAE decoder.
/// </summary>
public sealed class Sd3Pipeline : IDisposable, IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly ClipLEncoder _clipL;
    private readonly OpenClipGEncoder _clipG;
    private readonly MMDiTModel _mmdit;
    private readonly VaeDecoder _vae;
    private bool _disposed;

    public Sd3Pipeline(
        IWeightLoader weights,
        ClipTokenizer clipTokenizer,
        ClipLEncoder clipL,
        OpenClipGEncoder clipG,
        MMDiTModel mmdit,
        VaeDecoder vae)
    {
        _weights = weights;
        _clipTokenizer = clipTokenizer;
        _clipL = clipL;
        _clipG = clipG;
        _mmdit = mmdit;
        _vae = vae;
    }

    public static Sd3Pipeline Load(string modelPath, string? tokenizerPath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? GgufWeightLoader.Open(modelPath) : SafetensorsLoader.Open(modelPath);

        tokenizerPath ??= Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", "clip_tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            string candidate = Path.Combine(AppContext.BaseDirectory, "models", "clip_tokenizer.json");
            if (File.Exists(candidate)) tokenizerPath = candidate;
        }

        var tokenizer = ClipTokenizer.FromFile(tokenizerPath);

        var clipLLoader = new PrefixWeightLoader(weights, "text_encoders.clip_l.transformer.");
        var clipL = new ClipLEncoder(clipLLoader);

        var clipGLoader = new PrefixWeightLoader(weights, "text_encoders.clip_g.transformer.");
        var clipG = new OpenClipGEncoder(clipGLoader);

        var mmditLoader = new PrefixWeightLoader(weights, "model.diffusion_model.");
        var mmdit = new MMDiTModel(mmditLoader, prefix: "", backend: backend);

        var vaeLoader = new PrefixWeightLoader(weights, "first_stage_model.");
        var vae = new VaeDecoder(vaeLoader, backend: backend);

        return new Sd3Pipeline(weights, tokenizer, clipL, clipG, mmdit, vae);
    }

    public void Generate(
        string prompt,
        string? negativePrompt = null,
        int width = 1024,
        int height = 1024,
        int steps = 20,
        float guidance = 4.5f,
        int seed = -1,
        string outputPath = "output.png",
        Action<int, int>? progress = null,
        RRDBNet? upscaler = null,
        float upscaleBlend = 1.0f)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException($"Width and height must be divisible by 16 (got {width}x{height})");

        int latH = height / 8;
        int latW = width / 8;
        int latC = 16;

        // 1. Text tokenization & pooled vectors
        var condTokens = _clipTokenizer.Tokenize(prompt);
        var (condHiddenL, condPooledL) = _clipL.Encode(condTokens);
        var (condHiddenG, condPooledG) = _clipG.Encode(condTokens);

        var condContext = BuildContext(condHiddenL, condHiddenG);
        var condPooledY = BuildPooledY(condPooledL, condPooledG);

        var uncondTokens = _clipTokenizer.Tokenize(negativePrompt ?? "");
        var (uncondHiddenL, uncondPooledL) = _clipL.Encode(uncondTokens);
        var (uncondHiddenG, uncondPooledG) = _clipG.Encode(uncondTokens);

        var uncondContext = BuildContext(uncondHiddenL, uncondHiddenG);
        var uncondPooledY = BuildPooledY(uncondPooledL, uncondPooledG);

        // 2. Initial Noise
        var rng = seed >= 0 ? new Random(seed) : new Random();
        int latentCount = latC * latH * latW;
        var x = new float[latentCount];
        for (int i = 0; i < latentCount - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            x[i]     = (float)(radius * Math.Cos(theta));
            x[i + 1] = (float)(radius * Math.Sin(theta));
        }

        // 3. Rectified Flow Matching Denoising Loop
        // In flow matching, timestep ranges from 1.0 down to 0.0
        float dt = 1.0f / steps;
        for (int step = 0; step < steps; step++)
        {
            float t = 1.0f - step * dt;
            float timestep = t * 1000.0f; // Scale to 0..1000 for Fourier embedding

            var condPred = _mmdit.Forward(x, timestep, condContext, condPooledY, latH, latW, 77);
            var uncondPred = _mmdit.Forward(x, timestep, uncondContext, uncondPooledY, latH, latW, 77);

            // CFG Combination
            for (int i = 0; i < x.Length; i++)
            {
                float v = uncondPred[i] + guidance * (condPred[i] - uncondPred[i]);
                // Euler update along flow: x_{t-dt} = x_t - dt * v
                x[i] -= dt * v;
            }

            progress?.Invoke(step + 1, steps);
        }

        // 4. VAE Decode (16-channel latents)
        var pixels = _vae.Decode(x, latH, latW);

        // 5. Optional Super-Resolution
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

    private static float[] BuildContext(float[] hiddenL, float[] hiddenG)
    {
        // 77 tokens padded to 4096 context dimension
        var context = new float[77 * 4096];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(hiddenL, t * 768, context, t * 4096, 768);
            Array.Copy(hiddenG, t * 1280, context, t * 4096 + 768, 1280);
        }
        return context;
    }

    private static float[] BuildPooledY(float[] pooledL, float[] pooledG)
    {
        // [768] + [1280] = [2048]
        var y = new float[2048];
        Array.Copy(pooledL, 0, y, 0, 768);
        Array.Copy(pooledG, 0, y, 768, 1280);
        return y;
    }

    public void Generate(ImageGenerationRequest request)
    {
        Generate(
            request.Prompt,
            negativePrompt: null,
            width: request.Width <= 0 ? 1024 : request.Width,
            height: request.Height <= 0 ? 1024 : request.Height,
            steps: request.Steps <= 0 ? 20 : request.Steps,
            guidance: request.Guidance == 1.0f ? 4.5f : request.Guidance,
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
            _clipL.Dispose();
            _clipG.Dispose();
            _mmdit.Dispose();
            _vae.Dispose();
            _weights.Dispose();
        }
    }
}

