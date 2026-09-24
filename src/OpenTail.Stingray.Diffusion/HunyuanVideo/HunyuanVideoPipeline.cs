
namespace OpenTail.Stingray.Diffusion.HunyuanVideo;

/// <summary>
/// Native C# HunyuanVideo Text-to-Video and Text-to-Image inference pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::HUNYUAN_VIDEO
/// </summary>
public sealed class HunyuanVideoPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader _weights;
    private readonly IWeightLoader? _vaeWeights;
    private readonly HunyuanVideoModel _transformer;
    private readonly HunyuanVaeDecoder3D _vae;
    private readonly GgufModel? _textEncoderModel;
    private readonly Engine.ForwardPass? _textEncoderForward;
    private readonly GgufTokenizer? _textEncoderTokenizer;
    private readonly Cpu.CpuBackend? _textEncoderBackend;
    private TextEncoders.ClipLEncoder? _clipL;
    private TextEncoders.ClipTokenizer? _clipTokenizer;
    private bool _disposed;

    public string Architecture => "HunyuanVideo";

    public HunyuanVideoPipeline(
        IWeightLoader weights,
        HunyuanVideoModel transformer,
        HunyuanVaeDecoder3D vae,
        IWeightLoader? vaeWeights = null)
    {
        _weights = weights;
        _transformer = transformer;
        _vae = vae;
        _vaeWeights = vaeWeights;
    }

    private HunyuanVideoPipeline(
        IWeightLoader weights, HunyuanVideoModel transformer, HunyuanVaeDecoder3D vae, IWeightLoader? vaeWeights,
        GgufModel textEncoderModel, Engine.ForwardPass textEncoderForward, GgufTokenizer textEncoderTokenizer, Cpu.CpuBackend textEncoderBackend)
        : this(weights, transformer, vae, vaeWeights)
    {
        _textEncoderModel = textEncoderModel;
        _textEncoderForward = textEncoderForward;
        _textEncoderTokenizer = textEncoderTokenizer;
        _textEncoderBackend = textEncoderBackend;
    }

    /// <summary>
    /// Loads a HunyuanVideo pipeline. <paramref name="vaePath"/> must point to a real HunyuanVideo
    /// VAE checkpoint (e.g. a community `hunyuan_video_vae_*.safetensors` repack) -- the main DiT
    /// checkpoint bundles zero VAE tensors (confirmed by direct safetensors header inspection), so
    /// omitting it means <see cref="Generate"/> will throw when it reaches the decode stage rather
    /// than silently producing wrong pixels.
    /// </summary>
    public static HunyuanVideoPipeline Load(string modelPath, string? vaePath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new HunyuanVideoModel(weights, prefix: "", backend: backend);

        IWeightLoader? vaeLoader = null;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }
        else if (weights.Contains("decoder.conv_in.conv.weight") || weights.Contains("vae.decoder.conv_in.conv.weight"))
        {
            // Rare but possible: a repack that bundles VAE tensors into the main checkpoint.
            vaeLoader = weights;
        }

        var vae = new HunyuanVaeDecoder3D(vaeLoader);

        return new HunyuanVideoPipeline(weights, transformer, vae, vaeLoader == weights ? null : vaeLoader);
    }

    /// <summary>
    /// Loads a HunyuanVideo pipeline WITH real llava-llama-3-8b-v1_1 text conditioning (docs/088)
    /// -- <c>Generate</c> will use <see cref="HunyuanVideoTextConditioning.Encode"/> automatically
    /// when the caller doesn't supply an explicit <c>textContext</c>.
    /// </summary>
    public static HunyuanVideoPipeline Load(string modelPath, string textEncoderPath, string? vaePath, IComputeBackend? backend = null,
        string? clipLPath = null, string? clipTokenizerPath = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new HunyuanVideoModel(weights, prefix: "", backend: backend);

        IWeightLoader? vaeLoader = null;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
        }
        else if (weights.Contains("decoder.conv_in.conv.weight") || weights.Contains("vae.decoder.conv_in.conv.weight"))
        {
            vaeLoader = weights;
        }

        var vae = new HunyuanVaeDecoder3D(vaeLoader);

        var textEncoderModel = GgufModel.Open(textEncoderPath);
        var hp = ModelHyperparams.FromGgufMetadata(textEncoderModel.Metadata, textEncoderModel);
        var tokenizer = GgufTokenizer.FromGgufModel(textEncoderModel);
        var textEncoderBackend = new Cpu.CpuBackend();
        var textEncoderForward = new Engine.ForwardPass(textEncoderModel, textEncoderBackend, hp);

        var pipe = new HunyuanVideoPipeline(weights, transformer, vae, vaeLoader == weights ? null : vaeLoader,
            textEncoderModel, textEncoderForward, tokenizer, textEncoderBackend);
        // CLIP-L pooled output feeds the DiT's vector_in (the second text encoder of the real
        // pipeline, `text_encoder_2` in diffusers). Optional: without it vector_in is skipped.
        if (clipLPath is not null && File.Exists(clipLPath) && clipTokenizerPath is not null && File.Exists(clipTokenizerPath))
        {
            pipe._clipL = new TextEncoders.ClipLEncoder(clipLPath);
            pipe._clipTokenizer = TextEncoders.ClipTokenizer.FromFile(clipTokenizerPath);
        }
        return pipe;
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

        // 1. Text conditioning context [seqLen, 4096] -- real llava-llama-3-8b-v1_1 conditioning
        //    (HunyuanVideoTextConditioning.Encode, docs/088) when this pipeline owns a real text
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
            (condContext, _) = HunyuanVideoTextConditioning.Encode(_textEncoderForward, _textEncoderTokenizer, prompt);
            (uncondContext, _) = HunyuanVideoTextConditioning.Encode(_textEncoderForward, _textEncoderTokenizer, negativePrompt ?? "");
        }
        else
        {
            int seqLen = 256;
            condContext = new float[seqLen * HunyuanVideoModel.TextDim];
            uncondContext = new float[seqLen * HunyuanVideoModel.TextDim];
        }

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

        bool dumpDiag = Environment.GetEnvironmentVariable("STINGRAY_HUNYUAN_DUMP_LATENT") == "1";
        if (dumpDiag) Console.Error.WriteLine($"[HunyuanVideo] init noise: nanCount={latent.Count(v => !float.IsFinite(v))}/{latent.Length}");

        // CLIP-L pooled vectors for vector_in (null when no CLIP-L was loaded).
        float[]? condPooled = null, uncondPooled = null;
        if (_clipL is not null && _clipTokenizer is not null)
        {
            condPooled = _clipL.Encode(_clipTokenizer.Tokenize(prompt)).pooled;
            uncondPooled = _clipL.Encode(_clipTokenizer.Tokenize(negativePrompt ?? "")).pooled;
        }

        // Guidance-distilled checkpoints (guidance_in present) take the guidance scale as an
        // embedding -- diffusers passes guidance_scale * 1000 with true CFG off by default -- so
        // one forward per step. Running real CFG on them (as this did until 2026-09-24) is the
        // wrong algorithm and doubles the cost.
        bool distilled = _transformer.HasGuidanceEmbedding;
        float? distilledGuidance = distilled ? guidance * 1000.0f : null;

        // 4. Euler Flow trajectory loop
        for (int step = 0; step < steps; step++)
        {
            float t = timesteps[step];
            float tNext = timesteps[step + 1];
            float dt = t - tNext;

            var condVelocity = _transformer.Forward(latent, t * 1000.0f, condContext, numFrames, latH, latW, condPooled, distilledGuidance);
            if (dumpDiag) Console.Error.WriteLine($"[HunyuanVideo] step={step} t={t:F4} condVelocity nanCount={condVelocity.Count(v => !float.IsFinite(v))}/{condVelocity.Length}");
            float[] velocity;

            if (!distilled && guidance > 1.0f)
            {
                var uncondVelocity = _transformer.Forward(latent, t * 1000.0f, uncondContext, numFrames, latH, latW, uncondPooled);
                velocity = new float[condVelocity.Length];
                for (int i = 0; i < velocity.Length; i++)
                    velocity[i] = uncondVelocity[i] + guidance * (condVelocity[i] - uncondVelocity[i]);
            }
            else
            {
                velocity = condVelocity;
            }

            if (dumpDiag)
            {
                // x0 estimate = x_t - t*v. A working model gives a spatially smooth x0 (high
                // neighbour correlation along W); noise-like x0 (~0) means the DiT isn't denoising.
                int frameLen = latH * latW;
                double num = 0, den = 0, sq = 0; int cnt = 0;
                for (int ch = 0; ch < latC; ch++)
                    for (int y = 0; y < latH; y++)
                        for (int x = 0; x + 1 < latW; x++)
                        {
                            int i0 = ch * numFrames * frameLen + y * latW + x;
                            float a = latent[i0] - t * velocity[i0], b = latent[i0 + 1] - t * velocity[i0 + 1];
                            num += a * b; den += a * a; sq += a * a; cnt++;
                        }
                Console.Error.WriteLine($"[HunyuanVideo] step={step} t={t:F3} x0 std={Math.Sqrt(sq / cnt):F3} neighbourCorr={num / den:F3}");
            }

            for (int i = 0; i < latent.Length; i++)
                latent[i] -= dt * velocity[i];

            progress?.Invoke(step + 1, steps);
        }

        if (Environment.GetEnvironmentVariable("STINGRAY_HUNYUAN_DUMP_LATENT") == "1")
        {
            double sum = 0, sumSq = 0; int nanCount = 0; float min = float.MaxValue, max = float.MinValue;
            foreach (var v in latent)
            {
                if (!float.IsFinite(v)) { nanCount++; continue; }
                sum += v; sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            double mean = sum / latent.Length;
            double std = Math.Sqrt(sumSq / latent.Length - mean * mean);
            Console.Error.WriteLine($"[HunyuanVideo] pre-decode latent: mean={mean:F4} std={std:F4} min={min:F4} max={max:F4} nanCount={nanCount}/{latent.Length}");
        }

        // 5. Decode the full video latent [16, numFrames, latH, latW] to RGB pixels via the real
        // causal 3D VAE in one call -- unlike per-frame decoding, this reproduces the real
        // temporal upsampling/causal-conv context between neighboring latent frames (see
        // HunyuanVaeDecoder3D's class doc comment).
        string? latentDumpPath = Environment.GetEnvironmentVariable("STINGRAY_HUNYUAN_DUMP_LATENT_PATH");
        if (latentDumpPath is not null)
            File.WriteAllBytes(latentDumpPath, System.Runtime.InteropServices.MemoryMarshal.AsBytes(latent.AsSpan()).ToArray());

        var decodedFrames = _vae.Decode(latent, numFrames, latH, latW);
        var allFrames = new List<float[]>(decodedFrames.Count);

        foreach (var framePixels0 in decodedFrames)
        {
            var framePixels = framePixels0;

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
        numFrames = allFrames.Count;

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
            _vaeWeights?.Dispose();
            _transformer.Dispose();
            _vae.Dispose();
            _textEncoderForward?.Dispose();
            _textEncoderBackend?.Dispose();
            _textEncoderModel?.Dispose();
            _clipL?.Dispose();
        }
    }
}
