
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Full Z-Image-Turbo image generation pipeline.
///
/// Components:
///   ditDir         — transformer/ directory (multi-shard safetensors)
///                    OR single model.safetensors file
///   vaeDir         — vae/ directory (ae.safetensors or multi-shard)
///   qwenPath       — Qwen3-4B GGUF file (text encoder)
///   tokenizerPath  — tokenizer.json for Qwen3 BPE tokenizer
///
/// Model download (HuggingFace):
///   https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
///   https://huggingface.co/unsloth/Z-Image-Turbo-GGUF  (quantized DiT)
///   Qwen3-4B GGUF: https://huggingface.co/Qwen/Qwen3-4B-GGUF
///
/// Usage:
///   var pipeline = ZImagePipeline.Load(ditDir, vaeDir, qwenPath, tokenizerPath);
///   pipeline.Generate("a cat", width: 512, height: 512, output: "out.png");
/// </summary>
public sealed class ZImagePipeline : IDisposable, IDiffusionPipeline
{
    private readonly ZImageDiT _dit;
    private readonly VaeDecoder _vae;
    private readonly QwenTextEncoder _encoder;
    private readonly QwenTokenizer _tokenizer;
    private readonly ZImageParams _p;
    private string? _cachedPrompt;
    private float[]? _cachedEmbeds;
    private bool _disposed;

    private ZImagePipeline(ZImageDiT dit, VaeDecoder vae,
                            QwenTextEncoder encoder, QwenTokenizer tokenizer,
                            ZImageParams p)
    {
        _dit       = dit;
        _vae       = vae;
        _encoder   = encoder;
        _tokenizer = tokenizer;
        _p         = p;
    }

    /// <summary>
    /// Load all Z-Image pipeline components.
    /// </summary>
    /// <param name="ditPath">
    ///   Path to the DiT weights: either a directory containing safetensors shards
    ///   (e.g. the Z-Image transformer/ dir) or a single .safetensors file.
    /// </param>
    /// <param name="vaePath">
    ///   Path to the VAE weights: directory or single .safetensors file.
    /// </param>
    /// <param name="qwenPath">
    ///   Path to the Qwen3-4B GGUF file used as text encoder.
    /// </param>
    /// <param name="tokenizerPath">
    ///   Path to the Qwen3 tokenizer.json file.
    /// </param>
    /// <param name="backend">
    ///   Optional GPU compute backend. When supplied, large weight projections are
    ///   accelerated via batched SGEMM on the GPU.
    /// </param>
    public static ZImagePipeline Load(string ditPath, string vaePath,
                                       string qwenPath, string tokenizerPath,
                                       IComputeBackend? backend = null,
                                       bool useFp8Weights = false)
    {
        var p = new ZImageParams();

        IWeightLoader ditLoader;
        if (string.Equals(Path.GetExtension(ditPath), ".gguf", StringComparison.OrdinalIgnoreCase))
            ditLoader = GgufWeightLoader.Open(ditPath);
        else if (Directory.Exists(ditPath))
            ditLoader = SafetensorsLoader.OpenDirectory(ditPath);
        else
            ditLoader = SafetensorsLoader.Open(ditPath);

        var vaeLoader  = Directory.Exists(vaePath) ? SafetensorsLoader.OpenDirectory(vaePath)
                                                   : SafetensorsLoader.Open(vaePath);

        var dit       = new ZImageDiT(ditLoader, p, backend);
        var vae       = new VaeDecoder(vaeLoader, backend);
        var qwen      = new QwenTextEncoder(GgufModel.Open(qwenPath), p, backend);
        var tokenizer = QwenTokenizer.FromFile(tokenizerPath);

        return new ZImagePipeline(dit, vae, qwen, tokenizer, p);
    }

    // ── Main generation entry point ───────────────────────────────────────

    /// <summary>
    /// Generate an image from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the image.</param>
    /// <param name="width">Output image width in pixels (must be divisible by 16).</param>
    /// <param name="height">Output image height in pixels (must be divisible by 16).</param>
    /// <param name="steps">Number of denoising steps (default: 9 = 8 NFEs).</param>
    /// <param name="seed">RNG seed for reproducibility (-1 = random).</param>
    /// <param name="outputPath">Path to write the output PNG file.</param>
    /// <param name="upscaler">
    ///   Optional <see cref="RRDBNet"/> upscaler. When supplied, the decoded image is
    ///   passed through the upscaler before being written to disk. The output PNG will
    ///   be <c>Scale</c>× larger in each dimension (e.g. 512→2048 for a ×4 model).
    /// </param>
    /// <param name="upscaleBlend">
    ///   Blend factor for the upscaled result: 1.0 = full RRDB (default, sharpest),
    ///   values below 1.0 blend with a bicubic upscale to soften the output.
    ///   0.8 gives a natural look for portraits; 0.0 = pure bicubic.
    /// </param>
    /// <param name="progress">Optional progress callback (step, totalSteps).</param>
    /// <param name="statusCallback">Optional status text callback for spinner updates.</param>
    public void Generate(string prompt,
                          int width  = 512,
                          int height = 512,
                          int steps  = -1,
                          int seed   = -1,
                          string outputPath = "output.png",
                          RRDBNet? upscaler = null,
                          float upscaleBlend = 1.0f,
                          Action<int, int>? progress = null,
                          Action<string>? statusCallback = null)
    {
        if (steps < 0) steps = _p.DefaultSteps;

        int latH = height / _p.VaeScaleFactor;  // e.g. 64 for 512px
        int latW = width  / _p.VaeScaleFactor;
        int patchH = latH / _p.PatchSize;         // e.g. 32
        int patchW = latW / _p.PatchSize;
        int nImg   = patchH * patchW;

        // ── 1. Encode text ────────────────────────────────────────────────
        progress?.Invoke(0, steps + 2);
        float[] txtEmbeds;
        int[] tokenIds;
        if (_cachedPrompt == prompt && _cachedEmbeds is not null)
        {
            statusCallback?.Invoke("Using cached text embeddings…");
            txtEmbeds = _cachedEmbeds;
            tokenIds  = _tokenizer.EncodeWithTemplate(prompt);
        }
        else
        {
            statusCallback?.Invoke("Encoding text prompt (Qwen3-4B)…");
            tokenIds  = _tokenizer.EncodeWithTemplate(prompt);
            int totalEncLayers = _p.QwenEncoderLayer + 1;
            txtEmbeds = _encoder.Encode(tokenIds,
                encodeProgress: (layer, _) =>
                    statusCallback?.Invoke($"Encoding text: layer {layer + 1}/{totalEncLayers}…"));
            _cachedPrompt  = prompt;
            _cachedEmbeds  = txtEmbeds;
        }
        int nTxt = tokenIds.Length;

        // ── 2. Build position IDs ─────────────────────────────────────────
        int[] txtPosIds = ZImageRoPE.TextPosIds(nTxt);
        int[] imgPosIds = ZImageRoPE.ImagePosIds(nTxt, patchH, patchW);

        // ── 3. Sample noise latent ────────────────────────────────────────
        var noiseLatent = EulerFlowScheduler.SampleNoise(_p.LatentChannels * latH * latW, seed);

        // Pack into patches [nImg, 64] — Z-Image uses spatial-first (ky,kx,ch) ordering
        var noisePacked = EulerFlowScheduler.PackLatentSpatialFirst(noiseLatent, _p.LatentChannels, latH, latW, _p.PatchSize);

        // ── 4. Denoise with Euler flow scheduler ──────────────────────────
        // Z-Image-Turbo uses shift=3 (from scheduler_config.json)
        var scheduler = EulerFlowScheduler.Linear(steps, shift: 3.0f);

        // sign: +1, NOT the scheduler's default -1 (2026-09-12 regression fix). EulerFlowScheduler
        // is shared with FLUX.1 (ImagePipeline.cs), whose Euler-integration sign was corrected in
        // commit 0c52407 (verified against real diffusers sampling.py) -- but that fix silently
        // flipped the SAME shared method's behavior for every other caller too, and Z-Image's
        // S3-DiT returns its velocity prediction in the OPPOSITE convention from FLUX's DiT.
        // Root-caused by dumping the pre-VAE latent (std ballooned from a healthy ~1.3 to ~2.2,
        // range ±10 instead of ±4 -- diverging away from a clean image, not converging) and
        // confirming empirically: sign=+1 restores both a healthy latent distribution and a real,
        // correct, coherent output image (see docs/00-current-work.md's 2026-09-12 entry for the
        // before/after latent stats and image). Do not "fix" this back to -1 to match FLUX --
        // the two DiTs are architecturally independent and were verified to need opposite signs.
        var resultPacked = scheduler.Denoise(noisePacked, (patches, t) =>
        {
            return _dit.Forward(patches, imgPosIds, txtEmbeds, txtPosIds, t);
        }, (step, total) =>
        {
            progress?.Invoke(step + 1, steps + 2);
            statusCallback?.Invoke($"Denoising step {step + 1}/{total}…");
        }, sign: 1);

        // ── 5. Unpack and decode through VAE ──────────────────────────────
        progress?.Invoke(steps + 1, steps + 2);
        var latent = EulerFlowScheduler.UnpackLatentSpatialFirst(resultPacked, _p.LatentChannels, latH, latW, _p.PatchSize);

        // VaeDecoder.Decode() applies the FLUX VAE scaling internally:
        //   z = latent / 0.3611 + 0.1159
        // so we pass the raw scheduler output directly.
        if (Environment.GetEnvironmentVariable("STINGRAY_ZIMAGE_DUMP_LATENT") == "1")
        {
            float min = float.MaxValue, max = float.MinValue, sum = 0, sumSq = 0;
            foreach (var v in latent)
            {
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / latent.Length;
            float variance = sumSq / latent.Length - mean * mean;
            Console.Error.WriteLine($"[ZImage] pre-VAE latent stats: min={min:F4} max={max:F4} mean={mean:F4} std={MathF.Sqrt(MathF.Max(variance, 0)):F4} n={latent.Length}");
        }
        float[] rgb = _vae.Decode(latent, latH, latW);

        // ── 6. Write PNG ──────────────────────────────────────────────────
        int outWidth = width, outHeight = height;
        if (upscaler is not null)
        {
            statusCallback?.Invoke($"Upscaling {width}×{height} → {width * upscaler.Scale}×{height * upscaler.Scale} (RRDBNet ×{upscaler.Scale})…");
            var preUpscaleRgb = rgb;
            (rgb, outWidth, outHeight) = upscaler.Upscale(rgb, width, height);
            if (upscaleBlend < 1f)
            {
                // Blend RRDB output with bicubic to soften aggressive texture enhancement.
                var bicubic = DiffusionOps.UpsampleBicubic(preUpscaleRgb, 3, height, width, outHeight, outWidth);
                rgb = DiffusionOps.BlendRgb(rgb, bicubic, upscaleBlend);
            }
        }
        PngWriter.Write(outputPath, rgb, outWidth, outHeight);
        progress?.Invoke(steps + 2, steps + 2);
    }

    /// <summary>IDiffusionPipeline adapter — delegates to <see cref="Generate(string,int,int,int,int,string,RRDBNet?,float,Action{int,int}?,Action{string}?)"/> unchanged.</summary>
    void IDiffusionPipeline.Generate(ImageGenerationRequest request) => Generate(
        request.Prompt, request.Width, request.Height,
        steps: request.Steps, seed: request.Seed, outputPath: request.OutputPath,
        upscaler: request.Upscaler, upscaleBlend: request.UpscaleBlend,
        progress: request.Progress, statusCallback: request.StatusCallback);

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _dit.Dispose();
            _vae.Dispose();
            _encoder.Dispose();
        }
    }
}
