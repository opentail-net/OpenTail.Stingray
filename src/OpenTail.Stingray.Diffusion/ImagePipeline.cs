using System.Diagnostics;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Full FLUX.1 image generation pipeline.
///
/// Components loaded:
///   ditPath        — FLUX DiT weights as GGUF (.gguf)
///   vaePath        — FLUX VAE as safetensors (.safetensors)
///   clipPath       — CLIP-L encoder as safetensors (.safetensors)
///   clipTokenizerPath — tokenizer.json for CLIP BPE tokenizer
///   t5Path         — T5-XXL encoder as safetensors (.safetensors)
///   t5TokenizerPath — tokenizer.json for T5 unigram tokenizer
///
/// Usage:
///   var pipeline = ImagePipeline.Load(...);
///   pipeline.Generate("a cat", width: 512, height: 512, steps: 4, output: "out.png");
/// </summary>
public sealed class ImagePipeline : IDisposable, IDiffusionPipeline
{
    private readonly FluxDiT _dit;
    private readonly VaeDecoder _vae;
    private readonly ClipLEncoder _clip;
    private readonly T5Encoder _t5;
    private readonly ClipTokenizer _clipTok;
    private readonly T5Tokenizer _t5Tok;
    private readonly FluxParams _params;
    private readonly IComputeBackend _backend;
    private readonly bool _ownsBackend;
    private bool _disposed;

    private ImagePipeline(FluxDiT dit, VaeDecoder vae, ClipLEncoder clip, T5Encoder t5,
                           ClipTokenizer clipTok, T5Tokenizer t5Tok,
                           FluxParams p, IComputeBackend backend, bool ownsBackend)
    {
        _dit         = dit;
        _vae         = vae;
        _clip        = clip;
        _t5          = t5;
        _clipTok     = clipTok;
        _t5Tok       = t5Tok;
        _params      = p;
        _backend     = backend;
        _ownsBackend = ownsBackend;
    }

    /// <summary>
    /// Load all pipeline components. All paths are required unless noted.
    /// </summary>
    public static ImagePipeline Load(
        string ditPath,
        string vaePath,
        string clipPath,
        string clipTokenizerPath,
        string t5Path,
        string t5TokenizerPath,
        IComputeBackend? backend = null)
    {
        var model       = GgufModel.Open(ditPath);
        var meta        = model.Metadata;
        var p           = FluxParams.FromMetadata(meta);
        bool ownsBackend = backend is null;
        var compBackend = backend ?? new CpuBackend();
        var dit         = new FluxDiT(model, p, compBackend);
        var vae         = new VaeDecoder(SafetensorsLoader.Open(vaePath), compBackend);
        var clip        = new ClipLEncoder(clipPath);
        var t5          = new T5Encoder(t5Path);
        var clipTok     = ClipTokenizer.FromFile(clipTokenizerPath);
        var t5Tok       = T5Tokenizer.FromFile(t5TokenizerPath, maxLen: 256); // FLUX's real T5 sequence length (see Generate())
        return new ImagePipeline(dit, vae, clip, t5, clipTok, t5Tok, p, compBackend, ownsBackend);
    }

    /// <summary>
    /// Generate an image from a text prompt and save as PNG.
    /// </summary>
    /// <param name="prompt">Text description of the desired image.</param>
    /// <param name="width">Output width in pixels (must be divisible by 16).</param>
    /// <param name="height">Output height in pixels (must be divisible by 16).</param>
    /// <param name="steps">Denoising steps (4 = schnell quality, 20-28 = dev quality).</param>
    /// <param name="guidance">Guidance scale (1.0 for schnell; 3.5 for dev).</param>
    /// <param name="seed">Random seed (-1 = random).</param>
    /// <param name="outputPath">Path to write the output PNG.</param>
    /// <param name="progress">Optional progress callback (step, totalSteps).</param>
    /// <param name="upscaler">
    ///   Optional <see cref="RRDBNet"/> upscaler. When supplied, the decoded image is
    ///   passed through the upscaler before being written to disk. The output PNG will
    ///   be <c>Scale</c>× larger in each dimension (e.g. 512→2048 for a ×4 model).
    /// </param>
    /// <param name="upscaleBlend">
    ///   Blend factor: 1.0 = full RRDB (sharpest), &lt;1.0 blends with bicubic to soften.
    ///   0.8 gives a natural look for portraits.
    /// </param>
    public void Generate(
        string prompt,
        int width = 512, int height = 512,
        int steps = 4, float guidance = 1.0f,
        int seed = -1,
        string outputPath = "output.png",
        Action<int, int>? progress = null,
        RRDBNet? upscaler = null,
        float upscaleBlend = 1.0f)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException("Width and height must be divisible by 16.");

        int latH = height / _params.VaeScaleFactor;
        int latW = width  / _params.VaeScaleFactor;
        int latC = _params.LatentChannels;  // 16

        // TEMPORARY diagnostic instrumentation (perf-sweep Phase 12, docs/perf-sweep-plan.md) for
        // FLUX.1-schnell's unexplained ~3% Vulkan speedup vs CPU -- no STINGRAY_PROFILE_DECODE-
        // equivalent exists for diffusion pipelines, so this reuses the same env var and mirrors
        // AceStepPipeline's own gated Stopwatch pattern. Remove once the real bottleneck is found.
        bool profEnabled = Environment.GetEnvironmentVariable("STINGRAY_PROFILE_DECODE") == "1";
        var swTotal = profEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;
        var sw = profEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;

        // ── 1. Encode text ────────────────────────────────────────────────
        var clipTokens = _clipTok.Tokenize(prompt);
        var (_, _, pooledEmbed) = _clip.Encode(clipTokens);   // [768]
        double msClip = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        // Real FLUX (both the vendored diffusers `_get_t5_prompt_embeds` — `padding="max_length"`,
        // `max_length=max_sequence_length` — and the real C++ reference `stable-diffusion.cpp`'s
        // `FluxCLIPEmbedder` — `chunk_len = 256`) always pads the T5 sequence to a FIXED length
        // before encoding, feeding the padding-token embeddings into the DiT's attention UNMASKED
        // (neither reference applies an attention mask for the base FLUX pipeline). This project's
        // implementation previously encoded only the real (short, ~7-9 token) prompt length —
        // found 2026-09-13 while chasing the long-standing "repeating tiled background" artifact
        // (docs/056-flux-tiling-artifact-handoff.md) as a genuine, previously-unexamined structural
        // divergence from both references.
        const int T5MaxSequenceLength = 256; // matches stable-diffusion.cpp's FluxCLIPEmbedder::chunk_len
        var t5TokensRaw = _t5Tok.Tokenize(prompt);
        var t5Tokens = new int[T5MaxSequenceLength];
        Array.Copy(t5TokensRaw, t5Tokens, Math.Min(t5TokensRaw.Length, T5MaxSequenceLength));
        // Remaining entries stay 0 (T5's <pad> token id), matching real T5 padding.
        // T5 device: on an integrated GPU the CPU encode is faster (measured 2026-09-24, FLUX.1 512²:
        // 14.8s CPU vs 32.4s on the Vega 8 iGPU, which has to take ~9.5 GB of T5 weights out of the
        // same shared RAM for a single encode). A discrete GPU keeps the GPU encode. Override with
        // STINGRAY_FLUX_T5_DEVICE=cpu|gpu.
        string? t5Device = Environment.GetEnvironmentVariable("STINGRAY_FLUX_T5_DEVICE");
        bool t5OnGpu = _backend is not CpuBackend && _backend is IVisionOpsBackend
            && (t5Device == "gpu" || (t5Device != "cpu" && _backend is not OpenTail.Stingray.Vulkan.VulkanBackend { IsIntegratedGpu: true }));
        var txtEmbeds = t5OnGpu
            ? _t5.EncodeGpu(t5Tokens, (IVisionOpsBackend)_backend)
            : _t5.Encode(t5Tokens);             // [seq, 4096]
        // Free T5's ~19GB FP32 host cache (and GPU weights, if any) before the DiT runs: on GPU it
        // collides with the ~24GB FP16 DiT upload; on CPU it pushes a 32GB machine into paging
        // alongside the mmap'd DiT + its repacked Q4_K cache. T5 is re-read on the next Generate.
        _t5.ReleaseMemory();
        int nTxt = t5Tokens.Length;
        double msT5 = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        // ── 2. Build position ids ─────────────────────────────────────────
        int pH = latH / _params.PatchSize;
        int pW = latW / _params.PatchSize;
        int nImg = pH * pW;

        var imgIds = Flux2DRoPE.ImagePatchIds(pH, pW);     // [nImg, 2]
        var txtIds = new int[nTxt * 2];                    // all zeros

        // ── 3. Sample initial noise ───────────────────────────────────────
        var noise = EulerFlowScheduler.SampleNoise(latC * latH * latW, seed);

        // Pack noise into patch sequence
        var noisePacked = EulerFlowScheduler.PackLatent(noise, latC, latH, latW, _params.PatchSize);

        // ── 4. Denoising loop ─────────────────────────────────────────────
        float timeShift = _params.HasGuidanceIn ? 3.0f : 1.0f; // dev vs schnell
        var scheduler = EulerFlowScheduler.Linear(steps, timeShift);

        double msSetup = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        float[] denoised;
        if (_backend is not CpuBackend && _backend is IVisionOpsBackend visionOps && _backend is IImageOpsBackend imageOps)
        {
            int d = _params.HiddenSize;
            int nSeq = nTxt + nImg;
            var allIds = new int[nSeq * 2];
            imgIds.CopyTo(allIds, nTxt * 2);
            var (ropeC, ropeS) = Flux2DRoPE.BuildFreqs(allIds, nSeq, _params.HeadDim);
            var (imgRopeC, imgRopeS) = Flux2DRoPE.BuildFreqs(imgIds, nImg, _params.HeadDim);

            using var ws = new FluxGpuWorkspace(_backend, nSeq, nImg, nTxt, d, imgRopeC, imgRopeS, ropeC, ropeS);
            var weights = _dit.GetOrCreateGpuWeights();

            // Upload initial noise to ws.Latent
            using var initNoiseGpu = _backend.Upload(noisePacked, TensorShape.D2(nImg, _params.InChannels), exact: true);
            imageOps.ScaleInPlace(ws.Latent, 0f);
            _backend.AddInPlace(ws.Latent, initNoiseGpu);

            // Precompute text projections on GPU once (invariant across steps)
            _dit.PrecomputeTxtGpu(txtEmbeds, ws, weights, visionOps);

            var timesteps = scheduler.Timesteps;
            int nSteps = timesteps.Length;
            var stepStopwatch = Stopwatch.StartNew();
            for (int i = 0; i < nSteps; i++)
            {
                stepStopwatch.Restart();
                float t = timesteps[i];
                float tNext = (i + 1 < nSteps) ? timesteps[i + 1] : 0f;
                float dt = t - tNext;
                const float sign = -1f;

                // NOTE: batching is done per-block INSIDE ForwardGpu (see FluxDiT.cs), not as one
                // BeginBatch/EndBatch spanning the whole step. A single step-wide command buffer
                // (~160s of uninterrupted GPU compute, one unbounded vkWaitForFences) was found to
                // freeze the desktop UI on this shared iGPU (the same engine DWM composites with)
                // for the run's entire duration — see PerformanceLeague.md's 2026-09-13 entry.
                var velGpu = _dit.ForwardGpu(ws.Latent, txtEmbeds, pooledEmbed, t, guidance, ws, weights, visionOps);
                imageOps.BeginBatch();
                visionOps.FluxEulerStep(ws.Latent, velGpu, sign * dt, nImg * _params.InChannels);
                imageOps.EndBatch();

                if (profEnabled)
                {
                    Console.Error.WriteLine($"[FluxProfile] Step {i + 1}/{nSteps} (t={t:F3} -> {tNext:F3}): {stepStopwatch.Elapsed.TotalMilliseconds:F1}ms");
                }

                progress?.Invoke(i + 1, nSteps);
            }

            denoised = new float[nImg * _params.InChannels];
            _backend.Download(ws.Latent, denoised);
        }
        else
        {
            int stepCounter = 0;
            var stepStopwatch = Stopwatch.StartNew();
            denoised = scheduler.Denoise(
                noisePacked,
                (x, t) =>
                {
                    stepStopwatch.Restart();
                    var vel = _dit.Forward(x, imgIds, txtEmbeds, txtIds, pooledEmbed, t, guidance);
                    stepCounter++;
                    if (profEnabled)
                    {
                        Console.Error.WriteLine($"[FluxProfile] Step {stepCounter}/{steps} (t={t:F3}): {stepStopwatch.Elapsed.TotalMilliseconds:F1}ms");
                    }
                    return vel;
                },
                progress);
        }
        double msDiT = sw?.Elapsed.TotalMilliseconds ?? 0; sw?.Restart();

        // ── 5. Unpack and decode ──────────────────────────────────────────
        var latent  = EulerFlowScheduler.UnpackLatent(denoised, latC, latH, latW, _params.PatchSize);
        var pixels  = _vae.Decode(latent, latH, latW);    // [3, H, W] in [0,1]
        double msVae = sw?.Elapsed.TotalMilliseconds ?? 0;

        if (profEnabled)
        {
            double msTotal = swTotal!.Elapsed.TotalMilliseconds;
            Console.Error.WriteLine("[FluxProfile] Stage split summary (vs C++ sd-cli):");
            Console.Error.WriteLine($"  CLIP-L encode        {msClip,10:F2}ms  {100.0 * msClip / msTotal,6:F2}% (C++ clip+t5: ~11.29s)");
            Console.Error.WriteLine($"  T5-XXL encode        {msT5,10:F2}ms  {100.0 * msT5 / msTotal,6:F2}%");
            Console.Error.WriteLine($"  Noise/pos-id setup   {msSetup,10:F2}ms  {100.0 * msSetup / msTotal,6:F2}%");
            Console.Error.WriteLine($"  DiT denoise loop     {msDiT,10:F2}ms  {100.0 * msDiT / msTotal,6:F2}% (C++ sampling 4 steps: ~165.6s, ~41.4s/step)");
            Console.Error.WriteLine($"  VAE decode           {msVae,10:F2}ms  {100.0 * msVae / msTotal,6:F2}% (C++ VAE decode: ~14.79s)");
            Console.Error.WriteLine($"  Total                {msTotal,10:F2}ms (C++ wall: ~191.7s)");
        }

        // ── 6. Write PNG ──────────────────────────────────────────────────
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

    /// <summary>IDiffusionPipeline adapter — delegates to <see cref="Generate(string,int,int,int,float,int,string,Action{int,int}?,RRDBNet?,float)"/> unchanged.</summary>
    void IDiffusionPipeline.Generate(ImageGenerationRequest request) => Generate(
        request.Prompt, request.Width, request.Height,
        steps: request.Steps < 0 ? 4 : request.Steps,
        guidance: request.Guidance, seed: request.Seed, outputPath: request.OutputPath,
        progress: request.Progress, upscaler: request.Upscaler, upscaleBlend: request.UpscaleBlend);

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _dit.Dispose();
            _vae.Dispose();
            _clip.Dispose();
            _t5.Dispose();
            if (_ownsBackend)
                _backend.Dispose();
        }
    }
}
