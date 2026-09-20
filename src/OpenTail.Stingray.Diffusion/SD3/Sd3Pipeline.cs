using OpenTail.Stingray.Diffusion.StableDiffusion;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// Stable Diffusion 3 / 3.5 Pipeline.
/// Triple text conditioning (CLIP-L + OpenCLIP-bigG + T5), MMDiT DiT transformer,
/// Rectified Flow-Matching scheduler, and 16-channel VAE decoder.
///
/// <para><b>Real bug found and fixed 2026-09-19 (docs/094)</b>: the real reference
/// (<c>pipeline_stable_diffusion_3.py</c>'s <c>encode_prompt</c>) builds the joint-attention text
/// sequence as CLIP-L+G (channel-padded from 2048 to 4096, 77 tokens) concatenated ALONG THE TOKEN
/// AXIS with T5-XXL's own 4096-dim hidden states (<c>max_sequence_length=256</c> by default) --
/// total 333 text tokens, not 77. This pipeline previously built only the 77-token CLIP block and
/// never called T5 at all (zero T5 wiring existed despite this class's own doc comment claiming
/// "Triple text conditioning"), silently feeding the DiT a text sequence less than a quarter of its
/// trained length with the other three-quarters simply absent rather than even zero-padded --
/// producing garbled, incoherent output on both CPU and GPU (visually confirmed, not numeric-only).
/// T5-XXL is optional in the real pipeline (falls back to an all-zero block, same shape, when no T5
/// checkpoint is supplied) -- mirrored here via <paramref name="t5EncoderPath"/>/<paramref
/// name="t5TokenizerPath"/> being optional constructor inputs.</para>
/// </summary>
public sealed class Sd3Pipeline : IDisposable, IDiffusionPipeline
{
    private const int T5MaxTokens = 256;
    private const int ContextDim = 4096;

    private readonly IWeightLoader _weights;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly ClipLEncoder _clipL;
    private readonly OpenClipGEncoder _clipG;
    private readonly MMDiTModel _mmdit;
    private readonly VaeDecoder _vae;
    private readonly T5Encoder? _t5;
    private readonly T5Tokenizer? _t5Tokenizer;
    private bool _disposed;

    public Sd3Pipeline(
        IWeightLoader weights,
        ClipTokenizer clipTokenizer,
        ClipLEncoder clipL,
        OpenClipGEncoder clipG,
        MMDiTModel mmdit,
        VaeDecoder vae,
        T5Encoder? t5 = null,
        T5Tokenizer? t5Tokenizer = null)
    {
        _weights = weights;
        _clipTokenizer = clipTokenizer;
        _clipL = clipL;
        _clipG = clipG;
        _mmdit = mmdit;
        _vae = vae;
        _t5 = t5;
        _t5Tokenizer = t5Tokenizer;
    }

    public MMDiTModel MMDiT => _mmdit;

    /// <summary>Test-support method (docs/094 Phase 1, 2026-09-20 GPU trajectory investigation):
    /// exposes the exact same real text-conditioning encode <see cref="Generate(string,string?,int,int,int,float,int,string,Action{int,int}?,RRDBNet?,float)"/>
    /// uses internally, so a test can build the real (context, pooledY, numTextTokens) tuple for a
    /// prompt without duplicating the CLIP-L/G/T5 encode+BuildContext+BuildPooledY logic. Text
    /// encoders (ClipLEncoder/OpenClipGEncoder/T5Encoder) take no backend parameter and always run
    /// on CPU, so this is identical whether called on a CPU-backed or GPU-backed pipeline instance.</summary>
    public (float[] context, float[] pooledY, int numTextTokens) EncodePromptForTesting(string prompt)
    {
        var tokens = _clipTokenizer.Tokenize(prompt);
        var (hiddenL, pooledL) = _clipL.Encode(tokens);
        var (hiddenG, pooledG) = _clipG.Encode(tokens);
        var t5 = EncodeT5(prompt);
        return (BuildContext(hiddenL, hiddenG, t5), BuildPooledY(pooledL, pooledG), 77 + T5MaxTokens);
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

        // TODO(docs/094): the combined single-file checkpoint variant (StabilityAI's
        // "..._incl_clips_t5xxlfp8.safetensors") embeds T5-XXL too, presumably under a
        // "text_encoders.t5xxl.transformer." prefix mirroring the clip_l/clip_g siblings above --
        // NOT verified against a real checkpoint (this project only has the gated single-file
        // variant's separate-file diffusers-layout sibling, wired via LoadSeparate below, which
        // DOES have real T5 wiring). Wire this once a real combined checkpoint is available to
        // confirm the tensor prefix rather than guessing it here.
        return new Sd3Pipeline(weights, tokenizer, clipL, clipG, mmdit, vae);
    }

    /// <summary>
    /// Loads from the standard HuggingFace diffusers multi-file layout (separate
    /// text_encoder/text_encoder_2/transformer/vae safetensors) instead of a single combined
    /// checkpoint. The official StabilityAI single-file "..._incl_clips[_t5xxlfp8].safetensors"
    /// variant <see cref="Load"/> targets is gated on HuggingFace and has no ungated mirror; the
    /// separate-file diffusers layout (e.g. ckpt/stable-diffusion-3.5-medium) is freely available.
    /// Each of these files is already tensor-name-stripped to what <see cref="Load"/>'s
    /// <see cref="PrefixWeightLoader"/> calls would produce (confirmed by direct inspection: the
    /// standalone transformer file's tensors are named e.g. "context_embedder.bias", matching
    /// the combined file's "model.diffusion_model.context_embedder.bias" with that prefix
    /// stripped) — so no prefix wrapping is needed here, just direct per-file loaders.
    /// </summary>
    public static Sd3Pipeline LoadSeparate(
        string clipLPath, string clipGPath, string transformerPath, string vaePath,
        string? tokenizerPath = null, IComputeBackend? backend = null,
        string? t5EncoderPath = null, string? t5TokenizerPath = null)
    {
        tokenizerPath ??= Path.Combine(AppContext.BaseDirectory, "models", "clip_tokenizer.json");
        var tokenizer = ClipTokenizer.FromFile(tokenizerPath);

        // The standalone HF diffusers text_encoder/text_encoder_2 safetensors keep the real
        // "text_model." prefix on every tensor (confirmed by direct inspection); the encoder
        // classes' own tensor lookups (shared with the combined-checkpoint Load() path above,
        // where PrefixWeightLoader already strips down past this level) expect it stripped.
        var clipLWeights = new PrefixWeightLoader(SafetensorsLoader.Open(clipLPath), "text_model.");
        var clipL = new ClipLEncoder(clipLWeights);
        var clipGWeights = new PrefixWeightLoader(SafetensorsLoader.Open(clipGPath), "text_model.");
        var clipG = new OpenClipGEncoder(clipGWeights);

        // The transformer file may be either a standalone safetensors export using the REAL
        // StabilityAI/ComfyUI tensor names (joint_blocks/x_embedder/context_embedder -- what
        // MMDiTModel expects, matching the combined Load() path after prefix-stripping) or a
        // GGUF conversion (which also preserves those same real names at the root, no prefix at
        // all) -- NOT the HF `diffusers`-native re-export (transformer_blocks/pos_embed.proj/
        // separate add_q_proj|add_k_proj|add_v_proj), which is a structurally different fused-QKV
        // layout MMDiTModel does not implement and was not attempted here.
        IWeightLoader mmditWeights = transformerPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(transformerPath)
            : SafetensorsLoader.Open(transformerPath);
        var mmdit = new MMDiTModel(mmditWeights, prefix: "", backend: backend);

        var vaeWeights = SafetensorsLoader.Open(vaePath);
        var vae = new VaeDecoder(vaeWeights, backend: backend);

        T5Encoder? t5 = null;
        T5Tokenizer? t5Tok = null;
        if (t5EncoderPath is not null && File.Exists(t5EncoderPath) && t5TokenizerPath is not null && File.Exists(t5TokenizerPath))
        {
            t5 = new T5Encoder(t5EncoderPath);
            t5Tok = T5Tokenizer.FromFile(t5TokenizerPath, maxLen: T5MaxTokens);
        }

        return new Sd3Pipeline(mmditWeights, tokenizer, clipL, clipG, mmdit, vae, t5, t5Tok);
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
        var condT5 = EncodeT5(prompt);

        var condContext = BuildContext(condHiddenL, condHiddenG, condT5);
        var condPooledY = BuildPooledY(condPooledL, condPooledG);

        var uncondTokens = _clipTokenizer.Tokenize(negativePrompt ?? "");
        var (uncondHiddenL, uncondPooledL) = _clipL.Encode(uncondTokens);
        var (uncondHiddenG, uncondPooledG) = _clipG.Encode(uncondTokens);
        var uncondT5 = EncodeT5(negativePrompt ?? "");

        var uncondContext = BuildContext(uncondHiddenL, uncondHiddenG, uncondT5);
        var uncondPooledY = BuildPooledY(uncondPooledL, uncondPooledG);
        int numTextTokens = 77 + T5MaxTokens;

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

            float[] condPred;
            float[] uncondPred;
            // GPU: `MMDiTModel.Forward` locks `this` for the whole call (one Vulkan context can't
            // run two forward passes concurrently), so `Parallel.Invoke` buys zero real overlap
            // here -- it's already fully serialized by that lock -- while still paying real
            // ThreadPool scheduling and lock-contention overhead between the two calls, a plausible
            // contributor to the "far-apart GPU activity" gaps observed on a live utilization graph
            // (2026-09-20 perf pass, docs/094 Phase 1). CPU: the two forward passes are genuinely
            // independent CPU-bound work, so Parallel.Invoke is a real, kept win there.
            if (_mmdit.IsGpuBacked)
            {
                condPred = _mmdit.Forward(x, timestep, condContext, condPooledY, latH, latW, numTextTokens);
                uncondPred = _mmdit.Forward(x, timestep, uncondContext, uncondPooledY, latH, latW, numTextTokens);
            }
            else
            {
                float[] cp = null!, up = null!;
                Parallel.Invoke(
                    () => cp = _mmdit.Forward(x, timestep, condContext, condPooledY, latH, latW, numTextTokens),
                    () => up = _mmdit.Forward(x, timestep, uncondContext, uncondPooledY, latH, latW, numTextTokens)
                );
                condPred = cp;
                uncondPred = up;
            }

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
        // Real SD3/3.5 VAE scaling_factor=1.5305/shift_factor=0.0609 (confirmed from the real
        // stabilityai/stable-diffusion-3.5-medium vae/config.json) -- a DIFFERENT 16-channel VAE
        // checkpoint than FLUX/Z-Image-Turbo's, so VaeDecoder's channel-count-based default (tuned
        // for FLUX/Z-Image) is wrong here; pass the real values explicitly.
        var pixels = _vae.Decode(x, latH, latW, scaleOverride: 1f / 1.5305f, shiftOverride: 0.0609f);

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

    private float[] EncodeT5(string prompt)
    {
        if (_t5 is null || _t5Tokenizer is null) return new float[T5MaxTokens * ContextDim];
        // T5Tokenizer.Tokenize returns unpadded ids; the real reference always pads/truncates to a
        // FIXED length before encoding (same convention already established for FLUX.1 in
        // ImagePipeline.cs) -- remaining slots stay 0 (T5's <pad> token id).
        var raw = _t5Tokenizer.Tokenize(prompt);
        var tokens = new int[T5MaxTokens];
        Array.Copy(raw, tokens, Math.Min(raw.Length, T5MaxTokens));
        return _t5.Encode(tokens);
    }

    /// <summary>
    /// Real SD3 text sequence per <c>pipeline_stable_diffusion_3.py</c>'s <c>encode_prompt</c>:
    /// CLIP-L+G channel-concatenated (768+1280=2048) then zero-padded to the full 4096
    /// <c>joint_attention_dim</c>, occupying the FIRST 77 token rows; T5-XXL's own native
    /// 4096-dim hidden states (or an all-zero block of the same shape when T5 is unavailable)
    /// occupy the NEXT <see cref="T5MaxTokens"/> token rows. Concatenation is along the token
    /// axis (<c>dim=-2</c>), not the channel axis -- the two blocks are never mixed per-token.
    /// </summary>
    private static float[] BuildContext(float[] hiddenL, float[] hiddenG, float[] t5Context)
    {
        var context = new float[(77 + T5MaxTokens) * ContextDim];
        for (int t = 0; t < 77; t++)
        {
            Array.Copy(hiddenL, t * 768, context, t * ContextDim, 768);
            Array.Copy(hiddenG, t * 1280, context, t * ContextDim + 768, 1280);
        }
        Array.Copy(t5Context, 0, context, 77 * ContextDim, T5MaxTokens * ContextDim);
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
            _t5?.Dispose();
            _weights.Dispose();
        }
    }
}

