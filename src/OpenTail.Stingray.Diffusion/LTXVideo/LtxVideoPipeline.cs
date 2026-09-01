
namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// Native C# LTX-Video (Lightricks Video DiT) Text-to-Video and Image-to-Video Inference Pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::LTXV
/// </summary>
public sealed class LtxVideoPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader? _weights;
    private readonly LtxVideoModel _transformer;
    private readonly LtxVaeDecoder? _vae;
    private readonly IWeightLoader? _t5Weights;
    private readonly TextEncoders.T5Encoder? _t5;
    private readonly TextEncoders.T5Tokenizer? _t5Tokenizer;
    private readonly int _temporalScale;
    private readonly int _spatialScale;
    private readonly int _fps;
    private bool _disposed;

    public string Architecture => "LTX-Video";

    public LtxVideoPipeline(
        LtxVideoModel transformer,
        LtxVaeDecoder? vae = null,
        IWeightLoader? weights = null,
        TextEncoders.T5Encoder? t5 = null,
        TextEncoders.T5Tokenizer? t5Tokenizer = null,
        IWeightLoader? t5Weights = null,
        int temporalScale = 8,
        int spatialScale = 32,
        int fps = 24)
    {
        _transformer = transformer;
        _vae = vae;
        _weights = weights;
        _t5 = t5;
        _t5Tokenizer = t5Tokenizer;
        _t5Weights = t5Weights;
        _temporalScale = temporalScale;
        _spatialScale = spatialScale;
        _fps = fps;
    }

    /// <param name="textEncoderDir">Real T5-v1.1-XXL encoder directory (HF sharded-safetensors
    /// layout, e.g. `Lightricks/LTX-Video`'s own `text_encoder/` subfolder) -- optional; without it
    /// the pipeline falls back to placeholder random-noise text conditioning (structurally runnable,
    /// not semantically meaningful).</param>
    /// <param name="tokenizerJsonPath">Matching `tokenizer.json` (fast-tokenizer format) for
    /// <paramref name="textEncoderDir"/> -- required if that parameter is provided.</param>
    public static LtxVideoPipeline Load(
        string modelPath,
        string? vaePath = null,
        IComputeBackend? backend = null,
        string? textEncoderDir = null,
        string? tokenizerJsonPath = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new LtxVideoModel(weights);

        // Real LTX-Video single-file checkpoints (like `ltx-video-2b-v0.9.1.safetensors`) bundle
        // the VAE decoder's own `vae.decoder.*` tensors in the SAME file as the transformer -- no
        // separate VAE checkpoint needed for the common case. `vaePath` remains supported for a
        // split checkpoint layout (e.g. a GGUF conversion that separates them).
        LtxVaeDecoder? vae = null;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            IWeightLoader vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
            vae = new LtxVaeDecoder(vaeLoader);
        }
        else if (weights.Contains("vae.decoder.conv_in.conv.weight"))
        {
            vae = new LtxVaeDecoder(weights);
        }

        IWeightLoader? t5Weights = null;
        TextEncoders.T5Encoder? t5 = null;
        TextEncoders.T5Tokenizer? t5Tokenizer = null;
        if (!string.IsNullOrWhiteSpace(textEncoderDir) && Directory.Exists(textEncoderDir)
            && !string.IsNullOrWhiteSpace(tokenizerJsonPath) && File.Exists(tokenizerJsonPath))
        {
            t5Weights = SafetensorsLoader.OpenDirectory(textEncoderDir);
            t5 = TextEncoders.T5Encoder.FromLoader(t5Weights);
            t5Tokenizer = TextEncoders.T5Tokenizer.FromFile(tokenizerJsonPath, maxLen: 256);
        }

        return new LtxVideoPipeline(transformer, vae, weights, t5, t5Tokenizer, t5Weights);
    }

    public void Generate(ImageGenerationRequest request)
    {
        var frames = GenerateVideo(
            prompt: request.Prompt,
            width: request.Width,
            height: request.Height,
            numFrames: 1,
            steps: request.Steps > 0 ? request.Steps : 25,
            guidance: request.Guidance > 0 ? request.Guidance : 3.0f,
            seed: request.Seed,
            progress: request.Progress);

        if (frames.Count > 0 && !string.IsNullOrEmpty(request.OutputPath))
        {
            SaveRgbPlanarAsPng(frames[0], request.Width, request.Height, request.OutputPath);
        }
    }

    /// <param name="guidance">Real classifier-free-guidance scale; the real pipeline's own default
    /// is 4.5 (`pipeline_ltx_video.py`'s `guidance_scale` parameter), not the previous placeholder
    /// default of 3.0. CFG only actually applies (an extra unconditional forward pass per step,
    /// blended in) when a real T5 encoder/tokenizer was supplied to <see cref="Load"/> -- without
    /// one there is no real negative-prompt embedding to guide away from.</param>
    public List<float[]> GenerateVideo(
        string prompt,
        int width = 768,
        int height = 512,
        int numFrames = 25,
        int steps = 25,
        float guidance = 4.5f,
        int seed = -1,
        Action<int, int>? progress = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int patchH = Math.Max(1, height / _spatialScale);
        int patchW = Math.Max(1, width / _spatialScale);
        int numLatentFrames = Math.Max(1, (numFrames - 1) / _temporalScale + 1);

        int inChannels = _transformer.InChannels; // 128
        int totalLatentElements = numLatentFrames * patchH * patchW * inChannels;
        var latents = new float[totalLatentElements];

        var rng = seed >= 0 ? new Random(seed) : new Random();
        for (int i = 0; i < totalLatentElements; i++)
        {
            float u1 = Math.Max(1e-7f, rng.NextSingle());
            float u2 = rng.NextSingle();
            latents[i] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }

        int textDim = _transformer.CaptionChannels;
        float[] textContext;
        float[]? negativeTextContext = null; // real CFG's unconditional (empty-prompt) branch
        if (_t5 is not null && _t5Tokenizer is not null)
        {
            // Real T5-v1.1-XXL text conditioning (numerically verified against HuggingFace
            // `transformers`' real T5EncoderModel, >0.999 cosine similarity -- see
            // LtxT5EncoderGoldenParityTests). Feeds caption_projection's 4096-dim input directly.
            var tokenIds = _t5Tokenizer.Tokenize(prompt);
            textContext = _t5.Encode(tokenIds);

            if (guidance > 1.0f)
            {
                var negativeIds = _t5Tokenizer.Tokenize(string.Empty);
                negativeTextContext = _t5.Encode(negativeIds);
            }
        }
        else
        {
            // Placeholder text conditioning: no real T5-v1.1-XXL encoder wired for this Load() call
            // (pass textEncoderDir/tokenizerJsonPath to enable it). Structurally runnable, not
            // semantically meaningful. No CFG without a real negative-prompt embedding to guide
            // away from.
            int textTokens = 128;
            textContext = new float[textTokens * textDim];
            for (int i = 0; i < textContext.Length; i++)
            {
                textContext[i] = 0.01f * (rng.NextSingle() - 0.5f);
            }
        }

        // Real `RectifiedFlowScheduler` (config: `sampler="Uniform"` (default), `shifting="SD3"`,
        // `target_shift_terminal=None` -- confirmed against this checkpoint's own embedded
        // `__metadata__["config"]["scheduler"]` JSON): base timesteps are a plain
        // `linspace(1, 1/steps, steps)`, then resolution-DEPENDENT-shifted via
        // `sd3_resolution_dependent_timestep_shift` -- `shift = get_normal_shift(numLatentTokens)`
        // (linear interpolation between (1024 tokens -> 0.95) and (4096 tokens -> 2.05)), then
        // `time_shift(shift, sigma=1, t) = exp(shift) / (exp(shift) + (1/t - 1))` per real
        // `ltx_video/schedulers/rf.py`. This is NOT the ad-hoc `shift/(1+(shift-1)*t)` formula (a
        // fixed shift=3.0) this pipeline used previously -- that was a plausible-looking
        // approximation, not the real, resolution-dependent formula.
        int numLatentTokens = numLatentFrames * patchH * patchW;
        float sd3Shift = GetNormalShift(numLatentTokens);

        var shiftedTimesteps = new float[steps + 1]; // padded with a trailing 0, per real `step()`
        for (int i = 0; i < steps; i++)
        {
            float tRaw = 1.0f - (float)i / steps; // linspace(1, 1/steps, steps)
            shiftedTimesteps[i] = TimeShift(sd3Shift, 1.0f, tRaw);
        }
        shiftedTimesteps[steps] = 0f;

        for (int step = 0; step < steps; step++)
        {
            float tShifted = shiftedTimesteps[step];
            float dt = tShifted - shiftedTimesteps[step + 1]; // real: current - next lower timestep
            float timestep = tShifted * _transformer.TimestepScale;

            var vPred = _transformer.Forward(latents, timestep, textContext, numLatentFrames, patchH, patchW);

            if (negativeTextContext is not null)
            {
                // Real classifier-free guidance: `noise_pred = uncond + guidance_scale * (cond -
                // uncond)` (`pipeline_ltx_video.py`'s real combine step).
                var vPredUncond = _transformer.Forward(latents, timestep, negativeTextContext, numLatentFrames, patchH, patchW);
                for (int i = 0; i < vPred.Length; i++)
                    vPred[i] = vPredUncond[i] + guidance * (vPred[i] - vPredUncond[i]);
            }

            for (int i = 0; i < totalLatentElements; i++)
            {
                latents[i] -= dt * vPred[i];
            }

            progress?.Invoke(step + 1, steps);
        }

        var results = new List<float[]>();
        int spatialSize = patchH * patchW;

        if (_vae != null)
        {
            // Real decoder operates on the WHOLE latent volume at once (its 3 compress_all stages
            // do real cross-frame temporal upsampling, not a per-frame-independent operation) --
            // channel-first [C,F,H,W] layout, converted from this pipeline's token-major
            // [numTokens, C] latent buffer.
            var chFirst = new float[inChannels * numLatentFrames * spatialSize];
            for (int f = 0; f < numLatentFrames; f++)
            {
                int tokenBase = f * spatialSize;
                for (int p = 0; p < spatialSize; p++)
                {
                    int srcOff = (tokenBase + p) * inChannels;
                    for (int c = 0; c < inChannels; c++)
                        chFirst[(c * numLatentFrames + f) * spatialSize + p] = latents[srcOff + c];
                }
            }

            // Real pipeline default: `decode_timestep=0.0` (not the "nominal 0.05" this project's
            // earlier planning-pass research guessed -- see LtxVaeDecoder's own doc comment).
            var video = _vae.Decode(chFirst, decodeTimestep: 0f, numLatentFrames, patchH, patchW);

            int outF = 8 * (numLatentFrames - 1) + 1; // real `F_out = 8*(F_latent-1)+1` (LtxVaeDecoder.TemporalScale)
            int outH = patchH * LtxVaeDecoder.SpatialScale;
            int outW = patchW * LtxVaeDecoder.SpatialScale;
            int outSpatial = outH * outW;

            for (int f = 0; f < outF; f++)
            {
                var frameRgb = new float[3 * outSpatial];
                for (int c = 0; c < 3; c++)
                {
                    int srcBase = (c * outF + f) * outSpatial;
                    int dstBase = c * outSpatial;
                    for (int p = 0; p < outSpatial; p++)
                    {
                        frameRgb[dstBase + p] = Math.Clamp((video[srcBase + p] + 1.0f) * 0.5f, 0.0f, 1.0f);
                    }
                }
                results.Add(frameRgb);
            }

            return results;
        }

        for (int f = 0; f < numLatentFrames; f++)
        {
            var frameRgb = new float[3 * height * width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int py = Math.Clamp(y / _spatialScale, 0, patchH - 1);
                    int px = Math.Clamp(x / _spatialScale, 0, patchW - 1);
                    int p = py * patchW + px;
                    int frameOffset = f * spatialSize * inChannels;

                    float r = Math.Clamp(0.5f + latents[frameOffset + p * inChannels + 0] * 0.2f, 0f, 1f);
                    float g = Math.Clamp(0.5f + latents[frameOffset + p * inChannels + 1] * 0.2f, 0f, 1f);
                    float b = Math.Clamp(0.5f + latents[frameOffset + p * inChannels + 2] * 0.2f, 0f, 1f);

                    frameRgb[0 * (height * width) + y * width + x] = r;
                    frameRgb[1 * (height * width) + y * width + x] = g;
                    frameRgb[2 * (height * width) + y * width + x] = b;
                }
            }

            results.Add(frameRgb);
        }

        return results;
    }

    /// <summary>Real `get_normal_shift`: linear interpolation of the SD3 resolution-dependent shift
    /// parameter between (1024 tokens -&gt; 0.95) and (4096 tokens -&gt; 2.05).</summary>
    private static float GetNormalShift(int nTokens, int minTokens = 1024, int maxTokens = 4096,
        float minShift = 0.95f, float maxShift = 2.05f)
    {
        float m = (maxShift - minShift) / (maxTokens - minTokens);
        float b = minShift - m * minTokens;
        return m * nTokens + b;
    }

    /// <summary>Real `time_shift(mu, sigma, t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.</summary>
    private static float TimeShift(float mu, float sigma, float t)
        => MathF.Exp(mu) / (MathF.Exp(mu) + MathF.Pow(1.0f / t - 1.0f, sigma));

    private static void SaveRgbPlanarAsPng(float[] rgbPlanar, int width, int height, string outputPath)
    {
        PngWriter.Write(outputPath, rgbPlanar, width, height);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _vae?.Dispose();
            _t5?.Dispose();
            _t5Weights?.Dispose();
            _weights?.Dispose();
        }
    }
}

