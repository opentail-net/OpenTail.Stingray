
namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// Native C# LTX-Video (Lightricks Video DiT) Text-to-Video and Image-to-Video Inference Pipeline.
/// Reference: stable-diffusion.cpp:src/stable-diffusion.cpp:sd_type_t::LTXV
/// </summary>
public sealed class LtxVideoPipeline : IDiffusionPipeline
{
    private readonly IWeightLoader? _weights;
    private readonly LtxVideoModel _transformer;
    private readonly VaeDecoder? _vae;
    private readonly int _temporalScale;
    private readonly int _spatialScale;
    private readonly int _fps;
    private bool _disposed;

    public string Architecture => "LTX-Video";

    public LtxVideoPipeline(
        LtxVideoModel transformer,
        VaeDecoder? vae = null,
        IWeightLoader? weights = null,
        int temporalScale = 8,
        int spatialScale = 32,
        int fps = 24)
    {
        _transformer = transformer;
        _vae = vae;
        _weights = weights;
        _temporalScale = temporalScale;
        _spatialScale = spatialScale;
        _fps = fps;
    }

    public static LtxVideoPipeline Load(string modelPath, string? vaePath = null, IComputeBackend? backend = null)
    {
        IWeightLoader weights = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(modelPath)
            : SafetensorsLoader.Open(modelPath);

        var transformer = new LtxVideoModel(weights);

        // NOTE: LTX-Video's real VAE decoder is timestep-conditioned (decode_timestep/
        // decode_noise_scale threaded through the decoder's own residual blocks), NOT a plain
        // causal-VAE the way Wan's `VaeDecoder`/`WanVaeDecoder3D` is -- reusing `VaeDecoder` here
        // is a known, deliberate placeholder (see docs/055-ltx-video-implementation-plan.md's "The
        // single biggest gotcha" section) until the real LTX VAE gets its own port.
        VaeDecoder? vae = null;
        if (!string.IsNullOrWhiteSpace(vaePath) && File.Exists(vaePath))
        {
            IWeightLoader vaeLoader = vaePath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                ? GgufWeightLoader.Open(vaePath)
                : SafetensorsLoader.Open(vaePath);
            vae = new VaeDecoder(vaeLoader, backend: backend);
        }

        return new LtxVideoPipeline(transformer, vae, weights);
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

    public List<float[]> GenerateVideo(
        string prompt,
        int width = 768,
        int height = 512,
        int numFrames = 25,
        int steps = 25,
        float guidance = 3.0f,
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

        // Placeholder text conditioning: real T5-v1.1-XXL encoder is not yet ported/wired (not
        // downloaded locally -- see docs/055-ltx-video-implementation-plan.md step 6). This feeds
        // caption_projection's 4096-dim input, not `CrossAttentionDim` (2048, the projected size).
        int textTokens = 128;
        int textDim = _transformer.CaptionChannels;
        var textContext = new float[textTokens * textDim];
        for (int i = 0; i < textContext.Length; i++)
        {
            textContext[i] = 0.01f * (rng.NextSingle() - 0.5f);
        }

        float shift = 3.0f;
        float dt = 1.0f / steps;

        for (int step = 0; step < steps; step++)
        {
            float tRaw = 1.0f - (float)step / steps;
            float tShifted = (shift * tRaw) / (1.0f + (shift - 1.0f) * tRaw);
            float timestep = tShifted * 1000.0f;

            var vPred = _transformer.Forward(latents, timestep, textContext, numLatentFrames, patchH, patchW);

            for (int i = 0; i < totalLatentElements; i++)
            {
                latents[i] -= dt * vPred[i];
            }

            progress?.Invoke(step + 1, steps);
        }

        var results = new List<float[]>();
        int spatialSize = patchH * patchW;

        for (int f = 0; f < numLatentFrames; f++)
        {
            float[] frameRgb;
            if (_vae != null)
            {
                var frameLatents16 = new float[16 * spatialSize];
                int frameOffset = f * spatialSize * inChannels;
                for (int c = 0; c < 16; c++)
                {
                    for (int p = 0; p < spatialSize; p++)
                    {
                        frameLatents16[c * spatialSize + p] = latents[frameOffset + p * inChannels + c];
                    }
                }
                frameRgb = _vae.Decode(frameLatents16, patchH, patchW);
            }
            else
            {
                frameRgb = new float[3 * height * width];
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
            }

            results.Add(frameRgb);
        }

        return results;
    }

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
            _weights?.Dispose();
        }
    }
}

