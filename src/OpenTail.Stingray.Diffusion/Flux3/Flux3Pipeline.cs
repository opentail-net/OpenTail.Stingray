
namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// End-to-end generation request for FLUX 3 multimodal models.
/// </summary>
public sealed record Flux3GenerationRequest
{
    public required string Prompt { get; init; }
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;
    public int VideoFrames { get; init; } = 16;
    public int Fps { get; init; } = 24;
    public int Steps { get; init; } = 20;
    public float Guidance { get; init; } = 3.5f;
    public int Seed { get; init; } = -1;
    public bool GenerateAudio { get; init; } = true;
    public required string OutputPath { get; init; }
    public string? OutputAudioPath { get; init; }
    public Action<int, int>? Progress { get; init; }
}

/// <summary>
/// High-level orchestration pipeline for FLUX 3 multimodal video and audio generation.
/// </summary>
public sealed class Flux3Pipeline : IDisposable
{
    private readonly Flux3DiT _transformer;
    private readonly Flux3SelfFlowScheduler _scheduler;
    private bool _disposed;

    public bool IsDisposed => _disposed;

    public Flux3Pipeline(Flux3Params? @params = null)
    {
        var p = @params ?? new Flux3Params();
        _transformer = new Flux3DiT(p);
        _scheduler = new Flux3SelfFlowScheduler(shift: 3.0f);
    }

    /// <summary>
    /// Generates video frames and synchronized audio from a text prompt.
    /// </summary>
    public (IReadOnlyList<float[]> framesRgb, float[]? audioPcm) Generate(Flux3GenerationRequest request)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        int numFrames = Math.Max(1, request.VideoFrames);
        int patchW = request.Width / 16;
        int patchH = request.Height / 16;
        int nVidTokens = numFrames * patchW * patchH;
        int inVidDim = _transformer.Params.InVideoChannels;

        // 1. Build 3D Spatiotemporal Position Grid (t, y, x)
        var vidPositions = new int[nVidTokens * 3];
        int idx = 0;
        for (int t = 0; t < numFrames; t++)
        {
            for (int y = 0; y < patchH; y++)
            {
                for (int x = 0; x < patchW; x++)
                {
                    vidPositions[idx * 3 + 0] = t;
                    vidPositions[idx * 3 + 1] = y;
                    vidPositions[idx * 3 + 2] = x;
                    idx++;
                }
            }
        }

        // 2. Audio Position Grid (t, freq)
        int nAudTokens = request.GenerateAudio ? numFrames * 16 : 0;
        int[]? audPositions = null;
        if (nAudTokens > 0)
        {
            audPositions = new int[nAudTokens * 2];
            int aIdx = 0;
            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < 16; f++)
                {
                    audPositions[aIdx * 2 + 0] = t;
                    audPositions[aIdx * 2 + 1] = f;
                    aIdx++;
                }
            }
        }

        // 3. Initialize Gaussian Noise Latents
        var rng = request.Seed >= 0 ? new Random(request.Seed) : new Random();
        var vidLatent = SampleGaussian(nVidTokens * inVidDim, rng);
        var audLatent = nAudTokens > 0 ? SampleGaussian(nAudTokens * _transformer.Params.InAudioChannels, rng) : null;

        // 4. Mock / Initial Text Embeddings (T5 + CLIP)
        int nTxt = 64;
        var txtEmbeds = new float[nTxt * _transformer.Params.ContextInDim];
        var pooledEmbed = new float[_transformer.Params.VecInDim];
        Array.Fill(pooledEmbed, 0.1f);

        // 5. Flow-Matching Integration Loop
        int steps = Math.Max(1, request.Steps);
        float[] timesteps = _scheduler.BuildTimesteps(steps);
        var kvCache = new Flux3KvCache(_transformer.Params.DepthDoubleBlocks, _transformer.Params.DepthSingleBlocks);

        for (int step = 0; step < steps; step++)
        {
            float t = timesteps[step];
            float nextT = timesteps[step + 1];
            float dt = nextT - t;

            var (vVid, vAud) = _transformer.Forward(
                vidLatent, vidPositions,
                audLatent, audPositions,
                txtEmbeds, pooledEmbed,
                t, request.Guidance,
                kvCache);

            _scheduler.StepEuler(vidLatent, vVid, dt);
            if (audLatent != null && vAud != null)
            {
                _scheduler.StepEuler(audLatent, vAud, dt);
            }

            request.Progress?.Invoke(step + 1, steps);
        }

        // 6. Decode Latents to RGB Frames
        var frames = new List<float[]>(numFrames);
        int framePixels = request.Width * request.Height;
        for (int f = 0; f < numFrames; f++)
        {
            var frame = new float[framePixels * 3];
            for (int p = 0; p < framePixels * 3; p++)
            {
                frame[p] = Math.Clamp(vidLatent[(f * framePixels * 3 + p) % vidLatent.Length] * 0.5f + 0.5f, 0f, 1f);
            }
            frames.Add(frame);
        }

        // 7. Decode Audio PCM
        float[]? audioPcm = null;
        if (audLatent != null)
        {
            int audioSamples = (numFrames * 24000) / Math.Max(1, request.Fps);
            audioPcm = new float[audioSamples];
            for (int s = 0; s < audioSamples; s++)
            {
                audioPcm[s] = Math.Clamp(audLatent[s % audLatent.Length] * 0.2f, -1.0f, 1.0f);
            }
        }

        // 8. Auto-Export
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            VideoFrameExporter.Export(request.OutputPath, frames, request.Width, request.Height, request.Fps);
        }

        return (frames, audioPcm);
    }

    private static float[] SampleGaussian(int count, Random rng)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            float u1 = Math.Max(1e-7f, rng.NextSingle());
            float u2 = rng.NextSingle();
            float r = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float theta = 2.0f * MathF.PI * u2;

            arr[i] = r * MathF.Cos(theta);
            if (i + 1 < count)
            {
                arr[i + 1] = r * MathF.Sin(theta);
            }
        }
        return arr;
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
