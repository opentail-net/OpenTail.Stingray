namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Generation request for FLUX.2 (Klein &amp; Kontext).
/// </summary>
public sealed record Flux2GenerationRequest
{
    public required string Prompt { get; init; }
    public IReadOnlyList<float[]>? ReferenceImagesRgb { get; init; }
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;
    public int Steps { get; init; } = 20;
    public float Guidance { get; init; } = 3.5f;
    public int Seed { get; init; } = -1;
    public required string OutputPath { get; init; }
    public Action<int, int>? Progress { get; init; }
}

/// <summary>
/// High-level orchestration pipeline for FLUX.2 multi-reference and contextual image generation.
/// </summary>
public sealed class Flux2Pipeline : IDisposable
{
    private readonly Flux2DiT _transformer;
    private readonly IWeightLoader? _ditWeights;
    private readonly GgufModel? _mistralModel;
    private readonly Engine.ForwardPass? _mistralForward;
    private readonly GgufTokenizer? _mistralTokenizer;
    private readonly IWeightLoader? _vaeWeights;
    private readonly Cpu.CpuBackend? _mistralBackend;
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public Flux2Params Params => _transformer.Params;

    /// <summary>Structural-only constructor (no real weights) -- unchanged, backs the existing
    /// conformance tests. <see cref="Generate"/> in this mode uses synthetic conditioning/decode.</summary>
    public Flux2Pipeline(Flux2Params? @params = null)
    {
        var p = @params ?? new Flux2Params();
        _transformer = new Flux2DiT(p);
    }

    private Flux2Pipeline(
        Flux2DiT transformer, IWeightLoader ditWeights,
        GgufModel mistralModel, Engine.ForwardPass mistralForward, GgufTokenizer mistralTokenizer, Cpu.CpuBackend mistralBackend,
        IWeightLoader vaeWeights)
    {
        _transformer = transformer;
        _ditWeights = ditWeights;
        _mistralModel = mistralModel;
        _mistralForward = mistralForward;
        _mistralTokenizer = mistralTokenizer;
        _mistralBackend = mistralBackend;
        _vaeWeights = vaeWeights;
    }

    /// <summary>
    /// Loads a real FLUX.2 pipeline: real DiT weights (GGUF), real Mistral-Small-24B text encoder
    /// (GGUF), and real VAE (safetensors). See docs/087 for the full architecture derivation --
    /// all three components independently verified against real weights before this wiring.
    /// </summary>
    public static Flux2Pipeline Load(string ditPath, string mistralPath, string vaePath, Flux2Params? @params = null)
    {
        var ditWeights = GgufWeightLoader.Open(ditPath);
        var p = @params ?? new Flux2Params();
        var transformer = new Flux2DiT(ditWeights, p);

        var mistralModel = GgufModel.Open(mistralPath);
        var hp = ModelHyperparams.FromGgufMetadata(mistralModel.Metadata, mistralModel);
        var tokenizer = GgufTokenizer.FromGgufModel(mistralModel);
        var mistralBackend = new Cpu.CpuBackend();
        var mistralForward = new Engine.ForwardPass(mistralModel, mistralBackend, hp);

        var vaeWeights = SafetensorsLoader.Open(vaePath);

        return new Flux2Pipeline(transformer, ditWeights, mistralModel, mistralForward, tokenizer, mistralBackend, vaeWeights);
    }

    /// <summary>
    /// Generates an image conditioned on prompt and optional reference images.
    /// </summary>
    public float[] Generate(Flux2GenerationRequest request)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        int patchW = request.Width / 16;
        int patchH = request.Height / 16;
        int nTargetTokens = patchW * patchH;
        int inChannels = _transformer.Params.InChannels;

        // 1. Build Target 4D Position Grid (t, h, w, l), per the real BFL scheme
        //    (examples/flux2/src/flux2/sampling.py's prc_img): t=image index (0 for the
        //    target/generated image), h/w=patch row/col, l=0 (dummy -- l is only meaningful for
        //    text tokens, see nTxt positions below).
        var targetPositions = new int[nTargetTokens * 4];
        int idx = 0;
        for (int y = 0; y < patchH; y++)
        {
            for (int x = 0; x < patchW; x++)
            {
                targetPositions[idx * 4 + 0] = 0; // t: target image index = 0
                targetPositions[idx * 4 + 1] = y; // h
                targetPositions[idx * 4 + 2] = x; // w
                targetPositions[idx * 4 + 3] = 0; // l: dummy for image tokens
                idx++;
            }
        }

        // 2. Build Reference Images Grids & Mock Latents (t=r+1, h, w, l=0)
        List<float[]>? refLatents = null;
        List<int[]>? refPositions = null;

        if (request.ReferenceImagesRgb != null && request.ReferenceImagesRgb.Count > 0)
        {
            refLatents = new List<float[]>();
            refPositions = new List<int[]>();

            for (int r = 0; r < request.ReferenceImagesRgb.Count; r++)
            {
                var rLatent = new float[nTargetTokens * inChannels];
                Array.Fill(rLatent, 0.2f * (r + 1));
                refLatents.Add(rLatent);

                var rPos = new int[nTargetTokens * 4];
                int rIdx = 0;
                for (int y = 0; y < patchH; y++)
                {
                    for (int x = 0; x < patchW; x++)
                    {
                        rPos[rIdx * 4 + 0] = r + 1; // t: reference image index (temporal offset)
                        rPos[rIdx * 4 + 1] = y;     // h
                        rPos[rIdx * 4 + 2] = x;     // w
                        rPos[rIdx * 4 + 3] = 0;     // l: dummy for image tokens
                        rIdx++;
                    }
                }
                refPositions.Add(rPos);
            }
        }

        // 3. Initialize Target Gaussian Latents
        var rng = request.Seed >= 0 ? new Random(request.Seed) : new Random();
        var targetLatent = SampleGaussian(nTargetTokens * inChannels, rng);

        // 4. Text Embeddings -- real Mistral-Small-24B conditioning when real weights are loaded
        //    (Flux2TextConditioning.Encode, docs/087); synthetic fallback for the structural-only
        //    constructor. Text token positions per the real scheme (prc_txt): t=0/h=0/w=0 (dummy),
        //    l=sequential index -- the only axis that varies for text.
        int nTxt;
        float[] txtEmbeds;
        if (_mistralForward != null && _mistralTokenizer != null)
        {
            (txtEmbeds, nTxt) = Flux2TextConditioning.Encode(_mistralForward, _mistralTokenizer, request.Prompt);
        }
        else
        {
            nTxt = 64;
            txtEmbeds = new float[nTxt * _transformer.Params.ContextInDim];
        }
        var txtPositions = new int[nTxt * 4];
        for (int i = 0; i < nTxt; i++)
        {
            txtPositions[i * 4 + 0] = 0;
            txtPositions[i * 4 + 1] = 0;
            txtPositions[i * 4 + 2] = 0;
            txtPositions[i * 4 + 3] = i; // l: sequential text position
        }
        var pooledEmbed = Array.Empty<float>(); // FLUX.2 has no CLIP pooled conditioning at all (VecInDim=0, see Flux2Params.cs)

        // 5. Flow-Matching Integration Loop
        int steps = Math.Max(1, request.Steps);
        var scheduler = EulerFlowScheduler.Linear(steps, shift: 3.0f);

        for (int step = 0; step < steps; step++)
        {
            float t = 1.0f - (float)step / steps;
            float nextT = 1.0f - (float)(step + 1) / steps;
            float dt = nextT - t;

            var v = _transformer.Forward(
                targetLatent, targetPositions,
                refLatents, refPositions,
                txtEmbeds, txtPositions,
                pooledEmbed,
                t, request.Guidance);

            for (int i = 0; i < targetLatent.Length; i++)
            {
                targetLatent[i] += dt * v[i];
            }

            request.Progress?.Invoke(step + 1, steps);
        }

        // 6. Decode Latent to RGB -- real VAE (Flux2Vae + VaeDecoder, docs/087) when real weights
        //    are loaded; synthetic channel-repeat fallback for the structural-only constructor.
        int pixelCount = request.Width * request.Height;
        float[] rgb;
        if (_vaeWeights != null)
        {
            // targetLatent is token-major [nTarget, 128]; Flux2Vae needs channel-major [128, h, w].
            var chw = new float[nTargetTokens * inChannels];
            for (int i = 0; i < nTargetTokens; i++)
                for (int c = 0; c < inChannels; c++)
                    chw[c * nTargetTokens + i] = targetLatent[i * inChannels + c];

            var runningMean = _vaeWeights.ReadF32("bn.running_mean");
            var runningVar = _vaeWeights.ReadF32("bn.running_var");
            var (rawLatent, latH, latW) = Flux2Vae.UnnormalizeAndUnshuffle(chw, patchH, patchW, runningMean, runningVar);

            using var vae = new VaeDecoder(_vaeWeights);
            rgb = vae.Decode(rawLatent, latH, latW, scaleOverride: 1f, shiftOverride: 0f);
        }
        else
        {
            rgb = new float[pixelCount * 3];
            for (int p = 0; p < pixelCount * 3; p++)
                rgb[p] = Math.Clamp(targetLatent[p % targetLatent.Length] * 0.5f + 0.5f, 0f, 1f);
        }

        // 7. Save Image
        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            var dir = Path.GetDirectoryName(request.OutputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            PngWriter.Write(request.OutputPath, rgb, request.Width, request.Height);
        }

        return rgb;
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
        if (_disposed) return;
        _disposed = true;
        _mistralForward?.Dispose();
        _mistralBackend?.Dispose();
        _mistralModel?.Dispose();
        (_ditWeights as IDisposable)?.Dispose();
        (_vaeWeights as IDisposable)?.Dispose();
    }
}
