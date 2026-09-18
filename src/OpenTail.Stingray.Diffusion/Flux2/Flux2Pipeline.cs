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
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public Flux2Params Params => _transformer.Params;

    public Flux2Pipeline(Flux2Params? @params = null)
    {
        var p = @params ?? new Flux2Params();
        _transformer = new Flux2DiT(p);
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

        // 4. Mock / Initial Text Embeddings (Mistral-Small-24B conditioning -- real wiring not
        //    yet landed, see docs/087). Text token positions per the real scheme (prc_txt):
        //    t=0/h=0/w=0 (dummy), l=sequential index -- the only axis that varies for text.
        int nTxt = 64;
        var txtEmbeds = new float[nTxt * _transformer.Params.ContextInDim];
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

        // 6. Decode Latent to RGB
        int pixelCount = request.Width * request.Height;
        var rgb = new float[pixelCount * 3];
        for (int p = 0; p < pixelCount * 3; p++)
        {
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
        _disposed = true;
    }
}
