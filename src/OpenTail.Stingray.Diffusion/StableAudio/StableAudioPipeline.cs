using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Generation request for Stable Audio 3. Supply either <see cref="Prompt"/> (a raw string,
/// tokenized internally by <see cref="StableAudioPipeline"/>'s real T5Gemma tokenizer) or
/// <see cref="PromptTokenIds"/> directly (pre-tokenized, for callers with their own tokenization or
/// tests) -- exactly one must be set. Either way, ids are padded/truncated to 256 tokens internally,
/// with padding rows replaced by the real T5GemmaConditioner's learned padding embedding, matching
/// `padding_mode="learned"`.
/// </summary>
public sealed record StableAudioRequest
{
    public string? Prompt { get; init; }
    public int[]? PromptTokenIds { get; init; }
    public float DurationSeconds { get; init; } = 10.0f;
    public int Steps { get; init; } = 25;
    public int Seed { get; init; } = -1;

    /// <summary>Real classifier-free guidance scale (`cfg_scale` in the reference). Default 6.0
    /// matches the real Gradio demo interface's own default (the `DiffusionTransformer` class-level
    /// API default is 1.0/no-CFG; 6.0 is what the shipped generation experience actually uses).
    /// Set to 1.0 to disable CFG entirely (skips the extra unconditioned forward pass).</summary>
    public float CfgScale { get; init; } = 6.0f;

    public required string OutputPath { get; init; }
    public Action<int, int>? Progress { get; init; }
}

/// <summary>
/// End-to-end text-to-audio and music synthesis pipeline for Stable Audio 3.
///
/// <para><b>Status, 2026-09-02</b>: the text encoder (<see cref="T5GemmaEncoder"/>), DiT
/// (<see cref="StableAudioDiT"/>), and VAE (<see cref="AcousticVae"/>) are all real, weight-driven,
/// golden-verified-or-verified-in-progress ports -- see
/// docs/057-stable-audio-3-implementation-plan.md for the full status of each. All three share the
/// same <see cref="IWeightLoader"/> (the real checkpoint bundles DiT + VAE weights together).</para>
/// </summary>
public sealed class StableAudioPipeline : IDisposable
{
    private const int PromptMaxLength = 256;
    private const int CondTokenDim = 768;

    private readonly T5GemmaEncoder _textEncoder;
    private readonly StableAudioDiT _transformer;
    private readonly AcousticVae _vae;
    private readonly StableAudioParams _params;
    private readonly IWeightLoader _weights;
    private readonly GgufTokenizer? _tokenizer;
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public StableAudioParams Params => _params;

    /// <param name="dit">Loader open on the real checkpoint (e.g. `small-music-base/model.safetensors`) -- holds both the DiT and VAE weights.</param>
    /// <param name="textEncoderWeights">Loader open on the real T5Gemma encoder checkpoint (e.g. the bundled `t5gemma-b-b-ul2/` subfolder).</param>
    /// <param name="textEncoderDir">Directory holding the real T5Gemma `tokenizer.json` (typically
    /// the same directory <paramref name="textEncoderWeights"/> was opened on). Required for
    /// <see cref="StableAudioRequest.Prompt"/> (raw-string) requests; omit if every caller supplies
    /// <see cref="StableAudioRequest.PromptTokenIds"/> directly instead.</param>
    public StableAudioPipeline(IWeightLoader dit, IWeightLoader textEncoderWeights, string? textEncoderDir = null, StableAudioParams? @params = null)
    {
        _params = @params ?? new StableAudioParams();
        _weights = dit;
        _transformer = StableAudioDiT.FromLoader(dit);
        _textEncoder = T5GemmaEncoder.FromLoader(textEncoderWeights);
        _vae = AcousticVae.FromLoader(dit);

        if (textEncoderDir is not null)
        {
            var tokSource = HuggingFaceTokenizerSource.Load(textEncoderDir);
            if (!tokSource.IsUsable || tokSource.Source is null)
                throw new InvalidOperationException(
                    $"Failed to load T5Gemma tokenizer from '{textEncoderDir}': {string.Join("; ", tokSource.Rejections)}");
            _tokenizer = GgufTokenizer.FromSource(tokSource.Source);
        }
    }

    /// <summary>
    /// Generates high-fidelity stereo audio from a text prompt (or pre-tokenized ids) and a
    /// duration specification.
    /// </summary>
    public float[] Generate(StableAudioRequest request)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        int[] promptTokenIds = ResolvePromptTokenIds(request);

        float duration = Math.Max(0.5f, request.DurationSeconds);
        int seqLen = (int)Math.Ceiling(duration * _params.LatentFrameRate);
        int totalLatentElements = seqLen * _params.LatentChannels;

        var rng = request.Seed >= 0 ? new Random(request.Seed) : new Random();
        var latent = SampleGaussian(totalLatentElements, rng);

        float[] pcm = GenerateFromLatent(
            latent, seqLen, promptTokenIds, duration,
            Math.Max(1, request.Steps), request.CfgScale, request.Progress);

        if (!string.IsNullOrEmpty(request.OutputPath))
        {
            var dir = Path.GetDirectoryName(request.OutputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            WavWriter.WriteWav(request.OutputPath, pcm, _params.SampleRate, _params.AudioChannels, DitherMode.Tpdf);
        }

        return pcm;
    }

    /// <summary>
    /// Real generation core (conditioning assembly + Euler/CFG loop + VAE decode) from an explicit
    /// starting latent, factored out of <see cref="Generate"/> so a golden test can drive it with a
    /// fixed (not randomly-sampled) latent and compare the full real pipeline's output directly
    /// against a real end-to-end Python reference run -- <see cref="Generate"/> is just this plus
    /// real Gaussian noise sampling and the WAV-file side effect. <c>internal</c>, not private, for
    /// exactly that test (see <c>StableAudioPipelineGoldenParityTests</c>).
    /// </summary>
    internal float[] GenerateFromLatent(
        float[] initialLatent, int seqLen, int[] promptTokenIds, float durationSeconds,
        int steps, float cfgScale, Action<int, int>? progress = null)
    {
        var latent = initialLatent;
        var (condTokens, secondsTotalRaw) = BuildConditioning(promptTokenIds, durationSeconds);
        int nCond = condTokens.Length / CondTokenDim;
        var nullCondTokens = new float[condTokens.Length]; // real `null_embed = torch.zeros_like(...)`

        for (int step = 0; step < steps; step++)
        {
            float t = 1.0f - (float)step / steps;
            float nextT = 1.0f - (float)(step + 1) / steps;
            float dt = nextT - t;

            var v = PredictVelocity(latent, seqLen, condTokens, nullCondTokens, nCond, secondsTotalRaw, t, cfgScale);

            for (int i = 0; i < latent.Length; i++)
            {
                latent[i] += dt * v[i];
            }

            progress?.Invoke(step + 1, steps);
        }

        return _vae.Decode(latent, seqLen);
    }

    private int[] ResolvePromptTokenIds(StableAudioRequest request)
    {
        bool hasPrompt = !string.IsNullOrEmpty(request.Prompt);
        bool hasIds = request.PromptTokenIds is not null;
        if (hasPrompt == hasIds)
            throw new ArgumentException("Exactly one of StableAudioRequest.Prompt or PromptTokenIds must be set.");

        if (hasIds) return request.PromptTokenIds!;

        if (_tokenizer is null)
            throw new InvalidOperationException(
                "StableAudioRequest.Prompt requires the pipeline to be constructed with a textEncoderDir (real T5Gemma tokenizer.json).");

        return [.. _tokenizer.Encode(request.Prompt!)];
    }

    /// <summary>
    /// Real classifier-free guidance, matching `DiffusionTransformer.forward`'s CFG branch exactly
    /// (rectified-flow objective, real default `apg_scale=1.0` -- full Adaptive Projected Guidance,
    /// not vanilla CFG; the real Gradio demo interface's own default, not just the class's raw API
    /// default of no-CFG): runs the DiT twice (conditioned + unconditioned, the latter with an
    /// all-zero cross-attn context, `global_embed`/`seconds_total` conditioning shared unchanged
    /// between both passes), converts both outputs to denoised-`x` estimates, projects their
    /// difference to keep only the component ORTHOGONAL to the conditioned estimate (`apg_project`
    /// in the reference -- a single global dot product/norm over the whole `[seqLen,256]` latent,
    /// not per-channel), then re-derives the guided velocity from the guided denoised estimate.
    /// </summary>
    private float[] PredictVelocity(
        float[] latent, int seqLen,
        float[] condTokens, float[] nullCondTokens, int nCond,
        float[] secondsTotalRaw, float sigma, float cfgScale)
    {
        var condOutput = _transformer.Forward(latent, seqLen, condTokens, nCond, secondsTotalRaw, timestep: sigma);
        if (cfgScale == 1.0f) return condOutput;

        var uncondOutput = _transformer.Forward(latent, seqLen, nullCondTokens, nCond, secondsTotalRaw, timestep: sigma);

        int n = latent.Length;
        var condDenoised = new float[n];
        var uncondDenoised = new float[n];
        var diff = new float[n];
        for (int i = 0; i < n; i++)
        {
            condDenoised[i] = latent[i] - condOutput[i] * sigma;
            uncondDenoised[i] = latent[i] - uncondOutput[i] * sigma;
            diff[i] = condDenoised[i] - uncondDenoised[i];
        }

        // apg_project: orthogonal component of `diff` relative to `condDenoised`, treating the
        // whole latent as one vector (real `dim=[-1,-2]` reduction over both channel and time).
        double normSq = 0, dot = 0;
        for (int i = 0; i < n; i++) normSq += (double)condDenoised[i] * condDenoised[i];
        float invNorm = (float)(1.0 / (Math.Sqrt(normSq) + 1e-12));
        for (int i = 0; i < n; i++) dot += (double)diff[i] * (condDenoised[i] * invNorm);

        var velocity = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v1Normalized = condDenoised[i] * invNorm;
            float parallel = (float)dot * v1Normalized;
            float orthogonal = diff[i] - parallel;
            float cfgDenoised = condDenoised[i] + (cfgScale - 1f) * orthogonal;
            velocity[i] = (latent[i] - cfgDenoised) / sigma;
        }
        return velocity;
    }

    /// <summary>
    /// Real cross-attention conditioning assembly, matching `diffusion.py`'s
    /// `get_conditioning_inputs`: `cross_attn_cond = concat([prompt(256,768), seconds_total(1,768)])`,
    /// with the real T5GemmaConditioner's `padding_mode="learned"` substitution applied to padded
    /// prompt rows, and `seconds_total`'s own raw embedding returned separately for the DiT's
    /// global (AdaLN) conditioning input.
    /// </summary>
    private (float[] condTokens, float[] secondsTotalRaw) BuildConditioning(int[] promptTokenIds, float durationSeconds)
    {
        int realLen = Math.Min(promptTokenIds.Length, PromptMaxLength);
        var paddedIds = new int[PromptMaxLength];
        var mask = new bool[PromptMaxLength];
        for (int i = 0; i < PromptMaxLength; i++)
        {
            if (i < realLen)
            {
                paddedIds[i] = promptTokenIds[i];
                mask[i] = true;
            }
        }

        var promptEmbed = _textEncoder.Encode(paddedIds, mask);

        var paddingEmbedding = _weights.ReadF32("conditioner.conditioners.prompt.padding_embedding");
        for (int t = 0; t < PromptMaxLength; t++)
        {
            if (!mask[t]) paddingEmbedding.AsSpan().CopyTo(promptEmbed.AsSpan(t * CondTokenDim, CondTokenDim));
        }

        var secondsTotalRaw = SecondsTotalEmbedding(durationSeconds);

        var condTokens = new float[(PromptMaxLength + 1) * CondTokenDim];
        promptEmbed.AsSpan().CopyTo(condTokens);
        secondsTotalRaw.AsSpan().CopyTo(condTokens.AsSpan(PromptMaxLength * CondTokenDim, CondTokenDim));

        return (condTokens, secondsTotalRaw);
    }

    /// <summary>
    /// Real `NumberConditioner`/`NumberEmbedder` for `seconds_total`: normalize to [0,1] over
    /// `[min_val,max_val]=[0,384]`, `ExpoFourierFeatures(256, 0.5, 10000.0)`, then the real
    /// `conditioner.conditioners.seconds_total.embedder.embedding.1` linear (256→768, WITH bias).
    /// </summary>
    private float[] SecondsTotalEmbedding(float durationSeconds)
    {
        float normalized = Math.Clamp(durationSeconds, 0f, 384f) / 384f;

        const int fDim = 256;
        int half = fDim / 2;
        var feats = new float[fDim];
        float logMin = MathF.Log(0.5f);
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < half; i++)
        {
            float ramp = half == 1 ? 0f : (float)i / (half - 1);
            float freq = MathF.Exp(ramp * (logMax - logMin) + logMin);
            float arg = normalized * freq * 2f * MathF.PI;
            feats[i] = MathF.Cos(arg);
            feats[half + i] = MathF.Sin(arg);
        }

        var w = _weights.ReadF32("conditioner.conditioners.seconds_total.embedder.embedding.1.weight");
        var b = _weights.ReadF32("conditioner.conditioners.seconds_total.embedder.embedding.1.bias");
        return DiffusionOps.Linear(feats, w, b, 1, fDim, CondTokenDim);
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
