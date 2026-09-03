using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// End-to-end text-to-audio/music synthesis pipeline for Stable Audio 3 Medium. Structurally
/// identical to <see cref="StableAudioPipeline"/> (Small) -- same real conditioning assembly, CFG/
/// APG formula, and Euler loop -- but wired to <see cref="StableAudioMediumDiT"/> (real differential
/// attention) and <see cref="SameLargeVae"/> (real single-pass banded attention) instead of Small's
/// <see cref="StableAudioDiT"/>/<see cref="AcousticVae"/>. The T5Gemma text encoder checkpoint is
/// the literal same file as Small's (confirmed identical sha256, see docs/057-stable-audio-3-
/// implementation-plan.md's SA3_MODEL_MATRIX section), so <see cref="T5GemmaEncoder"/> is reused
/// completely unchanged.
///
/// <para>Duplicated from `StableAudioPipeline` rather than made generic over a DiT/VAE interface,
/// deliberately (same discipline as `StableAudioMediumDiT`/`SameLargeVae` themselves) -- unifying
/// behind docs/065's own proposed `IStableAudio3Engine` is a real, worthwhile next step once this
/// class is itself real-weight verified, not done speculatively in the same change that adds
/// Medium.</para>
/// </summary>
public sealed class StableAudioMediumPipeline : IDisposable
{
    private const int PromptMaxLength = 256;
    private const int CondTokenDim = 768;

    private readonly T5GemmaEncoder _textEncoder;
    private readonly StableAudioMediumDiT _transformer;
    private readonly SameLargeVae _vae;
    private readonly StableAudioParams _params;
    private readonly IWeightLoader _weights;
    private readonly GgufTokenizer? _tokenizer;
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public StableAudioParams Params => _params;

    /// <param name="dit">Loader open on the real `stable-audio-3-medium-base/model.safetensors` -- holds both the DiT and SAME-L VAE weights.</param>
    /// <param name="textEncoderWeights">Loader open on the real T5Gemma encoder checkpoint (identical file to Small's).</param>
    /// <param name="textEncoderDir">Directory holding the real T5Gemma `tokenizer.json`. Required for <see cref="StableAudioRequest.Prompt"/> (raw-string) requests.</param>
    public StableAudioMediumPipeline(IWeightLoader dit, IWeightLoader textEncoderWeights, string? textEncoderDir = null, StableAudioParams? @params = null)
    {
        _params = @params ?? new StableAudioParams { HiddenSize = 1536, Depth = 24, NumHeads = 24 };
        _weights = dit;
        _transformer = StableAudioMediumDiT.FromLoader(dit);
        _textEncoder = T5GemmaEncoder.FromLoader(textEncoderWeights);
        _vae = SameLargeVae.FromLoader(dit);

        if (textEncoderDir is not null)
        {
            var tokSource = HuggingFaceTokenizerSource.Load(textEncoderDir);
            if (!tokSource.IsUsable || tokSource.Source is null)
                throw new InvalidOperationException(
                    $"Failed to load T5Gemma tokenizer from '{textEncoderDir}': {string.Join("; ", tokSource.Rejections)}");
            _tokenizer = GgufTokenizer.FromSource(tokSource.Source);
        }
    }

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

    internal float[] GenerateFromLatent(
        float[] initialLatent, int seqLen, int[] promptTokenIds, float durationSeconds,
        int steps, float cfgScale, Action<int, int>? progress = null)
    {
        var latent = initialLatent;
        var (condTokens, secondsTotalRaw) = BuildConditioning(promptTokenIds, durationSeconds);
        int nCond = condTokens.Length / CondTokenDim;
        var nullCondTokens = new float[condTokens.Length];

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
