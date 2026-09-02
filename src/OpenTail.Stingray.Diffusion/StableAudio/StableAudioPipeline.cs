using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Diffusion.TextEncoders;

namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Generation request for Stable Audio 3.
///
/// <para><b>Known gap, 2026-09-02</b>: no T5Gemma tokenizer has been wired into this project yet
/// (see docs/057-stable-audio-3-implementation-plan.md), so the prompt must be supplied as
/// already-tokenized ids (real `google/t5gemma-b-b-ul2` SentencePiece vocab, padded/truncated to
/// 256 tokens with a matching attention mask) rather than a raw string. <see cref="PromptTokenIds"/>
/// with length &lt; 256 is padded internally with the real T5GemmaConditioner's learned padding
/// embedding, matching `padding_mode="learned"`.</para>
/// </summary>
public sealed record StableAudioRequest
{
    public required int[] PromptTokenIds { get; init; }
    public float DurationSeconds { get; init; } = 10.0f;
    public int Steps { get; init; } = 25;
    public int Seed { get; init; } = -1;
    public required string OutputPath { get; init; }
    public Action<int, int>? Progress { get; init; }
}

/// <summary>
/// End-to-end text-to-audio and music synthesis pipeline for Stable Audio 3.
///
/// <para><b>Status, 2026-09-02</b>: the text encoder (<see cref="T5GemmaEncoder"/>) and DiT
/// (<see cref="StableAudioDiT"/>) are real, weight-driven ports, golden-verified for the encoder
/// (see <c>StableAudioT5GemmaEncoderGoldenParityTests</c>) and spec-complete-but-not-yet-verified
/// for the DiT. <see cref="AcousticVaeDecoder"/> is still the ORIGINAL placeholder stub (no real
/// weights) -- the real VAE (`TransformerResamplingBlock`) has not been ported yet, so end-to-end
/// output from this pipeline is not real audio until that lands. See
/// docs/057-stable-audio-3-implementation-plan.md for the full status.</para>
/// </summary>
public sealed class StableAudioPipeline : IDisposable
{
    private const int PromptMaxLength = 256;
    private const int CondTokenDim = 768;

    private readonly T5GemmaEncoder _textEncoder;
    private readonly StableAudioDiT _transformer;
    private readonly AcousticVaeDecoder _decoder;
    private readonly StableAudioParams _params;
    private readonly IWeightLoader _weights;
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public StableAudioParams Params => _params;

    /// <param name="dit">Loader open on the real DiT checkpoint (e.g. `small-music-base/model.safetensors`).</param>
    /// <param name="textEncoderWeights">Loader open on the real T5Gemma encoder checkpoint (e.g. the bundled `t5gemma-b-b-ul2/` subfolder).</param>
    public StableAudioPipeline(IWeightLoader dit, IWeightLoader textEncoderWeights, StableAudioParams? @params = null)
    {
        _params = @params ?? new StableAudioParams();
        _weights = dit;
        _transformer = StableAudioDiT.FromLoader(dit);
        _textEncoder = T5GemmaEncoder.FromLoader(textEncoderWeights);
        _decoder = new AcousticVaeDecoder(_params.LatentChannels, _params.AudioChannels, upsampleRatio: 1024);
    }

    /// <summary>
    /// Generates high-fidelity stereo audio from a pre-tokenized prompt and duration specification.
    /// </summary>
    public float[] Generate(StableAudioRequest request)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(request);

        float duration = Math.Max(0.5f, request.DurationSeconds);
        int seqLen = (int)Math.Ceiling(duration * _params.LatentFrameRate);
        int totalLatentElements = seqLen * _params.LatentChannels;

        var rng = request.Seed >= 0 ? new Random(request.Seed) : new Random();
        var latent = SampleGaussian(totalLatentElements, rng);

        var (condTokens, condMask, secondsTotalRaw) = BuildConditioning(request.PromptTokenIds, duration);

        int steps = Math.Max(1, request.Steps);
        for (int step = 0; step < steps; step++)
        {
            float t = 1.0f - (float)step / steps;
            float nextT = 1.0f - (float)(step + 1) / steps;
            float dt = nextT - t;

            var v = _transformer.Forward(
                latent, seqLen,
                condTokens, condTokens.Length / CondTokenDim, condMask,
                secondsTotalRaw,
                timestep: t);

            for (int i = 0; i < latent.Length; i++)
            {
                latent[i] += dt * v[i];
            }

            request.Progress?.Invoke(step + 1, steps);
        }

        float[] pcm = _decoder.Decode(latent, seqLen);

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
    /// Real cross-attention conditioning assembly, matching `diffusion.py`'s
    /// `get_conditioning_inputs`: `cross_attn_cond = concat([prompt(256,768), seconds_total(1,768)])`,
    /// with the real T5GemmaConditioner's `padding_mode="learned"` substitution applied to padded
    /// prompt rows, and `seconds_total`'s own raw embedding returned separately for the DiT's
    /// global (AdaLN) conditioning input.
    /// </summary>
    private (float[] condTokens, bool[] condMask, float[] secondsTotalRaw) BuildConditioning(int[] promptTokenIds, float durationSeconds)
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

        var condMask = new bool[PromptMaxLength + 1];
        mask.AsSpan().CopyTo(condMask);
        condMask[PromptMaxLength] = true;

        return (condTokens, condMask, secondsTotalRaw);
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
