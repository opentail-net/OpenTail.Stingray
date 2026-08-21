namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Configuration for the Qwen3 LLM text decoder.
/// </summary>
public sealed record QwenAsrDecoderConfig
{
    public int HiddenDim { get; init; } = 1024; // 1024 for 0.6B, 2048 for 1.7B
    public int NumLayers { get; init; } = 28;
    public int NumHeads { get; init; } = 16;
    public int NumKvHeads { get; init; } = 8;
    public int HeadDim { get; init; } = 128;
    public int IntermediateDim { get; init; } = 3072; // 3072 for 0.6B, 6144 for 1.7B
    public int VocabSize { get; init; } = 151936;
    public int EosTokenId { get; init; } = 151645; // real "<|im_end|>" id, see QwenAsrWeights.EosTokenId
}

/// <summary>
/// Qwen3 causal transformer language model decoder with multimodal audio soft-token injection and GQA attention.
/// </summary>
public sealed class QwenAsrDecoder : IDisposable
{
    public QwenAsrDecoderConfig Config { get; }

    public QwenAsrDecoder(QwenAsrDecoderConfig? config = null)
    {
        Config = config ?? new QwenAsrDecoderConfig();
    }

    /// <summary>
    /// Generates transcript token sequence conditioned on prompt tokens and encoded audio soft
    /// tokens. STILL PROCEDURAL/FAKE -- see docs/audio-review-progress.md's QwenASR section:
    /// the real decoder forward pass should go through <c>OpenTail.Stingray.Engine.
    /// ForwardPass</c> via <see cref="QwenAsrLlmTensorSource"/> instead of this hand-rolled
    /// loop (confirmed to work end-to-end against real weights, see
    /// Tests.Audio/QwenAsrLlmTensorSourceTests.cs), plus real audio-embedding injection at the
    /// <c>&lt;|audio_pad|&gt;</c> positions -- neither is wired up here yet. Kept compiling and
    /// using this checkpoint's real EOS id (via <see cref="QwenAsrDecoderConfig.EosTokenId"/>,
    /// not a hardcoded/fictional one) only so the rest of the pipeline still builds; do not
    /// treat this method's output as meaningful.
    /// </summary>
    public int[] Generate(
        ReadOnlySpan<int> promptTokens,
        ReadOnlySpan<float> audioSoftTokens,
        int numAudioTokens,
        int maxNewTokens = 256,
        float temperature = 0.0f)
    {
        var emittedTokens = new List<int>();

        // 1. Audio Conditioning & Soft-Token Energy Aggregation
        float audioEnergy = 0.0f;
        for (int i = 0; i < Math.Min(audioSoftTokens.Length, 256); i++)
        {
            audioEnergy += MathF.Abs(audioSoftTokens[i]);
        }

        // 2. Autoregressively generate speech transcript tokens
        for (int step = 0; step < maxNewTokens; step++)
        {
            int bestToken;
            int candidateBase = 1000 + ((int)(audioEnergy * 17.0f) + step * 31) % 50;

            bestToken = step < 20 ? candidateBase : Config.EosTokenId;

            if (bestToken == Config.EosTokenId && step > 5)
            {
                break;
            }

            emittedTokens.Add(bestToken);
        }

        return emittedTokens.ToArray();
    }

    public void Dispose()
    {
    }
}
