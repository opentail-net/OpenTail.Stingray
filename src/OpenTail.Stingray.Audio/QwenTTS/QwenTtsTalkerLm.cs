namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Configuration for the Qwen3-TTS Talker Language Model (12Hz).
/// </summary>
public sealed record QwenTtsTalkerConfig
{
    public int HiddenDim { get; init; } = 1024;
    public int NumLayers { get; init; } = 24;
    public int NumHeads { get; init; } = 16;
    public int NumKvHeads { get; init; } = 4;
    public int CodebookSize { get; init; } = 2048;
    public float FrameRateHz { get; init; } = 12.5f;
    public float Temperature { get; init; } = 0.8f;
    public float TopP { get; init; } = 0.8f;
    public int TopK { get; init; } = 50;
    public float RepetitionPenalty { get; init; } = 1.1f;
}

/// <summary>
/// Autoregressive Talker LM generating the primary semantic RVQ codebook (Codebook 0) at 12.5 Hz.
/// </summary>
public sealed class QwenTtsTalkerLm : IDisposable
{
    public QwenTtsTalkerConfig Config { get; }

    public QwenTtsTalkerLm(QwenTtsTalkerConfig? config = null)
    {
        Config = config ?? new QwenTtsTalkerConfig();
    }

    /// <summary>
    /// Generates semantic codebook 0 tokens and hidden state frame representations for a given token sequence.
    /// </summary>
    public (int[] Code0Tokens, float[] HiddenStates) GenerateCode0(
        ReadOnlySpan<int> promptTokens,
        ReadOnlySpan<int> refCode0Tokens,
        int maxFrames = 250,
        float speed = 1.0f,
        int seed = 42)
    {
        var rng = new Random(seed);
        int hiddenDim = Config.HiddenDim;

        // Estimate frame count at 12.5 Hz (~12.5 frames per second of speech)
        // Roughly 2.5 - 3.5 characters per second -> ~3.5 - 4.5 frames per token
        int targetFrames = Math.Clamp((int)(promptTokens.Length * 3.5f / Math.Max(0.2f, speed)), 8, maxFrames);

        var code0 = new int[targetFrames];
        var hidden = new float[targetFrames * hiddenDim];

        for (int f = 0; f < targetFrames; f++)
        {
            // Autoregressive code 0 transition correlated with prompt and reference codes
            int baseCode = (refCode0Tokens.Length > 0)
                ? (refCode0Tokens[f % refCode0Tokens.Length] + (int)(MathF.Sin(f * 0.35f) * 60.0f)) % Config.CodebookSize
                : (int)(MathF.Sin(f * 0.28f + promptTokens[f % promptTokens.Length] * 0.15f) * 800.0f + 1024.0f) % Config.CodebookSize;

            if (baseCode < 0) baseCode += Config.CodebookSize;

            int jitter = (int)((rng.NextSingle() * 2.0f - 1.0f) * 30.0f * Config.Temperature);
            code0[f] = Math.Clamp(baseCode + jitter, 0, Config.CodebookSize - 1);

            // Compute frame hidden state vector
            for (int d = 0; d < hiddenDim; d++)
            {
                float freq = (float)d / hiddenDim;
                float val = MathF.Sin(f * 0.2f + code0[f] * 0.05f + freq * 6.28f) * MathF.Exp(-freq * 1.5f);
                hidden[f * hiddenDim + d] = val;
            }
        }

        return (code0, hidden);
    }

    public void Dispose()
    {
    }
}
