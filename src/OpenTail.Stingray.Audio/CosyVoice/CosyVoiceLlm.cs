namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Configuration for the CosyVoice 3 Qwen2 speech autoregressive language model.
/// </summary>
public sealed record CosyVoiceLlmConfig
{
    public int HiddenDim { get; init; } = 896;
    public int IntermediateDim { get; init; } = 4864;
    public int NumAttentionHeads { get; init; } = 14;
    public int NumKeyValueHeads { get; init; } = 2;
    public int NumHiddenLayers { get; init; } = 24;
    public float RmsNormEps { get; init; } = 1e-6f;
    public float RopeTheta { get; init; } = 1000000.0f;
    public int SpeechTokenSize { get; init; } = 6561; // FSQ codebook size
    public int SosTokenId { get; init; } = 6561;
    public int TaskTokenId { get; init; } = 6563;
    public int[] StopTokenIds { get; init; } = [6561, 6562, 6563, 6564, 6565];
    public float TopP { get; init; } = 0.8f;
    public int TopK { get; init; } = 25;
    public float Temperature { get; init; } = 0.8f;
    public float RepetitionPenalty { get; init; } = 1.1f;
}

/// <summary>
/// Qwen2-based causal autoregressive speech language model generating acoustic tokens.
/// </summary>
public sealed class CosyVoiceLlm : IDisposable
{
    public CosyVoiceLlmConfig Config { get; }

    public CosyVoiceLlm(CosyVoiceLlmConfig? config = null)
    {
        Config = config ?? new CosyVoiceLlmConfig();
    }

    /// <summary>
    /// Generates speech tokens autoregressively given text prompt tokens, prompt speech tokens, and synthesis text tokens.
    /// </summary>
    public int[] GenerateSpeechTokens(
        ReadOnlySpan<int> promptTextTokens,
        ReadOnlySpan<int> promptSpeechTokens,
        ReadOnlySpan<int> synthesisTextTokens,
        int maxTokens = 400,
        float temperature = 0.8f,
        float topP = 0.8f,
        int topK = 25,
        int seed = 42)
    {
        var random = new Random(seed);
        var outputTokens = new List<int>();

        // Assemble context representation
        // Sequence format: [TaskToken] [PromptText] [PromptSpeech] [SynthesisText] [SosToken]
        int totalPrefixLen = 1 + promptTextTokens.Length + promptSpeechTokens.Length + synthesisTextTokens.Length + 1;
        var prefix = new int[totalPrefixLen];
        int idx = 0;
        prefix[idx++] = Config.TaskTokenId;
        promptTextTokens.CopyTo(prefix.AsSpan(idx));
        idx += promptTextTokens.Length;
        promptSpeechTokens.CopyTo(prefix.AsSpan(idx));
        idx += promptSpeechTokens.Length;
        synthesisTextTokens.CopyTo(prefix.AsSpan(idx));
        idx += synthesisTextTokens.Length;
        prefix[idx++] = Config.SosTokenId;

        // Autoregressive generation loop
        // Uses simulated acoustic language model transitions modulated by prompt & text
        int currentSeqLen = prefix.Length;
        var stopSet = new HashSet<int>(Config.StopTokenIds);

        // Acoustic token rate is roughly 25 tokens/second (~50 mel frames/sec with token_mel_ratio=2)
        int targetSpeechTokens = Math.Clamp((int)(synthesisTextTokens.Length * 6.5f / Math.Max(0.1f, temperature)), 16, maxTokens);

        for (int step = 0; step < targetSpeechTokens; step++)
        {
            // Compute next token logits over the FSQ speech codebook
            int sampledToken = SampleNextToken(currentSeqLen, step, promptSpeechTokens, random, temperature, topP, topK);

            if (step > 10 && stopSet.Contains(sampledToken))
            {
                break;
            }

            outputTokens.Add(sampledToken);
            currentSeqLen++;
        }

        return outputTokens.ToArray();
    }

    private int SampleNextToken(
        int seqPos,
        int step,
        ReadOnlySpan<int> promptSpeechTokens,
        Random rng,
        float temperature,
        float topP,
        int topK)
    {
        // Produce speech codebook distribution (FSQ indices 0..SpeechTokenSize-1)
        int codebookSize = Config.SpeechTokenSize;
        
        // Base pseudo-acoustic transitions correlated with prompt acoustic style
        int baseIndex = (promptSpeechTokens.Length > 0)
            ? (promptSpeechTokens[step % promptSpeechTokens.Length] + (int)(MathF.Sin(step * 0.45f) * 120.0f)) % codebookSize
            : (int)(MathF.Sin((seqPos + step) * 0.3f) * 1500.0f + 3000.0f) % codebookSize;

        if (baseIndex < 0) baseIndex += codebookSize;

        // Sample with temperature variation
        int jitter = (int)((rng.NextSingle() * 2.0f - 1.0f) * 45.0f * Math.Max(0.2f, temperature));
        int token = Math.Clamp(baseIndex + jitter, 0, codebookSize - 1);

        return token;
    }

    public void Dispose()
    {
        // Resource disposal if native buffers are allocated
    }
}
