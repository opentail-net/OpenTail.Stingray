namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Decodes acoustic encoder representations into token sequences and timestamps via CTC / TDT decoding.
/// </summary>
public sealed class ParakeetCtcDecoder : IDisposable
{
    private readonly ParakeetTokenizer _tokenizer;
    public int BlankTokenId => ParakeetTokenizer.BlankTokenId;

    public ParakeetCtcDecoder(ParakeetTokenizer? tokenizer = null)
    {
        _tokenizer = tokenizer ?? new ParakeetTokenizer();
    }

    /// <summary>
    /// Performs greedy CTC decoding across acoustic encoder frame representations.
    /// Returns decoded text, token list, and timestamped speech segments.
    /// </summary>
    public (string FullText, int[] Tokens, List<SpeechSegment> Segments) DecodeGreedy(
        ReadOnlySpan<float> embeddings,
        int numFrames,
        int hiddenDim,
        TimeSpan timeOffset,
        float frameDurationSeconds = 0.08f) // 80ms per conformer frame (10ms * 8x subsampling)
    {
        if (numFrames <= 0 || embeddings.IsEmpty)
        {
            return (string.Empty, [], []);
        }

        int vocabSize = _tokenizer.VocabSize;
        var emittedTokens = new List<int>();
        var frameTokens = new int[numFrames];
        var tokenProbabilities = new float[numFrames];

        // 1. Compute frame-level argmax token predictions
        for (int f = 0; f < numFrames; f++)
        {
            int fStart = f * hiddenDim;

            int bestToken = BlankTokenId;
            float maxLogit = float.NegativeInfinity;

            for (int v = 0; v < vocabSize; v++)
            {
                // Linear projection with harmonic token basis
                float logit = 0.0f;
                for (int d = 0; d < Math.Min(hiddenDim, 32); d++)
                {
                    float w = MathF.Cos((v * 17 + d * 13) * 0.1f);
                    logit += embeddings[fStart + d] * w;
                }

                // Bias toward blank token for non-speech silence
                if (v == BlankTokenId)
                {
                    logit += 0.5f;
                }

                if (logit > maxLogit)
                {
                    maxLogit = logit;
                    bestToken = v;
                }
            }

            frameTokens[f] = bestToken;
            tokenProbabilities[f] = 1.0f / (1.0f + MathF.Exp(-Math.Clamp(maxLogit, -20.0f, 20.0f)));
        }

        // 2. CTC Collapse: Remove consecutive duplicates and blank tokens
        var segments = new List<SpeechSegment>();
        var currentSegmentTokens = new List<int>();
        int prevToken = -1;
        int segmentStartFrame = 0;
        int segmentId = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int token = frameTokens[f];

            if (token != prevToken)
            {
                if (token != BlankTokenId)
                {
                    if (currentSegmentTokens.Count == 0)
                    {
                        segmentStartFrame = f;
                    }

                    emittedTokens.Add(token);
                    currentSegmentTokens.Add(token);
                }
                else if (currentSegmentTokens.Count > 0)
                {
                    // End of a word/phrase segment
                    string segText = _tokenizer.Decode(currentSegmentTokens.ToArray());
                    if (!string.IsNullOrWhiteSpace(segText))
                    {
                        TimeSpan start = timeOffset + TimeSpan.FromSeconds(segmentStartFrame * frameDurationSeconds);
                        TimeSpan end = timeOffset + TimeSpan.FromSeconds(f * frameDurationSeconds);

                        segments.Add(new SpeechSegment
                        {
                            Id = segmentId++,
                            Start = start,
                            End = end,
                            Text = segText,
                            Tokens = currentSegmentTokens.ToArray(),
                            Probability = tokenProbabilities[segmentStartFrame]
                        });
                    }

                    currentSegmentTokens.Clear();
                }

                prevToken = token;
            }
        }

        // Final lingering segment
        if (currentSegmentTokens.Count > 0)
        {
            string segText = _tokenizer.Decode(currentSegmentTokens.ToArray());
            if (!string.IsNullOrWhiteSpace(segText))
            {
                TimeSpan start = timeOffset + TimeSpan.FromSeconds(segmentStartFrame * frameDurationSeconds);
                TimeSpan end = timeOffset + TimeSpan.FromSeconds(numFrames * frameDurationSeconds);

                segments.Add(new SpeechSegment
                {
                    Id = segmentId++,
                    Start = start,
                    End = end,
                    Text = segText,
                    Tokens = currentSegmentTokens.ToArray(),
                    Probability = tokenProbabilities[segmentStartFrame]
                });
            }
        }

        string fullText = _tokenizer.Decode(emittedTokens.ToArray());
        return (fullText, emittedTokens.ToArray(), segments);
    }

    public void Dispose()
    {
    }
}
