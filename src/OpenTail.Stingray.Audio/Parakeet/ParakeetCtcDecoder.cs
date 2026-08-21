using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Audio.Parakeet;

/// <summary>
/// Decodes acoustic encoder representations into token sequences and timestamps via CTC / TDT decoding.
/// Supports both procedural basis and real GGUF weights via <see cref="ParakeetWeights"/>.
/// </summary>
public sealed class ParakeetCtcDecoder : IDisposable
{
    private readonly ParakeetTokenizer _tokenizer;
    private readonly ParakeetWeights? _weights;

    public int BlankTokenId => _weights?.BlankTokenId ?? ParakeetTokenizer.BlankTokenId;

    public ParakeetCtcDecoder(ParakeetTokenizer? tokenizer = null, ParakeetWeights? weights = null)
    {
        _tokenizer = tokenizer ?? new ParakeetTokenizer();
        _weights = weights;
    }

    /// <summary>
    /// Performs greedy CTC decoding over real per-frame CTC logits (`ctcLogits[frame][vocab+1]`,
    /// as produced by <see cref="ParakeetConformerEncoder.Forward"/>'s CTC head -- no synthetic
    /// scoring, just argmax + collapse-repeats + drop-blank, the standard greedy CTC recipe).
    /// Returns decoded text, token list, and timestamped speech segments.
    /// </summary>
    public (string FullText, int[] Tokens, List<SpeechSegment> Segments) DecodeGreedy(
        IReadOnlyList<float[]> ctcLogits,
        int numFrames,
        TimeSpan timeOffset,
        float frameDurationSeconds = 0.08f) // 80ms per conformer frame (10ms * 8x subsampling)
    {
        if (numFrames <= 0 || ctcLogits.Count == 0)
        {
            return (string.Empty, [], []);
        }

        var emittedTokens = new List<int>();
        var frameTokens = new int[numFrames];
        var tokenProbabilities = new float[numFrames];

        // 1. Compute frame-level argmax token predictions directly from the real CTC logits.
        for (int f = 0; f < numFrames; f++)
        {
            var logits = ctcLogits[f];
            int bestToken = BlankTokenId;
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < logits.Length; v++)
            {
                if (logits[v] > maxLogit)
                {
                    maxLogit = logits[v];
                    bestToken = v;
                }
            }

            frameTokens[f] = bestToken;
            tokenProbabilities[f] = 1.0f / (1.0f + MathF.Exp(-Math.Clamp(maxLogit, -20f, 20f)));
        }

        // 2. CTC Collapse: remove repeated consecutive tokens and blank tokens
        int prevToken = -1;
        var segmentTokens = new List<int>();
        var segments = new List<SpeechSegment>();
        int segmentStartFrame = 0;
        int segId = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int t = frameTokens[f];

            if (t != BlankTokenId && t != prevToken)
            {
                emittedTokens.Add(t);
                segmentTokens.Add(t);

                if (segmentTokens.Count == 1)
                {
                    segmentStartFrame = f;
                }
            }

            // Word / clause boundary detection on whitespace token prefix
            string tokStr = _tokenizer.GetToken(t);
            if (tokStr.StartsWith("\u2581") && segmentTokens.Count > 1)
            {
                TimeSpan start = timeOffset + TimeSpan.FromSeconds(segmentStartFrame * frameDurationSeconds);
                TimeSpan end = timeOffset + TimeSpan.FromSeconds((f + 1) * frameDurationSeconds);

                string segText = _tokenizer.Decode(segmentTokens.ToArray());
                if (!string.IsNullOrWhiteSpace(segText))
                {
                    segments.Add(new SpeechSegment
                    {
                        Id = segId++,
                        Start = start,
                        End = end,
                        Text = segText,
                        Tokens = segmentTokens.ToArray(),
                        Probability = tokenProbabilities[f]
                    });
                }
                segmentTokens.Clear();
                segmentTokens.Add(t);
                segmentStartFrame = f;
            }

            prevToken = t;
        }

        // Flush trailing segment
        if (segmentTokens.Count > 0)
        {
            TimeSpan start = timeOffset + TimeSpan.FromSeconds(segmentStartFrame * frameDurationSeconds);
            TimeSpan end = timeOffset + TimeSpan.FromSeconds(numFrames * frameDurationSeconds);
            string segText = _tokenizer.Decode(segmentTokens.ToArray());

            if (!string.IsNullOrWhiteSpace(segText))
            {
                segments.Add(new SpeechSegment
                {
                    Id = segId,
                    Start = start,
                    End = end,
                    Text = segText,
                    Tokens = segmentTokens.ToArray(),
                    Probability = tokenProbabilities[^1]
                });
            }
        }

        string fullText = _tokenizer.Decode(emittedTokens.ToArray());

        if (segments.Count == 0 && !string.IsNullOrWhiteSpace(fullText))
        {
            segments.Add(new SpeechSegment
            {
                Id = 0,
                Start = timeOffset,
                End = timeOffset + TimeSpan.FromSeconds(numFrames * frameDurationSeconds),
                Text = fullText,
                Tokens = emittedTokens.ToArray(),
                Probability = 0.95f
            });
        }

        return (fullText, emittedTokens.ToArray(), segments);
    }

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
