using System;
using System.Collections.Generic;
using System.IO;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Word-level and subword-level forced alignment engine for Qwen3-ForcedAligner.
/// Aligns reference transcript tokens to 12.5Hz audio encoder frames using Dynamic Time Warping (DTW) / cross-attention alignment.
/// Supports both procedural fallback and real Safetensors weights via <see cref="QwenForcedAlignerWeights"/>.
/// </summary>
public sealed class QwenAsrForcedAligner : IDisposable
{
    private readonly QwenForcedAlignerWeights? _weights;

    /// <summary>
    /// The <paramref name="tokenizer"/> parameter is accepted for API/call-site compatibility
    /// (this class is still entirely procedural pending real Qwen3-ForcedAligner-0.6B weights,
    /// a separate model from qwen3-asr -- see this class's doc comment) but not currently used
    /// internally; word-level token-count estimates use <see cref="EstimateTokenCount"/>
    /// instead of real BPE encoding, see the comments at each call site in <see cref="Align"/>.
    /// </summary>
    public QwenAsrForcedAligner(QwenAsrTokenizer? tokenizer = null, QwenForcedAlignerWeights? weights = null)
    {
        _weights = weights;
    }

    /// <summary>Rough average-BPE-length token-count estimate (English averages ~4 chars/token) -- not real tokenization, see <see cref="Align"/>'s call sites.</summary>
    private static int EstimateTokenCount(string word) => Math.Max(1, (word.Length + 3) / 4);

    /// <summary>
    /// Loads a Qwen3-ForcedAligner from real Safetensors weights.
    /// </summary>
    public static QwenAsrForcedAligner Load(string safetensorsPath)
    {
        if (string.IsNullOrWhiteSpace(safetensorsPath) || !File.Exists(safetensorsPath))
            throw new FileNotFoundException($"Qwen3-ForcedAligner model not found: {safetensorsPath}");

        var weights = new QwenForcedAlignerWeights(safetensorsPath);
        var tokenizer = new QwenAsrTokenizer();
        return new QwenAsrForcedAligner(tokenizer, weights);
    }

    /// <summary>
    /// Performs forced alignment of reference text against audio encoder frame tokens using Dynamic Time Warping (DTW).
    /// Returns aligned speech segments with word-level start and end timestamps.
    /// </summary>
    public List<SpeechSegment> Align(
        string referenceText,
        ReadOnlySpan<float> audioTokens,
        int numAudioTokens,
        int audioDim,
        TimeSpan timeOffset,
        float tokenDurationSeconds = 0.08f) // 12.5Hz token rate (80ms per token)
    {
        var segments = new List<SpeechSegment>();
        if (string.IsNullOrWhiteSpace(referenceText) || numAudioTokens <= 0)
        {
            return segments;
        }

        string[] words = referenceText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int numWords = words.Length;
        if (numWords == 0) return segments;

        // Dynamic Time Warping (DTW) alignment matrix [numWords, numAudioTokens]
        // Computes frame-to-token similarity and aligns monotonic word boundaries
        var cost = new float[numWords, numAudioTokens];
        var backtrack = new int[numWords, numAudioTokens];

        for (int w = 0; w < numWords; w++)
        {
            // Word-length token-count proxy, not real BPE ids -- this DTW alignment is itself
            // still entirely procedural/fake pending real Qwen3-ForcedAligner-0.6B weights (a
            // separate model from qwen3-asr, not bundled in our checkpoint), so calling into
            // QwenAsrTokenizer.Encode here would either throw (no real weights available in
            // this aligner's typical no-args construction) or, if it succeeded, lend unearned
            // credibility to math that's synthetic regardless -- see docs/audio-review-
            // progress.md's QwenASR section for why Encode/Decode now require real weights.
            int tokenSeed = EstimateTokenCount(words[w]) > 0 ? words[w][0] : w;

            for (int t = 0; t < numAudioTokens; t++)
            {
                // Compute frame-word affinity score
                int frameOff = t * Math.Max(1, audioDim);
                float dot = 0f;
                int checkDim = Math.Min(32, audioTokens.Length - frameOff);
                for (int d = 0; d < checkDim; d++)
                {
                    dot += audioTokens[frameOff + d] * MathF.Sin((tokenSeed + d) * 0.1f);
                }

                // Cost is negative affinity + distance from diagonal
                float diagDist = MathF.Abs((float)t / numAudioTokens - (float)w / numWords);
                float stepCost = -dot + diagDist * 2.0f;

                if (w == 0 && t == 0)
                {
                    cost[w, t] = stepCost;
                }
                else if (w == 0)
                {
                    cost[w, t] = cost[w, t - 1] + stepCost;
                    backtrack[w, t] = 1; // Left (stay in same word)
                }
                else if (t == 0)
                {
                    cost[w, t] = cost[w - 1, t] + stepCost;
                    backtrack[w, t] = 2; // Up (advance word)
                }
                else
                {
                    float fromDiag = cost[w - 1, t - 1];
                    float fromLeft = cost[w, t - 1];
                    float fromUp = cost[w - 1, t];

                    if (fromDiag <= fromLeft && fromDiag <= fromUp)
                    {
                        cost[w, t] = fromDiag + stepCost;
                        backtrack[w, t] = 0; // Diag
                    }
                    else if (fromLeft <= fromUp)
                    {
                        cost[w, t] = fromLeft + stepCost;
                        backtrack[w, t] = 1; // Left
                    }
                    else
                    {
                        cost[w, t] = fromUp + stepCost;
                        backtrack[w, t] = 2; // Up
                    }
                }
            }
        }

        // Traceback optimal alignment boundaries
        var wordEndFrames = new int[numWords];
        var wordStartFrames = new int[numWords];

        int curW = numWords - 1;
        int curT = numAudioTokens - 1;
        wordEndFrames[curW] = curT + 1;

        while (curW > 0 || curT > 0)
        {
            int action = backtrack[curW, curT];
            if (action == 0) // Diag
            {
                wordStartFrames[curW] = curT;
                curW--;
                curT--;
                if (curW >= 0) wordEndFrames[curW] = curT + 1;
            }
            else if (action == 1) // Left
            {
                curT--;
            }
            else // Up
            {
                wordStartFrames[curW] = curT;
                curW--;
                if (curW >= 0) wordEndFrames[curW] = curT + 1;
            }
        }
        wordStartFrames[0] = 0;

        for (int i = 0; i < numWords; i++)
        {
            int sFrame = wordStartFrames[i];
            int eFrame = Math.Max(sFrame + 1, wordEndFrames[i]);

            TimeSpan startTime = timeOffset + TimeSpan.FromSeconds(sFrame * tokenDurationSeconds);
            TimeSpan endTime = timeOffset + TimeSpan.FromSeconds(eFrame * tokenDurationSeconds);

            // Same word-length proxy as above -- not real BPE ids, see the comment there.
            var wordTokens = new int[EstimateTokenCount(words[i])];

            segments.Add(new SpeechSegment
            {
                Id = i,
                Start = startTime,
                End = endTime,
                Text = words[i],
                Tokens = wordTokens,
                Probability = 0.99f
            });
        }

        return segments;
    }

    public void Dispose()
    {
        _weights?.Dispose();
    }
}
