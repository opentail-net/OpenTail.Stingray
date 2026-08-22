using System;
using System.Collections.Generic;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Audio.QwenTTS;

/// <summary>
/// Real Talker autoregressive generation loop: real prompt (<see cref="QwenTtsTalkerPromptBuilder"/>)
/// prefilled through the real 28-layer <see cref="ForwardPass"/> trunk via
/// <see cref="QwenTtsTalkerTensorSource.SetPromptEmbedding"/>'s synthetic-embedding-table
/// technique, then decoded step-by-step to produce the semantic codebook (codebook 0) token
/// sequence.
///
/// <para><b>Real, confirmed engine-capability gap, not worked around</b>: the real Code
/// Predictor (per `examples/qwentts.cpp/src/code-predictor-forward.h`) needs the Talker's own
/// last-position transformer hidden state (post final norm) as its prefill conditioning --
/// `IForwardPass.LastHidden`/`ForwardEmbedding` exist on the interface but are unimplemented on
/// the concrete <see cref="ForwardPass"/> class this project's Engine actually runs (checked
/// directly, not assumed). This class therefore only produces the semantic codebook; wiring the
/// Code Predictor's real acoustic-codebook expansion needs either an Engine-level change to
/// expose `LastHidden` for `ForwardPass`, or a from-scratch Talker forward pass outside the
/// shared Engine (both real, scoped follow-ups -- see docs/audio-review-progress.md).</para>
/// </summary>
public static class QwenTtsTalkerGeneration
{
    /// <summary>
    /// Generates the semantic codebook (codebook 0) token sequence for the given utterance
    /// text. Greedy decoding, stopping at the real codec EOS id or maxNewTokens.
    /// </summary>
    public static int[] GenerateSemanticCodes(GgufModel rawModel, string utteranceText, int numLayers, int maxNewTokens = 200)
    {
        var weights = QwenTtsTalkerPromptBuilder.Weights.Load(rawModel);
        var tokenizer = GgufTokenizer.FromGgufModel(rawModel);
        var (promptEmbed, tRows) = QwenTtsTalkerPromptBuilder.BuildBasePrompt(weights, tokenizer, utteranceText);

        using var source = new QwenTtsTalkerTensorSource(rawModel, numLayers);
        source.SetPromptEmbedding(promptEmbed, tRows);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prefillIds = new int[tRows];
        for (int i = 0; i < tRows; i++) prefillIds[i] = i;
        var logits = fwd.Prefill(prefillIds).ToArray();

        var generated = new List<int>();
        int pos = tRows;
        var specials = weights.Specials;
        for (int step = 0; step < maxNewTokens; step++)
        {
            int codeId = ArgMax(logits);
            if (codeId == specials.CodecEosId) break;
            generated.Add(codeId);

            var stepRow = QwenTtsTalkerPromptBuilder.ProjectTextIds(weights, [specials.TtsPadId]);
            var codecVec = QwenTtsTalkerPromptBuilder.CodecEmbedRow(weights, codeId);
            for (int d = 0; d < QwenTtsTalkerPromptBuilder.TalkerHiddenDim; d++) stepRow[d] += codecVec[d];

            source.SetPromptEmbedding(stepRow, 1);
            logits = fwd.Forward(0, pos).ToArray();
            pos++;
        }

        return [.. generated];
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
