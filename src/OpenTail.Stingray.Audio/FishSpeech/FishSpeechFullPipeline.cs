using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Full Fish Speech S2 Pro text-to-speech pipeline: text -&gt; <see cref="FishSpeechPipeline"/>'s
/// slow-AR semantic-token generation (which internally also runs the real fast-AR codebook
/// expansion per frame, see <see cref="FishSpeechPipeline.GenerateFrames"/>) -&gt; real
/// <see cref="FishSpeechCodec"/> decode -&gt; mono float32 PCM.
///
/// <para>Analogous to <c>OrpheusPipeline.Synthesize</c>: wires together already golden-verified
/// components (slow-AR, fast-AR, codec -- each independently proven numerically correct against
/// real oracles, see docs/audio-review-progress.md's Fish Speech section) into one callable
/// end-to-end path. No new model math here -- purely plumbing.</para>
/// </summary>
public sealed class FishSpeechFullPipeline : IDisposable
{
    private readonly FishSpeechPipeline _talker;
    private readonly FishSpeechCodecWeights _codecWeights;

    public FishSpeechFullPipeline(string talkerGgufPath, string tokenizerDir, string codecGgufPath, int numLayers = 36, int ctxSize = 2048)
    {
        _talker = new FishSpeechPipeline(talkerGgufPath, tokenizerDir, numLayers, ctxSize);
        _codecWeights = new FishSpeechCodecWeights(codecGgufPath);
    }

    /// <summary>Full pipeline: text -&gt; mono float32 PCM (44.1kHz, matching the real codec's native rate).</summary>
    public float[] Synthesize(string text, int maxTokens = 200)
    {
        var (semanticTokens, codebooksPerFrame) = _talker.GenerateFrames(text, maxTokens);
        if (semanticTokens.Count == 0) return [];

        int t = semanticTokens.Count;
        var semanticCodes = semanticTokens.ToArray();

        // codebooksPerFrame[frame] = [semantic, residual_0, .., residual_8] (NumCodebooks=10 total,
        // index 0 duplicates the already-known semantic code -- see FishSpeechPipeline.GenerateFrames).
        int numResidual = codebooksPerFrame[0].Length - 1;
        var residualCodes = new int[numResidual][];
        for (int cb = 0; cb < numResidual; cb++)
        {
            residualCodes[cb] = new int[t];
            for (int ti = 0; ti < t; ti++)
                residualCodes[cb][ti] = codebooksPerFrame[ti][cb + 1];
        }

        return FishSpeechCodec.Decode(_codecWeights, semanticCodes, residualCodes);
    }

    public void Dispose()
    {
        _talker.Dispose();
        _codecWeights.Dispose();
    }
}
