namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real intermediate checkpoint between autoregressive generation (Global + Local/depth) and Flow
/// synthesis -- the real synthesis architecture conditions on FUSED CONTINUOUS hidden states, not
/// on the RVQ tokens themselves (docs/066-minimax-music3-future-plan.md, "preserve hidden states
/// correctly"). `GlobalHiddenStates[t]`/`LocalHiddenStates[t]` are the real per-frame hidden states
/// that feed <c>MiniMaxMusic3ConditionEncoder</c>.
/// </summary>
public sealed class Music3Representation
{
    public required int[] SemanticTokens { get; init; } // [frameCount], real 0..SemanticVocabSize-1
    public required int[][] AcousticTokens { get; init; } // [frameCount][7], real 0..RvqDepthDecoderAudioVocabSize-1
    public required float[][] GlobalHiddenStates { get; init; } // [frameCount][LanguageModelHiddenSize]
    public required float[][] LocalHiddenStates { get; init; } // [frameCount][7*RvqDepthDecoderHiddenSize], real c1..c7 concat order

    public int FrameCount => SemanticTokens.Length;
}
