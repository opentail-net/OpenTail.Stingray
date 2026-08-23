using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Shared capability both real Qwen3-ASR LLM tensor sources (<see cref="QwenAsrLlmTensorSource"/>
/// GGUF-based, <see cref="QwenAsrLlmSafetensorsTensorSource"/> Safetensors-based) implement: real
/// multimodal audio-embedding injection via the synthetic-combined-embedding-table technique (see
/// either class's own doc comment for the full real derivation). Lets
/// <see cref="QwenAsrDecoder"/>'s real generation loop run against either weight format through
/// one shared code path instead of duplicating the loop per format.
/// </summary>
public interface IQwenAsrAudioConditionableSource : IModelTensorSource
{
    /// <summary>
    /// The token id an audio frame position maps to once <see cref="EnableAudioConditioning"/>
    /// has been called: audio frame `f` becomes <c>AudioTokenIdOffset + f</c> in the synthetic
    /// combined embedding space. -1 until enabled.
    /// </summary>
    int AudioTokenIdOffset { get; }

    /// <summary>
    /// Builds a synthetic combined `token_embd.weight` (real text-vocab rows followed by
    /// <paramref name="numAudioTokens"/> rows taken directly from
    /// <paramref name="audioEmbeddings"/>) and presents it under the same name. Irreversible on
    /// this instance and only valid for the audio clip it was built from.
    /// </summary>
    void EnableAudioConditioning(ReadOnlySpan<float> audioEmbeddings, int numAudioTokens);
}
