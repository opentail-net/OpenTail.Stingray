namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real per-layer streaming state for <see cref="VibeVoiceTokenizerEncoder.EncodeStreaming"/>/
/// <see cref="VibeVoiceTokenizerDecoder.DecodeStreaming"/>, ported from `tokenizer_audio.cpp`'s
/// `VibeVoiceTokenizerStreamingState` (not guessed): one real cache buffer per streaming conv
/// layer (downsample/stem/transpose convs and every ConvNeXt block's depthwise mixer conv, in the
/// exact real call order `build_encoder_streaming`/`build_decoder_streaming` uses), lazily
/// zero-initialized on first use (matching the reference's own lazy `state.caches` allocation).
/// Construct ONE instance per real audio stream and reuse it across every chunk -- constructing a
/// fresh instance per chunk is equivalent to never streaming at all (this is exactly the gap
/// `VibeVoiceGenerator`'s one-shot per-chunk `Encode`/`Decode` calls had before this class
/// existed).
/// </summary>
public sealed class VibeVoiceTokenizerStreamingState
{
    internal float[][]?[] Caches;
    private int _cursor;

    private VibeVoiceTokenizerStreamingState(int cacheCount) => Caches = new float[][]?[cacheCount];

    internal static VibeVoiceTokenizerStreamingState ForEncoder(VibeVoiceTokenizerEncoderWeights w) =>
        new(w.Stages.Sum(stage => 1 + stage.Length) + 1);

    internal static VibeVoiceTokenizerStreamingState ForDecoder(VibeVoiceTokenizerDecoderWeights w) =>
        new(w.Stages.Sum(stage => 1 + stage.Length) + 1);

    internal void BeginChunk() => _cursor = 0;

    internal ref float[][]? Next()
    {
        if (_cursor >= Caches.Length) throw new InvalidOperationException("VibeVoice streaming state cache count mismatch.");
        return ref Caches[_cursor++];
    }
}
