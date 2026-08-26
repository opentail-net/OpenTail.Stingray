using System.Collections.Generic;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// Abstraction for neural speech decoding in Chatterbox-Turbo.
/// </summary>
public interface IChatterboxDecoder
{
    /// <summary>
    /// Synthesizes 24kHz audio waveform samples from discrete speech tokens and speaker conditioning.
    /// </summary>
    float[] Decode(IReadOnlyList<int> speechTokens, float[] speakerFeatures);
}
