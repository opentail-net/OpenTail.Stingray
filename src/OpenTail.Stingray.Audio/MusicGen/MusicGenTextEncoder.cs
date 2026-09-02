
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real stock-`t5-base` encoder forward pass for MusicGen's text conditioning. Thin wrapper over
/// the shared <see cref="Primitives.T5EncoderKernels"/> (see <see cref="MusicGenTextEncoderWeights"/>'s
/// doc comment for the 2026-09-02 DRY extraction that created it).
/// </summary>
public static class MusicGenTextEncoder
{
    public static float[][] Forward(NonGatedT5EncoderWeights w, int[] tokenIds) =>
        T5EncoderKernels.Forward(MusicGenTextEncoderWeights.Dims, w, tokenIds);
}
