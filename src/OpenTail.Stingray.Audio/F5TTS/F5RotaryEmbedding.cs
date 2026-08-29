
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>Precomputes cos/sin tables for the interleaved RoPE used by every DiT block's attention.</summary>
public static class F5RotaryEmbedding
{
    /// <summary>Returns (cos, sin), each [numFrames * headDim/2], using the checkpoint's real `inv_freq` (not a recomputed theta formula).</summary>
    public static (float[] Cos, float[] Sin) Precompute(float[] invFreq, int numFrames)
    {
        int halfHead = invFreq.Length;
        var cos = new float[numFrames * halfHead];
        var sin = new float[numFrames * halfHead];
        for (int pos = 0; pos < numFrames; pos++)
        {
            int off = pos * halfHead;
            for (int k = 0; k < halfHead; k++)
            {
                float angle = pos * invFreq[k];
                cos[off + k] = MathF.Cos(angle);
                sin[off + k] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }
}
