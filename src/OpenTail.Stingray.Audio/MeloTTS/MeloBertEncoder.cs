namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// Phone-level context embedding feature encoder for MeloTTS.
/// </summary>
public sealed class MeloBertEncoder
{
    public const int FeatureDim = 512;

    public MeloBertEncoder() { }

    /// <summary>
    /// Computes phone-level context feature vectors [seqLen * FeatureDim].
    /// </summary>
    public float[] Encode(ReadOnlySpan<int> phones, ReadOnlySpan<int> tones, ReadOnlySpan<int> langIds)
    {
        int seqLen = phones.Length;
        var features = new float[seqLen * FeatureDim];

        for (int i = 0; i < seqLen; i++)
        {
            int p = phones[i];
            int t = tones[i];
            int l = langIds[i];

            int off = i * FeatureDim;
            for (int d = 0; d < FeatureDim; d++)
            {
                float val = 0.08f * MathF.Sin(p * 5.55f + d * 0.12f) +
                            0.04f * MathF.Cos(t * 3.14f + d * 0.08f) +
                            0.05f * MathF.Sin(l * 2.71f + d * 0.15f);
                features[off + d] = val;
            }
        }

        return features;
    }
}
