namespace OpenTail.Stingray.Audio.Parler;

/// <summary>
/// Real self-/cross-attention KV cache split for Parler-TTS's decoder, matching the real
/// `EncoderDecoderCache(self_attention_cache, cross_attention_cache)` architecture confirmed from
/// the real `huggingface/parler-tts` `modeling_parler_tts.py`/`_get_cache` (see
/// docs/audio-review-progress.md's Parler-TTS generation-loop section for the full derivation).
///
/// <para><b>Self-attention cache grows every step</b> (recomputed from the current decoder hidden
/// state and appended). <b>Cross-attention cache is built exactly once per layer</b>, projected
/// from the T5 encoder's fixed output the first time it's needed, then reused unchanged for every
/// subsequent decode step -- its sequence length is the encoder's length, not the decoder's, and
/// never grows.</para>
/// </summary>
public sealed class ParlerDecoderKvCache
{
    public readonly List<float[]>[] SelfK;
    public readonly List<float[]>[] SelfV;
    public readonly float[][]?[] CrossK;
    public readonly float[][]?[] CrossV;

    public ParlerDecoderKvCache(int numLayers)
    {
        SelfK = new List<float[]>[numLayers];
        SelfV = new List<float[]>[numLayers];
        CrossK = new float[][]?[numLayers];
        CrossV = new float[][]?[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            SelfK[i] = [];
            SelfV[i] = [];
        }
    }

    public int SelfLength(int layer) => SelfK[layer].Count;
    public bool CrossBuilt(int layer) => CrossK[layer] is not null;
}
