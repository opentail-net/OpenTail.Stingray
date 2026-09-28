
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real MusicGen decoder-only LM forward pass, transcribed from the real `transformers`
/// `modeling_musicgen.py` (`MusicgenDecoder`/`MusicgenDecoderLayer`/`MusicgenAttention`) --
/// standard OPT/Bart-style PRE-norm decoder: `LayerNorm -> SelfAttn -> +residual`, `LayerNorm ->
/// CrossAttn(encoder_hidden_states) -> +residual`, `LayerNorm -> fc1 -> GELU -> fc2 ->
/// +residual`, then one final `LayerNorm` after the last layer. Self-attention IS scaled by
/// `1/sqrt(headDim)` (unlike T5) -- a real, easy-to-get-backwards difference between the two
/// attention flavors this single checkpoint contains.
///
/// <para>Embedding: <see cref="MusicGenConfig.NumCodebooks"/> separate embedding tables, SUMMED
/// (not concatenated) per step, plus the checkpoint's own precomputed sinusoidal position
/// embedding buffer added on top (`embed_scale` is 1.0 here -- real config `scale_embedding:
/// false`). `KvCache` carries self-attention K/V incrementally across autoregressive steps;
/// cross-attention K/V is computed ONCE from the text encoder's output and reused every step
/// (the real encoder_hidden_states never change during decoding).</para>
///
/// <para>The layer math is shared with AudioGen in <see cref="AudiocraftLmKernels"/>.</para>
/// </summary>
public static class MusicGenTransformer
{
    /// <summary>Per-layer self-attention KV cache plus the once-computed cross-attention K/V, both grown/set as decoding proceeds.</summary>
    public sealed class KvCache() : AudiocraftLmKvCache(MusicGenConfig.DecoderNumLayers);

    /// <summary>Precomputes cross-attention K/V projections from the text encoder's output hidden states -- done once per generation, reused every decode step. Real MusicGen projects the T5 encoder's 768-dim output up to the decoder's 1024-dim hidden size via `enc_to_dec_proj` FIRST (see <see cref="MusicGenTransformerWeights.EncToDecProjWeight"/>'s doc comment) -- do not feed raw T5 output straight into the cross-attention K/V weights.</summary>
    public static void PrepareCrossAttention(MusicGenTransformerWeights w, float[][] encoderHiddenStates, KvCache cache) =>
        AudiocraftLmKernels.PrepareCrossAttention(w.Lm, encoderHiddenStates, cache);

    /// <summary>
    /// Runs one decode step: `tokenColumn[codebook]` (one input token per codebook, from
    /// <see cref="DelayPattern.NextInputColumn"/>) -&gt; summed embedding -&gt; N decoder layers
    /// (growing <paramref name="cache"/>) -&gt; final LayerNorm -&gt; per-codebook logits.
    /// Returns `[codebook][CodebookSize]`.
    /// </summary>
    public static float[][] Step(MusicGenTransformerWeights w, int[] tokenColumn, KvCache cache) =>
        AudiocraftLmKernels.StepBatch(w.Lm, tokenColumn, [cache])[0];

    /// <summary>One decode step for sequences sharing the token column (CFG branches), weights streamed once. Returns `[sequence][codebook][CodebookSize]`.</summary>
    public static float[][][] StepBatch(MusicGenTransformerWeights w, int[] tokenColumn, KvCache[] caches) =>
        AudiocraftLmKernels.StepBatch(w.Lm, tokenColumn, caches);
}
