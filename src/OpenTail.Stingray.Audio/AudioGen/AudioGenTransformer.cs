
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Real AudioGen decoder-only LM forward pass, transcribed from the real `audiocraft.modules
/// .transformer` (`StreamingTransformerLayer`/`StreamingMultiheadAttention`) and
/// `audiocraft.models.lm` (`LMModel.forward`) source (pip-installed and read directly,
/// 2026-09-02 -- see docs/done/063-audiogen-implementation-plan.md).
///
/// <para><b>Real differences from MusicGen's HF-format decoder</b> (do not copy MusicGen's
/// assumptions here): (1) self- AND cross-attention use a single FUSED `in_proj_weight`
/// (`[3*hidden,hidden]`, Q/K/V concatenated) rather than separate Q/K/V matrices -- for
/// cross-attention, the query third is applied to the hidden state, the key/value thirds to the
/// conditioning tensor (real `nn.MultiheadAttention`-style behavior when `query != key`).
/// (2) Positional embedding is real sinusoidal, COMPUTED not loaded (`positional_embedding:
/// sin`): `phase = pos / max_period^(i/(halfDim-1))`, embedding = `concat([cos(phase),
/// sin(phase)])` -- note COS FIRST, sin second (the opposite half-order from MusicGen HF's
/// stored `[sin,cos]` buffer; confirmed from the real `create_sin_embedding` source, do not
/// assume the two conventions match). (3) Self-attention Q IS scaled by `1/sqrt(headDim)` (same
/// as MusicGen). (4) Real layer order (confirmed `norm_first=true`, `StreamingTransformerLayer
/// .forward`): `x += self_attn(norm1(x)); x += cross_attn(norm_cross(x), condSrc); x +=
/// ffn(norm2(x))` -- same shape as MusicGen's pre-norm layer, just different norm names/no
/// linear-layer bias anywhere (`bias_ff`/`bias_attn`/`bias_proj` all false; LayerNorms still
/// carry bias). (5) Final `out_norm` after the last layer, then per-codebook `linears.{q}` heads
/// (no bias).</para>
///
/// <para>The layer math is shared with MusicGen in <see cref="AudiocraftLmKernels"/>.</para>
/// </summary>
public static class AudioGenTransformer
{
    public sealed class KvCache() : AudiocraftLmKvCache(AudioGenConfig.NumLayers);

    /// <summary>Precomputes cross-attention K/V from the T5 text encoder's output, projected once through `output_proj` (T5's 1024-dim -&gt; 1536-dim) then through each layer's fused cross in_proj_weight's K/V thirds -- done once per generation, reused every decode step.</summary>
    public static void PrepareCrossAttention(AudioGenTransformerWeights w, float[][] encoderHiddenStates, KvCache cache) =>
        AudiocraftLmKernels.PrepareCrossAttention(w.Lm, encoderHiddenStates, cache);

    /// <summary>Runs one decode step: `tokenColumn[codebook]` -&gt; summed embedding + sinusoidal position -&gt; N decoder layers (growing <paramref name="cache"/>) -&gt; final norm -&gt; per-codebook logits. Returns `[codebook][CodebookSize]`.</summary>
    public static float[][] Step(AudioGenTransformerWeights w, int[] tokenColumn, KvCache cache) =>
        AudiocraftLmKernels.StepBatch(w.Lm, tokenColumn, [cache])[0];

    /// <summary>
    /// <see cref="Step"/> for several sequences that share the token column and position (classifier-free guidance's
    /// conditional and unconditional branches): every projection runs as one B-row matmul, so each weight is streamed
    /// once per step instead of once per branch. Returns `[b][codebook][CodebookSize]`.
    /// </summary>
    public static float[][][] StepBatch(AudioGenTransformerWeights w, int[] tokenColumn, KvCache[] caches) =>
        AudiocraftLmKernels.StepBatch(w.Lm, tokenColumn, caches);

    /// <summary>Real `create_sin_embedding`: `phase = pos / maxPeriod^(i/(halfDim-1))` for `i` in `[0,halfDim)`, embedding = `concat([cos(phase), sin(phase)])` -- cos in the FIRST half, sin in the second (confirmed from the real `audiocraft.modules.transformer` source; do not assume MusicGen HF's `[sin,cos]` order applies here).</summary>
    internal static void AddSinusoidalPositionEmbedding(float[] x, int position, int dim)
    {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            float exponent = i / (float)(halfDim - 1);
            float divisor = MathF.Pow(AudioGenConfig.SinusoidalMaxPeriod, exponent);
            float phase = position / divisor;
            x[i] += MathF.Cos(phase);
            x[halfDim + i] += MathF.Sin(phase);
        }
    }
}
