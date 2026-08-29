
namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// MeloTTS's VITS2 TextEncoder (enc_p), ported from `examples/melotts-py/models.py`'s
/// `TextEncoder.forward` against real ONNX weights.
///
/// x = (emb(tok) + tone_emb(tone) + language_emb(lang) + bert_proj(bert) + ja_bert_proj(ja_bert))
///     * sqrt(hidden)
/// then `attentions.Encoder`: N relative-attention layers (identical math to Piper's, shared via
/// <see cref="VitsAttentionKernels"/>), with a speaker-embedding injection
/// (`x = x + spk_emb_linear(g)`) at a fixed layer index (cond_layer_idx, default 2) -- Piper has
/// no such conditioning (single-speaker checkpoint), this is new versus the Piper port.
///
/// Per <see cref="MeloOnnxWeights"/>'s doc comment, for THIS checkpoint bert/ja_bert are always
/// zero (so their conv projections contribute bias only) and `language` is not a real per-
/// utterance input -- it resolves to a fixed alternating pattern keyed by token position parity
/// (id 0 for even index, id 3 for odd index), confirmed empirically against the real ONNX graph.
/// </summary>
public static class MeloTextEncoder
{
    /// <summary>
    /// tokens/tones are phoneme/tone ids, same length. speakerId indexes emb_g (256 speakers).
    /// Returns (encoderHidden, mu, logs), all channel-first [hidden, T].
    /// </summary>
    public static (float[] EncoderHidden, float[] Mu, float[] Logs) Forward(
        MeloOnnxWeights w, ReadOnlySpan<int> tokens, ReadOnlySpan<int> tones, int speakerId) =>
        Forward(w, tokens, tones, speakerId, out _);

    /// <summary>Overload exposing the pre-encoder embedding sum (channel-first [dim,t]) for golden-output debugging.</summary>
    public static (float[] EncoderHidden, float[] Mu, float[] Logs) Forward(
        MeloOnnxWeights w, ReadOnlySpan<int> tokens, ReadOnlySpan<int> tones, int speakerId, out float[] preEncoderX)
    {
        int t = tokens.Length;
        int dim = w.HiddenDim;
        float embScale = MathF.Sqrt(dim);

        // bert_proj/ja_bert_proj: input is always zero for this checkpoint (see class doc), so
        // their contribution reduces to the conv bias, broadcast identically across every timestep.
        var x = new float[dim * t]; // channel-first [dim, t]
        for (int ti = 0; ti < t; ti++)
        {
            int tok = tokens[ti];
            int tone = tones[ti];
            int lang = (ti % 2 == 0) ? 0 : 3; // fixed position-parity pattern, see class doc

            int tokRow = tok * dim;
            int toneRow = tone * dim;
            int langRow = lang * dim;

            for (int c = 0; c < dim; c++)
            {
                float v = w.EmbWeight[tokRow + c] + w.ToneEmbWeight[toneRow + c] + w.LanguageEmbWeight[langRow + c]
                          + w.BertProjBias[c] + w.JaBertProjBias[c];
                x[c * t + ti] = v * embScale;
            }
        }

        preEncoderX = (float[])x.Clone();

        // Speaker embedding g = emb_g[speakerId], projected once (spk_emb_linear), added at
        // cond_layer_idx (default 2) inside the per-layer loop below -- attentions.py Encoder.forward.
        var g = new float[w.GinChannels];
        int gRow = speakerId * w.GinChannels;
        Array.Copy(w.EmbGWeight, gRow, g, 0, w.GinChannels);
        var spkProjected = MeloRelativeEncoder.LinearVec(g, w.SpkEmbLinearWeight, w.SpkEmbLinearBias, dim, w.GinChannels);

        x = MeloRelativeEncoder.Forward(x, t, dim, w.NumHeads, w.WindowSize, w.FfnKernel, w.Layers, w.SpkEmbCondLayerIdx, spkProjected);

        var stats = VitsAttentionKernels.Conv1x1(x, dim, t, w.ProjWeight, w.ProjBias, 2 * dim);
        var mu = new float[dim * t];
        var logs = new float[dim * t];
        Array.Copy(stats, 0, mu, 0, dim * t);
        Array.Copy(stats, dim * t, logs, 0, dim * t);
        return (x, mu, logs);
    }

}
