using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Diffusion.AceStep.Conditioning;

/// <summary>
/// Real ACE-Step `AceStepTimbreEncoder`, transcribed from the real `diffusers`
/// `pipelines/ace_step/modeling_ace_step.py` -- see docs/064-acestep-implementation-plan.md.
///
/// <para><b>V1 scope</b>: only the real "no reference audio" path is implemented -- a real 30s
/// (750-frame @ 25Hz) slice of a real, self-derived `silence_latent` (see
/// <see cref="Vae.AceStepOobleckEncoder"/>'s doc comment for why this project derives it itself
/// rather than needing an external asset) is embedded, run through the real 4-layer bidirectional
/// encoder (SAME `AceStepEncoderLayer` class/weights shape as the lyric encoder -- shared via
/// <see cref="AceStepConditionEncoder.RunBidirectionalEncoder"/>), and CLS-like pooled (the real
/// reference takes hidden_states[:,0,:] -- the first sequence position's post-layer, post-norm
/// output -- as the pooled per-reference embedding, NOT an actual prepended CLS token). Real
/// multi-reference-audio batching (`unpack_timbre_embeddings`'s general N-references-per-batch
/// case) is out of scope for V1's single always-one-reference (silence) case, where it reduces to
/// a no-op reshape of the single pooled row.</para>
///
/// <para><b>Real, confirmed-unused-in-inference tensor</b>: `encoder.timbre_encoder.special_token`
/// (`[1,1,hidden]`) is a real learned parameter in the checkpoint, loaded here for completeness, but
/// the real `AceStepTimbreEncoder.forward` never references `self.special_token` anywhere -- traced
/// through the full real method body to confirm this is not a missed step (unlike the earlier
/// `condition_embedder` bug this session found and fixed, where an unused-looking loaded tensor
/// WAS a real omission).</para>
/// </summary>
public sealed class AceStepTimbreEncoderWeights
{
    public required CfmLinearWeight EmbedTokensWeight { get; init; } // [hidden,timbreHiddenDim], WITH bias
    public required float[] EmbedTokensBias { get; init; }
    public required AceStepEncoderLayerWeights[] Layers { get; init; } // 4 real layers
    public required float[] NormWeight { get; init; }
    public required float[] SpecialToken { get; init; } // real tensor, confirmed unused in forward (see class doc comment)

    public static AceStepTimbreEncoderWeights Load(SafetensorsLoader loader)
    {
        int hidden = AceStepConfig.HiddenSize;
        int timbreDim = AceStepConfig.TimbreHiddenDim;

        var layers = new AceStepEncoderLayerWeights[AceStepConfig.NumTimbreEncoderHiddenLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = LoadEncoderLayer(loader, $"encoder.timbre_encoder.layers.{i}");

        return new AceStepTimbreEncoderWeights
        {
            EmbedTokensWeight = CfmLinearWeight.FromF32(loader.ReadF32("encoder.timbre_encoder.embed_tokens.weight"), outDim: hidden, inDim: timbreDim),
            EmbedTokensBias = loader.ReadF32("encoder.timbre_encoder.embed_tokens.bias"),
            Layers = layers,
            NormWeight = loader.ReadF32("encoder.timbre_encoder.norm.weight"),
            SpecialToken = loader.ReadF32("encoder.timbre_encoder.special_token"),
        };
    }

    private static AceStepEncoderLayerWeights LoadEncoderLayer(SafetensorsLoader loader, string p)
    {
        int hidden = AceStepConfig.HiddenSize;
        int qDim = AceStepConfig.NumAttentionHeads * AceStepConfig.HeadDim;
        int kvDim = AceStepConfig.NumKeyValueHeads * AceStepConfig.HeadDim;
        int ffn = AceStepConfig.IntermediateSize;

        return new AceStepEncoderLayerWeights
        {
            InputLayerNormWeight = loader.ReadF32($"{p}.input_layernorm.weight"),
            QWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.q_proj.weight"), outDim: qDim, inDim: hidden),
            KWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.k_proj.weight"), outDim: kvDim, inDim: hidden),
            VWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.v_proj.weight"), outDim: kvDim, inDim: hidden),
            OWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.self_attn.o_proj.weight"), outDim: hidden, inDim: qDim),
            QNormWeight = loader.ReadF32($"{p}.self_attn.q_norm.weight"),
            KNormWeight = loader.ReadF32($"{p}.self_attn.k_norm.weight"),
            PostAttnLayerNormWeight = loader.ReadF32($"{p}.post_attention_layernorm.weight"),
            MlpGateWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.gate_proj.weight"), outDim: ffn, inDim: hidden),
            MlpUpWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.up_proj.weight"), outDim: ffn, inDim: hidden),
            MlpDownWeight = CfmLinearWeight.FromF32(loader.ReadF32($"{p}.mlp.down_proj.weight"), outDim: hidden, inDim: ffn),
        };
    }
}

public static class AceStepTimbreEncoder
{
    /// <summary>Real V1-scoped forward: embed a real `[frames][timbreHiddenDim(64)]` acoustic latent (the real "no reference audio" path always uses a 750-frame slice of the real `silence_latent`), run the 4-layer bidirectional encoder, and CLS-like pool (return position 0's post-norm row). Returns a single `[hidden(2048)]` row.</summary>
    public static unsafe float[] Forward(AceStepTimbreEncoderWeights w, float[][] acousticLatent)
    {
        int hidden = AceStepConfig.HiddenSize;
        int seqLen = acousticLatent.Length;

        var embeds = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            var row = new float[hidden];
            fixed (float* rp = acousticLatent[i], pp = row, bp = w.EmbedTokensBias)
                w.EmbedTokensWeight.MatMul(rp, 1, pp, bp);
            embeds[i] = row;
        }

        var encoded = AceStepConditionEncoder.RunBidirectionalEncoder(w.Layers, embeds, w.NormWeight);
        return encoded[0]; // real CLS-like pooling: first sequence position.
    }
}
