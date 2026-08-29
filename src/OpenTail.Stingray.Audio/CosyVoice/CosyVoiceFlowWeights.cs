
namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// Real weights for CosyVoice2's flow "token -> mu conditioning" encoder
/// (`models/cosyvoice2_flow.safetensors`, converted this session from the official `flow.pt`).
/// Architecturally identical to `Chatterbox/ChatterboxS3GenWeights.cs`'s `UpsampleConformerEncoder`
/// (same S3Gen lineage, confirmed by real tensor names during the Phase-0 audit -- see
/// docs/audio-review-progress.md's CosyVoice section) but sourced from plain HF-style
/// safetensors names (`encoder.encoders.{i}.self_attn.linear_q.weight` etc) rather than
/// Chatterbox's GGUF-converted short names (`sa.lq.weight`), so this is a parallel loader/
/// layer-struct pair rather than a direct reuse of Chatterbox's classes.
///
/// Real dims (read directly, not assumed): hidden=512, heads=8, head_dim=64, ffn=2048,
/// mel_channels=80, spk_embed_dim=192 (CAMPPlus x-vector), speech_vocab=6561 (this flow-local
/// `input_embedding` table is separate from the LLM's own `speech_embedding`), 6 `encoders` +
/// 4 `up_encoders` blocks (both counts read from the file, not hardcoded).
/// </summary>
public sealed class CosyVoiceFlowWeights : IDisposable, IS3GenFlowEncoderWeights
{
    public SafetensorsLoader Loader { get; }

    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int FfnDim { get; }
    public int MelChannels { get; }
    public int SpkEmbedDim { get; }
    public int SpeechVocabSize { get; }
    public int NumEncoders { get; }
    public int NumUpEncoders { get; }

    public float[] InputEmbeddingWeight { get; }   // [speech_vocab, hidden]
    public float[] SpkEmbedAffineWeight { get; }    // [mel, spk_embed_dim]
    public float[] SpkEmbedAffineBias { get; }
    public float[] EncoderProjWeight { get; }       // [mel, hidden]
    public float[] EncoderProjBias { get; }
    public float[] AfterNormWeight { get; }
    public float[] AfterNormBias { get; }

    public float[] PlaConv1Weight { get; }  // pre_lookahead_layer.conv1 [hidden, hidden, 4]
    public float[] PlaConv1Bias { get; }
    public float[] PlaConv2Weight { get; }  // pre_lookahead_layer.conv2 [hidden, hidden, 3]
    public float[] PlaConv2Bias { get; }

    public float[] EmbedLinearWeight { get; }  // encoder.embed.out.0 [hidden, hidden]
    public float[] EmbedLinearBias { get; }
    public float[] EmbedLnWeight { get; }      // encoder.embed.out.1
    public float[] EmbedLnBias { get; }

    public float[] UlConvWeight { get; }  // encoder.up_layer.conv [hidden, hidden, 5]
    public float[] UlConvBias { get; }

    public float[] UpEmbedLinearWeight { get; }  // encoder.up_embed.out.0
    public float[] UpEmbedLinearBias { get; }
    public float[] UpEmbedLnWeight { get; }      // encoder.up_embed.out.1
    public float[] UpEmbedLnBias { get; }

    public CosyVoiceFlowLayerWeights[] EncLayers { get; }
    public CosyVoiceFlowLayerWeights[] UpEncLayers { get; }
    IS3GenConformerLayerWeights[] IS3GenFlowEncoderWeights.EncLayers => (IS3GenConformerLayerWeights[])EncLayers;
    IS3GenConformerLayerWeights[] IS3GenFlowEncoderWeights.UpEncLayers => (IS3GenConformerLayerWeights[])UpEncLayers;

    public CosyVoiceFlowWeights(string safetensorsPath)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"CosyVoice flow model file not found: {safetensorsPath}");

        Loader = SafetensorsLoader.Open(safetensorsPath);

        InputEmbeddingWeight = GetTensor("input_embedding.weight");
        int[] embShape = Loader.GetShape("input_embedding.weight");
        SpeechVocabSize = embShape[0];
        HiddenDim = embShape[1];

        SpkEmbedAffineWeight = GetTensor("spk_embed_affine_layer.weight");
        SpkEmbedAffineBias = GetTensor("spk_embed_affine_layer.bias");
        int[] spkShape = Loader.GetShape("spk_embed_affine_layer.weight");
        MelChannels = spkShape[0];
        SpkEmbedDim = spkShape[1];

        EncoderProjWeight = GetTensor("encoder_proj.weight");
        EncoderProjBias = GetTensor("encoder_proj.bias");
        AfterNormWeight = GetTensor("encoder.after_norm.weight");
        AfterNormBias = GetTensor("encoder.after_norm.bias");

        PlaConv1Weight = GetTensor("encoder.pre_lookahead_layer.conv1.weight");
        PlaConv1Bias = GetTensor("encoder.pre_lookahead_layer.conv1.bias");
        PlaConv2Weight = GetTensor("encoder.pre_lookahead_layer.conv2.weight");
        PlaConv2Bias = GetTensor("encoder.pre_lookahead_layer.conv2.bias");

        EmbedLinearWeight = GetTensor("encoder.embed.out.0.weight");
        EmbedLinearBias = GetTensor("encoder.embed.out.0.bias");
        EmbedLnWeight = GetTensor("encoder.embed.out.1.weight");
        EmbedLnBias = GetTensor("encoder.embed.out.1.bias");

        UlConvWeight = GetTensor("encoder.up_layer.conv.weight");
        UlConvBias = GetTensor("encoder.up_layer.conv.bias");

        UpEmbedLinearWeight = GetTensor("encoder.up_embed.out.0.weight");
        UpEmbedLinearBias = GetTensor("encoder.up_embed.out.0.bias");
        UpEmbedLnWeight = GetTensor("encoder.up_embed.out.1.weight");
        UpEmbedLnBias = GetTensor("encoder.up_embed.out.1.bias");

        NumEncoders = CountLayers("encoder.encoders.");
        NumUpEncoders = CountLayers("encoder.up_encoders.");

        EncLayers = new CosyVoiceFlowLayerWeights[NumEncoders];
        for (int i = 0; i < NumEncoders; i++)
            EncLayers[i] = new CosyVoiceFlowLayerWeights(this, $"encoder.encoders.{i}");

        UpEncLayers = new CosyVoiceFlowLayerWeights[NumUpEncoders];
        for (int i = 0; i < NumUpEncoders; i++)
            UpEncLayers[i] = new CosyVoiceFlowLayerWeights(this, $"encoder.up_encoders.{i}");

        int[] qShape = Loader.GetShape("encoder.encoders.0.self_attn.linear_q.weight");
        int[] posBiasShape = Loader.GetShape("encoder.encoders.0.self_attn.pos_bias_u");
        NumHeads = posBiasShape[0];
        HeadDim = posBiasShape[1];
        int[] ffnShape = Loader.GetShape("encoder.encoders.0.feed_forward.w_1.weight");
        FfnDim = ffnShape[0];
    }

    private int CountLayers(string prefix)
    {
        int n = 0;
        while (Loader.Contains($"{prefix}{n}.norm_mha.weight")) n++;
        return n;
    }

    public float[] GetTensor(string name) => Loader.ReadF32(name);

    public void Dispose() => Loader.Dispose();
}

/// <summary>One S3Gen-family Conformer block: rel-pos self-attention (untied u/v biases) + Swish FFN, no macaron, no conv module -- see `Chatterbox/ChatterboxFlowEncoder.cs`'s block-math doc comment for the full derivation this mirrors.</summary>
public sealed class CosyVoiceFlowLayerWeights : IS3GenConformerLayerWeights
{
    public float[] NormMhaWeight { get; }
    public float[] NormMhaBias { get; }
    public float[] QWeight { get; }
    public float[] QBias { get; }
    public float[] KWeight { get; }
    public float[] KBias { get; }
    public float[] VWeight { get; }
    public float[] VBias { get; }
    public float[] OutWeight { get; }
    public float[] OutBias { get; }
    public float[] PosWeight { get; }  // no bias
    public float[] PosBiasU { get; }
    public float[] PosBiasV { get; }

    public float[] NormFfWeight { get; }
    public float[] NormFfBias { get; }
    public float[] Ff1Weight { get; }
    public float[] Ff1Bias { get; }
    public float[] Ff2Weight { get; }
    public float[] Ff2Bias { get; }

    public CosyVoiceFlowLayerWeights(CosyVoiceFlowWeights w, string prefix)
    {
        NormMhaWeight = w.GetTensor($"{prefix}.norm_mha.weight");
        NormMhaBias = w.GetTensor($"{prefix}.norm_mha.bias");
        QWeight = w.GetTensor($"{prefix}.self_attn.linear_q.weight");
        QBias = w.GetTensor($"{prefix}.self_attn.linear_q.bias");
        KWeight = w.GetTensor($"{prefix}.self_attn.linear_k.weight");
        KBias = w.GetTensor($"{prefix}.self_attn.linear_k.bias");
        VWeight = w.GetTensor($"{prefix}.self_attn.linear_v.weight");
        VBias = w.GetTensor($"{prefix}.self_attn.linear_v.bias");
        OutWeight = w.GetTensor($"{prefix}.self_attn.linear_out.weight");
        OutBias = w.GetTensor($"{prefix}.self_attn.linear_out.bias");
        PosWeight = w.GetTensor($"{prefix}.self_attn.linear_pos.weight");
        PosBiasU = w.GetTensor($"{prefix}.self_attn.pos_bias_u");
        PosBiasV = w.GetTensor($"{prefix}.self_attn.pos_bias_v");

        NormFfWeight = w.GetTensor($"{prefix}.norm_ff.weight");
        NormFfBias = w.GetTensor($"{prefix}.norm_ff.bias");
        Ff1Weight = w.GetTensor($"{prefix}.feed_forward.w_1.weight");
        Ff1Bias = w.GetTensor($"{prefix}.feed_forward.w_1.bias");
        Ff2Weight = w.GetTensor($"{prefix}.feed_forward.w_2.weight");
        Ff2Bias = w.GetTensor($"{prefix}.feed_forward.w_2.bias");
    }
}
