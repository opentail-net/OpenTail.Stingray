using System;
using System.IO;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Chatterbox;

/// <summary>
/// S3Gen weights (chatterbox-turbo-s3gen-q4_k.gguf): the CosyVoice-derived token-to-mel flow
/// pipeline (examples/chatterbox-tts-py/chatterbox/models/s3gen/{s3gen,flow}.py). This loader
/// currently covers stage 1, the "flow encoder" (`s3.fe.*` / `s3.flow.*` tensors) --
/// UpsampleConformerEncoder (transformer/upsample_encoder.py) plus the token embedding, speaker
/// x-vector projection, and encoder_proj that produce the CFM decoder's `mu`/`spks` conditioning.
/// The CFM UNet (`s3.fd.*`) and HiFTGenerator vocoder (`s3.v.*`) are separate, not-yet-loaded
/// stages -- see ChatterboxDecoder.cs for the overall status.
/// </summary>
public sealed class ChatterboxS3GenWeights : IDisposable, IS3GenFlowEncoderWeights, IHiFTVocoderWeights
{
    // Explicit IS3GenFlowEncoderWeights implementation: aliases this class's existing public
    // property names (EncHidden/EncHeads/EncHeadDim/EncFfn/SpkEncDim -- kept as-is, no public
    // API to preserve here per se, but renaming would touch every other Chatterbox call site
    // referencing these names, e.g. ChatterboxCfmDecoder.cs/ChatterboxVocoder.cs) onto the
    // shared interface's generic names, added when `S3GenConformerKernels` was extracted so
    // this class's Conformer-encoder logic could move there -- see docs/audio-review-
    // progress.md's CosyVoice section for the extraction rationale.
    int IS3GenFlowEncoderWeights.HiddenDim => EncHidden;
    int IS3GenFlowEncoderWeights.NumHeads => EncHeads;
    int IS3GenFlowEncoderWeights.HeadDim => EncHeadDim;
    int IS3GenFlowEncoderWeights.FfnDim => EncFfn;
    IS3GenConformerLayerWeights[] IS3GenFlowEncoderWeights.EncLayers => (IS3GenConformerLayerWeights[])EncLayers;
    IS3GenConformerLayerWeights[] IS3GenFlowEncoderWeights.UpEncLayers => (IS3GenConformerLayerWeights[])UpEncLayers;

    // Explicit IHiFTVocoderWeights implementation: aliases this class's existing Voc*/Vocoder.*
    // names (added when `HiFTVocoderKernels` was extracted so this class's vocoder DSP logic
    // could move there) -- same rationale as the IS3GenFlowEncoderWeights block above. Both
    // conv_pre/conv_post kernels are hardcoded 7 for this checkpoint (verified from the
    // original ChatterboxVocoder.cs `Decode`'s literal `kernel: 7` arguments before this
    // extraction, not re-derived).
    int[] IHiFTVocoderWeights.UpsampleRates => VocUpsampleRates;
    int[] IHiFTVocoderWeights.UpsampleKernels => VocUpsampleKernels;
    int[] IHiFTVocoderWeights.ResblockKernels => VocResblockKernels;
    int[] IHiFTVocoderWeights.SourceResblockKernels => VocSourceResblockKernels;
    int IHiFTVocoderWeights.BaseChannels => VocBaseChannels;
    int IHiFTVocoderWeights.NbHarmonics => VocNbHarmonics;
    int IHiFTVocoderWeights.IstftNFft => VocIstftNFft;
    int IHiFTVocoderWeights.IstftHopLen => VocIstftHopLen;
    int IHiFTVocoderWeights.SampleRate => SampleRate;
    int IHiFTVocoderWeights.ConvPreKernel => 7;
    int IHiFTVocoderWeights.ConvPostKernel => 7;
    float[] IHiFTVocoderWeights.ConvPreWeight => Vocoder.ConvPreWeight;
    float[] IHiFTVocoderWeights.ConvPreBias => Vocoder.ConvPreBias;
    float[] IHiFTVocoderWeights.ConvPostWeight => Vocoder.ConvPostWeight;
    float[] IHiFTVocoderWeights.ConvPostBias => Vocoder.ConvPostBias;
    float[][] IHiFTVocoderWeights.UpWeight => Vocoder.UpWeight;
    float[][] IHiFTVocoderWeights.UpBias => Vocoder.UpBias;
    float[][] IHiFTVocoderWeights.SourceDownWeight => Vocoder.SourceDownWeight;
    float[][] IHiFTVocoderWeights.SourceDownBias => Vocoder.SourceDownBias;
    IHifiResBlockWeights[] IHiFTVocoderWeights.SourceResBlocks => (IHifiResBlockWeights[])Vocoder.SourceResBlocks;
    IHifiResBlockWeights[] IHiFTVocoderWeights.ResBlocks => (IHifiResBlockWeights[])Vocoder.ResBlocks;
    IF0PredictorWeights IHiFTVocoderWeights.F0Predictor => F0Predictor;
    float[] IHiFTVocoderWeights.MSourceLinearWeight => Vocoder.MSourceLinearWeight;
    float[] IHiFTVocoderWeights.MSourceLinearBias => Vocoder.MSourceLinearBias;

    public GgufModel Model { get; }

    public int EncHidden { get; }
    public int EncHeads { get; }
    public int EncHeadDim { get; }
    public int EncFfn { get; }
    public int EncNumLayers { get; }
    public int EncUpNumLayers { get; }
    public int MelChannels { get; }
    public int SpkEncDim { get; }
    public int S3VocabSize { get; }

    // --- s3.flow.* : token embedding, speaker projection, encoder->mel projection ---
    public float[] InputEmbeddingWeight { get; }   // [S3VocabSize, EncHidden]
    public float[] SpkEmbedAffineWeight { get; }    // [MelChannels, SpkEncDim]
    public float[] SpkEmbedAffineBias { get; }      // [MelChannels]
    public float[] EncoderProjWeight { get; }       // [MelChannels, EncHidden]
    public float[] EncoderProjBias { get; }         // [MelChannels]

    // --- s3.fe.* : UpsampleConformerEncoder ---
    public float[] EmbedLinearWeight { get; }       // [EncHidden, EncHidden]
    public float[] EmbedLinearBias { get; }
    public float[] EmbedLnWeight { get; }
    public float[] EmbedLnBias { get; }

    public float[] PlaConv1Weight { get; }          // [EncHidden, EncHidden, 4] (PreLookaheadLayer)
    public float[] PlaConv1Bias { get; }
    public float[] PlaConv2Weight { get; }          // [EncHidden, EncHidden, 3]
    public float[] PlaConv2Bias { get; }

    public ChatterboxS3GenConformerLayer[] EncLayers { get; }     // 6 blocks, pre-upsample

    public float[] UlConvWeight { get; }            // [EncHidden, EncHidden, 5] (Upsample1D conv)
    public float[] UlConvBias { get; }

    public float[] UpEmbedLinearWeight { get; }     // [EncHidden, EncHidden]
    public float[] UpEmbedLinearBias { get; }
    public float[] UpEmbedLnWeight { get; }
    public float[] UpEmbedLnBias { get; }

    public ChatterboxS3GenConformerLayer[] UpEncLayers { get; }   // 4 blocks, post-upsample

    public float[] AfterNormWeight { get; }         // [EncHidden]
    public float[] AfterNormBias { get; }

    // --- s3.fd.* : ConditionalDecoder (CFM flow-matching UNet, "flow decoder") ---
    public int DecChannels { get; }
    public int DecInChannels { get; }
    public int DecOutChannels { get; }
    public int DecHeadDim { get; }
    public int DecNumHeads { get; }
    public int DecNumBlocks { get; }        // transformer blocks per stage
    public int DecNumMid { get; }           // number of mid stages

    public float[] TimeMlpLinear1Weight { get; }    // [DecChannels*4, DecInChannels]
    public float[] TimeMlpLinear1Bias { get; }
    public float[] TimeMlpLinear2Weight { get; }    // [DecChannels*4, DecChannels*4]
    public float[] TimeMlpLinear2Bias { get; }
    public float[] TimeMixerWeight { get; }         // [DecChannels*4, DecChannels*8], no bias (meanflow t/r mixer)

    public ChatterboxCfmStageWeights DownStage { get; }   // in=DecInChannels, out=DecChannels
    public ChatterboxCfmStageWeights[] MidStages { get; } // in=out=DecChannels
    public ChatterboxCfmStageWeights UpStage { get; }     // in=DecChannels*2 (skip concat), out=DecChannels

    public float[] FinalBlockConvWeight { get; }    // [DecChannels, DecChannels, 3] (causal)
    public float[] FinalBlockConvBias { get; }
    public float[] FinalBlockLnWeight { get; }
    public float[] FinalBlockLnBias { get; }
    public float[] FinalProjWeight { get; }         // [DecOutChannels, DecChannels, 1]
    public float[] FinalProjBias { get; }

    // --- s3.v.* / s3.v.f0.* : HiFTGenerator vocoder (mel -> waveform) ---
    public int[] VocUpsampleRates { get; }
    public int[] VocUpsampleKernels { get; }
    public int[] VocResblockKernels { get; }         // [3, 7, 11]
    public int[] VocSourceResblockKernels { get; }    // [7, 7, 11], one per upsample stage
    public int VocBaseChannels { get; }
    public int VocNbHarmonics { get; }
    public int VocIstftNFft { get; }
    public int VocIstftHopLen { get; }
    public int SampleRate { get; }

    public ChatterboxF0PredictorWeights F0Predictor { get; }
    public ChatterboxVocoderWeights Vocoder { get; }

    public ChatterboxS3GenWeights(string s3GenPath)
    {
        if (!File.Exists(s3GenPath))
            throw new FileNotFoundException($"Chatterbox S3Gen model file not found: {s3GenPath}");

        Model = GgufModel.Open(s3GenPath);

        EncHidden = GetInt("chatterbox.s3gen.enc_hidden", 512);
        EncHeads = GetInt("chatterbox.s3gen.enc_heads", 8);
        EncHeadDim = GetInt("chatterbox.s3gen.enc_head_dim", EncHidden / EncHeads);
        EncFfn = GetInt("chatterbox.s3gen.enc_ffn", 2048);
        EncNumLayers = GetInt("chatterbox.s3gen.enc_n_layers", 6);
        EncUpNumLayers = GetInt("chatterbox.s3gen.enc_up_n_layers", 4);
        MelChannels = GetInt("chatterbox.s3gen.mel_channels", 80);
        SpkEncDim = GetInt("chatterbox.s3gen.spk_enc_dim", 192);
        S3VocabSize = GetInt("chatterbox.s3gen.s3_vocab_size", 6561);

        InputEmbeddingWeight = GetTensor("s3.flow.input_embedding.weight");
        SpkEmbedAffineWeight = GetTensor("s3.flow.spk_embed_affine_layer.weight");
        SpkEmbedAffineBias = GetTensor("s3.flow.spk_embed_affine_layer.bias");
        EncoderProjWeight = GetTensor("s3.flow.encoder_proj.weight");
        EncoderProjBias = GetTensor("s3.flow.encoder_proj.bias");

        EmbedLinearWeight = GetTensor("s3.fe.embed.out.0.weight");
        EmbedLinearBias = GetTensor("s3.fe.embed.out.0.bias");
        EmbedLnWeight = GetTensor("s3.fe.embed.out.1.weight");
        EmbedLnBias = GetTensor("s3.fe.embed.out.1.bias");

        PlaConv1Weight = GetTensor("s3.fe.pla.conv1.weight");
        PlaConv1Bias = GetTensor("s3.fe.pla.conv1.bias");
        PlaConv2Weight = GetTensor("s3.fe.pla.conv2.weight");
        PlaConv2Bias = GetTensor("s3.fe.pla.conv2.bias");

        EncLayers = new ChatterboxS3GenConformerLayer[EncNumLayers];
        for (int i = 0; i < EncNumLayers; i++)
            EncLayers[i] = new ChatterboxS3GenConformerLayer(this, $"s3.fe.enc.{i}");

        UlConvWeight = GetTensor("s3.fe.ul.conv.weight");
        UlConvBias = GetTensor("s3.fe.ul.conv.bias");

        UpEmbedLinearWeight = GetTensor("s3.fe.uemb.out.0.weight");
        UpEmbedLinearBias = GetTensor("s3.fe.uemb.out.0.bias");
        UpEmbedLnWeight = GetTensor("s3.fe.uemb.out.1.weight");
        UpEmbedLnBias = GetTensor("s3.fe.uemb.out.1.bias");

        UpEncLayers = new ChatterboxS3GenConformerLayer[EncUpNumLayers];
        for (int i = 0; i < EncUpNumLayers; i++)
            UpEncLayers[i] = new ChatterboxS3GenConformerLayer(this, $"s3.fe.ue.{i}");

        AfterNormWeight = GetTensor("s3.fe.an.weight");
        AfterNormBias = GetTensor("s3.fe.an.bias");

        DecChannels = GetInt("chatterbox.s3gen.dec_channels", 256);
        DecInChannels = GetInt("chatterbox.s3gen.dec_in_channels", 320);
        DecOutChannels = GetInt("chatterbox.s3gen.dec_out_channels", 80);
        DecHeadDim = GetInt("chatterbox.s3gen.dec_head_dim", 64);
        DecNumHeads = GetInt("chatterbox.s3gen.dec_n_heads", 8);
        DecNumBlocks = GetInt("chatterbox.s3gen.dec_n_blocks", 4);
        DecNumMid = GetInt("chatterbox.s3gen.dec_n_mid", 12);

        TimeMlpLinear1Weight = GetTensor("s3.fd.tm.linear_1.weight");
        TimeMlpLinear1Bias = GetTensor("s3.fd.tm.linear_1.bias");
        TimeMlpLinear2Weight = GetTensor("s3.fd.tm.linear_2.weight");
        TimeMlpLinear2Bias = GetTensor("s3.fd.tm.linear_2.bias");
        TimeMixerWeight = GetTensor("s3.fd.tmx.weight");

        DownStage = new ChatterboxCfmStageWeights(this, "s3.fd.db.0", DecNumBlocks);
        MidStages = new ChatterboxCfmStageWeights[DecNumMid];
        for (int i = 0; i < DecNumMid; i++)
            MidStages[i] = new ChatterboxCfmStageWeights(this, $"s3.fd.mb.{i}", DecNumBlocks, hasResample: false);
        UpStage = new ChatterboxCfmStageWeights(this, "s3.fd.ub.0", DecNumBlocks);

        FinalBlockConvWeight = GetTensor("s3.fd.fb.block.0.weight");
        FinalBlockConvBias = GetTensor("s3.fd.fb.block.0.bias");
        FinalBlockLnWeight = GetTensor("s3.fd.fb.block.2.weight");
        FinalBlockLnBias = GetTensor("s3.fd.fb.block.2.bias");
        FinalProjWeight = GetTensor("s3.fd.fp.weight");
        FinalProjBias = GetTensor("s3.fd.fp.bias");

        VocUpsampleRates = GetIntArray("chatterbox.s3gen.voc_upsample_rates", [8, 5, 3]);
        VocUpsampleKernels = GetIntArray("chatterbox.s3gen.voc_upsample_kernels", [16, 11, 7]);
        VocResblockKernels = GetIntArray("chatterbox.s3gen.voc_resblock_kernels", [3, 7, 11]);
        VocSourceResblockKernels = GetIntArray("chatterbox.s3gen.voc_source_resblock_kernels", [7, 7, 11]);
        VocBaseChannels = GetInt("chatterbox.s3gen.voc_base_channels", 512);
        VocNbHarmonics = GetInt("chatterbox.s3gen.voc_nb_harmonics", 8);
        VocIstftNFft = GetInt("chatterbox.s3gen.voc_istft_n_fft", 16);
        VocIstftHopLen = GetInt("chatterbox.s3gen.voc_istft_hop_len", 4);
        SampleRate = GetInt("chatterbox.s3gen.sample_rate", 24000);

        F0Predictor = new ChatterboxF0PredictorWeights(this);
        Vocoder = new ChatterboxVocoderWeights(this);
    }

    private int[] GetIntArray(string key, int[] fallback)
    {
        if (!Model.Metadata.TryGetValue(key, out var v)) return fallback;

        // Some converters store small int lists as a native GGUF array; this one stores them as
        // a comma-separated string (confirmed via `list-metadata`, which prints e.g. "3,7,11"
        // with no "(array: N items)" annotation, unlike the tokenizer's real array fields).
        if (v is string s)
        {
            var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            return parts.Length > 0 ? Array.ConvertAll(parts, int.Parse) : fallback;
        }
        if (v is System.Collections.IEnumerable e)
        {
            var list = new System.Collections.Generic.List<int>();
            foreach (var item in e) list.Add(Convert.ToInt32(item));
            return list.Count > 0 ? list.ToArray() : fallback;
        }
        return [Convert.ToInt32(v)];
    }

    private int GetInt(string key, int fallback) =>
        Model.Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    public float[] GetTensor(string name)
    {
        var info = Model.FindTensor(name) ?? throw new InvalidDataException($"Chatterbox S3Gen GGUF missing required tensor '{name}'.");
        return DequantTensor(Model, info);
    }

    public float[]? TryGetTensor(string name)
    {
        var info = Model.FindTensor(name);
        return info is null ? null : DequantTensor(Model, info.Value);
    }

    private static float[] DequantTensor(GgufModel model, GgufTensorInfo info)
    {
        var bytes = model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(bytes, dst, info.DType, info.ElementCount);
        return dst;
    }

    public void Dispose() => Model.Dispose();
}

/// <summary>
/// One ConformerEncoderLayer's weights, with macaron_style=False and use_cnn_module=False (per
/// UpsampleConformerEncoder's config in s3gen.py) -- so this is just rel-pos self-attention +
/// FFN, both pre-LN, no macaron FFN and no depthwise conv branch.
/// </summary>
public sealed class ChatterboxS3GenConformerLayer : IS3GenConformerLayerWeights
{
    public float[] NormMhaWeight { get; }
    public float[] NormMhaBias { get; }
    public float[] QWeight { get; }    // sa.lq [EncHidden, EncHidden]
    public float[] QBias { get; }
    public float[] KWeight { get; }    // sa.lk
    public float[] KBias { get; }
    public float[] VWeight { get; }    // sa.lv
    public float[] VBias { get; }
    public float[] OutWeight { get; }  // sa.lo
    public float[] OutBias { get; }
    public float[] PosWeight { get; }  // sa.lp, no bias
    public float[] PosBiasU { get; }   // sa.pbu [EncHeadDim, EncHeads] ggml -> torch [EncHeads, EncHeadDim]
    public float[] PosBiasV { get; }   // sa.pbv

    public float[] NormFfWeight { get; }
    public float[] NormFfBias { get; }
    public float[] Ff1Weight { get; }  // ff.w_1 [EncHidden, EncFfn]
    public float[] Ff1Bias { get; }
    public float[] Ff2Weight { get; }  // ff.w_2 [EncFfn, EncHidden]
    public float[] Ff2Bias { get; }

    public ChatterboxS3GenConformerLayer(ChatterboxS3GenWeights w, string prefix)
    {
        NormMhaWeight = w.GetTensor($"{prefix}.nmha.weight");
        NormMhaBias = w.GetTensor($"{prefix}.nmha.bias");
        QWeight = w.GetTensor($"{prefix}.sa.lq.weight");
        QBias = w.GetTensor($"{prefix}.sa.lq.bias");
        KWeight = w.GetTensor($"{prefix}.sa.lk.weight");
        KBias = w.GetTensor($"{prefix}.sa.lk.bias");
        VWeight = w.GetTensor($"{prefix}.sa.lv.weight");
        VBias = w.GetTensor($"{prefix}.sa.lv.bias");
        OutWeight = w.GetTensor($"{prefix}.sa.lo.weight");
        OutBias = w.GetTensor($"{prefix}.sa.lo.bias");
        PosWeight = w.GetTensor($"{prefix}.sa.lp.weight");
        PosBiasU = w.GetTensor($"{prefix}.sa.pbu");
        PosBiasV = w.GetTensor($"{prefix}.sa.pbv");

        NormFfWeight = w.GetTensor($"{prefix}.nff.weight");
        NormFfBias = w.GetTensor($"{prefix}.nff.bias");
        Ff1Weight = w.GetTensor($"{prefix}.ff.w_1.weight");
        Ff1Bias = w.GetTensor($"{prefix}.ff.w_1.bias");
        Ff2Weight = w.GetTensor($"{prefix}.ff.w_2.weight");
        Ff2Bias = w.GetTensor($"{prefix}.ff.w_2.bias");
    }
}

/// <summary>
/// One ConditionalDecoder stage (decoder.py): a CausalResnetBlock1D (with FiLM-style additive
/// time-embedding conditioning) followed by <see cref="ChatterboxS3GenWeights.DecNumBlocks"/>
/// BasicTransformerBlocks (self-attention only -- cross_attention_dim is never set in
/// decoder.py's construction, so attn2/norm2 don't exist), then a resample conv. Since
/// ConditionalDecoder is built with channels=[256] (a single-element list) both the "down" and
/// "up" resample convs are the is_last case, i.e. plain CausalConv1d(k=3) with no real
/// upsampling/downsampling -- see decoder.py's down_blocks/up_blocks construction.
/// </summary>
public sealed class ChatterboxCfmStageWeights : IUnetStageWeights
{
    public ChatterboxCfmResnetWeights Resnet { get; }
    IResnetBlockWeights IUnetStageWeights.Resnet => Resnet;
    public ChatterboxCfmTransformerBlockWeights[] TransformerBlocks { get; }
    IUnetTransformerBlockWeights[] IUnetStageWeights.TransformerBlocks => (IUnetTransformerBlockWeights[])TransformerBlocks;

    /// <summary>
    /// Resample conv, present only for the down/up stages (decoder.py's down_blocks/up_blocks
    /// each append a 3rd ModuleList element; mid_blocks is just [resnet, transformer_blocks],
    /// no resample). Null for mid stages.
    /// </summary>
    public float[]? ResampleConvWeight { get; }  // [outCh, outCh, 3] causal
    public float[]? ResampleConvBias { get; }

    public ChatterboxCfmStageWeights(ChatterboxS3GenWeights w, string prefix, int numBlocks, bool hasResample = true)
    {
        Resnet = new ChatterboxCfmResnetWeights(w, $"{prefix}.0");
        TransformerBlocks = new ChatterboxCfmTransformerBlockWeights[numBlocks];
        for (int i = 0; i < numBlocks; i++)
            TransformerBlocks[i] = new ChatterboxCfmTransformerBlockWeights(w, $"{prefix}.1.{i}");
        if (hasResample)
        {
            ResampleConvWeight = w.GetTensor($"{prefix}.2.weight");
            ResampleConvBias = w.GetTensor($"{prefix}.2.bias");
        }
    }
}

/// <summary>
/// CausalResnetBlock1D (matcha/decoder.py): block1 (CausalConv1d k=3 + LayerNorm + Mish) ->
/// += mlp(time_emb) broadcast over time -> block2 (same as block1) -> + res_conv(x) (1x1 conv,
/// always applied, even when dim==dim_out).
/// </summary>
public sealed class ChatterboxCfmResnetWeights : IResnetBlockWeights
{
    public float[] Block1ConvWeight { get; }  // [dimOut, dimIn, 3]
    public float[] Block1ConvBias { get; }
    public float[] Block1LnWeight { get; }
    public float[] Block1LnBias { get; }
    public float[] Block2ConvWeight { get; }  // [dimOut, dimOut, 3]
    public float[] Block2ConvBias { get; }
    public float[] Block2LnWeight { get; }
    public float[] Block2LnBias { get; }
    public CfmLinearWeight MlpWeight { get; }         // [dimOut, timeEmbedDim]
    public float[] MlpBias { get; }
    public float[] ResConvWeight { get; }     // [dimOut, dimIn, 1]
    public float[] ResConvBias { get; }

    public ChatterboxCfmResnetWeights(ChatterboxS3GenWeights w, string prefix)
    {
        Block1ConvWeight = w.GetTensor($"{prefix}.b1.0.weight");
        Block1ConvBias = w.GetTensor($"{prefix}.b1.0.bias");
        Block1LnWeight = w.GetTensor($"{prefix}.b1.2.weight");
        Block1LnBias = w.GetTensor($"{prefix}.b1.2.bias");
        Block2ConvWeight = w.GetTensor($"{prefix}.b2.0.weight");
        Block2ConvBias = w.GetTensor($"{prefix}.b2.0.bias");
        Block2LnWeight = w.GetTensor($"{prefix}.b2.2.weight");
        Block2LnBias = w.GetTensor($"{prefix}.b2.2.bias");
        var mlpWeightF32 = w.GetTensor($"{prefix}.mlp.1.weight");
        MlpBias = w.GetTensor($"{prefix}.mlp.1.bias");
        MlpWeight = CfmLinearWeight.FromF32(mlpWeightF32, outDim: MlpBias.Length, inDim: mlpWeightF32.Length / MlpBias.Length);
        ResConvWeight = w.GetTensor($"{prefix}.rc.weight");
        ResConvBias = w.GetTensor($"{prefix}.rc.bias");
    }
}

/// <summary>
/// BasicTransformerBlock (matcha/transformer.py), self-attention only (no cross-attention: this
/// decoder never sets cross_attention_dim): x = x + attn1(LayerNorm(x)); x = x + FF(LayerNorm(x)).
/// attn1's q/k/v projections have no bias (diffusers Attention default); the output projection
/// does. FeedForward uses activation_fn="gelu" (decoder.py's act_fn param) -- a plain (non-gated)
/// Linear-GELU-Linear MLP, not the diffusers-default GEGLU (confirmed by a single up/down weight
/// pair per block, not the two up-projections GEGLU would need).
/// </summary>
public sealed class ChatterboxCfmTransformerBlockWeights : IUnetTransformerBlockWeights
{
    public float[] Norm1Weight { get; }
    public float[] Norm1Bias { get; }
    public CfmLinearWeight QWeight { get; }    // attn1.q [dim, heads*headDim], no bias
    public CfmLinearWeight KWeight { get; }    // attn1.k, no bias
    public CfmLinearWeight VWeight { get; }    // attn1.v, no bias
    public CfmLinearWeight OutWeight { get; }  // attn1.o [heads*headDim, dim]
    public float[] OutBias { get; }

    public float[] Norm3Weight { get; }
    public float[] Norm3Bias { get; }
    public CfmLinearWeight FfUpWeight { get; }    // [dim*4, dim]
    public float[] FfUpBias { get; }
    public CfmLinearWeight FfDownWeight { get; }  // [dim, dim*4]
    public float[] FfDownBias { get; }

    public ChatterboxCfmTransformerBlockWeights(ChatterboxS3GenWeights w, string prefix)
    {
        Norm1Weight = w.GetTensor($"{prefix}.norm1.weight");
        Norm1Bias = w.GetTensor($"{prefix}.norm1.bias");
        var qF32 = w.GetTensor($"{prefix}.attn1.q.weight");
        var kF32 = w.GetTensor($"{prefix}.attn1.k.weight");
        var vF32 = w.GetTensor($"{prefix}.attn1.v.weight");
        var outF32 = w.GetTensor($"{prefix}.attn1.o.weight");
        OutBias = w.GetTensor($"{prefix}.attn1.o.bias");
        int dim = OutBias.Length;
        QWeight = CfmLinearWeight.FromF32(qF32, outDim: qF32.Length / dim, inDim: dim);
        KWeight = CfmLinearWeight.FromF32(kF32, outDim: kF32.Length / dim, inDim: dim);
        VWeight = CfmLinearWeight.FromF32(vF32, outDim: vF32.Length / dim, inDim: dim);
        OutWeight = CfmLinearWeight.FromF32(outF32, outDim: dim, inDim: outF32.Length / dim);

        Norm3Weight = w.GetTensor($"{prefix}.norm3.weight");
        Norm3Bias = w.GetTensor($"{prefix}.norm3.bias");
        var ffUpF32 = w.GetTensor($"{prefix}.ff.up.weight");
        FfUpBias = w.GetTensor($"{prefix}.ff.up.bias");
        FfUpWeight = CfmLinearWeight.FromF32(ffUpF32, outDim: FfUpBias.Length, inDim: ffUpF32.Length / FfUpBias.Length);
        var ffDownF32 = w.GetTensor($"{prefix}.ff.down.weight");
        FfDownBias = w.GetTensor($"{prefix}.ff.down.bias");
        FfDownWeight = CfmLinearWeight.FromF32(ffDownF32, outDim: FfDownBias.Length, inDim: ffDownF32.Length / FfDownBias.Length);
    }
}

/// <summary>ConvRNNF0Predictor (f0_predictor.py): 5x (Conv1d k=3 pad=1 + ELU), then Linear(512,1) + abs().</summary>
public sealed class ChatterboxF0PredictorWeights : IF0PredictorWeights
{
    public float[][] ConvWeight { get; } = new float[5][];  // conv 0: [512,80,3]; convs 1-4: [512,512,3]
    public float[][] ConvBias { get; } = new float[5][];
    public float[] ClassifierWeight { get; }  // [1, 512]
    public float[] ClassifierBias { get; }    // [1]

    public ChatterboxF0PredictorWeights(ChatterboxS3GenWeights w)
    {
        int[] idx = [0, 2, 4, 6, 8]; // nn.Sequential slot indices (odd slots are ELU, no params)
        for (int i = 0; i < 5; i++)
        {
            ConvWeight[i] = w.GetTensor($"s3.v.f0.cn.{idx[i]}.weight");
            ConvBias[i] = w.GetTensor($"s3.v.f0.cn.{idx[i]}.bias");
        }
        ClassifierWeight = w.GetTensor("s3.v.f0.cls.weight");
        ClassifierBias = w.GetTensor("s3.v.f0.cls.bias");
    }
}

/// <summary>
/// HiFTGenerator (hifigan.py): conv_pre -> per-upsample-stage [LeakyReLU, ConvTranspose1d,
/// (last stage: ReflectionPad), += source_resblocks[i](source_downs[i](sourceSpectrum)),
/// average of 3 Snake-activated HiFiGAN resblocks (kernel sizes from VocResblockKernels)] ->
/// LeakyReLU -> conv_post -> exp(mag)/sin(phase) -> learned inverse-STFT.
/// </summary>
public sealed class ChatterboxVocoderWeights
{
    public float[] ConvPreWeight { get; }   // [512, 80, 7]
    public float[] ConvPreBias { get; }
    public float[] ConvPostWeight { get; }  // [nfft+2, lastCh, 7]
    public float[] ConvPostBias { get; }

    public float[][] UpWeight { get; }      // ConvTranspose1d per stage: [chIn, chOut, kernel]
    public float[][] UpBias { get; }

    public float[][] SourceDownWeight { get; }  // Conv1d(nfft+2, chOut, ...) per stage
    public float[][] SourceDownBias { get; }
    public ChatterboxHifiResBlockWeights[] SourceResBlocks { get; }  // one per stage

    public ChatterboxHifiResBlockWeights[] ResBlocks { get; }  // numStages * numKernels (rb.0..rb.{n-1})

    // m_source.l_linear: merges (nb_harmonics+1) sine harmonics into a single excitation.
    public float[] MSourceLinearWeight { get; }  // [1, nb_harmonics+1]
    public float[] MSourceLinearBias { get; }    // [1]

    public ChatterboxVocoderWeights(ChatterboxS3GenWeights w)
    {
        ConvPreWeight = w.GetTensor("s3.v.cpre.weight");
        ConvPreBias = w.GetTensor("s3.v.cpre.bias");
        ConvPostWeight = w.GetTensor("s3.v.cpost.weight");
        ConvPostBias = w.GetTensor("s3.v.cpost.bias");

        int numStages = w.VocUpsampleRates.Length;
        UpWeight = new float[numStages][];
        UpBias = new float[numStages][];
        SourceDownWeight = new float[numStages][];
        SourceDownBias = new float[numStages][];
        SourceResBlocks = new ChatterboxHifiResBlockWeights[numStages];
        for (int i = 0; i < numStages; i++)
        {
            UpWeight[i] = w.GetTensor($"s3.v.ups.{i}.weight");
            UpBias[i] = w.GetTensor($"s3.v.ups.{i}.bias");
            SourceDownWeight[i] = w.GetTensor($"s3.v.sd.{i}.weight");
            SourceDownBias[i] = w.GetTensor($"s3.v.sd.{i}.bias");
            SourceResBlocks[i] = new ChatterboxHifiResBlockWeights(w, $"s3.v.srb.{i}");
        }

        int numKernels = w.VocResblockKernels.Length;
        ResBlocks = new ChatterboxHifiResBlockWeights[numStages * numKernels];
        for (int i = 0; i < ResBlocks.Length; i++)
            ResBlocks[i] = new ChatterboxHifiResBlockWeights(w, $"s3.v.rb.{i}");

        MSourceLinearWeight = w.GetTensor("s3.v.ms.ll.weight");
        MSourceLinearBias = w.GetTensor("s3.v.ms.ll.bias");
    }
}

/// <summary>
/// HiFiGAN/BigVGAN ResBlock (hifigan.py): 3 dilated conv pairs (dilations [1,3,5]), each preceded
/// by a per-channel learned Snake activation (x + (1/alpha)*sin(alpha*x)^2) -- same structural
/// shape as Kokoro's AdaINResBlock1 (see KokoroDecoder.cs), but WITHOUT AdaIN style-conditioning
/// (Snake only, no per-block speaker modulation).
/// </summary>
public sealed class ChatterboxHifiResBlockWeights : IHifiResBlockWeights
{
    public float[][] Convs1Weight { get; } = new float[3][];
    public float[][] Convs1Bias { get; } = new float[3][];
    public float[][] Convs2Weight { get; } = new float[3][];
    public float[][] Convs2Bias { get; } = new float[3][];
    public float[][] Alpha1 { get; } = new float[3][];
    public float[][] Alpha2 { get; } = new float[3][];

    public ChatterboxHifiResBlockWeights(ChatterboxS3GenWeights w, string prefix)
    {
        for (int i = 0; i < 3; i++)
        {
            Convs1Weight[i] = w.GetTensor($"{prefix}.c1.{i}.weight");
            Convs1Bias[i] = w.GetTensor($"{prefix}.c1.{i}.bias");
            Convs2Weight[i] = w.GetTensor($"{prefix}.c2.{i}.weight");
            Convs2Bias[i] = w.GetTensor($"{prefix}.c2.{i}.bias");
            Alpha1[i] = w.GetTensor($"{prefix}.a1.{i}.alpha");
            Alpha2[i] = w.GetTensor($"{prefix}.a2.{i}.alpha");
        }
    }
}
