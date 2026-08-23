using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice2's real flow-matching estimator (`decoder.estimator.*` in
/// `models/cosyvoice2_flow.safetensors`) -- the real Python class is
/// `cosyvoice.flow.decoder.CausalConditionalDecoder`, confirmed (not guessed) via:
///   1. The real HF `cosyvoice2.yaml` config (fetched via `gh api .../conf/cosyvoice2.yaml`):
///      `in_channels=320, out_channels=80, channels=[256], attention_head_dim=64, n_blocks=4,
///      num_mid_blocks=12, num_heads=8`.
///   2. The real, actual local checkpoint's tensor names/shapes (dumped via
///      `safetensors.safe_open`), which match #1 exactly: single down-stage (320-&gt;256, is_last
///      so its resample is a plain `CausalConv1d(256,256,3)`, not `Downsample1D` -- decoder.py's
///      `CausalConditionalDecoder.__init__`, `channels=(256,)` means every stage is `is_last`),
///      12 mid-stages (256-&gt;256, no resample), single up-stage (512-&gt;256 [skip concat], same
///      is_last CausalConv1d resample instead of `Upsample1D`), `final_block`+`final_proj`.
///   3. `ff.net.0.proj`/`ff.net.2` have NO `alpha`/`beta` snake params in the real checkpoint
///      (checked directly), so despite `ConditionalDecoder`'s default `act_fn="snake"`, this
///      checkpoint's FeedForward is the plain GELU MLP -- architecturally identical in shape to
///      Chatterbox's (see <see cref="CfmUNetKernels"/>'s shared `TransformerBlock`), so the
///      shared kernel (already GELU-based) applies as-is with no new activation needed.
///
/// The UNet body is run via the same <see cref="CfmUNetKernels.RunEstimator"/> used by
/// Chatterbox's (golden-verified) `ConditionalDecoder` -- see that kernel's doc comment for the
/// full architectural cross-check. The ONE real difference from Chatterbox is the time
/// embedding: this checkpoint has no meanflow `time_embed_mixer` (`s3.fd.tmx` in Chatterbox) --
/// only `time_mlp.linear_1/2`, standard single-timestep flow matching -- see
/// <see cref="CosyVoiceCfmDecoder"/>.
/// </summary>
public sealed class CosyVoiceCfmDecoderWeights
{
    public const int InChannels = 320;
    public const int Channels = 256;
    public const int OutChannels = 80;
    public const int NumHeads = 8;
    public const int HeadDim = 64;
    public const int NumBlocksPerStage = 4;
    public const int NumMidStages = 12;

    public float[] TimeMlpLinear1Weight { get; }  // [1024, 320]
    public float[] TimeMlpLinear1Bias { get; }
    public float[] TimeMlpLinear2Weight { get; }  // [1024, 1024]
    public float[] TimeMlpLinear2Bias { get; }

    public CosyVoiceCfmStageWeights DownStage { get; }
    public CosyVoiceCfmStageWeights[] MidStages { get; }
    public CosyVoiceCfmStageWeights UpStage { get; }

    public float[] FinalBlockConvWeight { get; }
    public float[] FinalBlockConvBias { get; }
    public float[] FinalBlockLnWeight { get; }
    public float[] FinalBlockLnBias { get; }
    public float[] FinalProjWeight { get; }
    public float[] FinalProjBias { get; }

    public CosyVoiceCfmDecoderWeights(CosyVoiceFlowWeights flow)
    {
        TimeMlpLinear1Weight = flow.GetTensor("decoder.estimator.time_mlp.linear_1.weight");
        TimeMlpLinear1Bias = flow.GetTensor("decoder.estimator.time_mlp.linear_1.bias");
        TimeMlpLinear2Weight = flow.GetTensor("decoder.estimator.time_mlp.linear_2.weight");
        TimeMlpLinear2Bias = flow.GetTensor("decoder.estimator.time_mlp.linear_2.bias");

        DownStage = new CosyVoiceCfmStageWeights(flow, "decoder.estimator.down_blocks.0", NumBlocksPerStage, hasResample: true);
        MidStages = new CosyVoiceCfmStageWeights[NumMidStages];
        for (int i = 0; i < NumMidStages; i++)
            MidStages[i] = new CosyVoiceCfmStageWeights(flow, $"decoder.estimator.mid_blocks.{i}", NumBlocksPerStage, hasResample: false);
        UpStage = new CosyVoiceCfmStageWeights(flow, "decoder.estimator.up_blocks.0", NumBlocksPerStage, hasResample: true);

        FinalBlockConvWeight = flow.GetTensor("decoder.estimator.final_block.block.0.weight");
        FinalBlockConvBias = flow.GetTensor("decoder.estimator.final_block.block.0.bias");
        FinalBlockLnWeight = flow.GetTensor("decoder.estimator.final_block.block.2.weight");
        FinalBlockLnBias = flow.GetTensor("decoder.estimator.final_block.block.2.bias");
        FinalProjWeight = flow.GetTensor("decoder.estimator.final_proj.weight");
        FinalProjBias = flow.GetTensor("decoder.estimator.final_proj.bias");
    }
}

/// <summary>One down/mid/up stage: CausalResnetBlock1D + N BasicTransformerBlocks + (down/up only) a plain CausalConv1d(k=3) resample (this checkpoint's single stage is always `is_last`).</summary>
public sealed class CosyVoiceCfmStageWeights : IUnetStageWeights
{
    public CosyVoiceCfmResnetWeights Resnet { get; }
    IResnetBlockWeights IUnetStageWeights.Resnet => Resnet;
    public CosyVoiceCfmTransformerBlockWeights[] TransformerBlocks { get; }
    IUnetTransformerBlockWeights[] IUnetStageWeights.TransformerBlocks => (IUnetTransformerBlockWeights[])TransformerBlocks;

    public float[]? ResampleConvWeight { get; }
    public float[]? ResampleConvBias { get; }

    public CosyVoiceCfmStageWeights(CosyVoiceFlowWeights flow, string prefix, int numBlocks, bool hasResample)
    {
        Resnet = new CosyVoiceCfmResnetWeights(flow, $"{prefix}.0");
        TransformerBlocks = new CosyVoiceCfmTransformerBlockWeights[numBlocks];
        for (int i = 0; i < numBlocks; i++)
            TransformerBlocks[i] = new CosyVoiceCfmTransformerBlockWeights(flow, $"{prefix}.1.{i}");
        if (hasResample)
        {
            ResampleConvWeight = flow.GetTensor($"{prefix}.2.weight");
            ResampleConvBias = flow.GetTensor($"{prefix}.2.bias");
        }
    }
}

/// <summary>CausalResnetBlock1D: block1(CausalConv1d k=3 + LayerNorm + Mish) -> += mlp(mish(timeEmb)) broadcast -> block2(same) -> + res_conv(x) (1x1 conv, always applied).</summary>
public sealed class CosyVoiceCfmResnetWeights : IResnetBlockWeights
{
    public float[] Block1ConvWeight { get; }
    public float[] Block1ConvBias { get; }
    public float[] Block1LnWeight { get; }
    public float[] Block1LnBias { get; }
    public float[] Block2ConvWeight { get; }
    public float[] Block2ConvBias { get; }
    public float[] Block2LnWeight { get; }
    public float[] Block2LnBias { get; }
    public float[] MlpWeight { get; }
    public float[] MlpBias { get; }
    public float[] ResConvWeight { get; }
    public float[] ResConvBias { get; }

    public CosyVoiceCfmResnetWeights(CosyVoiceFlowWeights flow, string prefix)
    {
        Block1ConvWeight = flow.GetTensor($"{prefix}.block1.block.0.weight");
        Block1ConvBias = flow.GetTensor($"{prefix}.block1.block.0.bias");
        Block1LnWeight = flow.GetTensor($"{prefix}.block1.block.2.weight");
        Block1LnBias = flow.GetTensor($"{prefix}.block1.block.2.bias");
        Block2ConvWeight = flow.GetTensor($"{prefix}.block2.block.0.weight");
        Block2ConvBias = flow.GetTensor($"{prefix}.block2.block.0.bias");
        Block2LnWeight = flow.GetTensor($"{prefix}.block2.block.2.weight");
        Block2LnBias = flow.GetTensor($"{prefix}.block2.block.2.bias");
        MlpWeight = flow.GetTensor($"{prefix}.mlp.1.weight");
        MlpBias = flow.GetTensor($"{prefix}.mlp.1.bias");
        ResConvWeight = flow.GetTensor($"{prefix}.res_conv.weight");
        ResConvBias = flow.GetTensor($"{prefix}.res_conv.bias");
    }
}

/// <summary>BasicTransformerBlock, self-attention only (no cross-attention, cross_attention_dim unset): x += attn1(norm1(x)); x += ff(norm3(x)). attn1's q/k/v have no bias; to_out.0 does. FF is a plain GELU MLP (real checkpoint has no snake alpha/beta params despite the class's act_fn="snake" default).</summary>
public sealed class CosyVoiceCfmTransformerBlockWeights : IUnetTransformerBlockWeights
{
    public float[] Norm1Weight { get; }
    public float[] Norm1Bias { get; }
    public float[] QWeight { get; }
    public float[] KWeight { get; }
    public float[] VWeight { get; }
    public float[] OutWeight { get; }
    public float[] OutBias { get; }
    public float[] Norm3Weight { get; }
    public float[] Norm3Bias { get; }
    public float[] FfUpWeight { get; }
    public float[] FfUpBias { get; }
    public float[] FfDownWeight { get; }
    public float[] FfDownBias { get; }

    public CosyVoiceCfmTransformerBlockWeights(CosyVoiceFlowWeights flow, string prefix)
    {
        Norm1Weight = flow.GetTensor($"{prefix}.norm1.weight");
        Norm1Bias = flow.GetTensor($"{prefix}.norm1.bias");
        QWeight = flow.GetTensor($"{prefix}.attn1.to_q.weight");
        KWeight = flow.GetTensor($"{prefix}.attn1.to_k.weight");
        VWeight = flow.GetTensor($"{prefix}.attn1.to_v.weight");
        OutWeight = flow.GetTensor($"{prefix}.attn1.to_out.0.weight");
        OutBias = flow.GetTensor($"{prefix}.attn1.to_out.0.bias");

        Norm3Weight = flow.GetTensor($"{prefix}.norm3.weight");
        Norm3Bias = flow.GetTensor($"{prefix}.norm3.bias");
        FfUpWeight = flow.GetTensor($"{prefix}.ff.net.0.proj.weight");
        FfUpBias = flow.GetTensor($"{prefix}.ff.net.0.proj.bias");
        FfDownWeight = flow.GetTensor($"{prefix}.ff.net.2.weight");
        FfDownBias = flow.GetTensor($"{prefix}.ff.net.2.bias");
    }
}
